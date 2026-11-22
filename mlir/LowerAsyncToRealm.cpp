// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LoweringUtils.h"
#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::realm {
  namespace {
    bool localExecute(Value value, func::FuncOp parent)
    {
      auto execute = value.getDefiningOp<async::ExecuteOp>();
      return execute && execute.getOperation()->getParentOp() == parent.getOperation();
    }

    struct LowerAsyncToRealm : PassWrapper<LowerAsyncToRealm, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAsyncToRealm)

      StringRef getArgument() const override { return "convert-async-to-realm"; }
      StringRef getDescription() const override
      {
        return "Outline token-only Async regions as local Realm CPU tasks";
      }

      void getDependentDialects(DialectRegistry &registry) const override
      {
        registry.insert<RealmDialect, arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect>();
      }

      void runOnOperation() override
      {
        ModuleOp module = getOperation();
        SmallVector<Operation *> operations;
        llvm::DenseMap<Operation *, uint32_t> taskIds;
        llvm::DenseMap<Operation *, uint64_t> nextIds;
        llvm::DenseSet<uint32_t> usedIds;
        module.walk([&](TaskRegisterOp op) {
          if(op.getKind() == ProcessorKind::LocProc)
            usedIds.insert(op.getId());
        });

        WalkResult checked = module.walk([&](Operation *op) {
          if(op->getName().getDialectNamespace() != "async")
            return WalkResult::advance();
          if(isa<async::YieldOp>(op) && isa<async::ExecuteOp>(op->getParentOp()))
            return WalkResult::advance();
          if(!getTaskInitializer(module)) {
            op->emitOpError("requires a module realm.task_initializer naming the "
                            "collective startup function");
            return WalkResult::interrupt();
          }
          auto parent = dyn_cast<func::FuncOp>(op->getParentOp());
          auto firstId = parent
                             ? parent->getAttrOfType<IntegerAttr>("realm.async_task_base")
                             : IntegerAttr{};
          if(!parent || parent->getParentOp() != module ||
             !parent.getBody().hasOneBlock() || parent.getNumArguments() == 0 ||
             !isa<ProcessorType>(parent.getArgument(0).getType()) || !firstId ||
             !firstId.getType().isSignlessInteger(32)) {
            op->emitOpError(
                "requires a single-block Realm Async entry with a CPU processor as the "
                "first argument and a realm.async_task_base i32 attribute");
            return WalkResult::interrupt();
          }
          if(auto exec = dyn_cast<async::ExecuteOp>(op)) {
            if(exec->getNumResults() != 1 || !exec.getBodyOperands().empty() ||
               !llvm::all_of(exec.getDependencies(),
                             [&](Value dep) { return localExecute(dep, parent); })) {
              op->emitOpError("requires token-only dependencies and no async values");
              return WalkResult::interrupt();
            }
            llvm::SetVector<Value> captures;
            getUsedValuesDefinedAbove(exec->getRegions(), captures);
            if(!llvm::all_of(captures, [](Value value) {
                 return isTaskArgumentType(value.getType());
               })) {
              op->emitOpError(
                  "captures must be instances or i32, i64, f32, or f64 scalars");
              return WalkResult::interrupt();
            }
            uint64_t &next =
                nextIds.try_emplace(parent, firstId.getValue().getZExtValue())
                    .first->second;
            if(next < 4 || next > UINT32_MAX || !usedIds.insert(next).second) {
              op->emitOpError("requires a task ID in an unused application range");
              return WalkResult::interrupt();
            }
            taskIds[op] = next++;
          } else if(auto await = dyn_cast<async::AwaitOp>(op)) {
            if(!localExecute(await.getOperand(), parent)) {
              op->emitOpError("requires a token produced by an execute in this function");
              return WalkResult::interrupt();
            }
          } else {
            op->emitOpError("is outside the supported token-only Async subset");
            return WalkResult::interrupt();
          }
          operations.push_back(op);
          return WalkResult::advance();
        });
        if(checked.wasInterrupted()) {
          signalPassFailure();
          return;
        }

        // Check token uses before replacing them with unlike-typed Realm events.
        llvm::DenseSet<Operation *> supported(operations.begin(), operations.end());
        for(Operation *op : operations) {
          auto exec = dyn_cast<async::ExecuteOp>(op);
          if(exec && llvm::any_of(exec.getToken().getUsers(), [&](Operation *user) {
               return !supported.contains(user);
             })) {
            op->emitOpError("token escapes the supported Async subset");
            signalPassFailure();
            return;
          }
        }

        auto eventType = EventType::get(&getContext());
        auto kind = ProcessorKindAttr::get(&getContext(), ProcessorKind::LocProc);
        for(Operation *op : operations) {
          OpBuilder builder(op);
          Location loc = op->getLoc();
          if(auto exec = dyn_cast<async::ExecuteOp>(op)) {
            llvm::SetVector<Value> captures;
            getUsedValuesDefinedAbove(exec->getRegions(), captures);
            auto signature = builder.getFunctionType(
                ValueRange(captures.getArrayRef()).getTypes(), {});
            auto parent = cast<func::FuncOp>(op->getParentOp());
            uint32_t id = taskIds.lookup(op);
            auto fn = [&] {
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToEnd(module.getBody());
              std::string base =
                  ("__realm_async_" + parent.getSymName() + "_" + Twine(id)).str();
              std::string name = base;
              unsigned suffix = 0;
              while(module.lookupSymbol(name))
                name = (Twine(base) + "_" + Twine(++suffix)).str();
              auto outlined = func::FuncOp::create(builder, loc, name, signature);
              outlined.setPrivate();
              Block *entry = outlined.addEntryBlock();
              builder.setInsertionPointToStart(entry);
              IRMapping mapping;
              mapping.map(captures.getArrayRef(), entry->getArguments());
              for(Operation &bodyOp : exec.getBody()->without_terminator())
                builder.clone(bodyOp, mapping);
              func::ReturnOp::create(builder, loc);
              return outlined;
            }();
            auto binding = TaskRegisterOp::create(
                builder, loc, TaskBindingType::get(&getContext(), signature), eventType,
                FlatSymbolRefAttr::get(fn.getSymNameAttr()),
                builder.getI32IntegerAttr(id), kind);
            SmallVector<Value> dependencies(exec.getDependencies());
            dependencies.push_back(binding.getRegistered());
            auto spawn = SpawnTypedOp::create(builder, loc, eventType, binding.getTask(),
                                              captures.getArrayRef(),
                                              parent.getArgument(0), dependencies);
            exec.getToken().replaceAllUsesWith(spawn.getDone());
          } else {
            auto await = cast<async::AwaitOp>(op);
            Value poisoned =
                AwaitOp::create(builder, loc, builder.getI1Type(), await.getOperand());
            Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
            Value successful = arith::XOrIOp::create(builder, loc, poisoned, one);
            cf::AssertOp::create(builder, loc, successful,
                                 "Realm task lowered from async.execute was poisoned");
          }
          op->erase();
        }
        for(auto &entry : nextIds)
          entry.first->removeAttr("realm.async_task_base");
      }
    };
  } // namespace

  std::unique_ptr<Pass> createLowerAsyncToRealmPass()
  {
    return std::make_unique<LowerAsyncToRealm>();
  }
} // namespace mlir::realm
