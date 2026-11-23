// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LoweringUtils.h"
#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#include <cassert>
#include <type_traits>

namespace mlir::realm {
  namespace {
    LLVM::LLVMStructType layoutType(MLIRContext *context)
    {
      auto word = IntegerType::get(context, 64);
      return LLVM::LLVMStructType::getLiteral(context, {word, word});
    }

    struct RuntimeCalls {
      OpBuilder &builder;
      Location loc;
      ModuleOp module;

      Value integer(int64_t value, unsigned width = 64)
      {
        return LLVM::ConstantOp::create(builder, loc, builder.getIntegerType(width),
                                        value);
      }

      Value null()
      {
        return LLVM::ZeroOp::create(builder, loc,
                                    LLVM::LLVMPointerType::get(builder.getContext()));
      }

      Value slot(Type element, unsigned count = 1)
      {
        OpBuilder::InsertionGuard guard(builder);
        Operation *op = builder.getInsertionBlock()->getParentOp();
        while(!isa<func::FuncOp, LLVM::LLVMFuncOp>(op))
          op = op->getParentOp();
        builder.setInsertionPointToStart(&op->getRegion(0).front());
        return LLVM::AllocaOp::create(builder, loc,
                                      LLVM::LLVMPointerType::get(builder.getContext()),
                                      element, integer(count), 0);
      }

      LLVM::CallOp call(StringRef name, Type result, ValueRange args)
      {
        SmallVector<Type> inputs(args.getTypes());
        SmallVector<Type> results;
        if(!isa<LLVM::LLVMVoidType>(result))
          results.push_back(result);
        auto type = LLVM::LLVMFunctionType::get(result, inputs, false);
        Operation *existing = module.lookupSymbol(name);
        if(!existing) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(module.getBody());
          LLVM::LLVMFuncOp::create(builder, loc, name, type);
        } else if(auto fn = dyn_cast<LLVM::LLVMFuncOp>(existing);
                  fn && fn.getFunctionType() == type) {
          // Reuse a declaration from another lowered operation.
        } else if(auto fn = dyn_cast<func::FuncOp>(existing);
                  fn &&
                  fn.getFunctionType() == builder.getFunctionType(inputs, results)) {
          // Its type is already compatible. The func conversion will lower it.
        } else {
          emitError(loc) << "conflicting declaration of " << name;
          return {};
        }
        return LLVM::CallOp::create(builder, loc, results, name, args);
      }

      LogicalResult statusCall(StringRef name, ValueRange args)
      {
        auto callOp = call(name, builder.getI32Type(), args);
        if(!callOp ||
           !call("realm_mlir_check_status", LLVM::LLVMVoidType::get(builder.getContext()),
                 callOp.getResult()))
          return failure();
        return success();
      }

      Value runtime()
      {
        auto callOp = call("realm_mlir_runtime",
                           LLVM::LLVMPointerType::get(builder.getContext()), {});
        return callOp ? callOp.getResult() : Value{};
      }

      Value load(Type type, Value address)
      {
        return LLVM::LoadOp::create(builder, loc, type, address);
      }

      Value wordAddress(Value base, size_t index)
      {
        // GEPArg holds a constant index in kGEPConstantBitWidth signed bits.
        assert(index < (size_t(1) << (LLVM::kGEPConstantBitWidth - 1)));
        return LLVM::GEPOp::create(builder, loc, base.getType(), builder.getI64Type(),
                                   base,
                                   ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
      }

      Value packWords(ValueRange values)
      {
        if(values.empty())
          return null();
        Value base = slot(builder.getI64Type(), values.size());
        for(auto [index, value] : llvm::enumerate(values))
          LLVM::StoreOp::create(builder, loc, value, wordAddress(base, index));
        return base;
      }

      Value merge(Value runtime, ValueRange events, bool ignoreFaults)
      {
        if(events.empty())
          return integer(0);
        Value out = slot(builder.getI64Type());
        if(failed(statusCall("realm_event_merge",
                             {runtime, packWords(events), integer(events.size()), out,
                              integer(ignoreFaults, 32)})))
          return {};
        return load(builder.getI64Type(), out);
      }

      Value precondition(Value runtime, ValueRange events)
      {
        llvm::SmallSetVector<Value, 4> active;
        for(Value event : events)
          if(!matchPattern(event, m_Zero()))
            active.insert(event);
        return active.size() == 1 ? active.front()
                                  : merge(runtime, active.getArrayRef(), false);
      }
    };

    // A typed payload has one 64-bit word per argument, including floating-point
    // bit patterns. The same convention is used in the generated task wrapper.
    Value packArgument(RuntimeCalls &calls, Value argument)
    {
      Type type = argument.getType();
      if(auto fp = dyn_cast<FloatType>(type)) {
        type = calls.builder.getIntegerType(fp.getWidth());
        argument = LLVM::BitcastOp::create(calls.builder, calls.loc, type, argument);
      }
      if(type.isInteger(32))
        return LLVM::ZExtOp::create(calls.builder, calls.loc, calls.builder.getI64Type(),
                                    argument);
      return argument;
    }

    Value unpackArgument(RuntimeCalls &calls, Type type, Value word)
    {
      if(type.isInteger(32) || type.isF32())
        word = LLVM::TruncOp::create(calls.builder, calls.loc, calls.builder.getI32Type(),
                                     word);
      if(isa<FloatType>(type))
        return LLVM::BitcastOp::create(calls.builder, calls.loc, type, word);
      return word;
    }

    struct Registration {
      StringAttr callee;
      FunctionType signature;
      Location loc;
    };

    using RegistrationTable =
        llvm::MapVector<std::pair<uint32_t, uint32_t>, Registration>;

    Value registerTask(RuntimeCalls &calls, const TypeConverter &converter,
                       const Registration &registration, uint32_t id, uint32_t kind,
                       Value runtime)
    {
      OpBuilder &builder = calls.builder;
      auto *context = builder.getContext();
      auto word = builder.getI64Type();
      auto ptr = LLVM::LLVMPointerType::get(context);
      auto voidTy = LLVM::LLVMVoidType::get(context);
      auto wrapperType =
          LLVM::LLVMFunctionType::get(voidTy, {ptr, word, ptr, word, word});
      SmallVector<Type> inputs;
      if(failed(converter.convertTypes(registration.signature.getInputs(), inputs)))
        return {};

      auto wrapper = [&]() -> LLVM::LLVMFuncOp {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(calls.module.getBody());
        std::string base = ("__realm_task_" + registration.callee.getValue()).str();
        std::string name = base;
        unsigned suffix = 0;
        while(calls.module.lookupSymbol(name))
          name = (Twine(base) + "_" + Twine(++suffix)).str();
        auto fn = LLVM::LLVMFuncOp::create(builder, registration.loc, name, wrapperType,
                                           LLVM::Linkage::Internal);
        Block *entry = fn.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);
        RuntimeCalls body{builder, registration.loc, calls.module};
        Value payload = entry->getArgument(0);
        if(!body.call("realm_mlir_check_task_payload", voidTy,
                      {payload, entry->getArgument(1),
                       body.integer(inputs.size() * sizeof(uint64_t))}))
          return {};
        SmallVector<Value> arguments;
        for(auto [index, type] : llvm::enumerate(inputs)) {
          Value wordValue = LLVM::LoadOp::create(builder, registration.loc, word,
                                                 body.wordAddress(payload, index), 1);
          arguments.push_back(unpackArgument(body, type, wordValue));
        }
        LLVM::CallOp::create(builder, registration.loc, TypeRange{},
                             registration.callee.getValue(), arguments);
        LLVM::ReturnOp::create(builder, registration.loc, ValueRange{});
        return fn;
      }();
      if(!wrapper)
        return {};

      Value out = calls.slot(word);
      Value function = LLVM::AddressOfOp::create(builder, registration.loc, wrapper);
      if(failed(calls.statusCall("realm_processor_register_task_by_kind",
                                 {runtime, calls.integer(kind, 32), calls.integer(0, 32),
                                  calls.integer(id, 32), function, calls.null(),
                                  calls.integer(0), out})))
        return {};
      return calls.load(word, out);
    }

    Registration getRegistration(TaskRegisterOp op)
    {
      return {op.getCalleeAttr().getAttr(),
              cast<TaskBindingType>(op.getTask().getType()).getSignature(), op.getLoc()};
    }

    LogicalResult collectRegistrations(ModuleOp module, StringRef initName,
                                       RegistrationTable &registrations)
    {
      if(Operation *existing = module.lookupSymbol(initName)) {
        auto fn = dyn_cast<func::FuncOp>(existing);
        if(!fn || !fn.isExternal() || fn.getNumArguments() != 0 ||
           fn.getNumResults() != 1 || !isa<EventType>(fn.getResultTypes().front()))
          return existing->emitOpError(
              "must declare the Realm initializer as () -> !realm.event");
      }
      return module.walk([&](TaskRegisterOp op) {
                     if(op->getParentOfType<ModuleOp>() != module) {
                       op.emitOpError("requires the initializer in its own module");
                       return WalkResult::interrupt();
                     }
                     auto key =
                         std::make_pair(op.getId(), static_cast<uint32_t>(op.getKind()));
                     Registration registration = getRegistration(op);
                     auto [it, inserted] = registrations.try_emplace(key, registration);
                     if(!inserted && (it->second.callee != registration.callee ||
                                      it->second.signature != registration.signature)) {
                       op.emitOpError(
                           "conflicting task definitions for one ID and processor kind");
                       return WalkResult::interrupt();
                     }
                     return WalkResult::advance();
                   }).wasInterrupted()
                 ? failure()
                 : success();
    }

    LogicalResult emitInitializer(ModuleOp module, StringRef name,
                                  const RegistrationTable &registrations,
                                  const LLVMTypeConverter &converter)
    {
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToStart(module.getBody());
      Location loc = module.getLoc();
      auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if(!fn) {
        auto type = LLVM::LLVMFunctionType::get(builder.getI64Type(), {});
        fn = LLVM::LLVMFuncOp::create(builder, loc, name, type);
      }
      fn.setPublic();
      fn.setLinkage(LLVM::Linkage::External);
      builder.setInsertionPointToStart(fn.addEntryBlock(builder));
      RuntimeCalls calls{builder, loc, module};
      SmallVector<Value> ready;
      if(!registrations.empty()) {
        Value runtime = calls.runtime();
        if(!runtime)
          return failure();
        for(const auto &[key, registration] : registrations) {
          Value event = registerTask(calls, converter, registration, key.first,
                                     key.second, runtime);
          if(!event)
            return failure();
          ready.push_back(event);
        }
        Value allReady = calls.precondition(runtime, ready);
        if(!allReady)
          return failure();
        LLVM::ReturnOp::create(builder, loc, allReady);
      } else {
        LLVM::ReturnOp::create(builder, loc, calls.integer(0));
      }
      return success();
    }

    template <typename Op>
    struct LowerOp : OpConversionPattern<Op> {
      LowerOp(const LLVMTypeConverter &converter, MLIRContext *context,
              bool initializeTasks = false)
        : OpConversionPattern<Op>(converter, context)
        , initializeTasks(initializeTasks)
      {}

      LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override
      {
        if(!op->template getParentOfType<func::FuncOp>() &&
           !op->template getParentOfType<LLVM::LLVMFuncOp>())
          return op.emitOpError("requires an enclosing function for LLVM lowering");
        RuntimeCalls calls{rewriter, op.getLoc(),
                           op->template getParentOfType<ModuleOp>()};
        auto word = rewriter.getI64Type();
        auto intTy = rewriter.getI32Type();
        auto exists = [&](Value value) {
          return LLVM::ICmpOp::create(rewriter, op.getLoc(), LLVM::ICmpPredicate::ne,
                                      value, calls.integer(0));
        };

        if constexpr(std::is_same_v<Op, ReadyOp>) {
          rewriter.replaceOp(op, calls.integer(0));
        } else if constexpr(std::is_same_v<Op, DenseSpaceOp>) {
          rewriter.replaceOp(op, adaptor.getExtent());
        } else if constexpr(std::is_same_v<Op, DenseLayoutOp>) {
          Value layout = LLVM::PoisonOp::create(rewriter, op.getLoc(),
                                                layoutType(rewriter.getContext()));
          layout = LLVM::InsertValueOp::create(rewriter, op.getLoc(), layout,
                                               adaptor.getSpace(), ArrayRef<int64_t>{0});
          rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
              op, layout, calls.integer(op.getFieldBytes()), ArrayRef<int64_t>{1});
        } else if constexpr(std::is_same_v<Op, InstanceExistsOp>) {
          rewriter.replaceOp(op, exists(adaptor.getInstance()).getResult());
        } else if constexpr(std::is_same_v<Op, FindProcessorOp> ||
                            std::is_same_v<Op, FindMemoryOp>) {
          Value same = adaptor.getSameAddressSpace();
          Value affinity = adaptor.getAffinity();
          SmallVector<Value> args{calls.integer(static_cast<int32_t>(op.getKind()), 32),
                                  calls.integer(op.getLocal(), 32),
                                  calls.integer(same ? 1 : 0, 32),
                                  same ? same : calls.integer(0),
                                  calls.integer(affinity ? 1 : 0, 32),
                                  affinity ? affinity : calls.integer(0)};
          StringRef name = "realm_mlir_find_processor";
          if constexpr(std::is_same_v<Op, FindMemoryOp>) {
            name = "realm_mlir_find_memory";
            args.push_back(calls.integer(op.getMinCapacity().value_or(0)));
          }
          auto result = calls.call(name, word, args);
          if(!result)
            return failure();
          rewriter.replaceOp(op, {result.getResult(), exists(result.getResult())});
        } else if constexpr(std::is_same_v<Op, MergeOp> ||
                            std::is_same_v<Op, JoinSettledOp>) {
          if(adaptor.getEvents().empty()) {
            rewriter.replaceOp(op, calls.integer(0));
            return success();
          }
          Value runtime = calls.runtime();
          if(!runtime)
            return failure();
          Value result = calls.merge(runtime, adaptor.getEvents(),
                                     std::is_same_v<Op, JoinSettledOp>);
          if(!result)
            return failure();
          rewriter.replaceOp(op, result);
        } else if constexpr(std::is_same_v<Op, TaskRegisterOp>) {
          Value id = calls.integer(op.getId(), 32);
          Value ready;
          if(initializeTasks) {
            // The caller consumes the initializer's readiness at collective startup.
            ready = calls.integer(0);
          } else {
            Value runtime = calls.runtime();
            if(!runtime)
              return failure();
            ready =
                registerTask(calls, *this->getTypeConverter(), getRegistration(op),
                             op.getId(), static_cast<uint32_t>(op.getKind()), runtime);
            if(!ready)
              return failure();
          }
          rewriter.replaceOp(op, {id, ready});
        } else {
          Value runtime = calls.runtime();
          if(!runtime)
            return failure();
          if constexpr(std::is_same_v<Op, AwaitOp>) {
            Value poisoned = calls.slot(intTy);
            if(failed(calls.statusCall(
                   "realm_event_wait",
                   {runtime, adaptor.getEvent(), calls.integer(INT64_MIN), poisoned})))
              return failure();
            rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::ne,
                                                      calls.load(intTy, poisoned),
                                                      calls.integer(0, 32));
          } else {
            Value wait = calls.precondition(runtime, adaptor.getDependencies());
            if(!wait)
              return failure();
            if constexpr(std::is_same_v<Op, SpawnOp> ||
                         std::is_same_v<Op, SpawnTypedOp>) {
              SmallVector<Value> args;
              if constexpr(std::is_same_v<Op, SpawnOp>)
                llvm::append_range(args, adaptor.getInstances());
              else
                for(Value argument : adaptor.getArgs())
                  args.push_back(packArgument(calls, argument));
              Value out = calls.slot(word);
              if(failed(
                     calls.statusCall("realm_processor_spawn",
                                      {runtime, adaptor.getProcessor(), adaptor.getTask(),
                                       calls.packWords(args),
                                       calls.integer(args.size() * sizeof(uint64_t)),
                                       calls.null(), wait, calls.integer(0, 32), out})))
                return failure();
              rewriter.replaceOp(op, calls.load(word, out));
            } else if constexpr(std::is_same_v<Op, CreateInstanceOp>) {
              Value instance = calls.slot(word);
              Value ready = calls.slot(word);
              auto extent = LLVM::ExtractValueOp::create(
                  rewriter, op.getLoc(), adaptor.getLayout(), ArrayRef<int64_t>{0});
              auto bytes = LLVM::ExtractValueOp::create(
                  rewriter, op.getLoc(), adaptor.getLayout(), ArrayRef<int64_t>{1});
              if(failed(calls.statusCall("realm_mlir_create_dense",
                                         {runtime, adaptor.getMemory(), extent, bytes,
                                          wait, instance, ready})))
                return failure();
              rewriter.replaceOp(op,
                                 {calls.load(word, instance), calls.load(word, ready)});
            } else if constexpr(std::is_same_v<Op, CopyOp>) {
              Value out = calls.slot(word);
              if(failed(calls.statusCall(
                     "realm_mlir_copy_dense",
                     {runtime, adaptor.getSource(),
                      calls.integer(op.getSourceField(), 32), adaptor.getTarget(),
                      calls.integer(op.getTargetField(), 32), adaptor.getSpace(),
                      calls.integer(op.getBytes()), wait, out})))
                return failure();
              rewriter.replaceOp(op, calls.load(word, out));
            } else {
              static_assert(std::is_same_v<Op, DestroyInstanceOp>);
              if(failed(calls.statusCall("realm_region_instance_destroy",
                                         {runtime, adaptor.getInstance(), wait})))
                return failure();
              rewriter.eraseOp(op);
            }
          }
        }
        return success();
      }

      bool initializeTasks;
    };

    struct LowerRealmToLLVM : PassWrapper<LowerRealmToLLVM, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRealmToLLVM)

      StringRef getArgument() const override { return "convert-realm-to-llvm"; }
      StringRef getDescription() const override
      {
        return "Lower Realm host code to LLVM and the Realm C ABI";
      }

      void getDependentDialects(DialectRegistry &registry) const override
      {
        registry.insert<LLVM::LLVMDialect, cf::ControlFlowDialect>();
      }

      void runOnOperation() override
      {
        ModuleOp module = getOperation();
        auto *context = &getContext();
        if(failed(checkUnsupportedTypes<LayoutType, IndexSpaceType, ReservationType,
                                        BarrierPhaseType>(
               module, "LLVM", "--convert-realm-to-emitc for Realm C++ handles"))) {
          signalPassFailure();
          return;
        }
        for(auto gpuModule : module.getOps<gpu::GPUModuleOp>()) {
          gpuModule.emitOpError("device IR must be extracted or removed before Realm "
                                "LLVM lowering of the host");
          signalPassFailure();
          return;
        }
        ConversionTarget scfTarget(*context);
        scfTarget.addIllegalDialect<scf::SCFDialect>();
        scfTarget.addLegalDialect<cf::ControlFlowDialect>();
        RewritePatternSet scfPatterns(context);
        populateSCFToControlFlowConversionPatterns(scfPatterns);
        if(failed(applyPartialConversion(module, scfTarget, std::move(scfPatterns)))) {
          signalPassFailure();
          return;
        }

        LLVMTypeConverter converter(context);
        auto word = IntegerType::get(context, 64);
        converter.addConversion([&](EventType) { return word; });
        converter.addConversion([&](InstanceType) { return word; });
        converter.addConversion([&](ProcessorType) { return word; });
        converter.addConversion([&](MemoryType) { return word; });
        converter.addConversion([&](DenseLayoutType) { return layoutType(context); });
        converter.addConversion([&](DenseIndexSpaceType) { return word; });
        converter.addConversion([&](TaskType) { return IntegerType::get(context, 32); });
        converter.addConversion(
            [&](TaskBindingType) { return IntegerType::get(context, 32); });

        RegistrationTable registrations;
        auto initializer = getTaskInitializer(module);
        if(module->hasAttr("realm.task_initializer") && !initializer) {
          module.emitOpError("realm.task_initializer requires a nonempty symbol name");
          signalPassFailure();
          return;
        }
        if(initializer &&
           failed(collectRegistrations(module, initializer.getValue(), registrations))) {
          signalPassFailure();
          return;
        }

        // MLIR symbol visibility does not determine the exported LLVM symbols.
        // Declarations (including a declared initializer) keep their linkage.
        module.walk([&](func::FuncOp fn) {
          if(fn.isPrivate() && !fn.isExternal() && !fn->hasAttr("llvm.linkage"))
            fn->setAttr("llvm.linkage",
                        LLVM::LinkageAttr::get(context, LLVM::Linkage::Internal));
        });

        ConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        populateFuncToLLVMConversionPatterns(converter, patterns);
        arith::populateArithToLLVMConversionPatterns(converter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
        cf::populateAssertToLLVMConversionPattern(converter, patterns);
        patterns.add<LowerOp<ReadyOp>, LowerOp<MergeOp>, LowerOp<JoinSettledOp>,
                     LowerOp<AwaitOp>, LowerOp<DenseSpaceOp>, LowerOp<DenseLayoutOp>,
                     LowerOp<CreateInstanceOp>, LowerOp<DestroyInstanceOp>,
                     LowerOp<InstanceExistsOp>, LowerOp<CopyOp>, LowerOp<FindProcessorOp>,
                     LowerOp<FindMemoryOp>, LowerOp<SpawnOp>, LowerOp<SpawnTypedOp>>(
            converter, context);
        patterns.add<LowerOp<TaskRegisterOp>>(converter, context, bool(initializer));
        if(failed(applyFullConversion(module, target, std::move(patterns)))) {
          signalPassFailure();
          return;
        }
        if(initializer) {
          if(failed(emitInitializer(module, initializer.getValue(), registrations,
                                    converter))) {
            signalPassFailure();
            return;
          }
          module->removeAttr("realm.task_initializer");
        }
      }
    };
  } // namespace

  std::unique_ptr<Pass> createLowerRealmToLLVMPass()
  {
    return std::make_unique<LowerRealmToLLVM>();
  }
} // namespace mlir::realm
