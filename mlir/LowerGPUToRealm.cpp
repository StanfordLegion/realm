// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LoweringUtils.h"
#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::realm {
  namespace {
    bool denseBuffer(MemRefType type)
    {
      if(!type || type.getRank() != 1 || !type.hasStaticShape() ||
         type.getDimSize(0) <= 0 || !type.getLayout().isIdentity())
        return false;
      if(Attribute space = type.getMemorySpace()) {
        auto integerSpace = dyn_cast<IntegerAttr>(space);
        auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(space);
        if(!(integerSpace && integerSpace.getValue() == 1) &&
           !(gpuSpace && gpuSpace.getValue() == gpu::AddressSpace::Global))
          return false;
      }
      Type element = type.getElementType();
      return element.isSignlessInteger(32) || element.isSignlessInteger(64) ||
             element.isF32() || element.isF64();
    }

    struct Buffer {
      Value instance;
      Value ready;
      SmallVector<Value> uses;
      bool destroyed = false;
    };

    struct GPUSubmission {
      using Dependencies = llvm::SmallSetVector<Value, 4>;

      func::FuncOp function;
      ModuleOp module;
      llvm::DenseMap<Value, Value> tokens;
      llvm::DenseMap<Value, Buffer> buffers;
      llvm::DenseMap<uint32_t, TaskRegisterOp> tasks;

      GPUSubmission(func::FuncOp function, ModuleOp module)
        : function(function)
        , module(module)
      {}

      Value denseSpace(OpBuilder &builder, Location loc, MemRefType type)
      {
        Value extent = arith::ConstantIntOp::create(builder, loc, type.getDimSize(0), 64);
        return DenseSpaceOp::create(
            builder, loc, DenseIndexSpaceType::get(builder.getContext()), extent);
      }

      Buffer *lookupBuffer(Operation *op, Value memref, Dependencies &deps)
      {
        auto it = buffers.find(memref);
        if(it == buffers.end() || it->second.destroyed) {
          op->emitOpError("requires a live dense buffer from this entry function");
          return nullptr;
        }
        Buffer &buffer = it->second;
        if(buffer.ready)
          deps.insert(buffer.ready);
        return &buffer;
      }

      LogicalResult lowerLaunch(gpu::LaunchFuncOp op, OpBuilder &builder,
                                Dependencies &dependencies, EventType event)
      {
        if(!op.getAsyncToken() || op.getAsyncObject() || op.getClusterSizeX() ||
           op.getDynamicSharedMemorySize())
          return op.emitOpError(
              "requires a token-based launch without clusters or dynamic shared memory");
        auto symbol = op->getAttrOfType<FlatSymbolRefAttr>("realm.task_entry");
        auto idAttr = op->getAttrOfType<IntegerAttr>("realm.task_id");
        auto entry = symbol ? module.lookupSymbol<func::FuncOp>(symbol.getValue())
                            : func::FuncOp{};
        if(!entry || entry.isExternal() || !idAttr ||
           !idAttr.getType().isSignlessInteger(32) || idAttr.getValue().ult(4))
          return op.emitOpError("requires realm.task_entry naming a host launcher and "
                                "realm.task_id as i32");
        if(entry->getAttrOfType<SymbolRefAttr>("realm.gpu_kernel") != op.getKernelAttr())
          return op.emitOpError()
                 << "requires the host launcher's realm.gpu_kernel to name "
                 << op.getKernelAttr();

        SmallVector<Value> inputs;
        SmallVector<Buffer *> liveBuffers;
        for(Value operand : op.getKernelOperands()) {
          if(isa<MemRefType>(operand.getType())) {
            Buffer *buffer = lookupBuffer(op, operand, dependencies);
            if(!buffer)
              return failure();
            inputs.push_back(buffer->instance);
            liveBuffers.push_back(buffer);
          } else {
            inputs.push_back(operand);
          }
        }
        Location loc = op.getLoc();
        for(Value dim : {op.getGridSizeX(), op.getGridSizeY(), op.getGridSizeZ(),
                         op.getBlockSizeX(), op.getBlockSizeY(), op.getBlockSizeZ()}) {
          if(dim.getType().isIndex())
            dim = arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(), dim);
          else if(dim.getType().isSignlessInteger(32))
            dim = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), dim);
          else if(!dim.getType().isSignlessInteger(64))
            return op.emitOpError("requires index, i32, or i64 launch dimensions");
          inputs.push_back(dim);
        }
        SmallVector<Type> inputTypes;
        for(Value value : inputs) {
          Type type = value.getType();
          inputTypes.push_back(
              isa<MemRefType>(type) ? InstanceType::get(builder.getContext()) : type);
        }
        auto signature = builder.getFunctionType(inputTypes, {});
        if(!llvm::all_of(inputTypes, isTaskArgumentType) ||
           entry.getFunctionType() != signature)
          return op.emitOpError("host launcher must accept converted kernel operands "
                                "followed by six i64 launch dimensions");

        uint32_t id = idAttr.getValue().getZExtValue();
        auto binding = tasks.lookup(id);
        if(binding && binding.getCalleeAttr() != symbol)
          return op.emitOpError("task ID maps to multiple host launchers");
        if(!binding) {
          binding = TaskRegisterOp::create(
              builder, loc, TaskBindingType::get(builder.getContext(), signature), event,
              symbol, idAttr,
              ProcessorKindAttr::get(builder.getContext(), ProcessorKind::TocProc));
          tasks[id] = binding;
        }

        dependencies.insert(binding.getRegistered());
        auto spawn =
            SpawnTypedOp::create(builder, loc, event, binding.getTask(), inputs,
                                 function.getArgument(0), dependencies.getArrayRef());
        tokens[op.getAsyncToken()] = spawn.getDone();
        for(Buffer *buffer : liveBuffers)
          buffer->uses.push_back(spawn.getDone());
        return success();
      }

      LogicalResult lower()
      {
        if(!function.getBody().hasOneBlock() || function.getNumArguments() < 2 ||
           !isa<ProcessorType>(function.getArgument(0).getType()) ||
           !isa<MemoryType>(function.getArgument(1).getType()))
          return function.emitOpError("requires a single-block GPU entry with a "
                                      "processor and memory as its first arguments");
        if(!SymbolTable::symbolKnownUseEmpty(function, module))
          return function.emitOpError(
              "GPU entry must be called through its external ABI");

        auto *context = function.getContext();
        auto instanceType = InstanceType::get(context);
        auto event = EventType::get(context);
        for(BlockArgument arg : function.getArguments()) {
          if(auto type = dyn_cast<MemRefType>(arg.getType())) {
            if(!denseBuffer(type))
              return function.emitOpError(
                  "requires static contiguous rank-one buffer arguments");
            buffers.try_emplace(arg, Buffer{arg, {}, {}});
          }
        }

        SmallVector<Operation *> original;
        for(Operation &op : function.getBody().front())
          original.push_back(&op);
        SmallVector<Operation *> lowered;
        for(Operation *op : original) {
          auto async = dyn_cast<gpu::AsyncOpInterface>(op);
          if(!async) {
            bool hasUnsupportedUse = llvm::any_of(op->getOperandTypes(), [](Type type) {
              return isa<MemRefType, gpu::AsyncTokenType>(type);
            });
            if(op->getName().getDialectNamespace() == "gpu" || hasUnsupportedUse ||
               op->getNumRegions() != 0)
              return op->emitOpError("is outside the supported straight-line GPU subset");
            continue;
          }

          Location loc = op->getLoc();
          OpBuilder builder(op);
          Dependencies dependencies;
          for(Value dependency : async.getAsyncDependencies()) {
            Value translated = tokens.lookup(dependency);
            if(!translated)
              return op->emitOpError("requires a GPU token produced inside this entry");
            if(!translated.getDefiningOp<ReadyOp>())
              dependencies.insert(translated);
          }
          auto require = [&](Value condition, StringRef message) {
            cf::AssertOp::create(builder, loc, condition, message);
          };
          if(auto alloc = dyn_cast<gpu::AllocOp>(op)) {
            auto type = alloc.getMemref().getType();
            if(!alloc.getAsyncToken() || alloc.getHostShared() || !denseBuffer(type))
              return op->emitOpError(
                  "requires an asynchronous static contiguous device allocation");
            Value space = denseSpace(builder, loc, type);
            Value layout =
                DenseLayoutOp::create(builder, loc, DenseLayoutType::get(context), space,
                                      type.getElementTypeBitWidth() / 8);
            auto create = CreateInstanceOp::create(builder, loc, instanceType, event,
                                                   function.getArgument(1), layout,
                                                   dependencies.getArrayRef());
            Value exists = InstanceExistsOp::create(builder, loc, builder.getI1Type(),
                                                    create.getInstance());
            require(exists, "Realm GPU allocation returned no instance");
            tokens[alloc.getAsyncToken()] = create.getReady();
            buffers.try_emplace(alloc.getMemref(),
                                Buffer{create.getInstance(), create.getReady(), {}});
          } else if(auto copy = dyn_cast<gpu::MemcpyOp>(op)) {
            auto type = copy.getSrc().getType();
            if(!copy.getAsyncToken() || !denseBuffer(type) ||
               type.getShape() != copy.getDst().getType().getShape() ||
               type.getElementType() != copy.getDst().getType().getElementType())
              return op->emitOpError(
                  "requires compatible asynchronous dense buffer copies");
            Buffer *source = lookupBuffer(op, copy.getSrc(), dependencies);
            Buffer *target = lookupBuffer(op, copy.getDst(), dependencies);
            if(!source || !target)
              return failure();
            auto transfer = CopyOp::create(
                builder, loc, event, source->instance, 0, target->instance, 0,
                denseSpace(builder, loc, type), type.getElementTypeBitWidth() / 8,
                dependencies.getArrayRef());
            tokens[copy.getAsyncToken()] = transfer.getDone();
            source->uses.push_back(transfer.getDone());
            target->uses.push_back(transfer.getDone());
          } else if(auto launch = dyn_cast<gpu::LaunchFuncOp>(op)) {
            if(failed(lowerLaunch(launch, builder, dependencies, event)))
              return failure();
          } else if(auto dealloc = dyn_cast<gpu::DeallocOp>(op)) {
            if(!dealloc.getAsyncToken() || !dealloc.getAsyncToken().use_empty())
              return op->emitOpError("observes deallocation completion, which Realm "
                                     "destroy does not return");
            Buffer *buffer = lookupBuffer(op, dealloc.getMemref(), dependencies);
            if(!buffer)
              return failure();
            if(!buffer->ready)
              return op->emitOpError("cannot destroy a borrowed buffer");
            dependencies.insert(buffer->uses.begin(), buffer->uses.end());
            Value settled =
                JoinSettledOp::create(builder, loc, event, dependencies.getArrayRef());
            DestroyInstanceOp::create(builder, loc, buffer->instance, settled);
            buffer->destroyed = true;
          } else if(auto wait = dyn_cast<gpu::WaitOp>(op)) {
            Value merged;
            if(dependencies.empty())
              merged = ReadyOp::create(builder, loc, event);
            else if(dependencies.size() == 1)
              merged = dependencies.front();
            else
              merged = MergeOp::create(builder, loc, event, dependencies.getArrayRef());
            if(wait.getAsyncToken()) {
              tokens[wait.getAsyncToken()] = merged;
            } else {
              Value poisoned = AwaitOp::create(builder, loc, builder.getI1Type(), merged);
              Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
              Value successful = arith::XOrIOp::create(builder, loc, poisoned, one);
              require(successful, "Realm GPU work was poisoned");
            }
          } else {
            return op->emitOpError("is outside the supported token-based GPU subset");
          }
          lowered.push_back(op);
        }

        for(const auto &entry : buffers)
          if(entry.second.ready && !entry.second.destroyed)
            return entry.first.getDefiningOp()->emitOpError(
                "requires a matching gpu.dealloc");
        for(Operation *op : llvm::reverse(lowered))
          op->erase();
        SmallVector<Type> types;
        for(BlockArgument arg : function.getArguments()) {
          if(isa<MemRefType>(arg.getType()))
            arg.setType(instanceType);
          types.push_back(arg.getType());
        }
        function.setType(FunctionType::get(context, types, function.getResultTypes()));
        function->removeAttr("realm.gpu_entry");
        return success();
      }
    };

    struct LowerGPUToRealm : PassWrapper<LowerGPUToRealm, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToRealm)

      StringRef getArgument() const override { return "convert-gpu-to-realm"; }
      StringRef getDescription() const override
      {
        return "Lower a token-based GPU entry to Realm submissions and typed tasks";
      }

      void getDependentDialects(DialectRegistry &registry) const override
      {
        registry.insert<RealmDialect, arith::ArithDialect, cf::ControlFlowDialect>();
      }

      void runOnOperation() override
      {
        ModuleOp module = getOperation();
        bool lowered = false;
        for(auto fn : module.getOps<func::FuncOp>()) {
          if(!fn->hasAttrOfType<UnitAttr>("realm.gpu_entry"))
            continue;
          if(!getTaskInitializer(module)) {
            fn.emitOpError("requires a module realm.task_initializer naming the "
                           "collective startup function");
            signalPassFailure();
            return;
          }
          if(failed(GPUSubmission(fn, module).lower())) {
            signalPassFailure();
            return;
          }
          lowered = true;
        }
        // These annotations are only for host launch conversion. The device
        // module remains available for extraction or symbol DCE.
        if(lowered)
          for(auto fn : module.getOps<func::FuncOp>())
            fn->removeAttr("realm.gpu_kernel");
      }
    };
  } // namespace

  std::unique_ptr<Pass> createLowerGPUToRealmPass()
  {
    return std::make_unique<LowerGPUToRealm>();
  }
} // namespace mlir::realm
