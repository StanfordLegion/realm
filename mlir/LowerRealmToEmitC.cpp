// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LoweringUtils.h"
#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>
#include <type_traits>

namespace mlir::realm {
  namespace {
    template <typename QueryOp>
    struct LowerMachineQuery : OpConversionPattern<QueryOp> {
      using OpConversionPattern<QueryOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override
      {
        SmallVector<Type> resultTypes;
        if(failed(
               this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
          return failure();

        constexpr bool isProcessorQuery = std::is_same_v<QueryOp, FindProcessorOp>;
        std::string kind = (isProcessorQuery ? "Realm::Processor::" : "Realm::Memory::") +
                           stringifyEnum(op.getKind()).upper();
        auto cpp = [&](StringRef expression) {
          return emitc::OpaqueAttr::get(rewriter.getContext(), expression);
        };

        SmallVector<Attribute> args{cpp(kind), cpp(op.getLocal() ? "true" : "false")};
        SmallVector<Value> operands;
        for(Value reference : {adaptor.getSameAddressSpace(), adaptor.getAffinity()}) {
          if(reference) {
            args.push_back(rewriter.getIndexAttr(operands.size()));
            operands.push_back(reference);
          } else {
            args.push_back(cpp("std::nullopt"));
          }
        }
        if constexpr(!isProcessorQuery)
          args.push_back(cpp(std::to_string(op.getMinCapacity().value_or(0)) + "ULL"));

        StringRef callee =
            isProcessorQuery ? "Realm::MLIR::findProcessor" : "Realm::MLIR::findMemory";
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, resultTypes, callee, operands, rewriter.getArrayAttr(args));
        return success();
      }
    };

    constexpr StringLiteral readyEvent = "Realm::MLIR::readyEvent";
    constexpr StringLiteral mergeEvents = "Realm::MLIR::mergeEvents";

    bool supportsIndexSpace(IndexSpaceType space)
    {
      IntegerType coordinate = space.getCoordinateType();
      return coordinate.isSignedInteger(32) || coordinate.isUnsignedInteger(32) ||
             coordinate.isSignedInteger(64);
    }

    Value lowerPrecondition(Location loc, ValueRange dependencies,
                            ConversionPatternRewriter &rewriter)
    {
      if(dependencies.size() == 1)
        return dependencies.front();
      Type event = emitc::OpaqueType::get(rewriter.getContext(), "Realm::Event");
      return emitc::CallOpaqueOp::create(rewriter, loc, event,
                                         dependencies.empty() ? readyEvent : mergeEvents,
                                         dependencies)
          .getResult(0);
    }

    template <typename RealmOp>
    struct LowerRuntimeCall : OpConversionPattern<RealmOp> {
      LowerRuntimeCall(const TypeConverter &converter, MLIRContext *context,
                       StringRef callee)
        : OpConversionPattern<RealmOp>(converter, context)
        , callee(callee)
      {}

      LogicalResult matchAndRewrite(RealmOp op, typename RealmOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override
      {
        SmallVector<Type> types;
        if(failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), types)))
          return failure();

        SmallVector<Value> operands;
        if constexpr(std::is_same_v<RealmOp, CreateInstanceOp>) {
          operands = {
              adaptor.getMemory(), adaptor.getLayout(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
        } else if constexpr(std::is_same_v<RealmOp, DestroyInstanceOp>) {
          operands = {
              adaptor.getInstance(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
        } else if constexpr(std::is_same_v<RealmOp, SpawnOp>) {
          operands = {
              adaptor.getProcessor(), adaptor.getTask(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
          llvm::append_range(operands, adaptor.getInstances());
        } else if constexpr(std::is_same_v<RealmOp, CopyOp>) {
          operands = {
              adaptor.getSpace(), adaptor.getSource(), adaptor.getTarget(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
          auto args = rewriter.getArrayAttr(
              {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
               rewriter.getIndexAttr(2), adaptor.getSourceFieldAttr(),
               adaptor.getTargetFieldAttr(), adaptor.getBytesAttr(),
               rewriter.getIndexAttr(3)});
          rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, types, callee, operands,
                                                           args);
          return success();
        } else if constexpr(std::is_same_v<RealmOp, AcquireOp> ||
                            std::is_same_v<RealmOp, ReleaseOp> ||
                            std::is_same_v<RealmOp, DestroyReservationOp>) {
          operands = {
              adaptor.getReservation(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
        } else if constexpr(std::is_same_v<RealmOp, ArriveBarrierOp>) {
          operands = {
              adaptor.getPhase(), adaptor.getValue(),
              lowerPrecondition(op.getLoc(), adaptor.getDependencies(), rewriter)};
        } else {
          static_assert(std::is_same_v<RealmOp, ReadyOp> ||
                            std::is_same_v<RealmOp, MergeOp> ||
                            std::is_same_v<RealmOp, JoinSettledOp> ||
                            std::is_same_v<RealmOp, AwaitOp> ||
                            std::is_same_v<RealmOp, InstanceExistsOp> ||
                            std::is_same_v<RealmOp, CreateReservationOp> ||
                            std::is_same_v<RealmOp, AdvanceBarrierOp> ||
                            std::is_same_v<RealmOp, BarrierEventOp> ||
                            std::is_same_v<RealmOp, BarrierResultOp>,
                        "Specify how to lower this operation's operands");
          llvm::append_range(operands, adaptor.getOperands());
        }
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, types, callee, operands);
        return success();
      }

      StringRef callee;
    };

    struct LowerRealmToEmitC : PassWrapper<LowerRealmToEmitC, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRealmToEmitC)

      StringRef getArgument() const override { return "convert-realm-to-emitc"; }
      StringRef getDescription() const override
      {
        return "Lower Realm operations to calls to the Realm C++ API";
      }

      void getDependentDialects(DialectRegistry &registry) const override
      {
        registry.insert<emitc::EmitCDialect>();
      }

      void runOnOperation() override
      {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();
        if(failed(checkUnsupportedTypes<DenseLayoutType, DenseIndexSpaceType,
                                        TaskBindingType>(
               module, "EmitC",
               "--convert-realm-to-llvm for dense descriptors and "
               "typed tasks"))) {
          signalPassFailure();
          return;
        }
        if(module
               .walk([](CopyOp copy) {
                 if(supportsIndexSpace(cast<IndexSpaceType>(copy.getSpace().getType())))
                   return WalkResult::advance();
                 copy.emitOpError(
                     "requires si32, ui32, or si64 coordinates for Realm lowering");
                 return WalkResult::interrupt();
               })
               .wasInterrupted()) {
          signalPassFailure();
          return;
        }
        bool hasQueries = module
                              .walk([](Operation *op) {
                                return isa<FindProcessorOp, FindMemoryOp>(op)
                                           ? WalkResult::interrupt()
                                           : WalkResult::advance();
                              })
                              .wasInterrupted();
        TypeConverter converter;
        converter.addConversion([](Type type) { return type; });
        converter.addConversion([](ProcessorType type) {
          return emitc::OpaqueType::get(type.getContext(), "Realm::Processor");
        });
        converter.addConversion([](MemoryType type) {
          return emitc::OpaqueType::get(type.getContext(), "Realm::Memory");
        });

        ConversionTarget target(*context);
        target.addLegalDialect<emitc::EmitCDialect>();
        target.addIllegalDialect<RealmDialect>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
        target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
            [&](Operation *op) { return converter.isLegal(op); });
        target.markUnknownOpDynamicallyLegal(
            [&](Operation *op) { return converter.isLegal(op); });

        RewritePatternSet patterns(context);
        patterns.add<LowerMachineQuery<FindProcessorOp>, LowerMachineQuery<FindMemoryOp>>(
            converter, context);
        converter.addConversion(
            [&](EventType) { return emitc::OpaqueType::get(context, "Realm::Event"); });
        converter.addConversion([&](InstanceType) {
          return emitc::OpaqueType::get(context, "Realm::RegionInstance");
        });
        converter.addConversion([&](LayoutType) {
          return emitc::PointerType::get(
              emitc::OpaqueType::get(context, "const Realm::InstanceLayoutGeneric"));
        });
        converter.addConversion([&](TaskType) {
          return emitc::OpaqueType::get(context, "Realm::Processor::TaskFuncID");
        });
        converter.addConversion([&](IndexSpaceType type) -> Type {
          if(!supportsIndexSpace(type))
            return {};
          return emitc::PointerType::get(
              emitc::OpaqueType::get(context, "const Realm::IndexSpaceGeneric"));
        });
        converter.addConversion([&](ReservationType) {
          return emitc::OpaqueType::get(context, "Realm::Reservation");
        });
        converter.addConversion([&](BarrierPhaseType) {
          return emitc::OpaqueType::get(context, "Realm::Barrier");
        });
        patterns.add<LowerRuntimeCall<ReadyOp>>(converter, context, readyEvent);
        patterns.add<LowerRuntimeCall<MergeOp>>(converter, context, mergeEvents);
        patterns.add<LowerRuntimeCall<JoinSettledOp>>(converter, context,
                                                      "Realm::MLIR::joinSettled");
        patterns.add<LowerRuntimeCall<AwaitOp>>(converter, context,
                                                "Realm::MLIR::awaitEvent");
        patterns.add<LowerRuntimeCall<CreateInstanceOp>>(converter, context,
                                                         "Realm::MLIR::createInstance");
        patterns.add<LowerRuntimeCall<InstanceExistsOp>>(converter, context,
                                                         "Realm::MLIR::instanceExists");
        patterns.add<LowerRuntimeCall<DestroyInstanceOp>>(converter, context,
                                                          "Realm::MLIR::destroyInstance");
        patterns.add<LowerRuntimeCall<SpawnOp>>(converter, context, "Realm::MLIR::spawn");
        patterns.add<LowerRuntimeCall<CopyOp>>(converter, context, "Realm::MLIR::copy");
        patterns.add<LowerRuntimeCall<CreateReservationOp>>(
            converter, context, "Realm::MLIR::createReservation");
        patterns.add<LowerRuntimeCall<AcquireOp>>(converter, context,
                                                  "Realm::MLIR::acquireReservation");
        patterns.add<LowerRuntimeCall<ReleaseOp>>(converter, context,
                                                  "Realm::MLIR::releaseReservation");
        patterns.add<LowerRuntimeCall<DestroyReservationOp>>(
            converter, context, "Realm::MLIR::destroyReservation");
        patterns.add<LowerRuntimeCall<AdvanceBarrierOp>>(converter, context,
                                                         "Realm::MLIR::advanceBarrier");
        patterns.add<LowerRuntimeCall<BarrierEventOp>>(converter, context,
                                                       "Realm::MLIR::barrierEvent");
        patterns.add<LowerRuntimeCall<ArriveBarrierOp>>(converter, context,
                                                        "Realm::MLIR::arriveBarrier");
        patterns.add<LowerRuntimeCall<BarrierResultOp>>(converter, context,
                                                        "Realm::MLIR::barrierResult");
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                       converter);
        populateCallOpTypeConversionPattern(patterns, converter);
        populateReturnOpTypeConversionPattern(patterns, converter);
        scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);

        if(failed(applyPartialConversion(module, target, std::move(patterns)))) {
          signalPassFailure();
          return;
        }
        OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
        if(hasQueries)
          emitc::IncludeOp::create(builder, module.getLoc(), "RealmMachineQueries.h");
        emitc::IncludeOp::create(builder, module.getLoc(), "RealmRuntime.h");
      }
    };
  } // namespace

  std::unique_ptr<Pass> createLowerRealmToEmitCPass()
  {
    return std::make_unique<LowerRealmToEmitC>();
  }
} // namespace mlir::realm
