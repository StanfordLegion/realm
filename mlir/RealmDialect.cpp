// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RealmDialect.cpp.inc"
#include "RealmEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RealmAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "RealmTypes.cpp.inc"

namespace mlir::realm {
  static ParseResult parseInstanceLayoutType(OpAsmParser &parser, Type &layout)
  {
    if(succeeded(parser.parseOptionalColon()))
      return parser.parseType(layout);
    layout = LayoutType::get(parser.getContext());
    return success();
  }

  static void printInstanceLayoutType(OpAsmPrinter &printer, Operation *, Type layout)
  {
    if(!isa<LayoutType>(layout))
      printer << " : " << layout;
  }
} // namespace mlir::realm

#define GET_OP_CLASSES
#include "RealmOps.cpp.inc"

void mlir::realm::RealmDialect::initialize()
{
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RealmAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "RealmTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "RealmOps.cpp.inc"
      >();
}

mlir::LogicalResult
mlir::realm::IndexSpaceType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                                    unsigned rank, IntegerType coordinateType)
{
  if(rank == 0)
    return emitError() << "expected a positive index space rank";
  if(coordinateType.isSignless())
    return emitError() << "expected a signed or unsigned coordinate type";
  return success();
}

mlir::LogicalResult mlir::realm::CopyOp::verify()
{
  if(!getBytesAttr().getValue().isStrictlyPositive())
    return emitOpError("requires a positive field size in bytes");
  if(getSourceFieldAttr().getInt() == -1 || getTargetFieldAttr().getInt() == -1)
    return emitOpError("reserves field ID -1 for fills");
  return success();
}

mlir::LogicalResult mlir::realm::DenseLayoutOp::verify()
{
  if(!getFieldBytesAttr().getValue().isStrictlyPositive())
    return emitOpError("requires a positive field size in bytes");
  return success();
}

mlir::LogicalResult mlir::realm::TaskRegisterOp::verify()
{
  auto signature = cast<TaskBindingType>(getTask().getType()).getSignature();
  if(signature.getNumResults() != 0 ||
     !llvm::all_of(signature.getInputs(), isTaskArgumentType))
    return emitOpError(
        "requires instance, i32, i64, f32, or f64 arguments and no results");
  if(getIdAttr().getValue().ult(4))
    return emitOpError("requires a task ID of at least 4");
  return success();
}

mlir::LogicalResult
mlir::realm::TaskRegisterOp::verifySymbolUses(SymbolTableCollection &symbols)
{
  auto fn = symbols.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  auto signature = cast<TaskBindingType>(getTask().getType()).getSignature();
  if(!fn || fn.isExternal() || fn.getFunctionType() != signature)
    return emitOpError("requires a module-defined func.func with the task signature");
  return success();
}

mlir::LogicalResult mlir::realm::SpawnTypedOp::verify()
{
  auto signature = cast<TaskBindingType>(getTask().getType()).getSignature();
  if(!llvm::equal(signature.getInputs(), getArgs().getTypes()) ||
     signature.getNumResults() != 0 ||
     !llvm::all_of(signature.getInputs(), isTaskArgumentType))
    return emitOpError(
        "requires instance or scalar operands matching a task signature with no results");
  return success();
}

namespace mlir::realm {
  bool isTaskArgumentType(Type type)
  {
    return isa<InstanceType>(type) || type.isSignlessInteger(32) ||
           type.isSignlessInteger(64) || type.isF32() || type.isF64();
  }

  namespace {
    template <typename Op>
    struct FoldEmptyMerge : OpRewritePattern<Op> {
      using OpRewritePattern<Op>::OpRewritePattern;

      LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override
      {
        if(!op.getEvents().empty())
          return failure();
        rewriter.replaceOp(op, ReadyOp::create(rewriter, op.getLoc(),
                                               EventType::get(rewriter.getContext()))
                                   .getEvent());
        return success();
      }
    };
  } // namespace

  void MergeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context)
  {
    patterns.add<FoldEmptyMerge<MergeOp>>(context);
  }

  void JoinSettledOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context)
  {
    patterns.add<FoldEmptyMerge<JoinSettledOp>>(context);
  }
} // namespace mlir::realm
