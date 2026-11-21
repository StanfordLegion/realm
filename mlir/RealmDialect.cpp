// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RealmDialect.cpp.inc"
#include "RealmEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RealmAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "RealmTypes.cpp.inc"

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

namespace mlir::realm {
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
