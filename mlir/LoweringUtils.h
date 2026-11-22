// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_LOWERING_UTILS_H
#define REALM_MLIR_LOWERING_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::realm {
  inline StringAttr getTaskInitializer(ModuleOp module)
  {
    auto name = module->getAttrOfType<StringAttr>("realm.task_initializer");
    return name && !name.getValue().empty() ? name : StringAttr{};
  }

  template <typename... UnsupportedTypes>
  LogicalResult checkUnsupportedTypes(ModuleOp module, StringRef backend,
                                      StringRef alternative)
  {
    WalkResult checked = module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto check = [&](TypeRange types) -> LogicalResult {
        for(Type type : types)
          if(isa<UnsupportedTypes...>(type))
            return op->emitOpError() << backend << " lowering does not support " << type
                                     << ". Use " << alternative;
        return success();
      };
      if(auto fn = dyn_cast<FunctionOpInterface>(op))
        if(failed(check(fn.getArgumentTypes())) || failed(check(fn.getResultTypes())))
          return WalkResult::interrupt();
      if(failed(check(op->getOperandTypes())) || failed(check(op->getResultTypes())))
        return WalkResult::interrupt();
      for(Region &region : op->getRegions())
        for(Block &block : region)
          if(failed(check(block.getArgumentTypes())))
            return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(checked.wasInterrupted());
  }
} // namespace mlir::realm

#endif // REALM_MLIR_LOWERING_UTILS_H
