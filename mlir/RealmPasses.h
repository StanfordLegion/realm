// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_PASSES_H
#define REALM_MLIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::realm {
  std::unique_ptr<Pass> createLowerRealmToEmitCPass();
  std::unique_ptr<Pass> createSimplifyRealmDataflowPass();
  std::unique_ptr<Pass> createLowerRealmToLLVMPass();
  std::unique_ptr<Pass> createLowerAsyncToRealmPass();
  std::unique_ptr<Pass> createLowerGPUToRealmPass();
  void registerRealmPasses();
} // namespace mlir::realm

#endif // REALM_MLIR_PASSES_H
