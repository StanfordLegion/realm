// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv)
{
  mlir::realm::registerRealmPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::realm::RealmDialect, mlir::arith::ArithDialect,
                  mlir::emitc::EmitCDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                  mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Realm MLIR optimizer\n", registry));
}
