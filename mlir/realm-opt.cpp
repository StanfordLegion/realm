// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv)
{
  mlir::realm::registerRealmPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::realm::RealmDialect, mlir::arith::ArithDialect,
                  mlir::emitc::EmitCDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Realm MLIR optimizer\n", registry));
}
