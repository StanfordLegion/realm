// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_DIALECT_H
#define REALM_MLIR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// CMake generates the Realm .inc files from RealmOps.td in the build directory.
#include "RealmDialect.h.inc"
#include "RealmEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "RealmAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "RealmTypes.h.inc"

#define GET_OP_CLASSES
#include "RealmOps.h.inc"

namespace mlir::realm {
  bool isTaskArgumentType(Type type);
}

#endif // REALM_MLIR_DIALECT_H
