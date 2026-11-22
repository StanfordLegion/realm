// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_OPS
#define REALM_MLIR_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Realm_Dialect : Dialect {
  let name = "realm";
  let cppNamespace = "::mlir::realm";
  let summary = "Asynchronous operation submission to Realm";
  let description = [{
    Operations submit requests in the order of the enclosing control flow.
    An `after` clause supplies preconditions for the requested work using
    Realm's native merge semantics. With no preconditions the request uses
    `NO_EVENT`. To wait for all inputs regardless of poison, first combine
    the inputs with `realm.event.join_settled`.

    Allocation and destruction are ordered host submissions to Realm's
    asynchronous allocator. A deletion precondition must never come to
    depend on the readiness of an allocation submitted after that deletion.
    Lowerings must preserve this ordering even when the returned events do
    not express a direct SSA dependence between requests.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class Realm_Type<string name, string keyword> : TypeDef<Realm_Dialect, name> {
  let mnemonic = keyword;
}

def Realm_Event : Realm_Type<"Event", "event">;
def Realm_Instance : Realm_Type<"Instance", "instance">;
def Realm_Processor : Realm_Type<"Processor", "processor">;
def Realm_Task : Realm_Type<"Task", "task">;
def Realm_TaskBinding : Realm_Type<"TaskBinding", "task_binding"> {
  let parameters = (ins "::mlir::FunctionType":$signature);
  let assemblyFormat = "`<` $signature `>`";
}
def Realm_Memory : Realm_Type<"Memory", "memory">;
def Realm_Layout : Realm_Type<"Layout", "layout">;
def Realm_DenseLayout : Realm_Type<"DenseLayout", "dense_layout">;
def Realm_DenseIndexSpace : Realm_Type<"DenseIndexSpace", "dense_index_space">;
def Realm_Reservation : Realm_Type<"Reservation", "reservation">;
def Realm_BarrierPhase : Realm_Type<"BarrierPhase", "barrier_phase">;

def Realm_IndexSpace : Realm_Type<"IndexSpace", "index_space"> {
  let parameters = (ins "unsigned":$rank, "::mlir::IntegerType":$coordinateType);
  let assemblyFormat = "`<` $rank `,` $coordinateType `>`";
  let genVerifyDecl = 1;
}

class Realm_Op<string mnemonic, list<Trait> traits = []>
    : Op<Realm_Dialect, mnemonic, traits>;

class Realm_Kind<string name, string description, list<I32EnumCase> cases>
    : I32Enum<name, description, cases> {
  let cppNamespace = "::mlir::realm";
}

def Realm_ProcessorKind : Realm_Kind<"ProcessorKind", "Realm processor kind", [
  I32EnumCase<"TocProc", 1, "toc_proc">,
  I32EnumCase<"LocProc", 2, "loc_proc">,
  I32EnumCase<"UtilProc", 3, "util_proc">,
  I32EnumCase<"IoProc", 4, "io_proc">,
  I32EnumCase<"ProcGroup", 5, "proc_group">,
  I32EnumCase<"ProcSet", 6, "proc_set">,
  I32EnumCase<"OmpProc", 7, "omp_proc">,
  I32EnumCase<"PyProc", 8, "py_proc">
]>;

def Realm_MemoryKind : Realm_Kind<"MemoryKind", "Realm memory kind", [
  I32EnumCase<"GlobalMem", 1, "global_mem">,
  I32EnumCase<"SystemMem", 2, "system_mem">,
  I32EnumCase<"RegdmaMem", 3, "regdma_mem">,
  I32EnumCase<"SocketMem", 4, "socket_mem">,
  I32EnumCase<"ZCopyMem", 5, "z_copy_mem">,
  I32EnumCase<"GpuFbMem", 6, "gpu_fb_mem">,
  I32EnumCase<"DiskMem", 7, "disk_mem">,
  I32EnumCase<"HdfMem", 8, "hdf_mem">,
  I32EnumCase<"FileMem", 9, "file_mem">,
  I32EnumCase<"Level3Cache", 10, "level3_cache">,
  I32EnumCase<"Level2Cache", 11, "level2_cache">,
  I32EnumCase<"Level1Cache", 12, "level1_cache">,
  I32EnumCase<"GpuManagedMem", 13, "gpu_managed_mem">,
  I32EnumCase<"GpuDynamicMem", 14, "gpu_dynamic_mem">
]>;

class Realm_KindAttr<Realm_Kind kind, string name>
    : EnumAttr<Realm_Dialect, kind, name> {
  let assemblyFormat = "`<` $value `>`";
}

def Realm_ProcessorKindAttr : Realm_KindAttr<Realm_ProcessorKind, "processor_kind">;
def Realm_MemoryKindAttr : Realm_KindAttr<Realm_MemoryKind, "memory_kind">;

// The runtime machine model can change between queries.
def Realm_FindProcessorOp : Realm_Op<"machine.find_processor", [AttrSizedOperandSegments]> {
  let summary = "Find the first processor matching the machine query";
  let description = [{
    Query an initialized Realm runtime synchronously. All supplied filters
    apply. An empty match returns `NO_PROC` and false.
  }];
  let arguments = (ins Realm_ProcessorKindAttr:$kind, UnitAttr:$local,
                       Optional<Realm_Memory>:$sameAddressSpace,
                       Optional<Realm_Memory>:$affinity);
  let results = (outs Realm_Processor:$processor, I1:$found);
  let assemblyFormat = [{
    `kind` qualified($kind) (`local` $local^)?
    (`same_address_space_as` $sameAddressSpace^)?
    (`with_affinity_to` $affinity^)? attr-dict
  }];
}

def Realm_FindMemoryOp : Realm_Op<"machine.find_memory", [AttrSizedOperandSegments]> {
  let summary = "Find the first memory matching the machine query";
  let description = [{
    Query an initialized Realm runtime synchronously. All supplied filters
    apply. Capacity is the minimum total memory capacity in bytes. An empty
    match returns `NO_MEMORY` and false.
  }];
  let arguments = (ins Realm_MemoryKindAttr:$kind, UnitAttr:$local,
                       Optional<Realm_Processor>:$sameAddressSpace,
                       Optional<Realm_Processor>:$affinity,
                       OptionalAttr<UI64Attr>:$minCapacity);
  let results = (outs Realm_Memory:$memory, I1:$found);
  let assemblyFormat = [{
    `kind` qualified($kind) (`local` $local^)?
    (`same_address_space_as` $sameAddressSpace^)?
    (`with_affinity_to` $affinity^)?
    (`min_capacity` $minCapacity^)? attr-dict
  }];
}

def Realm_ReadyOp : Realm_Op<"event.ready", [Pure]> {
  let summary = "Return the already satisfied Realm NO_EVENT sentinel";
  let results = (outs Realm_Event:$event);
  let assemblyFormat = "attr-dict";
}

def Realm_MergeOp : Realm_Op<"event.merge"> {
  let summary = "Merge the input events using Realm's native poison propagation";
  let arguments = (ins Variadic<Realm_Event>:$events);
  let results = (outs Realm_Event:$event);
  let assemblyFormat = "` ` `[` $events `]` attr-dict";
  let hasCanonicalizer = 1;
}

def Realm_JoinSettledOp : Realm_Op<"event.join_settled"> {
  let summary = "Signal success after every input event has triggered, ignoring poison";
  let arguments = (ins Variadic<Realm_Event>:$events);
  let results = (outs Realm_Event:$event);
  let assemblyFormat = "` ` `[` $events `]` attr-dict";
  let hasCanonicalizer = 1;
}

def Realm_AwaitOp : Realm_Op<"event.await"> {
  let summary = "Wait for a Realm event and report whether it was poisoned";
  let arguments = (ins Realm_Event:$event);
  let results = (outs I1:$poisoned);
  let assemblyFormat = "$event attr-dict";
}

def Realm_CreateInstanceOp : Realm_Op<"instance.create", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Submit instance allocation and return its handle and readiness event";
  let description = [{
    The handle can be available before allocation and may be null if Realm
    cannot obtain an instance slot. Uses of the storage require successful
    completion of the readiness event. Cancellation poisons readiness even
    when the handle exists. The current runtime lowering uses Realm's default
    allocation failure reporting, which can abort on out-of-memory.
  }];
  let arguments = (ins Realm_Memory:$memory,
                       AnyTypeOf<[Realm_Layout, Realm_DenseLayout]>:$layout,
                       Variadic<Realm_Event>:$dependencies);
  let results = (outs Realm_Instance:$instance, Realm_Event:$ready);
  let assemblyFormat = [{
    `in` $memory `layout` $layout custom<InstanceLayoutType>(type($layout))
    (`after` ` ` `[` $dependencies^ `]`)? attr-dict
  }];
}

def Realm_InstanceExistsOp : Realm_Op<"instance.exists", [Pure]> {
  let summary = "Test whether a value is a non-null instance handle";
  let arguments = (ins Realm_Instance:$instance);
  let results = (outs I1:$exists);
  let assemblyFormat = "$instance attr-dict";
}

def Realm_DenseSpaceOp : Realm_Op<"index_space.dense", [Pure]> {
  let summary = "Describe a zero-based, one-dimensional dense domain";
  let arguments = (ins I64:$extent);
  let results = (outs Realm_DenseIndexSpace:$space);
  let assemblyFormat = "$extent attr-dict";
}

def Realm_DenseLayoutOp : Realm_Op<"layout.dense", [Pure]> {
  let summary = "Describe contiguous storage for a single field with ID zero";
  let arguments = (ins Realm_DenseIndexSpace:$space, I64Attr:$fieldBytes);
  let results = (outs Realm_DenseLayout:$layout);
  let assemblyFormat = "$space `field_bytes` $fieldBytes attr-dict";
  let hasVerifier = 1;
}

def Realm_TaskRegisterOp : Realm_Op<"task.register", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Register a module-defined function on local processors of a kind";
  let arguments = (ins FlatSymbolRefAttr:$callee, I32Attr:$id,
                       Realm_ProcessorKindAttr:$kind);
  let results = (outs Realm_TaskBinding:$task, Realm_Event:$registered);
  let assemblyFormat = "$callee `id` $id `kind` qualified($kind) `as` qualified(type($task)) attr-dict";
  let hasVerifier = 1;
}

def Realm_SpawnTypedOp : Realm_Op<"spawn_typed", [AttrSizedOperandSegments]> {
  let summary = "Submit a typed task with scalar and instance arguments";
  let arguments = (ins Realm_TaskBinding:$task, Variadic<AnyType>:$args,
                       Realm_Processor:$processor, Variadic<Realm_Event>:$dependencies);
  let results = (outs Realm_Event:$done);
  let assemblyFormat = [{
    $task `(` $args `)` `on` $processor (`after` ` ` `[` $dependencies^ `]`)?
    attr-dict `:` qualified(type($task)) (`,` type($args)^)?
  }];
  let hasVerifier = 1;
}

def Realm_SpawnOp : Realm_Op<"spawn", [AttrSizedOperandSegments]> {
  let summary = "Submit a registered task and return its completion event";
  let description = [{
    A task handle identifies an implementation already registered on the target
    processor, or whose registration is covered by a precondition. This form
    packs the instance handles into the argument buffer as consecutive Realm
    instance IDs, in operand order. The task entry point uses that argument ABI.
  }];
  let arguments = (ins Realm_Task:$task, Variadic<Realm_Instance>:$instances,
                       Realm_Processor:$processor, Variadic<Realm_Event>:$dependencies);
  let results = (outs Realm_Event:$done);
  let assemblyFormat = [{
    $task `(` $instances `)` `on` $processor
    (`after` ` ` `[` $dependencies^ `]`)? attr-dict
  }];
}

def Realm_CopyOp : Realm_Op<"copy"> {
  let summary = "Submit a direct copy between a pair of instance fields";
  let arguments = (ins Realm_Instance:$source, I32Attr:$sourceField,
                       Realm_Instance:$target, I32Attr:$targetField,
                       AnyTypeOf<[Realm_IndexSpace, Realm_DenseIndexSpace]>:$space,
                       I64Attr:$bytes,
                       Variadic<Realm_Event>:$dependencies);
  let results = (outs Realm_Event:$done);
  let assemblyFormat = [{
    $source `field` $sourceField `to` $target `field` $targetField
    `over` $space `bytes` $bytes (`after` ` ` `[` $dependencies^ `]`)?
    attr-dict `:` qualified(type($space))
  }];
  let hasVerifier = 1;
}

def Realm_DestroyInstanceOp : Realm_Op<"instance.destroy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Submit deferred instance destruction";
  let arguments = (ins Realm_Instance:$instance, Variadic<Realm_Event>:$dependencies);
  let assemblyFormat = [{
    $instance (`after` ` ` `[` $dependencies^ `]`)? attr-dict
  }];
}

def Realm_CreateReservationOp : Realm_Op<"reservation.create"> {
  let summary = "Create a Realm reservation";
  let results = (outs Realm_Reservation:$reservation);
  let assemblyFormat = "attr-dict";
}

def Realm_AcquireOp : Realm_Op<"reservation.acquire"> {
  let summary = "Request exclusive ownership and return the grant event";
  let arguments = (ins Realm_Reservation:$reservation, Variadic<Realm_Event>:$dependencies);
  let results = (outs Realm_Event:$granted);
  let assemblyFormat = "$reservation (`after` ` ` `[` $dependencies^ `]`)? attr-dict";
}

def Realm_ReleaseOp : Realm_Op<"reservation.release"> {
  let summary = "Schedule relinquishing a granted reservation";
  let arguments = (ins Realm_Reservation:$reservation, Variadic<Realm_Event>:$dependencies);
  let assemblyFormat = "$reservation (`after` ` ` `[` $dependencies^ `]`)? attr-dict";
}

def Realm_DestroyReservationOp : Realm_Op<"reservation.destroy"> {
  let summary = "Submit deferred reservation destruction";
  let arguments = (ins Realm_Reservation:$reservation, Variadic<Realm_Event>:$dependencies);
  let assemblyFormat = "$reservation (`after` ` ` `[` $dependencies^ `]`)? attr-dict";
}

def Realm_AdvanceBarrierOp : Realm_Op<"barrier.advance", [Pure]> {
  let summary = "Get the handle for the next barrier phase";
  let arguments = (ins Realm_BarrierPhase:$phase);
  let results = (outs Realm_BarrierPhase:$next);
  let assemblyFormat = "$phase attr-dict";
}

def Realm_BarrierEventOp : Realm_Op<"barrier.as_event", [Pure]> {
  let summary = "Get the event for completion of a barrier phase";
  let arguments = (ins Realm_BarrierPhase:$phase);
  let results = (outs Realm_Event:$event);
  let assemblyFormat = "$phase attr-dict";
}

def Realm_ArriveBarrierOp : Realm_Op<"barrier.arrive"> {
  let summary = "Submit one arrival carrying an i32 reduction contribution";
  let description = [{
    The phase must be configured for an i32 contribution. Every precondition
    must trigger successfully. Realm's barrier implementation can assert on
    a poisoned precondition.
  }];
  let arguments = (ins Realm_BarrierPhase:$phase, I32:$value, Variadic<Realm_Event>:$dependencies);
  let assemblyFormat = "$phase `value` $value (`after` ` ` `[` $dependencies^ `]`)? attr-dict";
}

def Realm_BarrierResultOp : Realm_Op<"barrier.get_result"> {
  let summary = "Try to read the i32 reduction result of a barrier phase";
  let description = [{
    The phase must be configured for an i32 result. Inspect the returned value
    only when the availability flag is true.
  }];
  let arguments = (ins Realm_BarrierPhase:$phase);
  let results = (outs I32:$value, I1:$available);
  let assemblyFormat = "$phase attr-dict";
}

#endif // REALM_MLIR_OPS
