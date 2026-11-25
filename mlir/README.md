<!--
Copyright 2026 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Realm MLIR dialect

The `realm` dialect models a host program submitting asynchronous work to Realm. `realm-opt` can verify the IR, simplify dependency graphs, and lower operations through EmitC or directly to LLVM. The examples show Realm orchestration with CPU tasks and CUDA kernels. The interface is a prototype.

## Build and inspect IR

Run the commands below from the Realm repository root. Use CMake 3.22 or newer and an LLVM/MLIR installation with headers, libraries, `mlir-tblgen`, `mlir-opt`, and `mlir-translate`. Set `LLVM_PREFIX` to that installation and `MLIR_DIR` to the directory containing `MLIRConfig.cmake`. LLVM/MLIR 22.1.7 is the tested baseline. The examples below use the Homebrew layout.

```sh
LLVM_PREFIX="${LLVM_PREFIX:-$(brew --prefix llvm@22)}"
MLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir"

cmake -S mlir -B build/mlir -DMLIR_DIR="$MLIR_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/mlir --parallel
ctest --test-dir build/mlir --output-on-failure

build/mlir/realm-opt --verify-roundtrip mlir/examples/pipeline.mlir
build/mlir/realm-opt --simplify-realm-dataflow mlir/examples/copy-staging.mlir
build/mlir/realm-opt --simplify-realm-dataflow mlir/examples/transitive-reduction.mlir
```

This build works without a Realm installation. The IR tests exercise parsing, round trips, diagnostics, and lowering. When `FileCheck` is available, they also verify the optimized copy and dependency edges and check that the lowered transitive-reduction example omits the merge.

## Run the CPU examples

Use a Realm installation built with `REALM_ENABLE_LLVM=ON` and an LLVM version compatible with the MLIR tools. The following configures a small local Realm build and installs it under `build/realm-jit-install`.

```sh
cmake -S . -B build/realm-jit -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/realm-jit-install" -DCMAKE_INSTALL_LIBDIR=lib \
  -DLLVM_CONFIG_EXECUTABLE="$LLVM_PREFIX/bin/llvm-config" \
  -DREALM_ENABLE_LLVM=ON -DREALM_MAX_DIM=1 \
  -DREALM_BUILD_TESTS=OFF -DREALM_BUILD_EXAMPLES=OFF \
  -DREALM_BUILD_BENCHMARKS=OFF -DREALM_BUILD_TUTORIALS=OFF -DREALM_BUILD_DOCS=OFF \
  -DREALM_ENABLE_CUDA=OFF -DREALM_ENABLE_HIP=OFF -DREALM_ENABLE_HDF5=OFF \
  -DREALM_ENABLE_OPENMP=OFF -DREALM_ENABLE_PAPI=OFF -DREALM_ENABLE_PYTHON=OFF
cmake --build build/realm-jit --parallel
cmake --install build/realm-jit

cmake -S mlir -B build/mlir-jit -DMLIR_DIR="$MLIR_DIR" \
  -DRealm_DIR="$PWD/build/realm-jit-install/lib/cmake/realm" \
  -DREALM_MLIR_BUILD_LLVM_TASK_EXAMPLE=ON -DREALM_MLIR_TEST_RUNTIME=ON
cmake --build build/mlir-jit --parallel
ctest --test-dir build/mlir-jit --output-on-failure
build/mlir-jit/examples/realm-llvm-task -ll:cpu 2
```

For an existing installation, set `Realm_DIR` to the directory containing its `RealmConfig.cmake`. `REALM_MLIR_TEST_RUNTIME=ON` adds a machine-query test that also works with a Realm build without LLVM JIT.

Each example has its own MLIR file in `mlir/examples/`. The [shared C++ launcher](examples/main.cpp) registers the [LLVM task bodies](examples/tasks.mlir), which the build exports to `tasks.ll`. The executable runs and checks all eight examples.

| Example | What to look for |
| --- | --- |
| [pipeline.mlir](examples/pipeline.mlir) | Discovers a CPU and a memory, then submits two allocations and two tasks before the start event fires. The tasks produce `42`, and the program exercises deferred destruction. |
| [fork-join.mlir](examples/fork-join.mlir) | Two tasks produce `21` each. A third waits for both and produces `42`. |
| [fault-cleanup.mlir](examples/fault-cleanup.mlir) | A native merge reports poison while another use is pending. `join_settled` keeps the instance live until cleanup is eligible. |
| [copy-staging.mlir](examples/copy-staging.mlir) | The optimized copy produces identical bytes and leaves the temporary untouched. |
| [transitive-reduction.mlir](examples/transitive-reduction.mlir) | The final spawn depends on a single event after transitive reduction. |
| [exclusive-reservation.mlir](examples/exclusive-reservation.mlir) | Two exclusive acquisitions serialize increments to the same counter. |
| [barrier-phases.mlir](examples/barrier-phases.mlir) | Worker completions drive arrivals. The two phases produce sums of `10` and `26`. |
| [replay.mlir](examples/replay.mlir) | The same function runs with different bindings. The first invocation's completion becomes the second invocation's input event. |

The build leaves each example's optimized MLIR, EmitC, and generated C++ in `build/mlir-jit/examples/`. The replay demo illustrates graph rebinding. Legion trace capture and Subgraph replay are future integrations.

## Compile directly to LLVM IR

`--convert-realm-to-llvm` produces LLVM dialect with calls to the Realm C ABI and a small runtime library. Build against an installed 64-bit Realm with the C API. The direct path works with `REALM_ENABLE_LLVM=OFF` in Realm. The application build uses `realm-opt`, `mlir-translate`, and `llc`, then links the generated object and `RealmMLIRLLVMRuntime` against Realm.

```sh
cmake -S mlir -B build/mlir-native -DMLIR_DIR="$MLIR_DIR" \
  -DRealm_DIR="$PWD/build/realm-jit-install/lib/cmake/realm" \
  -DREALM_MLIR_BUILD_LLVM_EXAMPLES=ON
cmake --build build/mlir-native --parallel
build/mlir-native/examples/realm-llvm-cpu -ll:cpu 2
```

[llvm-cpu.mlir](examples/llvm-cpu.mlir) defines its own tasks, layout, and `main`. It submits two allocations, a scalar-argument task, a copy, and another task that takes a floating-point scalar. It then checks two tasks outlined from Async and verifies the final value of `44`. The generated `.llvm.mlir` and `.ll` files are next to the executable.

### Collective task startup

Set the module attribute `realm.task_initializer = "my_module_init"` to emit a C ABI initializer that returns a Realm registration-readiness event. Within that module, the lowering collects each task ID and processor kind once and checks that all uses agree on the implementation. The generated initializer registers on the matching local processors. The compiled entries use those bindings and can run repeatedly.

Every job process calls the initializer once after Realm starts, in the same startup phase. Pass the combined local registration-readiness event to the program's first collective spawn. Realm merges the events contributed by every process before launching the task. The examples use the same protocol with a collective NOP and start the main host computation once that rendezvous completes. Modules that perform explicit registration can keep calling `realm.task.register` in their own setup code.

Private `func.func` definitions default to internal LLVM linkage. Imported declarations and the generated C ABI initializer have external linkage.

### Backend support

| Operations or types | EmitC | LLVM |
| --- | --- | --- |
| Machine queries, events, instance existence and destruction, instance-only `spawn` | Yes | Yes |
| Instance creation and copies | C++ pointers described by `!realm.layout` and `!realm.index_space<rank, coordinate-type>` | Value descriptors described by `!realm.dense_layout` and `!realm.dense_index_space` |
| Dense descriptor builders, `task.register`, `spawn_typed`, collective startup initializer | — | Yes |
| Reservations and barrier phases | Yes | — |

During LLVM execution, immediate C ABI errors terminate the program with the Realm status. Event poison remains observable through `realm.event.await`.

`--convert-async-to-realm` outlines straight-line token-only `async.execute` operations in a single-block entry. Set the module's `realm.task_initializer` and give the entry a `realm.async_task_base` `i32` attribute that begins an unused CPU task ID range. The entry uses its first argument as the local CPU processor for every outlined task. The pass lowers token waits to Realm waits and reports poison as a failed assertion. Async values and groups require additional runtime representation.

## Run the CUDA example

Use a Realm installation built with `REALM_ENABLE_CUDA=ON`, an NVIDIA driver and CUDA Toolkit, and an LLVM install with the NVPTX backend. Configure MLIR with `REALM_MLIR_BUILD_CUDA_EXAMPLE=ON` and point `Realm_DIR` at that installation. The test runs on one local NVIDIA GPU of compute capability 7.0 or newer.

```sh
cmake -S mlir -B build/mlir-cuda -DMLIR_DIR="$MLIR_DIR" \
  -DRealm_DIR="$REALM_CUDA_INSTALL/lib/cmake/realm" \
  -DREALM_MLIR_BUILD_LLVM_EXAMPLES=ON -DREALM_MLIR_BUILD_CUDA_EXAMPLE=ON
cmake --build build/mlir-cuda --parallel
ctest --test-dir build/mlir-cuda --output-on-failure
build/mlir-cuda/examples/realm-llvm-cuda -ll:cpu 1 -ll:gpu 1
```

[llvm-cuda.mlir](examples/llvm-cuda.mlir) runs two separate Realm GPU tasks and then two kernel launches within one task. [gpu-input.mlir](examples/gpu-input.mlir) expresses the same allocation, upload, launches, download, and cleanup in the GPU dialect. All three paths check `i + 14` across 513 elements. [cuda-kernels.mlir](examples/cuda-kernels.mlir) supplies the NVVM device entry point and is compiled to PTX. The driver adapter loads the PTX and launches on the stream Realm assigns to the task.

`--convert-gpu-to-realm` lowers the host side of a straight-line, token-based function marked `realm.gpu_entry`. Set the module's `realm.task_initializer`. The first two arguments bind a local CUDA processor and its framebuffer memory. Static, contiguous, rank-one memref arguments become already-ready borrowed Realm instances with field zero. The pass supports `gpu.wait`, asynchronous allocation and copies, and token-based kernel launch.

On each launch, set `realm.task_entry` to the host launcher function and `realm.task_id` to its application task ID. Declare the exact kernel symbol on the launcher with an attribute such as `realm.gpu_kernel = @kernels::@add_to_buffer`. Multiple host entries may share the same task ID for the same launcher. The launcher receives the converted kernel operands followed by the six grid and block dimensions as `i64`. It is responsible for launching the corresponding separately compiled device implementation and adapting its ABI. In the example, that implementation comes from `cuda-kernels.mlir`. The host pass leaves the original `gpu.module` available for a device pipeline, and the example uses `--symbol-dce` to remove the unused private device module from the host artifact before LLVM lowering.

An asynchronous `gpu.dealloc` is supported when its token is unused. The conversion submits the free with the allocation and every use of the owned instance as preconditions. A synchronous deallocation or a live deallocation token needs an event that reports actual destruction. The pass diagnoses both cases.

The [CUDA driver adapter](examples/run-cuda.cpp) also imports a caller-owned CUDA allocation and event, runs a Realm task against that storage, and returns a CUDA event recorded after successful Realm completion. The adapter uses a host wait and records the CUDA event before handing it to a consumer stream. The caller keeps the allocation alive through all Realm uses. The check expects `28` in each element after round-trip.

### Early GPU submission

[Realm PR #482](https://github.com/StanfordLegion/realm/pull/482) uses `DeferredEffectsProperty` and the stream-aware task prototype in compiled subgraphs. Same-GPU successors can be submitted once the predecessor has recorded its stream token. Ordinary `realm.spawn` and `realm.spawn_typed` in this prototype use Realm completion-event dependencies.

For ordinary programs, an early path needs to distinguish a recorded device dependency from a triggered Realm completion. A compiler or runtime would have to verify context compatibility, hold the recorded token until all stream waits are enqueued, and establish that registration, instance metadata, allocation readiness, and any other host-visible prerequisites are ready. The consumer must order every dependent effect on its stream. An unresolved host, copy, remote or cross-GPU dependency stays on the normal completion path. The failure policy must also account for work that has already been enqueued when a predecessor fails. Instructions for validating the current examples on the cluster are in [CUDA validation](CUDA_VALIDATION.md).

## Use the dialect

MLIR control flow determines when the host issues requests. An `after [...]` list supplies event preconditions for the work itself. `realm.event.await` is an explicit wait and returns `true` for poison.

### Allocator issue order

Allocations and pending frees must be submitted in topological order. A free's precondition must never come to depend, even through a later user-event trigger or task, on the readiness of an allocation submitted after the free. Lowerings preserve the allocator-visible order of allocation and destruction requests. Clients that introduce dependencies through opaque events or external code must uphold the same contract.

Submit a free as soon as its last-use precondition is available. The allocator can take that pending free into account when it receives a later allocation request. Use the new instance's readiness event before accessing its storage.

A host function can submit an allocation, launch, and destruction using an initialized runtime and a registered task binding.

```mlir
%buffer, %ready = realm.instance.create in %memory layout %layout after [%start]
%has_buffer = realm.instance.exists %buffer
scf.if %has_buffer {
  %done = realm.spawn %task(%buffer) on %cpu after [%ready]
  %settled = realm.event.join_settled [%ready, %done]
  realm.instance.destroy %buffer after [%settled]
}
```

| Operations | Current contract |
| --- | --- |
| `machine.find_processor`, `machine.find_memory` | Query an initialized Realm runtime. Return a handle and `found` flag. Filters include kind, address space, and affinity. Memory capacity means total capacity in bytes. |
| `instance.create`, `instance.exists`, `instance.destroy` | Creation returns an instance handle and a separate readiness event. `exists` tests handle presence. Destruction accepts preconditions and has zero results. |
| `index_space.dense`, `layout.dense` | Build a zero-based rank-one `!realm.dense_index_space` from a nonnegative `i64` extent and a `!realm.dense_layout` for field zero from that domain and a positive field width in bytes. Spell out `: !realm.dense_layout` after the layout operand in `instance.create`. |
| `spawn` | Takes an already registered task ID as `!realm.task`, a processor, instance arguments, and preconditions. Packs the instance IDs in operand order and returns task completion. Supply registration readiness as a dependency when needed. |
| `task.register` | Binds a module-defined function to an application task ID on local processors of the specified kind. Returns the typed binding and a registration-readiness event. With explicit registration, order the first spawn after that event. |
| `spawn_typed` | Submits work with arguments matching the binding and returns task completion. The task takes instances, `i32`, `i64`, `f32`, or `f64` and has no SSA results. The generated payload packs one 64-bit word per argument. |
| `copy` | Submits a single-field direct copy over an index space. Supply matching layout and field information. Field ID `-1` is reserved for Realm fills. |
| `event.ready`, `event.merge`, `event.join_settled` | Produce the `NO_EVENT` sentinel, native merge, and all-inputs-settled join respectively. Native merge can report poison early. The settled join waits for every input and suppresses poison. |
| `reservation.create`, `reservation.acquire`, `reservation.release`, `reservation.destroy` | Model an exclusive acquisition and the associated lifetime. The grant event orders protected work. Release is valid after a successful grant and can be deferred. |
| `barrier.advance`, `barrier.as_event`, `barrier.arrive`, `barrier.get_result` | Work with a phase of a Realm barrier configured by the caller for an `i32` reduction. An arrival contributes one value. The phase event represents completion of the counted arrivals, and `get_result` returns a value plus an availability flag. |

Use `!realm.index_space<rank, coordinate-type>` for a domain. The EmitC lowering accepts `si32`, `ui32`, and `si64` coordinates. Supply an `IndexSpaceGeneric` of the matching dimension and coordinate type, within the installed Realm build's `REALM_MAX_DIM`. Processor and memory kind attributes use `#realm.processor_kind<loc_proc>` and `#realm.memory_kind<system_mem>` syntax. The enum definitions list all accepted spellings. [Types and kinds](RealmOps.td)

### Failure and cleanup

Keep allocation readiness on every path that uses an instance's storage. Realm can return a null handle when it cannot obtain an instance slot. A poisoned readiness event can also accompany a live handle, such as on cancellation. Schedule destruction for that handle once its uses are settled. The prototype passes an empty profiling request set, so native Realm can abort the process on out-of-memory.

| Operation | Effect of a poisoned precondition |
| --- | --- |
| `instance.create` | Cancels allocation and poisons the readiness event. |
| `spawn`, `spawn_typed`, `copy` | Cancel the work and poison the completion event. |
| `reservation.acquire` | Skips acquisition and poisons the grant event. |
| `reservation.release`, `instance.destroy` | Skip the pending effect, which can leave a held reservation or a live instance. |
| `reservation.destroy` | A deferred request can be skipped. An already poisoned event can fault at submission. |
| `barrier.arrive` | Requires successful preconditions. A poisoned precondition can assert in Realm. |

Use `join_settled` over the original in-flight uses to make cleanup wait for each of them despite early poison in a native merge. Before arranging reservation release on a fault path, establish that the reservation was granted. The barrier example assumes a successful gate and successful worker completions. [Fault cleanup](examples/fault-cleanup.mlir), [Barrier phases](examples/barrier-phases.mlir)

The EmitC examples take task IDs, user events, and barriers from their C++ launcher. The direct LLVM examples define and register their tasks in MLIR and use runtime helpers to start the job and access local instance storage.

### Optimization contracts

`--simplify-realm-dataflow` reduces redundant `spawn` dependencies in functions annotated with `realm.assume_success`. The annotation selects successful-dataflow equivalence and permits changes to the timing of poison propagation and profiling. Transitive reduction follows task completion edges.

For staging elimination, also annotate the function with `realm.private_staging`. The caller guarantees a legal direct transfer, disjoint instances, a private temporary whose contents the program can discard, a destination ready at the initial precondition, and source data that stays stable through completion. The pass checks for adjacent compatible copies over the same domain. For an owned temporary, it can preserve the allocation, handle checks, and destruction while removing the staging copy. Any cleanup join that used the first copy's event must also cover allocation readiness and the second copy's completion. The copy staging example shows both the baseline `staged_copy` and the annotated `elide_staging`. [Copy staging](examples/copy-staging.mlir), [Pass implementation](SimplifyRealmDataflow.cpp)

## Source map for extensions

| File | Role |
| --- | --- |
| [`RealmOps.td`](RealmOps.td) | TableGen definitions for types, operands, results, and textual assembly. Read an operation's `ins`, `outs`, and `assemblyFormat` alongside its use in an example. |
| [`RealmDialect.cpp`](RealmDialect.cpp) | Dialect registration and custom verifiers. |
| [`LowerRealmToEmitC.cpp`](LowerRealmToEmitC.cpp) | `--convert-realm-to-emitc` handles the supported runtime operations and machine queries. The old machine-only flag is a compatibility alias for the same pass. |
| [`LowerRealmToLLVM.cpp`](LowerRealmToLLVM.cpp) and [`RealmLLVMRuntime.cpp`](RealmLLVMRuntime.cpp) | Direct LLVM lowering and the companion C ABI library. |
| [`LowerAsyncToRealm.cpp`](LowerAsyncToRealm.cpp) and [`LowerGPUToRealm.cpp`](LowerGPUToRealm.cpp) | Scoped host-side conversions from upstream Async and GPU operations. |
| [`RealmPasses.h`](RealmPasses.h) | Pass factories and registration for C++ clients linking `MLIRRealmTransforms`. |
| [`RealmRuntime.h`](RealmRuntime.h) and [`RealmMachineQueries.h`](RealmMachineQueries.h) | C++ helpers called by generated host code. |
| [`examples/CMakeLists.txt`](examples/CMakeLists.txt) and [`test/`](test/) | Build pipelines and verification fixtures for both EmitC and direct LLVM examples. |

Build `realm-opt` to generate the `.inc` headers included by `RealmDialect.h`. They live in the MLIR build directory. Point editor tooling at the build's `compile_commands.json` to resolve both the generated headers and the installed MLIR headers. When extending an operation, update its TableGen definition, its lowering and C++ helper as needed, and a fixture that exercises the behavior through the corresponding entry point.

`realm-opt` registers `arith`, `async`, `cf`, `emitc`, `func`, `gpu`, `llvm`, `memref`, `realm`, and `scf` along with `--canonicalize`, `--cse`, and `--symbol-dce`. When generating C++, run the relevant upstream EmitC conversions for the other dialects before invoking `mlir-translate`.
