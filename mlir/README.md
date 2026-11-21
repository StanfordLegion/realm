<!--
Copyright 2026 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Realm MLIR dialect

The `realm` dialect models a host program submitting asynchronous work to Realm. `realm-opt` can verify the IR, simplify dependency graphs, and lower the supported operations to EmitC calls into Realm's C++ API. The runnable examples use the Realm dialect for orchestration and MLIR's LLVM dialect for CPU task bodies. The interface is a prototype.

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

## Use the dialect

MLIR control flow determines when the host issues requests. An `after [...]` list supplies event preconditions for the work itself. `realm.event.await` is an explicit wait and returns `true` for poison.

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
| `spawn` | Takes an already registered task ID as `!realm.task`, a processor, instance arguments, and preconditions. Packs the instance IDs in operand order and returns task completion. Supply registration readiness as a dependency when needed. |
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
| `spawn`, `copy` | Cancel the work and poison the completion event. |
| `reservation.acquire` | Skips acquisition and poisons the grant event. |
| `reservation.release`, `instance.destroy` | Skip the pending effect, which can leave a held reservation or a live instance. |
| `reservation.destroy` | A deferred request can be skipped. An already poisoned event can fault at submission. |
| `barrier.arrive` | Requires successful preconditions. A poisoned precondition can assert in Realm. |

Use `join_settled` over the original in-flight uses to make cleanup wait for each of them despite early poison in a native merge. Before arranging reservation release on a fault path, establish that the reservation was granted. The barrier example assumes a successful gate and successful worker completions. [Fault cleanup](examples/fault-cleanup.mlir), [Barrier phases](examples/barrier-phases.mlir)

Task registration, user-event creation, and barrier creation are supplied by the example's host code. The MLIR functions take those resources as inputs. The current task example uses the local 64-bit host task ABI and provides a registration callback to access local CPU-visible storage.

### Optimization contracts

`--simplify-realm-dataflow` reduces redundant `spawn` dependencies in functions annotated with `realm.assume_success`. The annotation selects successful-dataflow equivalence and permits changes to the timing of poison propagation and profiling. Transitive reduction follows task completion edges.

For staging elimination, also annotate the function with `realm.private_staging`. The caller guarantees a legal direct transfer, disjoint instances, a private temporary whose contents the program can discard, a destination ready at the initial precondition, and source data that stays stable through completion. The pass checks for adjacent compatible copies over the same domain. For an owned temporary, it can preserve the allocation, handle checks, and destruction while removing the staging copy. Any cleanup join that used the first copy's event must also cover allocation readiness and the second copy's completion. The copy staging example shows both the baseline `staged_copy` and the annotated `elide_staging`. [Copy staging](examples/copy-staging.mlir), [Pass implementation](SimplifyRealmDataflow.cpp)

## Source map for extensions

| File | Role |
| --- | --- |
| [`RealmOps.td`](RealmOps.td) | TableGen definitions for types, operands, results, and textual assembly. Read an operation's `ins`, `outs`, and `assemblyFormat` alongside its use in an example. |
| [`RealmDialect.cpp`](RealmDialect.cpp) | Dialect registration and custom verifiers. |
| [`LowerRealmToEmitC.cpp`](LowerRealmToEmitC.cpp) | `--convert-realm-to-emitc` handles the supported runtime operations and machine queries. The old machine-only flag is a compatibility alias for the same pass. |
| [`RealmPasses.h`](RealmPasses.h) | Pass factories and registration for C++ clients linking `MLIRRealmTransforms`. |
| [`RealmRuntime.h`](RealmRuntime.h) and [`RealmMachineQueries.h`](RealmMachineQueries.h) | C++ helpers called by generated host code. |
| [`examples/CMakeLists.txt`](examples/CMakeLists.txt) and [`test/`](test/) | Translation commands and verification fixtures. The examples lower host MLIR through EmitC, export task MLIR to LLVM IR, and run both through Realm. |

Build `realm-opt` to generate the `.inc` headers included by `RealmDialect.h`. They live in the MLIR build directory. Point editor tooling at the build's `compile_commands.json` to resolve both the generated headers and the installed MLIR headers. When extending an operation, update its TableGen definition, its lowering and C++ helper as needed, and a fixture that exercises the behavior through the corresponding entry point.

`realm-opt` registers `arith`, `emitc`, `func`, `realm`, and `scf` along with `--canonicalize` and `--cse`. When generating C++, run the relevant upstream EmitC conversions for the other dialects before invoking `mlir-translate`.
