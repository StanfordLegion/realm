# Compiled subgraphs: full feature set on the hardened engine

Supersedes the CPU-task-only subset of #417. Realm subgraphs are now
compiled-only: `Subgraph::create_subgraph` compiles the definition into
per-processor ready queues, NUMA-placed counters and copy plans, and
`instantiate` replays it. Everything the engine does not implement is a hard
error naming the operation and the feature. `docs/subgraphs.md` is the
reference for what is supported, what is refused, the semantics, and the
remaining gaps.

23 commits, 38 files, +7550 / -2561. History is intended to be squashed.

## What landed

1. **Compiled-only engine.** Interpreted mode, `ExecutionMode` and the
   separate "bgwork partition" are gone. One operation table per subgraph with
   NUMA-aware placement: precondition counters live with their writers when
   those share a domain, otherwise with the consumer; queue slots and tails in
   the consumer's domain; interpolation scratch per domain. Blocks are
   allocated on the right node once and recycled across instantiations.
2. **Graph priority.** `priority_adjust` of `instantiate` is the
   instantiation's priority. While an instantiation of priority P is active on
   a processor, and the external inputs its operations there depend on have
   triggered (compile-time analysis), that processor runs only work of
   priority P or higher, normal tasks and other instantiations alike; already
   started tasks always resume.
3. **Arrivals, external pre/postconditions, interpolation.** Preconditions are
   counted inputs with per-operation and per-postcondition bitsets, so a
   poisoned input skips exactly the operations downstream of it and poisons
   exactly the postconditions downstream of it, plus the finish event.
4. **Blocking, finish events, profiling.** Subgraph tasks may wait on events
   and ask for a finish event, which is created on demand. Profiling requests
   come from the definition and from `SubgraphInstantiationProfiling` (sparse
   per-kind lists, serializable for remote instantiation), merged per
   operation.
5. **GPU tasks.** `DeferredEffectsProperty` on `CodeDescriptor` marks a task
   whose work is all on the stream it is given; stream-aware prototypes imply
   it. The CUDA processor implements new `LocalTaskProcessor` hooks: push the
   context, assign a stream, record a CUDA-event token after the function.
   Same-GPU successors of a deferred-effects task start when its function
   returns and wait on the token in-stream; everything else waits for
   completion (stream event, or the context synchronizer on drivers without
   `cuCtxRecordEvent`). `-ll:pin_gpu` gives GPU processors a dedicated core.
6. **Copies, fills, reductions, indirections.** One `TransferDesc` per copy,
   analyzed at compile time and replayed by a `TransferOperation` per
   instantiation (fresh XDs; XD reuse waits for the DMA refactor). Typed
   indirections through `CopyDesc::add_indirection<N,T>()`, with type
   mismatches caught at compile.
7. **Remote XDs** are created per replay through the existing factory messages
   and complete through XD ids (came with 6).
8. **Gap document, stencil benchmark, layout.** `docs/subgraphs.md`;
   `benchmarks/stencil_subgraph` (tiled 5-point stencil with halo copies, run
   directly and as subgraph replays, result checked, CPU or GPU);
   implementation moved into `src/realm/subgraph/` split into compile,
   execution and API/lifecycle files.

## API changes

- `SubgraphDefinition::ExecutionMode` and the mode argument are removed.
- `Subgraph::destroy` returns an `Event` that covers outstanding
  instantiations.
- `instantiate` gains overloads taking `SubgraphInstantiationProfiling`; the
  legacy overload's `ProfilingRequestSet` must be empty.
- `CopyDesc` gains `indirects` and `add_indirection<N,T>()`;
  `redop_id`/`red_fold` apply to destination fields without their own
  reduction; `priority` is added to the instantiation's priority.
- `DeferredEffectsProperty` in `realm/codedesc.h`; `CodeDescriptor` now
  serializes portable properties.
- Tasks must be registered on their processor when the subgraph compiles;
  `TaskDesc::priority` must be 0; tasks must run on local CPUs or CUDA GPUs.
- Use after destroy (`instantiate` or a second `destroy`) is fatal while the
  slot is unused; a deferred-effects GPU task calling
  `Cuda::set_task_ctxsync_required(true)` is fatal.
- Internal: `LocalTaskProcessor` subgraph hooks; `TransferDesc::analyze` split
  from `perform_analysis`; `TransferOperation` constructor with explicit
  profiling requests; `GPUStream::add_event(..., return_event)`;
  `ContextSynchronizer::add_notification`; options `-ll:subgraph_poll`,
  `-ll:pin_gpu`.

## Refused (fatal) in this version

`SERIALIZABLE` and `CONCURRENT`, nested instantiations, reservation
acquires/releases, tasks on remote nodes or on processors other than CPUs and
CUDA GPUs, collective pre/postconditions, deferred or profiled
`create_subgraph`, and malformed definitions (cycles, bad indices,
out-of-range interpolations, reduction size mismatches, indirection type
mismatches, arrivals without a barrier).

## Tests

`tests/subgraph_unit_tests`: 52 tests and 14 death scenarios with their own
harness (`-list`, `-only`, `-skip`, `-death`). Coverage: tasks, copies, fills,
reductions, gather and scatter, multi-field copies, large fills, arrivals,
interpolation, external conditions and precise poison across all operation
kinds, random DAGs over many processors, destroy ordering, thousands of
instantiations, blocking tasks, finish events, profiling (including
`CANCELLED` for skipped tasks), graph priority (two graphs, and a
high-priority graph fed by lower-priority work on its own processor), GPU
chains in deferred, stream-aware and plain flavours, cross-GPU chains, GPU to
CPU and GPU to copy dependencies, GPU poison, remote instantiation with
arguments, pre/postconditions and profiling, copies to, from and between
remote instances. ctest registers the unit test, a GPU variant and every death
scenario.

Verification of the final commit on Eos (gcc 11, CUDA 12.8, H100, driver 535):

- One node: ctest 778/778 in Debug and RelWithDebInfo; unit tests in four CPU
  configurations and one- and two-GPU configurations; all death scenarios;
  stencil result checks on CPU and GPU.
- Two nodes over UCX: unit tests, all remote tests, remote-task death
  scenario, stencil check.
- The pre-port branch's CUDA ctest is also 100% green, so nothing was
  regressed by the build changes. TSAN is left to CI.

## Measurements (one Eos node, RelWithDebInfo)

Chain of 64 tasks, compiled replay chained on the previous instantiation:

| processors | µs per instantiation | ns per task |
| --- | --- | --- |
| 1 | 11.4 | 178 |
| 4 | 54.3 | 848 |
| 16 | 69.6 | 1088 |
| 32 | 82.4 | 1288 |

Plain spawns versus compiled, 4 processors, 64 operations:

| shape | spawn ns/task | compiled ns/task | ratio |
| --- | --- | --- | --- |
| chain | 9129 | 848 | 10.8× |
| independent | 3722 | 365 | 10.2× |
| fan | 4562 | 338 | 13.5× |
| layers 8×8 | 5156 | 543 | 9.5× |
| random | 3546 | 419 | 8.5× |

Chain on one processor: 575 ns/task at 8 tasks, 194 at 64, 125 at 512, 114 at
4096 (spawns: 2.4 to 3.0 µs/task).

Copy chains of 16 sysmem-to-sysmem copies:

| bytes per copy | per-copy issue µs | compiled replay µs |
| --- | --- | --- |
| 256 | 5.5 | 2.8 |
| 16 K | 6.2 | 3.2 |
| 1 M | 40.7 | 36.8 |

GPU: a chain of 12 deferred-effects kernels of 200 µs is enqueued by the host
in 0.1 ms and finishes in 2.8 ms; without the property each task waits for the
previous kernel. 500 replays of a 4-kernel chain: 10.6 µs per kernel end to
end.

Stencil, 2×2 tiles, 40 steps: CPU 128² on 4 CPUs, 138 µs/step direct versus
112 compiled; GPU 256² on one H100, 184 versus 133.

Two ranks: copy to a remote instance and back, 86 µs per round trip.

Normal task path, main versus branch, interleaved medians over 12 runs:
`task_ubench` completion 3.87 vs 3.89 Mtasks/s, `task_throughput` 4.66 vs
4.43 µs/task. No measurable cost from the scheduler-loop changes.

## Follow-ups

XD reuse across replays (after the DMA refactor), execution-state recycling if
the ~4 µs fixed instantiation cost shows up in Legion profiles, nested
instantiations, reservations, `SERIALIZABLE`/`CONCURRENT`, remote tasks,
collective conditions, subgraph hooks for utility/HIP/OpenMP/Python
processors, non-CUDA context managers for subgraph tasks, a long randomized
soak for TSAN, a 64-CPU scale run.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
