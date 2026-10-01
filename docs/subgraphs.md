# Compiled subgraphs

Realm subgraphs (`realm/subgraph.h`) describe a fixed graph of operations
once and replay it many times with `instantiate`. This document records what
the compiled implementation supports, what it refuses, the semantics that are
not obvious from the header, how it is tested, and what remains to be done.

## What `compile` does

`Subgraph::create_subgraph` compiles the definition synchronously:

- Tasks are grouped per processor and assigned slots in per-processor ready
  queues. Each operation gets a precondition counter; counters live in the
  NUMA domain of the operation's predecessors when they share one, otherwise
  in the consumer's. Queue slots, tails and per-processor counters live in
  the consumer's domain. Interpolated arguments get a per-domain scratch copy.
- Dependencies become CSR successor lists. For GPU tasks, successors are split
  into those released when the task function returns (same-GPU tasks after a
  deferred-effects task, see below) and those released when its work completes.
- External preconditions become counted inputs; per-operation and
  per-postcondition bitsets record which inputs each depends on, so a poisoned
  input skips exactly the operations downstream of it and poisons exactly the
  postconditions downstream of it, plus the finish event.
- Each copy is turned into a transfer plan (`TransferDesc`) and analyzed once.
  Instantiations replay the plan with fresh transfer descriptors (XDs). Remote
  XDs are created through the existing factory messages on every replay.
- Blocks for all of the above are allocated on the right NUMA node, pooled per
  subgraph and recycled across instantiations.

Everything the compiled implementation cannot run makes `create_subgraph` (or
`instantiate`) abort with a message naming the operation and the feature.

## Supported

| Feature | Notes |
| --- | --- |
| Tasks on CPUs (`LOC_PROC`) | Function runs on the processor's scheduler thread, no `Task` object. Tasks may block on events, spawn work, and ask for their finish event (created on demand). |
| Tasks on CUDA GPUs (`TOC_PROC`) | Context pushed, a task stream assigned, completion tracked with a CUDA event. See "GPU tasks". |
| Copies, fills, reductions | One plan per copy, analyzed at compile time. Instances must outlive the subgraph. |
| Indirect copies (gather/scatter) | `CopyDesc::add_indirection<N,T>()` records the index space type; mismatches are fatal at compile. |
| Barrier arrivals | Performed by whoever satisfies the arrival's last precondition. |
| Interpolation | Into task arguments and into an arrival's barrier or reduction value, with or without a reduction operator. |
| External preconditions and postconditions | Counted inputs and outputs; precise poison propagation. |
| `ONE_SHOT`, `INSTANTIATION_ORDER` | Instantiation order is enforced by chaining the start on the previous finish. |
| Priorities | `priority_adjust` of `instantiate` is the instantiation's priority (see below). `CopyDesc::priority` is added to it for the copy. |
| Profiling | Definition-time requests on tasks and copies; instantiation-time requests through `SubgraphInstantiationProfiling` (sparse lists per kind, serializable for remote instantiation), merged with the definition's. Tasks report timeline, processor usage, status and finish event; copies report whatever Realm's transfer operations report. |
| Remote `instantiate` and `destroy` | Messages to the owner; `destroy` returns an event that covers outstanding instantiations. |
| Many instantiations in flight | Any number per processor, round robin among the ready ones. |

## Refused (fatal), and why

| Request | Reason |
| --- | --- |
| `SERIALIZABLE`, `CONCURRENT` | Not implemented in the first pass; nothing in the engine makes them harder later. |
| Nested instantiations (`instantiations`) | Not implemented. |
| Reservation acquires and releases | Not implemented. |
| Tasks on another node's processors | Only local tasks are compiled; remote work goes through copies. |
| Tasks on processors other than CPUs and CUDA GPUs | Utility, I/O, OpenMP, Python and HIP processors have no subgraph hooks yet. |
| A task id not registered on its processor at compile time | The deferred-effects decision needs the registration. |
| `TaskDesc::priority != 0` | The executor schedules by instantiation priority only. |
| Collective pre/postconditions | Not implemented. |
| `create_subgraph` with an untriggered `wait_on`, or with profiling requests | Compile is synchronous and unprofiled. |
| Profiling requests on the `instantiate` overloads that take a `ProfilingRequestSet` for the instantiation itself | Use `SubgraphInstantiationProfiling`. |
| Copies with mismatched source and destination counts, missing index space, or an indirection for another index space type | Caught at compile. |
| Dependency cycles, bad indices, arrivals without a barrier | Caught at compile. |

The public API has no way to interpolate into copies (there are no
`TARGET_COPY_*` interpolation targets), so that is not a gap in the
implementation.

## Semantics worth knowing

**Priority.** While an instantiation of priority P is active on a processor,
and all external inputs its operations on that processor depend on have
triggered, that processor runs only work of priority P or higher: normal tasks
and other instantiations alike. Equal priority still runs. Tasks that already
started always resume. The analysis of which inputs each processor depends on
is done at compile time, so a processor is not held hostage by a graph that is
waiting on something that processor might have to produce. Consequence to be
aware of: a task inside a high-priority instantiation that blocks on work of
lower priority on the same processor will wait until the instantiation leaves
that processor.

**Completion.** A processor's share of the finish counter is released when its
last operation completes, whichever one that is. Tasks that block and finish
out of order, and GPU tasks whose work outlives their function, are accounted
for correctly.

**Poison.** Only poisoned external preconditions propagate poison. Operations
downstream are skipped (tasks and copies are not run, arrivals are not
performed); their successors still drain. Misuse is fatal, not poisoned.

**Finish events.** `Processor::get_current_finish_event()` inside a subgraph
task creates an event on demand. It triggers when the task completes: at
function return on a CPU, when the launched work completes on a GPU.

**Blocking.** Subgraph tasks may wait on events; the scheduler treats them
like any blocked task.

## GPU tasks

A GPU task in a subgraph runs on the GPU processor's scheduler thread with the
CUDA context pushed and `Cuda::get_task_cuda_stream()` returning its stream.
After the function returns a CUDA event, the task's *token*, is recorded on the
stream. The task completes when the token fires; completion releases the
remaining successors, postconditions and finish accounting. Tokens are returned
to the GPU's event pool when the instantiation is released.

A task registered with `DeferredEffectsProperty` (`realm/codedesc.h`), or with
the stream-aware prototype `Cuda::StreamAwareTaskFuncPtr`, promises that all of
its work is on the stream it was given. Then:

- GPU tasks on the same GPU that depend on it start as soon as its function
  returns, with `cuStreamWaitEvent` on its token, so the host can enqueue a
  whole chain while the device is still working on the first kernel.
- No context synchronization is used for its completion (unless it calls
  `Cuda::set_task_ctxsync_required(true)`).

Tasks without the promise follow Realm's usual rule: completion covers the
whole context (`cuCtxRecordEvent` on drivers that have it, the context
synchronizer threads otherwise), and every successor waits for completion.

`-ll:pin_gpu` gives GPU processors a dedicated core, like `-ll:pin_util`.

## Copies

Each `CopyDesc` is compiled into a `TransferDesc` whose analysis (paths,
intermediate buffers, XD templates) is performed once at compile time; this is
also what keeps concurrent instantiations from racing on the analysis. Every
instantiation creates a `TransferOperation` on the shared plan, which
allocates intermediate buffers and creates transfer descriptors, locally or on
remote nodes through the existing factory messages, and reports back when the
data has moved. Reusing the XDs themselves across replays is deferred to the
DMA refactor.

## Configuration

| Option | Default | Meaning |
| --- | --- | --- |
| `-ll:subgraph_poll <us>` | 100 | After an instantiation's work on a processor, how long its scheduler keeps polling for more before sleeping. 0 disables polling. |
| `-ll:pin_gpu` | off | Dedicated core for each GPU processor. |

## Testing

`tests/subgraph_unit_tests` is an integration test with its own harness
(`-list`, `-only PREFIX`, `-skip NAME`, `-iters`, `-seed`, `-hang_timeout`):

- CPU: simple tasks, copies/fills/reductions, arrivals, interpolation, external
  pre/postconditions, random DAGs over many processors, destroy ordering,
  thousands of instantiations, poison, mixed normal and subgraph work,
  blocking tasks, finish events, profiling, graph priority (normal tasks and
  two graphs), copy replay with interpolated values, copy profiling, poisoned
  copies, indirect gather, concurrent subgraphs.
- GPU (`-ll:gpu 1 -ll:zsize 32`, built with CUDA): deferred, stream-aware and
  plain chains (checking that the host ran ahead or did not), GPU to CPU
  dependencies and postconditions, finish events, profiling, 500 replays,
  mixed GPU/CPU chains.
- Two ranks (`-only Remote`): remote instantiate/destroy, copies to and from a
  remote instance.
- Death scenarios (`-death NAME`, one per process, must abort): unsupported
  operation, profiling on the legacy instantiate, external preconditions on
  the legacy instantiate, concurrent mode, dependency cycle, unregistered
  task, indirection type mismatch.

`benchmarks/subgraph_ubench` compares plain spawns with compiled replays for
several graph shapes (`-shape chain|layers|random|copychain`), and
`benchmarks/stencil_subgraph` runs a tiled 5-point stencil with halo copies
both directly and as subgraph replays and checks the result.

## Measurements

See the pull request description for the numbers from the Eos runs (per-task
cost, instantiation cost, copy replay cost, stencil step time).

## Follow-ups

- XD reuse across replays (after the DMA refactor); inline copy analysis to
  skip the analyzer hop is already the common case since plans are analyzed.
- Nested instantiations, reservations, `SERIALIZABLE`/`CONCURRENT`, remote
  tasks, collective conditions.
- Subgraph hooks for utility processors (trivial), HIP, OpenMP and Python
  processors.
- Task context managers other than the CUDA hooks are not applied to subgraph
  tasks (Cuhook validation and CUPTI correlation included).
- Moving the implementation into `src/realm/subgraph/`.
