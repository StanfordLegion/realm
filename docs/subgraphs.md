<!--
Copyright 2025 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

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
- Blocks for all of the above are allocated on the right NUMA node, carved
  from per-domain slabs. The execution state of an instantiation (blocks,
  launch and cleanup waiters, one waiter per external input) is pooled per
  subgraph and recycled, so an instantiation without copies or profiling
  allocates nothing but its finish event (copies create Realm transfer
  operations with a finish event each; profiling allocates its records). The
  state is recycled in the finish event's trigger context, or on a
  background worker when the subgraph has GPU tasks (any GPU task's work may
  outlive its function, so those states hold processor tokens).

Everything the compiled implementation cannot run makes `create_subgraph` (or
`instantiate`) abort with a message naming the operation and the feature.

A note on first use: the first event a thread creates on behalf of a processor
makes Realm allocate that processor's event table (65536 entries), which takes
tens of milliseconds. Instantiations create a finish event, and copies create
one per transfer on whichever thread launches them, so the first replay of a
graph may pay this once per processor and DMA thread involved; replays after
that do not. `subgraph_ubench -cold K` and the stencil benchmark's warm-up
round show the effect.

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
| `CONCURRENT`, `INSTANTIATION_ORDER`, `ONE_SHOT` | `CONCURRENT` (the default, as before) runs instantiations independently, ordered only by the events the application gives them. `INSTANTIATION_ORDER` chains each start on the previous finish. A `ONE_SHOT` subgraph instantiated twice is fatal. |
| Priorities | `priority_adjust` of `instantiate` is the instantiation's priority (see below). `CopyDesc::priority` is added to it for the copy. |
| Profiling | Definition-time requests on tasks and copies; instantiation-time requests through `SubgraphInstantiationProfiling` (sparse lists per kind, serializable for remote instantiation), merged with the definition's. Tasks report timeline, processor usage, status and finish event; copies report whatever Realm's transfer operations report. |
| Remote `instantiate` and `destroy` | Messages to the owner; `destroy` returns an event that covers outstanding instantiations. Active messages are not delivered in order (not across types, and a large payload can arrive after a smaller one sent later), so the node's own issue order is carried explicitly: a remote `destroy` message is sent only after every instantiation that node sent before it has finished, and each instantiate carries a per-node sequence number that the owner uses to run an `INSTANTIATION_ORDER` subgraph's instantiations in the order each node issued them (across nodes, the owner's arrival order). The numbering never resets, on either side: it continues across a subgraph's destroy and the reuse of its slot, since ids carry no generation. The destroy's own precondition is passed to the owner as given. |
| Many instantiations in flight | Any number per processor, round robin among the ready ones. |

## Refused (fatal), and why

| Request | Reason |
| --- | --- |
| `SERIALIZABLE` | Not implemented in the first pass; nothing in the engine makes it harder later. |
| An arrival as a dependency source, an external postcondition as a source, an external precondition as a target, or a precondition wired directly to a postcondition | Arrivals and postconditions have no outputs, preconditions have no inputs, and a postcondition must be fed by an operation. |
| An external postcondition index with no source | Every declared postcondition must be fed by some operation (indices are dense). |
| Nested instantiations (`instantiations`) | Not implemented. |
| Reservation acquires and releases | Not implemented. |
| Tasks on another node's processors | Only local tasks are compiled; remote work goes through copies. |
| Tasks on processors other than CPUs and CUDA GPUs | Utility, I/O, OpenMP, Python and HIP processors have no subgraph hooks yet. |
| A task id not registered on its processor at compile time | The deferred-effects decision needs the registration. |
| `instantiate` or `destroy` on a subgraph that was destroyed | Detected while the handle's slot has not been reused (subgraph ids carry no generation). |
| A deferred-effects GPU task calling `Cuda::set_task_ctxsync_required(true)` | Contradicts the promise the compile relied on. |
| `TaskDesc::priority != 0` | The executor schedules by instantiation priority only. |
| `Processor::set_current_task_priority` inside a graph task | Graph tasks have no `Operation`; the instantiation's priority applies. (`Processor::add_finish_event_precondition` returns `REALM_PROCESSOR_ERROR_OUTSIDE_TASK` there, for the same reason.) |
| A copy that fails or is cancelled | Its successors would run on bad data; nothing in a graph can recover. |
| Profiling measurements on tasks other than `OperationTimeline`, `OperationProcessorUsage`, `OperationStatus`, `OperationFinishEvent` | Graph tasks have no `Operation`, which is where every other measurement is produced. Refused at creation for the definition's requests and at instantiate for per-instantiation ones, as a whitelist: a measurement kind added to Realm later is refused rather than silently never delivered (main drops unsupported requests silently). |
| Collective pre/postconditions | Not implemented. |
| `create_subgraph` with an untriggered `wait_on`, or with profiling requests | Compile is synchronous and unprofiled. |
| Profiling requests on the `instantiate` overloads that take a `ProfilingRequestSet` for the instantiation itself | Use `SubgraphInstantiationProfiling`. |
| Copies with mismatched source and destination counts, missing index space, or an indirection for another index space type | Caught at compile. |
| Dependency cycles, bad indices, arrivals without a barrier | Caught at compile. |

The public API has no way to interpolate into copies (there are no
`TARGET_COPY_*` interpolation targets), so that is not a gap in the
implementation.

## Semantics worth knowing

**Priority.** While an instantiation of priority P is active on a processor
(it has operations still to run there, and the external inputs they depend
on have triggered), and the instantiation is making progress somewhere (a
task of it is running, or an operation of it is ready or in flight), that
processor runs only work of priority P or higher: normal tasks and other
instantiations alike. Equal priority still runs, and ties between a ready
normal task and subgraph work alternate: the side that lost the previous tie
wins the next one, so neither a graph replayed back to back nor a task that
keeps respawning itself can starve the other. Tasks that already started always resume. A processor whose last operation of the graph has been
dequeued stays held while that task runs. The analysis of which inputs each
processor depends on is done at compile time, so a processor is not held
hostage by a graph that is waiting on something that processor might have to
produce. An instantiation that is stalled everywhere (every started task of
it is blocked and nothing of it is ready) holds nothing: holding would only
idle its processors and, if a blocked task waits on lower-priority work that
needs one of them, deadlock. The holds return as soon as a task of it
resumes. Consequences: lower-priority work can start on a graph's processors
while the whole graph is blocked (or, for a GPU, once a GPU task's function
has returned and nothing else of the graph is running), and the graph then
waits for that work to finish or block before its task resumes; nothing ever
waits forever on a hold. When an instantiation stalls, the processors it may
have been holding are woken explicitly, so lower-priority work they held back
runs even if nothing else would have woken them. This rests on one rule for
graph tasks: a task that waits for anything must wait through Realm (an
event wait). A graph task that spins on a flag, blocks in a library call or
otherwise waits outside Realm keeps its instantiation counted as running, so
its holds stay in place; if what it waits for needs a held processor, that is
a deadlock of the application's making, one that ordinary tasks could not
have because their priorities never reach across processors. A GPU task whose function has returned counts as
not running, whether or not its work on the stream is complete.

**Completion.** A processor's share of the finish counter is released when its
last operation completes, whichever one that is. Tasks that block and finish
out of order, and GPU tasks whose work outlives their function, are accounted
for correctly.

**Shutdown.** The contract is the one every Realm operation has: an
application must chain the events Realm hands it (instantiation finish events,
external postconditions) into the precondition it gives `Runtime::shutdown`,
in some form. An instantiation whose finish event has not triggered when the
runtime shuts down (after that precondition and the processors' flush tasks)
is pending application work, and the runtime aborts with a diagnostic naming
the count ("pending subgraph instantiation(s) during shutdown"), as it does
for an untriggered operation. The event a `destroy` returns is no exception:
like every event Realm hands out, it must be chained into the shutdown
precondition (or waited for), and a destroy that has not taken effect when the
runtime shuts down is reported the same way ("subgraph destroy(s) not yet in
effect during shutdown"). A destroy that returned `NO_EVENT` took effect on
the spot and needs nothing. The check is eager and exact for programs that
follow this rule: the destroy's event triggers only after the destroy has
taken effect, so nothing is ever counted that the application had already
been told was done. What the runtime drains on its own is only the internal
work behind those events: the release of finished instantiations' states and
the destroy itself.

**Destroy.** A destroy takes effect when its precondition triggers, not when
it is called. Instantiations handled before that point are covered by it
(the subgraph goes away once the last of them has released its resources),
so `d = sg.destroy(G); e = sg.instantiate(); G.trigger(e)` is valid, on one
node or across nodes. Every instantiation must either be issued on the
destroying node before the destroy call, or be ordered before the destroy's
precondition (by its finish event, or something that follows it).
Instantiating after the precondition has triggered is fatal, as is a second
destroy while one is pending.

**Poison.** Only a poisoned start event or poisoned external preconditions
propagate poison; in `INSTANTIATION_ORDER` mode the previous instantiation's
poison does not reach the next one (the chain is ordering, not a
precondition). An external precondition index below the declared count that
gates no operation is still waited for, and its poison still poisons the
finish event; the number of preconditions passed to `instantiate` must equal
the declared count (one more than the highest index used), anything else is
fatal. A poisoned start event skips every operation, poisons every
postcondition and the finish event, and still waits for the external
preconditions before the finish event triggers. A poisoned
`destroy` precondition destroys the subgraph anyway (with a warning). Operations
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
the stream-aware prototype `Cuda::StreamAwareTaskFuncPtr` (the one that takes
a `CUstream` argument), promises that all of its work is on the stream it was
given. The inference from the prototype is overridden by an explicit
`-cuda:contextsync 1` on the command line: then only the property makes a task
deferred-effects, and stream-aware tasks without it get the conservative path
(dependents released on completion, context synchronization honored). Then,
for a deferred-effects task:

- GPU tasks on the same GPU that depend on it start as soon as its function
  returns, with `cuStreamWaitEvent` on its token, so the host can enqueue a
  whole chain while the device is still working on the first kernel.
- No context synchronization is used for its completion. Calling
  `Cuda::set_task_ctxsync_required(true)` from such a task is fatal, since
  its dependents were ordered after its stream alone when the graph compiled.

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
  thousands of instantiations, poison (tasks, and fills/arrivals/profiled
  tasks/multi-source postconditions), mixed normal and subgraph work,
  blocking tasks, finish events, profiling, graph priority (normal tasks, two
  graphs, and a high-priority graph fed by lower-priority work on its own
  processor, equal-priority fairness under replay), default-mode
  instantiations that depend on later ones, copy replay with interpolated
  values, copy profiling, poisoned copies, indirect gather and scatter,
  multi-field copies, large fills, concurrent subgraphs.
- GPU (`-ll:gpu 1` or `2`, `-ll:zsize 32`, built with CUDA): deferred,
  stream-aware and plain chains (checking that the host ran ahead or did
  not), a chain alternating between two GPUs, GPU to CPU dependencies and
  postconditions, a GPU task feeding a copy out of frame-buffer memory,
  finish events, profiling, poison, 500 replays, mixed GPU/CPU chains.
- Two ranks (`-only Remote`): remote instantiate/destroy, remote instantiation
  with interpolated arguments, pre/postconditions and instantiation-time
  profiling, copies to and from a remote instance, copies between two remote
  instances.
- Death scenarios (`-death NAME`, one per process, must abort; run with
  `-ll:util 1`, plus `-ll:gpu 1` for the GPU one): unsupported operation,
  profiling on the legacy instantiate, external preconditions on the legacy
  instantiate, serializable mode, dependency cycle, unregistered task,
  indirection type mismatch, interpolation out of range, reduction size
  mismatch, deferred creation, task on a utility processor, destroy twice,
  instantiate after destroy, deferred-effects GPU task asking for a context
  synchronization. Each must die of its own diagnostic. Two shutdown
  scenarios (`-exit NAME`) must abort with the pending-instantiation
  diagnostic: an instantiation gated on an event nobody triggers, and a
  graph task blocked forever.

`benchmarks/subgraph_ubench` compares plain spawns with compiled replays for
several graph shapes (`-shape chain|layers|random|copychain`), and
`benchmarks/stencil_subgraph` runs a tiled 5-point stencil with halo copies
both directly and as subgraph replays and checks the result.

## Measurements

See the pull request description for the numbers from the Eos runs (per-task
cost, instantiation cost, copy replay cost, stencil step time).

## Code layout

- `realm/subgraph.h`, `realm/subgraph.inl`: the public API.
- `realm/subgraph/subgraph_impl.h`: the compiled form (`CompiledSubgraph`), the
  per-instantiation state, the per-processor executor and the lifecycle classes.
- `realm/subgraph/subgraph_compile.cc`: validation and compilation.
- `realm/subgraph/subgraph_exec.cc`: execution (state, executor, GPU and copy paths).
- `realm/subgraph/subgraph_api.cc`: the API entry points, lifecycle, messages.
- Processor hooks live with the processors (`proc_impl.h`, `cuda/cuda_module.cc`);
  the scheduler integration is in `tasks.cc`.

## Follow-ups

- XD reuse across replays (after the DMA refactor); inline copy analysis to
  skip the analyzer hop is already the common case since plans are analyzed.
- Nested instantiations, reservations, `SERIALIZABLE`, remote tasks,
  collective conditions.
- Subgraph hooks for utility processors (trivial), HIP, OpenMP and Python
  processors.
- Task context managers other than the CUDA hooks are not applied to subgraph
  tasks (Cuhook validation and CUPTI correlation included).
