gpu_copy_streams measures copy latency and compute progress under competing GPU
work, using public Realm APIs. Build with REALM_BUILD_BENCHMARKS and CUDA or HIP.

Paths are local (same framebuffer), peer (two GPUs), h2d and d2h. The default is
all available paths. A peer is normally the second local GPU. With one GPU per
MPI rank, use -path peer -peer-rank N; -peer-gpu I selects framebuffer I on that
rank (default 0). Colocated ranks exercise IPC, while different hosts exercise
the network route. RESULT prints source/destination ranks and profiled hop count.

Examples (prepend your MPI/cluster launcher):
  gpu_copy_streams -ll:gpu 2 -path peer -bg 4 -ll:p2p_streams 8
  gpu_copy_streams -ll:gpu 1 -path peer -peer-rank 1 -verify
  gpu_copy_streams -ll:gpu 2 -path peer -load-tasks 4 -load-place 1 -samples
  gpu_copy_streams -ll:gpu 1 -path h2d -load-tasks 4 -load-place 2

Copy shape: -bytes payload, -rects separated rectangles, -pad uncopied AoS field.
The defaults are 88064 bytes, four rectangles and eight bytes of padding. A dense
one-rectangle, pad=0 case is a copy-engine control. Multiple rectangles generally
use copy kernels. -bg N queues N independent background destinations on the same
path; -bg-bytes and -bg-rects select their shape. Instances use the exact layouts
used in capacity accounting, including field/allocation alignment. Preflight
includes background destinations and every selected GPU's load buffer. It cannot
account for unrelated concurrent allocations outside this benchmark.

GPU load: -load-tasks tasks per selected GPU, -load-blocks, -load-threads and
-load-spin control kernels. -load-place is a bit mask: 1 source, 2 destination,
3 both. A local source/destination is loaded once even with mask 3; mask 2 also
loads it. For h2d use mask 2 or 3 since the source is host memory. -load-wave is
the number of measured copies per load wave; -load-delay-us follows submission
of all load kernels. Immediate kernel-launch failures terminate with an error.

RESULT measures copy OperationTimeline complete_time - start_time after warmup.
LOAD retains full-wave mean_wave_ms, which includes copy draining and the launch
delay. LOAD_GPU separately reports each GPU's mean_task_span_ms: earliest task
start through latest completion in that wave. These public task timelines include
host dispatch and completion-event handling; they are not exact GPU kernel-only
timestamps. Per-GPU spans avoid combining clocks from different MPI ranks.

A SAMPLE's source_pending/destination_pending flags refer to selected GPUs' task
completion events. They describe pending work, not SM occupancy. The legacy busy
flag means any selected GPU's task event is pending. A partly drained grid or a
busy other endpoint can coexist with fast copies. LOAD therefore also reports
first_n/first_median_us/first_p99_us for each fresh wave's first copy, and separate
pending subsets per endpoint. Inspect tails and per-GPU task spans when evaluating
priority: a fast copy can delay compute completion. -samples prints every copy
and per-GPU LOAD_WAVE task span with wave indices.

-verify checks the final measured destination against a position-dependent pattern
outside the timing loop. It adds two system-memory instances to capacity accounting.
It does not validate every timed iteration or background-copy destination; use a
functional concurrency test for event-ordering/data-integrity stress. Verification
and preparation traffic are outside measured copy/load waves.

No measurable path (including capacity skips) returns 77. Invalid arguments,
failed launches or incorrect verified data return nonzero errors. CTest regression
cases cover placement, launch errors, aligned/load-buffer capacity and verification
without asserting hardware-dependent timing thresholds. Stream priority is a
policy choice; the runtime default is unchanged by this benchmark.
