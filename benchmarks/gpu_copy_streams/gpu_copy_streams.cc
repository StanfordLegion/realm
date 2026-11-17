/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// gpu_copy_streams: latency of one GPU copy while other work competes with it.
//
// The GPU modules run copies on a fixed set of streams (-ll:d2d_streams,
// -ll:p2p_streams, -ll:hd_streams) at a priority relative to task streams
// (-ll:dma_stream_priority).  A stream is a FIFO queue, so two things can delay a
// copy that has nothing to do with the copy itself:
//
//  * other copies on the same path.  If they occupy every stream of that path, the
//    measured copy queues behind one of them.  -bg N issues N background copies on
//    the measured path immediately before each measured copy.  Streams are assigned
//    round-robin, so the measured copy shares a stream with a background copy once
//    N reaches the number of streams.
//
//  * application kernels on the same GPU.  A copy that needs a kernel (more than one
//    rectangle) competes with task kernels for the GPU, and stream priority decides
//    which runs first.  -load-tasks N launches N GPU tasks, each with one kernel of
//    -load-blocks blocks, and measures copies while those kernels are pending.
//    A copy done by the copy engine (a single rectangle) is the control: it should
//    not be affected.
//
// Measured paths (-path): local (one framebuffer to itself), peer (one GPU's
// framebuffer to another's, needs -ll:gpu 2 or more), h2d and d2h (host memory to
// and from a framebuffer).  The default runs every path the machine has.
//
// Use -peer-rank N (and optionally -peer-gpu I) to select a framebuffer in
// another rank, exercising IPC on one host or network staging across hosts.
// The time reported is the copy's OperationTimeline, complete_time - start_time.
// -verify checks the measured payload outside the timing loop.  Results go to standard
// output, because the logger's print level is compiled out of release builds.
//
// Examples (two GPUs):
//   stream count, copy-engine target behind 4 background copies:
//     gpu_copy_streams -ll:gpu 2 -path peer -rects 1 -bg 4 -ll:p2p_streams {1,4,8}
//   stream priority, kernel target under compute load:
//     gpu_copy_streams -ll:gpu 2 -path peer -load-tasks 4 -ll:dma_stream_priority {0,-1}

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/inst_layout.h"

#if defined(REALM_USE_CUDA)
#include "realm/cuda/cuda_module.h"
#include <cuda_runtime.h>
#elif defined(REALM_USE_HIP)
#include "realm/hip/hip_module.h"
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  COPY_PROF_TASK,
  LOAD_TASK,
  LOAD_PROF_TASK,
};

enum
{
  FID_PAYLOAD = 0,
  FID_PAD = 1,
};

typedef long long elem_t;

namespace TestConfig {
  std::string path = "all";    // all, local, peer, h2d, d2h
  std::string host = "zc";     // host endpoint for h2d/d2h: zc (Z_COPY_MEM) or sys
  size_t bytes = 88064;        // payload bytes of the measured copy
  size_t rects = 4;            // rectangles in the measured copy (1 = dense)
  size_t pad = sizeof(elem_t); // bytes of a second, uncopied field (0 = none)
  size_t iters = 100;          // measured copies per path, after one warmup
  bool verify = false;         // verify the measured copy outside the timing loop
  int peer_rank = -1;          // -1: second local GPU; otherwise a remote Realm rank
  size_t peer_gpu = 0;         // framebuffer index within peer_rank
  bool samples = false;        // print every sample

  size_t bg = 0;              // background copies issued before each measured copy
  size_t bg_bytes = 16 << 20; // bytes per background copy
  size_t bg_rects = 1;        // rectangles per background copy

  size_t load_tasks = 0;       // GPU tasks launched per wave (0 = no compute load)
  int load_place = 1;          // bit 0: source GPU, bit 1: destination GPU
  size_t load_blocks = 262144; // blocks per load kernel
  size_t load_threads = 256;   // threads per block
  size_t load_spin = 1000;     // arithmetic iterations per thread
  size_t load_wave = 2;        // measured copies per wave of load tasks
  size_t load_delay_us = 1000; // delay between launching a wave and the first copy
};                             // namespace TestConfig

// ---------------------------------------------------------------------------
// copy shapes
// ---------------------------------------------------------------------------

// `rects` equal runs of 8-byte elements, separated by gaps of the same length so
// that they cannot be merged.  With more than one rectangle the GPU modules use a
// kernel; with one they use the copy engine.  A nonzero pad adds a second field
// and lays the instance out as an array of structs, so that the payload is not
// contiguous in memory even within a rectangle.
struct Shape {
  IndexSpace<1> inst_space;
  IndexSpace<1> domain;
  bool sparse = false;
  std::vector<size_t> field_sizes;
  size_t block_size = 0;
  size_t bytes = 0;

  std::unique_ptr<InstanceLayoutGeneric> layout;

  Shape(size_t want_bytes, size_t num_rects, size_t pad)
  {
    num_rects = std::max<size_t>(num_rects, 1);
    long long run = std::max<size_t>(want_bytes / (num_rects * sizeof(elem_t)), 1);
    long long stride = 2 * run;
    long long last = (static_cast<long long>(num_rects) - 1) * stride + run - 1;
    inst_space = IndexSpace<1>(Rect<1>(0, last));
    if(num_rects > 1) {
      std::vector<Rect<1>> rects;
      for(size_t i = 0; i < num_rects; i++) {
        long long lo = static_cast<long long>(i) * stride;
        rects.push_back(Rect<1>(lo, lo + run - 1));
      }
      domain = IndexSpace<1>(rects);
      sparse = true;
    } else {
      domain = inst_space;
    }
    field_sizes.push_back(sizeof(elem_t));
    if(pad > 0) {
      field_sizes.push_back(pad);
      block_size = 1; // array of structs
    }
    bytes = num_rects * run * sizeof(elem_t);
    const int dim_order[1] = {0};
    layout.reset(InstanceLayoutGeneric::choose_instance_layout(
        inst_space, InstanceLayoutConstraints(field_sizes, block_size), dim_order));
  }

  void destroy()
  {
    if(sparse) {
      domain.destroy();
    }
  }

  // bytes one instance of this shape occupies
  size_t footprint() const
  {
    // Account for alignment between consecutive allocations as well as AoS
    // field padding. Use this same layout when creating the actual instances.
    const size_t alignment = layout->alignment_reqd;
    const size_t padding = (alignment - layout->bytes_used % alignment) % alignment;
    if(layout->bytes_used > std::numeric_limits<size_t>::max() - padding)
      return std::numeric_limits<size_t>::max();
    return layout->bytes_used + padding;
  }

  bool create(Memory mem, RegionInstance &inst) const
  {
    RegionInstance::create_instance(inst, mem, *layout, ProfilingRequestSet()).wait();
    return inst.exists();
  }

  Event copy(RegionInstance src, RegionInstance dst, const ProfilingRequestSet &prs) const
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(src, FID_PAYLOAD, sizeof(elem_t));
    dsts[0].set_field(dst, FID_PAYLOAD, sizeof(elem_t));
    return domain.copy(srcs, dsts, prs);
  }
};

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

struct CopyProfResult {
  long long *nanoseconds;
  unsigned *num_hops;
  UserEvent done;
};

static void copy_prof_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(CopyProfResult));
  const CopyProfResult *result = static_cast<const CopyProfResult *>(resp.user_data());

  ProfilingMeasurements::OperationTimeline timeline;
  if(resp.get_measurement(timeline)) {
    *(result->nanoseconds) = timeline.complete_time - timeline.start_time;
  } else {
    log_app.fatal() << "no operation timeline in profiling response!";
    assert(0);
  }
  ProfilingMeasurements::OperationCopyInfo copy_info;
  if((result->num_hops != nullptr) && resp.get_measurement(copy_info) &&
     !copy_info.inst_info.empty()) {
    *(result->num_hops) = copy_info.inst_info[0].num_hops;
  }
  result->done.trigger();
}

// ---------------------------------------------------------------------------
// compute load
// ---------------------------------------------------------------------------

struct LoadTiming {
  long long start = -1, complete = -1;
  UserEvent measured;
};

static void load_prof_task(const void *args, size_t arglen, const void *, size_t,
                           Processor)
{
  ProfilingResponse response(args, arglen);
  assert(response.user_data_size() == sizeof(LoadTiming *));
  auto *timing = *static_cast<LoadTiming *const *>(response.user_data());
  ProfilingMeasurements::OperationTimeline timeline;
  bool found = response.get_measurement(timeline);
  assert(found && timeline.start_time >= 0 &&
         timeline.complete_time >= timeline.start_time);
  timing->start = timeline.start_time;
  timing->complete = timeline.complete_time;
  timing->measured.trigger();
}

struct LoadArgs {
  RegionInstance buffer;
  size_t elements;
  size_t blocks;
  size_t threads;
  size_t spin;
  UserEvent launched;
};

#if defined(REALM_USE_CUDA)
void launch_load_kernel(float *buffer, size_t elements, size_t blocks, size_t threads,
                        size_t spin, cudaStream_t stream);
#elif defined(REALM_USE_HIP)
void launch_load_kernel(float *buffer, size_t elements, size_t blocks, size_t threads,
                        size_t spin, hipStream_t stream);
#endif

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
// Launches one kernel on this task's stream and returns without waiting for it.
// Context synchronization is turned off so that the task's completion event
// follows the kernel on that stream and nothing else.
static void load_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  assert(arglen == sizeof(LoadArgs));
  const LoadArgs &largs = *static_cast<const LoadArgs *>(args);
  AffineAccessor<float, 1> acc(largs.buffer, FID_PAYLOAD);
  float *buffer = acc.ptr(Point<1>(0));
#if defined(REALM_USE_CUDA)
  cudaStream_t stream = Cuda::get_task_cuda_stream();
  Cuda::set_task_ctxsync_required(false);
#else
  hipStream_t stream = Hip::get_task_hip_stream();
  Hip::set_task_ctxsync_required(false);
#endif
  launch_load_kernel(buffer, largs.elements, largs.blocks, largs.threads, largs.spin,
                     stream);
  largs.launched.trigger();
}
#endif

// The load on one GPU: a buffer for the kernels to write to, and the processor
// that runs the tasks.
struct LoadGPU {
  Memory fb = Memory::NO_MEMORY;
  Event done = Event::NO_EVENT;
  long long task_span_total = 0;
  Processor proc = Processor::NO_PROC;
  RegionInstance buffer = RegionInstance::NO_INST;
  size_t elements = 0;
};

struct Load {
  static constexpr size_t buffer_bytes = (1 << 20) * sizeof(float);
  Processor profiling_proc;
  std::vector<LoadTiming> timings;
  std::vector<LoadGPU> gpus;
  Event done = Event::NO_EVENT;
  long long wave_start = 0;
  long long wave_total = 0;
  size_t waves = 0;

  bool enabled() const { return !gpus.empty(); }

  void add_gpu(Memory fb)
  {
    for(const auto &existing : gpus)
      if(existing.fb == fb)
        return;
    LoadGPU gpu;
    gpu.fb = fb;
    gpu.proc = Machine::ProcessorQuery(Machine::get_machine())
                   .only_kind(Processor::TOC_PROC)
                   .best_affinity_to(fb)
                   .first();
    if(!gpu.proc.exists()) {
      log_app.warning() << "no GPU processor for " << fb << ", not loading it";
      return;
    }
    gpu.elements = 1 << 20;
    std::vector<size_t> field_sizes(1, sizeof(float));
    RegionInstance::create_instance(gpu.buffer, fb,
                                    Rect<1>(0, static_cast<long long>(gpu.elements) - 1),
                                    field_sizes, 0 /*SOA*/, ProfilingRequestSet())
        .wait();
    if(!gpu.buffer.exists()) {
      log_app.warning() << "no room for the load buffer in " << fb;
      return;
    }
    // The instance was created by the root task; a remote load task needs its
    // metadata before constructing an accessor. Keep this setup outside waves.
    gpu.buffer.fetch_metadata(gpu.proc).wait();
    gpus.push_back(gpu);
  }

  // launch one wave and return once every kernel has been submitted
  void start()
  {
    std::vector<Event> launched, finished;
    wave_start = Clock::current_time_in_nanoseconds();
    timings.clear();
    timings.resize(gpus.size() * TestConfig::load_tasks);
    size_t index = 0;
    for(LoadGPU &gpu : gpus) {
      std::vector<Event> gpu_finished;
      for(size_t i = 0; i < TestConfig::load_tasks; i++) {
        LoadArgs largs;
        largs.buffer = gpu.buffer;
        largs.elements = gpu.elements;
        largs.blocks = TestConfig::load_blocks;
        largs.threads = TestConfig::load_threads;
        largs.spin = TestConfig::load_spin;
        largs.launched = UserEvent::create_user_event();
        launched.push_back(largs.launched);
        LoadTiming *timing = &timings[index++];
        timing->measured = UserEvent::create_user_event();
        ProfilingRequestSet requests;
        requests.add_request(profiling_proc, LOAD_PROF_TASK, &timing, sizeof(timing))
            .add_measurement<ProfilingMeasurements::OperationTimeline>();
        gpu_finished.push_back(
            gpu.proc.spawn(LOAD_TASK, &largs, sizeof(largs), requests));
      }
      gpu.done = Event::merge_events(gpu_finished);
      finished.push_back(gpu.done);
    }
    done = Event::merge_events(finished);
    Event::merge_events(launched).wait();
    std::this_thread::sleep_for(std::chrono::microseconds(TestConfig::load_delay_us));
  }

  // Pending task events describe completion, not instantaneous SM occupancy.
  bool busy() const { return !done.has_triggered(); }
  bool pending(Memory fb) const
  {
    for(const auto &gpu : gpus)
      if(gpu.fb == fb)
        return !gpu.done.has_triggered();
    return false;
  }

  void finish()
  {
    done.wait();
    wave_total += Clock::current_time_in_nanoseconds() - wave_start;
    std::vector<Event> measured;
    for(const auto &timing : timings)
      measured.push_back(timing.measured);
    Event::merge_events(measured).wait();
    for(size_t g = 0; g < gpus.size(); g++) {
      long long first = std::numeric_limits<long long>::max(), last = 0;
      for(size_t t = 0; t < TestConfig::load_tasks; t++) {
        const auto &timing = timings[g * TestConfig::load_tasks + t];
        first = std::min(first, timing.start);
        last = std::max(last, timing.complete);
      }
      gpus[g].task_span_total += last - first;
      if(TestConfig::samples)
        std::cout << "LOAD_WAVE gpu=" << gpus[g].fb << " wave=" << waves
                  << " task_span_ms=" << (last - first) / 1e6 << std::endl;
    }
    waves++;
  }

  void destroy()
  {
    for(LoadGPU &gpu : gpus) {
      gpu.buffer.destroy();
    }
    gpus.clear();
  }
};

// ---------------------------------------------------------------------------
// one measured path
// ---------------------------------------------------------------------------

struct PathSpec {
  const char *name;
  Memory src;
  Memory dst;
};

struct Sample {
  long long nanoseconds;
  bool busy, source_pending, destination_pending, wave_first;
};

// nearest-rank quantile of a sorted vector
static double quantile_us(const std::vector<long long> &sorted, double q)
{
  if(sorted.empty()) {
    return 0.0;
  }
  size_t rank = static_cast<size_t>(std::ceil(q * sorted.size()));
  rank = std::min(std::max<size_t>(rank, 1), sorted.size());
  return sorted[rank - 1] / 1e3;
}

static bool run_path(const PathSpec &spec, Processor p)
{
  Shape shape(TestConfig::bytes, TestConfig::rects, TestConfig::pad);
  Shape bg_shape(TestConfig::bg_bytes, TestConfig::bg_rects, 0);

  RegionInstance src = RegionInstance::NO_INST, dst = RegionInstance::NO_INST;
  RegionInstance bg_src = RegionInstance::NO_INST;
  std::vector<RegionInstance> bg_dsts;
  std::set<Memory> load_memories;
  if(TestConfig::load_tasks > 0) {
    if((TestConfig::load_place & 1) && spec.src.kind() == Memory::GPU_FB_MEM)
      load_memories.insert(spec.src);
    if((TestConfig::load_place & 2) && spec.dst.kind() == Memory::GPU_FB_MEM)
      load_memories.insert(spec.dst);
  }
  Memory verify_mem = Memory::NO_MEMORY;
  if(TestConfig::verify)
    verify_mem = Machine::MemoryQuery(Machine::get_machine())
                     .local_address_space()
                     .only_kind(Memory::SYSTEM_MEM)
                     .first();
  // Reserve the complete footprint per memory before attempting any allocation.
  // Division before multiplication also prevents overflow for large -bg values.
  std::map<Memory, size_t> needed;
  auto reserve = [&](Memory mem, size_t bytes, size_t count = 1) {
    if(!mem.exists())
      return false;
    size_t &used = needed[mem];
    if(bytes && count > (mem.capacity() - used) / bytes)
      return false;
    used += bytes * count;
    return true;
  };
  bool ok = reserve(spec.src, shape.footprint()) && reserve(spec.dst, shape.footprint());
  if(TestConfig::bg > 0)
    ok = ok && reserve(spec.src, bg_shape.footprint()) &&
         reserve(spec.dst, bg_shape.footprint(), TestConfig::bg);
  for(Memory mem : load_memories)
    ok = ok && reserve(mem, Load::buffer_bytes);
  if(TestConfig::verify)
    ok = ok && reserve(verify_mem, shape.footprint(), 2);
  ok = ok && shape.create(spec.src, src) && shape.create(spec.dst, dst);
  if(ok && (TestConfig::bg > 0)) {
    ok = bg_shape.create(spec.src, bg_src);
    // every background copy has its own destination, so that none of them
    // depends on another through the data it writes
    for(size_t i = 0; ok && (i < TestConfig::bg); i++) {
      RegionInstance inst = RegionInstance::NO_INST;
      ok = bg_shape.create(spec.dst, inst);
      if(ok) {
        bg_dsts.push_back(inst);
      }
    }
  }

  Load load;
  load.profiling_proc = p;
  if(ok) {
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    for(Memory mem : load_memories)
      load.add_gpu(mem);
#endif
    if(TestConfig::load_tasks && !load.enabled())
      log_app.warning() << "path " << spec.name << ": no GPU selected for load";
  }
  RegionInstance verify_src = RegionInstance::NO_INST,
                 verify_dst = RegionInstance::NO_INST;
  if(ok && TestConfig::verify) {
    ok = shape.create(verify_mem, verify_src) && shape.create(verify_mem, verify_dst);
    if(ok) {
      AffineAccessor<elem_t, 1> input(verify_src, FID_PAYLOAD);
      for(PointInRectIterator<1> it(shape.inst_space.bounds); it.valid; it.step())
        input[it.p] = elem_t(42) + it.p[0] * 17;
      shape.copy(verify_src, src, ProfilingRequestSet()).wait();
    }
  }

  if(!ok) {
    std::cout << "SKIP path=" << spec.name << ": instances do not fit in " << spec.src
              << " and " << spec.dst
              << ", skipping (raise -ll:fsize / -ll:zsize / -ll:csize)" << std::endl;
  } else {
    // warmup, which also tells us how many hops the measured copy takes
    long long warmup_time = -1;
    unsigned num_hops = 0;
    {
      CopyProfResult result;
      result.nanoseconds = &warmup_time;
      result.num_hops = &num_hops;
      result.done = UserEvent::create_user_event();
      ProfilingRequestSet prs;
      prs.add_request(p, COPY_PROF_TASK, &result, sizeof(CopyProfResult))
          .add_measurement<ProfilingMeasurements::OperationTimeline>()
          .add_measurement<ProfilingMeasurements::OperationCopyInfo>();
      shape.copy(src, dst, prs).wait();
      result.done.wait();
      if(!bg_dsts.empty()) {
        bg_shape.copy(bg_src, bg_dsts[0], ProfilingRequestSet()).wait();
      }
    }

    std::vector<Sample> samples;
    for(size_t it = 0; it < TestConfig::iters; it++) {
      if(load.enabled() && ((it % TestConfig::load_wave) == 0)) {
        load.start();
      }

      Sample sample;
      sample.nanoseconds = -1;
      sample.busy = load.enabled() && load.busy();
      sample.source_pending = load.pending(spec.src);
      sample.destination_pending = load.pending(spec.dst);
      sample.wave_first = load.enabled() && ((it % TestConfig::load_wave) == 0);

      CopyProfResult result;
      result.nanoseconds = &sample.nanoseconds;
      result.num_hops = nullptr;
      result.done = UserEvent::create_user_event();
      ProfilingRequestSet prs;
      prs.add_request(p, COPY_PROF_TASK, &result, sizeof(CopyProfResult))
          .add_measurement<ProfilingMeasurements::OperationTimeline>();

      // background copies first, so that they are ahead of the measured copy
      std::vector<Event> events;
      for(RegionInstance bg_dst : bg_dsts) {
        events.push_back(bg_shape.copy(bg_src, bg_dst, ProfilingRequestSet()));
      }
      events.push_back(shape.copy(src, dst, prs));
      events.push_back(result.done);
      Event::merge_events(events).wait();

      samples.push_back(sample);
      if(TestConfig::samples) {
        std::cout << "SAMPLE path=" << spec.name << " iter=" << it
                  << " busy=" << sample.busy
                  << " source_pending=" << sample.source_pending
                  << " destination_pending=" << sample.destination_pending
                  << " wave_first=" << sample.wave_first << " ns=" << sample.nanoseconds
                  << std::endl;
      }

      if(load.enabled() &&
         ((((it + 1) % TestConfig::load_wave) == 0) || ((it + 1) == TestConfig::iters))) {
        load.finish();
      }
    }

    std::vector<long long> all, busy, idle, first, source_pending, destination_pending;
    size_t over_1ms = 0;
    for(const Sample &sample : samples) {
      all.push_back(sample.nanoseconds);
      if(sample.wave_first)
        first.push_back(sample.nanoseconds);
      if(sample.source_pending)
        source_pending.push_back(sample.nanoseconds);
      if(sample.destination_pending)
        destination_pending.push_back(sample.nanoseconds);
      (sample.busy ? busy : idle).push_back(sample.nanoseconds);
      if(sample.nanoseconds > 1000000) {
        over_1ms++;
      }
    }
    std::sort(all.begin(), all.end());
    std::sort(busy.begin(), busy.end());
    std::sort(idle.begin(), idle.end());
    std::sort(first.begin(), first.end());
    std::sort(source_pending.begin(), source_pending.end());
    std::sort(destination_pending.begin(), destination_pending.end());

    std::cout << "RESULT path=" << spec.name << " src=" << spec.src << " dst=" << spec.dst
              << " src_rank=" << spec.src.address_space()
              << " dst_rank=" << spec.dst.address_space() << " bytes=" << shape.bytes
              << " rects=" << TestConfig::rects << " pad=" << TestConfig::pad
              << " hops=" << num_hops << " bg=" << bg_dsts.size()
              << " bg_bytes=" << (bg_dsts.empty() ? 0 : bg_shape.bytes)
              << " load_gpus=" << load.gpus.size() << " n=" << all.size()
              << " min_us=" << quantile_us(all, 0.0)
              << " median_us=" << quantile_us(all, 0.5)
              << " p90_us=" << quantile_us(all, 0.9)
              << " p99_us=" << quantile_us(all, 0.99)
              << " max_us=" << quantile_us(all, 1.0) << " over_1ms=" << over_1ms
              << std::endl;
    if(load.enabled()) {
      // a sample is "busy" if the load tasks had not finished when it was issued
      std::cout << "LOAD path=" << spec.name << " waves=" << load.waves
                << " mean_wave_ms="
                << (load.waves ? (load.wave_total / 1e6 / load.waves) : 0.0)
                << " first_n=" << first.size()
                << " first_median_us=" << quantile_us(first, 0.5)
                << " first_p99_us=" << quantile_us(first, 0.99)
                << " source_pending_n=" << source_pending.size()
                << " source_pending_median_us=" << quantile_us(source_pending, 0.5)
                << " destination_pending_n=" << destination_pending.size()
                << " destination_pending_median_us="
                << quantile_us(destination_pending, 0.5) << " busy_n=" << busy.size()
                << " busy_median_us=" << quantile_us(busy, 0.5)
                << " idle_n=" << idle.size()
                << " idle_median_us=" << quantile_us(idle, 0.5) << std::endl;
      for(const auto &gpu : load.gpus)
        std::cout << "LOAD_GPU path=" << spec.name << " gpu=" << gpu.fb
                  << " rank=" << gpu.fb.address_space()
                  << " mean_task_span_ms=" << gpu.task_span_total / 1e6 / load.waves
                  << " waves=" << load.waves << std::endl;
    }
    if(TestConfig::verify) {
      shape.copy(dst, verify_dst, ProfilingRequestSet()).wait();
      AffineAccessor<elem_t, 1> output(verify_dst, FID_PAYLOAD);
      size_t checked = 0, bad = 0;
      for(IndexSpaceIterator<1> r(shape.domain); r.valid; r.step())
        for(PointInRectIterator<1> it(r.rect); it.valid; it.step()) {
          bad += output[it.p] != elem_t(42) + it.p[0] * 17;
          checked++;
        }
      std::cout << "VERIFY path=" << spec.name << " checked=" << checked
                << " errors=" << bad << std::endl;
      if(bad)
        std::exit(1);
    }
  }

  if(verify_src.exists())
    verify_src.destroy();
  if(verify_dst.exists())
    verify_dst.destroy();
  load.destroy();
  if(src.exists()) {
    src.destroy();
  }
  if(dst.exists()) {
    dst.destroy();
  }
  if(bg_src.exists()) {
    bg_src.destroy();
  }
  for(RegionInstance inst : bg_dsts) {
    inst.destroy();
  }
  shape.destroy();
  bg_shape.destroy();
  return ok;
}

static void top_level_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  Machine machine = Machine::get_machine();

  std::vector<Memory> fbs;
  Machine::MemoryQuery fb_query =
      Machine::MemoryQuery(machine).local_address_space().only_kind(Memory::GPU_FB_MEM);
  for(Machine::MemoryQuery::iterator it = fb_query.begin(); it; ++it) {
    if((*it).capacity() > 0) {
      fbs.push_back(*it);
    }
  }
  if(fbs.empty()) {
    std::cout << "no GPU framebuffer memories (run with -ll:gpu), nothing to do"
              << std::endl;
    Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), 77);
    return;
  }

  // the largest host memory of the requested kind
  Memory host = Memory::NO_MEMORY;
  Machine::MemoryQuery host_query =
      Machine::MemoryQuery(machine).local_address_space().only_kind(
          (TestConfig::host == "sys") ? Memory::SYSTEM_MEM : Memory::Z_COPY_MEM);
  for(Machine::MemoryQuery::iterator it = host_query.begin(); it; ++it) {
    if(((*it).capacity() > 0) &&
       (!host.exists() || ((*it).capacity() > host.capacity()))) {
      host = *it;
    }
  }

  const std::string &want = TestConfig::path;
  const bool all = (want == "all");
  std::vector<PathSpec> paths;
  if(all || (want == "local")) {
    paths.push_back(PathSpec{"local", fbs[0], fbs[0]});
  }
  if(all || (want == "peer")) {
    if(TestConfig::peer_rank < 0) {
      if(fbs.size() >= 2)
        paths.push_back(PathSpec{"peer", fbs[0], fbs[1]});
      else
        std::cout << "path peer: needs two local GPUs or -peer-rank, skipping"
                  << std::endl;
    } else {
      std::vector<Memory> peers;
      Machine::MemoryQuery peer_query(machine);
      peer_query.only_kind(Memory::GPU_FB_MEM);
      for(auto mem : peer_query)
        if(int(mem.address_space()) == TestConfig::peer_rank && mem.capacity() > 0)
          peers.push_back(mem);
      if(TestConfig::peer_gpu < peers.size() && peers[TestConfig::peer_gpu] != fbs[0])
        paths.push_back(PathSpec{"peer", fbs[0], peers[TestConfig::peer_gpu]});
      else
        std::cout << "path peer: requested rank/GPU is unavailable, skipping"
                  << std::endl;
    }
  }

  if(all || (want == "h2d") || (want == "d2h")) {
    if(!host.exists()) {
      std::cout << "paths h2d/d2h: no " << TestConfig::host << " host memory, skipping"
                << std::endl;
    } else {
      if(all || (want == "h2d")) {
        paths.push_back(PathSpec{"h2d", host, fbs[0]});
      }
      if(all || (want == "d2h")) {
        paths.push_back(PathSpec{"d2h", fbs[0], host});
      }
    }
  }
  if(paths.empty()) {
    std::cout << "no path to measure for -path " << want << std::endl;
  }

  size_t measured = 0;
  for(const PathSpec &spec : paths)
    measured += run_path(spec, p);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(),
                                  measured ? 0 : 77);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_string("-path", TestConfig::path)
      .add_option_string("-host", TestConfig::host)
      .add_option_int("-bytes", TestConfig::bytes)
      .add_option_int("-rects", TestConfig::rects)
      .add_option_int("-pad", TestConfig::pad)
      .add_option_int("-iters", TestConfig::iters)
      .add_option_bool("-samples", TestConfig::samples)
      .add_option_bool("-verify", TestConfig::verify)
      .add_option_int("-peer-rank", TestConfig::peer_rank)
      .add_option_int("-peer-gpu", TestConfig::peer_gpu)
      .add_option_int("-bg", TestConfig::bg)
      .add_option_int("-bg-bytes", TestConfig::bg_bytes)
      .add_option_int("-bg-rects", TestConfig::bg_rects)
      .add_option_int("-load-tasks", TestConfig::load_tasks)
      .add_option_int("-load-place", TestConfig::load_place)
      .add_option_int("-load-blocks", TestConfig::load_blocks)
      .add_option_int("-load-threads", TestConfig::load_threads)
      .add_option_int("-load-spin", TestConfig::load_spin)
      .add_option_int("-load-wave", TestConfig::load_wave)
      .add_option_int("-load-delay-us", TestConfig::load_delay_us);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  ok = ok && TestConfig::iters > 0 && TestConfig::rects > 0 && TestConfig::bytes > 0 &&
       TestConfig::bg_rects > 0 && TestConfig::bg_bytes > 0 &&
       TestConfig::load_wave > 0 && TestConfig::load_place >= 0 &&
       TestConfig::load_place <= 3 && TestConfig::peer_rank >= -1;
  ok = ok &&
       (TestConfig::path == "all" || TestConfig::path == "local" ||
        TestConfig::path == "peer" || TestConfig::path == "h2d" ||
        TestConfig::path == "d2h") &&
       (TestConfig::host == "zc" || TestConfig::host == "sys");
  if(!ok) {
    std::cerr << "invalid gpu_copy_streams arguments" << std::endl;
    rt.shutdown(Event::NO_EVENT, 1);
    return rt.wait_for_shutdown();
  }

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet())
      .wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false, LOAD_PROF_TASK,
                                   CodeDescriptor(load_prof_task), ProfilingRequestSet())
      .wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, COPY_PROF_TASK,
                                   CodeDescriptor(copy_prof_task), ProfilingRequestSet())
      .wait();
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, LOAD_TASK,
                                   CodeDescriptor(load_task), ProfilingRequestSet())
      .wait();
#endif

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}
