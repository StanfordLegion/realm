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
// The time reported is the copy's OperationTimeline, complete_time - start_time.
// The benchmark does not check the copied data.  Results go to standard output,
// because the logger's print level is compiled out of release builds.
//
// Examples (two GPUs):
//   stream count, copy-engine target behind 4 background copies:
//     gpu_copy_streams -ll:gpu 2 -path peer -rects 1 -bg 4 -ll:p2p_streams {1,4,8}
//   stream priority, kernel target under compute load:
//     gpu_copy_streams -ll:gpu 2 -path peer -load-tasks 4 -ll:dma_stream_priority {0,-1}

#include "realm.h"
#include "realm/cmdline.h"

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
}; // namespace TestConfig

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
    size_t element = 0;
    for(size_t field_size : field_sizes) {
      element += field_size;
    }
    return inst_space.volume() * element;
  }

  bool create(Memory mem, RegionInstance &inst) const
  {
    RegionInstance::create_instance(inst, mem, inst_space, field_sizes, block_size,
                                    ProfilingRequestSet())
        .wait();
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
  Processor proc = Processor::NO_PROC;
  RegionInstance buffer = RegionInstance::NO_INST;
  size_t elements = 0;
};

struct Load {
  std::vector<LoadGPU> gpus;
  Event done = Event::NO_EVENT;
  long long wave_start = 0;
  long long wave_total = 0;
  size_t waves = 0;

  bool enabled() const { return !gpus.empty(); }

  void add_gpu(Memory fb)
  {
    LoadGPU gpu;
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
    gpus.push_back(gpu);
  }

  // launch one wave and return once every kernel has been submitted
  void start()
  {
    std::vector<Event> launched, finished;
    wave_start = Clock::current_time_in_nanoseconds();
    for(const LoadGPU &gpu : gpus) {
      for(size_t i = 0; i < TestConfig::load_tasks; i++) {
        LoadArgs largs;
        largs.buffer = gpu.buffer;
        largs.elements = gpu.elements;
        largs.blocks = TestConfig::load_blocks;
        largs.threads = TestConfig::load_threads;
        largs.spin = TestConfig::load_spin;
        largs.launched = UserEvent::create_user_event();
        launched.push_back(largs.launched);
        finished.push_back(gpu.proc.spawn(LOAD_TASK, &largs, sizeof(largs)));
      }
    }
    done = Event::merge_events(finished);
    Event::merge_events(launched).wait();
    std::this_thread::sleep_for(std::chrono::microseconds(TestConfig::load_delay_us));
  }

  bool busy() const { return !done.has_triggered(); }

  void finish()
  {
    done.wait();
    wave_total += Clock::current_time_in_nanoseconds() - wave_start;
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
  bool busy;
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

static void run_path(const PathSpec &spec, Processor p)
{
  Shape shape(TestConfig::bytes, TestConfig::rects, TestConfig::pad);
  Shape bg_shape(TestConfig::bg_bytes, TestConfig::bg_rects, 0);

  RegionInstance src = RegionInstance::NO_INST, dst = RegionInstance::NO_INST;
  RegionInstance bg_src = RegionInstance::NO_INST;
  std::vector<RegionInstance> bg_dsts;
  // Realm aborts when an instance does not fit, so check the sizes first.  This
  // ignores whatever else is in the memories and the load buffer.
  size_t src_need = shape.footprint();
  size_t dst_need = shape.footprint() + TestConfig::bg * bg_shape.footprint();
  if(TestConfig::bg > 0) {
    src_need += bg_shape.footprint();
  }
  bool ok = (spec.src == spec.dst) ? ((src_need + dst_need) <= spec.src.capacity())
                                   : ((src_need <= spec.src.capacity()) &&
                                      (dst_need <= spec.dst.capacity()));
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
  if(ok && (TestConfig::load_tasks > 0)) {
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    if(((TestConfig::load_place & 1) != 0) && (spec.src.kind() == Memory::GPU_FB_MEM)) {
      load.add_gpu(spec.src);
    }
    if(((TestConfig::load_place & 2) != 0) && (spec.dst.kind() == Memory::GPU_FB_MEM) &&
       (spec.dst != spec.src)) {
      load.add_gpu(spec.dst);
    }
#endif
    if(!load.enabled()) {
      log_app.warning() << "path " << spec.name << ": no GPU to put the load on";
    }
  }

  if(!ok) {
    log_app.warning() << "path " << spec.name << ": instances do not fit in " << spec.src
                      << " and " << spec.dst
                      << ", skipping (raise -ll:fsize / -ll:zsize / -ll:csize)";
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
                  << " busy=" << sample.busy << " ns=" << sample.nanoseconds << std::endl;
      }

      if(load.enabled() &&
         ((((it + 1) % TestConfig::load_wave) == 0) || ((it + 1) == TestConfig::iters))) {
        load.finish();
      }
    }

    std::vector<long long> all, busy, idle;
    size_t over_1ms = 0;
    for(const Sample &sample : samples) {
      all.push_back(sample.nanoseconds);
      (sample.busy ? busy : idle).push_back(sample.nanoseconds);
      if(sample.nanoseconds > 1000000) {
        over_1ms++;
      }
    }
    std::sort(all.begin(), all.end());
    std::sort(busy.begin(), busy.end());
    std::sort(idle.begin(), idle.end());

    std::cout << "RESULT path=" << spec.name << " src=" << spec.src << " dst=" << spec.dst
              << " bytes=" << shape.bytes << " rects=" << TestConfig::rects
              << " pad=" << TestConfig::pad << " hops=" << num_hops
              << " bg=" << bg_dsts.size()
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
                << " busy_n=" << busy.size()
                << " busy_median_us=" << quantile_us(busy, 0.5)
                << " idle_n=" << idle.size()
                << " idle_median_us=" << quantile_us(idle, 0.5) << std::endl;
    }
  }

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
    if(fbs.size() >= 2) {
      paths.push_back(PathSpec{"peer", fbs[0], fbs[1]});
    } else {
      std::cout << "path peer: needs two GPUs in this process, skipping" << std::endl;
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

  for(const PathSpec &spec : paths) {
    run_path(spec, p);
  }
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
  assert(ok);
  TestConfig::load_wave = std::max<size_t>(TestConfig::load_wave, 1);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet())
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

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  rt.shutdown(e);

  return rt.wait_for_shutdown();
}
