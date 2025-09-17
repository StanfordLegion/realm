/*
 * Copyright 2025 Stanford University, NVIDIA
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <vector>
#include <set>
#include <cstdio>
#include <cstring>
#include "realm.h"
#include "realm/id.h"
#include "realm/machine.h"
#include "realm/cmdline.h"
#include "philox.h"

using namespace Realm;

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_memcpy.h"
#include "realm/cuda/cuda_module.h"
#endif
#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hip_module.h"
#endif

#ifdef REALM_USE_CUDA
using namespace Realm::Cuda;
#endif
#ifdef REALM_USE_HIP
using namespace Realm::Hip;
#endif

Logger log_app("app");

// ---------------- Config (matches transpose_test style) ----------------
namespace TestConfig {
  int    num_nodes   = 1000;
  int    num_pieces  = 4;
  int    random      = 0;           // 0 deterministic, 1 random
  unsigned long long seed = 123456789ULL;
  int    show        = 0;           // print assigned ids
  int    verify      = 1;           // do correctness check
};
static const FieldID FID_SUBGRAPH = 0;

// ---------------- Small helpers (same idioms as transpose_test) --------
template <int N, typename T, typename DT, typename Fn>
static void fill_index_space(RegionInstance inst,
                             FieldID fid,
                             const IndexSpace<N,T>& is,
                             Fn gen)
{
  GenericAccessor<DT, N, T> acc(inst, fid);
  for (IndexSpaceIterator<N,T> it(is); it.valid; it.step()) {
    for (PointInRectIterator<N,T> p(it.rect); p.valid; p.step())
      acc[p.p] = gen(p.p);
  }
}

template <int N, typename T, typename DT>
static void copy_field(const IndexSpace<N,T>& is,
                       RegionInstance src, RegionInstance dst, FieldID fid)
{
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src, fid, sizeof(DT));
  dsts[0].set_field(dst, fid, sizeof(DT));
  is.copy(srcs, dsts, ProfilingRequestSet()).wait();
}

static void choose_cpu_and_gpu_mems(Memory& cpu_mem, Memory& gpu_mem, bool& have_gpu)
{
  have_gpu = false;
  for (auto mem : Machine::MemoryQuery(Machine::get_machine())) {
    if (!cpu_mem.exists() && (mem.kind() == Memory::SYSTEM_MEM))
      cpu_mem = mem;
    if (!gpu_mem.exists() && (mem.kind() == Memory::GPU_FB_MEM)) {
      gpu_mem = mem;
      have_gpu = true;
    }
  }
}

// For brevity, we use the simple vector<size_t> layout helper (as in many Realm tests)
static Event make_instance(RegionInstance& ri,
                           Memory mem,
                           const IndexSpace<1,int>& is,
                           size_t elem_size)
{
  std::vector<size_t> fields(1, elem_size);
  return RegionInstance::create_instance(ri, mem, is, fields,
                                         /*soa=*/0, ProfilingRequestSet());
}

// Compare two partitions index-space-by-index-space
static int compare_partitions(const std::vector<IndexSpace<1,int>>& A,
                              const std::vector<IndexSpace<1,int>>& B)
{
  int errors = 0;
  if (A.size() != B.size()) return 1;
  for (size_t i = 0; i < A.size(); i++) {
    // Check A minus B
    for (IndexSpaceIterator<1,int> it(A[i]); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        if (!B[i].contains(p.p)) { errors++; }
    // Check B minus A
    for (IndexSpaceIterator<1,int> it(B[i]); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        if (!A[i].contains(p.p)) { errors++; }
  }
  return errors;
}

// ---------------- Top-level task (like transpose_test_gpu) --------------
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 300,
};

static void top_level_task(const void*, size_t, const void*, size_t, Processor)
{
  log_app.print() << "deppart_byfield_itest starting";

  // Build the 1D node space [0 .. N-1]
  IndexSpace<1,int> is(Rect<1,int>(0, TestConfig::num_nodes - 1));

  // Choose memories
  Memory cpu_mem, gpu_mem;
  bool have_gpu = false;
  choose_cpu_and_gpu_mems(cpu_mem, gpu_mem, have_gpu);
  if (!cpu_mem.exists()) {
    log_app.fatal() << "No SYSTEM_MEM found";
    assert(0);
    return;
  }
  if (!have_gpu) {
    log_app.warning() << "No GPU_FB_MEM found; running CPU-only check.";
  }

  // Create CPU instance holding subgraph ids
  RegionInstance cpu_inst;
  make_instance(cpu_inst, cpu_mem, is, sizeof(int)).wait();

  // Fill ids (deterministic or random)
  auto gen_id = [&](Point<1,int> p)->int {
    if (TestConfig::random) {
      return Philox_2x32<>::rand_int(TestConfig::seed,
                                     /*counter=*/p[0],
                                     /*stream=*/0,
                                     /*bound=*/TestConfig::num_pieces);
    } else {
      // even split
      return int((long long)p[0] * TestConfig::num_pieces / TestConfig::num_nodes);
    }
  };
  fill_index_space<1,int,int>(cpu_inst, FID_SUBGRAPH, is, gen_id);

  if (TestConfig::show) {
    GenericAccessor<int,1,int> acc(cpu_inst, FID_SUBGRAPH);
    for (IndexSpaceIterator<1,int> it(is); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        log_app.print() << "id[" << p.p << "]=" << acc[p.p];
  }

  // Describe the field data (CPU)
  FieldDataDescriptor<IndexSpace<1,int>, int> cpu_field;
  cpu_field.index_space  = is;
  cpu_field.inst         = cpu_inst;
  cpu_field.field_offset = 0;

  std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>> cpu_fields(1, cpu_field);

  // Colors 0..num_pieces-1
  std::vector<int> colors(TestConfig::num_pieces);
  for (int i = 0; i < TestConfig::num_pieces; i++) colors[i] = i;

  // CPU partitioning
  std::vector<IndexSpace<1,int>> p_cpu;
  Event e_cpu = is.create_subspaces_by_field(cpu_fields, colors, p_cpu,
                                             ProfilingRequestSet());
  e_cpu.wait();

  // GPU path (optional if GPU exists)
  std::vector<IndexSpace<1,int>> p_gpu;
  if (have_gpu) {
    RegionInstance gpu_inst;
    make_instance(gpu_inst, gpu_mem, is, sizeof(int)).wait();

    // Copy field data CPU -> GPU
    copy_field<1,int,int>(is, cpu_inst, gpu_inst, FID_SUBGRAPH);

    FieldDataDescriptor<IndexSpace<1,int>, int> gpu_field;
    gpu_field.index_space  = is;
    gpu_field.inst         = gpu_inst;
    gpu_field.field_offset = 0;
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>> gpu_fields(1, gpu_field);

    Event e_gpu = is.create_subspaces_by_field(gpu_fields, colors, p_gpu,
                                               ProfilingRequestSet());
    e_gpu.wait();

    // Compare CPU vs GPU partitions
    if (TestConfig::verify) {
      int errs = compare_partitions(p_cpu, p_gpu);
      if (errs) {
        log_app.fatal() << "Mismatch between CPU and GPU partitions, errors=" << errs;
        assert(0);
      }
    }

    gpu_inst.destroy();
  } else {
    // No GPU: at least assert CPU produced a sane partitioning
    if (TestConfig::verify) {
      // Every point should appear exactly once across p_cpu
      std::vector<char> seen(TestConfig::num_nodes, 0);
      for (size_t i = 0; i < p_cpu.size(); i++)
        for (IndexSpaceIterator<1,int> it(p_cpu[i]); it.valid; it.step())
          for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step()) {
            int idx = p.p[0];
            if (idx < 0 || idx >= TestConfig::num_nodes || seen[idx]) {
              log_app.fatal() << "CPU partition invalid at " << p.p << " (dup/out of range)";
              assert(0);
            }
            seen[idx] = 1;
          }
    }
  }

  // Cleanup
  cpu_inst.destroy();
  is.destroy();

  log_app.print() << "deppart_byfield_itest: PASS";
}

// ---------------- Main (same as transpose_test pattern) -----------------
int main(int argc, char** argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  // Parse simple flags similar to the example
  CommandLineParser cp;
  cp.add_option_int("-n",      TestConfig::num_nodes)
    .add_option_int("-p",      TestConfig::num_pieces)
    .add_option_int("-random", TestConfig::random)
    .add_option_int("-show",   TestConfig::show)
    .add_option_int("-verify", TestConfig::verify);
  bool ok = cp.parse_command_line(argc, const_cast<const char**>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                  .only_kind(Processor::LOC_PROC)
                  .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, nullptr, 0);
  rt.shutdown(e);
  rt.wait_for_shutdown();
  return 0;
}