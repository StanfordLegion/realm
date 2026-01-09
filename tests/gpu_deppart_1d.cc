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
  int    num_edges   = 5000;
  int    num_pieces  = 4;
  int    random      = 0;           // 0 deterministic, 1 random
  unsigned long long seed = 123456789ULL;
  int    show        = 0;           // print assigned ids
  int    verify      = 1;           // do correctness check
};
static const FieldID FID_SUBGRAPH = 0;
static const FieldID FID_SRC = 0;
static const FieldID FID_DST = sizeof(Point<1, int>);

// ---------------- Small helpers (same idioms as transpose_test) --------
template <int N, typename T, typename DT, typename Fn>
static void fill_index_space(RegionInstance inst,
                             FieldID fid,
                             const IndexSpace<N,T>& is,
                             Fn gen)
{
  AffineAccessor<DT, N, T> acc(inst, fid);
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
                           std::vector<size_t> fields)
{
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
  IndexSpace<1,int> is_nodes(Rect<1,int>(0, TestConfig::num_nodes - 1));
  IndexSpace<1,int> is_edges(Rect<1, int>(0, TestConfig::num_edges - 1));

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
  RegionInstance cpu_inst_nodes;
  make_instance(cpu_inst_nodes, cpu_mem, is_nodes, {sizeof(int)}).wait();

  RegionInstance cpu_inst_edges;
  make_instance(cpu_inst_edges, cpu_mem, is_edges, {sizeof(Point<1, int>), sizeof(Point<1, int>)}).wait();

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
  fill_index_space<1,int,int>(cpu_inst_nodes, FID_SUBGRAPH, is_nodes, gen_id);

  auto gen_src = [&](Point<1,int> p)->Point<1, int> {
    if (TestConfig::random) {
      return Point<1, int>(Philox_2x32<>::rand_int(TestConfig::seed,
                                     /*counter=*/p[0],
                                     /*stream=*/0,
                                     /*bound=*/TestConfig::num_nodes));
    } else {
      return Point<1, int>(p[0] % TestConfig::num_nodes);
    }
  };

  fill_index_space<1,int,Point<1,int>>(cpu_inst_edges, FID_SRC, is_edges, gen_src);

  auto gen_dst = [&](Point<1,int> p)->Point<1, int> {
    if (TestConfig::random) {
      return Point<1, int>(Philox_2x32<>::rand_int(TestConfig::seed,
                                     /*counter=*/p[0]+TestConfig::num_edges,
                                     /*stream=*/0,
                                     /*bound=*/TestConfig::num_nodes));
    } else {
      return Point<1, int>((p[0]+1) % TestConfig::num_nodes);
    }
  };

  fill_index_space<1,int,Point<1,int>>(cpu_inst_edges, FID_DST, is_edges, gen_dst);

  if (TestConfig::show) {
    AffineAccessor<int,1,int> acc(cpu_inst_nodes, FID_SUBGRAPH);
    for (IndexSpaceIterator<1,int> it(is_nodes); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        log_app.print() << "id[" << p.p << "]=" << acc[p.p];

    AffineAccessor<Point<1,int>,1,int> acc_src(cpu_inst_edges, FID_SRC);
    AffineAccessor<Point<1,int>,1,int> acc_dst(cpu_inst_edges, FID_DST);
    for (IndexSpaceIterator<1,int> it(is_edges); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        log_app.print() << "edge[" << p.p << "]=" << acc_src[p.p] << "->" << acc_dst[p.p];
  }

  // Describe the field data (CPU)
  FieldDataDescriptor<IndexSpace<1,int>, int> cpu_field_nodes;
  cpu_field_nodes.index_space  = is_nodes;
  cpu_field_nodes.inst         = cpu_inst_nodes;
  cpu_field_nodes.field_offset = 0;

  FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>> cpu_field_src;
  cpu_field_src.index_space  = is_edges;
  cpu_field_src.inst         = cpu_inst_edges;
  cpu_field_src.field_offset = 0;

  FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>> cpu_field_dst;
  cpu_field_dst.index_space  = is_edges;
  cpu_field_dst.inst         = cpu_inst_edges;
  cpu_field_dst.field_offset = sizeof(Point<1,int>);

  std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>> cpu_nodes(1, cpu_field_nodes);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>>> cpu_src(1, cpu_field_src);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>>> cpu_dst(1, cpu_field_dst);


  // Colors 0..num_pieces-1
  std::vector<int> colors(TestConfig::num_pieces);
  for (int i = 0; i < TestConfig::num_pieces; i++) colors[i] = i;

  // CPU partitioning
  std::vector<IndexSpace<1,int>> p_cpu_nodes, p_cpu_edges, p_cpu_rd;
  Event e_cpu_byfield = is_nodes.create_subspaces_by_field(cpu_nodes, colors, p_cpu_nodes, ProfilingRequestSet());
  Event e_cpu_bypreimage = is_edges.create_subspaces_by_preimage(cpu_dst, p_cpu_nodes, p_cpu_edges, ProfilingRequestSet(), e_cpu_byfield);
  Event e_cpu_image = is_nodes.create_subspaces_by_image(cpu_src, p_cpu_edges, p_cpu_rd, ProfilingRequestSet(), e_cpu_bypreimage);

  // GPU path (optional if GPU exists)
  std::vector<IndexSpace<1,int>> p_gpu_nodes, p_gpu_edges, p_gpu_rd;
  if (have_gpu) {
    RegionInstance gpu_inst_nodes, gpu_inst_edges;
    make_instance(gpu_inst_nodes, gpu_mem, is_nodes, {sizeof(int)}).wait();
    make_instance(gpu_inst_edges, gpu_mem, is_edges, {sizeof(Point<1, int>), sizeof(Point<1, int>)}).wait();

    // Copy field data CPU -> GPU
    copy_field<1,int,int>(is_nodes, cpu_inst_nodes, gpu_inst_nodes, FID_SUBGRAPH);
    copy_field<1,int,Point<1,int>>(is_edges, cpu_inst_edges, gpu_inst_edges, FID_SRC);
    copy_field<1,int,Point<1,int>>(is_edges, cpu_inst_edges, gpu_inst_edges, FID_DST);

    // Describe the field data (CPU)
    FieldDataDescriptor<IndexSpace<1,int>, int> gpu_field_nodes;
    gpu_field_nodes.index_space  = is_nodes;
    gpu_field_nodes.inst         = gpu_inst_nodes;
    gpu_field_nodes.field_offset = 0;

    FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>> gpu_field_src;
    gpu_field_src.index_space  = is_edges;
    gpu_field_src.inst         = gpu_inst_edges;
    gpu_field_src.field_offset = 0;

    FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>> gpu_field_dst;
    gpu_field_dst.index_space  = is_edges;
    gpu_field_dst.inst         = cpu_inst_edges;
    gpu_field_dst.field_offset = sizeof(Point<1,int>);

    std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>> gpu_nodes(1, gpu_field_nodes);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>>> gpu_src(1, gpu_field_src);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1, int>>> gpu_dst(1, gpu_field_dst);

    std::vector<IndexSpace<1,int>> p_gpu_nodes, p_gpu_edges, p_gpu_rd;
    Event e_gpu_byfield = is_nodes.create_subspaces_by_field(gpu_nodes, colors, p_gpu_nodes,
                                               ProfilingRequestSet());
    Event e_gpu_bypreimage = is_edges.create_subspaces_by_preimage(gpu_dst, p_gpu_nodes, p_gpu_edges, ProfilingRequestSet(), e_gpu_byfield);
    Event e_gpu_image = is_nodes.create_subspaces_by_image(gpu_src, p_gpu_edges, p_gpu_rd, ProfilingRequestSet(), e_gpu_bypreimage);

    e_cpu_image.wait();
    e_gpu_image.wait();
    // Compare CPU vs GPU partitions
    if (TestConfig::verify) {
      int errs = compare_partitions(p_cpu_nodes, p_gpu_nodes) +
                 compare_partitions(p_cpu_edges, p_gpu_edges) +
                 compare_partitions(p_cpu_rd, p_gpu_rd);
      if (errs) {
        log_app.fatal() << "Mismatch between CPU and GPU partitions, errors=" << errs;
        assert(0);
      }
    }
    gpu_inst_nodes.destroy();
    gpu_inst_edges.destroy();
  } else {
    e_cpu_image.wait();
  }

  // Cleanup
  cpu_inst_nodes.destroy();
  cpu_inst_edges.destroy();
  is_nodes.destroy();
  is_edges.destroy();

  log_app.print() << "deppart_1d_itest: PASS";
}

// ---------------- Main (same as transpose_test pattern) -----------------
int main(int argc, char** argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  // Parse simple flags similar to the example
  CommandLineParser cp;
  cp.add_option_int("-n",      TestConfig::num_nodes)
    .add_option_int("-e",      TestConfig::num_edges)
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