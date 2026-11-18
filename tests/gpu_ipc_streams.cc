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

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#if defined(REALM_USE_CUDA)
#include "realm/cuda/cuda_internal.h"
#elif defined(REALM_USE_HIP)
#include "realm/hip/hip_internal.h"
#endif
#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <set>
#include <thread>

using namespace Realm;
#if defined(REALM_USE_CUDA)
namespace Backend = Realm::Cuda;
using GPUModule = Backend::CudaModule;
static const char *module_name = "cuda";
#else
namespace Backend = Realm::Hip;
using GPUModule = Backend::HipModule;
static const char *module_name = "hip";
#endif
static int root_rank = 0, require_ipc = 0;
enum
{
  TOP = Processor::TASK_ID_FIRST_AVAILABLE
};

static void top(const void *, size_t, const void *, size_t, Processor proc)
{
  auto *module = Runtime::get_runtime().get_module<GPUModule>(module_name);
  assert(module);
  const unsigned count = std::max(1u, module->config->cfg_p2p_streams);
  size_t tested = 0, checked = 0;
  for(auto *gpu : module->gpus) {
    // Copies within one GPU use the d2d streams, so no peer pool targets the GPU
    // itself.
    const size_t self = gpu->info->index * gpu->p2p_streams_per_peer;
    for(unsigned k = 0; k < gpu->p2p_streams_per_peer; k++)
      assert(gpu->peer_to_peer_streams[self + k] == nullptr);
    // Inspect real imported pools, then interleave selections across peers. A
    // shared rotation counter degenerates when peer count matches pool size.
    std::vector<Backend::GPU::IpcStreamPool *> pools;
#if defined(REALM_USE_CUDA)
    auto &peer_streams = gpu->cudaipc_streams;
    const auto &mappings = gpu->cudaipc_mappings;
#else
    auto &peer_streams = gpu->hipipc_streams;
    const auto &mappings = gpu->hipipc_mappings;
#endif
    for(auto &peer : peer_streams) {
      auto &pool = peer.second;
      assert(pool.streams.size() == count);
      assert(std::set<Backend::GPUStream *>(pool.streams.begin(), pool.streams.end())
                 .size() == count);
      pools.push_back(&pool);
    }
    std::vector<std::vector<Backend::GPUStream *>> first(pools.size());
    for(unsigned n = 0; n < 2 * count; n++)
      for(size_t p = 0; p < pools.size(); p++) {
        auto *stream = pools[p]->get_next_stream();
        if(n < count)
          first[p].push_back(stream);
        else
          assert(stream == first[p][n - count]);
      }
    for(auto &cycle : first)
      assert(std::set<Backend::GPUStream *>(cycle.begin(), cycle.end()).size() == count);

#if defined(REALM_USE_CUDA)
    CUcontext previous;
    assert(cuCtxPushCurrent(gpu->context) == CUDA_SUCCESS);
#else
    int previous;
    assert(hipGetDevice(&previous) == hipSuccess);
    assert(hipSetDevice(gpu->device_id) == hipSuccess);
#endif
    const int expected = std::max(
        gpu->greatest_stream_priority,
        std::min(gpu->least_stream_priority,
                 gpu->greatest_stream_priority + module->config->cfg_dma_stream_priority +
                     (gpu->least_stream_priority - gpu->greatest_stream_priority + 1) /
                         2));
    for(auto *pool : pools) {
      for(auto *stream : pool->streams) {
        int priority;
#if defined(REALM_USE_CUDA)
        assert(cuStreamGetPriority(stream->get_stream(), &priority) == CUDA_SUCCESS);
#else
        assert(hipStreamGetPriority(stream->get_stream(), &priority) == hipSuccess);
#endif
        assert(priority == expected);
      }
      // Concurrent DMA producers must share the atomic per-peer rotation safely.
      auto hits = std::make_unique<std::atomic<unsigned>[]>(count);
      for(unsigned k = 0; k < count; k++)
        hits[k] = 0;
      std::vector<std::thread> threads;
      for(unsigned t = 0; t < 4; t++)
        threads.emplace_back([&]() {
          for(unsigned n = 0; n < count * 32; n++) {
            auto *stream = pool->get_next_stream();
            auto it = std::find(pool->streams.begin(), pool->streams.end(), stream);
            assert(it != pool->streams.end());
            hits[it - pool->streams.begin()].fetch_add(1);
          }
        });
      for(auto &thread : threads)
        thread.join();
      for(unsigned k = 0; k < count; k++)
        assert(hits[k] == 128);
    }
#if defined(REALM_USE_CUDA)
    assert(cuCtxPopCurrent(&previous) == CUDA_SUCCESS);
#else
    assert(hipSetDevice(previous) == hipSuccess);
#endif

    std::vector<Memory> targets;
    std::set<Backend::GPU::IpcStreamPool *> target_pools;
    for(const auto &mapping : mappings) {
      assert(mapping.stream_pool && mapping.stream_pool->streams.size() == count);
      if(!ID(mapping.mem).is_ib_memory() && mapping.mem.kind() == Memory::GPU_FB_MEM) {
        targets.push_back(mapping.mem);
        // Distinct imported framebuffers must have independent peer pools.
        assert(target_pools.insert(mapping.stream_pool).second);
      }
    }
    if(targets.empty())
      continue;
    tested += targets.size();
    Memory host = Machine::MemoryQuery(Machine::get_machine())
                      .local_address_space()
                      .only_kind(Memory::Z_COPY_MEM)
                      .first();
    Memory cpu = Machine::MemoryQuery(Machine::get_machine())
                     .local_address_space()
                     .only_kind(Memory::SYSTEM_MEM)
                     .first();
    assert(host.exists() && cpu.exists());
    IndexSpace<1> backing(Rect<1>(0, 7 * 256 - 1));
    std::vector<Rect<1>> rectangles;
    for(int r = 0; r < 4; r++)
      rectangles.emplace_back(2 * r * 256, (2 * r + 1) * 256 - 1);
    IndexSpace<1> domain(rectangles);
    auto create = [&](Memory mem) {
      RegionInstance inst;
      RegionInstance::create_instance(inst, mem, backing, std::vector<size_t>{8, 8}, 1,
                                      ProfilingRequestSet())
          .wait();
      assert(inst.exists());
      return inst;
    };
    auto copy = [&](RegionInstance src, RegionInstance dst, Event pre = Event::NO_EVENT) {
      std::vector<CopySrcDstField> a(1), b(1);
      a[0].set_field(src, 0, 8);
      b[0].set_field(dst, 0, 8);
      return domain.copy(a, b, ProfilingRequestSet(), pre);
    };
    struct Slot {
      RegionInstance host, src;
      std::vector<RegionInstance> dst, readback;
    };
    std::vector<Slot> slots(2 * count + 1);
    for(auto &slot : slots) {
      slot.host = create(host);
      slot.src = create(gpu->fbmem->me);
      for(Memory target : targets) {
        slot.dst.push_back(create(target));
        slot.readback.push_back(create(cpu));
      }
    }
    for(unsigned round = 0; round < 2; round++) {
      std::vector<Event> done;
      for(size_t s = 0; s < slots.size(); s++) {
        auto &slot = slots[s];
        uint64_t seed = 1 + round * 100000 + s * 1000 + gpu->info->index * 1000000;
        AffineAccessor<uint64_t, 1> input(slot.host, 0);
        for(PointInRectIterator<1> p(backing.bounds); p.valid; p.step())
          input[p.p] = seed + p.p[0] * 17;
        Event ready = copy(slot.host, slot.src);
        for(size_t p = 0; p < targets.size(); p++)
          done.push_back(
              copy(slot.dst[p], slot.readback[p], copy(slot.src, slot.dst[p], ready)));
      }
      Event::merge_events(done).wait();
      for(size_t s = 0; s < slots.size(); s++)
        for(auto inst : slots[s].readback) {
          uint64_t seed = 1 + round * 100000 + s * 1000 + gpu->info->index * 1000000;
          AffineAccessor<uint64_t, 1> output(inst, 0);
          for(IndexSpaceIterator<1> r(domain); r.valid; r.step())
            for(PointInRectIterator<1> p(r.rect); p.valid; p.step()) {
              assert(output[p.p] == seed + p.p[0] * 17);
              checked++;
            }
        }
    }
    for(auto &slot : slots) {
      slot.host.destroy();
      slot.src.destroy();
      for(auto inst : slot.dst)
        inst.destroy();
      for(auto inst : slot.readback)
        inst.destroy();
    }
    domain.destroy();
  }
  if(tested)
    std::cout << "IPC PASS peers=" << tested << " streams=" << count
              << " checked=" << checked << std::endl;
  else
    std::cout << "SKIP: no imported GPU framebuffers (use colocated MPI ranks)"
              << std::endl;
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(),
                                  tested ? 0 : (require_ipc ? 1 : 77));
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);
  CommandLineParser cp;
  cp.add_option_int("-root-rank", root_rank).add_option_int("-require-ipc", require_ipc);
  if(!cp.parse_command_line(argc, const_cast<const char **>(argv)))
    return 1;
  Processor::register_task_by_kind(Processor::LOC_PROC, false, TOP, CodeDescriptor(top),
                                   ProfilingRequestSet())
      .wait();
  Processor root = Processor::NO_PROC;
  Machine::ProcessorQuery roots(Machine::get_machine());
  roots.only_kind(Processor::LOC_PROC);
  for(auto proc : roots) {
    if(int(proc.address_space()) == root_rank) { root = proc; break; }
  }
  assert(root.exists());
  rt.collective_spawn(root, TOP, nullptr, 0);
  return rt.wait_for_shutdown();
}
