
/* Copyright 2024 NVIDIA Corporation
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

/**
 * Realm Machine Model Tutorial
 *
 * This example illustrates how to query the machine model from Realm.
 * It outlines various Realm types that define the resources of the
 * underlying hardware and the affinities between these resources.
 *
 * Topics Covered:
 * - Machine: the top-level abstraction of all computing resources.
 * - Processor: execution resources that run tasks.
 * - Memory: data storage locations with specific access properties.
 * - Affinity: bandwidth/latency relationships between processors and memories.
 * - ID: Realm object identifiers.
 *
 * Why a Machine Model?
 * --------------------
 * A machine model describes the structure of hardware resources available
 * to Realm and enables applications, runtimes, or schedulers to make informed
 * mapping decisions.
 *
 * Realm uses a flat, flexible, graph-based representation:
 *   - Nodes: Processors and Memories
 *   - Edges: Affinities (bandwidth and latency hints)
 *
 * This model captures real hardware topology more accurately than rigid hierarchies.
 * For example, NUMA and GPU memory access paths may not align with CPU socket layouts.
 *
 * Performance Portability:
 * Applications can use Realm’s model to remap across different machines
 * without rewriting core logic, thanks to this abstraction layer.
 *
 * Example System Topology:
 *
 *   +------+      +------+      +------+      +------+      +-------+
 *   | x86  |      | x86  |      | x86  |      | x86  |      | CUDA  |
 *   +------+      +------+      +------+      +------+      +-------+
 *      |              |              |              |            |
 *      +--------------+--------------+--------------+------------+
 *                     |              |              |
 *                  +------+       +------+       +------+
 *                  |NUMA1|       |NUMA2|       |  ZC  |  (Zero-Copy Memory)
 *                  +------+       +------+       +------+
 *                                                |
 *                                             +------+
 *                                             |  FB  | (GPU Framebuffer)
 *                                             +------+
 *
 * This structure represents processors, memory types, and connections (affinities).
 * Realm can use this model to choose the best path for computation and data movement.
 */

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#include <assert.h>

    using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

/**
 * The main task is launched on a CPU (LOC_PROC) and performs machine model
 * inspection by querying available processors and their memory affinities.
 */
void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor)
{

  /**
   * MACHINE
   * -------
   * A Machine represents all compute nodes an application can occupy.
   * Use Machine::get_machine() to access this singleton instance.
   */
  Machine machine = Machine::get_machine();

  /**
   * PROCESSOR
   * ---------
   * Any hardware or software entity capable of running a task.
   * Examples: CPU core (Pthread), CPU socket (OpenMP), GPU (CUDA), Python interpreter.
   *
   * Realm enables consistent handling of diverse hardware (CPUs, GPUs, etc.) in
   * distributed environments. Applications can be (re)mapped to different hardware
   * targets without rewriting core logic.
   *
   * Use Machine::ProcessorQuery to enumerate processors.
   * The address space shows which node (rank) the processor resides on.
   */
  for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin(); it;
      ++it) {
    Processor p = *it;
    ID proc_id = ID(p.id);

    /**
     * ID
     * --
     * Realm uses 64-bit IDs to uniquely identify objects (processors, memories, etc).
     * Use is_processor() or is_procgroup() to verify processor IDs.
     */
    assert(proc_id.is_processor() || proc_id.is_procgroup());

    Processor::Kind kind = p.kind();
    switch(kind) {
    case Processor::LOC_PROC:
    {
      /** LOC_PROC: latency-optimized core (CPU), specified by -ll:cpu */
      log_app.print("Rank %u, Processor ID " IDFMT " is CPU.", p.address_space(), p.id);
      break;
    }
    case Processor::TOC_PROC:
    {
      /** TOC_PROC: throughput-optimized core (GPU), specified by -ll:gpu */
      log_app.print("Rank %u, Processor ID " IDFMT " is GPU.", p.address_space(), p.id);
      break;
    }
    case Processor::IO_PROC:
    {
      /** IO_PROC: used for file or socket I/O operations, specified by -ll:io */
      log_app.print("Rank %u, Processor ID " IDFMT " is I/O Proc.", p.address_space(),
                    p.id);
      break;
    }
    case Processor::UTIL_PROC:
    {
      /** UTIL_PROC: utility thread for background Realm work, specified by -ll:util */
      log_app.print("Rank %u, Processor ID " IDFMT " is utility.", p.address_space(),
                    p.id);
      break;
    }
    default:
    {
      log_app.print("Rank %u, Processor " IDFMT " is unknown (kind=%d)",
                    p.address_space(), p.id, p.kind());
      break;
    }
    }

    /**
     * MEMORY
     * ------
     * Memories describe the location of application data
     * System memory (DRAM), GPU framebuffer memory, NIC-registered (RDMA) memory, flash
     * storage (e.g., burst buffers), file-backed storage.
     *
     * MemoryQuery returns memory objects that have affinity to the given processor.
     * has_affinity_to(p) filters memories accessible to processor p.
     */
    log_app.print("Has Affinity with:");
    Machine::MemoryQuery mq = Machine::MemoryQuery(machine).has_affinity_to(p, 0, 0);

    for(Machine::MemoryQuery::iterator it = mq.begin(); it; ++it) {
      Memory m = *it;
      ID mem_id = ID(m.id);

      assert(mem_id.is_memory() || mem_id.is_ib_memory());

      size_t memory_size_in_kb = m.capacity() >> 10;

      /**
       * AFFINITY
       * --------
       * ProcessorMemoryAffinity describes the connection between a processor and memory.
       * Includes bandwidth (in MB/s) and latency (in arbitrary units).
       */
      std::vector<Machine::ProcessorMemoryAffinity> pm_affinity;
      machine.get_proc_mem_affinity(pm_affinity, p, m, true);
      assert(pm_affinity.size() == 1);

      unsigned bandwidth = pm_affinity[0].bandwidth;
      unsigned latency = pm_affinity[0].latency;

      /**
       * MEMORY KIND
       * -----------
       * Realm supports many types of memory:
       * SYSTEM_MEM, REGDMA_MEM, GPU_FB_MEM, Z_COPY_MEM, etc.
       * Each has specific usage and visibility traits.
       */
      switch(m.kind()) {
      case Memory::GLOBAL_MEM:
        log_app.print("\tGASNet Global Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d.",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::SYSTEM_MEM:
        log_app.print("\tSystem Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::REGDMA_MEM:
        log_app.print("\tPinned Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::SOCKET_MEM:
        log_app.print("\tSocket Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::Z_COPY_MEM:
        log_app.print("\tZero-Copy Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::GPU_FB_MEM:
        log_app.print("\tGPU Frame Buffer Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::GPU_MANAGED_MEM:
        log_app.print("\tGPU Managed Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::GPU_DYNAMIC_MEM:
        log_app.print("\tGPU Dynamic-allocated Frame Buffer Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::DISK_MEM:
        log_app.print("\tDisk Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::HDF_MEM:
        log_app.print("\tHDF Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::FILE_MEM:
        log_app.print("\tFile Memory ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::LEVEL3_CACHE:
        log_app.print("\tLevel 3 Cache ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::LEVEL2_CACHE:
        log_app.print("\tLevel 2 Cache ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      case Memory::LEVEL1_CACHE:
        log_app.print("\tLevel 1 Cache ID " IDFMT
                      " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                      m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
        break;
      default:
        log_app.print("\tMemory " IDFMT " is unknown (kind=%d).", it->id, it->kind());
        break;
      }
    }
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);

  return rt.wait_for_shutdown();
}