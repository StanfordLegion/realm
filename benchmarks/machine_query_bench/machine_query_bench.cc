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

// Micro-benchmark for Realm machine-model queries.
//
// Times the query shapes that a task-based runtime's mapper issues on every
// mapping decision (processors by kind, memories visible from a processor,
// best memory for a processor, point affinity lookups, bandwidth between two
// memories, ...) both from a single thread and from several concurrent
// threads, so that the per-query cost and the contention inside the machine
// model can be measured as the machine grows.  Shapes come in fresh-query
// form (a query object is built, run and destroyed per operation) and, for a
// few, in reused form (one query object run repeatedly by every thread) and
// cache-miss form (a kind and memory with no affinity).  Only the first
// processor in the machine runs the measurement; all other ranks only
// contribute to the size of the machine model.
//
// Output is one RESULT line per (shape, thread count):
//   RESULT shape=<name> threads=<T> iters=<n> elems=<k> ns_per_op=<median>
//          min=<min> max=<max>
// where ns_per_op is the median over -reps repetitions of the mean per-thread
// latency of one operation.  "elems" is the number of elements the shape
// touches per operation (e.g. the number of processors an iteration yields).

#include <realm.h>
#include <realm/cmdline.h>
#include <realm/timers.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

namespace {

  // Benchmark knobs (parsed on every rank, read by the measuring task).
  int cfg_iters = 0; // 0 = calibrate per shape against cfg_target_ms
  int cfg_reps = 5;
  int cfg_target_ms = 50;
  int cfg_max_iters = 1000000;
  std::string cfg_threads = "1,4,16";
  std::string cfg_shapes; // comma-separated substrings, empty = all shapes
  int cfg_sample = 64;    // handles sampled across the machine for mixed shapes

#ifdef __linux__
  // affinity mask of the process before Realm pins any threads
  cpu_set_t initial_mask;
  bool have_initial_mask = false;
#endif

  void inherit_process_affinity(void)
  {
#ifdef __linux__
    if(have_initial_mask)
      pthread_setaffinity_np(pthread_self(), sizeof(initial_mask), &initial_mask);
#endif
  }

  struct Shape {
    std::string name;
    std::function<uint64_t(void)> fn;
    size_t elems;
    bool enabled;
  };

  // A fixed sample of handles from the machine so that the shapes below can
  // issue realistic queries without re-deriving their inputs in the timed loop.
  struct Sample {
    Machine machine;
    std::vector<Processor> local_procs, all_procs, sampled_procs;
    std::vector<Memory> local_mems, all_mems, sampled_mems, source_mems;
    Processor cpu, gpu, p;
    Memory sysmem, fb, mem_hit, mem_miss, mm_src, mm_dst, kind_miss;
    Memory::Kind best_kind;
    size_t pmas_all;
    Sample(void)
      : machine(Machine::get_machine())
    {}
  };

  template <typename T>
  void spread(const std::vector<T> &in, size_t n, std::vector<T> &out)
  {
    out.clear();
    if(in.empty())
      return;
    if(in.size() <= n) {
      out = in;
      return;
    }
    for(size_t i = 0; i < n; i++)
      out.push_back(in[(i * in.size()) / n]);
  }

  bool build_sample(Sample &s)
  {
    Machine &m = s.machine;
    {
      std::set<Processor> ps;
      m.get_all_processors(ps);
      s.all_procs.assign(ps.begin(), ps.end());
      std::set<Processor> lp;
      m.get_local_processors(lp);
      s.local_procs.assign(lp.begin(), lp.end());
      std::set<Memory> ms;
      m.get_all_memories(ms);
      s.all_mems.assign(ms.begin(), ms.end());
      for(Machine::MemoryQuery::iterator it =
              Machine::MemoryQuery(m).local_address_space().begin();
          it; ++it)
        s.local_mems.push_back(*it);
    }
    s.cpu = Machine::ProcessorQuery(m)
                .local_address_space()
                .only_kind(Processor::LOC_PROC)
                .first();
    s.gpu = Machine::ProcessorQuery(m)
                .local_address_space()
                .only_kind(Processor::TOC_PROC)
                .first();
    s.p = s.gpu.exists() ? s.gpu : s.cpu;
    s.sysmem = Machine::MemoryQuery(m)
                   .local_address_space()
                   .only_kind(Memory::SYSTEM_MEM)
                   .first();
    s.fb = Machine::MemoryQuery(m)
               .local_address_space()
               .only_kind(Memory::GPU_FB_MEM)
               .first();
    s.best_kind = s.fb.exists() ? Memory::GPU_FB_MEM : Memory::SYSTEM_MEM;
    if(!s.cpu.exists() || !s.sysmem.exists()) {
      log_app.error() << "need at least one local CPU processor and system memory";
      return false;
    }
    // a (proc, mem) pair with affinity, and one without (worst case for scans)
    s.mem_hit = s.sysmem;
    s.mem_miss = Memory::NO_MEMORY;
    for(std::vector<Memory>::reverse_iterator it = s.all_mems.rbegin();
        it != s.all_mems.rend(); ++it)
      if(!m.has_affinity(s.cpu, *it)) {
        s.mem_miss = *it;
        break;
      }
    // a memory no CPU processor has an affinity to (a cache miss for the
    // kind-and-memory lists)
    s.kind_miss = Memory::NO_MEMORY;
    for(std::vector<Memory>::reverse_iterator it = s.all_mems.rbegin();
        it != s.all_mems.rend(); ++it)
      if(Machine::ProcessorQuery(m).only_kind(Processor::LOC_PROC).has_affinity_to(*it).count() ==
         0) {
        s.kind_miss = *it;
        break;
      }
    // a memory pair with a copy path from local system memory
    s.mm_src = s.sysmem;
    s.mm_dst = Memory::NO_MEMORY;
    for(std::vector<Memory>::reverse_iterator it = s.all_mems.rbegin();
        it != s.all_mems.rend(); ++it)
      if((*it != s.sysmem) && m.has_affinity(s.sysmem, *it)) {
        s.mm_dst = *it;
        break;
      }
    if(!s.mm_dst.exists())
      s.mm_dst = s.sysmem;
    spread(s.all_procs, cfg_sample, s.sampled_procs);
    spread(s.all_mems, cfg_sample, s.sampled_mems);
    spread(s.all_mems, 8, s.source_mems);
    {
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      s.pmas_all = m.get_proc_mem_affinity(pmas, Processor::NO_PROC, Memory::NO_MEMORY,
                                           false /*!local_only*/);
    }
    return true;
  }

  template <typename Q>
  uint64_t iterate(const Q &q, size_t *count = 0)
  {
    uint64_t acc = 0;
    size_t n = 0;
    for(typename Q::iterator it = q.begin(); it; ++it) {
      acc += (*it).id;
      n++;
    }
    if(count)
      *count = n;
    return acc;
  }

  void build_shapes(const Sample &s, std::vector<Shape> &shapes)
  {
    const Machine machine = s.machine;
    const Processor cpu = s.cpu, p = s.p;
    const Memory sysmem = s.sysmem, mem_hit = s.mem_hit, mem_miss = s.mem_miss;
    const Memory mm_src = s.mm_src, mm_dst = s.mm_dst, kind_miss = s.kind_miss;
    const Memory::Kind best_kind = s.best_kind;
    const std::vector<Processor> local_procs = s.local_procs;
    const std::vector<Processor> sampled_procs = s.sampled_procs;
    const std::vector<Memory> local_mems = s.local_mems;
    const std::vector<Memory> sampled_mems = s.sampled_mems;
    const std::vector<Memory> source_mems = s.source_mems;

    // helper to count the elements a query yields once, for the elems column
    size_t n;
    auto add = [&](const char *name, std::function<uint64_t(void)> fn, size_t elems,
                   bool enabled = true) {
      Shape sh;
      sh.name = name;
      sh.fn = fn;
      sh.elems = elems;
      sh.enabled = enabled;
      shapes.push_back(sh);
    };

    // --- processor queries ------------------------------------------------
    add(
        "procs_kind_count_global",
        [=]() -> uint64_t {
          return Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC).count();
        },
        Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC).count());
    {
      // one query object, executed again and again (and by every thread at
      // once): the cost of a query that is already attached to its result
      Machine::ProcessorQuery reused(machine);
      reused.only_kind(Processor::LOC_PROC);
      add(
          "procs_kind_count_reused", [=]() -> uint64_t { return reused.count(); },
          reused.count());
    }
    iterate(Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC), &n);
    add(
        "procs_kind_iter_global",
        [=]() -> uint64_t {
          return iterate(Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC));
        },
        n);
    iterate(Machine::ProcessorQuery(machine).local_address_space().only_kind(
                Processor::LOC_PROC),
            &n);
    add(
        "procs_kind_iter_local",
        [=]() -> uint64_t {
          return iterate(Machine::ProcessorQuery(machine).local_address_space().only_kind(
              Processor::LOC_PROC));
        },
        n);
    add(
        "procs_kind_first_local",
        [=]() -> uint64_t {
          return Machine::ProcessorQuery(machine)
              .local_address_space()
              .only_kind(Processor::LOC_PROC)
              .first()
              .id;
        },
        1);
    iterate(Machine::ProcessorQuery(machine)
                .only_kind(Processor::LOC_PROC)
                .has_affinity_to(sysmem),
            &n);
    add(
        "procs_sharing_mem_iter",
        [=]() -> uint64_t {
          return iterate(Machine::ProcessorQuery(machine)
                             .only_kind(Processor::LOC_PROC)
                             .has_affinity_to(sysmem));
        },
        n);
    add(
        "procs_sharing_mem_count",
        [=]() -> uint64_t {
          return Machine::ProcessorQuery(machine)
              .only_kind(Processor::LOC_PROC)
              .has_affinity_to(sysmem)
              .count();
        },
        n);
    // a kind and memory with no affinity between them
    add(
        "procs_kind_affinity_miss",
        [=]() -> uint64_t {
          return Machine::ProcessorQuery(machine)
              .only_kind(Processor::LOC_PROC)
              .has_affinity_to(kind_miss)
              .count();
        },
        0, kind_miss.exists());
    add(
        "procs_best_for_mem_global",
        [=]() -> uint64_t {
          return Machine::ProcessorQuery(machine)
              .only_kind(Processor::LOC_PROC)
              .best_affinity_to(sysmem)
              .first()
              .id;
        },
        1);
    iterate(
        Machine::ProcessorQuery(machine).same_address_space_as(sysmem).has_affinity_to(
            sysmem),
        &n);
    add(
        "procs_sharing_mem_iter_local",
        [=]() -> uint64_t {
          return iterate(Machine::ProcessorQuery(machine)
                             .same_address_space_as(sysmem)
                             .has_affinity_to(sysmem));
        },
        n);

    // --- memory queries ---------------------------------------------------
    add(
        "mems_kind_count_global",
        [=]() -> uint64_t {
          return Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).count();
        },
        Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).count());
    {
      Machine::MemoryQuery reused(machine);
      reused.only_kind(Memory::SYSTEM_MEM);
      add(
          "mems_kind_count_reused", [=]() -> uint64_t { return reused.count(); },
          reused.count());
    }
    iterate(Machine::MemoryQuery(machine).has_affinity_to(p), &n);
    add(
        "mems_visible_iter_global",
        [=]() -> uint64_t {
          return iterate(Machine::MemoryQuery(machine).has_affinity_to(p));
        },
        n);
    iterate(Machine::MemoryQuery(machine).same_address_space_as(p).has_affinity_to(p),
            &n);
    add(
        "mems_visible_iter_local",
        [=]() -> uint64_t {
          return iterate(
              Machine::MemoryQuery(machine).same_address_space_as(p).has_affinity_to(p));
        },
        n);
    add(
        "mems_visible_kind_first_global",
        [=]() -> uint64_t {
          return Machine::MemoryQuery(machine)
              .only_kind(best_kind)
              .has_affinity_to(p)
              .first()
              .id;
        },
        1);
    add(
        "mems_visible_kind_first_local",
        [=]() -> uint64_t {
          return Machine::MemoryQuery(machine)
              .same_address_space_as(p)
              .only_kind(best_kind)
              .has_affinity_to(p)
              .first()
              .id;
        },
        1);
    add(
        "mems_best_for_proc_global",
        [=]() -> uint64_t {
          return Machine::MemoryQuery(machine)
              .only_kind(best_kind)
              .best_affinity_to(p)
              .first()
              .id;
        },
        1);
    add(
        "mems_best_for_proc_local",
        [=]() -> uint64_t {
          return Machine::MemoryQuery(machine)
              .same_address_space_as(p)
              .only_kind(best_kind)
              .best_affinity_to(p)
              .first()
              .id;
        },
        1);

    // --- direct affinity lookups -----------------------------------------
    add(
        "pma_proc_mem",
        [=]() -> uint64_t {
          std::vector<Machine::ProcessorMemoryAffinity> v;
          return machine.get_proc_mem_affinity(v, cpu, sysmem);
        },
        1);
    {
      std::vector<Machine::ProcessorMemoryAffinity> v;
      n = machine.get_proc_mem_affinity(v, cpu);
    }
    add(
        "pma_proc_all",
        [=]() -> uint64_t {
          std::vector<Machine::ProcessorMemoryAffinity> v;
          return machine.get_proc_mem_affinity(v, cpu);
        },
        n);
    add(
        "mma_mem_mem",
        [=]() -> uint64_t {
          std::vector<Machine::MemoryMemoryAffinity> v;
          return machine.get_mem_mem_affinity(v, mm_src, mm_dst);
        },
        1);
    add(
        "has_affinity_pm_hit",
        [=]() -> uint64_t { return machine.has_affinity(cpu, mem_hit) ? 1 : 0; }, 1);
    add(
        "has_affinity_pm_miss",
        [=]() -> uint64_t { return machine.has_affinity(cpu, mem_miss) ? 1 : 0; }, 1,
        mem_miss.exists());
    add(
        "has_affinity_mm",
        [=]() -> uint64_t { return machine.has_affinity(mm_src, mm_dst) ? 1 : 0; }, 1);
    {
      std::set<Memory> ms;
      machine.get_visible_memories(cpu, ms);
      n = ms.size();
    }
    add(
        "visible_memories_set",
        [=]() -> uint64_t {
          std::set<Memory> ms;
          machine.get_visible_memories(cpu, ms);
          return ms.size();
        },
        n);
    {
      std::set<Processor> ps;
      machine.get_shared_processors(sysmem, ps);
      n = ps.size();
    }
    add(
        "shared_processors_set",
        [=]() -> uint64_t {
          std::set<Processor> ps;
          machine.get_shared_processors(sysmem, ps);
          return ps.size();
        },
        n);

    // --- handle properties (hit the runtime's processor/memory tables) ---
    {
      std::shared_ptr<std::atomic<size_t>> cursor =
          std::make_shared<std::atomic<size_t>>(0);
      add(
          "handle_props",
          [=]() -> uint64_t {
            size_t i = cursor->fetch_add(1);
            Processor pp = sampled_procs[i % sampled_procs.size()];
            Memory mm = sampled_mems[i % sampled_mems.size()];
            return (uint64_t)pp.kind() + (uint64_t)mm.kind() + mm.capacity() +
                   pp.address_space() + mm.address_space();
          },
          2);
    }

    // --- composite patterns taken from Legion's DefaultMapper -----------
    add(
        "composite_must_epoch",
        [=]() -> uint64_t {
          uint64_t acc = 0;
          for(size_t i = 0; i < local_procs.size(); i++)
            for(size_t j = 0; j < local_mems.size(); j++)
              acc += machine.has_affinity(local_procs[i], local_mems[j]) ? 1 : 0;
          return acc;
        },
        local_procs.size() * local_mems.size());
    add(
        "composite_select_sources",
        [=]() -> uint64_t {
          uint64_t acc = 0;
          std::vector<Machine::MemoryMemoryAffinity> v;
          for(size_t i = 0; i < source_mems.size(); i++) {
            machine.get_mem_mem_affinity(v, source_mems[i], sysmem);
            acc += v.empty() ? 0 : v[0].bandwidth;
          }
          return acc;
        },
        source_mems.size());
  }

  double time_loop(const Shape &s, int iters, uint64_t &sink)
  {
    uint64_t acc = 0;
    long long t0 = Clock::current_time_in_nanoseconds();
    for(int i = 0; i < iters; i++)
      acc += s.fn();
    long long t1 = Clock::current_time_in_nanoseconds();
    sink += acc;
    return double(t1 - t0) / double(iters);
  }

  // mean per-thread latency of one operation with T threads issuing
  // operations concurrently
  double run_threads(const Shape &s, int nthreads, int iters, uint64_t &sink)
  {
    std::vector<double> per(nthreads, 0.0);
    std::vector<uint64_t> sinks(nthreads, 0);
    std::atomic<int> ready(0);
    std::atomic<bool> go(false);
    std::vector<std::thread> threads;
    for(int t = 0; t < nthreads; t++)
      threads.emplace_back([&, t]() {
        inherit_process_affinity();
        ready.fetch_add(1);
        while(!go.load())
          std::this_thread::yield();
        per[t] = time_loop(s, iters, sinks[t]);
      });
    while(ready.load() < nthreads)
      std::this_thread::yield();
    go.store(true);
    double sum = 0;
    for(int t = 0; t < nthreads; t++) {
      threads[t].join();
      sum += per[t];
      sink += sinks[t];
    }
    return sum / nthreads;
  }

  int calibrate(const Shape &s, uint64_t &sink)
  {
    if(cfg_iters > 0)
      return cfg_iters;
    time_loop(s, 1, sink); // warm
    int iters = 8;
    double ns = time_loop(s, iters, sink);
    while((ns * iters < 2e6) && (iters < cfg_max_iters)) {
      iters *= 4;
      ns = time_loop(s, iters, sink);
    }
    double target = double(cfg_target_ms) * 1e6;
    long long want = (long long)(target / std::max(ns, 1.0));
    if(want < 8)
      want = 8;
    if(want > cfg_max_iters)
      want = cfg_max_iters;
    return (int)want;
  }

  bool shape_selected(const std::string &name)
  {
    if(cfg_shapes.empty())
      return true;
    std::stringstream ss(cfg_shapes);
    std::string item;
    while(std::getline(ss, item, ','))
      if(!item.empty() && (name.find(item) != std::string::npos))
        return true;
    return false;
  }

  void top_level_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor proc)
  {
    Sample s;
    if(!build_sample(s)) {
      printf("FATAL: could not sample the machine\n");
      fflush(stdout);
      return;
    }
    size_t gpus =
        Machine::ProcessorQuery(s.machine).only_kind(Processor::TOC_PROC).count();
    printf("MACHINE ranks=%zu procs=%zu mems=%zu local_procs=%zu local_mems=%zu "
           "pmas_all=%zu gpus=%zu best_kind=%d\n",
           s.machine.get_address_space_count(), s.all_procs.size(), s.all_mems.size(),
           s.local_procs.size(), s.local_mems.size(), s.pmas_all, gpus, (int)s.best_kind);
    fflush(stdout);

    std::vector<int> thread_counts;
    {
      std::stringstream ss(cfg_threads);
      std::string item;
      while(std::getline(ss, item, ','))
        if(!item.empty())
          thread_counts.push_back(atoi(item.c_str()));
    }
    if(thread_counts.empty())
      thread_counts.push_back(1);

    std::vector<Shape> shapes;
    build_shapes(s, shapes);

    uint64_t sink = 0;
    for(size_t i = 0; i < shapes.size(); i++) {
      const Shape &sh = shapes[i];
      if(!sh.enabled || !shape_selected(sh.name)) {
        printf("SKIP shape=%s\n", sh.name.c_str());
        continue;
      }
      int iters = calibrate(sh, sink);
      for(size_t ti = 0; ti < thread_counts.size(); ti++) {
        int T = thread_counts[ti];
        std::vector<double> samples;
        for(int r = 0; r < cfg_reps; r++)
          samples.push_back(run_threads(sh, T, iters, sink));
        std::sort(samples.begin(), samples.end());
        double median = samples[samples.size() / 2];
        printf("RESULT shape=%s threads=%d iters=%d elems=%zu ns_per_op=%.1f min=%.1f "
               "max=%.1f\n",
               sh.name.c_str(), T, iters, sh.elems, median, samples.front(),
               samples.back());
        fflush(stdout);
      }
    }
    printf("DONE sink=%llu\n", (unsigned long long)sink);
    fflush(stdout);
  }

} // namespace

int main(int argc, char **argv)
{
#ifdef __linux__
  if(sched_getaffinity(0, sizeof(initial_mask), &initial_mask) == 0)
    have_initial_mask = true;
#endif
  Runtime rt;
  bool ok = rt.init(&argc, &argv);
  assert(ok);

  CommandLineParser cp;
  cp.add_option_int("-iters", cfg_iters)
      .add_option_int("-reps", cfg_reps)
      .add_option_int("-target_ms", cfg_target_ms)
      .add_option_int("-max_iters", cfg_max_iters)
      .add_option_int("-sample", cfg_sample)
      .add_option_string("-threads", cfg_threads)
      .add_option_string("-shapes", cfg_shapes);
  ok = cp.parse_command_line(argc, (const char **)argv);
  if(!ok) {
    fprintf(stderr, "error parsing command line\n");
    return 1;
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // the first CPU processor in the machine runs the measurement
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  return rt.wait_for_shutdown();
}
