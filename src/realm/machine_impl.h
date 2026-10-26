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

// implementation for Realm Machine class

#ifndef REALM_MACHINE_IMPL_H
#define REALM_MACHINE_IMPL_H

#include "realm/machine.h"
#include "realm/network.h"
#include "realm/mutex.h"
#include "realm/atomics.h"

#include <vector>
#include <set>
#include <optional>
#include <memory>

namespace Realm {

  template <typename KT, typename AT>
  struct MachineAffinityInfo {
    MachineAffinityInfo(void);
    ~MachineAffinityInfo(void);
    bool add_affinity(KT key, const AT &aff, bool is_local);

    std::map<KT, AT *> all;
    std::map<KT, AT *> local;
    std::map<KT, AT *> best;
  };

  struct MachineProcInfo {
    MachineProcInfo(Processor _p);
    ~MachineProcInfo(void);

    bool add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma);

    Processor p;
    MachineAffinityInfo<Memory, Machine::ProcessorMemoryAffinity> pmas;
  };

  struct MachineMemInfo {
    MachineMemInfo(Memory _m);
    ~MachineMemInfo(void);

    bool add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma);
    bool add_mem_mem_affinity(const Machine::MemoryMemoryAffinity &mma);

    Memory m;
    MachineAffinityInfo<Processor, Machine::ProcessorMemoryAffinity> pmas;
    MachineAffinityInfo<Memory, Machine::MemoryMemoryAffinity> mmas_out;
    MachineAffinityInfo<Memory, Machine::MemoryMemoryAffinity> mmas_in;
  };

  struct MachineNodeInfo {
    MachineNodeInfo(int _node, RuntimeImpl *_runtime_impl);
    ~MachineNodeInfo(void);
    bool add_processor(Processor p);
    bool add_memory(Memory m);
    bool add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma);
    bool add_mem_mem_affinity(const Machine::MemoryMemoryAffinity &mma);
    bool add_process_info(const Machine::ProcessInfo &proc_info);

    void update_kind_maps(void);

    int node;

    // The runtime_impl is not const because we calls get_processor_impl(), which is not
    // const but the runtime_impl should not be modified in this class
    RuntimeImpl *runtime_impl{nullptr};

    Machine::ProcessInfo *process_info{nullptr};

    std::map<Processor, MachineProcInfo *> procs;
    std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>> proc_by_kind;

    std::map<Memory, MachineMemInfo *> mems;
    std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>> mem_by_kind;
  };

  class MachineImpl {
  public:
    MachineImpl(RuntimeImpl *_runtime_impl);
    ~MachineImpl(void);

    RuntimeImpl *get_runtime_impl(void) const;

    void get_all_memories(std::set<Memory> &mset) const;
    void get_all_processors(std::set<Processor> &pset) const;

    void get_local_processors(std::set<Processor> &pset) const;
    void get_local_processors_by_kind(std::set<Processor> &pset,
                                      Processor::Kind kind) const;

    // Return the set of memories visible from a processor
    void get_visible_memories(Processor p, std::set<Memory> &mset, bool local_only) const;

    // Return the set of memories visible from a memory
    void get_visible_memories(Memory m, std::set<Memory> &mset, bool local_only) const;

    // Return the set of processors which can all see a given memory
    void get_shared_processors(Memory m, std::set<Processor> &pset,
                               bool local_only) const;

    bool get_process_info(Processor p, Machine::ProcessInfo *info) const;

    bool has_affinity(Processor p, Memory m, Machine::AffinityDetails *details = 0) const;
    bool has_affinity(Memory m1, Memory m2, Machine::AffinityDetails *details = 0) const;

    int get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity> &result,
                              Processor restrict_proc = Processor::NO_PROC,
                              Memory restrict_memory = Memory::NO_MEMORY,
                              bool local_only = true) const;

    int get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity> &result,
                             Memory restrict_mem1 = Memory::NO_MEMORY,
                             Memory restrict_mem2 = Memory::NO_MEMORY,
                             bool local_only = true) const;

    void parse_node_announce_data(int node_id, const void *args, size_t arglen,
                                  bool remote);

    void add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma,
                               bool lock_held = false);

    void add_subscription(Machine::MachineUpdateSubscriber *subscriber);
    void remove_subscription(Machine::MachineUpdateSubscriber *subscriber);

    void update_kind_maps(void);

    void enumerate_mem_mem_affinities(void);

    void add_process_info(int node_id, const Machine::ProcessInfo &process_info,
                          bool lock_held = false);

    // guards nodeinfos and everything reachable from it. Readers: the
    //  accessors above and every machine walk in the query implementations
    //  below (which may not call the locking accessors while they hold it -
    //  the lock is not recursive - and use the *_unlocked variants instead).
    //  Writers: node announcements, affinity updates and the kind map
    //  rebuild. Reads vastly outnumber writes (which in practice all happen
    //  at startup), so this is a striped lock: readers never touch a shared
    //  cache line, and a writer is not starved by a steady stream of them.
    mutable StripedRWLock mutex;
    std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
    std::set<Machine::MachineUpdateSubscriber *> subscribers;

    std::map<int, MachineNodeInfo *> nodeinfos;

  protected:
    friend class ProcessorQueryImpl;
    friend class MemoryQueryImpl;

    // the bodies of the public accessors, for callers already holding 'mutex'
    int
    get_proc_mem_affinity_unlocked(std::vector<Machine::ProcessorMemoryAffinity> &result,
                                   Processor restrict_proc, Memory restrict_memory,
                                   bool local_only) const;
    int get_mem_mem_affinity_unlocked(std::vector<Machine::MemoryMemoryAffinity> &result,
                                      Memory restrict_mem1, Memory restrict_mem2,
                                      bool local_only) const;

    MachineNodeInfo *get_nodeinfo(int node) const;
    MachineNodeInfo *get_nodeinfo(Processor p) const;
    MachineNodeInfo *get_nodeinfo(Memory m) const;
    void invalidate_query_caches();

    RuntimeImpl *runtime_impl{nullptr};
  };

  template <typename T, typename T2>
  class QueryPredicate {
  public:
    virtual ~QueryPredicate(void){};

    virtual QueryPredicate<T, T2> *clone(void) const = 0;

    virtual bool matches_predicate(const MachineImpl *machine, T thing,
                                   const T2 *info = 0) const = 0;
  };

  typedef QueryPredicate<Processor, MachineProcInfo> ProcQueryPredicate;

  class ProcessorHasAffinityPredicate : public ProcQueryPredicate {
  public:
    ProcessorHasAffinityPredicate(Memory _memory, unsigned _min_bandwidth,
                                  unsigned _max_latency);

    virtual ProcQueryPredicate *clone(void) const;

    virtual bool matches_predicate(const MachineImpl *machine, Processor thing,
                                   const MachineProcInfo *info = 0) const;

  protected:
    Memory memory;
    unsigned min_bandwidth;
    unsigned max_latency;
  };

  namespace Config {
    extern bool use_machine_query_cache;
  };

  enum QueryType
  {
    QUERY_NEXT = 0,
    QUERY_FIRST,
    QUERY_RANDOM,
  };

  // the number of values of Processor::Kind and Memory::Kind, for the arrays
  //  indexed by kind below
#define REALM_COUNT_KIND(name, desc) +1
  static constexpr unsigned NUM_PROCESSOR_KINDS =
      0 REALM_PROCESSOR_KINDS(REALM_COUNT_KIND);
  static constexpr unsigned NUM_MEMORY_KINDS = 0 REALM_MEMORY_KINDS(REALM_COUNT_KIND);
#undef REALM_COUNT_KIND

  // The machine query cache.
  //
  // A query result is an immutable, sorted list stamped with the generation
  //  of the cache it was built for (the generation is bumped by every machine
  //  update). Lists live in snapshots: a snapshot holds the lists for one
  //  generation and is itself immutable once published. Readers take the
  //  current snapshot with one atomic load, so a query that the cache can
  //  serve writes no shared memory at all; this is what lets concurrent
  //  queries scale. A query remembers the list it last used (see
  //  ProcessorQueryImpl::attached), which is still valid while its generation
  //  is current, so iterating a cached list costs a binary search per step.
  //
  // A replaced snapshot is not freed until cleanup_query_caches(), because a
  //  query may still hold a pointer into it; the number of replaced snapshots
  //  is bounded by the number of machine updates that happened while queries
  //  were being made, which is small since updates happen at startup.
  template <typename T>
  struct QueryResultList {
    unsigned generation = 0;
    std::vector<T> items; // sorted by id
  };

  // the lists of all things of each kind
  template <typename T, unsigned NUM_KINDS>
  struct QueryKindCache {
    unsigned generation = 0;
    QueryResultList<T> by_kind[NUM_KINDS];
  };

  // the publication slot for one kind of snapshot
  template <typename S>
  class QueryCacheSlot {
  public:
    QueryCacheSlot(void)
      : current(nullptr)
    {}

    // the published snapshot if it is for 'generation', else null
    const S *get(unsigned generation) const
    {
      const S *s = current.load_acquire();
      return (s && (s->generation == generation)) ? s : nullptr;
    }

    // publishes 'built' (taking ownership) and returns it - unless a snapshot
    //  for the same generation got there first, which is returned instead, or
    //  the cache generation has moved on from 'built', in which case null is
    //  returned (and the caller walks the machine)
    const S *install(S *built, unsigned live_generation);

    // frees every snapshot ever published
    void cleanup(void);

  protected:
    atomic<const S *> current;
    Mutex mutex; // serializes install and cleanup
    std::vector<const S *> retired;
  };

  class ProcessorQueryImpl {
  public:
    ProcessorQueryImpl(const Machine &_machine);
    ProcessorQueryImpl(const MachineImpl *_machine_impl);

    typedef QueryResultList<Processor> CachedList;
    typedef QueryKindCache<Processor, NUM_PROCESSOR_KINDS> KindCache;

    // the processors of each kind with an affinity to each memory - any
    //  affinity in the model, local or not, exactly as the walk sees them. A
    //  kind and memory that are not in the map have no such processors (and
    //  are served the empty list), so a miss costs no more than a hit.
    struct AffinityCache {
      unsigned generation = 0;
      std::map<Memory, CachedList> by_kind_and_mem[NUM_PROCESSOR_KINDS];
      CachedList none;
    };

    // the shared query cache, rebuilt lazily after the machine changes
    static atomic<unsigned int> cache_invalid_count;
    static QueryCacheSlot<KindCache> kind_cache;
    static QueryCacheSlot<AffinityCache> affinity_cache;

  protected:
    // these things are refcounted and copied-on-write
    ProcessorQueryImpl(const ProcessorQueryImpl &copy_from);
    ~ProcessorQueryImpl(void);

  public:
    void add_reference(void);
    void remove_reference(void);
    // makes and returns if a copy if more than one reference is held
    ProcessorQueryImpl *writeable_reference(void);

    void restrict_to_node(int new_node_id);
    void restrict_to_kind(Processor::Kind new_kind);
    void add_predicate(ProcQueryPredicate *pred);

    Processor first_match(void) const;
    Processor next_match(Processor after);
    size_t count_matches(void) const;
    Processor random_match(void) const;

    void set_cached_mem(Memory m)
    {
      cached_mem = m;
      if(predicates.size() == 1)
        is_cached_mem = true;
      else
        is_cached_mem = false;
    };
    void reset_cached_mem()
    {
      cached_mem = Memory::NO_MEMORY;
      is_cached_mem = false;
    };
    Processor cache_next(Processor after);

  public:
    // Best affinity cost function structure
    struct BestAffinityCostFn {
      Memory memory;
      int bandwidth_weight;
      int latency_weight;
    };

    // Friend declarations for Machine::ProcessorQuery methods
    friend Machine::ProcessorQuery &
    Machine::ProcessorQuery::best_affinity_to(Memory m, int bandwidth_weight,
                                              int latency_weight);

  protected:
    atomic<int> references;
    const MachineImpl *machine;
    bool is_restricted_node;
    int restricted_node_id;
    bool is_restricted_kind;
    Processor::Kind restricted_kind;
    std::vector<ProcQueryPredicate *> predicates;

    // Best affinity cost function (not a predicate)
    std::optional<BestAffinityCostFn> best_affinity_cost;

    Memory cached_mem;
    bool is_cached_mem;

    // the list this query last used: one in a shared snapshot, or one of
    //  the best-affinity lists it built itself. It is current while its
    //  generation matches cache_invalid_count. Copies of a query share this
    //  object and may read concurrently, so this is an atomic; the lists it
    //  points to are immutable and outlive it (see QueryCacheSlot)
    mutable atomic<const CachedList *> attached;
    // best-affinity results depend on the query's cost function, so they are
    //  built per query. Every list this query built is kept until the query
    //  is destroyed, because another copy of it may still be reading an older
    //  one; a query builds one list per machine update it observes
    mutable UnfairMutex build_mutex;
    mutable std::vector<std::unique_ptr<CachedList>> own_lists;

    // the shared list that serves this query (attaching to it as a side
    //  effect), or null if the shared cache does not cover this query
    const CachedList *cached_list() const;
    // builders of the shared snapshots for 'generation'; null if the machine
    //  changed meanwhile (the caller then walks it)
    const KindCache *build_kind_cache(unsigned generation) const;
    const AffinityCache *build_affinity_cache(unsigned generation) const;
    // the list of this query's best-affinity results, building it if needed
    const CachedList *best_affinity_list() const;
    // builds the best-affinity results of this query into 'result'
    void build_best_affinity_list(std::vector<Processor> &result) const;
    // remembers 'list' as this query's current result
    const CachedList *attach(const CachedList *list) const;
    // this query's attached list if it is still current, else null
    const CachedList *attached_list() const;
    // drops the attached list - every change to the query's constraints does this
    void reset_cached_list();
    bool cached_query(Processor p, Processor &pval);
    bool cached_query(Processor &pval, QueryType q) const;
    bool cached_query(size_t &count) const;
  };

  typedef QueryPredicate<Memory, MachineMemInfo> MemoryQueryPredicate;

  class MemoryHasProcAffinityPredicate : public MemoryQueryPredicate {
  public:
    MemoryHasProcAffinityPredicate(Processor _proc, unsigned _min_bandwidth,
                                   unsigned _max_latency);

    virtual MemoryQueryPredicate *clone(void) const;

    virtual bool matches_predicate(const MachineImpl *machine, Memory thing,
                                   const MachineMemInfo *info = 0) const;

  protected:
    Processor proc;
    unsigned min_bandwidth;
    unsigned max_latency;
  };

  class MemoryHasMemAffinityPredicate : public MemoryQueryPredicate {
  public:
    MemoryHasMemAffinityPredicate(Memory _memory, unsigned _min_bandwidth,
                                  unsigned _max_latency);

    virtual MemoryQueryPredicate *clone(void) const;

    virtual bool matches_predicate(const MachineImpl *machine, Memory thing,
                                   const MachineMemInfo *info = 0) const;

  protected:
    Memory memory;
    unsigned min_bandwidth;
    unsigned max_latency;
  };

  class MemoryQueryImpl {
  public:
    MemoryQueryImpl(const Machine &_machine);
    MemoryQueryImpl(const MachineImpl *_machine_impl);

    // see ProcessorQueryImpl
    typedef QueryResultList<Memory> CachedList;
    typedef QueryKindCache<Memory, NUM_MEMORY_KINDS> KindCache;

    // the shared query cache: memories by kind (see ProcessorQueryImpl)
    static atomic<unsigned int> cache_invalid_count;
    static QueryCacheSlot<KindCache> kind_cache;

  protected:
    // these things are refcounted and copied-on-write
    MemoryQueryImpl(const MemoryQueryImpl &copy_from);
    ~MemoryQueryImpl(void);

  public:
    void add_reference(void);
    void remove_reference(void);
    // makes and returns if a copy if more than one reference is held
    MemoryQueryImpl *writeable_reference(void);

    void restrict_to_node(int new_node_id);
    void restrict_to_kind(Memory::Kind new_kind);
    void restrict_by_capacity(size_t new_min_bytes);
    void add_predicate(MemoryQueryPredicate *pred);
    // records the processor of a has_affinity_to(Processor) predicate when it
    // is the query's only predicate, so matches can be enumerated from that
    // processor's affinity map instead of every memory in the machine
    void set_affinity_proc(Processor p);

    Memory first_match(void) const;
    Memory next_match(Memory after) const;
    size_t count_matches(void) const;
    Memory random_match(void) const;
    Memory cache_next(Memory after);
    bool cached_query(Memory p, Memory &pval);
    bool cached_query(Memory &pval, QueryType q) const;
    bool cached_query(size_t &count) const;

  public:
    // Best affinity cost function structures
    struct BestProcAffinityCostFn {
      Processor proc;
      int bandwidth_weight;
      int latency_weight;
    };
    struct BestMemAffinityCostFn {
      Memory memory;
      int bandwidth_weight;
      int latency_weight;
    };

    // Friend declarations for Machine::MemoryQuery methods
    friend Machine::MemoryQuery &
    Machine::MemoryQuery::best_affinity_to(Memory m, int bandwidth_weight,
                                           int latency_weight);
    friend Machine::MemoryQuery &
    Machine::MemoryQuery::best_affinity_to(Processor p, int bandwidth_weight,
                                           int latency_weight);

  protected:
    atomic<int> references;
    const MachineImpl *machine;
    bool is_restricted_node;
    int restricted_node_id;
    bool is_restricted_kind;
    Memory::Kind restricted_kind;
    size_t restricted_min_capacity;
    // see ProcessorQueryImpl
    mutable atomic<const CachedList *> attached;
    mutable UnfairMutex build_mutex;
    mutable std::vector<std::unique_ptr<CachedList>> own_lists;
    std::vector<MemoryQueryPredicate *> predicates;
    Processor affinity_proc; // see set_affinity_proc

    // Best affinity cost functions (not predicates)
    std::optional<BestProcAffinityCostFn> best_proc_affinity_cost;
    std::optional<BestMemAffinityCostFn> best_mem_affinity_cost;

    // see ProcessorQueryImpl
    const CachedList *cached_list() const;
    const KindCache *build_kind_cache(unsigned generation) const;
    const CachedList *best_affinity_list() const;
    void build_best_affinity_list(std::vector<Memory> &result) const;
    const CachedList *attach(const CachedList *list) const;
    const CachedList *attached_list() const;
    void reset_cached_list();
    // the affinity map of a processor the query is anchored to, or NULL
    const std::map<Memory, Machine::ProcessorMemoryAffinity *> *
    anchor_affinities(Processor p, bool local_only) const;
    // applies the node, kind, capacity and predicate restrictions to one memory
    bool candidate_matches(Memory m) const;
  };

  extern MachineImpl *machine_singleton;
  inline MachineImpl *get_machine(void) { return machine_singleton; }
  extern void cleanup_query_caches();
  // active messages

  enum NodeAnnounceTag
  {
    NODE_ANNOUNCE_INVALID = 0,
    NODE_ANNOUNCE_PROC,   // PROC id kind
    NODE_ANNOUNCE_MEM,    // MEM id size
    NODE_ANNOUNCE_IB_MEM, // IB_MEM id size
    NODE_ANNOUNCE_PMA,    // PMA proc_id mem_id bw latency
    NODE_ANNOUNCE_DMA_CHANNEL,
    NODE_ANNOUNCE_PROCESS_INFO,
  };

}; // namespace Realm

  // include "machine_impl.inl"

#endif // ifndef REALM_MACHINE_IMPL_H
