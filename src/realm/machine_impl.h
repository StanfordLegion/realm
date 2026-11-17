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
  //  is current, and each thread remembers where its last step through a list
  //  stopped, so iterating a cached list in order costs a few comparisons per
  //  step.
  //
  // A replaced snapshot is retired, not freed, until cleanup_query_caches()
  //  at runtime shutdown, because a query may still hold a pointer into it
  //  (attached_list() dereferences the pointer to compare generations). So
  //  every machine update that is followed by a query retains one snapshot of
  //  each kind, each a copy of the machine's lists. That is bounded today
  //  because the machine model only changes at startup. Snapshot builds are
  //  single-flight (QueryCacheSlot::get_or_build): an update costs one build
  //  per snapshot kind, not one per querying thread.
  //
  // Before the machine model may change at run time (elasticity), retired
  //  snapshots need reclaiming. What would have to change: a query would keep
  //  the generation of its attached list beside the pointer (one 64-bit atomic
  //  holding the generation and an index into a table of per-generation
  //  snapshot headers, say), so that staleness is decided without touching
  //  the snapshot; then only a call in progress can be inside a retired
  //  snapshot, and since calls are short and never block while reading one,
  //  an epoch or quiescent-state scheme (readers announce the generation they
  //  entered; a snapshot is freed once every reader has moved past it) can
  //  free retired snapshots safely. Until then the retention is by design.
  template <typename T>
  struct QueryResultList {
    unsigned generation = 0;
    std::vector<T> items; // sorted by id
  };

  // the lists of all things of each kind, and of every kind together (for a
  //  query with no kind restriction)
  template <typename T, unsigned NUM_KINDS>
  struct QueryKindCache {
    unsigned generation = 0;
    QueryResultList<T> by_kind[NUM_KINDS];
    QueryResultList<T> all;
  };

  // the publication slot for one kind of snapshot
  template <typename S>
  class QueryCacheSlot {
  public:
    QueryCacheSlot(void)
      : current(nullptr)
      , builds(0)
    {}
    // the snapshots are freed by cleanup_query_caches() at runtime shutdown,
    //  not by this object's destructor: a process that exits without shutting
    //  Realm down may still have threads querying, and a hit on a leaked
    //  snapshot is harmless where a freed one is not. (The slot's own members
    //  are destroyed with the other statics at exit, like every Realm static;
    //  a thread still building a snapshot at that point is an application
    //  bug, as it was before.)

    // the published snapshot if it is for 'generation', else null
    const S *get(unsigned generation) const
    {
      const S *s = current.load_acquire();
      return (s && (s->generation == generation)) ? s : nullptr;
    }

    // the snapshot for 'generation', built by 'build(generation)' if it is
    //  missing. Builders are serialized: after a machine update the first
    //  thread to miss builds, and the others wait for its snapshot instead of
    //  each building their own, so an update costs one build however many
    //  threads query. Returns null if the live generation has moved on from
    //  'generation' (the caller then walks the machine), whether before the
    //  build started or while it ran. A build is attempted only for the live
    //  generation and generations only move forward, so each generation is
    //  built at most once. Lock order: build_mutex, then the machine's reader
    //  lock inside 'build', then 'mutex' in install(); machine writers take
    //  neither of this class's locks
    template <typename BuildFn>
    const S *get_or_build(unsigned generation, const atomic<unsigned int> &live,
                          BuildFn &&build)
    {
      AutoLock<UnfairMutex> al(build_mutex);
      const S *s = get(generation);
      if(s)
        return s; // built while we waited
      if(live.load() != generation)
        return nullptr;
      builds.fetch_add(1);
      S *built = build(generation);
      // the live generation is read after the build (call arguments are not
      //  sequenced), so a change during the build is seen by install()
      return install(built, live.load());
    }

    // publishes 'built' (taking ownership) and returns it - unless a snapshot
    //  for the same generation got there first, which is returned instead, or
    //  the cache generation has moved on from 'built', in which case null is
    //  returned (and the caller walks the machine)
    const S *install(S *built, unsigned live_generation);

    // frees every snapshot ever published (runtime shutdown)
    void cleanup(void);

    // how many snapshots have been built (for tests)
    unsigned long build_count(void) const { return builds.load(); }

  protected:
    atomic<const S *> current;
    Mutex mutex; // serializes install and cleanup
    std::vector<const S *> retired;
    UnfairMutex build_mutex; // serializes builders (see get_or_build)
    atomic<unsigned long> builds;
  };

  class ProcessorQueryImpl {
  public:
    ProcessorQueryImpl(const Machine &_machine);
    ProcessorQueryImpl(const MachineImpl *_machine_impl);

    typedef QueryResultList<Processor> CachedList;
    typedef QueryKindCache<Processor, NUM_PROCESSOR_KINDS> KindCache;

    // the processors of each kind, and of every kind, with an affinity to
    //  each memory - any affinity in the model, local or not, exactly as the
    //  walk sees them. A kind and memory that are not in the map have no such
    //  processors (and are served the empty list), so a miss costs no more
    //  than a hit.
    struct AffinityCache {
      unsigned generation = 0;
      std::map<Memory, CachedList> by_kind_and_mem[NUM_PROCESSOR_KINDS];
      std::map<Memory, CachedList> by_mem;
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

    // how many calls on the calling thread were answered by a pass over the
    //  whole machine model, and by a pass over one affinity map (for tests)
    static unsigned long walk_count(void);
    static unsigned long anchored_count(void);

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
    // records the memory of a has_affinity_to(Memory) predicate: a query the
    //  shared cache does not serve enumerates that memory's affinity map (the
    //  first such predicate's; every predicate is applied to each candidate)
    //  instead of every processor in the machine. "No memory" anchors too: it
    //  has no map, and no processor has an affinity to it
    void set_anchor_mem(Memory m)
    {
      if(!anchor_mem)
        anchor_mem = m;
    }
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
    std::optional<Memory> anchor_mem; // see set_anchor_mem

    // the list this query last used: one in a shared snapshot, or one of
    //  the best-affinity lists it built itself. It is current while its
    //  generation matches cache_invalid_count. Copies of a query share this
    //  object and may read concurrently, so this is an atomic; the lists it
    //  points to are immutable and outlive it (see QueryCacheSlot)
    mutable atomic<const CachedList *> attached;
    // a query's own result lists: best-affinity results depend on the query's
    //  cost function, so they are built per query; and a query the shared
    //  cache does not serve builds the list of its matches on its first
    //  next(), the start of an iteration, so that each step is a lookup in
    //  the list rather than another pass over the candidates (first(),
    //  count() and random() use that list while it is current but do not
    //  build one: a one-shot call keeps its one pass). The exception is a
    //  memory query anchored on a processor: its candidates are the few
    //  memories that processor reaches, so a step through them costs less
    //  than building a list would (see MemoryQueryImpl::cache_next). Across machine
    //  updates every list is kept, because another copy may still be reading an older
    //  one. Refinement holds the only reference to the implementation and
    //  frees all its lists; otherwise they live until the query is destroyed.
    //  Between refinements a query builds one list per machine update it
    //  observes. The first list lives in the query itself, the rest on the
    //  heap; the optional is constructed only when a list is built, so a query
    //  that never needs one pays nothing for it
    mutable UnfairMutex build_mutex;
    mutable std::optional<CachedList> first_own;
    mutable std::vector<std::unique_ptr<CachedList>> own_lists;

    // whether the shared cache can serve this query at all: at most one kind
    //  and one node, no best-affinity cost, and no predicate other than the one
    //  memory affinity the by-memory lists cover; anything else enumerates
    //  the anchor memory's affinity map, or walks the machine without one
    bool shared_cache_covers() const
    {
      return !(is_restricted_node && (restricted_node_id < 0)) &&
             !best_affinity_cost.has_value() && (predicates.empty() || is_cached_mem) &&
             (!is_restricted_kind || (unsigned(restricted_kind) < NUM_PROCESSOR_KINDS));
    }
    // the shared list that serves this query (attaching to it as a side
    //  effect), or null if the shared cache does not cover this query. The
    //  attached-list check (the hit path, which every cached call takes) and
    //  the coverage test (which every call of an uncovered query takes) are
    //  defined here so that they are inlined into the callers; the snapshot
    //  lookup, taken once per query per machine update, is out of line
    const CachedList *cached_list() const
    {
      const CachedList *list = attached_list();
      if(list)
        return list;
      return shared_cache_covers() ? cached_list_lookup() : nullptr;
    }
    // the rest of cached_list(), for a query the shared cache covers: its list
    //  in the current snapshot, attached
    REALM_ATTR_NOINLINE(const CachedList *cached_list_lookup() const);
    // the miss path of cached_list_lookup(): the snapshot for 'generation' is
    //  built (or waited for) and the query attached to its list
    REALM_ATTR_NOINLINE(const CachedList *cached_list_miss(unsigned generation) const);
    // the list for this query's kind and memory in an affinity snapshot
    const CachedList *list_in(const AffinityCache *ac) const;
    // the list for this query's kind (or of every kind) in a kind snapshot
    const CachedList *list_in(const KindCache *kc) const
    {
      return is_restricted_kind ? &kc->by_kind[restricted_kind] : &kc->all;
    }
    // builders of the shared snapshots for 'generation', run one at a time
    //  by QueryCacheSlot::get_or_build
    KindCache *build_kind_cache(unsigned generation) const;
    AffinityCache *build_affinity_cache(unsigned generation) const;
    // the list of this query's own results - its best-affinity results, or
    //  the matches of a query the shared cache does not serve - building it
    //  if it is missing or stale
    const CachedList *own_list() const;
    // next() over that list (see the definition)
    REALM_ATTR_NOINLINE(Processor own_list_next(Processor after) const);
    // builds this query's results into 'result' (one of the two below)
    void build_own_list(std::vector<Processor> &result) const;
    // builds the best-affinity results of this query into 'result'
    void build_best_affinity_list(std::vector<Processor> &result) const;
    // builds the matches of this query into 'result', from the anchor
    //  memory's affinity map or by a walk of the machine
    void build_match_list(std::vector<Processor> &result) const;
    // remembers 'list' as this query's current result
    const CachedList *attach(const CachedList *list) const;
    // this query's attached list if it is still current, else null
    const CachedList *attached_list() const
    {
      const CachedList *list = attached.load_acquire();
      return (list && (list->generation == cache_invalid_count.load())) ? list : nullptr;
    }
    // drops the attached list - every change to the query's constraints does this
    void reset_cached_list();
    bool cached_query(Processor p, Processor &pval);
    bool cached_query(Processor &pval, QueryType q) const;
    bool cached_query(size_t &count) const;
    // the affinity map of the memory the query is anchored to (every
    //  processor with an affinity to it), or NULL if it is not in the model
    const std::map<Processor, Machine::ProcessorMemoryAffinity *> *
    anchor_affinities(Memory m) const;
    // the processors of 'node' this query may return (its node and kind
    //  restrictions applied), or NULL if there are none
    const std::map<Processor, MachineProcInfo *> *candidate_list(NodeID node) const;
    // the matches of an anchored query after 'after' (all of them for "no
    //  processor"), in id order, each handed to fn until it returns false;
    //  the caller holds the machine's reader lock
    template <typename Fn>
    void for_each_anchored(Processor after, Fn &&fn) const;
    // the accessors over the anchor's map, for a query the shared cache does
    //  not serve (see first_match)
    Processor anchored_first(void) const;
    Processor anchored_next(Processor after) const;
    size_t anchored_count_matches(void) const;
    Processor anchored_random(void) const;
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

    // the memories of each kind, and of every kind, with an affinity to (an
    //  edge into) each memory, as MemoryHasMemAffinityPredicate sees them
    //  (see ProcessorQueryImpl::AffinityCache)
    struct AffinityCache {
      unsigned generation = 0;
      std::map<Memory, CachedList> by_kind_and_mem[NUM_MEMORY_KINDS];
      std::map<Memory, CachedList> by_mem;
      CachedList none;
    };

    // the shared query cache: memories by kind, and by the memory they have
    //  an affinity to (see ProcessorQueryImpl)
    static atomic<unsigned int> cache_invalid_count;
    static QueryCacheSlot<KindCache> kind_cache;
    static QueryCacheSlot<AffinityCache> affinity_cache;

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
    // the anchor of a query with affinity predicates: a query the shared
    //  cache does not serve enumerates the anchor's affinity map (every
    //  memory a processor reaches, or every memory with an edge into a
    //  memory), applying every restriction and predicate to each entry,
    //  instead of walking every memory in the machine. Any affinity
    //  predicate's target would do; a processor's map is always small (the
    //  memories one processor reaches) where a memory's incoming map may
    //  span the machine, so a processor anchors the query whenever it has a
    //  has_affinity_to(Processor), else the first has_affinity_to(Memory)
    //  does. A target that is not in the model ("no processor", "no
    //  memory") anchors too: it has no map, and nothing has an affinity to it
    void set_anchor_proc(Processor p)
    {
      if(!anchor_proc)
        anchor_proc = p;
    }
    void set_anchor_mem(Memory m)
    {
      if(!anchor_mem)
        anchor_mem = m;
    }
    // the memory of a has_affinity_to(Memory) predicate with no bounds, which
    //  the shared by-memory lists serve while it is the only predicate
    void set_cached_mem(Memory m)
    {
      cached_mem = m;
      is_cached_mem = (predicates.size() == 1);
    }
    void reset_cached_mem()
    {
      cached_mem = Memory::NO_MEMORY;
      is_cached_mem = false;
    }

    Memory first_match(void) const;
    Memory next_match(Memory after) const;
    size_t count_matches(void) const;
    Memory random_match(void) const;
    // see ProcessorQueryImpl
    static unsigned long walk_count(void);
    static unsigned long anchored_count(void);
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
    mutable std::optional<CachedList> first_own;
    mutable std::vector<std::unique_ptr<CachedList>> own_lists;
    std::vector<MemoryQueryPredicate *> predicates;
    // see set_anchor_proc and set_cached_mem
    Memory cached_mem;
    bool is_cached_mem;
    std::optional<Processor> anchor_proc;
    std::optional<Memory> anchor_mem;
    bool anchored() const { return anchor_proc.has_value() || anchor_mem.has_value(); }

    // Best affinity cost functions (not predicates)
    std::optional<BestProcAffinityCostFn> best_proc_affinity_cost;
    std::optional<BestMemAffinityCostFn> best_mem_affinity_cost;

    // see ProcessorQueryImpl
    bool shared_cache_covers() const
    {
      return !(is_restricted_node && (restricted_node_id < 0)) &&
             (predicates.empty() || is_cached_mem) && (restricted_min_capacity == 0) &&
             !best_proc_affinity_cost.has_value() &&
             !best_mem_affinity_cost.has_value() &&
             (!is_restricted_kind || (unsigned(restricted_kind) < NUM_MEMORY_KINDS));
    }
    const CachedList *list_in(const KindCache *kc) const
    {
      return is_restricted_kind ? &kc->by_kind[restricted_kind] : &kc->all;
    }
    const CachedList *list_in(const AffinityCache *ac) const;
    const CachedList *cached_list() const
    {
      const CachedList *list = attached_list();
      if(list)
        return list;
      return shared_cache_covers() ? cached_list_lookup() : nullptr;
    }
    REALM_ATTR_NOINLINE(const CachedList *cached_list_lookup() const);
    REALM_ATTR_NOINLINE(const CachedList *cached_list_miss(unsigned generation) const);
    KindCache *build_kind_cache(unsigned generation) const;
    AffinityCache *build_affinity_cache(unsigned generation) const;
    // see ProcessorQueryImpl
    const CachedList *own_list() const;
    REALM_ATTR_NOINLINE(Memory own_list_next(Memory after) const);
    void build_own_list(std::vector<Memory> &result) const;
    void build_best_affinity_list(std::vector<Memory> &result) const;
    void build_match_list(std::vector<Memory> &result) const;
    const CachedList *attach(const CachedList *list) const;
    const CachedList *attached_list() const
    {
      const CachedList *list = attached.load_acquire();
      return (list && (list->generation == cache_invalid_count.load())) ? list : nullptr;
    }
    void reset_cached_list();
    // the affinity map of a processor the query is anchored to, or NULL
    const std::map<Memory, Machine::ProcessorMemoryAffinity *> *
    anchor_affinities(Processor p, bool local_only) const;
    // the incoming affinity map of a memory (every memory with an edge into
    //  it, on any node), or NULL if it is not in the model
    const std::map<Memory, Machine::MemoryMemoryAffinity *> *
    anchor_sources(Memory m) const;
    // the memories of 'node' this query may return (its node and kind
    //  restrictions applied), or NULL if there are none
    const std::map<Memory, MachineMemInfo *> *candidate_list(NodeID node) const;
    // the matches of an anchored query after 'after' (all of them for "no
    //  memory"), in id order, each handed to fn until it returns false; the
    //  caller holds the machine's reader lock
    template <typename Fn>
    void for_each_anchored(Memory after, Fn &&fn) const;
    // the matches among the keys of an anchor's map (see for_each_anchored)
    template <typename Map, typename Fn>
    void scan_anchor(const Map &map, Memory after, Fn &&fn) const;
    // the accessors over the anchor's map, for a query the shared cache does
    //  not serve (see ProcessorQueryImpl::first_match)
    Memory anchored_first(void) const;
    Memory anchored_next(Memory after) const;
    size_t anchored_count_matches(void) const;
    Memory anchored_random(void) const;
    // the same-node outgoing affinity map of a memory, or NULL if it is not in
    //  the machine model (which "no memory" never is)
    const std::map<Memory, Machine::MemoryMemoryAffinity *> *
    local_mem_affinities(Memory m) const;
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
