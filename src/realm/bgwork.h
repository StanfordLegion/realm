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

// manager for background work that can be performed by available threads

#ifndef REALM_BGWORK_H
#define REALM_BGWORK_H

#include "realm/atomics.h"
#include "realm/threads.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"
#include "realm/timers.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace Realm {

  class BackgroundWorkItem;
  class BackgroundWorkThread;
  class BackgroundWorkManager;

  // Background work profiling for Realm
  //
  // Binary file format specification (RBWP = Realm Background Work Profile):
  //
  // FILE HEADER (36 bytes, written at start, counts/offset patched at shutdown):
  //   Magic:                    4 bytes  "RBWP"
  //   Version:                  uint16_t (currently 1)
  //   Flags:                    uint16_t (bit 0 = has fine-grained data)
  //   Node ID:                  uint32_t
  //   Clock zero time:          int64_t  (nanoseconds, absolute)
  //   Work item descriptor count: uint32_t  (patched at shutdown)
  //   Sub-item descriptor count:  uint32_t  (patched at shutdown)
  //   Descriptor table offset:    uint64_t  (patched at shutdown)
  //
  // DATA BLOCKS (appended during run and at shutdown, starting at offset 36):
  //   Block header:
  //     Thread ID:       uint64_t
  //     Block sequence:  uint32_t
  //     Record count:    uint32_t
  //     Base timestamp:  int64_t
  //     Data size:       uint32_t  (uncompressed)
  //     Compressed size: uint32_t  (0 = uncompressed)
  //   Block data:        uint8_t[compressed_size or data_size]
  //
  // DESCRIPTOR TABLES (written at shutdown, at descriptor_table_offset):
  //
  // WORK ITEM DESCRIPTOR TABLE:
  //   For each work item:
  //     Slot:       uint16_t
  //     Name length: uint16_t
  //     Name:       char[name_length]  (not null-terminated)
  //
  // SUB-ITEM DESCRIPTOR TABLE (follows work item table):
  //   For each sub-item:
  //     ID:         uint16_t
  //     Type:       uint8_t  (0=AM_HANDLER, 1=XFER_CHANNEL, 2=DEPPART_OP, 3=GPU_REAP)
  //     Name length: uint16_t
  //     Name:       char[name_length]  (not null-terminated)
  //
  // RECORDS within a block (variable-length, packed):
  //   Timestamp delta:  2, 4, or 8 bytes (see encoding below)
  //   Record type:      uint8_t
  //   Payload:          depends on record type
  //
  // Timestamp delta encoding:
  //   If delta fits in 15 bits:  2 bytes, high bit 0:  0bbb bbbb bbbb bbbb
  //   If delta fits in 30 bits:  4 bytes, high bits 10: 10bb bbbb ... bbbb bbbb
  //   Otherwise:                 8 bytes, high bits 11: 11xx xxxx + 7 more bytes
  //     (stores absolute timestamp, not delta)
  //
  // Record types and payloads:
  //   COARSE_BEGIN (0x01): uint8_t slot
  //   COARSE_END   (0x02): (no payload)
  //   FINE_BEGIN   (0x11): uint16_t sub_item_id
  //   FINE_END     (0x12): (no payload)
  //   GPU_WORK     (0x21): uint64_t proc_id, uint8_t slot, int64_t start, int64_t stop

  // Record type constants
  enum BgWorkProfileRecordType : uint8_t
  {
    BGWP_COARSE_BEGIN = 0x01,
    BGWP_COARSE_END = 0x02,
    BGWP_FINE_BEGIN = 0x11,
    BGWP_FINE_END = 0x12,
    BGWP_GPU_WORK = 0x21,
  };

  // Sub-item type constants
  enum BgWorkProfileSubItemType : uint8_t
  {
    BGWP_SUB_AM_HANDLER = 0,
    BGWP_SUB_XFER_CHANNEL = 1,
    BGWP_SUB_DEPPART_OP = 2,
    BGWP_SUB_GPU_REAP = 3,
  };

  // File format constants
  static const char BGWP_MAGIC[4] = {'R', 'B', 'W', 'P'};
  static const uint16_t BGWP_VERSION = 1;
  static const uint16_t BGWP_FLAG_HAS_FINE = 0x0001;

  struct ProfileBlock {
    static const size_t BLOCK_SIZE = 16384; // 16KB
    uint8_t data[BLOCK_SIZE];
    uint32_t used;
    int64_t base_timestamp;
    uint32_t num_records;
    uint64_t thread_id;
    uint32_t sequence;
    ProfileBlock *next;
  };

  class BgWorkProfileState {
  private:
    const int level; // 0 = disabled, 1 = coarse, 2 = fine
    ProfileBlock *current_block = nullptr;
    int64_t last_timestamp = 0; // for delta encoding
    const uint64_t thread_id = 0;

    // did_work flag: set to true by fine_begin, fine_end, gpu_work,
    // or explicitly by do_work implementations; checked by end()
    // to decide whether to record or discard
    bool did_work = false;

    // saved state for discard (set by begin())
    uint32_t begin_block_used = 0;
    uint32_t begin_block_num_records = 0;
    int64_t begin_last_timestamp = 0;

  public:
    BgWorkProfileState(void);
    ~BgWorkProfileState(void);
    // recording methods (all no-op when level == 0)
    inline void begin(uint8_t slot);
    inline void end(const TimeLimit &time_limit);
    inline void set_worked(bool worked)
    {
      if(level > 0)
        did_work = worked;
    }
    inline void discard(void);
    inline void fine_begin(uint16_t sub_item_id);
    inline void fine_end();
    inline void gpu_work(uint64_t proc_id, uint8_t slot, int64_t start, int64_t stop);
    ProfileBlock *flush(void);

  private:
    uint8_t *ensure_space(size_t needed);
    static size_t encode_timestamp(uint8_t *buf, int64_t delta, int64_t absolute);
  };

  namespace ThreadLocal {
    inline thread_local BgWorkProfileState *bgwork_profstate = nullptr;
  };

  struct BgWorkItemDescriptor {
    uint16_t slot;
    std::string name;
  };

  struct BgWorkSubItemDescriptor {
    uint16_t id;
    uint8_t type;
    std::string name;
  };

  class BgWorkProfileManager {
  public:
    BgWorkProfileManager();
    ~BgWorkProfileManager();

    // configuration (called before initialize)
    void set_level(int level);
    void set_logfile(const std::string &filename);
    void set_bufsize(size_t megabytes);

    // returns the configured profiling level (0, 1, or 2)
    int get_level() const;

    // lifecycle
    void initialize(uint32_t node_id);
    void shutdown();

    // descriptor registration (called during module init, before recording starts)
    void register_work_item(uint16_t slot, const std::string &name);
    uint16_t register_sub_item(uint8_t type, const std::string &name);

    // retroactively register any work items that were added to the manager
    // before the profiler was configured (e.g., network layer items)
    void register_existing_items(BackgroundWorkManager &mgr);

    // block management (called by recording functions)
    ProfileBlock *alloc_block(uint64_t thread_id);
    void complete_block(ProfileBlock *block);

    // state management: Workers register their profstate so shutdown can flush
    void register_thread_state(BgWorkProfileState *state);
    void unregister_thread_state(BgWorkProfileState *state);

  private:
    void write_file_header();
    void write_descriptor_tables();
    void flush_blocks_to_disk(size_t target_size);
    void flush_all_blocks();

    static const size_t HEADER_SIZE = 36;

    int profile_level;
    std::string logfile_pattern;
    bool initialized;
    size_t max_buffer_bytes;

    // file state
    int fd;
    uint32_t node_id;

    // descriptors
    Mutex desc_mutex;
    std::vector<BgWorkItemDescriptor> work_item_descs;
    std::vector<BgWorkSubItemDescriptor> sub_item_descs;
    uint16_t next_sub_item_id;

    // block pool and completed list
    Mutex block_mutex;
    ProfileBlock *free_blocks;
    ProfileBlock *completed_head;
    ProfileBlock *completed_tail;
    size_t buffered_bytes;
    uint32_t next_sequence;

    // registered states (for shutdown flushing)
    Mutex thread_mutex;
    std::vector<BgWorkProfileState *> thread_states;
  };

  // global instance
  extern BgWorkProfileManager bgwork_profiler;

  class BackgroundWorkManager {
  public:
    BackgroundWorkManager(void);
    ~BackgroundWorkManager(void);

    struct Config {
      unsigned generic_workers = 2; // non-numa-specific workers
      unsigned per_numa_workers = 0;
      bool pin_generic = false;
      bool pin_numa = false;
      size_t worker_stacksize_in_kb = 1024;
      long long worker_spin_interval = 0;
      long long work_item_timeslice = 100000;
    };

    void configure_from_cmdline(std::vector<std::string> &cmdline);

    void start_dedicated_workers(Realm::CoreReservationSet &crs);

    void stop_dedicated_workers(void);

    typedef unsigned long long BitMask;
    static const size_t MAX_WORK_ITEMS = 256;
    static const size_t BITMASK_BITS = 8 * sizeof(BitMask);
    static const size_t BITMASK_ARRAY_SIZE =
        (MAX_WORK_ITEMS + BITMASK_BITS - 1) / BITMASK_BITS;

    class Worker {
    public:
      Worker(void);
      ~Worker(void);

      void set_manager(BackgroundWorkManager *_manager);

      // configuration settings impact which work items this worker can handle
      void set_max_timeslice(long long _timeslice_in_ns);
      void set_numa_domain(int _numa_domain); // -1 == dont care

      bool do_work(long long max_time_in_ns, atomic<bool> *interrupt_flag);

    protected:
      BackgroundWorkManager *manager;
      unsigned starting_slot;
      BitMask known_work_item_mask[BITMASK_ARRAY_SIZE];
      BitMask allowed_work_item_mask[BITMASK_ARRAY_SIZE];
      long long max_timeslice;
      int numa_domain;
    };

  protected:
    friend class BackgroundWorkManager::Worker;
    friend class BackgroundWorkItem;

    unsigned assign_slot(BackgroundWorkItem *item);
    void release_slot(unsigned slot);
    void advertise_work(unsigned slot);

    Config cfg;

    // mutex protects assignment of work items to slots
    Mutex mutex;
    atomic<unsigned> num_work_items;
    atomic<BitMask> active_work_item_mask[BITMASK_ARRAY_SIZE];

    atomic<int> work_item_usecounts[MAX_WORK_ITEMS];
    BackgroundWorkItem *work_items[MAX_WORK_ITEMS];

    friend class BackgroundWorkThread;
    friend class BgWorkProfileManager;

    // to manage sleeping workers, we need to stuff three things into a
    //  single atomically-updatable state variable:
    // a) the number of active work items - no worker should sleep if there are
    //     any active work items, and any increment of the active work items
    //     should wake up one sleeping worker (unless there are none)
    //     (NOTE: this counter can temporarily underflow, so needs to be the top
    //       field in the variable to avoid temporarily corrupting other fields)
    // b) the number of sleeping workers
    // c) a bit indicating if a shutdown has been requested (which should wake
    //     up all remaining workers)
    static const uint32_t STATE_SHUTDOWN_BIT = 1;
    static const uint32_t STATE_ACTIVE_ITEMS_MASK = 0xFFFF;
    static const unsigned STATE_ACTIVE_ITEMS_SHIFT = 16;
    static const uint32_t STATE_SLEEPING_WORKERS_MASK = 0xFFF;
    static const unsigned STATE_SLEEPING_WORKERS_SHIFT = 4;
    atomic<uint32_t> worker_state;

    // sleeping workers go in a doorbell list with a delegating mutex
    DelegatingMutex db_mutex;
    DoorbellList db_list;

    std::vector<BackgroundWorkThread *> dedicated_workers;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE BackgroundWorkItem {
  public:
    BackgroundWorkItem(const std::string &_name);
    virtual ~BackgroundWorkItem(void);

    void add_to_manager(BackgroundWorkManager *_manager, int _numa_domain = -1,
                        long long _min_timeslice_needed = -1);

    // perform work, trying to respect the 'work_until' time limit - return
    //  true to request requeuing (this is more efficient than calling
    //  'make_active' at the end of 'do_work') or false if all work has been
    //  completed (or if 'make_active' has already been called)
    virtual bool do_work(TimeLimit work_until) = 0;

    // returns the slot index assigned by the background work manager
    unsigned get_slot() const { return index; }

  protected:
    friend class BackgroundWorkManager::Worker;
    friend class BgWorkProfileManager;

    // mark this work item as active (i.e. having work to do)
    void make_active(void);

    std::string name;
    BackgroundWorkManager *manager;
    int numa_domain;
    long long min_timeslice_needed;
    unsigned index;

#ifdef DEBUG_REALM
  public:
    // in debug mode, we'll track the state of a work item to avoid
    //  duplicate activations or activations after shutdown
    enum State
    {
      STATE_IDLE,
      STATE_ACTIVE,
      STATE_SHUTDOWN,
    };

    void make_inactive(void); // called immediately before 'do_work'
    void shutdown_work_item(void);

  protected:
    atomic<State> state;
#endif
  };

}; // namespace Realm

#include "realm/bgwork.inl"

#endif
