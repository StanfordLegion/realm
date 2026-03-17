/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef REALM_BGWORK_PROFILE_H
#define REALM_BGWORK_PROFILE_H

#include "realm/realm_config.h"
#include "realm/mutex.h"
#include "realm/atomics.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace Realm {

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

  struct BgWorkProfileState {
    ProfileBlock *current_block;
    int64_t last_timestamp; // for delta encoding
    uint64_t thread_id;
  };

  // thread-local pointer: null when profiling is disabled
  extern thread_local BgWorkProfileState *tl_bgwork_profile;

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

    // block management (called by recording functions)
    ProfileBlock *alloc_block(uint64_t thread_id);
    void complete_block(ProfileBlock *block);

    // thread state management
    void register_thread_state(BgWorkProfileState *state);

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

    // thread states (for shutdown flushing)
    Mutex thread_mutex;
    std::vector<BgWorkProfileState *> thread_states;
  };

  // global instance
  extern BgWorkProfileManager bgwork_profiler;

  // call at thread entry to set up thread-local profiling state
  // safe to call when profiling is disabled (does nothing)
  void bgwork_profile_thread_init();

  // call at thread exit to flush thread-local block
  void bgwork_profile_thread_fini();

}; // namespace Realm

#include "realm/bgwork_profile.inl"

#endif // REALM_BGWORK_PROFILE_H
