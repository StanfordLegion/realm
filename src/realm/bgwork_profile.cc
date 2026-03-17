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

// Background work profiling manager implementation

#include "realm/bgwork_profile.h"
#include "realm/timers.h"
#include "realm/logging.h"
#include "realm/network.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef REALM_ON_WINDOWS
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef REALM_BGWORK_PROFILE_USE_ZLIB
#include <zlib.h>
#endif

namespace Realm {

  Logger log_bgwork_profile("bgwork_profile");

  thread_local BgWorkProfileState *tl_bgwork_profile = nullptr;

  BgWorkProfileManager bgwork_profiler;

  BgWorkProfileManager::BgWorkProfileManager()
    : profile_level(0)
    , initialized(false)
    , max_buffer_bytes(1ULL << 30) // 1GB default
    , fd(-1)
    , node_id(0)
    , next_sub_item_id(0)
    , free_blocks(nullptr)
    , completed_head(nullptr)
    , completed_tail(nullptr)
    , buffered_bytes(0)
    , next_sequence(0)
  {}

  BgWorkProfileManager::~BgWorkProfileManager()
  {
    // free any remaining blocks in the free list
    while(free_blocks) {
      ProfileBlock *next = free_blocks->next;
      delete free_blocks;
      free_blocks = next;
    }
  }

  void BgWorkProfileManager::set_level(int level) { profile_level = level; }

  void BgWorkProfileManager::set_logfile(const std::string &filename)
  {
    logfile_pattern = filename;
  }

  void BgWorkProfileManager::set_bufsize(size_t megabytes)
  {
    max_buffer_bytes = (megabytes == 0) ? SIZE_MAX : megabytes * (1ULL << 20);
  }

  int BgWorkProfileManager::get_level() const { return profile_level; }

  void BgWorkProfileManager::initialize(uint32_t _node_id)
  {
    if(profile_level == 0)
      return;

    node_id = _node_id;

    // determine output filename
    std::string filename = logfile_pattern;
    if(filename.empty())
      filename = "bgwork_profile_%.bin";

    // replace % with node ID
    size_t pct = filename.find('%');
    if(pct != std::string::npos) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%u", node_id);
      filename.replace(pct, 1, buf);
    } else if(Network::max_node_id > 0) {
      log_bgwork_profile.fatal()
          << "multi-node run requires '%' in bgwork profile filename: " << filename;
      abort();
    }

    // open output file
#ifdef REALM_ON_WINDOWS
    fd = _open(filename.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, 0644);
#else
    fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
    if(fd < 0) {
      log_bgwork_profile.fatal() << "failed to open bgwork profile file: " << filename;
      abort();
    }

    log_bgwork_profile.info() << "bgwork profiling enabled: level=" << profile_level
                              << " file=" << filename
                              << " bufsize=" << (max_buffer_bytes >> 20) << "MB";

    // Write header with placeholder counts/offset.  Data blocks are appended
    // starting at offset HEADER_SIZE.  Descriptor tables and final header
    // patch happen at shutdown.
    write_file_header();

    initialized = true;
  }

  void BgWorkProfileManager::shutdown()
  {
    if(!initialized)
      return;

    // flush all thread-local blocks
    {
      AutoLock<> al(thread_mutex);
      for(BgWorkProfileState *state : thread_states) {
        if(state->current_block) {
          if(state->current_block->num_records > 0) {
            AutoLock<> bl(block_mutex);
            ProfileBlock *block = state->current_block;
            if(completed_tail) {
              completed_tail->next = block;
            } else {
              completed_head = block;
            }
            completed_tail = block;
            block->next = nullptr;
            buffered_bytes += block->used;
          } else {
            // empty block goes back to free list
            AutoLock<> bl(block_mutex);
            state->current_block->next = free_blocks;
            free_blocks = state->current_block;
          }
          state->current_block = nullptr;
        }
        // disable profiling for this thread
        // (the thread_local pointer was set to state, but we can't clear
        //  other threads' TLS - they should have stopped by now)
        delete state;
      }
      thread_states.clear();
    }

    // Flush all remaining in-memory data blocks to disk
    flush_all_blocks();

    // Record current file position — this is where descriptor tables start
    uint64_t desc_offset = lseek(fd, 0, SEEK_CUR);

    // Write descriptor tables (now complete) at end of file
    write_descriptor_tables();

    // Patch header with final counts and descriptor offset
    {
      AutoLock<> al(desc_mutex);
      uint32_t work_count = static_cast<uint32_t>(work_item_descs.size());
      uint32_t sub_count = static_cast<uint32_t>(sub_item_descs.size());
      lseek(fd, 20, SEEK_SET);
      write(fd, &work_count, sizeof(work_count));
      write(fd, &sub_count, sizeof(sub_count));
      write(fd, &desc_offset, sizeof(desc_offset));
      lseek(fd, 0, SEEK_END);
    }

    // close file
    if(fd >= 0) {
#ifdef REALM_ON_WINDOWS
      _close(fd);
#else
      close(fd);
#endif
      fd = -1;
    }

    log_bgwork_profile.info() << "bgwork profiling shutdown complete";
    initialized = false;
  }

  void BgWorkProfileManager::register_work_item(uint16_t slot, const std::string &name)
  {
    AutoLock<> al(desc_mutex);
    // check for duplicate
    for(const auto &d : work_item_descs) {
      if(d.slot == slot)
        return;
    }
    work_item_descs.push_back({slot, name});
    log_bgwork_profile.debug() << "registered work item: slot=" << slot
                               << " name=" << name;
  }

  uint16_t BgWorkProfileManager::register_sub_item(uint8_t type, const std::string &name)
  {
    AutoLock<> al(desc_mutex);
    uint16_t id = next_sub_item_id++;
    sub_item_descs.push_back({id, type, name});
    log_bgwork_profile.debug() << "registered sub-item: id=" << id
                               << " type=" << (int)type << " name=" << name;
    return id;
  }

  ProfileBlock *BgWorkProfileManager::alloc_block(uint64_t thread_id)
  {
    AutoLock<> al(block_mutex);

    ProfileBlock *block;
    if(free_blocks) {
      block = free_blocks;
      free_blocks = block->next;
    } else {
      block = new ProfileBlock;
    }

    block->used = 0;
    block->base_timestamp = 0;
    block->num_records = 0;
    block->thread_id = thread_id;
    block->sequence = next_sequence++;
    block->next = nullptr;

    return block;
  }

  void BgWorkProfileManager::complete_block(ProfileBlock *block)
  {
    bool need_flush = false;

    {
      AutoLock<> al(block_mutex);

      if(completed_tail) {
        completed_tail->next = block;
      } else {
        completed_head = block;
      }
      completed_tail = block;
      block->next = nullptr;
      buffered_bytes += block->used;

      need_flush = (buffered_bytes >= max_buffer_bytes);
    }

    // Flush half the buffer to keep memory bounded while retaining some
    // buffering to reduce write syscall frequency
    if(need_flush)
      flush_blocks_to_disk(max_buffer_bytes / 2);
  }

  void BgWorkProfileManager::register_thread_state(BgWorkProfileState *state)
  {
    AutoLock<> al(thread_mutex);
    thread_states.push_back(state);
  }

  void BgWorkProfileManager::write_file_header()
  {
    // header: magic(4) + version(2) + flags(2) + node_id(4) + zero_time(8) +
    //         work_item_count(4) + sub_item_count(4) + desc_offset(8) = 36 bytes
    uint8_t header[HEADER_SIZE];
    uint8_t *p = header;

    memcpy(p, BGWP_MAGIC, 4);
    p += 4;

    uint16_t version = BGWP_VERSION;
    memcpy(p, &version, 2);
    p += 2;

    uint16_t flags = 0;
    if(profile_level >= 2)
      flags |= BGWP_FLAG_HAS_FINE;
    memcpy(p, &flags, 2);
    p += 2;

    memcpy(p, &node_id, 4);
    p += 4;

    int64_t zero_time = Clock::get_zero_time();
    memcpy(p, &zero_time, 8);
    p += 8;

    // descriptor counts and offset will be patched at shutdown
    uint32_t zero32 = 0;
    uint64_t zero64 = 0;
    memcpy(p, &zero32, 4);
    p += 4; // work item count
    memcpy(p, &zero32, 4);
    p += 4; // sub item count
    memcpy(p, &zero64, 8);
    p += 8; // descriptor table offset

    ssize_t written = write(fd, header, sizeof(header));
    (void)written;
  }

  void BgWorkProfileManager::write_descriptor_tables()
  {
    AutoLock<> al(desc_mutex);

    // write work item descriptors
    for(const auto &d : work_item_descs) {
      uint16_t slot = d.slot;
      uint16_t name_len = static_cast<uint16_t>(d.name.size());
      write(fd, &slot, sizeof(slot));
      write(fd, &name_len, sizeof(name_len));
      write(fd, d.name.data(), name_len);
    }

    // write sub-item descriptors
    for(const auto &d : sub_item_descs) {
      uint16_t id = d.id;
      uint8_t type = d.type;
      uint16_t name_len = static_cast<uint16_t>(d.name.size());
      write(fd, &id, sizeof(id));
      write(fd, &type, sizeof(type));
      write(fd, &name_len, sizeof(name_len));
      write(fd, d.name.data(), name_len);
    }
  }

  void BgWorkProfileManager::flush_blocks_to_disk(size_t target_size)
  {
    while(true) {
      ProfileBlock *block = nullptr;
      {
        AutoLock<> al(block_mutex);
        if(!completed_head || buffered_bytes <= target_size)
          return;
        block = completed_head;
        completed_head = block->next;
        if(!completed_head)
          completed_tail = nullptr;
        buffered_bytes -= block->used;
      }

      // write block header fields individually to avoid padding
      uint64_t bh_thread_id = block->thread_id;
      uint32_t bh_sequence = block->sequence;
      uint32_t bh_record_count = block->num_records;
      int64_t bh_base_timestamp = block->base_timestamp;
      uint32_t bh_data_size = block->used;
      uint32_t bh_compressed_size = 0;

#ifdef REALM_BGWORK_PROFILE_USE_ZLIB
      // try to compress the block
      uLongf compressed_bound = compressBound(block->used);
      std::vector<uint8_t> compressed(compressed_bound);
      int zret = compress2(compressed.data(), &compressed_bound, block->data, block->used,
                           Z_DEFAULT_COMPRESSION);
      if(zret == Z_OK && compressed_bound < block->used) {
        bh_compressed_size = static_cast<uint32_t>(compressed_bound);
      }
#endif

      write(fd, &bh_thread_id, 8);
      write(fd, &bh_sequence, 4);
      write(fd, &bh_record_count, 4);
      write(fd, &bh_base_timestamp, 8);
      write(fd, &bh_data_size, 4);
      write(fd, &bh_compressed_size, 4);

#ifdef REALM_BGWORK_PROFILE_USE_ZLIB
      if(bh_compressed_size > 0)
        write(fd, compressed.data(), bh_compressed_size);
      else
        write(fd, block->data, block->used);
#else
      write(fd, block->data, block->used);
#endif

      // return block to free list
      {
        AutoLock<> al(block_mutex);
        block->next = free_blocks;
        free_blocks = block;
      }
    }
  }

  void BgWorkProfileManager::flush_all_blocks() { flush_blocks_to_disk(0); }

  void bgwork_profile_thread_init()
  {
    if(bgwork_profiler.get_level() == 0)
      return;
    if(tl_bgwork_profile)
      return; // already initialized

    BgWorkProfileState *state = new BgWorkProfileState;
    state->current_block = nullptr;
    state->last_timestamp = 0;
    // use hash of std::thread::id as our thread identifier
    state->thread_id =
        static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    tl_bgwork_profile = state;
    bgwork_profiler.register_thread_state(state);
  }

  void bgwork_profile_thread_fini()
  {
    BgWorkProfileState *state = tl_bgwork_profile;
    if(!state)
      return;

    // flush current block if it has data
    if(state->current_block && state->current_block->num_records > 0) {
      bgwork_profiler.complete_block(state->current_block);
      state->current_block = nullptr;
    }

    tl_bgwork_profile = nullptr;
    // note: state is not freed here - the manager owns the pointer list
    // and will clean up at shutdown
  }

}; // namespace Realm
