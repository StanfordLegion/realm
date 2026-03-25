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

// inline recording methods for BgWorkProfileState

#ifndef REALM_BGWORK_INL
#define REALM_BGWORK_INL

namespace Realm {

  // Timestamp delta encoding:
  //  15-bit: 2 bytes, MSB=0
  //  30-bit: 4 bytes, MSB=10
  //  64-bit: 8 bytes, MSB=11 (stores absolute timestamp)
  inline size_t BgWorkProfileState::encode_timestamp(uint8_t *buf, int64_t delta,
                                                      int64_t absolute)
  {
    if(delta >= 0 && delta < (1 << 15)) {
      uint16_t val = static_cast<uint16_t>(delta);
      buf[0] = (val >> 8) & 0x7F;
      buf[1] = val & 0xFF;
      return 2;
    } else if(delta >= 0 && delta < (1LL << 30)) {
      uint32_t val = static_cast<uint32_t>(delta) | 0x80000000U;
      buf[0] = (val >> 24) & 0xFF;
      buf[1] = (val >> 16) & 0xFF;
      buf[2] = (val >> 8) & 0xFF;
      buf[3] = val & 0xFF;
      return 4;
    } else {
      // 8-byte encoding: store absolute timestamp
      uint64_t val = static_cast<uint64_t>(absolute);
      buf[0] = 0xC0 | ((val >> 56) & 0x3F);
      buf[1] = (val >> 48) & 0xFF;
      buf[2] = (val >> 40) & 0xFF;
      buf[3] = (val >> 32) & 0xFF;
      buf[4] = (val >> 24) & 0xFF;
      buf[5] = (val >> 16) & 0xFF;
      buf[6] = (val >> 8) & 0xFF;
      buf[7] = val & 0xFF;
      return 8;
    }
  }

  // ensures enough space in the current block, rotating if needed
  // returns pointer to write position, or nullptr on failure
  inline uint8_t *BgWorkProfileState::ensure_space(size_t needed)
  {
    ProfileBlock *block = current_block;
    if(block && (block->used + needed <= ProfileBlock::BLOCK_SIZE))
      return block->data + block->used;

    // need a new block - complete old one and get fresh
    if(block)
      bgwork_profiler.complete_block(block);

    block = bgwork_profiler.alloc_block(thread_id);
    current_block = block;
    if(!block)
      return nullptr;

    // reset delta encoding for new block
    last_timestamp = 0;
    return block->data;
  }

  inline void BgWorkProfileState::begin(uint8_t slot)
  {
    if(level == 0)
      return;
    REALM_ASSERT(!did_work);
    did_work = true;

    int64_t now = Clock::current_time_in_nanoseconds(true /*absolute*/);

    // max record size: 8 (timestamp) + 1 (type) + 1 (slot) = 10
    uint8_t *buf = ensure_space(10);
    if(!buf)
      return;

    ProfileBlock *block = current_block;
    if(block->num_records == 0)
      block->base_timestamp = now;

    // save state for potential discard
    begin_block_used = block->used;
    begin_block_num_records = block->num_records;
    begin_last_timestamp = last_timestamp;

    int64_t delta = now - last_timestamp;
    size_t ts_size = encode_timestamp(buf, delta, now);
    buf += ts_size;

    *buf++ = BGWP_COARSE_BEGIN;
    *buf++ = slot;

    block->used += ts_size + 2;
    block->num_records++;
    last_timestamp = now;
  }

  inline void BgWorkProfileState::end(const TimeLimit& time_limit)
  {
    if(level == 0)
      return;

    if(did_work) {
      did_work = false;
    } else if(!time_limit.is_expired()) {
      // If the timelimit expired we still record this
      // because it is surprising that it took that long
      discard();
      return;
    }

    int64_t now = Clock::current_time_in_nanoseconds(true /*absolute*/);

    // max: 8 (timestamp) + 1 (type) = 9
    uint8_t *buf = ensure_space(9);
    if(!buf)
      return;

    ProfileBlock *block = current_block;
    if(block->num_records == 0)
      block->base_timestamp = now;

    int64_t delta = now - last_timestamp;
    size_t ts_size = encode_timestamp(buf, delta, now);
    buf += ts_size;

    *buf++ = BGWP_COARSE_END;

    block->used += ts_size + 1;
    block->num_records++;
    last_timestamp = now;
  }

  inline void BgWorkProfileState::discard(void)
  {
    // rewind the block state to what it was before begin()
    ProfileBlock *block = current_block;
    if(!block)
      return;

    block->used = begin_block_used;
    block->num_records = begin_block_num_records;
    last_timestamp = begin_last_timestamp;
  }

  inline void BgWorkProfileState::fine_begin(uint16_t sub_item_id)
  {
    if(level < 2)
      return;

    did_work = true;

    int64_t now = Clock::current_time_in_nanoseconds(true /*absolute*/);

    // max: 8 + 1 + 2 = 11
    uint8_t *buf = ensure_space(11);
    if(!buf)
      return;

    ProfileBlock *block = current_block;
    if(block->num_records == 0)
      block->base_timestamp = now;

    int64_t delta = now - last_timestamp;
    size_t ts_size = encode_timestamp(buf, delta, now);
    buf += ts_size;

    *buf++ = BGWP_FINE_BEGIN;
    memcpy(buf, &sub_item_id, sizeof(uint16_t));
    buf += sizeof(uint16_t);

    block->used += ts_size + 3;
    block->num_records++;
    last_timestamp = now;
  }

  inline void BgWorkProfileState::fine_end()
  {
    if(level < 2)
      return;

    int64_t now = Clock::current_time_in_nanoseconds(true /*absolute*/);

    // max: 8 + 1 = 9
    uint8_t *buf = ensure_space(9);
    if(!buf)
      return;

    ProfileBlock *block = current_block;
    if(block->num_records == 0)
      block->base_timestamp = now;

    int64_t delta = now - last_timestamp;
    size_t ts_size = encode_timestamp(buf, delta, now);
    buf += ts_size;

    *buf++ = BGWP_FINE_END;

    block->used += ts_size + 1;
    block->num_records++;
    last_timestamp = now;
  }

  inline void BgWorkProfileState::gpu_work(uint64_t proc_id, uint8_t slot,
                                            int64_t start_time, int64_t stop_time)
  {
    if(level == 0)
      return;

    int64_t now = Clock::current_time_in_nanoseconds(true /*absolute*/);

    // max: 8 (timestamp) + 1 (type) + 8 (proc_id) + 1 (slot) + 8 (start) + 8 (stop) = 34
    uint8_t *buf = ensure_space(34);
    if(!buf)
      return;

    ProfileBlock *block = current_block;
    if(block->num_records == 0)
      block->base_timestamp = now;

    int64_t delta = now - last_timestamp;
    size_t ts_size = encode_timestamp(buf, delta, now);
    buf += ts_size;

    *buf++ = BGWP_GPU_WORK;
    memcpy(buf, &proc_id, sizeof(uint64_t));
    buf += sizeof(uint64_t);
    *buf++ = slot;
    memcpy(buf, &start_time, sizeof(int64_t));
    buf += sizeof(int64_t);
    memcpy(buf, &stop_time, sizeof(int64_t));
    buf += sizeof(int64_t);

    block->used += ts_size + 26;
    block->num_records++;
    last_timestamp = now;
  }

}; // namespace Realm

#endif // REALM_BGWORK_INL
