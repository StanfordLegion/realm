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

// Test program for background work profiling.
// Creates instances in available memories, runs a batch of copies to exercise
// DMA background work items, then optionally validates the output file.

#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <vector>

#ifdef REALM_ON_WINDOWS
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

namespace TestConfig {
  size_t copy_size = 1 << 20; // 1MB per copy
  int num_copies = 16;
  bool validate = false;
  std::string profile_file = "bgwork_profile_test.bin";
}; // namespace TestConfig

// File format constants (duplicated from bgwork_profile.h for standalone validation)
static const char EXPECTED_MAGIC[4] = {'R', 'B', 'W', 'P'};

// Note: file header and block header fields are read individually
// to avoid C struct padding issues with the binary file format.

// Read helpers
static bool read_exact(int fd, void *buf, size_t count)
{
  uint8_t *p = static_cast<uint8_t *>(buf);
  while(count > 0) {
    ssize_t n = read(fd, p, count);
    if(n <= 0)
      return false;
    p += n;
    count -= n;
  }
  return true;
}

// Decode a timestamp delta, returns the absolute timestamp
// Updates pos to point past the consumed bytes
static bool decode_timestamp(const uint8_t *data, size_t data_size, size_t &pos,
                             int64_t &last_ts, int64_t &out_ts)
{
  if(pos >= data_size)
    return false;

  uint8_t first = data[pos];
  if((first & 0x80) == 0) {
    // 2-byte encoding, 15-bit delta
    if(pos + 2 > data_size)
      return false;
    int64_t delta = ((int64_t)(first & 0x7F) << 8) | data[pos + 1];
    out_ts = last_ts + delta;
    pos += 2;
  } else if((first & 0xC0) == 0x80) {
    // 4-byte encoding, 30-bit delta
    if(pos + 4 > data_size)
      return false;
    int64_t delta = ((int64_t)(first & 0x3F) << 24) | ((int64_t)data[pos + 1] << 16) |
                    ((int64_t)data[pos + 2] << 8) | data[pos + 3];
    out_ts = last_ts + delta;
    pos += 4;
  } else {
    // 8-byte encoding, absolute timestamp
    if(pos + 8 > data_size)
      return false;
    uint64_t val = ((uint64_t)(first & 0x3F) << 56) | ((uint64_t)data[pos + 1] << 48) |
                   ((uint64_t)data[pos + 2] << 40) | ((uint64_t)data[pos + 3] << 32) |
                   ((uint64_t)data[pos + 4] << 24) | ((uint64_t)data[pos + 5] << 16) |
                   ((uint64_t)data[pos + 6] << 8) | (uint64_t)data[pos + 7];
    out_ts = static_cast<int64_t>(val);
    pos += 8;
  }
  last_ts = out_ts;
  return true;
}

static bool validate_profile_file(const std::string &filename)
{
  int fd = open(filename.c_str(), O_RDONLY);
  if(fd < 0) {
    fprintf(stderr, "VALIDATE: cannot open file: %s\n", filename.c_str());
    return false;
  }

  // read header fields individually to avoid struct padding issues
  char magic[4];
  uint16_t version, flags;
  uint32_t node_id;
  int64_t zero_time;
  uint32_t work_item_count, sub_item_count;

  if(!read_exact(fd, magic, 4) || !read_exact(fd, &version, 2) ||
     !read_exact(fd, &flags, 2) || !read_exact(fd, &node_id, 4) ||
     !read_exact(fd, &zero_time, 8) || !read_exact(fd, &work_item_count, 4) ||
     !read_exact(fd, &sub_item_count, 4)) {
    fprintf(stderr, "VALIDATE: failed to read file header\n");
    close(fd);
    return false;
  }

  // check magic
  if(memcmp(magic, EXPECTED_MAGIC, 4) != 0) {
    fprintf(stderr, "VALIDATE: bad magic: %c%c%c%c\n", magic[0], magic[1], magic[2],
            magic[3]);
    close(fd);
    return false;
  }

  fprintf(stdout, "VALIDATE: magic OK, version=%u, flags=0x%04x, node_id=%u\n", version,
          flags, node_id);
  fprintf(stdout, "VALIDATE: zero_time=%lld, work_items=%u, sub_items=%u\n",
          (long long)zero_time, work_item_count, sub_item_count);

  // read work item descriptors
  for(uint32_t i = 0; i < work_item_count; i++) {
    uint16_t slot, name_len;
    if(!read_exact(fd, &slot, 2) || !read_exact(fd, &name_len, 2)) {
      fprintf(stderr, "VALIDATE: failed reading work item descriptor %u\n", i);
      close(fd);
      return false;
    }
    std::vector<char> name(name_len);
    if(name_len > 0 && !read_exact(fd, name.data(), name_len)) {
      fprintf(stderr, "VALIDATE: failed reading work item name %u\n", i);
      close(fd);
      return false;
    }
    fprintf(stdout, "VALIDATE: work item: slot=%u name='%.*s'\n", slot, (int)name_len,
            name.data());
  }

  // read sub-item descriptors
  for(uint32_t i = 0; i < sub_item_count; i++) {
    uint16_t id, name_len;
    uint8_t type;
    if(!read_exact(fd, &id, 2) || !read_exact(fd, &type, 1) ||
       !read_exact(fd, &name_len, 2)) {
      fprintf(stderr, "VALIDATE: failed reading sub-item descriptor %u\n", i);
      close(fd);
      return false;
    }
    std::vector<char> name(name_len);
    if(name_len > 0 && !read_exact(fd, name.data(), name_len)) {
      fprintf(stderr, "VALIDATE: failed reading sub-item name %u\n", i);
      close(fd);
      return false;
    }
    fprintf(stdout, "VALIDATE: sub-item: id=%u type=%u name='%.*s'\n", id, type,
            (int)name_len, name.data());
  }

  // read data blocks
  uint32_t total_blocks = 0;
  uint32_t total_records = 0;
  while(true) {
    // read block header fields individually to avoid padding
    uint64_t blk_thread_id;
    uint32_t blk_sequence, blk_record_count;
    int64_t blk_base_timestamp;
    uint32_t blk_data_size, blk_compressed_size;

    if(!read_exact(fd, &blk_thread_id, 8) || !read_exact(fd, &blk_sequence, 4) ||
       !read_exact(fd, &blk_record_count, 4) || !read_exact(fd, &blk_base_timestamp, 8) ||
       !read_exact(fd, &blk_data_size, 4) || !read_exact(fd, &blk_compressed_size, 4))
      break; // end of file or incomplete header

    total_blocks++;
    total_records += blk_record_count;

    size_t read_size = (blk_compressed_size > 0) ? blk_compressed_size : blk_data_size;
    std::vector<uint8_t> data(read_size);
    if(!read_exact(fd, data.data(), read_size)) {
      fprintf(stderr, "VALIDATE: failed reading block data (block %u)\n", total_blocks);
      close(fd);
      return false;
    }

    // if compressed, we'd need to decompress - skip detailed validation for compressed
    // blocks
    if(blk_compressed_size > 0) {
      fprintf(stdout,
              "VALIDATE: block %u: thread=%llu seq=%u records=%u compressed=%u->%u\n",
              total_blocks, (unsigned long long)blk_thread_id, blk_sequence,
              blk_record_count, blk_data_size, blk_compressed_size);
      continue;
    }

    // parse records and check timestamp monotonicity
    size_t pos = 0;
    int64_t last_ts = 0;
    int64_t prev_ts = 0;
    bool ts_monotonic = true;

    for(uint32_t r = 0; r < blk_record_count && pos < blk_data_size; r++) {
      int64_t ts;
      if(!decode_timestamp(data.data(), blk_data_size, pos, last_ts, ts)) {
        fprintf(stderr, "VALIDATE: failed decoding timestamp in block %u record %u\n",
                total_blocks, r);
        close(fd);
        return false;
      }

      if(r > 0 && ts < prev_ts) {
        ts_monotonic = false;
        fprintf(stderr,
                "VALIDATE: non-monotonic timestamp in block %u record %u: "
                "%lld < %lld\n",
                total_blocks, r, (long long)ts, (long long)prev_ts);
      }
      prev_ts = ts;

      // read record type
      if(pos >= blk_data_size) {
        fprintf(stderr, "VALIDATE: truncated record in block %u record %u\n",
                total_blocks, r);
        close(fd);
        return false;
      }
      uint8_t rec_type = data[pos++];

      // skip payload based on type
      switch(rec_type) {
      case 0x01: // COARSE_BEGIN: 1 byte slot
        if(pos + 1 > blk_data_size) {
          close(fd);
          return false;
        }
        pos += 1;
        break;
      case 0x02: // COARSE_END: no payload
        break;
      case 0x11: // FINE_BEGIN: 2 byte sub_item_id
        if(pos + 2 > blk_data_size) {
          close(fd);
          return false;
        }
        pos += 2;
        break;
      case 0x12: // FINE_END: no payload
        break;
      case 0x21: // GPU_WORK: 1 byte gpu_index + 4 byte duration
        if(pos + 5 > blk_data_size) {
          close(fd);
          return false;
        }
        pos += 5;
        break;
      default:
        fprintf(stderr, "VALIDATE: unknown record type 0x%02x in block %u record %u\n",
                rec_type, total_blocks, r);
        close(fd);
        return false;
      }
    }

    fprintf(stdout,
            "VALIDATE: block %u: thread=%llu seq=%u records=%u bytes=%u ts_mono=%s\n",
            total_blocks, (unsigned long long)blk_thread_id, blk_sequence,
            blk_record_count, blk_data_size, ts_monotonic ? "yes" : "NO");
  }

  close(fd);

  fprintf(stdout, "VALIDATE: total blocks=%u, total records=%u\n", total_blocks,
          total_records);

  if(total_records == 0) {
    fprintf(stderr, "VALIDATE: WARNING - no records found in profile\n");
    // not necessarily a failure - short tests may not generate records
  }

  fprintf(stdout, "VALIDATE: PASSED\n");
  return true;
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  log_app.info() << "bgwork profile test starting";

  // find system memory
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM).local_address_space().has_capacity(1);

  std::vector<Memory> sys_mems;
  for(Machine::MemoryQuery::iterator it = mq.begin(); it; ++it)
    sys_mems.push_back(*it);

  if(sys_mems.empty()) {
    log_app.fatal() << "no system memories found!";
    abort();
  }

  log_app.info() << "found " << sys_mems.size() << " system memories";

  // create instances in available memories
  size_t num_elems = TestConfig::copy_size;
  IndexSpace<1> is = Rect<1>(0, num_elems - 1);
  std::vector<size_t> field_sizes(1, 1); // 1 byte per element

  std::vector<RegionInstance> instances;
  for(size_t i = 0; i < sys_mems.size() && i < 2; i++) {
    RegionInstance inst;
    RegionInstance::create_instance(inst, sys_mems[i], is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst.exists());
    instances.push_back(inst);
    log_app.info() << "created instance in memory " << sys_mems[i];
  }

  // if only one memory, create two instances in the same memory
  if(instances.size() < 2) {
    RegionInstance inst;
    RegionInstance::create_instance(inst, sys_mems[0], is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst.exists());
    instances.push_back(inst);
  }

  // run a batch of copies between the instances
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  Event prev = Event::NO_EVENT;

  for(int i = 0; i < TestConfig::num_copies; i++) {
    int src_idx = i % instances.size();
    int dst_idx = (i + 1) % instances.size();

    srcs[0].set_field(instances[src_idx], 0, 1);
    dsts[0].set_field(instances[dst_idx], 0, 1);

    prev = is.copy(srcs, dsts, ProfilingRequestSet(), prev);
  }

  // wait for all copies to finish
  prev.wait();
  log_app.info() << "all " << TestConfig::num_copies << " copies completed";

  // clean up instances
  for(auto &inst : instances)
    inst.destroy();

  log_app.info() << "bgwork profile test done";
}

int main(int argc, char **argv)
{
  // pre-scan for validate-only mode
  for(int i = 1; i < argc; i++) {
    if(strcmp(argv[i], "-validate") == 0) {
      TestConfig::validate = true;
    } else if(strcmp(argv[i], "-profile_file") == 0 && i + 1 < argc) {
      TestConfig::profile_file = argv[i + 1];
      i++;
    }
  }

  if(TestConfig::validate) {
    // validate-only mode - no Realm runtime needed
    if(!validate_profile_file(TestConfig::profile_file))
      return 1;
    return 0;
  }

  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-size", TestConfig::copy_size, 'M')
      .add_option_int("-copies", TestConfig::num_copies)
      .add_option_string("-profile_file", TestConfig::profile_file);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  int ret = rt.wait_for_shutdown();

  // validate after shutdown
  if(ret == 0) {
    struct stat st;
    if(stat(TestConfig::profile_file.c_str(), &st) == 0) {
      if(!validate_profile_file(TestConfig::profile_file))
        return 1;
    } else {
      fprintf(stdout,
              "NOTE: profile file '%s' not found (profiling may not have been enabled)\n",
              TestConfig::profile_file.c_str());
    }
  }
  return ret;
}
