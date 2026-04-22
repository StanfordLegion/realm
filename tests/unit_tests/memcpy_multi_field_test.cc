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

#include "realm/bgwork.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/memcpy_channel.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include "test_common.h"
#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <numeric>

using namespace Realm;

namespace {

  struct MockHeap : public ReplicatedHeap {
    void *alloc_obj(std::size_t bytes, std::size_t align = 16) override
    {
      void *ptr = nullptr;
#ifdef REALM_ON_WINDOWS
      ptr = _aligned_malloc(bytes, align);
#else
      int ret = posix_memalign(&ptr, align, bytes);
      if(ret != 0)
        ptr = nullptr;
#endif
      assert(ptr != nullptr);
      return ptr;
    }

    void free_obj(void *ptr) override
    {
#ifdef REALM_ON_WINDOWS
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
  };

  // Build an AoS-style layout where all fields share one piece list and the
  // i-th field sits at offset i * field_size.  This matches what
  // InstanceLayoutGeneric produces when idindexed_fields is detected, which
  // is the shape that IDIndexedFieldsIterator expects.
  template <int N, typename T>
  static InstanceLayout<N, T> *make_aos_layout(Rect<N, T> bounds,
                                               const std::vector<FieldID> &field_ids,
                                               size_t field_size)
  {
    auto *inst_layout = new InstanceLayout<N, T>();
    inst_layout->bytes_used = 0;
    inst_layout->space = bounds;
    inst_layout->idindexed_fields = true;

    size_t stride = field_size;
    for(int d = 0; d < N; d++) {
      const size_t count = static_cast<size_t>(bounds.hi[d] - bounds.lo[d] + 1);
      // Fields are packed first, so the next-higher spatial dim stride is
      // (num_fields * field_size * prefix product).
      stride *= count;
    }
    const size_t total_per_field = stride;
    (void)total_per_field;

    inst_layout->piece_lists.resize(1);
    auto *affine = new AffineLayoutPiece<N, T>();
    affine->bounds = bounds;
    affine->offset = 0;
    size_t cur_stride = field_size;
    for(int d = 0; d < N; d++) {
      affine->strides[d] = cur_stride;
      affine->offset -= bounds.lo[d] * cur_stride;
      cur_stride *= static_cast<size_t>(bounds.hi[d] - bounds.lo[d] + 1);
    }
    inst_layout->piece_lists[0].pieces.push_back(affine);
    const size_t bytes_per_field = cur_stride;

    for(size_t i = 0; i < field_ids.size(); i++) {
      InstanceLayoutGeneric::FieldLayout fl;
      fl.list_idx = 0;
      fl.rel_offset = static_cast<size_t>(field_ids[i]) * bytes_per_field;
      fl.size_in_bytes = static_cast<int>(field_size);
      inst_layout->fields[field_ids[i]] = fl;
    }
    // Account for all N_fields fields' storage.  Pad bytes_used to cover
    // the highest-ID field in the block.
    size_t max_fid = 0;
    for(FieldID f : field_ids)
      max_fid = std::max(max_fid, static_cast<size_t>(f));
    inst_layout->bytes_used = (max_fid + 1) * bytes_per_field;
    return inst_layout;
  }

  template <int N, typename T>
  static RegionInstanceImpl *
  make_aos_inst(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
                size_t field_size, uintptr_t storage_base_offset = 0)
  {
    RegionInstance inst = ID::make_instance(0, 0, 0, 0).convert<RegionInstance>();
    auto *layout = make_aos_layout<N, T>(bounds, field_ids, field_size);
    auto *impl = new RegionInstanceImpl(nullptr, inst, nullptr);
    impl->metadata.layout = layout;
    impl->metadata.inst_offset = storage_base_offset;
    return impl;
  }

  template <typename T>
  static void fill_with_pattern(std::vector<T> &buf, T base)
  {
    for(size_t i = 0; i < buf.size(); i++) {
      buf[i] = base + static_cast<T>(i);
    }
  }

  // Drive one MemcpyXferDes through progress_xd until it stops making progress
  // and verify every field's byte pattern landed correctly at the right
  // destination offset.
  template <int N, typename T>
  static void
  run_multi_field_case(Rect<N, T> bounds, const std::vector<FieldID> &src_fids,
                       const std::vector<FieldID> &dst_fids, size_t field_size)
  {
    ASSERT_EQ(src_fids.size(), dst_fids.size());

    // allocate max-field-ID + 1 slots on each side so every id in the
    // copy lists is addressable.
    size_t src_max_fid = 0, dst_max_fid = 0;
    for(FieldID f : src_fids)
      src_max_fid = std::max(src_max_fid, static_cast<size_t>(f));
    for(FieldID f : dst_fids)
      dst_max_fid = std::max(dst_max_fid, static_cast<size_t>(f));

    const size_t volume = bounds.volume();
    const size_t bytes_per_field = volume * field_size;
    const size_t src_bytes = (src_max_fid + 1) * bytes_per_field;
    const size_t dst_bytes = (dst_max_fid + 1) * bytes_per_field;

    std::vector<uint8_t> src_storage(src_bytes, 0);
    std::vector<uint8_t> dst_storage(dst_bytes, 0xAB);

    // write a distinct byte pattern into each src field slot so we can
    // verify the right field landed at the right dst offset.
    for(FieldID f : src_fids) {
      const uint8_t base = 0x10u + static_cast<uint8_t>(f & 0xFFu);
      uint8_t *slot = src_storage.data() + static_cast<size_t>(f) * bytes_per_field;
      for(size_t i = 0; i < bytes_per_field; i++) {
        slot[i] = base + static_cast<uint8_t>(i);
      }
    }

    auto *src_inst = make_aos_inst<N, T>(bounds, src_fids, field_size);
    auto *dst_inst = make_aos_inst<N, T>(bounds, dst_fids, field_size);

    MockHeap heap;
    std::vector<int> dim_order(N);
    std::iota(dim_order.begin(), dim_order.end(), 0);

    auto *src_it = new IDIndexedFieldsIterator<N, T>(dim_order.data(), src_fids,
                                                     field_size, src_inst, bounds, &heap);
    auto *dst_it = new IDIndexedFieldsIterator<N, T>(dim_order.data(), dst_fids,
                                                     field_size, dst_inst, bounds, &heap);

    // -------- MemcpyXferDes wiring (mirrors memcpy_channel_test) --------
    Node node_data;
    std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
    XferDesRedopInfo redop_info;
    auto bgwork = std::make_unique<BackgroundWorkManager>();
    auto channel = std::make_unique<MemcpyChannel>(bgwork.get(), &node_data,
                                                   remote_shared_memory_mappings);

    std::vector<XferDesPortInfo> inputs_info, outputs_info;
    std::unique_ptr<MemcpyXferDes> xfer_desc(dynamic_cast<MemcpyXferDes *>(
        channel->create_xfer_des(0, 0 /*owner*/, 0 /*guid*/, inputs_info, outputs_info,
                                 0 /*priority*/, redop_info, nullptr, 0, 0)));

    auto src_mem =
        std::make_unique<LocalCPUMemory>(nullptr, Memory::NO_MEMORY, src_storage.size(),
                                         0, Memory::SYSTEM_MEM, src_storage.data());
    auto dst_mem =
        std::make_unique<LocalCPUMemory>(nullptr, Memory::NO_MEMORY, dst_storage.size(),
                                         0, Memory::SYSTEM_MEM, dst_storage.data());

    xfer_desc->input_ports.resize(1);
    auto &input_port = xfer_desc->input_ports[0];
    input_port.mem = src_mem.get();
    input_port.peer_port_idx = 0;
    input_port.iter = src_it;
    input_port.addrcursor.set_addrlist(&input_port.addrlist);

    xfer_desc->output_ports.resize(1);
    auto &output_port = xfer_desc->output_ports[0];
    output_port.mem = dst_mem.get();
    output_port.peer_port_idx = 0;
    output_port.iter = dst_it;
    output_port.addrcursor.set_addrlist(&output_port.addrlist);

    while(xfer_desc->progress_xd(channel.get(), TimeLimit::relative(10000000))) {
    }

    // check: every dst_fid slot contains the pattern we wrote into the
    // corresponding src_fid slot.  every other byte in dst remains 0xAB.
    std::vector<bool> dst_written(dst_storage.size(), false);
    for(size_t k = 0; k < src_fids.size(); k++) {
      const FieldID sf = src_fids[k];
      const FieldID df = dst_fids[k];
      const uint8_t base = 0x10u + static_cast<uint8_t>(sf & 0xFFu);
      const uint8_t *dst_slot =
          dst_storage.data() + static_cast<size_t>(df) * bytes_per_field;
      for(size_t i = 0; i < bytes_per_field; i++) {
        const uint8_t expected = base + static_cast<uint8_t>(i);
        EXPECT_EQ(dst_slot[i], expected)
            << " src_fid=" << sf << " dst_fid=" << df << " byte=" << i;
        dst_written[static_cast<size_t>(df) * bytes_per_field + i] = true;
      }
    }
    for(size_t i = 0; i < dst_storage.size(); i++) {
      if(!dst_written[i]) {
        EXPECT_EQ(dst_storage[i], 0xABu) << " byte=" << i << " unexpectedly modified";
      }
    }

    channel->shutdown();
  }

} // namespace

// ---- 1D cases ----

TEST(MemcpyMultiField, OneDim_TwoFields_Sequential)
{
  run_multi_field_case<1, int>(Rect<1, int>(0, 63),
                               /*src_fids=*/{0, 1},
                               /*dst_fids=*/{0, 1},
                               /*field_size=*/sizeof(int));
}

TEST(MemcpyMultiField, OneDim_FourFields_Sequential)
{
  run_multi_field_case<1, int>(Rect<1, int>(0, 255),
                               /*src_fids=*/{0, 1, 2, 3},
                               /*dst_fids=*/{0, 1, 2, 3},
                               /*field_size=*/sizeof(int));
}

TEST(MemcpyMultiField, OneDim_FieldReorder)
{
  // src reads fields 0..3, dst writes into fields 4..7 (reordered).
  // Confirms per-field offsets on src and dst can differ.
  run_multi_field_case<1, int>(Rect<1, int>(0, 127),
                               /*src_fids=*/{0, 1, 2, 3},
                               /*dst_fids=*/{4, 5, 6, 7},
                               /*field_size=*/sizeof(int));
}

TEST(MemcpyMultiField, OneDim_FieldPermutation)
{
  // src fields in different order than dst - validates that the per-field
  // offset pairing tracks src_fields[k] <-> dst_fields[k] rather than
  // assuming both arrays are sorted.
  run_multi_field_case<1, int>(Rect<1, int>(0, 63),
                               /*src_fids=*/{3, 1, 0, 2},
                               /*dst_fids=*/{0, 2, 3, 1},
                               /*field_size=*/sizeof(int));
}

TEST(MemcpyMultiField, OneDim_ManyFields)
{
  // 1024 fields exercises the multi-field iteration path at realistic
  // Legion workload scale; the 256-KiB per-field cap still keeps each
  // progress_xd iteration bounded.
  std::vector<FieldID> fids(1024);
  std::iota(fids.begin(), fids.end(), 0);
  run_multi_field_case<1, int>(Rect<1, int>(0, 15),
                               /*src_fids=*/fids,
                               /*dst_fids=*/fids,
                               /*field_size=*/sizeof(int));
}

// ---- 2D cases ----

TEST(MemcpyMultiField, TwoDim_FourFields)
{
  run_multi_field_case<2, int>(Rect<2, int>({0, 0}, {15, 15}),
                               /*src_fids=*/{0, 1, 2, 3},
                               /*dst_fids=*/{0, 1, 2, 3},
                               /*field_size=*/sizeof(int));
}

TEST(MemcpyMultiField, TwoDim_FieldReorder)
{
  run_multi_field_case<2, int>(Rect<2, int>({0, 0}, {31, 7}),
                               /*src_fids=*/{2, 0, 1},
                               /*dst_fids=*/{0, 1, 2},
                               /*field_size=*/sizeof(long long));
}

// ---- 3D cases ----

TEST(MemcpyMultiField, ThreeDim_ThreeFields)
{
  run_multi_field_case<3, int>(Rect<3, int>({0, 0, 0}, {7, 7, 7}),
                               /*src_fids=*/{0, 1, 2},
                               /*dst_fids=*/{0, 1, 2},
                               /*field_size=*/sizeof(int));
}

// ---- single-field sanity ----

TEST(MemcpyMultiField, OneDim_SingleField_LegacyPath)
{
  // fields.size() == 1 still exercises the FieldBlock code path (the
  // iterator attaches a FieldBlock unconditionally) but n collapses to 1
  // throughout.  Confirms the fast-path code does the right thing when
  // there's nothing to batch.
  run_multi_field_case<1, int>(Rect<1, int>(0, 63),
                               /*src_fids=*/{5},
                               /*dst_fids=*/{7},
                               /*field_size=*/sizeof(int));
}
