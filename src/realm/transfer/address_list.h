/*
 * Copyright 2025 Los Alamos National Laboratory, Stanford University, NVIDIA Corporation
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

#ifndef ADDRESS_LIST
#define ADDRESS_LIST

#include "realm/realm_config.h"
#include "realm/indexspace.h"
#include "realm/id.h"

namespace Realm {

  class AddressList {
  public:
    AddressList();

    size_t *begin_nd_entry(int max_dim, size_t payload_size = 0);
    void commit_nd_entry(int act_dim, size_t bytes, size_t payload_size = 0);

    size_t bytes_pending() const;

  protected:
    friend class AddressListCursor;

    const size_t *read_entry();

    size_t total_bytes;
    unsigned write_pointer;
    unsigned read_pointer;
    constexpr static const size_t MAX_ENTRIES = 1000;
    constexpr static size_t FLAG_HAS_EXTRA = (1UL << 63);
    std::vector<size_t> data;
    // size_t data[MAX_ENTRIES];
  };

  struct ParsedAddrlistEntry {
    int dim;
    size_t contig_bytes;
    uintptr_t base_offset;
    std::vector<std::pair<size_t, size_t>> count_strides;
    const size_t *payload = nullptr;
    size_t payload_count = 0;
  };

  class AddressListCursor {
  public:
    AddressListCursor();

    void set_addrlist(AddressList *_addrlist);

    int get_dim() const;
    uintptr_t get_offset() const;
    uintptr_t get_stride(int dim) const;
    size_t remaining(int dim) const;
    void advance(int dim, size_t amount);

    void skip_bytes(size_t bytes);

    const size_t *get_payload(size_t &count);

  protected:
    AddressList *addrlist;
    bool partial;
    // we need to be one larger than any index space realm supports, since
    //  we use the contiguous bytes within a field as a "dimension" in some
    //  cases
    static const int MAX_DIM = REALM_MAX_DIM + 1;
    int partial_dim;
    size_t pos[MAX_DIM];
  };

  struct SharedRectLayout {
    size_t base_offset;
    size_t total_bytes;
    size_t contig_bytes;
    int ndims;
    std::array<std::pair<size_t, size_t>, REALM_MAX_DIM> count_strides; // count, stride
  };

  struct PackedRect {
    SharedRectLayout layout;
    std::vector<FieldID> field_ids;
    size_t field_size;
  };

  class PackedRectAddressList {
  public:
    void add_rect(const SharedRectLayout &layout, const std::vector<FieldID> &field_ids,
                  size_t field_size)
    {
      rects.push_back(PackedRect{layout, field_ids, field_size});
      total_bytes += layout.total_bytes * field_ids.size();
    }

    bool get_next(PackedRect &out)
    {
      if(read_index >= rects.size())
        return false;
      out = rects[read_index++];
      return true;
    }

    size_t bytes_pending() const { return total_bytes; }

    void reset()
    {
      read_index = 0;
      rects.clear();
      total_bytes = 0;
    }

    void advance(size_t bytes)
    {
      assert(bytes <= total_bytes);
      total_bytes -= bytes;
    }

    bool done() const { return read_index >= rects.size(); }

  private:
    std::vector<PackedRect> rects;
    size_t read_index = 0;
    size_t total_bytes = 0;
  };

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc);
} // namespace Realm

#endif
