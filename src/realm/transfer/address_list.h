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
    AddressList(size_t _max_entries = 1000);

    size_t *begin_nd_entry(int max_dim, size_t payload_size = 0, bool wrap_mode = true);
    void commit_nd_entry(int act_dim, size_t bytes, size_t payload_size = 0);
    size_t bytes_pending() const;

    // entry packs the contiguous byte count (contig_bytes) in the upper bits the actual
    // dimension count (act_dim) in the lower 4 bits
    static size_t pack_entry_header(size_t contig_bytes, int dims);

    constexpr static size_t ADDRLIST_DIM_SLOTS = 2;
  protected:
    constexpr static size_t FLAG_HAS_EXTRA = (1UL << 63);
    constexpr static size_t ADDRLIST_DIM_MASK = 0xF;
    constexpr static size_t ADDRLIST_CONTIG_SHIFT = 4;

    friend class AddressListCursor;

    const size_t *read_entry();

    size_t total_bytes;
    unsigned write_pointer;
    unsigned read_pointer;
    std::vector<size_t> data;
    size_t max_entries;
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

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc);
} // namespace Realm

#endif
