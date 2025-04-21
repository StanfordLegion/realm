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

  struct FieldBlock {
    using FieldID = int;
    std::size_t count;
    FieldID fields[1];

    // allocate a FieldBlock via heap.alloc_obj and store n field IDs
    template <typename Heap>
    static FieldBlock *create(Heap &heap, const FieldID *src, std::size_t n,
                              std::size_t align = 16)
    {
      const std::size_t bytes = sizeof(FieldBlock) + (n - 1) * sizeof(FieldID);
      void *mem = heap.alloc_obj(bytes, align);
      auto *fb = new(mem) FieldBlock;
      fb->count = n;
      std::copy_n(src, n, fb->fields);
      return fb;
    }
  };

  // =================================================================================================
  //                                             AddressList
  // =================================================================================================
  class AddressList {
  public:
    AddressList(size_t _max_entries = 1000);

    // ─── entry construction ──────────────────────────────────────────────────────
    size_t *begin_entry(int max_dim, bool wrap_mode = true);
    void commit_entry(int act_dim, size_t bytes);

    size_t bytes_pending() const;
    void set_field_block(const FieldBlock *_field_block) { field_block = _field_block; }

    // entry packs:
    // the contiguous byte count (contig_bytes) in the upper bitsthe
    // the actual dimension count (act_dim) in the lower 4 bits
    static size_t pack_entry_header(size_t contig_bytes, int dims);

    // ─── layout constants ───────────────────────────────────────────────────────
    static constexpr size_t SLOT_HEADER = 0;
    static constexpr size_t SLOT_BASE = 1;
    static constexpr size_t DIM_SLOTS = 2;
    static constexpr size_t DIM_MASK = 0xF;
    static constexpr size_t CONTIG_SHIFT = 4;

  protected:
    friend class AddressListCursor;
    const size_t *read_entry();

    const FieldBlock *field_block{nullptr};

    size_t total_bytes{0};
    size_t write_pointer{0};
    size_t read_pointer{0};
    size_t max_entries{0};
    std::vector<size_t> data;
  };

  // =================================================================================================
  //                                           AddressListCursor
  // =================================================================================================
  class AddressListCursor {
  public:
    AddressListCursor();

    void set_addrlist(AddressList *_addrlist);

    int get_dim() const;
    uintptr_t get_offset() const;
    uintptr_t get_stride(int dim) const;
    size_t remaining(int dim) const;
    void advance(int dim, size_t amount, int f = 1);

    void skip_bytes(size_t bytes);

    const FieldBlock *field_block() const { return addrlist->field_block; }
    const FieldID *fields_data() const
    {
      return addrlist->field_block->fields + partial_fields;
    }

    size_t fields() const
    {
      if(addrlist->field_block) {
        return addrlist->field_block->count - partial_fields;
      }
      return 1;
    }

  protected:
    AddressList *addrlist{nullptr};
    bool partial{false}; // inside a dimension
    int partial_dim{0};  // dimension index
    size_t partial_fields{0};
    std::array<size_t, REALM_MAX_DIM + 1> pos{};
  };

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc);
} // namespace Realm

#endif
