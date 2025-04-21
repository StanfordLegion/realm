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

#include "realm/realm_config.h"

#ifdef REALM_ON_WINDOWS
#define NOMINMAX
#endif

#include "realm/transfer/address_list.h"
#include "realm/utils.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressList
  //

  AddressList::AddressList(size_t _max_entries)
    : max_entries(_max_entries)
  {
    data.reserve(max_entries);
  }

  size_t *AddressList::being_entry(int max_dim, size_t payload_size, bool wrap_mode)
  {
    size_t entries_needed =
        DIM_SLOTS * max_dim + (payload_size > 0 ? payload_size + 1 : 0);

    if(wrap_mode) {
      size_t new_wp = write_pointer + entries_needed;
      if(new_wp > max_entries) {
        if((read_pointer <= entries_needed) || (write_pointer < read_pointer))
          return nullptr;

        // fill remaining entries with 0's so reader skips
        while(write_pointer < max_entries)
          data[write_pointer++] = 0;

        write_pointer = 0;
        new_wp = entries_needed;
      } else {
        if((write_pointer < read_pointer) && (new_wp >= read_pointer))
          return nullptr;
        if((new_wp == max_entries) && (read_pointer == 0))
          return nullptr;
      }

      // ensure capacity upfront for max_entries once
      if(data.size() < max_entries)
        data.resize(max_entries);

      return data.data() + write_pointer;
    } else {
      if(data.size() < write_pointer + entries_needed)
        data.resize(write_pointer + entries_needed);
      return data.data() + write_pointer;
    }
  }

  void AddressList::commit_entry(int act_dim, size_t bytes, size_t payload_size)
  {
    size_t entries_used = act_dim * DIM_SLOTS;

    if(payload_size > 0) {
      data[write_pointer + entries_used] = payload_size;
      entries_used += 1;
    }

    entries_used += payload_size;
    data[write_pointer] |= (payload_size > 0 ? HAS_EXTRA : 0);

    write_pointer += entries_used;

    total_bytes += bytes * (field_block ? field_block->count : 1);
  }

  size_t AddressList::bytes_pending() const
  {
    return total_bytes;
  }

  size_t AddressList::pack_entry_header(size_t contig_bytes, int dims)
  {
    return (contig_bytes << CONTIG_SHIFT) | (dims & DIM_MASK);
  }

  const size_t *AddressList::read_entry()
  {
    // assert(total_bytes > 0);
    if(read_pointer >= max_entries) {
      assert(read_pointer == max_entries);
      read_pointer = 0;
    }
    // skip trailing 0's
    if(data[read_pointer] == 0)
      read_pointer = 0;
    return (data.data() + read_pointer);
  }

  // helpers
  // -----------------------------------------------------------------------------------------
  namespace detail {
    inline std::size_t contig_bytes(const std::size_t *e)
    {
      return ((e[AddressList::SLOT_HEADER] >> AddressList::CONTIG_SHIFT));
    }
    inline int actdim(const std::size_t *e)
    {
      return int(e[AddressList::SLOT_HEADER] & AddressList::DIM_MASK);
    }
  } // namespace detail

  ////////////////////////////////////////////////////////////////////////
  //
  // class AddressListCursor
  //

  AddressListCursor::AddressListCursor() { pos.fill(0); }

  void AddressListCursor::set_addrlist(AddressList *_addrlist) { addrlist = _addrlist; }

  int AddressListCursor::get_dim() const
  {
    assert(addrlist);
    // with partial progress, we restrict ourselves to just the rest of that dim
    if(partial) {
      return (partial_dim + 1);
    } else {
      const size_t *entry = addrlist->read_entry();
      int act_dim = (entry[0] & AddressList::DIM_MASK);
      return act_dim;
    }
  }

  uintptr_t AddressListCursor::get_offset() const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    uintptr_t ofs = entry[1];
    if(partial) {
      for(int i = partial_dim; i < act_dim; i++)
        if(i == 0) {
          // dim 0 is counted in bytes
          ofs += pos[0];
        } else {
          // rest use the strides from the address list
          ofs += pos[i] * entry[1 + (AddressList::DIM_SLOTS * i)];
        }
    }
    return ofs;
  }

  uintptr_t AddressListCursor::get_stride(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    assert((dim > 0) && (dim < act_dim));
    return entry[AddressList::DIM_SLOTS * dim + 1];
  }

  const size_t *AddressListCursor::get_payload(size_t &count)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    if(entry[0] & AddressList::HAS_EXTRA) {
      count = entry[act_dim * AddressList::DIM_SLOTS];
      return entry + act_dim * AddressList::DIM_SLOTS + 1;
    }
    return nullptr;
  }

  size_t AddressListCursor::remaining(int dim) const
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    assert(dim < act_dim);
    size_t r = entry[AddressList::DIM_SLOTS * dim];

    if(dim == 0) {
      r &= ~AddressList::HAS_EXTRA;
      r >>= AddressList::CONTIG_SHIFT;
    }

    if(partial) {
      if(dim > partial_dim)
        r = 1;
      if(dim == partial_dim) {
        assert(r > pos[dim]);
        r -= pos[dim];
      }
    }
    return r;
  }

  /*void AddressListCursor::advance(int dim, size_t amount, int f)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    assert(dim < act_dim);
    size_t r = entry[AddressList::DIM_SLOTS * dim];

    if(dim == 0) {
      r &= ~AddressList::HAS_EXTRA;
      r >>= AddressList::CONTIG_SHIFT;
    }

    size_t bytes = amount;
    if(dim > 0) {
#ifdef DEBUG_REALM
      for(int i = 0; i < dim; i++)
        assert(pos[i] == 0);
#endif
      bytes *= (entry[0] >> AddressList::CONTIG_SHIFT);
      for(int i = 1; i < dim; i++)
        bytes *= entry[AddressList::DIM_SLOTS * i];
    }

#ifdef DEBUG_REALM
    assert(addrlist->total_bytes >= bytes * f);
#endif
    addrlist->total_bytes -= bytes * f;

    const FieldBlock *fields = field_block();
    if(fields && partial_dim == 0 && f > 0) {
      partial_fields += f;
    }

    bool fields_complete = (!fields || (partial_fields == fields->count));

    if(!partial) {
      if((dim == (act_dim - 1)) && (amount == r) && fields_complete) {
        partial_fields = 0;
        // simple case - we consumed the whole thing
        addrlist->read_pointer += AddressList::DIM_SLOTS * act_dim;
        return;
      } else {
        // record partial consumption
        partial = true;
        partial_dim = dim;
        pos[partial_dim] = amount;
      }
    } else {
      // update a partial consumption in progress
      assert(dim <= partial_dim);
      partial_dim = dim;
      pos[partial_dim] += amount;
    }

    while(pos[partial_dim] == r) {
      pos[partial_dim++] = 0;

      if(partial_dim == act_dim) {
        partial = false;

        if(fields_complete) {
          partial_fields = 0;
          // all done
          addrlist->read_pointer += AddressList::DIM_SLOTS * act_dim;
        } else {
          partial_dim = 0;
          pos.fill(0);
        }

        break;
      } else {
        pos[partial_dim]++; // carry into next dimension
        r = entry[AddressList::DIM_SLOTS *
                  partial_dim]; // no shift because partial_dim > 0
      }
    }
  }*/

  void AddressListCursor::advance(int dim, size_t amount, int f)
  {
    const size_t *entry = addrlist->read_entry();
    int act_dim = (entry[0] & AddressList::DIM_MASK);
    assert(dim < act_dim);

    // size of this “slice” in dim
    size_t r = entry[AddressList::DIM_SLOTS * dim];
    if(dim == 0) {
      r &= ~AddressList::HAS_EXTRA;
      r >>= AddressList::CONTIG_SHIFT;
    }

    // compute how many bytes we’re really removing
    size_t bytes = amount;
    if(dim > 0) {
#ifdef DEBUG_REALM
      for(int i = 0; i < dim; i++)
        assert(pos[i] == 0);
#endif
      bytes *= (entry[0] >> AddressList::CONTIG_SHIFT);
      for(int i = 1; i < dim; i++)
        bytes *= entry[AddressList::DIM_SLOTS * i];
    }

#ifdef DEBUG_REALM
    assert(addrlist->total_bytes >= bytes * f);
#endif
    addrlist->total_bytes -= bytes * f;

    const FieldBlock *fields = field_block();

    // ——— NEW: if this call exactly finishes *one* rect (the last dim)
    if(dim == (act_dim - 1) && amount == r) {
      if(fields && f > 0) {
        // bump fields only on a full‐rect consume
        partial_fields += f;
        if(partial_fields >= fields->count) {
          if(addrlist->bytes_pending() > 0)
            partial_fields = 0;
          addrlist->read_pointer += AddressList::DIM_SLOTS * act_dim;
        }
      } else {
        // no fields at all: consume entry immediately
        addrlist->read_pointer += AddressList::DIM_SLOTS * act_dim;
      }
      // reset any in‐flight partial state
      partial = false;
      partial_dim = 0;
      pos.fill(0);
      return;
    }

    // ——— otherwise fall back to the existing “partial” logic
    if(!partial) {
      partial = true;
      partial_dim = dim;
      pos[dim] = amount;
    } else {
      assert(dim <= partial_dim);
      partial_dim = dim;
      pos[dim] += amount;
    }

    bool fields_complete = (!fields || (partial_fields == fields->count));

    while(pos[partial_dim] == r) {
      pos[partial_dim++] = 0;

      if(partial_dim == act_dim) {
        partial = false;

        if(fields_complete) {
          if(addrlist->bytes_pending() > 0)
            partial_fields = 0;
          // all done
          addrlist->read_pointer += AddressList::DIM_SLOTS * act_dim;
        } else {
          partial_fields += f;
          partial_dim = 0;
          pos.fill(0);
        }

        break;
      } else {
        pos[partial_dim]++; // carry into next dimension
        r = entry[AddressList::DIM_SLOTS *
                  partial_dim]; // no shift because partial_dim > 0
      }
    }
  }

  void AddressListCursor::skip_bytes(size_t bytes)
  {
    while(bytes > 0) {
      int act_dim = get_dim();

      if(act_dim == 0) {
        assert(0);
      } else {
        size_t chunk = remaining(0);
        if(chunk <= bytes) {
          int dim = 0;
          size_t count = chunk;
          while((dim + 1) < act_dim) {
            dim++;
            count = bytes / chunk;
            assert(count > 0);
            size_t r = remaining(dim + 1);
            if(count < r) {
              chunk *= count;
              break;
            } else {
              count = r;
              chunk *= count;
            }
          }
          advance(dim, count);
          bytes -= chunk;
        } else {
          advance(0, bytes);
          return;
        }
      }
    }
  }

  std::ostream &operator<<(std::ostream &os, const AddressListCursor &alc)
  {
    os << alc.remaining(0);
    for(int i = 1; i < alc.get_dim(); i++)
      os << 'x' << alc.remaining(i);
    os << ',' << alc.get_offset();
    for(int i = 1; i < alc.get_dim(); i++)
      os << '+' << alc.get_stride(i);
    return os;
  }
} // namespace Realm
