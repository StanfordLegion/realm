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

// Memory implementations for Realm

#ifndef REALM_MEMORY_IMPL_INL
#define REALM_MEMORY_IMPL_INL

// nop, but helpful for IDEs
#include "realm/mem_impl.h"
#include <unordered_set>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryImpl
  //

  template <typename T>
  T *MemoryImpl::find_module_specific()
  {
    ModuleSpecificInfo *info = module_specific;
    while(info) {
      T *downcast = dynamic_cast<T *>(info);
      if(downcast)
        return downcast;
      info = info->next;
    }
    return 0;
  }

  template <typename T>
  const T *MemoryImpl::find_module_specific() const
  {
    const ModuleSpecificInfo *info = module_specific;
    while(info) {
      const T *downcast = dynamic_cast<const T *>(info);
      if(downcast)
        return downcast;
      info = info->next;
    }
    return 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class BasicRangeAllocator<RT,TT>
  //

#if 0
  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT,TT>::Range::Range(RT _first, RT _last)
    : first(_first), last(_last)
    , prev(-1), next(-1)
    , prev_free(-1), next_free(-1)
  {}
#endif

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT, TT>::BasicRangeAllocator(void)
    : first_free_range(SENTINEL)
  {
    ranges.resize(1);
    Range &s = ranges[SENTINEL];
    s.first = RT(-1);
    s.last = 0;
    s.prev = s.next = s.prev_free = s.next_free = SENTINEL;
  }

  template <typename RT, typename TT>
  inline BasicRangeAllocator<RT, TT>::~BasicRangeAllocator(void)
  {}

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT, TT>::swap(BasicRangeAllocator<RT, TT> &swap_with)
  {
    allocated.swap(swap_with.allocated);
#ifdef DEBUG_REALM
    by_first.swap(swap_with.by_first);
#endif
    ranges.swap(swap_with.ranges);
    std::swap(first_free_range, swap_with.first_free_range);
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT, TT>::add_range(RT first, RT last)
  {
    // ignore empty ranges
    if(first == last)
      return;

    int new_idx = alloc_range(first, last);

    Range &newr = ranges[new_idx];
    Range &sentinel = ranges[SENTINEL];

    // simple case - starting range
    if(sentinel.next == SENTINEL) {
      // all block list
      newr.prev = newr.next = SENTINEL;
      sentinel.prev = sentinel.next = new_idx;
      // free block list
      newr.prev_free = newr.next_free = SENTINEL;
      sentinel.prev_free = sentinel.next_free = new_idx;

#ifdef DEBUG_REALM
      by_first[first] = new_idx;
#endif
      return;
    }

    abort();
  }

  template <typename RT, typename TT>
  inline unsigned BasicRangeAllocator<RT, TT>::alloc_range(RT first, RT last)
  {
    // find/make a free index in the range list for this range
    int new_idx;
    if(first_free_range != SENTINEL) {
      new_idx = first_free_range;
      first_free_range = ranges[new_idx].next;
    } else {
      new_idx = ranges.size();
      ranges.resize(new_idx + 1);
    }
    ranges[new_idx].first = first;
    ranges[new_idx].last = last;
    return new_idx;
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT, TT>::free_range(unsigned index)
  {
    ranges[index].next = first_free_range;
    first_free_range = index;
  }

  template <typename RT>
  static RT calculate_offset(size_t start, RT alignment)
  {
    RT offset = 0;
    if(alignment) {
      RT rem = start % alignment;
      if(rem > 0) {
        offset = alignment - rem;
      }
    }
    return offset;
  }

  template <typename RT, typename TT>
  inline size_t BasicRangeAllocator<RT, TT>::split_range(
      TT old_tag, const std::vector<TT> &new_tags, const std::vector<RT> &sizes,
      const std::vector<RT> &alignments, std::vector<RT> &allocs_first, bool missing_ok)
  {
    typename std::map<TT, unsigned>::iterator it = allocated.find(old_tag);
    if(it == allocated.end()) {
      REALM_ASSERT(missing_ok);
      return 0;
    }

    const size_t n = new_tags.size();
    REALM_ASSERT(n == sizes.size() && n == alignments.size());
    REALM_ASSERT(allocs_first.size() == n);

    const unsigned range_idx = it->second;
    if(range_idx == SENTINEL) {
      // this is a zero-sized range so we can redistrict only to zero-sized instances
      for(size_t i = 0; i < n; i++) {
        // No need to check for duplicate tags here since they are going
        // to be assigned the same sentinel value anyway
        if(sizes[i]) {
          deallocate(old_tag);
          return i;
        }
        allocated[new_tags[i]] = SENTINEL;
        // Make sure zero-sized instances have a valid offset
        allocs_first[i] = 0;
      }
      deallocate(old_tag);
      return n;
    }

    Range *r = &ranges[range_idx];
    for(size_t i = 0; i < n; i++) {
      REALM_ASSERT(allocated.find(new_tags[i]) == allocated.end());
      if(sizes[i]) {
        RT offset = calculate_offset(r->first, alignments[i]);
        // do we have enough space?
        if((r->last - r->first) < (sizes[i] + offset)) {
          deallocate(old_tag);
          return i;
        }
        allocs_first[i] = r->first + offset;
        RT alloc_last = allocs_first[i] + sizes[i];
        if(offset) { // Offset padding needs to be freed
          // See if we can merge with the previous free range before us
          unsigned pf_idx = r->prev;
          while((pf_idx != SENTINEL) && (ranges[pf_idx].prev_free == pf_idx)) {
            pf_idx = ranges[pf_idx].prev;
            REALM_ASSERT(pf_idx != range_idx); // wrapping around would be bad
          }
          if((pf_idx == r->prev) && (pf_idx != SENTINEL)) {
            // Previous range is free so we can expand it to include offset
            ranges[pf_idx].last = allocs_first[i];
          } else {
            // Create a new free range and insert it into the free list
            unsigned new_idx = alloc_range(r->first, allocs_first[i]);
            r = &ranges[range_idx]; // alloc may have moved this!
            Range &new_range = ranges[new_idx];
            new_range.prev = r->prev;
            new_range.next = range_idx;
            ranges[r->prev].next = new_idx;
            r->prev = new_idx;
            // Insert the new range into the free list
            Range &prev = ranges[pf_idx];
            new_range.prev_free = pf_idx;
            new_range.next_free = prev.next_free;
            ranges[prev.next_free].prev_free = new_idx;
            prev.next_free = new_idx;
          }
        }
        // Now make the new range for the tag
        unsigned new_idx = alloc_range(allocs_first[i], alloc_last);
        r = &ranges[range_idx]; // alloc may have moved this!
        r->first = alloc_last;
        Range &new_range = ranges[new_idx];
        new_range.prev = r->prev;
        new_range.next = range_idx;
        ranges[r->prev].next = new_idx;
        r->prev = new_idx;
        // tie this off because we use it to detect allocated-ness
        new_range.prev_free = new_range.next_free = new_idx;
        allocated[new_tags[i]] = new_idx;

        // Detect the case where the old range is empty
        if(r->first == r->last) {
          deallocate(it->first);
          return (i + 1);
        }
      } else { // Zero-sized instances are easy
        allocated[new_tags[i]] = SENTINEL;
        // Make sure zero-sized instances have a valid offset
        allocs_first[i] = 0;
      }
    }
    // deallocate whatever is left of the old instance
    deallocate(old_tag);
#ifdef DEBUG_REALM
    bool has_cycle = free_list_has_cycle();
    bool invalid = has_invalid_ranges();
    if(has_cycle || invalid) {
      REALM_ASSERT(has_cycle == false);
      REALM_ASSERT(invalid == false);
    }
#endif
    return n;
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT, TT>::free_list_has_cycle()
  {
    if(ranges.empty()) {
      return false;
    }

    unsigned tortoise = ranges[SENTINEL].next_free;
    unsigned hare = ranges[SENTINEL].next_free;

    while(true) {
      tortoise = ranges[tortoise].next_free;
      hare = ranges[ranges[hare].next_free].next_free;

      if(tortoise == SENTINEL || hare == SENTINEL) {
        return false;
      }

      if(tortoise == hare) {
        std::cerr << "ERROR - found cycle at index:" << tortoise << std::endl;
        return true;
      }
    }
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT, TT>::has_invalid_ranges()
  {
    // TODO(apryakhin@): Consider doing it more efficient.
    std::unordered_set<unsigned> range_indices;
    for(const auto &alloc : allocated) {
      if(alloc.second == SENTINEL)
        continue;
      if(!range_indices.insert(alloc.second).second) {
        std::cerr << "ERROR: found duplicate range idx:" << alloc.second
                  << " for tag:" << alloc.first << std::endl;
        return true;
      }
    }

    for(unsigned idx = ranges[SENTINEL].next; idx != SENTINEL; idx = ranges[idx].next) {
      if(ranges[idx].first > ranges[idx].last ||
         ranges[idx].last > ranges[ranges[idx].next].first) {
        std::cerr << "ERROR: found invalid range idx:" << idx << std::endl;
        return true;
      }
    }
    return false;
  }

  template <typename RT, typename TT>
  inline typename BasicRangeAllocator<RT, TT>::MemoryStats BasicRangeAllocator<RT, TT>::get_allocator_stats()
  {
    typename BasicRangeAllocator<RT, TT>::MemoryStats stats;
    size_t total_size = 0;
    unsigned range_idx = ranges[SENTINEL].next;
    while(range_idx != SENTINEL) {
      unsigned i = range_idx;
      size_t size = ranges[i].last - ranges[i].first;
      total_size += size;
      range_idx = ranges[range_idx].next;
    }

    stats.total_size = total_size;

    size_t largest_used_blocksize = 0;
    size_t total_used_size = 0;
    for(auto alloc_it = allocated.begin(); alloc_it != allocated.end(); ++alloc_it) {
      if(alloc_it->second == SENTINEL)
        continue;
      unsigned i = alloc_it->second;
      size_t size = ranges[i].last - ranges[i].first;
      if(largest_used_blocksize < size) {
        largest_used_blocksize = size;
      }
      total_used_size += size;
    }

    stats.total_used_size = total_used_size;

    size_t largest_free_blocksize = 0;
    size_t total_free_size = 0;
    unsigned free_idx = ranges[SENTINEL].next_free;
    while(free_idx != SENTINEL) {
      unsigned i = free_idx;
      size_t size = ranges[i].last - ranges[i].first;

      total_free_size += size;
      if(largest_free_blocksize < size) {
        largest_free_blocksize = size;
      }
      free_idx = ranges[free_idx].next_free;
    }

    stats.total_free_size = total_free_size;
    stats.largest_free_blocksize = largest_free_blocksize;

    REALM_ASSERT(total_size == total_used_size + total_free_size);
    return stats;
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT, TT>::can_allocate(TT tag, RT size, RT alignment)
  {
    // empty allocation requests are trivial
    if(size == 0) {
      return true;
    }

    // walk free ranges and just take the first that fits
    unsigned idx = ranges[SENTINEL].next_free;
    while(idx != SENTINEL) {
      Range *r = &ranges[idx];

      RT ofs = 0;
      if(alignment) {
        RT rem = r->first % alignment;
        if(rem > 0)
          ofs = alignment - rem;
      }
      // do we have enough space?
      if((r->last - r->first) >= (size + ofs))
        return true;

      // no, go to next one
      idx = r->next_free;
    }

    // allocation failed
    return false;
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT, TT>::allocate(TT tag, RT size, RT alignment,
                                                    RT &alloc_first)
  {
    // empty allocation requests are trivial
    if(size == 0) {
      allocated[tag] = SENTINEL;
      return true;
    }

    // walk free ranges and just take the first that fits
    unsigned idx = ranges[SENTINEL].next_free;
    while(idx != SENTINEL) {
      Range *r = &ranges[idx];

      RT ofs = 0;
      if(alignment) {
        RT rem = r->first % alignment;
        if(rem > 0)
          ofs = alignment - rem;
      }
      // do we have enough space?
      if((r->last - r->first) >= (size + ofs)) {
        // yes, but we may need to chop things up to make the exact range we want
        alloc_first = r->first + ofs;
        RT alloc_last = alloc_first + size;

        // do we need to carve off a new (free) block before us?
        if(alloc_first != r->first) {
          unsigned new_idx = alloc_range(r->first, alloc_first);
          Range *new_prev = &ranges[new_idx];
          r = &ranges[idx]; // alloc may have moved this!

          r->first = alloc_first;
          // insert into all-block dllist
          new_prev->prev = r->prev;
          new_prev->next = idx;
          ranges[r->prev].next = new_idx;
          r->prev = new_idx;
          // insert into free-block dllist
          new_prev->prev_free = r->prev_free;
          new_prev->next_free = idx;
          ranges[r->prev_free].next_free = new_idx;
          r->prev_free = new_idx;

#ifdef DEBUG_REALM
          // fix up by_first entries
          by_first[r->first] = new_idx;
          by_first[alloc_first] = idx;
#endif
        }

        // two cases to deal with
        if(alloc_last == r->last) {
          // case 1 - exact fit
          //
          // all we have to do here is remove this range from the free range dlist
          //  and add to the allocated lookup map
          ranges[r->prev_free].next_free = r->next_free;
          ranges[r->next_free].prev_free = r->prev_free;
        } else {
          // case 2 - leftover at end - put in new range
          unsigned after_idx = alloc_range(alloc_last, r->last);
          Range *r_after = &ranges[after_idx];
          r = &ranges[idx]; // alloc may have moved this!

#ifdef DEBUG_REALM
          by_first[alloc_last] = after_idx;
#endif
          r->last = alloc_last;

          // r_after goes after r in all block list
          r_after->prev = idx;
          r_after->next = r->next;
          r->next = after_idx;
          ranges[r_after->next].prev = after_idx;

          // r_after replaces r in the free block list
          r_after->prev_free = r->prev_free;
          r_after->next_free = r->next_free;
          ranges[r_after->next_free].prev_free = after_idx;
          ranges[r_after->prev_free].next_free = after_idx;
        }

        // tie this off because we use it to detect allocated-ness
        r->prev_free = r->next_free = idx;

        allocated[tag] = idx;

        return true;
      }

      // no, go to next one
      idx = r->next_free;
    }

    // allocation failed
    return false;
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT, TT>::deallocate(unsigned del_idx)
  {
    // if there was no Range associated with this tag, it was an zero-size
    //  allocation, and there's nothing to add to the free list
    if(del_idx == SENTINEL)
      return;

    Range &r = ranges[del_idx];

    unsigned pf_idx = r.prev;
    while((pf_idx != SENTINEL) && (ranges[pf_idx].prev_free == pf_idx)) {
      pf_idx = ranges[pf_idx].prev;
      REALM_ASSERT(pf_idx != del_idx); // wrapping around would be bad
    }
    unsigned nf_idx = r.next;
    while((nf_idx != SENTINEL) && (ranges[nf_idx].next_free == nf_idx)) {
      nf_idx = ranges[nf_idx].next;
      REALM_ASSERT(nf_idx != del_idx);
    }

    // do we need to merge?
    bool merge_prev = (pf_idx == r.prev) && (pf_idx != SENTINEL);
    bool merge_next = (nf_idx == r.next) && (nf_idx != SENTINEL);

    // four cases - ordered to match the allocation cases
    if(!merge_next) {
      if(!merge_prev) {
        // case 1 - no merging (exact match)
        // just add ourselves to the free list
        r.prev_free = pf_idx;
        r.next_free = nf_idx;
        ranges[pf_idx].next_free = del_idx;
        ranges[nf_idx].prev_free = del_idx;
      } else {
        // case 2 - merge before
        // merge ourselves into the range before
        Range &r_before = ranges[pf_idx];

        r_before.last = r.last;
        r_before.next = r.next;
        ranges[r.next].prev = pf_idx;
        // r_before was already in free list, so no changes to that

#ifdef DEBUG_REALM
        by_first.erase(r.first);
#endif
        free_range(del_idx);
      }
    } else {
      if(!merge_prev) {
        // case 3 - merge after
        // merge ourselves into the range after
        Range &r_after = ranges[nf_idx];

#ifdef DEBUG_REALM
        by_first[r.first] = nf_idx;
        by_first.erase(r_after.first);
#endif

        r_after.first = r.first;
        r_after.prev = r.prev;
        ranges[r.prev].next = nf_idx;
        // r_after was already in the free list, so no changes to that

        free_range(del_idx);
      } else {
        // case 4 - merge both
        // merge both ourselves and range after into range before
        Range &r_before = ranges[pf_idx];
        Range &r_after = ranges[nf_idx];

        r_before.last = r_after.last;
#ifdef DEBUG_REALM
        by_first.erase(r.first);
        by_first.erase(r_after.first);
#endif

        // adjust both normal list and free list
        r_before.next = r_after.next;
        ranges[r_after.next].prev = pf_idx;

        r_before.next_free = r_after.next_free;
        ranges[r_after.next_free].prev_free = pf_idx;

        free_range(del_idx);
        free_range(nf_idx);
      }
    }
  }

  template <typename RT, typename TT>
  inline void BasicRangeAllocator<RT, TT>::deallocate(TT tag, bool missing_ok /*= false*/)
  {
    typename std::map<TT, unsigned>::iterator it = allocated.find(tag);
    if(it == allocated.end()) {
      REALM_ASSERT(missing_ok);
      return;
    }
    unsigned del_idx = it->second;
    allocated.erase(it);

    deallocate(del_idx);
  }

  template <typename RT, typename TT>
  inline bool BasicRangeAllocator<RT, TT>::lookup(TT tag, RT &first, RT &size)
  {
    typename std::map<TT, unsigned>::iterator it = allocated.find(tag);

    if(it != allocated.end()) {
      // if there was no Range associated with this tag, it was an zero-size
      //  allocation
      if(it->second == SENTINEL) {
        first = 0;
        size = 0;
      } else {
        const Range &r = ranges[it->second];
        first = r.first;
        size = r.last - r.first;
      }

      return true;
    } else
      return false;
  }

  template <typename RT, typename TT, bool SORTED>
  inline SizedRangeAllocator<RT,TT,SORTED>::SizedRangeAllocator(void)
    : BasicRangeAllocator<RT,TT>()
  {}

  template <typename RT, typename TT, bool SORTED>
  inline SizedRangeAllocator<RT,TT,SORTED>::~SizedRangeAllocator(void)
  {}
  
  template <typename RT, typename TT, bool SORTED>
  inline void SizedRangeAllocator<RT,TT,SORTED>::swap(SizedRangeAllocator<RT,TT,SORTED>& swap_with)
  {
    BasicRangeAllocator<RT,TT>::swap(swap_with);
    size_based_free_lists.swap(swap_with.size_based_free_lists);
  }

  template <typename RT, typename TT, bool SORTED>
  inline typename BasicRangeAllocator<RT,TT>::MemoryStats SizedRangeAllocator<RT,TT,SORTED>::get_allocator_stats()
  {
    typename BasicRangeAllocator<RT, TT>::MemoryStats stats;
    size_t total_size = 0;

    const auto &allocated = this->allocated;
    const auto& ranges = this->ranges;

    unsigned range_idx = ranges[SENTINEL].next;
    while(range_idx != SENTINEL) {
      unsigned i = range_idx;
      size_t size = ranges[i].last - ranges[i].first;
      total_size += size;
      range_idx = ranges[range_idx].next;
    }

    stats.total_size = total_size;

    size_t largest_used_blocksize = 0;
    size_t total_used_size = 0;
    for(auto alloc_it = allocated.begin(); alloc_it != allocated.end(); ++alloc_it) {
      if(alloc_it->second == SENTINEL)
        continue;
      unsigned i = alloc_it->second;
      size_t size = ranges[i].last - ranges[i].first;
      if(largest_used_blocksize < size) {
        largest_used_blocksize = size;
      }
      total_used_size += size;
    }

    stats.total_used_size = total_used_size;

    size_t largest_free_blocksize = 0;
    size_t total_free_size = 0;

    for(unsigned idx = 1; idx < size_based_free_lists.size(); idx++) {
      unsigned index = size_based_free_lists[idx];
      while(index != SENTINEL) {
        Range *r = &this->ranges[index];
        size_t size = r->last - r->first;
        total_free_size += size;
        if(largest_free_blocksize < size) {
          largest_free_blocksize = size;
        }
        if(index == r->next_free)
          break;
        index = r->next_free;
      }
    }

    stats.total_free_size = total_free_size;
    stats.largest_free_blocksize = stats.largest_free_blocksize;

    REALM_ASSERT(total_size == total_used_size + total_free_size);
    return stats;
  }

  template <typename RT, typename TT, bool SORTED>
  inline void SizedRangeAllocator<RT,TT,SORTED>::add_range(RT first, RT last)
  {
    // ignore empty ranges
    if(first == last)
      return;

    int new_idx = this->alloc_range(first, last);

    Range& newr = this->ranges[new_idx];
    Range& sentinel = this->ranges[SENTINEL];

    // simple case - starting range
    if(sentinel.next == SENTINEL) {
      // all block list
      newr.prev = newr.next = SENTINEL;
      sentinel.prev = sentinel.next = new_idx;
      // free block list
      newr.prev_free = newr.next_free = SENTINEL;
      RT size = last - first;
      unsigned log2_size = floor_log2(size);
      size_based_free_lists.resize(log2_size + 1, SENTINEL);
      size_based_free_lists[log2_size] = new_idx;

#ifdef DEBUG_REALM
      this->by_first[first] = new_idx;
#endif
      return;
    }

    abort();
  }

  template <typename RT, typename TT, bool SORTED>
  inline bool SizedRangeAllocator<RT,TT,SORTED>::can_allocate(TT tag, RT size, RT alignment)
  {
    if(size == 0)
      return true;

    for (unsigned idx = floor_log2(size); idx < size_based_free_lists.size(); idx++) {
      unsigned index = size_based_free_lists[idx];
      while (index != SENTINEL) {
        Range *r = &this->ranges[index];

        RT offset = 0;
        if (alignment) {
          RT remainder = r->first % alignment;
          if (remainder)
            offset = alignment - remainder;
        }
        // do we have enough space?
        if ((r->last - r->first) >= (size + offset))
          return true;
        // No, keep going
        index = r->next_free;
      }
    }
    return false;
  }

  template <typename RT, typename TT, bool SORTED>
  inline bool SizedRangeAllocator<RT,TT,SORTED>::allocate(TT tag, RT size, RT alignment, RT& alloc_first)
  {
    // empty allocation requests are trivial
    if(size == 0) {
      this->allocated[tag] = SENTINEL;
      return true;
    }

    for (unsigned idx = floor_log2(size); idx < size_based_free_lists.size(); idx++) {
      unsigned index = size_based_free_lists[idx];
      while (index != SENTINEL) {
        Range *r = &this->ranges[index];

        RT offset = 0;
        if (alignment) {
          RT remainder = r->first % alignment;
          if (remainder)
            offset = alignment - remainder;
        }
        // do we have enough space?
        if ((r->last - r->first) < (size + offset)) {
          // No, keep going
          index = r->next_free;
          continue;
        }
        // We have enough space
        // Remove this from the current size free list
        remove_from_free_list(index, *r);
        // but we we may to chop things up to make the exact range
        alloc_first = r->first + offset;
        RT alloc_last = alloc_first + size;
        // do we need to carve off a new (free) block before us?
        if (offset) {
          unsigned new_index = this->alloc_range(r->first, alloc_first);
          Range &new_prev = this->ranges[new_index];
          r = &this->ranges[index];  // alloc may have moved this!
          r->first = alloc_first;
          // insert into all-block dllist
          new_prev.prev = r->prev;
          new_prev.next = index;
          this->ranges[r->prev].next = new_index;
          r->prev = new_index;
          // Insert into the free list of the appropriate size
          add_to_free_list(new_index, new_prev); 
        }
        // see if we have leftover space and need to make a new range
        // to represent the remainder
        if (alloc_last != r->last) {
          // case 2 - leftover at end - put in new range
          unsigned after_index = this->alloc_range(alloc_last, r->last);
          Range &r_after = this->ranges[after_index];
          r = &this->ranges[index];  // alloc may have moved this!
          r->last = alloc_last;
          // r_after goes after r in all block list
          r_after.prev = index;
          r_after.next = r->next;
          r->next = after_index;
          this->ranges[r_after.next].prev = after_index;
          // Put r_after in the free list of the right size
          add_to_free_list(after_index, r_after);
        }
        // tie this off because we use it to detect allocated-ness
        r->prev_free = r->next_free = index;
        this->allocated[tag] = index;
        return true;
      }
    }
    return false;
  }

  template <typename RT, typename TT, bool SORTED>
  inline void SizedRangeAllocator<RT,TT,SORTED>::deallocate(TT tag, bool missing_ok)
  {
    typename std::map<TT, unsigned>::iterator it = this->allocated.find(tag);
    if(it == this->allocated.end()) {
      REALM_ASSERT(missing_ok);
      return;
    }
    unsigned del_idx = it->second;
    this->allocated.erase(it);

    // if there was no Range associated with this tag, it was an zero-size
    //  allocation, and there's nothing to add to the free list
    if(del_idx == SENTINEL)
      return;

    Range& r = this->ranges[del_idx];

    // See if the previous range is free so we can merge with it
    unsigned pf_idx = r.prev;
    bool merge_prev = (pf_idx != SENTINEL) && (this->ranges[pf_idx].prev_free != pf_idx);
    // See if the next range is free so we can merge with it
    unsigned nf_idx = r.next;
    bool merge_next = (nf_idx != SENTINEL) && (this->ranges[nf_idx].next_free != nf_idx);

    // four cases - ordered to match the allocation cases
    if (!merge_next) 
    {
      if (!merge_prev) 
      {
        // case 1 - no merging (exact match)
        // just add ourselves to the free list
        add_to_free_list(del_idx, r);
      } 
      else 
      {
        // case 2 - merge before
        // merge ourselves into the range before
        Range& r_before = this->ranges[pf_idx];
        grow_hole(pf_idx, r_before, r.last, false/*before*/);
        r_before.next = r.next;
        this->ranges[r.next].prev = pf_idx;
        this->free_range(del_idx);
      }
    } 
    else 
    {
      if (!merge_prev) 
      {
        // case 3 - merge after
        // merge ourselves into the range after
        Range& r_after = this->ranges[nf_idx];
        grow_hole(nf_idx, r_after, r.first, true/*before*/);
        r_after.prev = r.prev;
        this->ranges[r.prev].next = nf_idx;
        this->free_range(del_idx);
      } 
      else 
      {
        // case 4 - merge both
        // merge both ourselves and range after into range before
        Range& r_before = this->ranges[pf_idx];
        Range& r_after = this->ranges[nf_idx];
        remove_from_free_list(nf_idx, r_after);
        grow_hole(pf_idx, r_before, r_after.last, false/*before*/);
        // adjust both normal list and free list
        r_before.next = r_after.next;
        this->ranges[r_after.next].prev = pf_idx;

        this->free_range(del_idx);
        this->free_range(nf_idx);
      }
    }
  }

  template <typename RT, typename TT, bool SORTED>
  size_t SizedRangeAllocator<RT, TT, SORTED>::split_range(TT old_tag,
                                                        const std::vector<TT> &new_tags,
                                                        const std::vector<RT> &sizes,
                                                        const std::vector<RT> &alignments,
                                                        std::vector<RT> &allocs_first,
                                                        bool missing_ok)
  {
    typename std::map<TT, unsigned>::iterator it = this->allocated.find(old_tag);
    if(it == this->allocated.end()) {
      REALM_ASSERT(missing_ok);
      return 0;
    }

    size_t n = new_tags.size();
    REALM_ASSERT(n == sizes.size() && n == alignments.size());

    unsigned index = it->second;
    if (index == SENTINEL) {
      // this is a zero-sized range so we can redistrict only to zero-sized instances
      for (size_t i = 0; i < n; i++) {
        // No need to check for duplicate tags here since they are going
        // to be assigned the same sentinel value anyway
        if (sizes[i]) {
          deallocate(old_tag);
          return i;
        }
        this->allocated[new_tags[i]] = SENTINEL;
        // Make sure zero-sized instances have a valid offset
        allocs_first[i] = 0;
      }
      deallocate(old_tag);
      return n;
    }

    Range *r = &this->ranges[index];
    for (unsigned idx = 0; idx < n; idx++) {
      RT offset = 0;
      if (alignments[idx]) {
        RT remainder = r->first % alignments[idx];
        if (remainder)
          offset = alignments[idx] - remainder;
      }
      // do we have enough space?
      if ((r->last - r->first) < (sizes[idx] + offset)) {
        deallocate(old_tag);
        return idx;
      }
      RT alloc_first = r->first + offset;
      allocs_first[idx] = r->first + offset;
      RT alloc_last = alloc_first + sizes[idx];
      if (offset) {
        // See if we can merge with a free range before us
        unsigned pf_idx = r->prev;
        bool merge_prev = (pf_idx != SENTINEL) && (this->ranges[pf_idx].prev_free != pf_idx);
        if (merge_prev) {
          Range &prev = this->ranges[pf_idx];
          grow_hole(pf_idx, prev, alloc_first, false/*before*/);
          r->first = alloc_first;
          add_to_free_list(pf_idx, prev);
        } else {
          unsigned new_index = this->alloc_range(r->first, alloc_first);
          Range &new_prev = this->ranges[new_index];
          r = &this->ranges[index];  // alloc may have moved this!
          r->first = alloc_first;
          // insert into all-block dllist
          new_prev.prev = r->prev;
          new_prev.next = index;
          this->ranges[r->prev].next = new_index;
          r->prev = new_index;
          // Insert into the free list of the appropriate size
          add_to_free_list(new_index, new_prev);
        }
      }
      // Now make the new range for the tag
      unsigned new_index = this->alloc_range(alloc_first, alloc_last);
      Range &new_range = this->ranges[new_index];
      r = &this->ranges[index];  // alloc may have moved this!
      r->first = alloc_last;
      new_range.prev = r->prev;
      new_range.next = index;
      this->ranges[r->prev].next = new_index;
      r->prev = new_index;
      // tie this off because we use it to detect allocated-ness
      new_range.prev_free = new_range.next_free = new_index;
      this->allocated[new_tags[idx]] = new_index;
    }
    // deallocate whatever is left of the old instance
    deallocate(old_tag);
    return n;
  }

  template <typename RT, typename TT, bool SORTED>
  void SizedRangeAllocator<RT,TT,SORTED>::add_to_free_list(unsigned index, Range& range)
  {
    RT size = range.last - range.first;
    if(size == 0) {
      // This can happen when we are splitting a range and the remainder ends up
      // being empty, in which case there is no "hole" to add to the free list
      // and we just need to remove this entry from the list of ranges and
      // recycle the tag for later
      this->ranges[range.prev].next = range.next;
      this->ranges[range.next].prev = range.prev;
      this->free_range(index);
      return;
    }
    unsigned log2_size = floor_log2(size);
    if (size_based_free_lists.size() <= log2_size)
      size_based_free_lists.resize(log2_size+1, SENTINEL);
    if (SORTED) {
      // Insert the range into the list such that it maintains a sorted
      // list from smallest to largest
      unsigned prev = SENTINEL;
      unsigned next = size_based_free_lists[log2_size];
      while (next != SENTINEL) {
        Range &next_range = this->ranges[next]; 
        RT next_size = next_range.last - next_range.first;
        if (size <= next_size) {
          // We can insert this here and we're done
          range.prev_free = next_range.prev_free;
          if (range.prev_free == SENTINEL)
            size_based_free_lists[log2_size] = index;
          else
            this->ranges[range.prev_free].next_free = index;
          range.next_free = next;
          next_range.prev_free = index;
          return;
        }
        // Step to the next entry
        prev = next;
        next = next_range.next_free;
      }
      // If we get here then we're adding ourselves to the end of the list
      range.prev_free = prev;
      range.next_free = SENTINEL;
      if (prev == SENTINEL)
        size_based_free_lists[log2_size] = index;
      else
        this->ranges[prev].next_free = index;
    } else {
      // Not sorted so just add it to the front of the free list
      range.prev_free = SENTINEL;
      range.next_free = size_based_free_lists[log2_size];
      if (range.next_free != SENTINEL)
        this->ranges[range.next_free].prev_free = index;
      size_based_free_lists[log2_size] = index;
    }
  }

  template <typename RT, typename TT, bool SORTED>
  void SizedRangeAllocator<RT,TT,SORTED>::remove_from_free_list(unsigned index, Range& range)
  {
    if (range.prev_free != SENTINEL) {
      if (range.next_free != SENTINEL) {
        // Remove an item in the middle of the list
        this->ranges[range.prev_free].next_free = range.next_free;
        this->ranges[range.next_free].prev_free = range.prev_free;
      } else // last item in the list can just be removed
        this->ranges[range.prev_free].next_free = SENTINEL;
    } else {
      // We're the first item in the list
      RT size = range.last - range.first;
      unsigned log2_size = floor_log2(size);
      REALM_ASSERT(log2_size < size_based_free_lists.size());
      REALM_ASSERT(size_based_free_lists[log2_size] == index);
      if (range.next_free != SENTINEL)
        this->ranges[range.next_free].prev_free = SENTINEL;
      size_based_free_lists[log2_size] = range.next_free;
    }
  }

  template <typename RT, typename TT, bool SORTED>
  void SizedRangeAllocator<RT,TT,SORTED>::grow_hole(unsigned index, Range& range,
                                                    RT bound, bool before)
  {
    // Check to see if it is going to change bin sizes
    unsigned old_bin = floor_log2(range.last - range.first);
    RT new_size = (before ? range.last : bound) - (before ? bound : range.first);
    unsigned new_bin = floor_log2(new_size);
    if (old_bin == new_bin) {
      if (before)
        range.first = bound;
      else
        range.last = bound;
      // Scan up the list until we've inserted ourselves
      if (SORTED) {
        // Bubble ourselves up the free list
        while (range.next_free != SENTINEL) {
          unsigned next_index = range.next_free;
          Range &next_range = this->ranges[next_index]; 
          RT next_size = next_range.last - next_range.first;
          if (new_size <= next_size)
            break;
          // Swap places with the next range
          if (range.prev_free != SENTINEL)
            this->ranges[range.prev_free].next_free = next_index;
          if (next_range.next_free != SENTINEL)
            this->ranges[next_range.next_free].prev_free = index;
          next_range.prev_free = range.prev_free;
          range.next_free = next_range.next_free;
          range.prev_free = next_index;
          next_range.next_free = index;
          // Make sure to handle the case where we're the first entry
          // in the free lists
          if (size_based_free_lists[old_bin] == index)
            size_based_free_lists[old_bin] = next_index;
        }
      }
    } else {
      remove_from_free_list(index, range);
      if (before)
        range.first = bound;
      else
        range.last = bound;
      add_to_free_list(index, range);
    }
  }

  template <typename RT, typename TT, bool SORTED>
  /*static*/ unsigned SizedRangeAllocator<RT,TT,SORTED>::floor_log2(uint64_t size)
  {
    // size should be non-zero
    REALM_ASSERT(size);
    // Round down to the nearest power of two to figure out which range
    // to put it in using DeBruijin algorithm to compute integer log2
    // Taken from Hacker's Delight
    static const unsigned tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5 };
    uint64_t value = size;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
  }
    
}; // namespace Realm

#endif // ifndef REALM_MEM_IMPL_INL
