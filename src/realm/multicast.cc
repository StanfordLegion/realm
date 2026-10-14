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

// Adaptive multicast target sets and encodings for Realm active messages, plus the
//  multicast envelope and the bounded-radix forwarding algorithm built on top of them

#include "realm/multicast.h"

#include "realm/logging.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ostream>
#include <sstream>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // multicast target encodings
  //

  const char *multicast_target_encoding_name(MulticastTargetEncoding kind)
  {
    switch(kind) {
    case MulticastTargetEncoding::EMPTY:
      return "EMPTY";
    case MulticastTargetEncoding::SINGLE:
      return "SINGLE";
    case MulticastTargetEncoding::SMALL_INLINE:
      return "SMALL_INLINE";
    case MulticastTargetEncoding::RANGES:
      return "RANGES";
    case MulticastTargetEncoding::DELTA_LIST:
      return "DELTA_LIST";
    case MulticastTargetEncoding::BITMAP:
      return "BITMAP";
    case MulticastTargetEncoding::ALL_NODES:
      return "ALL_NODES";
    case MulticastTargetEncoding::ALL_EXCEPT:
      return "ALL_EXCEPT";
    }
    return "INVALID";
  }

  std::ostream &operator<<(std::ostream &os, MulticastTargetEncoding kind)
  {
    os << multicast_target_encoding_name(kind);
    return os;
  }

  const char *multicast_decode_status_name(MulticastDecodeStatus status)
  {
    switch(status) {
    case MulticastDecodeStatus::OK:
      return "OK";
    case MulticastDecodeStatus::UNKNOWN_KIND:
      return "UNKNOWN_KIND";
    case MulticastDecodeStatus::TRUNCATED:
      return "TRUNCATED";
    case MulticastDecodeStatus::TRAILING_BYTES:
      return "TRAILING_BYTES";
    case MulticastDecodeStatus::BAD_CARDINALITY:
      return "BAD_CARDINALITY";
    case MulticastDecodeStatus::NODE_OUT_OF_RANGE:
      return "NODE_OUT_OF_RANGE";
    case MulticastDecodeStatus::RANGE_OVERFLOW:
      return "RANGE_OVERFLOW";
    case MulticastDecodeStatus::NOT_CANONICAL:
      return "NOT_CANONICAL";
    }
    return "INVALID";
  }

  std::ostream &operator<<(std::ostream &os, MulticastDecodeStatus status)
  {
    os << multicast_decode_status_name(status);
    return os;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // wire primitives
  //

  namespace MulticastWire {

    size_t varint_size(uint64_t value)
    {
      size_t bytes = 1;
      while(value >= 0x80) {
        value >>= 7;
        bytes++;
      }
      return bytes;
    }

    void append_varint(std::vector<unsigned char> &buf, uint64_t value)
    {
      while(value >= 0x80) {
        buf.push_back(static_cast<unsigned char>((value & 0x7f) | 0x80));
        value >>= 7;
      }
      buf.push_back(static_cast<unsigned char>(value));
    }

    MulticastDecodeStatus read_varint(const unsigned char *base, size_t bytes,
                                      size_t &pos, uint64_t &value)
    {
      uint64_t result = 0;
      unsigned shift = 0;
      size_t consumed = 0;
      while(true) {
        if(pos + consumed >= bytes)
          return MulticastDecodeStatus::TRUNCATED;
        unsigned char b = base[pos + consumed];
        consumed++;
        // an encoding longer than MAX_VARINT_BYTES, or one whose top byte would shift
        //  bits out of a 64-bit value, is not something our encoder can produce
        if((consumed > MAX_VARINT_BYTES) || ((shift == 63) && ((b & 0x7f) > 1)))
          return MulticastDecodeStatus::NOT_CANONICAL;
        result |= (static_cast<uint64_t>(b & 0x7f) << shift);
        if((b & 0x80) == 0) {
          // reject overlong encodings so that the byte string is canonical
          if((consumed > 1) && (b == 0))
            return MulticastDecodeStatus::NOT_CANONICAL;
          break;
        }
        shift += 7;
      }
      pos += consumed;
      value = result;
      return MulticastDecodeStatus::OK;
    }

  }; // namespace MulticastWire

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastTargetSet::const_iterator
  //

  MulticastTargetSet::const_iterator::const_iterator(const MulticastTargetSet &_set,
                                                     size_t _run_idx)
    : set(&_set)
    , run_idx(_run_idx)
    , cur_node(-1)
  {
    if(run_idx < set->runs.size())
      cur_node = set->runs[run_idx].first;
  }

  bool MulticastTargetSet::const_iterator::operator==(const const_iterator &rhs) const
  {
    return (set == rhs.set) && (run_idx == rhs.run_idx) && (cur_node == rhs.cur_node);
  }

  MulticastTargetSet::const_iterator &MulticastTargetSet::const_iterator::operator++(void)
  {
    assert((set != nullptr) && (run_idx < set->runs.size()));
    if(cur_node < set->runs[run_idx].last) {
      cur_node++;
    } else {
      run_idx++;
      cur_node = (run_idx < set->runs.size()) ? set->runs[run_idx].first : -1;
    }
    return *this;
  }

  MulticastTargetSet::const_iterator
  MulticastTargetSet::const_iterator::operator++(int /*postfix*/)
  {
    const_iterator old = *this;
    ++(*this);
    return old;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastTargetSet
  //

  MulticastTargetSet::MulticastTargetSet(const NodeSet &nodes) { add_nodeset(nodes); }

  void MulticastTargetSet::clear(void)
  {
    runs.clear();
    total = 0;
  }

  void MulticastTargetSet::add(NodeID id) { add_range(id, id); }

  void MulticastTargetSet::add_range(NodeID first, NodeID last)
  {
    if(last < first)
      return; // empty range - nothing to do
    assert(first >= 0);

    // all of the +/-1 comparisons below are done in 64 bits so that a range touching
    //  the top of the NodeID space cannot overflow
    const long long lo = first;
    const long long hi = last;

    // first run that could possibly merge with or follow [first, last]
    size_t left = 0, right = runs.size();
    while(left < right) {
      size_t mid = left + ((right - left) >> 1);
      if((static_cast<long long>(runs[mid].last) + 1) < lo)
        left = mid + 1;
      else
        right = mid;
    }

    // absorb every run that touches or is adjacent to the new one
    size_t end = left;
    NodeID merged_first = first;
    NodeID merged_last = last;
    size_t absorbed = 0;
    while((end < runs.size()) && (static_cast<long long>(runs[end].first) <= (hi + 1))) {
      merged_first = std::min(merged_first, runs[end].first);
      merged_last = std::max(merged_last, runs[end].last);
      absorbed += runs[end].count();
      end++;
    }

    Range merged;
    merged.first = merged_first;
    merged.last = merged_last;
    const size_t merged_count = merged.count();

    runs.erase(runs.begin() + left, runs.begin() + end);
    runs.insert(runs.begin() + left, merged);

    total += merged_count - absorbed;
  }

  void MulticastTargetSet::add_nodeset(const NodeSet &nodes)
  {
    // NOTE: NodeSet iteration is not guaranteed to be sorted, which is fine - add()
    //  handles arbitrary insertion order
    for(NodeSet::const_iterator it = nodes.begin(); it != nodes.end(); ++it)
      add(*it);
  }

  void MulticastTargetSet::add_targets(const MulticastTargetSet &other)
  {
    for(size_t i = 0; i < other.runs.size(); i++)
      add_range(other.runs[i].first, other.runs[i].last);
  }

  bool MulticastTargetSet::remove(NodeID id)
  {
    size_t idx = find_run(id);
    if(idx >= runs.size())
      return false;

    Range &r = runs[idx];
    if((r.first == id) && (r.last == id)) {
      runs.erase(runs.begin() + idx);
    } else if(r.first == id) {
      r.first = id + 1;
    } else if(r.last == id) {
      r.last = id - 1;
    } else {
      // split into two runs, neither of which can be empty
      Range tail;
      tail.first = id + 1;
      tail.last = r.last;
      r.last = id - 1;
      runs.insert(runs.begin() + idx + 1, tail);
    }
    total--;
    return true;
  }

  bool MulticastTargetSet::append_increasing_node(NodeID id)
  {
    if(id < 0)
      return false;
    if(runs.empty()) {
      Range r;
      r.first = id;
      r.last = id;
      runs.push_back(r);
      total++;
      return true;
    }
    if(id <= runs.back().last)
      return false;
    if(id == (runs.back().last + 1))
      runs.back().last = id;
    else {
      Range r;
      r.first = id;
      r.last = id;
      runs.push_back(r);
    }
    total++;
    return true;
  }

  bool MulticastTargetSet::append_canonical_run(NodeID first, NodeID last)
  {
    if((first < 0) || (last < first))
      return false;
    // must be strictly after the trailing run AND not adjacent to it, otherwise the
    //  incoming encoding was not canonical
    if(!runs.empty() &&
       (static_cast<long long>(first) <= (static_cast<long long>(runs.back().last) + 1)))
      return false;
    Range r;
    r.first = first;
    r.last = last;
    runs.push_back(r);
    total += r.count();
    return true;
  }

  size_t MulticastTargetSet::find_run(NodeID id) const
  {
    size_t left = 0, right = runs.size();
    while(left < right) {
      size_t mid = left + ((right - left) >> 1);
      if(runs[mid].last < id)
        left = mid + 1;
      else
        right = mid;
    }
    if((left < runs.size()) && (runs[left].first <= id) && (id <= runs[left].last))
      return left;
    return runs.size();
  }

  bool MulticastTargetSet::contains(NodeID id) const
  {
    return (find_run(id) < runs.size());
  }

  bool MulticastTargetSet::fits_node_count(NodeID num_nodes) const
  {
    if(runs.empty())
      return true;
    return ((runs.front().first >= 0) && (runs.back().last < num_nodes));
  }

  void MulticastTargetSet::to_nodeset(NodeSet &nodes) const
  {
    for(size_t i = 0; i < runs.size(); i++)
      nodes.add_range(runs[i].first, runs[i].last);
  }

  void MulticastTargetSet::partition(size_t max_slices,
                                     std::vector<MulticastTargetSet> &slices) const
  {
    slices.clear();
    if((total == 0) || (max_slices == 0))
      return;

    const size_t num_slices = std::min(max_slices, total);
    const size_t base = total / num_slices;
    const size_t extra = total % num_slices;

    slices.resize(num_slices);

    size_t run_idx = 0;
    NodeID cursor = runs[0].first;
    for(size_t s = 0; s < num_slices; s++) {
      size_t wanted = base + ((s < extra) ? 1 : 0);
      MulticastTargetSet &slice = slices[s];
      while(wanted > 0) {
        assert(run_idx < runs.size());
        const Range &r = runs[run_idx];
        const size_t avail =
            static_cast<size_t>(static_cast<long long>(r.last) - cursor + 1);
        const size_t taken = std::min(avail, wanted);
        // NOTE: we cut runs here, we never expand them - a run covering thousands of
        //  nodes costs one entry per slice it lands in
        const NodeID slice_last =
            static_cast<NodeID>(static_cast<long long>(cursor) + taken - 1);
        bool appended = slice.append_canonical_run(cursor, slice_last);
        assert(appended);
        (void)appended;
        wanted -= taken;
        if(taken == avail) {
          run_idx++;
          if(run_idx < runs.size())
            cursor = runs[run_idx].first;
        } else {
          cursor = static_cast<NodeID>(static_cast<long long>(cursor) + taken);
        }
      }
    }
  }

  bool MulticastTargetSet::operator==(const MulticastTargetSet &rhs) const
  {
    return (total == rhs.total) && (runs == rhs.runs);
  }

  std::ostream &operator<<(std::ostream &os, const MulticastTargetSet &targets)
  {
    os << "{";
    const std::vector<MulticastTargetSet::Range> &runs = targets.ranges();
    for(size_t i = 0; i < runs.size(); i++) {
      if(i > 0)
        os << ",";
      if(runs[i].first == runs[i].last)
        os << runs[i].first;
      else
        os << runs[i].first << "-" << runs[i].last;
    }
    os << "}";
    return os;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastEncodingTally
  //

  MulticastEncodingTally::MulticastEncodingTally(void)
  {
    for(size_t i = 0; i < MULTICAST_ENCODING_KINDS; i++)
      counts[i].store(0);
  }

  void MulticastEncodingTally::record(MulticastTargetEncoding kind)
  {
    size_t idx = static_cast<size_t>(kind);
    assert(idx < MULTICAST_ENCODING_KINDS);
    counts[idx].fetch_add(1);
  }

  uint64_t MulticastEncodingTally::get(MulticastTargetEncoding kind) const
  {
    size_t idx = static_cast<size_t>(kind);
    assert(idx < MULTICAST_ENCODING_KINDS);
    return counts[idx].load();
  }

  uint64_t MulticastEncodingTally::total(void) const
  {
    uint64_t sum = 0;
    for(size_t i = 0; i < MULTICAST_ENCODING_KINDS; i++)
      sum += counts[i].load();
    return sum;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class EncodedMulticastTargets
  //

  namespace {

    typedef MulticastTargetSet::Range Range;

    // The complement of 'runs' within [0, num_nodes), itself canonical.  This is
    //  O(runs.size()) - the complement of a small set of a huge node space is still
    //  only a couple of runs, which is what keeps the ALL_EXCEPT candidate cheap to
    //  evaluate.
    void complement_runs(const std::vector<Range> &runs, NodeID num_nodes,
                         std::vector<Range> &out)
    {
      out.clear();
      NodeID next = 0;
      for(size_t i = 0; i < runs.size(); i++) {
        if(runs[i].first > next) {
          Range r;
          r.first = next;
          r.last = runs[i].first - 1;
          out.push_back(r);
        }
        next = runs[i].last + 1;
      }
      if(next < num_nodes) {
        Range r;
        r.first = next;
        r.last = num_nodes - 1;
        out.push_back(r);
      }
    }

    // Bytes needed for "first node, then one positive varint delta per remaining node",
    //  computed from the runs rather than by walking individual nodes: inside a run
    //  every delta is 1 and therefore exactly one byte.
    uint64_t delta_list_bytes(const std::vector<Range> &runs)
    {
      uint64_t bytes = 0;
      long long prev = -1;
      for(size_t i = 0; i < runs.size(); i++) {
        if(prev < 0)
          bytes += MulticastWire::varint_size(static_cast<uint64_t>(runs[i].first));
        else
          bytes += MulticastWire::varint_size(
              static_cast<uint64_t>(static_cast<long long>(runs[i].first) - prev));
        bytes += runs[i].count() - 1; // interior deltas are all 1, i.e. one byte each
        prev = runs[i].last;
      }
      return bytes;
    }

    void emit_delta_list(std::vector<unsigned char> &buf, const std::vector<Range> &runs)
    {
      long long prev = -1;
      for(size_t i = 0; i < runs.size(); i++) {
        if(prev < 0)
          MulticastWire::append_varint(buf, static_cast<uint64_t>(runs[i].first));
        else
          MulticastWire::append_varint(
              buf, static_cast<uint64_t>(static_cast<long long>(runs[i].first) - prev));
        for(long long n = runs[i].first + 1; n <= runs[i].last; n++)
          MulticastWire::append_varint(buf, 1);
        prev = runs[i].last;
      }
    }

    // reads "count nodes as first-plus-positive-deltas" into 'out'
    MulticastDecodeStatus read_delta_list(const unsigned char *base, size_t bytes,
                                          size_t &pos, uint64_t count,
                                          uint64_t node_limit, MulticastTargetSet &out)
    {
      uint64_t cur = 0;
      MulticastDecodeStatus status = MulticastWire::read_varint(base, bytes, pos, cur);
      if(status != MulticastDecodeStatus::OK)
        return status;
      if(cur >= node_limit)
        return MulticastDecodeStatus::NODE_OUT_OF_RANGE;
      if(!out.append_increasing_node(static_cast<NodeID>(cur)))
        return MulticastDecodeStatus::NOT_CANONICAL;

      for(uint64_t i = 1; i < count; i++) {
        uint64_t delta = 0;
        status = MulticastWire::read_varint(base, bytes, pos, delta);
        if(status != MulticastDecodeStatus::OK)
          return status;
        // a zero delta would repeat the previous node, which is not canonical
        if(delta == 0)
          return MulticastDecodeStatus::NOT_CANONICAL;
        if(delta > (node_limit - 1 - cur))
          return MulticastDecodeStatus::NODE_OUT_OF_RANGE;
        cur += delta;
        if(!out.append_increasing_node(static_cast<NodeID>(cur)))
          return MulticastDecodeStatus::NOT_CANONICAL;
      }
      return MulticastDecodeStatus::OK;
    }

  }; // namespace

  EncodedMulticastTargets::EncodedMulticastTargets(void)
    : chosen(MulticastTargetEncoding::EMPTY)
    , count(0)
  {
    wire.push_back(static_cast<unsigned char>(MulticastTargetEncoding::EMPTY));
  }

  size_t EncodedMulticastTargets::encoded_size(const MulticastTargetSet &targets,
                                               NodeID num_nodes,
                                               MulticastTargetEncoding kind)
  {
    const std::vector<Range> &runs = targets.ranges();
    const size_t n = targets.size();

    switch(kind) {
    case MulticastTargetEncoding::EMPTY:
    {
      return (n == 0) ? 1 : 0;
    }

    case MulticastTargetEncoding::SINGLE:
    {
      if(n != 1)
        return 0;
      return 1 + MulticastWire::varint_size(static_cast<uint64_t>(targets.first_node()));
    }

    case MulticastTargetEncoding::SMALL_INLINE:
    {
      if((n < 1) || (n > MULTICAST_MAX_SMALL_INLINE))
        return 0;
      size_t bytes = 1 + MulticastWire::varint_size(n);
      // n is at most MULTICAST_MAX_SMALL_INLINE, so walking nodes here is bounded
      for(MulticastTargetSet::const_iterator it = targets.begin(); it != targets.end();
          ++it)
        bytes += MulticastWire::varint_size(static_cast<uint64_t>(*it));
      return bytes;
    }

    case MulticastTargetEncoding::RANGES:
    {
      if(n < 1)
        return 0;
      size_t bytes = 1 + MulticastWire::varint_size(runs.size());
      for(size_t i = 0; i < runs.size(); i++)
        bytes += MulticastWire::varint_size(static_cast<uint64_t>(runs[i].first)) +
                 MulticastWire::varint_size(runs[i].count());
      return bytes;
    }

    case MulticastTargetEncoding::DELTA_LIST:
    {
      if(n < 1)
        return 0;
      return 1 + MulticastWire::varint_size(n) +
             static_cast<size_t>(delta_list_bytes(runs));
    }

    case MulticastTargetEncoding::BITMAP:
    {
      if(n < 1)
        return 0;
      const uint64_t base = static_cast<uint64_t>(targets.first_node());
      const uint64_t bit_length = static_cast<uint64_t>(targets.last_node()) - base + 1;
      return 1 + MulticastWire::varint_size(base) +
             MulticastWire::varint_size(bit_length) +
             static_cast<size_t>((bit_length + 7) / 8);
    }

    case MulticastTargetEncoding::ALL_NODES:
    {
      if((num_nodes < 1) || (n != static_cast<size_t>(num_nodes)) ||
         !targets.fits_node_count(num_nodes))
        return 0;
      return 1;
    }

    case MulticastTargetEncoding::ALL_EXCEPT:
    {
      // must exclude at least one node (otherwise ALL_NODES) and must leave at least
      //  one node in the set (otherwise EMPTY)
      if((num_nodes < 1) || (n < 1) || (n >= static_cast<size_t>(num_nodes)))
        return 0;
      if(!targets.fits_node_count(num_nodes))
        return 0;
      const size_t excluded = static_cast<size_t>(num_nodes) - n;
      std::vector<Range> comp;
      complement_runs(runs, num_nodes, comp);
      return 1 + MulticastWire::varint_size(excluded) +
             static_cast<size_t>(delta_list_bytes(comp));
    }
    }
    return 0;
  }

  void EncodedMulticastTargets::emit(std::vector<unsigned char> &buf,
                                     const MulticastTargetSet &targets, NodeID num_nodes,
                                     MulticastTargetEncoding kind)
  {
    const std::vector<Range> &runs = targets.ranges();
    const size_t n = targets.size();

    buf.push_back(static_cast<unsigned char>(kind));

    switch(kind) {
    case MulticastTargetEncoding::EMPTY:
    case MulticastTargetEncoding::ALL_NODES:
    {
      break;
    }

    case MulticastTargetEncoding::SINGLE:
    {
      MulticastWire::append_varint(buf, static_cast<uint64_t>(targets.first_node()));
      break;
    }

    case MulticastTargetEncoding::SMALL_INLINE:
    {
      MulticastWire::append_varint(buf, n);
      for(MulticastTargetSet::const_iterator it = targets.begin(); it != targets.end();
          ++it)
        MulticastWire::append_varint(buf, static_cast<uint64_t>(*it));
      break;
    }

    case MulticastTargetEncoding::RANGES:
    {
      MulticastWire::append_varint(buf, runs.size());
      for(size_t i = 0; i < runs.size(); i++) {
        MulticastWire::append_varint(buf, static_cast<uint64_t>(runs[i].first));
        MulticastWire::append_varint(buf, runs[i].count());
      }
      break;
    }

    case MulticastTargetEncoding::DELTA_LIST:
    {
      MulticastWire::append_varint(buf, n);
      emit_delta_list(buf, runs);
      break;
    }

    case MulticastTargetEncoding::BITMAP:
    {
      const uint64_t base = static_cast<uint64_t>(targets.first_node());
      const uint64_t bit_length = static_cast<uint64_t>(targets.last_node()) - base + 1;
      MulticastWire::append_varint(buf, base);
      MulticastWire::append_varint(buf, bit_length);
      const size_t map_bytes = static_cast<size_t>((bit_length + 7) / 8);
      const size_t map_start = buf.size();
      buf.resize(map_start + map_bytes, 0);
      for(size_t i = 0; i < runs.size(); i++)
        for(long long node = runs[i].first; node <= runs[i].last; node++) {
          const uint64_t bit = static_cast<uint64_t>(node) - base;
          unsigned char &byte = buf[map_start + static_cast<size_t>(bit >> 3)];
          byte = static_cast<unsigned char>(byte | (1u << (bit & 7)));
        }
      break;
    }

    case MulticastTargetEncoding::ALL_EXCEPT:
    {
      const size_t excluded = static_cast<size_t>(num_nodes) - n;
      std::vector<Range> comp;
      complement_runs(runs, num_nodes, comp);
      MulticastWire::append_varint(buf, excluded);
      emit_delta_list(buf, comp);
      break;
    }
    }
  }

  /*static*/ EncodedMulticastTargets
  EncodedMulticastTargets::encode(const MulticastTargetSet &targets, NodeID num_nodes,
                                  MulticastEncodingTally *tally)
  {
    assert(targets.fits_node_count(num_nodes));

    // evaluate the ACTUAL serialized length of every candidate and keep the smallest -
    //  a fixed density heuristic is explicitly not allowed here (plan section 7.2)
    size_t best_size = 0;
    MulticastTargetEncoding best = MulticastTargetEncoding::EMPTY;
    for(size_t i = 0; i < MULTICAST_ENCODING_KINDS; i++) {
      MulticastTargetEncoding kind = static_cast<MulticastTargetEncoding>(i);
      size_t sz = encoded_size(targets, num_nodes, kind);
      if(sz == 0)
        continue; // this kind cannot represent this set
      // strict '<' leaves the earliest (cheapest to decode) kind winning a tie
      if((best_size == 0) || (sz < best_size)) {
        best_size = sz;
        best = kind;
      }
    }
    // EMPTY is always available for an empty set and RANGES for a nonempty one
    assert(best_size > 0);

    EncodedMulticastTargets result;
    result.wire.clear();
    result.wire.reserve(best_size);
    emit(result.wire, targets, num_nodes, best);
    assert(result.wire.size() == best_size);
    result.chosen = best;
    result.count = targets.size();

    if(tally != nullptr)
      tally->record(best);

    return result;
  }

  /*static*/ MulticastDecodeStatus
  EncodedMulticastTargets::decode(const void *data, size_t bytes, NodeID num_nodes,
                                  MulticastTargetSet &targets)
  {
    targets.clear();

    if(bytes < 1)
      return MulticastDecodeStatus::TRUNCATED;
    const unsigned char *base = static_cast<const unsigned char *>(data);
    const unsigned char kind_byte = base[0];
    if(kind_byte >= MULTICAST_ENCODING_KINDS)
      return MulticastDecodeStatus::UNKNOWN_KIND;
    const MulticastTargetEncoding kind = static_cast<MulticastTargetEncoding>(kind_byte);

    // a negative or zero node count can only ever describe an empty machine
    const uint64_t node_limit =
        (num_nodes > 0) ? static_cast<uint64_t>(num_nodes) : uint64_t(0);

    size_t pos = 1;
    MulticastDecodeStatus status = MulticastDecodeStatus::OK;

    switch(kind) {
    case MulticastTargetEncoding::EMPTY:
    {
      break;
    }

    case MulticastTargetEncoding::SINGLE:
    {
      uint64_t node = 0;
      status = MulticastWire::read_varint(base, bytes, pos, node);
      if(status != MulticastDecodeStatus::OK)
        break;
      if(node >= node_limit) {
        status = MulticastDecodeStatus::NODE_OUT_OF_RANGE;
        break;
      }
      targets.append_increasing_node(static_cast<NodeID>(node));
      break;
    }

    case MulticastTargetEncoding::SMALL_INLINE:
    {
      uint64_t n = 0;
      status = MulticastWire::read_varint(base, bytes, pos, n);
      if(status != MulticastDecodeStatus::OK)
        break;
      // NOTE: the transmitted cardinality is checked against both the configured node
      //  count and the number of bytes actually left before it is used for anything
      //  (plan section 22) - every element needs at least one byte
      if((n < 1) || (n > MULTICAST_MAX_SMALL_INLINE) || (n > node_limit) ||
         (n > (bytes - pos))) {
        status = MulticastDecodeStatus::BAD_CARDINALITY;
        break;
      }
      for(uint64_t i = 0; i < n; i++) {
        uint64_t node = 0;
        status = MulticastWire::read_varint(base, bytes, pos, node);
        if(status != MulticastDecodeStatus::OK)
          break;
        if(node >= node_limit) {
          status = MulticastDecodeStatus::NODE_OUT_OF_RANGE;
          break;
        }
        if(!targets.append_increasing_node(static_cast<NodeID>(node))) {
          status = MulticastDecodeStatus::NOT_CANONICAL;
          break;
        }
      }
      break;
    }

    case MulticastTargetEncoding::RANGES:
    {
      uint64_t num_runs = 0;
      status = MulticastWire::read_varint(base, bytes, pos, num_runs);
      if(status != MulticastDecodeStatus::OK)
        break;
      // each run needs at least two bytes, and canonical runs are separated by a gap,
      //  so at most ceil(num_nodes/2) of them can fit in the node space
      if((num_runs < 1) || (num_runs > ((bytes - pos) / 2)) ||
         (num_runs > ((node_limit + 1) / 2))) {
        status = MulticastDecodeStatus::BAD_CARDINALITY;
        break;
      }
      for(uint64_t i = 0; i < num_runs; i++) {
        uint64_t start = 0, length = 0;
        status = MulticastWire::read_varint(base, bytes, pos, start);
        if(status != MulticastDecodeStatus::OK)
          break;
        status = MulticastWire::read_varint(base, bytes, pos, length);
        if(status != MulticastDecodeStatus::OK)
          break;
        if(start >= node_limit) {
          status = MulticastDecodeStatus::NODE_OUT_OF_RANGE;
          break;
        }
        // checked this way round so that a huge length cannot wrap
        if((length < 1) || (length > (node_limit - start))) {
          status = MulticastDecodeStatus::RANGE_OVERFLOW;
          break;
        }
        const NodeID first = static_cast<NodeID>(start);
        const NodeID last = static_cast<NodeID>(start + length - 1);
        // rejects unsorted, duplicated, overlapping and adjacent-but-unmerged runs
        if(!targets.append_canonical_run(first, last)) {
          status = MulticastDecodeStatus::NOT_CANONICAL;
          break;
        }
      }
      break;
    }

    case MulticastTargetEncoding::DELTA_LIST:
    {
      uint64_t n = 0;
      status = MulticastWire::read_varint(base, bytes, pos, n);
      if(status != MulticastDecodeStatus::OK)
        break;
      if((n < 1) || (n > node_limit) || (n > (bytes - pos))) {
        status = MulticastDecodeStatus::BAD_CARDINALITY;
        break;
      }
      status = read_delta_list(base, bytes, pos, n, node_limit, targets);
      break;
    }

    case MulticastTargetEncoding::BITMAP:
    {
      uint64_t bitmap_base = 0, bit_length = 0;
      status = MulticastWire::read_varint(base, bytes, pos, bitmap_base);
      if(status != MulticastDecodeStatus::OK)
        break;
      status = MulticastWire::read_varint(base, bytes, pos, bit_length);
      if(status != MulticastDecodeStatus::OK)
        break;
      if(bitmap_base >= node_limit) {
        status = MulticastDecodeStatus::NODE_OUT_OF_RANGE;
        break;
      }
      if((bit_length < 1) || (bit_length > (node_limit - bitmap_base))) {
        status = MulticastDecodeStatus::RANGE_OVERFLOW;
        break;
      }
      const uint64_t map_bytes = (bit_length + 7) / 8;
      if(map_bytes != (bytes - pos)) {
        status = (map_bytes > (bytes - pos)) ? MulticastDecodeStatus::TRUNCATED
                                             : MulticastDecodeStatus::TRAILING_BYTES;
        break;
      }
      // canonical form pins the first and last bits and zeroes the padding
      const unsigned char last_byte = base[pos + map_bytes - 1];
      const unsigned pad_bits = static_cast<unsigned>((map_bytes * 8) - bit_length);
      if(((base[pos] & 1) == 0) || ((last_byte & (1u << ((bit_length - 1) & 7))) == 0) ||
         ((pad_bits > 0) && ((last_byte >> (8 - pad_bits)) != 0))) {
        status = MulticastDecodeStatus::NOT_CANONICAL;
        break;
      }
      for(uint64_t bit = 0; bit < bit_length; bit++)
        if((base[pos + (bit >> 3)] & (1u << (bit & 7))) != 0)
          targets.append_increasing_node(static_cast<NodeID>(bitmap_base + bit));
      pos += map_bytes;
      break;
    }

    case MulticastTargetEncoding::ALL_NODES:
    {
      if(node_limit < 1) {
        status = MulticastDecodeStatus::BAD_CARDINALITY;
        break;
      }
      targets.append_canonical_run(0, num_nodes - 1);
      break;
    }

    case MulticastTargetEncoding::ALL_EXCEPT:
    {
      uint64_t excluded = 0;
      status = MulticastWire::read_varint(base, bytes, pos, excluded);
      if(status != MulticastDecodeStatus::OK)
        break;
      // excluding every node would be an empty set, which is EMPTY's job
      if((excluded < 1) || (excluded >= node_limit) || (excluded > (bytes - pos))) {
        status = MulticastDecodeStatus::BAD_CARDINALITY;
        break;
      }
      MulticastTargetSet exclusions;
      status = read_delta_list(base, bytes, pos, excluded, node_limit, exclusions);
      if(status != MulticastDecodeStatus::OK)
        break;
      // complement of the exclusion set within [0, num_nodes)
      NodeID next = 0;
      const std::vector<Range> &excl_runs = exclusions.ranges();
      for(size_t i = 0; i < excl_runs.size(); i++) {
        if(excl_runs[i].first > next)
          targets.append_canonical_run(next, excl_runs[i].first - 1);
        next = excl_runs[i].last + 1;
      }
      if(next < num_nodes)
        targets.append_canonical_run(next, num_nodes - 1);
      break;
    }
    }

    if(status != MulticastDecodeStatus::OK) {
      targets.clear();
      return status;
    }
    if(pos != bytes) {
      targets.clear();
      return MulticastDecodeStatus::TRAILING_BYTES;
    }
    return MulticastDecodeStatus::OK;
  }

}; // namespace Realm
