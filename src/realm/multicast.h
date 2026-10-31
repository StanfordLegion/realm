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

// Adaptive multicast target sets and encodings for Realm active messages - see
//  sections 7.1, 7.2 and 7.3 of SCALABLE_BARRIERS_IMPLEMENTATION_PLAN.md.
//
// The logical target set and its wire encoding are deliberately separate types:
//
//    MulticastTargetSet       canonical logical set / builder / view
//    EncodedMulticastTargets  immutable wire representation
//
// This layer sits strictly above backend unicast (plan section 7.1) and deliberately
//  has no dependency on activemsg.h, so it can be unit tested standalone.

#ifndef REALM_MULTICAST_H
#define REALM_MULTICAST_H

#include "realm/atomics.h"
#include "realm/nodeset.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <vector>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // multicast target encodings (plan section 7.2)
  //

  // The eight representations the encoder chooses between.  The numeric values are the
  //  on-the-wire kind byte and must not be renumbered.  The encoder computes the actual
  //  serialized size of every candidate and takes the smallest; ties are broken in
  //  favor of the lower value, which is also the cheaper one to decode.
  enum class MulticastTargetEncoding : unsigned char
  {
    EMPTY = 0,        // no targets at all
    SINGLE = 1,       // exactly one target
    SMALL_INLINE = 2, // short sorted list of absolute node IDs
    RANGES = 3,       // sorted nonoverlapping (start, length) runs
    DELTA_LIST = 4,   // first node followed by positive varint deltas
    BITMAP = 5,       // explicit base node plus an explicit bit length
    ALL_NODES = 6,    // every node in the configured node count
    ALL_EXCEPT = 7,   // every node except a (small) explicit exclusion list
  };

  // number of distinct encodings - candidates are enumerated over [0, KINDS)
  static const size_t MULTICAST_ENCODING_KINDS = 8;

  // longest sorted absolute list that SMALL_INLINE will represent ("small" in 7.2);
  //  larger sparse sets fall through to DELTA_LIST
  static const size_t MULTICAST_MAX_SMALL_INLINE = 8;

  const char *multicast_target_encoding_name(MulticastTargetEncoding kind);

  std::ostream &operator<<(std::ostream &os, MulticastTargetEncoding kind);

  // Why a decode was rejected.  Plan section 21.1 requires "malformed multicast target
  //  encoding" to be a fatal diagnostic, but the decoder itself never aborts: it
  //  returns a status so that the caller can build the full fatal context (and so that
  //  the decoder stays unit testable).
  enum class MulticastDecodeStatus : unsigned char
  {
    OK = 0,
    UNKNOWN_KIND,      // kind byte is not one of the eight
    TRUNCATED,         // payload ended in the middle of a field
    TRAILING_BYTES,    // complete encoding did not consume the whole payload
    BAD_CARDINALITY,   // count is zero where >= 1 is required, or exceeds what the
                       //  payload length / configured node count could possibly hold
    NODE_OUT_OF_RANGE, // a node ID is negative or >= the configured node count
    RANGE_OVERFLOW,    // a run's length is zero, or start+length leaves the node space
    NOT_CANONICAL,     // unsorted, duplicated, overlapping, adjacent-but-unmerged, an
                       //  overlong varint, or a bitmap with slack at either end
  };

  const char *multicast_decode_status_name(MulticastDecodeStatus status);

  std::ostream &operator<<(std::ostream &os, MulticastDecodeStatus status);

  ////////////////////////////////////////////////////////////////////////
  //
  // wire primitives
  //

  // Exposed so that tests can build byte-exact malformed payloads, and so that the
  //  envelope code in the next stage can size fields without duplicating this.
  namespace MulticastWire {

    // LEB128, canonical (no overlong encodings), at most 10 bytes
    static const size_t MAX_VARINT_BYTES = 10;

    size_t varint_size(uint64_t value);

    void append_varint(std::vector<unsigned char> &buf, uint64_t value);

    // advances 'pos' on success; returns TRUNCATED if the buffer ends first and
    //  NOT_CANONICAL for an overlong or overflowing encoding
    MulticastDecodeStatus read_varint(const unsigned char *base, size_t bytes,
                                      size_t &pos, uint64_t &value);

  }; // namespace MulticastWire

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastTargetSet
  //

  // The canonical logical target set (plan section 7.2).  Stored as sorted, disjoint,
  //  nonadjacent runs so that a range covering thousands of nodes costs one entry: this
  //  is what lets partition() and the size estimator run without ever expanding a run
  //  into individual node IDs.
  class MulticastTargetSet {
  public:
    struct Range {
      NodeID first = 0;
      NodeID last = 0; // inclusive

      size_t count(void) const
      {
        return static_cast<size_t>(static_cast<long long>(last) - first + 1);
      }

      bool operator==(const Range &rhs) const
      {
        return (first == rhs.first) && (last == rhs.last);
      }
      bool operator!=(const Range &rhs) const { return !(*this == rhs); }
    };

    MulticastTargetSet(void) = default;
    explicit MulticastTargetSet(const NodeSet &nodes);

    // --- building ------------------------------------------------------

    void clear(void);

    void add(NodeID id);
    void add_range(NodeID first, NodeID last /*inclusive*/);
    void add_nodeset(const NodeSet &nodes);
    void add_targets(const MulticastTargetSet &other);

    // removal of the local relay before forwarding (plan section 7.3) - returns true if
    //  the node was actually present
    bool remove(NodeID id);

    // Appends a single node that must be strictly greater than every node already
    //  present; merges into the trailing run when adjacent so the result stays
    //  canonical.  Returns false (leaving the set unchanged) otherwise - this is the
    //  ordering check the decoder needs, done in O(1).
    bool append_increasing_node(NodeID id);

    // Appends a run that must be strictly greater than, and not adjacent to, everything
    //  already present.  Adjacency is rejected rather than merged so that a decoder can
    //  use this to prove the incoming encoding was canonical.
    bool append_canonical_run(NodeID first, NodeID last /*inclusive*/);

    // --- inspection ----------------------------------------------------

    bool empty(void) const { return runs.empty(); }
    size_t size(void) const { return total; }

    bool contains(NodeID id) const;

    // both require !empty()
    NodeID first_node(void) const { return runs.front().first; }
    NodeID last_node(void) const { return runs.back().last; }

    size_t num_ranges(void) const { return runs.size(); }
    const std::vector<Range> &ranges(void) const { return runs; }

    // every target is in [0, num_nodes)
    bool fits_node_count(NodeID num_nodes) const;

    void to_nodeset(NodeSet &nodes) const;

    // --- sorted iteration ----------------------------------------------

    class const_iterator {
    public:
      typedef std::input_iterator_tag iterator_category;
      typedef NodeID value_type;
      typedef std::ptrdiff_t difference_type;
      typedef const NodeID *pointer;
      typedef const NodeID &reference;

      const_iterator(void) = default;
      const_iterator(const MulticastTargetSet &_set, size_t _run_idx);

      bool operator==(const const_iterator &rhs) const;
      bool operator!=(const const_iterator &rhs) const { return !(*this == rhs); }

      NodeID operator*(void) const { return cur_node; }
      const NodeID *operator->(void) const { return &cur_node; }

      const_iterator &operator++(/*prefix*/);
      const_iterator operator++(int /*postfix*/);

    protected:
      const MulticastTargetSet *set = nullptr;
      size_t run_idx = 0;
      NodeID cur_node = -1;
    };

    const_iterator begin(void) const { return const_iterator(*this, 0); }
    const_iterator end(void) const { return const_iterator(*this, runs.size()); }

    // --- partitioning (plan section 7.3) -------------------------------

    // Splits into at most 'max_slices' slices of nearly equal cardinality (the first
    //  size()%k slices get one extra node).  Runs are cut, never expanded, so a run
    //  covering thousands of nodes costs O(1) per slice it lands in.  Empty slices are
    //  never produced, so an empty set yields no slices and a set smaller than
    //  'max_slices' yields size() singleton slices.
    void partition(size_t max_slices, std::vector<MulticastTargetSet> &slices) const;

    bool operator==(const MulticastTargetSet &rhs) const;
    bool operator!=(const MulticastTargetSet &rhs) const { return !(*this == rhs); }

  protected:
    // index of the run containing 'id', or runs.size() if there is none
    size_t find_run(NodeID id) const;

    std::vector<Range> runs; // sorted, disjoint, nonadjacent
    size_t total = 0;
  };

  std::ostream &operator<<(std::ostream &os, const MulticastTargetSet &targets);

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastEncodingTally
  //

  // Optional per-choice tally (plan section 21.3, "multicast target encoding choices").
  //  This layer has no barrier dependency, so the encoder records here and the barrier
  //  layer mirrors these into BarrierCounters::multicast_encoding_* using kind().
  struct MulticastEncodingTally {
    MulticastEncodingTally(void);

    void record(MulticastTargetEncoding kind);

    uint64_t get(MulticastTargetEncoding kind) const;
    uint64_t total(void) const;

    void reset(void) { *this = MulticastEncodingTally(); }

    atomic<uint64_t> counts[MULTICAST_ENCODING_KINDS];
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastMetricsSink
  //

  // Optional metrics hook for one multicast (plan section 21.3: "multicast target
  //  encoding choices" and "multicast tree depth and first-hop count").  The forwarding
  //  layer in realm/activemsg.h reports into whichever sink the caller supplies
  //  and never touches process-global state; BarrierMulticastMetrics in
  //  realm/barrier_impl.h is the implementation that mirrors these into a barrier's
  //  BarrierCounters.
  //
  // It is declared here, rather than next to the forwarding code, so that a counter
  //  owner does not have to pull in activemsg.h.
  class MulticastMetricsSink {
  public:
    virtual ~MulticastMetricsSink(void);

    // which of the eight encodings was selected for one outbound envelope
    virtual void record_encoding_choice(MulticastTargetEncoding kind) = 0;

    // number of first-hop envelopes this node issued for one multicast - plan section
    //  23 bounds this by the radix
    virtual void record_first_hops(size_t num_first_hops) = 0;

    // hops from the origin of an envelope this node relayed; the origin's own first-hop
    //  envelopes carry depth 1
    virtual void record_tree_depth(unsigned depth) = 0;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class EncodedMulticastTargets
  //

  // The immutable wire form of a target slice.  The byte string always starts with the
  //  kind byte and is entirely self-describing apart from the configured node count,
  //  which both sides must agree on.
  class EncodedMulticastTargets {
  public:
    // default-constructs to a valid EMPTY encoding
    EncodedMulticastTargets(void);

    // Evaluates every representation that can express 'targets', computes each
    //  candidate's ACTUAL serialized length (no density heuristic - plan section 7.2)
    //  and keeps the smallest.  Every target must be in [0, num_nodes).
    static EncodedMulticastTargets encode(const MulticastTargetSet &targets,
                                          NodeID num_nodes,
                                          MulticastEncodingTally *tally = nullptr);

    // Exact serialized length 'kind' would need for 'targets', or 0 if that kind cannot
    //  represent this set.  Runs are never expanded, so this is O(num_ranges()) even
    //  for DELTA_LIST and ALL_EXCEPT.
    static size_t encoded_size(const MulticastTargetSet &targets, NodeID num_nodes,
                               MulticastTargetEncoding kind);

    // Rebuilds the logical set.  'targets' is cleared first and left empty on failure.
    //  Validates the kind, every varint, the cardinality (against both the remaining
    //  payload length and 'num_nodes'), node bounds, run overflow and canonical
    //  ordering before touching 'targets', and never sizes an allocation from an
    //  unvalidated length (plan section 22).
    static MulticastDecodeStatus decode(const void *data, size_t bytes, NodeID num_nodes,
                                        MulticastTargetSet &targets);

    MulticastTargetEncoding kind(void) const { return chosen; }

    const void *data(void) const { return wire.data(); }
    size_t bytes(void) const { return wire.size(); }

    const std::vector<unsigned char> &wire_bytes(void) const { return wire; }

    // cardinality as known locally at encode time - a decoder must never trust a
    //  transmitted cardinality, so this is not on the wire for every kind
    size_t num_targets(void) const { return count; }

    MulticastDecodeStatus decode_into(NodeID num_nodes, MulticastTargetSet &targets) const
    {
      return decode(wire.data(), wire.size(), num_nodes, targets);
    }

  protected:
    static void emit(std::vector<unsigned char> &buf, const MulticastTargetSet &targets,
                     NodeID num_nodes, MulticastTargetEncoding kind);

    std::vector<unsigned char> wire;
    MulticastTargetEncoding chosen = MulticastTargetEncoding::EMPTY;
    size_t count = 0;
  };

}; // namespace Realm

#endif // ifndef REALM_MULTICAST_H
