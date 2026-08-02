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

// Stages 2a, 2b and 2c of SCALABLE_BARRIERS_IMPLEMENTATION_PLAN.md.
//
//  - stage 2a: adaptive multicast target sets and encodings (plan sections 7.2, 20.1)
//  - stage 2b: the multicast envelope, generic handler redispatch and bounded-radix
//    forwarding over unicast (plan sections 7.1, 7.3, 7.4, 20.1)
//  - stage 2c: payload, fragmentation and transient completion semantics (plan
//    sections 7.5, 20.1)

#include "realm/multicast.h"

#include "realm/activemsg.h"
#include "realm/threads.h"
#include "realm/timers.h"

#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <gtest/gtest.h>

using namespace Realm;

namespace {

  typedef std::vector<unsigned char> ByteVec;

  void put_kind(ByteVec &buf, MulticastTargetEncoding kind)
  {
    buf.push_back(static_cast<unsigned char>(kind));
  }

  void put_varint(ByteVec &buf, uint64_t value)
  {
    MulticastWire::append_varint(buf, value);
  }

  MulticastDecodeStatus decode_bytes(const ByteVec &buf, NodeID num_nodes,
                                     MulticastTargetSet &out)
  {
    return EncodedMulticastTargets::decode(buf.data(), buf.size(), num_nodes, out);
  }

  MulticastDecodeStatus decode_bytes(const ByteVec &buf, NodeID num_nodes)
  {
    MulticastTargetSet ignored;
    return decode_bytes(buf, num_nodes, ignored);
  }

  size_t size_of(const MulticastTargetSet &targets, NodeID num_nodes,
                 MulticastTargetEncoding kind)
  {
    return EncodedMulticastTargets::encoded_size(targets, num_nodes, kind);
  }

  // every node in the set, in iteration order
  std::vector<NodeID> expand(const MulticastTargetSet &targets)
  {
    std::vector<NodeID> nodes;
    for(MulticastTargetSet::const_iterator it = targets.begin(); it != targets.end();
        ++it)
      nodes.push_back(*it);
    return nodes;
  }

  // sorted, disjoint, nonadjacent runs and a consistent cardinality
  void check_canonical(const MulticastTargetSet &targets)
  {
    const std::vector<MulticastTargetSet::Range> &runs = targets.ranges();
    size_t total = 0;
    for(size_t i = 0; i < runs.size(); i++) {
      EXPECT_LE(runs[i].first, runs[i].last);
      EXPECT_GE(runs[i].first, 0);
      if(i > 0)
        EXPECT_GT(static_cast<long long>(runs[i].first),
                  static_cast<long long>(runs[i - 1].last) + 1)
            << "runs " << (i - 1) << " and " << i << " are unmerged or out of order";
      total += runs[i].count();
    }
    EXPECT_EQ(total, targets.size());
  }

  // the encoder must pick the smallest ACTUAL serialized size (plan section 7.2)
  void check_encoder_picked_minimum(const MulticastTargetSet &targets, NodeID num_nodes,
                                    const EncodedMulticastTargets &enc)
  {
    size_t best = 0;
    for(size_t i = 0; i < MULTICAST_ENCODING_KINDS; i++) {
      size_t sz = size_of(targets, num_nodes, static_cast<MulticastTargetEncoding>(i));
      if((sz != 0) && ((best == 0) || (sz < best)))
        best = sz;
    }
    ASSERT_GT(best, 0u);
    EXPECT_EQ(enc.bytes(), best) << "chose " << enc.kind() << " for " << targets;
    EXPECT_EQ(size_of(targets, num_nodes, enc.kind()), best);
  }

  // encode, verify minimality, verify the round trip, and hand back the encoding
  EncodedMulticastTargets round_trip(const MulticastTargetSet &targets, NodeID num_nodes)
  {
    EncodedMulticastTargets enc = EncodedMulticastTargets::encode(targets, num_nodes);
    check_encoder_picked_minimum(targets, num_nodes, enc);
    EXPECT_EQ(enc.num_targets(), targets.size());

    MulticastTargetSet decoded;
    EXPECT_EQ(enc.decode_into(num_nodes, decoded), MulticastDecodeStatus::OK)
        << "kind " << enc.kind();
    EXPECT_EQ(decoded, targets);
    check_canonical(decoded);
    return enc;
  }

}; // namespace

////////////////////////////////////////////////////////////////////////
//
// MulticastTargetSet: building, canonical form, iteration
//

TEST(MulticastTargetSetTest, EmptySet)
{
  MulticastTargetSet targets;
  EXPECT_TRUE(targets.empty());
  EXPECT_EQ(targets.size(), 0u);
  EXPECT_EQ(targets.num_ranges(), 0u);
  EXPECT_FALSE(targets.contains(0));
  EXPECT_TRUE(targets.begin() == targets.end());
  EXPECT_TRUE(expand(targets).empty());
}

TEST(MulticastTargetSetTest, AddMergesAdjacentAndDuplicates)
{
  MulticastTargetSet targets;
  targets.add(5);
  targets.add(5); // duplicate
  EXPECT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets.num_ranges(), 1u);

  targets.add(6); // adjacent above
  targets.add(4); // adjacent below
  EXPECT_EQ(targets.size(), 3u);
  EXPECT_EQ(targets.num_ranges(), 1u) << "adjacent nodes must merge into one run";
  EXPECT_EQ(targets.first_node(), 4);
  EXPECT_EQ(targets.last_node(), 6);

  targets.add(9); // disjoint
  EXPECT_EQ(targets.num_ranges(), 2u);
  targets.add(8); // closes the gap partially
  targets.add(7); // ...and fully
  EXPECT_EQ(targets.num_ranges(), 1u);
  EXPECT_EQ(targets.size(), 6u);
  check_canonical(targets);
}

TEST(MulticastTargetSetTest, ArbitraryRangeBoundaries)
{
  MulticastTargetSet targets;
  targets.add_range(10, 20);
  targets.add_range(40, 50);
  targets.add_range(70, 80);
  EXPECT_EQ(targets.num_ranges(), 3u);
  EXPECT_EQ(targets.size(), 33u);

  // fully inside an existing run
  targets.add_range(12, 15);
  EXPECT_EQ(targets.num_ranges(), 3u);
  EXPECT_EQ(targets.size(), 33u);

  // overlapping two runs and the gap between them
  targets.add_range(15, 45);
  EXPECT_EQ(targets.num_ranges(), 2u);
  EXPECT_EQ(targets.size(), 41u + 11u);

  // exactly adjacent on both sides
  targets.add_range(51, 69);
  EXPECT_EQ(targets.num_ranges(), 1u);
  EXPECT_EQ(targets.size(), 71u);
  EXPECT_EQ(targets.first_node(), 10);
  EXPECT_EQ(targets.last_node(), 80);

  // an inverted range is a no-op
  targets.add_range(200, 100);
  EXPECT_EQ(targets.size(), 71u);
  check_canonical(targets);
}

TEST(MulticastTargetSetTest, SortedIterationAndContains)
{
  MulticastTargetSet targets;
  // deliberately unsorted insertion order
  const NodeID added[] = {7, 1, 30, 31, 2, 29, 0};
  for(size_t i = 0; i < sizeof(added) / sizeof(added[0]); i++)
    targets.add(added[i]);

  std::vector<NodeID> nodes = expand(targets);
  std::vector<NodeID> expected = {0, 1, 2, 7, 29, 30, 31};
  EXPECT_EQ(nodes, expected);
  EXPECT_EQ(nodes.size(), targets.size());
  EXPECT_TRUE(std::is_sorted(nodes.begin(), nodes.end()));

  for(NodeID id = 0; id < 40; id++)
    EXPECT_EQ(targets.contains(id),
              std::find(expected.begin(), expected.end(), id) != expected.end())
        << "id " << id;
  check_canonical(targets);
}

TEST(MulticastTargetSetTest, RemoveLocalRelay)
{
  MulticastTargetSet targets;
  targets.add_range(0, 9);

  // from the middle: splits the run
  EXPECT_TRUE(targets.remove(5));
  EXPECT_EQ(targets.num_ranges(), 2u);
  EXPECT_EQ(targets.size(), 9u);
  EXPECT_FALSE(targets.contains(5));

  // from the front and back edges of a run
  EXPECT_TRUE(targets.remove(0));
  EXPECT_TRUE(targets.remove(9));
  EXPECT_EQ(targets.size(), 7u);
  EXPECT_EQ(targets.first_node(), 1);
  EXPECT_EQ(targets.last_node(), 8);

  // a singleton run disappears entirely
  targets.add(100);
  EXPECT_EQ(targets.num_ranges(), 3u);
  EXPECT_TRUE(targets.remove(100));
  EXPECT_EQ(targets.num_ranges(), 2u);

  // removing something that is not there changes nothing
  EXPECT_FALSE(targets.remove(5));
  EXPECT_FALSE(targets.remove(1000));
  EXPECT_EQ(targets.size(), 7u);
  check_canonical(targets);

  std::vector<NodeID> expected = {1, 2, 3, 4, 6, 7, 8};
  EXPECT_EQ(expand(targets), expected);
}

// NodeSet's bitmask encoding needs the (process-global) bitmask allocator configured,
//  exactly as nodeset_test.cc does
class MulticastNodeSetTest : public ::testing::Test {
protected:
  virtual void SetUp(void)
  {
    NodeSetBitmask::configure_allocator(max_node_id, 1024 /*bitsets_per_chunk*/,
                                        true /*use_twolevel*/);
  }

  virtual void TearDown(void) { NodeSetBitmask::free_allocations(); }

  NodeID max_node_id = 512;
};

TEST_F(MulticastNodeSetTest, NodeSetConversion)
{
  NodeSet nodes;
  nodes.add(17);
  nodes.add(3);
  nodes.add_range(100, 120);
  nodes.add(4);
  nodes.add(101); // already covered

  MulticastTargetSet targets(nodes);
  EXPECT_EQ(targets.size(), nodes.size());
  check_canonical(targets);
  for(NodeSet::const_iterator it = nodes.begin(); it != nodes.end(); ++it)
    EXPECT_TRUE(targets.contains(*it)) << "missing " << *it;
  EXPECT_FALSE(targets.contains(5));
  EXPECT_FALSE(targets.contains(121));

  // ...and back again
  NodeSet reconstructed;
  targets.to_nodeset(reconstructed);
  EXPECT_EQ(reconstructed.size(), nodes.size());
  for(NodeSet::const_iterator it = nodes.begin(); it != nodes.end(); ++it)
    EXPECT_TRUE(reconstructed.contains(*it));
}

TEST(MulticastTargetSetTest, AppendHelpersRejectNoncanonicalInput)
{
  MulticastTargetSet targets;
  EXPECT_TRUE(targets.append_canonical_run(10, 19));
  EXPECT_FALSE(targets.append_canonical_run(5, 8));   // out of order
  EXPECT_FALSE(targets.append_canonical_run(15, 25)); // overlapping
  EXPECT_FALSE(targets.append_canonical_run(20, 25)); // adjacent, must be merged
  EXPECT_TRUE(targets.append_canonical_run(21, 25));
  EXPECT_EQ(targets.num_ranges(), 2u);
  EXPECT_EQ(targets.size(), 15u);

  EXPECT_FALSE(targets.append_increasing_node(25)); // duplicate
  EXPECT_FALSE(targets.append_increasing_node(3));  // out of order
  EXPECT_TRUE(targets.append_increasing_node(26));  // merges with trailing run
  EXPECT_EQ(targets.num_ranges(), 2u);
  EXPECT_TRUE(targets.append_increasing_node(40));
  EXPECT_EQ(targets.num_ranges(), 3u);
  EXPECT_EQ(targets.size(), 17u);
  check_canonical(targets);
}

TEST(MulticastTargetSetTest, RandomizedAgainstReferenceSet)
{
  std::mt19937 rng(12345);
  for(int trial = 0; trial < 32; trial++) {
    const NodeID num_nodes = 512;
    std::set<NodeID> reference;
    MulticastTargetSet targets;

    for(int op = 0; op < 60; op++) {
      unsigned choice = rng() % 10;
      if(choice < 5) {
        NodeID id = static_cast<NodeID>(rng() % num_nodes);
        targets.add(id);
        reference.insert(id);
      } else if(choice < 8) {
        NodeID lo = static_cast<NodeID>(rng() % num_nodes);
        NodeID hi = lo + static_cast<NodeID>(rng() % 40);
        if(hi >= num_nodes)
          hi = num_nodes - 1;
        targets.add_range(lo, hi);
        for(NodeID id = lo; id <= hi; id++)
          reference.insert(id);
      } else {
        NodeID id = static_cast<NodeID>(rng() % num_nodes);
        EXPECT_EQ(targets.remove(id), reference.erase(id) > 0);
      }
    }

    ASSERT_EQ(targets.size(), reference.size());
    check_canonical(targets);
    std::vector<NodeID> nodes = expand(targets);
    std::vector<NodeID> expected(reference.begin(), reference.end());
    ASSERT_EQ(nodes, expected);

    // and the encoding of every one of these survives a round trip
    round_trip(targets, num_nodes);
  }
}

////////////////////////////////////////////////////////////////////////
//
// EncodedMulticastTargets: every encoding is forced deterministically
//

TEST(MulticastEncodingTest, Empty)
{
  MulticastTargetSet targets;
  EncodedMulticastTargets enc = round_trip(targets, 64);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::EMPTY);
  EXPECT_EQ(enc.bytes(), 1u);
  EXPECT_EQ(enc.num_targets(), 0u);
}

TEST(MulticastEncodingTest, DefaultConstructedIsEmpty)
{
  EncodedMulticastTargets enc;
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::EMPTY);
  EXPECT_EQ(enc.bytes(), 1u);
  MulticastTargetSet decoded;
  EXPECT_EQ(enc.decode_into(64, decoded), MulticastDecodeStatus::OK);
  EXPECT_TRUE(decoded.empty());
}

TEST(MulticastEncodingTest, Single)
{
  MulticastTargetSet targets;
  targets.add(5);
  EncodedMulticastTargets enc = round_trip(targets, 64);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::SINGLE);
  EXPECT_EQ(enc.bytes(), 2u);
}

TEST(MulticastEncodingTest, SingleAtNodeZeroAndAtMaxNodeID)
{
  const NodeID num_nodes = 1 << 20;

  MulticastTargetSet zero;
  zero.add(0);
  EncodedMulticastTargets enc_zero = round_trip(zero, num_nodes);
  EXPECT_EQ(enc_zero.kind(), MulticastTargetEncoding::SINGLE);

  MulticastTargetSet top;
  top.add(num_nodes - 1);
  EncodedMulticastTargets enc_top = round_trip(top, num_nodes);
  EXPECT_EQ(enc_top.kind(), MulticastTargetEncoding::SINGLE);

  MulticastTargetSet decoded;
  ASSERT_EQ(enc_top.decode_into(num_nodes, decoded), MulticastDecodeStatus::OK);
  EXPECT_EQ(decoded.first_node(), num_nodes - 1);
}

TEST(MulticastEncodingTest, SmallInline)
{
  // three widely separated nodes whose absolute IDs are single-byte varints: the
  //  inline list ties the delta list on size and wins the tie
  MulticastTargetSet targets;
  targets.add(2);
  targets.add(40);
  targets.add(61);
  EncodedMulticastTargets enc = round_trip(targets, 64);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::SMALL_INLINE);
  EXPECT_EQ(enc.bytes(), 5u);
}

TEST(MulticastEncodingTest, SmallInlineSpanningNodeZeroAndMaxNodeID)
{
  const NodeID num_nodes = 1 << 20;
  MulticastTargetSet targets;
  targets.add(0);
  targets.add(num_nodes - 1);
  EncodedMulticastTargets enc = round_trip(targets, num_nodes);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::SMALL_INLINE);

  MulticastTargetSet decoded;
  ASSERT_EQ(enc.decode_into(num_nodes, decoded), MulticastDecodeStatus::OK);
  EXPECT_EQ(decoded.first_node(), 0);
  EXPECT_EQ(decoded.last_node(), num_nodes - 1);
  EXPECT_EQ(decoded.size(), 2u);
}

TEST(MulticastEncodingTest, RangesForLargeContiguousSet)
{
  // plan section 23: a contiguous target set has compact range metadata
  MulticastTargetSet targets;
  targets.add_range(0, 999);
  EncodedMulticastTargets enc = round_trip(targets, 4096);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::RANGES);
  EXPECT_LE(enc.bytes(), 8u) << "1000 targets must not cost more than a few bytes";

  MulticastTargetSet decoded;
  ASSERT_EQ(enc.decode_into(4096, decoded), MulticastDecodeStatus::OK);
  EXPECT_EQ(decoded.num_ranges(), 1u) << "must decode back into exactly one run";
  EXPECT_EQ(decoded.size(), 1000u);
}

TEST(MulticastEncodingTest, RangesForSeveralLargeRuns)
{
  MulticastTargetSet targets;
  targets.add_range(0, 99);
  targets.add_range(500, 599);
  targets.add_range(1000, 1099);
  EncodedMulticastTargets enc = round_trip(targets, 4096);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::RANGES);
  EXPECT_LE(enc.bytes(), 16u);

  MulticastTargetSet decoded;
  ASSERT_EQ(enc.decode_into(4096, decoded), MulticastDecodeStatus::OK);
  EXPECT_EQ(decoded.num_ranges(), 3u);
}

TEST(MulticastEncodingTest, DeltaListForIrregularSparseSet)
{
  // more entries than SMALL_INLINE will carry, and far too sparse for a bitmap
  MulticastTargetSet targets;
  for(NodeID i = 0; i < 12; i++)
    targets.add(i * 100);
  EncodedMulticastTargets enc = round_trip(targets, 4096);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::DELTA_LIST);
  EXPECT_EQ(enc.bytes(), 14u);
}

TEST(MulticastEncodingTest, DeltaListWhenAbsoluteNodeIDsAreExpensive)
{
  // only four targets, but their absolute IDs need three varint bytes each while the
  //  deltas need one - the delta list beats the inline list on actual bytes
  MulticastTargetSet targets;
  targets.add(100000);
  targets.add(100050);
  targets.add(100100);
  targets.add(100150);
  EncodedMulticastTargets enc = round_trip(targets, 200000);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::DELTA_LIST);
  EXPECT_LT(size_of(targets, 200000, MulticastTargetEncoding::DELTA_LIST),
            size_of(targets, 200000, MulticastTargetEncoding::SMALL_INLINE));
}

TEST(MulticastEncodingTest, BitmapForIrregularDenseSet)
{
  MulticastTargetSet targets;
  for(NodeID i = 0; i < 200; i++)
    if((i % 3) != 0)
      targets.add(i);
  ASSERT_EQ(targets.size(), 133u);

  EncodedMulticastTargets enc = round_trip(targets, 1024);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::BITMAP);
  EXPECT_EQ(enc.bytes(), 29u);
}

TEST(MulticastEncodingTest, BitmapForIrregularDenseSetWithHighBase)
{
  MulticastTargetSet targets;
  for(NodeID i = 300; i < 500; i++)
    if((i % 5) < 3)
      targets.add(i);
  EncodedMulticastTargets enc = round_trip(targets, 4096);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::BITMAP);
}

TEST(MulticastEncodingTest, AllNodes)
{
  const NodeID num_nodes = 1024;
  MulticastTargetSet targets;
  targets.add_range(0, num_nodes - 1);
  EncodedMulticastTargets enc = round_trip(targets, num_nodes);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::ALL_NODES);
  EXPECT_EQ(enc.bytes(), 1u);
  EXPECT_EQ(enc.num_targets(), static_cast<size_t>(num_nodes));
}

TEST(MulticastEncodingTest, AllNodesOnASingleNodeMachine)
{
  MulticastTargetSet targets;
  targets.add(0);
  EncodedMulticastTargets enc = round_trip(targets, 1);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::ALL_NODES);
  EXPECT_EQ(enc.bytes(), 1u);
}

TEST(MulticastEncodingTest, AllExceptForNearlyEverything)
{
  const NodeID num_nodes = 1024;
  MulticastTargetSet targets;
  targets.add_range(0, num_nodes - 1);
  ASSERT_TRUE(targets.remove(7));

  EncodedMulticastTargets enc = round_trip(targets, num_nodes);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::ALL_EXCEPT);
  EXPECT_EQ(enc.bytes(), 3u);
  EXPECT_EQ(enc.num_targets(), static_cast<size_t>(num_nodes) - 1);
}

TEST(MulticastEncodingTest, AllExceptWithSeveralExclusions)
{
  const NodeID num_nodes = 1024;
  MulticastTargetSet targets;
  targets.add_range(0, num_nodes - 1);
  ASSERT_TRUE(targets.remove(0));
  ASSERT_TRUE(targets.remove(3));
  ASSERT_TRUE(targets.remove(900));
  ASSERT_TRUE(targets.remove(num_nodes - 1));

  EncodedMulticastTargets enc = round_trip(targets, num_nodes);
  EXPECT_EQ(enc.kind(), MulticastTargetEncoding::ALL_EXCEPT);
  EXPECT_LT(enc.bytes(), size_of(targets, num_nodes, MulticastTargetEncoding::RANGES));
}

TEST(MulticastEncodingTest, EncodedSizeReportsUnrepresentableKindsAsZero)
{
  MulticastTargetSet empty;
  EXPECT_EQ(size_of(empty, 64, MulticastTargetEncoding::EMPTY), 1u);
  EXPECT_EQ(size_of(empty, 64, MulticastTargetEncoding::SINGLE), 0u);
  EXPECT_EQ(size_of(empty, 64, MulticastTargetEncoding::RANGES), 0u);
  EXPECT_EQ(size_of(empty, 64, MulticastTargetEncoding::ALL_EXCEPT), 0u);

  MulticastTargetSet two;
  two.add(1);
  two.add(9);
  EXPECT_EQ(size_of(two, 64, MulticastTargetEncoding::EMPTY), 0u);
  EXPECT_EQ(size_of(two, 64, MulticastTargetEncoding::SINGLE), 0u);
  EXPECT_EQ(size_of(two, 64, MulticastTargetEncoding::ALL_NODES), 0u);

  // more than MULTICAST_MAX_SMALL_INLINE entries has no inline representation
  MulticastTargetSet many;
  for(NodeID i = 0; i <= static_cast<NodeID>(MULTICAST_MAX_SMALL_INLINE); i++)
    many.add(i * 3);
  EXPECT_EQ(size_of(many, 64, MulticastTargetEncoding::SMALL_INLINE), 0u);
}

TEST(MulticastEncodingTest, EncoderAlwaysPicksTheSmallestRepresentation)
{
  MulticastEncodingTally tally;
  std::mt19937 rng(987654321);
  const NodeID node_counts[] = {1, 2, 8, 64, 1024, 65536};
  for(size_t nc = 0; nc < sizeof(node_counts) / sizeof(node_counts[0]); nc++) {
    const NodeID num_nodes = node_counts[nc];
    for(int trial = 0; trial < 40; trial++) {
      MulticastTargetSet targets;
      // vary the density wildly so that every encoding gets its turn
      const unsigned density = 1 + (rng() % 100);
      for(NodeID id = 0; id < num_nodes; id++)
        if((rng() % 100) < density)
          targets.add(id);
      // one long contiguous block occasionally, so RANGES shows up too
      if((trial % 5) == 0)
        targets.add_range(0, num_nodes / 2);
      round_trip(targets, num_nodes);
      tally.record(EncodedMulticastTargets::encode(targets, num_nodes).kind());
    }
  }

  // this sweep should have exercised every one of the eight encodings
  for(size_t i = 0; i < MULTICAST_ENCODING_KINDS; i++) {
    MulticastTargetEncoding kind = static_cast<MulticastTargetEncoding>(i);
    EXPECT_GT(tally.get(kind), 0u) << "never chose " << kind;
  }
}

TEST(MulticastEncodingTest, EncodingTallyRecordsEveryChoice)
{
  MulticastEncodingTally tally;
  const NodeID num_nodes = 1024;

  MulticastTargetSet empty;
  EncodedMulticastTargets::encode(empty, num_nodes, &tally);

  MulticastTargetSet single;
  single.add(3);
  EncodedMulticastTargets::encode(single, num_nodes, &tally);

  MulticastTargetSet all;
  all.add_range(0, num_nodes - 1);
  EncodedMulticastTargets::encode(all, num_nodes, &tally);
  EncodedMulticastTargets::encode(all, num_nodes, &tally);

  MulticastTargetSet ranges;
  ranges.add_range(0, 511);
  EncodedMulticastTargets::encode(ranges, num_nodes, &tally);

  EXPECT_EQ(tally.get(MulticastTargetEncoding::EMPTY), 1u);
  EXPECT_EQ(tally.get(MulticastTargetEncoding::SINGLE), 1u);
  EXPECT_EQ(tally.get(MulticastTargetEncoding::ALL_NODES), 2u);
  EXPECT_EQ(tally.get(MulticastTargetEncoding::RANGES), 1u);
  EXPECT_EQ(tally.get(MulticastTargetEncoding::BITMAP), 0u);
  EXPECT_EQ(tally.total(), 5u);

  tally.reset();
  EXPECT_EQ(tally.total(), 0u);
}

////////////////////////////////////////////////////////////////////////
//
// malformed encodings must be rejected safely (plan sections 21.1 and 22)
//

TEST(MulticastDecodeTest, EmptyBufferAndUnknownKind)
{
  ByteVec buf;
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  buf.push_back(static_cast<unsigned char>(MULTICAST_ENCODING_KINDS));
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::UNKNOWN_KIND);

  buf.clear();
  buf.push_back(0xff);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::UNKNOWN_KIND);
}

TEST(MulticastDecodeTest, TrailingBytes)
{
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::EMPTY);
  buf.push_back(0x00);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRAILING_BYTES);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  put_varint(buf, 5);
  buf.push_back(0x00);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRAILING_BYTES);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_NODES);
  buf.push_back(0x00);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRAILING_BYTES);
}

TEST(MulticastDecodeTest, TruncatedPayloads)
{
  // SINGLE with no node at all
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // SMALL_INLINE whose element varint runs off the end
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 2);
  buf.push_back(0x80); // continuation...
  buf.push_back(0x80); // ...and another, then nothing
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // RANGES with a start but no length
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 1);
  put_varint(buf, 3);
  buf.push_back(0x80);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // DELTA_LIST whose final delta is cut off
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::DELTA_LIST);
  put_varint(buf, 3);
  put_varint(buf, 1);
  put_varint(buf, 1);
  buf.push_back(0x80);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // BITMAP with fewer map bytes than the bit length demands
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 16);
  buf.push_back(0xff);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // ALL_EXCEPT whose second exclusion is cut off
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 2);
  put_varint(buf, 1);
  buf.push_back(0x80);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRUNCATED);

  // NOTE: a declared count with no payload at all behind it is caught by the length
  //  guard before any element is read, which is the point of that guard
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 1);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);
}

TEST(MulticastDecodeTest, AbsurdCardinalityIsRejectedWithoutAllocating)
{
  // NOTE: the point of these is that the decoder must never size an allocation from a
  //  remote length before checking it against the payload size and the node count
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 1000000);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  // a cardinality that is fine against the node count but cannot fit in the bytes left
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 4);
  put_varint(buf, 1);
  put_varint(buf, 2);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::DELTA_LIST);
  put_varint(buf, 4000000000ull);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::DELTA_LIST);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 0xffffffffull);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  // more runs than could possibly fit in the node space, even though the payload is
  //  long enough to hold them
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 6);
  for(int i = 0; i < 6; i++) {
    put_varint(buf, static_cast<uint64_t>(i));
    put_varint(buf, 1);
  }
  EXPECT_EQ(decode_bytes(buf, 8), MulticastDecodeStatus::BAD_CARDINALITY);

  // excluding every node would leave nothing behind
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 64);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::BAD_CARDINALITY);

  // ALL_NODES only means something on a machine with nodes
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_NODES);
  EXPECT_EQ(decode_bytes(buf, 0), MulticastDecodeStatus::BAD_CARDINALITY);
}

TEST(MulticastDecodeTest, NodesOutsideTheConfiguredNodeCount)
{
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  put_varint(buf, 100);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);
  EXPECT_EQ(decode_bytes(buf, 101), MulticastDecodeStatus::OK);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 2);
  put_varint(buf, 1);
  put_varint(buf, 200);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 1);
  put_varint(buf, 100);
  put_varint(buf, 1);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::DELTA_LIST);
  put_varint(buf, 2);
  put_varint(buf, 1);
  put_varint(buf, 100);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 100);
  put_varint(buf, 8);
  buf.push_back(0xff);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 1);
  put_varint(buf, 100);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NODE_OUT_OF_RANGE);
}

TEST(MulticastDecodeTest, RangeOverflow)
{
  // zero length
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 1);
  put_varint(buf, 3);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::RANGE_OVERFLOW);

  // length that walks off the end of the node space
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 1);
  put_varint(buf, 60);
  put_varint(buf, 1000);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::RANGE_OVERFLOW);

  // a length that would wrap 64-bit arithmetic if it were simply added to the start
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 1);
  put_varint(buf, 60);
  put_varint(buf, 0xfffffffffffffffull);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::RANGE_OVERFLOW);

  // bitmap whose bit length leaves the node space
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 60);
  put_varint(buf, 100);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::RANGE_OVERFLOW);

  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::RANGE_OVERFLOW);
}

TEST(MulticastDecodeTest, NoncanonicalRanges)
{
  // out of order / overlapping
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 2);
  put_varint(buf, 0);
  put_varint(buf, 10);
  put_varint(buf, 5);
  put_varint(buf, 10);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // strictly descending
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 2);
  put_varint(buf, 40);
  put_varint(buf, 4);
  put_varint(buf, 10);
  put_varint(buf, 4);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // adjacent runs that should have been merged
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 2);
  put_varint(buf, 0);
  put_varint(buf, 5);
  put_varint(buf, 5);
  put_varint(buf, 5);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // ...but a one-node gap is fine
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 2);
  put_varint(buf, 0);
  put_varint(buf, 5);
  put_varint(buf, 6);
  put_varint(buf, 5);
  MulticastTargetSet decoded;
  EXPECT_EQ(decode_bytes(buf, 64, decoded), MulticastDecodeStatus::OK);
  EXPECT_EQ(decoded.size(), 10u);
  EXPECT_EQ(decoded.num_ranges(), 2u);
}

TEST(MulticastDecodeTest, NoncanonicalLists)
{
  // unsorted inline list
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 3);
  put_varint(buf, 5);
  put_varint(buf, 4);
  put_varint(buf, 9);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // duplicated inline entry
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SMALL_INLINE);
  put_varint(buf, 2);
  put_varint(buf, 5);
  put_varint(buf, 5);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // a zero delta repeats the previous node
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::DELTA_LIST);
  put_varint(buf, 3);
  put_varint(buf, 1);
  put_varint(buf, 2);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // ...and so does one in the exclusion list
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::ALL_EXCEPT);
  put_varint(buf, 2);
  put_varint(buf, 1);
  put_varint(buf, 0);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);
}

TEST(MulticastDecodeTest, NoncanonicalBitmaps)
{
  // the base node must be the first set bit
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 8);
  buf.push_back(0xfe);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // the last bit of the declared length must be set
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 8);
  buf.push_back(0x01);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // padding bits past the declared length must be zero
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 4);
  buf.push_back(0x19);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // ...whereas this one is well formed
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 4);
  buf.push_back(0x09);
  MulticastTargetSet decoded;
  EXPECT_EQ(decode_bytes(buf, 64, decoded), MulticastDecodeStatus::OK);
  std::vector<NodeID> expected = {0, 3};
  EXPECT_EQ(expand(decoded), expected);

  // extra map bytes beyond the declared bit length
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::BITMAP);
  put_varint(buf, 0);
  put_varint(buf, 8);
  buf.push_back(0xff);
  buf.push_back(0x00);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::TRAILING_BYTES);
}

TEST(MulticastDecodeTest, NoncanonicalVarints)
{
  // overlong encoding of 5
  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  buf.push_back(0x85);
  buf.push_back(0x00);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // a varint longer than any 64-bit value needs
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  for(size_t i = 0; i < MulticastWire::MAX_VARINT_BYTES; i++)
    buf.push_back(0x80);
  buf.push_back(0x01);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);

  // a 10-byte varint whose top byte would shift bits out of 64
  buf.clear();
  put_kind(buf, MulticastTargetEncoding::SINGLE);
  for(size_t i = 0; i < MulticastWire::MAX_VARINT_BYTES - 1; i++)
    buf.push_back(0xff);
  buf.push_back(0x7f);
  EXPECT_EQ(decode_bytes(buf, 64), MulticastDecodeStatus::NOT_CANONICAL);
}

TEST(MulticastDecodeTest, FailureLeavesTheTargetSetEmpty)
{
  MulticastTargetSet targets;
  targets.add_range(0, 100);
  ASSERT_FALSE(targets.empty());

  ByteVec buf;
  put_kind(buf, MulticastTargetEncoding::RANGES);
  put_varint(buf, 2);
  put_varint(buf, 0);
  put_varint(buf, 10);
  put_varint(buf, 5);
  put_varint(buf, 10);
  EXPECT_EQ(decode_bytes(buf, 64, targets), MulticastDecodeStatus::NOT_CANONICAL);
  EXPECT_TRUE(targets.empty());
  EXPECT_EQ(targets.size(), 0u);
}

TEST(MulticastDecodeTest, VarintRoundTrip)
{
  const uint64_t values[] = {0,          1,          127,         128,
                             255,        16383,      16384,       (1ull << 31) - 1,
                             1ull << 31, 1ull << 63, ~uint64_t(0)};
  for(size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
    ByteVec buf;
    put_varint(buf, values[i]);
    EXPECT_EQ(buf.size(), MulticastWire::varint_size(values[i]));
    EXPECT_LE(buf.size(), MulticastWire::MAX_VARINT_BYTES);
    size_t pos = 0;
    uint64_t got = 0;
    EXPECT_EQ(MulticastWire::read_varint(buf.data(), buf.size(), pos, got),
              MulticastDecodeStatus::OK);
    EXPECT_EQ(got, values[i]);
    EXPECT_EQ(pos, buf.size());
  }
}

namespace {

  // A buffer whose last byte sits immediately before an unmapped page, so that a
  //  decoder reading even one byte past the end of what it was given takes a SIGSEGV
  //  instead of quietly picking up adjacent heap.  This is what makes "no out-of-bounds
  //  read" (plan section 22) an observed property rather than a code-reading claim.
  class GuardedBuffer {
  public:
    GuardedBuffer(void)
    {
      page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      base = static_cast<unsigned char *>(mmap(nullptr, 2 * page_size,
                                               PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      assert(base != MAP_FAILED);
      // second page is unreadable - anything past the end of the payload faults
      int ret = mprotect(base + page_size, page_size, PROT_NONE);
      assert(ret == 0);
      (void)ret;
    }

    ~GuardedBuffer(void) { munmap(base, 2 * page_size); }

    size_t capacity(void) const { return page_size; }

    // returns a pointer to 'bytes' writable bytes that END exactly at the guard page
    unsigned char *place(size_t bytes)
    {
      assert(bytes <= page_size);
      return base + page_size - bytes;
    }

  protected:
    unsigned char *base = nullptr;
    size_t page_size = 0;
  };

}; // namespace

TEST(MulticastDecodeTest, FuzzedPayloadsNeverReadPastTheEndOfTheBuffer)
{
  GuardedBuffer guard;
  std::mt19937 rng(20250729);
  const NodeID node_counts[] = {1, 2, 7, 64, 1024, 1 << 20};

  size_t num_ok = 0;
  std::set<MulticastDecodeStatus> statuses_seen;

  for(int trial = 0; trial < 40000; trial++) {
    // deliberately favors short payloads (that is where truncation lives) but
    //  occasionally goes large
    size_t len = ((trial % 16) == 0) ? (1 + (rng() % 400)) : (rng() % 24);
    unsigned char *buf = guard.place(len);
    // bias the kind byte into the valid range half the time so that the deeper
    //  per-kind validation actually gets exercised
    for(size_t i = 0; i < len; i++)
      buf[i] = static_cast<unsigned char>(rng() & 0xff);
    if((len > 0) && ((rng() & 1) != 0))
      buf[0] = static_cast<unsigned char>(rng() % MULTICAST_ENCODING_KINDS);

    const NodeID num_nodes =
        node_counts[rng() % (sizeof(node_counts) / sizeof(node_counts[0]))];

    MulticastTargetSet targets;
    MulticastDecodeStatus status =
        EncodedMulticastTargets::decode(buf, len, num_nodes, targets);
    statuses_seen.insert(status);

    if(status != MulticastDecodeStatus::OK) {
      // a rejected encoding must leave nothing behind
      EXPECT_TRUE(targets.empty());
      EXPECT_EQ(targets.size(), 0u);
      continue;
    }

    num_ok++;
    // an accepted encoding must be canonical, in range, and must re-encode to
    //  something that decodes back to the same set
    ASSERT_TRUE(targets.fits_node_count(num_nodes));
    const std::vector<MulticastTargetSet::Range> &runs = targets.ranges();
    size_t total = 0;
    for(size_t i = 0; i < runs.size(); i++) {
      EXPECT_LE(runs[i].first, runs[i].last);
      if(i > 0)
        EXPECT_GT(runs[i].first, runs[i - 1].last + 1) << "runs not canonical";
      total += runs[i].count();
    }
    EXPECT_EQ(total, targets.size());

    EncodedMulticastTargets re = EncodedMulticastTargets::encode(targets, num_nodes);
    MulticastTargetSet round;
    ASSERT_EQ(re.decode_into(num_nodes, round), MulticastDecodeStatus::OK);
    EXPECT_TRUE(round == targets);
  }

  // non-vacuity: random bytes really do get accepted sometimes, and more than one
  //  rejection reason is reached
  EXPECT_GT(num_ok, 0u);
  EXPECT_GE(statuses_seen.size(), 5u);
}

////////////////////////////////////////////////////////////////////////
//
// partitioning (plan sections 7.3 and 20.1)
//

namespace {

  void check_partition(const MulticastTargetSet &targets, size_t radix)
  {
    std::vector<MulticastTargetSet> slices;
    targets.partition(radix, slices);

    if((targets.size() == 0) || (radix == 0)) {
      EXPECT_TRUE(slices.empty());
      return;
    }

    const size_t expected_slices = std::min(radix, targets.size());
    ASSERT_EQ(slices.size(), expected_slices);

    // nearly-equal cardinality: no two slices differ by more than one node
    size_t smallest = targets.size(), largest = 0, sum = 0;
    for(size_t i = 0; i < slices.size(); i++) {
      EXPECT_GT(slices[i].size(), 0u) << "slice " << i << " is empty";
      smallest = std::min(smallest, slices[i].size());
      largest = std::max(largest, slices[i].size());
      sum += slices[i].size();
      check_canonical(slices[i]);
    }
    EXPECT_LE(largest - smallest, 1u)
        << "radix " << radix << ": slice sizes " << smallest << ".." << largest;
    EXPECT_EQ(sum, targets.size()) << "partitions must not overlap";

    // disjoint, ordered, and their union is exactly the input
    MulticastTargetSet united;
    for(size_t i = 0; i < slices.size(); i++) {
      if(i > 0)
        EXPECT_GT(slices[i].first_node(), slices[i - 1].last_node());
      for(MulticastTargetSet::const_iterator it = slices[i].begin();
          it != slices[i].end(); ++it) {
        ASSERT_FALSE(united.contains(*it)) << "node " << *it << " appears twice";
        united.add(*it);
      }
    }
    EXPECT_EQ(united, targets);
    EXPECT_EQ(united.size(), targets.size());
  }

}; // namespace

TEST(MulticastPartitionTest, EmptyAndDegenerateRadix)
{
  MulticastTargetSet empty;
  std::vector<MulticastTargetSet> slices;
  empty.partition(4, slices);
  EXPECT_TRUE(slices.empty());

  MulticastTargetSet targets;
  targets.add_range(0, 9);
  targets.partition(0, slices);
  EXPECT_TRUE(slices.empty());
}

TEST(MulticastPartitionTest, RadixOneKeepsEverythingTogether)
{
  MulticastTargetSet targets;
  targets.add_range(0, 99);
  targets.add(500);
  std::vector<MulticastTargetSet> slices;
  targets.partition(1, slices);
  ASSERT_EQ(slices.size(), 1u);
  EXPECT_EQ(slices[0], targets);
  check_partition(targets, 1);
}

TEST(MulticastPartitionTest, RadixLargerThanCardinality)
{
  MulticastTargetSet targets;
  targets.add(3);
  targets.add(9);
  targets.add(27);
  std::vector<MulticastTargetSet> slices;
  targets.partition(16, slices);
  ASSERT_EQ(slices.size(), 3u);
  for(size_t i = 0; i < slices.size(); i++)
    EXPECT_EQ(slices[i].size(), 1u);
  check_partition(targets, 16);
}

TEST(MulticastPartitionTest, BalanceAndExactUnionAcrossRadices)
{
  const size_t radices[] = {1, 2, 3, 4, 16, 64, 1000};

  // a contiguous set, a set of several runs, an irregular sparse set and an irregular
  //  dense one
  std::vector<MulticastTargetSet> cases;

  MulticastTargetSet contiguous;
  contiguous.add_range(0, 99);
  cases.push_back(contiguous);

  MulticastTargetSet runs;
  runs.add_range(0, 9);
  runs.add_range(50, 52);
  runs.add_range(200, 287);
  cases.push_back(runs);

  MulticastTargetSet sparse;
  for(NodeID i = 0; i < 37; i++)
    sparse.add(i * 13 + 1);
  cases.push_back(sparse);

  MulticastTargetSet dense;
  for(NodeID i = 0; i < 300; i++)
    if((i % 7) != 3)
      dense.add(i);
  cases.push_back(dense);

  MulticastTargetSet prime;
  prime.add_range(0, 100); // 101 nodes, awkward for most radices
  cases.push_back(prime);

  for(size_t c = 0; c < cases.size(); c++)
    for(size_t r = 0; r < sizeof(radices) / sizeof(radices[0]); r++) {
      SCOPED_TRACE(testing::Message() << "case " << c << " radix " << radices[r]);
      check_partition(cases[c], radices[r]);
    }
}

TEST(MulticastPartitionTest, ExtraNodesGoToTheEarliestSlices)
{
  MulticastTargetSet targets;
  targets.add_range(0, 99); // 100 nodes into 7 slices -> 15,15,14,14,14,14,14
  std::vector<MulticastTargetSet> slices;
  targets.partition(7, slices);
  ASSERT_EQ(slices.size(), 7u);
  EXPECT_EQ(slices[0].size(), 15u);
  EXPECT_EQ(slices[1].size(), 15u);
  for(size_t i = 2; i < 7; i++)
    EXPECT_EQ(slices[i].size(), 14u);
  check_partition(targets, 7);
}

TEST(MulticastPartitionTest, LargeRangePartitionsWithoutExpanding)
{
  // plan section 7.2: partition by cardinality without requiring full expansion.  One
  //  run of 50000 nodes cut into 8 slices must stay eight single-run slices, each of
  //  which still encodes as a handful of RANGES bytes.
  const NodeID num_nodes = 1 << 20;
  const NodeID base = 1000;
  const size_t span = 50000;
  MulticastTargetSet targets;
  targets.add_range(base, base + static_cast<NodeID>(span) - 1);
  ASSERT_EQ(targets.num_ranges(), 1u);
  ASSERT_EQ(targets.size(), span);

  std::vector<MulticastTargetSet> slices;
  targets.partition(8, slices);
  ASSERT_EQ(slices.size(), 8u);

  size_t total = 0;
  NodeID expected_first = base;
  for(size_t i = 0; i < slices.size(); i++) {
    SCOPED_TRACE(testing::Message() << "slice " << i);
    EXPECT_EQ(slices[i].num_ranges(), 1u) << "the run must be cut, never expanded";
    EXPECT_EQ(slices[i].size(), span / 8);
    EXPECT_EQ(slices[i].first_node(), expected_first);
    expected_first = slices[i].last_node() + 1;
    total += slices[i].size();

    EncodedMulticastTargets enc = EncodedMulticastTargets::encode(slices[i], num_nodes);
    EXPECT_EQ(enc.kind(), MulticastTargetEncoding::RANGES);
    EXPECT_LE(enc.bytes(), 12u) << "6250 targets must stay compact range metadata";

    MulticastTargetSet decoded;
    ASSERT_EQ(enc.decode_into(num_nodes, decoded), MulticastDecodeStatus::OK);
    EXPECT_EQ(decoded, slices[i]);
  }
  EXPECT_EQ(total, span);

  // and recursing (as a relay would) keeps the slices range-shaped
  std::vector<MulticastTargetSet> subslices;
  slices[0].partition(8, subslices);
  ASSERT_EQ(subslices.size(), 8u);
  for(size_t i = 0; i < subslices.size(); i++)
    EXPECT_EQ(subslices[i].num_ranges(), 1u);
}

TEST(MulticastPartitionTest, PartitionAfterRemovingTheLocalRelay)
{
  // the shape a relay actually produces: it drops itself, then splits the rest
  MulticastTargetSet targets;
  targets.add_range(0, 63);
  const NodeID relay = targets.first_node();
  ASSERT_TRUE(targets.remove(relay));
  EXPECT_FALSE(targets.contains(relay));
  EXPECT_EQ(targets.size(), 63u);

  std::vector<MulticastTargetSet> slices;
  targets.partition(4, slices);
  ASSERT_EQ(slices.size(), 4u);
  for(size_t i = 0; i < slices.size(); i++)
    EXPECT_FALSE(slices[i].contains(relay));
  check_partition(targets, 4);
}

TEST(MulticastPartitionTest, RandomizedBalanceAndCoverage)
{
  std::mt19937 rng(24680);
  const size_t radices[] = {1, 2, 4, 16, 64};
  for(int trial = 0; trial < 24; trial++) {
    const NodeID num_nodes = 4096;
    MulticastTargetSet targets;
    const unsigned density = 1 + (rng() % 100);
    for(NodeID id = 0; id < num_nodes; id++)
      if((rng() % 100) < density)
        targets.add(id);
    // occasionally throw in a big contiguous block too
    if((trial % 3) == 0)
      targets.add_range(1000, 2500);

    for(size_t r = 0; r < sizeof(radices) / sizeof(radices[0]); r++) {
      SCOPED_TRACE(testing::Message() << "trial " << trial << " radix " << radices[r]);
      check_partition(targets, radices[r]);
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
// Stage 2b: envelope, handler redispatch and bounded-radix forwarding
//  (plan sections 7.1, 7.3, 7.4 and 20.1)
//

namespace {

  ////////////////////////////////////////////////////////////////////////
  //
  // trace shared by the simulated network and the test message handlers
  //

  struct TraceEvent {
    enum Kind
    {
      SEND_ENVELOPE,
      SEND_ORIGINAL,
      SEND_ACK,
      HANDLED,
    };
    Kind kind = HANDLED;
    NodeID node = 0; // node that performed the action
    NodeID peer = 0; // envelope destination for a send, apparent sender for HANDLED
  };

  struct HandledMessage {
    NodeID node = 0;   // where the handler ran, or -1 if it ran during a drain
    NodeID sender = 0; // sender as presented to the handler
    int value = 0;
    std::vector<char> payload;
  };

  std::vector<TraceEvent> g_trace;
  std::vector<HandledMessage> g_handled;
  // the node the simulated network is currently "running on"
  NodeID g_current_node = 0;
  // monotonically increasing tick, so that "the completion callback ran after the last
  //  handler" can be asserted rather than assumed
  size_t g_clock = 0;
  // tick of the most recent original-message handler invocation
  size_t g_last_handled_tick = 0;

  // Nodes whose "runtime shutdown" handler has already run.  Such a node has stopped
  //  making progress, so a send attributed to it afterwards would in production be a
  //  message that never leaves the node - i.e. a permanently stranded subtree.  This is
  //  exactly what plan section 7.3 step 4's forward-before-deliver rule prevents.  Both
  //  stay empty/zero for every test that does not use McastShutdownMessage.
  std::set<NodeID> g_stopped_nodes;
  size_t g_sends_from_stopped_nodes = 0;

  size_t next_tick(void) { return ++g_clock; }

  void reset_trace(void)
  {
    g_trace.clear();
    g_handled.clear();
    g_current_node = 0;
    g_clock = 0;
    g_last_handled_tick = 0;
    g_stopped_nodes.clear();
    g_sends_from_stopped_nodes = 0;
  }

  void record_handled(NodeID node, NodeID sender, int value, const void *payload,
                      size_t payload_size)
  {
    HandledMessage h;
    h.node = node;
    h.sender = sender;
    h.value = value;
    if(payload_size > 0) {
      const char *c = static_cast<const char *>(payload);
      h.payload.assign(c, c + payload_size);
    }
    g_handled.push_back(h);
    g_last_handled_tick = next_tick();
  }

  // An ordinary message type with an inline handler, so that local delivery is
  //  synchronous and therefore observable in the trace relative to the child sends.
  struct McastTestMessage {
    int value = 0;

    static bool handle_inline(NodeID sender, const McastTestMessage &hdr,
                              const void *payload, size_t payload_size,
                              TimeLimit /*work_until*/)
    {
      record_handled(g_current_node, sender, hdr.value, payload, payload_size);
      TraceEvent ev;
      ev.kind = TraceEvent::HANDLED;
      ev.node = g_current_node;
      ev.peer = sender;
      g_trace.push_back(ev);
      return true;
    }

    static void handle_message(NodeID sender, const McastTestMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };

  // A message type with NO inline handler, so that local delivery has to go through
  //  IncomingMessageManager's deferred queue and its TimeLimit-aware handler path.
  struct McastDeferredMessage {
    int value = 0;

    static void handle_message(NodeID sender, const McastDeferredMessage &hdr,
                               const void *payload, size_t payload_size,
                               TimeLimit /*work_until*/)
    {
      // this runs during the drain, not at delivery time, so the receiving node is not
      //  attributable from here - only the apparent sender matters
      record_handled(-1, sender, hdr.value, payload, payload_size);
    }
  };

  // Stand-in for Realm::RuntimeShutdownMessage (plan section 7.6: "Runtime shutdown is a
  //  critical forwarding-order test").  Its handler does what
  //  RuntimeShutdownMessage::handle_message ultimately does - it stops the node - so any
  //  send this node makes afterwards is recorded as a violation.
  //
  // The real handler has no handle_inline and therefore runs off the deferred queue,
  //  which would make "forward before deliver" trivially true.  Handling it INLINE here
  //  is deliberately the stronger model: the handler runs synchronously inside
  //  MulticastTransport::deliver_local, so the relay's child sends have to have already
  //  happened by then or the violation counter fires.
  struct McastShutdownMessage {
    int result_code = 0;

    static bool handle_inline(NodeID sender, const McastShutdownMessage &hdr,
                              const void *payload, size_t payload_size,
                              TimeLimit /*work_until*/)
    {
      record_handled(g_current_node, sender, hdr.result_code, payload, payload_size);
      TraceEvent ev;
      ev.kind = TraceEvent::HANDLED;
      ev.node = g_current_node;
      ev.peer = sender;
      g_trace.push_back(ev);
      // RuntimeShutdownMessage::handle_message asserts the request is not a duplicate,
      //  so a second delivery to the same node would be fatal in production
      EXPECT_EQ(g_stopped_nodes.count(g_current_node), 0u)
          << "node " << g_current_node << " was shut down twice";
      g_stopped_nodes.insert(g_current_node);
      return true;
    }

    static void handle_message(NodeID sender, const McastShutdownMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };

  namespace {
    // The multicast layer deliberately has no barrier dependency: it records through the
    //  abstract MulticastMetricsSink and the barrier layer supplies its own
    //  implementation (plan section 21.3).  These tests therefore carry their OWN sink
    //  rather than reaching into barrier state - which is also what keeps this file
    //  compiling independently of whatever the barrier protocol currently looks like.
    struct TestCounter {
      Realm::atomic<unsigned long long> value{0};
      void bump(unsigned long long by = 1) { value.fetch_add(by); }
      void bump_max(unsigned long long v)
      {
        unsigned long long cur = value.load();
        while((v > cur) && !value.compare_exchange(cur, v)) {
        }
      }
      unsigned long long get(void) const { return value.load(); }
    };

    struct TestMulticastCounters {
      TestCounter multicast_encoding_empty, multicast_encoding_single;
      TestCounter multicast_encoding_small_inline, multicast_encoding_ranges;
      TestCounter multicast_encoding_delta_list, multicast_encoding_bitmap;
      TestCounter multicast_encoding_all_nodes, multicast_encoding_all_except;
      TestCounter multicast_first_hops, multicast_max_depth;
    };

    class TestMulticastMetrics : public Realm::MulticastMetricsSink {
    public:
      TestMulticastMetrics(TestMulticastCounters *_c = nullptr)
        : counters(_c)
      {}
      void set_counters(TestMulticastCounters *_c) { counters = _c; }

      virtual void record_encoding_choice(Realm::MulticastTargetEncoding kind)
      {
        if(counters == nullptr)
          return;
        switch(kind) {
        case Realm::MulticastTargetEncoding::EMPTY:
          counters->multicast_encoding_empty.bump();
          break;
        case Realm::MulticastTargetEncoding::SINGLE:
          counters->multicast_encoding_single.bump();
          break;
        case Realm::MulticastTargetEncoding::SMALL_INLINE:
          counters->multicast_encoding_small_inline.bump();
          break;
        case Realm::MulticastTargetEncoding::RANGES:
          counters->multicast_encoding_ranges.bump();
          break;
        case Realm::MulticastTargetEncoding::DELTA_LIST:
          counters->multicast_encoding_delta_list.bump();
          break;
        case Realm::MulticastTargetEncoding::BITMAP:
          counters->multicast_encoding_bitmap.bump();
          break;
        case Realm::MulticastTargetEncoding::ALL_NODES:
          counters->multicast_encoding_all_nodes.bump();
          break;
        case Realm::MulticastTargetEncoding::ALL_EXCEPT:
          counters->multicast_encoding_all_except.bump();
          break;
        }
      }
      virtual void record_first_hops(size_t n)
      {
        if(counters != nullptr)
          counters->multicast_first_hops.bump(n);
      }
      virtual void record_tree_depth(unsigned depth)
      {
        if(counters != nullptr)
          counters->multicast_max_depth.bump_max(depth);
      }

    protected:
      TestMulticastCounters *counters;
    };
  }; // namespace

  class SimMulticastNetwork;

  // The multicast envelope as the simulated network transmits it when fragmentation is
  //  enabled.  This exists so that an oversized envelope can be pushed through the REAL
  //  reassembly machinery: registering this type also auto-registers
  //  WrappedWithFragInfo<SimEnvelopeMessage>, and IncomingMessageManager reassembles
  //  those chunks with FragmentedMessage before the (deliberately non-inline) handler
  //  below hands the whole envelope to MulticastForwarder::forward().  That is exactly
  //  the sequence ActiveMessage<MulticastEnvelopeMessage>'s chunked mode produces on a
  //  real network, which is what the production transport relies on (plan section 7.5).
  struct SimEnvelopeMessage {
    MulticastEnvelopeMessage env;
    NodeID to = 0; // the simulated node this envelope was addressed to

    static void handle_message(NodeID sender, const SimEnvelopeMessage &hdr,
                               const void *payload, size_t payload_size,
                               TimeLimit work_until);
  };

  ActiveMessageHandlerReg<McastTestMessage> mcast_test_message_reg;
  ActiveMessageHandlerReg<McastDeferredMessage> mcast_deferred_message_reg;
  ActiveMessageHandlerReg<McastShutdownMessage> mcast_shutdown_message_reg;
  ActiveMessageHandlerReg<SimEnvelopeMessage> sim_envelope_message_reg;

  // the simulated network currently under test - the reassembly handler above has no
  //  other way to find it
  SimMulticastNetwork *g_sim = nullptr;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimMulticastNetwork
  //

  // In-process stand-in for a multi-node network.  send_envelope() records the envelope
  //  in a FIFO instead of touching a backend; run() then replays each queued envelope
  //  into MulticastForwarder::forward() with the receiving node installed as "the local
  //  node".  Local delivery still goes through a real IncomingMessageManager, so the
  //  handler-table lookup, inline-handler decision and TimeLimit behavior are the real
  //  ones rather than a test reimplementation.
  class SimMulticastNetwork : public MulticastTransport {
  public:
    struct QueuedEnvelope {
      NodeID from = 0;
      NodeID to = 0;
      MulticastEnvelopeMessage env;
      std::vector<unsigned char> payload;
    };

    struct QueuedUnicast {
      NodeID from = 0;
      NodeID to = 0;
      ActiveMessageHandlerTable::MessageID msgid = 0;
      std::vector<unsigned char> hdr;
      std::vector<unsigned char> payload;
      // an ordinary active-message remote completion requested by the origin - it fires
      //  once the single target has HANDLED the message
      bool has_completion = false;
      MulticastCompletionToken completion;
    };

    struct QueuedAck {
      NodeID from = 0;
      NodeID to = 0;
      MulticastAckMessage ack;
    };

    // one fragment of an envelope that exceeded frag_chunk_size
    struct QueuedFragment {
      NodeID from = 0;
      NodeID to = 0;
      WrappedWithFragInfo<SimEnvelopeMessage> hdr;
      std::vector<unsigned char> chunk;
    };

    SimMulticastNetwork(NodeID _nodes, size_t _radix)
      : crs(nullptr)
      , mgr(_nodes, 0 /*dedicated_threads*/, crs)
      , metrics(&counters)
      , nodes(_nodes)
      , fan_radix(_radix)
    {
      // the transient completion state of plan section 7.5 is per node, and the
      //  simulation runs every node in one process
      for(NodeID i = 0; i < _nodes; i++)
        node_completion.emplace_back(new MulticastCompletionState);
      g_sim = this;
    }

    virtual ~SimMulticastNetwork(void)
    {
      // an IncomingMessageManager is a BackgroundWorkItem and insists on being shut
      //  down before it is destroyed
      mgr.shutdown();
      if(g_sim == this)
        g_sim = nullptr;
    }

    // A node that has already run its shutdown handler has stopped making progress -
    //  anything it tries to transmit afterwards would never actually leave.
    static void note_send(NodeID from)
    {
      if(g_stopped_nodes.count(from) != 0)
        g_sends_from_stopped_nodes++;
    }

    // --- MulticastTransport ---------------------------------------------

    virtual NodeID my_node_id(void) const { return cur; }
    virtual NodeID num_nodes(void) const { return nodes; }
    virtual size_t radix(void) const { return fan_radix; }

    virtual void send_envelope(NodeID relay, const MulticastEnvelopeMessage &env,
                               const void *payload, size_t payload_bytes)
    {
      note_send(cur);
      TraceEvent ev;
      ev.kind = TraceEvent::SEND_ENVELOPE;
      ev.node = cur;
      ev.peer = relay;
      g_trace.push_back(ev);

      QueuedEnvelope q;
      q.from = cur;
      q.to = relay;
      q.env = env;
      const unsigned char *p = static_cast<const unsigned char *>(payload);
      q.payload.assign(p, p + payload_bytes);
      all_envelopes.push_back(q);
      sends_per_node[cur]++;
      if(env.depth > max_depth_seen)
        max_depth_seen = env.depth;

      if((frag_chunk_size == 0) || (q.payload.size() <= frag_chunk_size)) {
        envelope_queue.push_back(q);
        return;
      }

      // oversized: hand it to the ordinary fragmentation machinery, exactly the way
      //  ActiveMessage<T>'s chunked mode does
      const size_t total = q.payload.size();
      const uint32_t total_chunks =
          static_cast<uint32_t>((total + frag_chunk_size - 1) / frag_chunk_size);
      const uint64_t msg_id = next_frag_msg_id++;
      size_t offset = 0;
      for(uint32_t chunk_id = 0; chunk_id < total_chunks; chunk_id++) {
        const size_t chunk_size = std::min(frag_chunk_size, total - offset);
        QueuedFragment f;
        f.from = cur;
        f.to = relay;
        f.hdr.frag_info = {chunk_id, total_chunks, msg_id};
        f.hdr.user.env = env;
        f.hdr.user.to = relay;
        f.chunk.assign(q.payload.begin() + offset,
                       q.payload.begin() + offset + chunk_size);
        fragment_queue.push_back(f);
        offset += chunk_size;
      }
      num_fragments_sent += total_chunks;
    }

    virtual bool can_send_original(size_t /*hdr_size*/, size_t /*payload_size*/) const
    {
      return allow_unicast_fastpath;
    }

    virtual void send_original(NodeID target, ActiveMessageHandlerTable::MessageID msgid,
                               const void *hdr, size_t hdr_size, const void *payload,
                               size_t payload_size,
                               const MulticastCompletionToken *completion)
    {
      note_send(cur);
      TraceEvent ev;
      ev.kind = TraceEvent::SEND_ORIGINAL;
      ev.node = cur;
      ev.peer = target;
      g_trace.push_back(ev);

      QueuedUnicast q;
      q.from = cur;
      q.to = target;
      q.msgid = msgid;
      const unsigned char *h = static_cast<const unsigned char *>(hdr);
      q.hdr.assign(h, h + hdr_size);
      if(payload_size > 0) {
        const unsigned char *p = static_cast<const unsigned char *>(payload);
        q.payload.assign(p, p + payload_size);
      }
      if(completion != nullptr) {
        q.has_completion = true;
        q.completion = *completion;
      }
      unicast_queue.push_back(q);
      sends_per_node[cur]++;
      num_unicasts++;
    }

    virtual void send_ack(NodeID from, NodeID parent, NodeID origin,
                          uint64_t multicast_id)
    {
      note_send(from);
      TraceEvent ev;
      ev.kind = TraceEvent::SEND_ACK;
      ev.node = from;
      ev.peer = parent;
      g_trace.push_back(ev);

      QueuedAck q;
      q.from = from;
      q.to = parent;
      q.ack.multicast_id = multicast_id;
      q.ack.origin_node = origin;
      ack_queue.push_back(q);
      acks_per_node[from]++;
      num_acks++;
    }

    virtual void deliver_local(NodeID origin, ActiveMessageHandlerTable::MessageID msgid,
                               const void *hdr, size_t hdr_size, const void *payload,
                               size_t payload_size, TimeLimit work_until,
                               const MulticastCompletionToken *completion)
    {
      delivered_senders[cur] = origin;
      num_local_deliveries++;
      bool handled =
          MulticastForwarder::dispatch_local(&mgr, origin, msgid, hdr, hdr_size, payload,
                                             payload_size, work_until, completion);
      if(!handled)
        deferred_queued++;
    }

    virtual MulticastCompletionState &completion_state(void)
    {
      assert((cur >= 0) && (cur < nodes));
      return *node_completion[cur];
    }

    // --- driving the simulation -----------------------------------------

    // issues the origin-side multicast but does NOT deliver anything yet
    void start(NodeID origin, const MulticastTargetSet &targets,
               ActiveMessageHandlerTable::MessageID msgid, const void *hdr,
               size_t hdr_size, const void *payload = nullptr, size_t payload_size = 0,
               MulticastCompletionCallback *on_remote_complete = nullptr)
    {
      cur = origin;
      g_current_node = origin;
      MulticastForwarder::send(*this, targets, msgid, hdr, hdr_size, payload,
                               payload_size, TimeLimit(), &metrics, on_remote_complete);
    }

    // called by SimEnvelopeMessage::handle_message once a fragmented envelope has been
    //  reassembled by the real IncomingMessageManager
    void receive_reassembled_envelope(NodeID sender, const SimEnvelopeMessage &hdr,
                                      const void *payload, size_t payload_size,
                                      TimeLimit work_until)
    {
      assert(pending_envelope_handlers > 0);
      pending_envelope_handlers--;
      num_reassembled++;
      cur = hdr.to;
      g_current_node = hdr.to;
      MulticastForwarder::forward(*this, sender, hdr.env, payload, payload_size,
                                  work_until, &metrics);
    }

    bool anything_queued(void) const
    {
      return (!envelope_queue.empty() || !unicast_queue.empty() ||
              !fragment_queue.empty() || !ack_queue.empty());
    }

    void run(void)
    {
      while(anything_queued()) {
        if(!envelope_queue.empty()) {
          QueuedEnvelope q = envelope_queue.front();
          envelope_queue.pop_front();
          cur = q.to;
          g_current_node = q.to;
          MulticastForwarder::forward(*this, q.from, q.env, q.payload.data(),
                                      q.payload.size(), TimeLimit(), &metrics);
        } else if(!fragment_queue.empty()) {
          QueuedFragment q = fragment_queue.front();
          fragment_queue.pop_front();
          cur = q.to;
          g_current_node = q.to;
          // the real reassembly path: only the final chunk produces a handler call, and
          //  that handler is not inline, so it needs the ordinary drain below
          pending_envelope_handlers++;
          bool handled = mgr.add_incoming_message(
              q.from, sim_envelope_msgid(), &q.hdr, sizeof(q.hdr), PAYLOAD_COPY,
              q.chunk.data(), q.chunk.size(), PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
          EXPECT_FALSE(handled) << "the envelope handler must not be an inline handler";
          if(q.hdr.frag_info.chunk_id + 1 < q.hdr.frag_info.total_chunks) {
            // incomplete - no handler was queued for this chunk
            pending_envelope_handlers--;
          } else {
            drain_envelope_handlers();
          }
        } else if(!unicast_queue.empty()) {
          QueuedUnicast q = unicast_queue.front();
          unicast_queue.pop_front();
          cur = q.to;
          g_current_node = q.to;
          // an ordinary unicast presents the node that sent it as the sender
          delivered_senders[q.to] = q.from;
          num_local_deliveries++;
          bool handled = MulticastForwarder::dispatch_local(
              &mgr, q.from, q.msgid, q.hdr.data(), q.hdr.size(),
              q.payload.empty() ? nullptr : q.payload.data(), q.payload.size(),
              TimeLimit(), q.has_completion ? &q.completion : nullptr);
          if(!handled)
            deferred_queued++;
        } else {
          QueuedAck q = ack_queue.front();
          ack_queue.pop_front();
          // sampled the moment the acknowledgement is consumed: plan section 7.5 says
          //  a relay reclaims its state before it acknowledges, so the acking node must
          //  already be holding nothing
          reclaim_log.push_back(
              std::make_pair(q.from, node_completion[q.from]->num_pending()));
          cur = q.to;
          g_current_node = q.to;
          MulticastForwarder::handle_ack(*this, q.from, q.ack);
        }
      }
    }

    void multicast(NodeID origin, const MulticastTargetSet &targets,
                   ActiveMessageHandlerTable::MessageID msgid, const void *hdr,
                   size_t hdr_size, const void *payload = nullptr,
                   size_t payload_size = 0,
                   MulticastCompletionCallback *on_remote_complete = nullptr)
    {
      start(origin, targets, msgid, hdr, hdr_size, payload, payload_size,
            on_remote_complete);
      run();
    }

    // runs whatever local deliveries could not be handled inline
    void drain_deferred(void)
    {
      while(deferred_queued > 0) {
        size_t before = g_handled.size();
        mgr.do_work(TimeLimit());
        size_t progressed = g_handled.size() - before;
        if(progressed == 0)
          break;
        deferred_queued -= std::min(deferred_queued, progressed);
      }
    }

    void drain_envelope_handlers(void)
    {
      while(pending_envelope_handlers > 0) {
        size_t before = pending_envelope_handlers;
        mgr.do_work(TimeLimit());
        if(pending_envelope_handlers == before)
          break;
      }
    }

    // alternates transmission and deferred handling until nothing is left anywhere,
    //  which is what a completion-tracked multicast with deferred handlers needs (an
    //  acknowledgement is only produced once a handler has actually run)
    void run_to_quiescence(void)
    {
      while(true) {
        if(anything_queued()) {
          run();
          continue;
        }
        if(deferred_queued > 0) {
          size_t before = deferred_queued;
          drain_deferred();
          if(deferred_queued == before)
            break; // no progress possible
          continue;
        }
        break;
      }
    }

    size_t max_fan_out(void) const
    {
      size_t worst = 0;
      for(std::map<NodeID, size_t>::const_iterator it = sends_per_node.begin();
          it != sends_per_node.end(); ++it)
        worst = std::max(worst, it->second);
      return worst;
    }

    size_t total_sends(void) const
    {
      size_t total = 0;
      for(std::map<NodeID, size_t>::const_iterator it = sends_per_node.begin();
          it != sends_per_node.end(); ++it)
        total += it->second;
      return total;
    }

    size_t total_pending_completions(void) const
    {
      size_t total = 0;
      for(size_t i = 0; i < node_completion.size(); i++)
        total += node_completion[i]->num_pending();
      return total;
    }

    size_t peak_pending_completions(void) const
    {
      size_t peak = 0;
      for(size_t i = 0; i < node_completion.size(); i++)
        peak = std::max(peak, node_completion[i]->peak_pending());
      return peak;
    }

    static ActiveMessageHandlerTable::MessageID sim_envelope_msgid(void)
    {
      return activemsg_handler_table
          .lookup_message_id<WrappedWithFragInfo<SimEnvelopeMessage>>();
    }

    CoreReservationSet crs;
    IncomingMessageManager mgr;
    TestMulticastCounters counters;
    TestMulticastMetrics metrics;

    NodeID nodes;
    size_t fan_radix;
    NodeID cur = 0;
    bool allow_unicast_fastpath = true;
    // 0 disables the simulated fragmentation path entirely
    size_t frag_chunk_size = 0;

    std::deque<QueuedEnvelope> envelope_queue;
    std::deque<QueuedUnicast> unicast_queue;
    std::deque<QueuedFragment> fragment_queue;
    std::deque<QueuedAck> ack_queue;
    std::vector<QueuedEnvelope> all_envelopes;
    std::vector<std::unique_ptr<MulticastCompletionState>> node_completion;
    std::map<NodeID, size_t> sends_per_node;
    std::map<NodeID, size_t> acks_per_node;
    std::map<NodeID, NodeID> delivered_senders;
    std::vector<std::pair<NodeID, size_t>> reclaim_log;
    size_t num_unicasts = 0;
    size_t num_local_deliveries = 0;
    size_t deferred_queued = 0;
    size_t num_acks = 0;
    size_t num_fragments_sent = 0;
    size_t num_reassembled = 0;
    size_t pending_envelope_handlers = 0;
    uint64_t next_frag_msg_id = 1;
    unsigned max_depth_seen = 0;
  };

  /*static*/ void SimEnvelopeMessage::handle_message(NodeID sender,
                                                     const SimEnvelopeMessage &hdr,
                                                     const void *payload,
                                                     size_t payload_size,
                                                     TimeLimit work_until)
  {
    ASSERT_NE(g_sim, nullptr);
    g_sim->receive_reassembled_envelope(sender, hdr, payload, payload_size, work_until);
  }

  class RecordingFatalReporter : public MulticastFatalReporter {
  public:
    virtual void report(const MulticastFatalContext &ctx)
    {
      contexts.push_back(ctx);
      descriptions.push_back(ctx.to_string());
    }

    std::vector<MulticastFatalContext> contexts;
    std::vector<std::string> descriptions;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // shared assertions
  //

  // no node sends a child envelope after it has delivered the original message locally
  //  (plan section 7.3 step 4)
  void expect_forward_before_deliver(void)
  {
    std::map<NodeID, size_t> handled_at;
    for(size_t i = 0; i < g_trace.size(); i++)
      if((g_trace[i].kind == TraceEvent::HANDLED) &&
         (handled_at.count(g_trace[i].node) == 0))
        handled_at[g_trace[i].node] = i;

    for(size_t i = 0; i < g_trace.size(); i++) {
      if(g_trace[i].kind == TraceEvent::HANDLED)
        continue;
      std::map<NodeID, size_t>::const_iterator it = handled_at.find(g_trace[i].node);
      if(it != handled_at.end())
        EXPECT_LT(i, it->second)
            << "node " << g_trace[i].node << " sent to " << g_trace[i].peer
            << " only after it had already delivered the message locally";
    }
  }

  // the deepest envelope the forwarding recurrence can produce: the origin hands out
  //  slices of ceil(M/R), a relay removes itself and splits the remaining s-1
  unsigned max_possible_depth(size_t remote_targets, size_t radix)
  {
    unsigned depth = 0;
    size_t s = remote_targets;
    while(s > 0) {
      s = (s + radix - 1) / radix;
      depth++;
      s -= 1;
    }
    return depth;
  }

  ActiveMessageHandlerTable::MessageID test_msgid(void)
  {
    return activemsg_handler_table.lookup_message_id<McastTestMessage>();
  }

  class MulticastForwardTest : public ::testing::Test {
  protected:
    void SetUp(void)
    {
      // the handler table is built from a static registration list, so one build is
      //  enough no matter which suite gets there first
      static bool table_built = false;
      if(!table_built) {
        activemsg_handler_table.construct_handler_table();
        table_built = true;
      }
      reset_trace();
    }

    void TearDown(void) { set_multicast_fatal_reporter(nullptr); }
  };

  // one multicast, plus every property plan sections 7.3/7.4/23 demand of it
  void check_multicast(NodeID num_nodes, size_t radix, NodeID origin,
                       const MulticastTargetSet &targets)
  {
    SCOPED_TRACE(testing::Message() << "nodes=" << num_nodes << " radix=" << radix
                                    << " origin=" << origin << " targets=" << targets);
    reset_trace();
    SimMulticastNetwork net(num_nodes, radix);

    McastTestMessage hdr;
    hdr.value = 0x5EED;
    net.multicast(origin, targets, test_msgid(), &hdr, sizeof(hdr));

    // exactly one delivery per target, and every handler saw the ORIGIN as its sender
    std::map<NodeID, int> per_node;
    for(size_t i = 0; i < g_handled.size(); i++) {
      EXPECT_EQ(g_handled[i].sender, origin);
      EXPECT_EQ(g_handled[i].value, 0x5EED);
      per_node[g_handled[i].node]++;
    }
    EXPECT_EQ(g_handled.size(), targets.size());
    EXPECT_EQ(per_node.size(), targets.size());
    for(MulticastTargetSet::const_iterator it = targets.begin(); it != targets.end();
        ++it)
      EXPECT_EQ(per_node[*it], 1) << "node " << *it << " was not delivered exactly once";

    // origin and relay fan-out are both bounded by R
    EXPECT_LE(net.max_fan_out(), radix);

    // one edge per remote target: total tree edges are O(M)
    const size_t remote = targets.size() - (targets.contains(origin) ? 1 : 0);
    EXPECT_EQ(net.total_sends(), remote);

    EXPECT_LE(net.max_depth_seen, max_possible_depth(remote, radix));

    expect_forward_before_deliver();

    // a fire-and-forget multicast keeps no acknowledgement state anywhere, ever
    EXPECT_EQ(net.peak_pending_completions(), 0u);
    EXPECT_EQ(net.num_acks, 0u);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // completion helpers (plan section 7.5)
  //

  // Records when (and how often) the origin's remote-completion callback ran, and how
  //  many callback objects are still alive - the forwarding layer takes ownership, so
  //  'callbacks_alive' must be back to zero on every path, including the paths where
  //  the callback is never invoked at all.
  struct CompletionProbe {
    size_t invocations = 0;
    size_t tick = 0;
    // handler invocations that had already happened when the callback ran
    size_t handled_when_invoked = 0;
    int callbacks_alive = 0;
  };

  class ProbeCallable {
  public:
    explicit ProbeCallable(CompletionProbe *_probe)
      : probe(_probe)
    {
      probe->callbacks_alive++;
    }
    ProbeCallable(const ProbeCallable &rhs)
      : probe(rhs.probe)
    {
      probe->callbacks_alive++;
    }
    ~ProbeCallable(void) { probe->callbacks_alive--; }

    void operator()(void) const
    {
      probe->invocations++;
      probe->tick = next_tick();
      probe->handled_when_invoked = g_handled.size();
    }

  private:
    ProbeCallable &operator=(const ProbeCallable &) = delete;
    CompletionProbe *probe;
  };

  MulticastCompletionCallback *probe_callback(CompletionProbe *probe)
  {
    return make_multicast_completion(ProbeCallable(probe));
  }

  // Builds an envelope by hand, so that malformed completion metadata can be fed to a
  //  relay without a real origin-side record being left behind.
  struct BuiltEnvelope {
    MulticastEnvelopeMessage env;
    std::vector<unsigned char> payload;
  };

  BuiltEnvelope build_envelope(const MulticastTargetSet &slice, NodeID num_nodes,
                               NodeID origin, uint64_t multicast_id, unsigned depth,
                               const McastTestMessage &hdr, uint32_t flags,
                               const std::vector<unsigned char> &completion_meta)
  {
    BuiltEnvelope built;
    EncodedMulticastTargets enc = EncodedMulticastTargets::encode(slice, num_nodes);

    built.env.multicast_id = multicast_id;
    built.env.origin_node = origin;
    built.env.original_payload_size = 0;
    built.env.target_encoding_size = static_cast<uint32_t>(enc.bytes());
    built.env.completion_size = static_cast<uint32_t>(completion_meta.size());
    built.env.flags = flags;
    built.env.depth = depth;
    built.env.original_message_id = test_msgid();
    built.env.original_header_size = sizeof(McastTestMessage);
    built.env.target_encoding_kind = static_cast<unsigned char>(enc.kind());

    built.payload.insert(built.payload.end(), enc.wire_bytes().begin(),
                         enc.wire_bytes().end());
    const unsigned char *h = reinterpret_cast<const unsigned char *>(&hdr);
    built.payload.insert(built.payload.end(), h, h + sizeof(McastTestMessage));
    built.payload.insert(built.payload.end(), completion_meta.begin(),
                         completion_meta.end());
    return built;
  }

}; // namespace

////////////////////////////////////////////////////////////////////////
//
// empty / singleton / origin membership
//

TEST_F(MulticastForwardTest, EmptyTargetSetIsSuccessfulNoOp)
{
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  EXPECT_TRUE(g_trace.empty());
  EXPECT_TRUE(g_handled.empty());
  EXPECT_EQ(net.total_sends(), 0u);
  EXPECT_EQ(net.counters.multicast_first_hops.get(), 0u);
}

TEST_F(MulticastForwardTest, SingletonRemoteTargetUsesUnicastFastPath)
{
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  targets.add(5);
  McastTestMessage hdr;
  hdr.value = 77;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  // no envelope at all - the origin already is the sender the handler must see
  EXPECT_EQ(net.all_envelopes.size(), 0u);
  EXPECT_EQ(net.num_unicasts, 1u);
  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 5);
  EXPECT_EQ(g_handled[0].sender, 0);
  EXPECT_EQ(g_handled[0].value, 77);
  EXPECT_EQ(net.counters.multicast_first_hops.get(), 1u);
  // the fast path builds no target encoding at all
  EXPECT_EQ(net.counters.multicast_encoding_single.get(), 0u);
}

TEST_F(MulticastForwardTest, SingletonFallsBackToEnvelopeWhenUnicastUnavailable)
{
  // a payload that would need fragmenting cannot ride an untyped unicast, so the
  //  envelope path has to take over (and the origin is still the apparent sender)
  SimMulticastNetwork net(8, 4);
  net.allow_unicast_fastpath = false;
  MulticastTargetSet targets;
  targets.add(5);
  McastTestMessage hdr;
  hdr.value = 78;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  EXPECT_EQ(net.all_envelopes.size(), 1u);
  EXPECT_EQ(net.num_unicasts, 0u);
  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 5);
  EXPECT_EQ(g_handled[0].sender, 0);
  EXPECT_EQ(net.counters.multicast_encoding_single.get(), 1u);
}

TEST_F(MulticastForwardTest, OriginIsTheOnlyTarget)
{
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  targets.add(3);
  McastTestMessage hdr;
  hdr.value = 9;
  net.multicast(3, targets, test_msgid(), &hdr, sizeof(hdr));

  EXPECT_EQ(net.total_sends(), 0u);
  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 3);
  EXPECT_EQ(g_handled[0].sender, 3);
}

TEST_F(MulticastForwardTest, OriginIncludedAndExcluded)
{
  MulticastTargetSet with_origin;
  with_origin.add_range(0, 15);
  check_multicast(16, 4, 0, with_origin);

  MulticastTargetSet without_origin;
  without_origin.add_range(1, 15);
  check_multicast(16, 4, 0, without_origin);

  // origin in the middle of the set, and origin entirely outside it
  MulticastTargetSet middle;
  middle.add_range(0, 15);
  check_multicast(16, 4, 7, middle);

  MulticastTargetSet elsewhere;
  elsewhere.add_range(8, 15);
  check_multicast(16, 4, 2, elsewhere);
}

////////////////////////////////////////////////////////////////////////
//
// radix sweeps
//

TEST_F(MulticastForwardTest, RadixOneDegeneratesToAChain)
{
  MulticastTargetSet targets;
  targets.add_range(1, 8);
  check_multicast(16, 1, 0, targets);

  // with R == 1 every relay has exactly one child, so the tree is a chain of depth M
  reset_trace();
  SimMulticastNetwork net(16, 1);
  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));
  EXPECT_EQ(net.max_fan_out(), 1u);
  EXPECT_EQ(net.max_depth_seen, 8u);
}

TEST_F(MulticastForwardTest, RadixTwoFourAndSixteen)
{
  const size_t radices[] = {1, 2, 4, 16, 64};
  for(size_t r = 0; r < sizeof(radices) / sizeof(radices[0]); r++) {
    MulticastTargetSet contiguous;
    contiguous.add_range(0, 63);
    check_multicast(64, radices[r], 0, contiguous);

    MulticastTargetSet sparse;
    for(NodeID id = 1; id < 64; id += 3)
      sparse.add(id);
    check_multicast(64, radices[r], 0, sparse);

    MulticastTargetSet irregular;
    irregular.add_range(2, 5);
    irregular.add(11);
    irregular.add_range(30, 47);
    irregular.add(63);
    check_multicast(64, radices[r], 17, irregular);
  }
}

TEST_F(MulticastForwardTest, LargeTargetSetKeepsOriginFanOutBounded)
{
  // plan section 19 exit criterion: a source sends at most R first-hop messages for a
  //  large target set
  const size_t radix = 4;
  reset_trace();
  SimMulticastNetwork net(1024, radix);
  MulticastTargetSet targets;
  targets.add_range(0, 1023);

  McastTestMessage hdr;
  hdr.value = 4242;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  EXPECT_EQ(net.sends_per_node[0], radix);
  EXPECT_EQ(net.counters.multicast_first_hops.get(), radix);
  EXPECT_LE(net.max_fan_out(), radix);
  EXPECT_EQ(g_handled.size(), 1024u);
  EXPECT_EQ(net.total_sends(), 1023u);
  // O(log_R M) depth, not O(M)
  EXPECT_LE(net.max_depth_seen, max_possible_depth(1023, radix));
  EXPECT_LE(net.max_depth_seen, 6u);

  std::set<NodeID> seen;
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].sender, 0);
    EXPECT_TRUE(seen.insert(g_handled[i].node).second)
        << "node " << g_handled[i].node << " was delivered twice";
  }
  EXPECT_EQ(seen.size(), 1024u);
}

TEST_F(MulticastForwardTest, RandomizedShapesDeliverExactlyOnce)
{
  std::mt19937 rng(1357);
  const size_t radices[] = {1, 2, 3, 4, 8};
  for(int trial = 0; trial < 20; trial++) {
    const NodeID num_nodes = 96;
    MulticastTargetSet targets;
    const unsigned density = 5 + (rng() % 95);
    for(NodeID id = 0; id < num_nodes; id++)
      if((rng() % 100) < density)
        targets.add(id);
    if(targets.empty())
      targets.add(static_cast<NodeID>(rng() % num_nodes));

    const NodeID origin = static_cast<NodeID>(rng() % num_nodes);
    const size_t radix = radices[rng() % (sizeof(radices) / sizeof(radices[0]))];
    check_multicast(num_nodes, radix, origin, targets);
  }
}

////////////////////////////////////////////////////////////////////////
//
// ordering, sender preservation, payloads
//

TEST_F(MulticastForwardTest, RelayForwardsBeforeInvokingLocalHandler)
{
  reset_trace();
  SimMulticastNetwork net(32, 2);
  MulticastTargetSet targets;
  targets.add_range(0, 31);
  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  expect_forward_before_deliver();

  // and make sure the test is not vacuous: at least one node both forwarded and
  //  delivered, and its forwards really are earlier in the trace
  size_t nodes_that_did_both = 0;
  std::map<NodeID, size_t> handled_at, last_send_at;
  for(size_t i = 0; i < g_trace.size(); i++) {
    if(g_trace[i].kind == TraceEvent::HANDLED)
      handled_at[g_trace[i].node] = i;
    else
      last_send_at[g_trace[i].node] = i;
  }
  for(std::map<NodeID, size_t>::const_iterator it = handled_at.begin();
      it != handled_at.end(); ++it) {
    std::map<NodeID, size_t>::const_iterator s = last_send_at.find(it->first);
    if(s == last_send_at.end())
      continue;
    nodes_that_did_both++;
    EXPECT_LT(s->second, it->second);
  }
  EXPECT_GE(nodes_that_did_both, 2u);
}

TEST_F(MulticastForwardTest, OriginalSenderSurvivesMultipleHops)
{
  reset_trace();
  SimMulticastNetwork net(64, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 63);
  const NodeID origin = 0;
  McastTestMessage hdr;
  hdr.value = 31337;
  net.multicast(origin, targets, test_msgid(), &hdr, sizeof(hdr));

  // the tree really is more than one hop deep
  EXPECT_GE(net.max_depth_seen, 2u);

  size_t relayed_by_someone_else = 0;
  for(size_t i = 0; i < net.all_envelopes.size(); i++) {
    // every envelope, at every depth, still names the ORIGIN
    EXPECT_EQ(net.all_envelopes[i].env.origin_node, origin);
    if(net.all_envelopes[i].from != origin) {
      relayed_by_someone_else++;
      // ... even though the node that physically transmitted it is not the origin
      EXPECT_NE(net.all_envelopes[i].from, origin);
    }
  }
  EXPECT_GT(relayed_by_someone_else, 0u);

  ASSERT_EQ(g_handled.size(), 63u);
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].sender, origin)
        << "node " << g_handled[i].node << " saw the wrong sender";
    EXPECT_EQ(g_handled[i].value, 31337);
  }

  // and the transport-level record agrees for every node that is not the origin
  for(std::map<NodeID, NodeID>::const_iterator it = net.delivered_senders.begin();
      it != net.delivered_senders.end(); ++it)
    EXPECT_EQ(it->second, origin);
}

TEST_F(MulticastForwardTest, HeaderOnlyAndCopiedPayloadMessages)
{
  // header only
  {
    reset_trace();
    SimMulticastNetwork net(16, 3);
    MulticastTargetSet targets;
    targets.add_range(0, 15);
    McastTestMessage hdr;
    hdr.value = 11;
    net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));
    ASSERT_EQ(g_handled.size(), 16u);
    for(size_t i = 0; i < g_handled.size(); i++) {
      EXPECT_EQ(g_handled[i].value, 11);
      EXPECT_TRUE(g_handled[i].payload.empty());
    }
  }

  // simple copied 1-D payload
  {
    reset_trace();
    SimMulticastNetwork net(16, 3);
    MulticastTargetSet targets;
    targets.add_range(0, 15);

    std::vector<char> payload(197);
    for(size_t i = 0; i < payload.size(); i++)
      payload[i] = static_cast<char>((i * 7) & 0xff);

    McastTestMessage hdr;
    hdr.value = 12;
    net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(),
                  payload.size());

    ASSERT_EQ(g_handled.size(), 16u);
    for(size_t i = 0; i < g_handled.size(); i++) {
      EXPECT_EQ(g_handled[i].value, 12);
      EXPECT_EQ(g_handled[i].payload, payload)
          << "node " << g_handled[i].node << " got the wrong payload";
    }
  }
}

TEST_F(MulticastForwardTest, PayloadOutlivesTheCallerBuffer)
{
  // plan section 7.5: the payload is copied into the first-hop envelopes before the
  //  origin-side call returns, so the caller may reuse its buffer immediately
  reset_trace();
  SimMulticastNetwork net(16, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 15);

  std::vector<char> payload(64, 'A');
  McastTestMessage hdr;
  hdr.value = 13;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(), payload.size());

  // scribble over the caller's buffer and the caller's header before anything is
  //  actually delivered
  std::fill(payload.begin(), payload.end(), 'Z');
  hdr.value = -1;

  net.run();

  const std::vector<char> expected(64, 'A');
  ASSERT_EQ(g_handled.size(), 15u);
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].value, 13);
    EXPECT_EQ(g_handled[i].payload, expected);
  }
}

TEST_F(MulticastForwardTest, DeferredHandlerPathIsUsedWhenThereIsNoInlineHandler)
{
  reset_trace();
  SimMulticastNetwork net(16, 2);
  MulticastTargetSet targets;
  targets.add_range(0, 15);

  McastDeferredMessage hdr;
  hdr.value = 4711;
  ActiveMessageHandlerTable::MessageID msgid =
      activemsg_handler_table.lookup_message_id<McastDeferredMessage>();
  net.multicast(0, targets, msgid, &hdr, sizeof(hdr));

  // nothing could be handled inline, so everything is sitting in the queue
  EXPECT_EQ(net.num_local_deliveries, 16u);
  EXPECT_EQ(net.deferred_queued, 16u);
  EXPECT_TRUE(g_handled.empty());

  net.drain_deferred();

  EXPECT_EQ(g_handled.size(), 16u);
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].sender, 0) << "deferred delivery lost the original sender";
    EXPECT_EQ(g_handled[i].value, 4711);
  }
}

////////////////////////////////////////////////////////////////////////
//
// envelope contents and identity
//

TEST_F(MulticastForwardTest, EnvelopeCarriesEverythingPlan74Requires)
{
  reset_trace();
  SimMulticastNetwork net(32, 4);
  MulticastTargetSet targets;
  targets.add_range(1, 31);

  std::vector<char> payload(40, 'p');
  McastTestMessage hdr;
  hdr.value = 5150;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(), payload.size());

  ASSERT_EQ(net.all_envelopes.size(), 4u);
  const uint64_t id = net.all_envelopes[0].env.multicast_id;
  EXPECT_NE(id, 0u);
  for(size_t i = 0; i < net.all_envelopes.size(); i++) {
    const MulticastEnvelopeMessage &env = net.all_envelopes[i].env;
    EXPECT_EQ(env.origin_node, 0);
    EXPECT_EQ(env.multicast_id, id) << "one multicast must use one ID";
    EXPECT_EQ(env.original_message_id, test_msgid());
    EXPECT_EQ(env.original_header_size, sizeof(McastTestMessage));
    EXPECT_EQ(env.original_payload_size, payload.size());
    EXPECT_EQ(env.completion_size, 0u);
    EXPECT_EQ(env.flags, 0u);
    EXPECT_EQ(env.depth, 1u);
    EXPECT_GT(env.target_encoding_size, 0u);
    EXPECT_EQ(net.all_envelopes[i].payload.size(), env.target_encoding_size +
                                                       env.original_header_size +
                                                       env.original_payload_size);
    // the kind byte in the header agrees with the encoded slice that follows
    EXPECT_EQ(net.all_envelopes[i].payload[0], env.target_encoding_kind);
    // and the relay is the first node of the slice it was sent
    MulticastTargetSet slice;
    ASSERT_EQ(EncodedMulticastTargets::decode(net.all_envelopes[i].payload.data(),
                                              env.target_encoding_size, 32, slice),
              MulticastDecodeStatus::OK);
    EXPECT_EQ(slice.first_node(), net.all_envelopes[i].to);
    EXPECT_TRUE(slice.contains(net.all_envelopes[i].to));
  }

  net.run();

  // a second multicast gets a different ID
  reset_trace();
  SimMulticastNetwork net2(32, 4);
  net2.start(0, targets, test_msgid(), &hdr, sizeof(hdr));
  ASSERT_FALSE(net2.all_envelopes.empty());
  EXPECT_NE(net2.all_envelopes[0].env.multicast_id, id);
  net2.run();
}

TEST_F(MulticastForwardTest, EnvelopeHandlerIsRegisteredAndDeliberatelyNotInline)
{
  ActiveMessageHandlerTable::MessageID id =
      activemsg_handler_table.lookup_message_id<MulticastEnvelopeMessage>();
  ActiveMessageHandlerTable::HandlerEntry *entry =
      activemsg_handler_table.lookup_message_handler(id);
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->handler, nullptr);
  // plan section 22: child sends must not be issued recursively out of an inline
  //  handler, so the envelope type deliberately provides no inline handler
  EXPECT_EQ(entry->handler_inline, nullptr);

  // the fragmentation wrapper is auto-registered too, so an oversized envelope is
  //  chunked and reassembled by the existing machinery on every hop
  ActiveMessageHandlerTable::MessageID wrapped_id =
      activemsg_handler_table
          .lookup_message_id<WrappedWithFragInfo<MulticastEnvelopeMessage>>();
  EXPECT_NE(wrapped_id, id);
  ActiveMessageHandlerTable::HandlerEntry *wrapped =
      activemsg_handler_table.lookup_message_handler(wrapped_id);
  ASSERT_NE(wrapped, nullptr);
  EXPECT_NE(wrapped->handler, nullptr);

  // the envelope itself carries no FragmentInfo of its own
  EXPECT_FALSE(is_wrapped_with_frag_info<MulticastEnvelopeMessage>::value);
  EXPECT_FALSE(entry->extract_frag_info.has_value());
}

TEST_F(MulticastForwardTest, ChildSlicesAreDisjointAndCoverTheParent)
{
  reset_trace();
  SimMulticastNetwork net(64, 3);
  MulticastTargetSet targets;
  targets.add_range(0, 63);
  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  // union of every slice ever transmitted, counted with multiplicity: each node must
  //  appear in exactly one slice per tree level it is part of, and in particular each
  //  node is the relay of exactly one envelope
  std::map<NodeID, int> relay_count;
  for(size_t i = 0; i < net.all_envelopes.size(); i++)
    relay_count[net.all_envelopes[i].to]++;
  EXPECT_EQ(relay_count.size(), 63u);
  for(std::map<NodeID, int>::const_iterator it = relay_count.begin();
      it != relay_count.end(); ++it)
    EXPECT_EQ(it->second, 1) << "node " << it->first << " received two envelopes";

  // siblings at the first hop are disjoint
  std::set<NodeID> first_hop_union;
  size_t first_hop_total = 0;
  for(size_t i = 0; i < net.all_envelopes.size(); i++) {
    if(net.all_envelopes[i].env.depth != 1)
      continue;
    MulticastTargetSet slice;
    ASSERT_EQ(EncodedMulticastTargets::decode(
                  net.all_envelopes[i].payload.data(),
                  net.all_envelopes[i].env.target_encoding_size, 64, slice),
              MulticastDecodeStatus::OK);
    first_hop_total += slice.size();
    for(MulticastTargetSet::const_iterator it = slice.begin(); it != slice.end(); ++it)
      EXPECT_TRUE(first_hop_union.insert(*it).second)
          << "node " << *it << " appears in two sibling slices";
  }
  EXPECT_EQ(first_hop_total, 63u);
  EXPECT_EQ(first_hop_union.size(), 63u);
}

////////////////////////////////////////////////////////////////////////
//
// fatal diagnostics (plan section 21.1)
//

namespace {

  // runs one multicast far enough to capture a real first-hop envelope
  struct CapturedEnvelope {
    MulticastEnvelopeMessage env;
    std::vector<unsigned char> payload;
    NodeID from = 0;
    NodeID to = 0;
  };

  CapturedEnvelope capture_first_hop(SimMulticastNetwork &net, NodeID origin,
                                     const MulticastTargetSet &targets)
  {
    McastTestMessage hdr;
    hdr.value = 1;
    net.start(origin, targets, test_msgid(), &hdr, sizeof(hdr));
    CapturedEnvelope captured;
    captured.env = net.all_envelopes.at(0).env;
    captured.payload = net.all_envelopes.at(0).payload;
    captured.from = net.all_envelopes.at(0).from;
    captured.to = net.all_envelopes.at(0).to;
    net.envelope_queue.clear();
    net.unicast_queue.clear();
    return captured;
  }

}; // namespace

TEST_F(MulticastForwardTest, MalformedTargetEncodingIsReportedAndDropped)
{
  SimMulticastNetwork net(32, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 31);
  CapturedEnvelope captured = capture_first_hop(net, 0, targets);

  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  // an impossible kind byte
  std::vector<unsigned char> corrupt = captured.payload;
  corrupt[0] = 200;
  MulticastEnvelopeMessage env = captured.env;
  env.target_encoding_kind = 200;

  reset_trace();
  net.cur = captured.to;
  g_current_node = captured.to;
  MulticastForwarder::forward(net, captured.from, env, corrupt.data(), corrupt.size(),
                              TimeLimit());

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_EQ(reporter.contexts[0].status, MulticastDecodeStatus::UNKNOWN_KIND);
  EXPECT_STREQ(reporter.contexts[0].rule, "malformed multicast target encoding");
  EXPECT_EQ(reporter.contexts[0].origin_node, 0);
  EXPECT_EQ(reporter.contexts[0].sender, captured.from);
  EXPECT_EQ(reporter.contexts[0].local_node, captured.to);
  EXPECT_NE(reporter.descriptions[0].find("malformed multicast target encoding"),
            std::string::npos);
  // nothing was delivered and nothing was forwarded
  EXPECT_TRUE(g_handled.empty());
  EXPECT_TRUE(net.envelope_queue.empty());
}

TEST_F(MulticastForwardTest, TruncatedTargetEncodingIsReportedAndDropped)
{
  SimMulticastNetwork net(32, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 31);
  CapturedEnvelope captured = capture_first_hop(net, 0, targets);

  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  // claim one more encoded byte than the payload actually holds
  MulticastEnvelopeMessage env = captured.env;
  env.target_encoding_size += 1;

  reset_trace();
  net.cur = captured.to;
  MulticastForwarder::forward(net, captured.from, env, captured.payload.data(),
                              captured.payload.size(), TimeLimit());

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast envelope length fields do not match the received payload");
  EXPECT_TRUE(g_handled.empty());
}

TEST_F(MulticastForwardTest, EncodingKindDisagreementIsReportedAndDropped)
{
  SimMulticastNetwork net(32, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 31);
  CapturedEnvelope captured = capture_first_hop(net, 0, targets);

  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastEnvelopeMessage env = captured.env;
  env.target_encoding_kind =
      static_cast<unsigned char>((captured.env.target_encoding_kind + 1) % 8);

  reset_trace();
  net.cur = captured.to;
  MulticastForwarder::forward(net, captured.from, env, captured.payload.data(),
                              captured.payload.size(), TimeLimit());

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast envelope encoding kind disagrees with its payload");
  EXPECT_TRUE(g_handled.empty());
}

TEST_F(MulticastForwardTest, RelayMissingFromItsOwnSliceIsReportedAndDropped)
{
  SimMulticastNetwork net(32, 2);
  MulticastTargetSet targets;
  targets.add_range(1, 15);
  CapturedEnvelope captured = capture_first_hop(net, 0, targets);

  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  // deliver a perfectly well-formed envelope to a node that is not in its slice
  reset_trace();
  net.cur = 31;
  g_current_node = 31;
  MulticastForwarder::forward(net, captured.from, captured.env, captured.payload.data(),
                              captured.payload.size(), TimeLimit());

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast relay is not a member of the slice it was sent");
  EXPECT_EQ(reporter.contexts[0].local_node, 31);
  EXPECT_TRUE(g_handled.empty());
  EXPECT_TRUE(net.envelope_queue.empty());
}

TEST_F(MulticastForwardTest, TargetOutsideTheConfiguredNodeCountIsReported)
{
  SimMulticastNetwork net(8, 2);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastTargetSet targets;
  targets.add(1);
  targets.add(99); // >= num_nodes

  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_EQ(reporter.contexts[0].status, MulticastDecodeStatus::NODE_OUT_OF_RANGE);
  EXPECT_TRUE(g_handled.empty());
  EXPECT_EQ(net.total_sends(), 0u);
}

////////////////////////////////////////////////////////////////////////
//
// metrics (plan section 21.3)
//

TEST_F(MulticastForwardTest, CountersRecordFanOutDepthAndEncodingChoices)
{
  reset_trace();
  const size_t radix = 4;
  SimMulticastNetwork net(64, radix);
  MulticastTargetSet targets;
  targets.add_range(1, 63);
  McastTestMessage hdr;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  const TestMulticastCounters &c = net.counters;

  // one first-hop count per origin-side send, bounded by R (plan section 23)
  EXPECT_EQ(c.multicast_first_hops.get(), radix);

  // one encoding choice per envelope
  const uint64_t encodings =
      c.multicast_encoding_empty.get() + c.multicast_encoding_single.get() +
      c.multicast_encoding_small_inline.get() + c.multicast_encoding_ranges.get() +
      c.multicast_encoding_delta_list.get() + c.multicast_encoding_bitmap.get() +
      c.multicast_encoding_all_nodes.get() + c.multicast_encoding_all_except.get();
  EXPECT_EQ(encodings, net.all_envelopes.size());
  EXPECT_EQ(encodings, 63u);

  // a contiguous target set uses the compact range encoding for its big slices
  EXPECT_GE(c.multicast_encoding_ranges.get(), radix);
  // and the leaves are single-node slices
  EXPECT_GT(c.multicast_encoding_single.get(), 0u);

  // depth is a gauge, and matches what the transport observed
  EXPECT_EQ(c.multicast_max_depth.get(), net.max_depth_seen);
  EXPECT_GE(c.multicast_max_depth.get(), 2u);
  EXPECT_LE(c.multicast_max_depth.get(), max_possible_depth(63, radix));
}

// NOTE: a test named BarrierMulticastMetricsAdapterWiring used to live here.  It covered
//  BarrierMulticastMetrics, the BARRIER-side implementation of MulticastMetricsSink, and
//  went with the barrier protocol when that was reverted.  The multicast layer's own side
//  of that contract - that it records an encoding choice per envelope, a first-hop count
//  per origin send, and a max tree depth - is covered by the counter assertions in
//  AcceptanceCriteriaScalingTable and friends, using this file's TestMulticastMetrics.

////////////////////////////////////////////////////////////////////////
//
// Stage 2c: payload, fragmentation and transient completion semantics
//  (plan sections 7.5, 20.1)
//

namespace {

  // exact payload bytes each target's handler saw, keyed by node
  std::map<NodeID, std::vector<char>> payload_per_node(void)
  {
    std::map<NodeID, std::vector<char>> result;
    for(size_t i = 0; i < g_handled.size(); i++)
      result[g_handled[i].node] = g_handled[i].payload;
    return result;
  }

  size_t count_trace(TraceEvent::Kind kind)
  {
    size_t n = 0;
    for(size_t i = 0; i < g_trace.size(); i++)
      if(g_trace[i].kind == kind)
        n++;
    return n;
  }

}; // namespace

////////////////////////////////////////////////////////////////////////
//
// payload semantics
//

TEST_F(MulticastForwardTest, PayloadBytesRoundTripAtEveryTarget)
{
  reset_trace();
  SimMulticastNetwork net(24, 3);
  MulticastTargetSet targets;
  targets.add_range(0, 23);

  std::vector<char> payload(613);
  for(size_t i = 0; i < payload.size(); i++)
    payload[i] = static_cast<char>((i * 31 + 7) & 0xff);

  McastTestMessage hdr;
  hdr.value = 0x2C2C;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(),
                payload.size());

  ASSERT_EQ(g_handled.size(), 24u);
  std::map<NodeID, std::vector<char>> seen = payload_per_node();
  ASSERT_EQ(seen.size(), 24u);
  for(NodeID n = 0; n < 24; n++) {
    ASSERT_EQ(seen.count(n), 1u) << "node " << n << " was never delivered";
    EXPECT_EQ(seen[n], payload) << "node " << n << " got the wrong payload bytes";
  }
}

TEST_F(MulticastForwardTest, HeaderOnlyMulticastCarriesNoPayloadAnywhere)
{
  reset_trace();
  SimMulticastNetwork net(24, 3);
  MulticastTargetSet targets;
  targets.add_range(0, 23);

  McastTestMessage hdr;
  hdr.value = 0x1234;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  ASSERT_EQ(g_handled.size(), 24u);
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].value, 0x1234);
    EXPECT_TRUE(g_handled[i].payload.empty());
  }
  for(size_t i = 0; i < net.all_envelopes.size(); i++)
    EXPECT_EQ(net.all_envelopes[i].env.original_payload_size, 0u);
}

TEST_F(MulticastForwardTest, PayloadOutlivesTheCallerBufferOnTheUnicastFastPath)
{
  // the singleton fast path copies too - the caller's PAYLOAD_KEEP guarantee is that
  //  the buffer is free the moment the origin-side call returns (plan section 7.5)
  reset_trace();
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  targets.add(3);

  std::vector<char> payload(37, 'k');
  McastTestMessage hdr;
  hdr.value = 55;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(), payload.size());

  EXPECT_EQ(net.num_unicasts, 1u);
  EXPECT_TRUE(net.all_envelopes.empty());

  std::fill(payload.begin(), payload.end(), 'X');
  hdr.value = -7;

  net.run();

  const std::vector<char> expected(37, 'k');
  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 3);
  EXPECT_EQ(g_handled[0].sender, 0);
  EXPECT_EQ(g_handled[0].value, 55);
  EXPECT_EQ(g_handled[0].payload, expected);
}

////////////////////////////////////////////////////////////////////////
//
// fragmentation (plan section 7.5)
//

TEST_F(MulticastForwardTest, OversizedEnvelopesAreFragmentedAndReassembledOnEveryHop)
{
  reset_trace();
  SimMulticastNetwork net(16, 3);
  // far below the envelope size, so every hop has to fragment
  net.frag_chunk_size = 64;

  MulticastTargetSet targets;
  targets.add_range(0, 15);

  std::vector<char> payload(1000);
  for(size_t i = 0; i < payload.size(); i++)
    payload[i] = static_cast<char>((i * 13 + 5) & 0xff);

  McastTestMessage hdr;
  hdr.value = 0xF00D;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(),
                payload.size());

  // every envelope really did go through the fragmentation machinery, and the whole
  //  envelope was reassembled before the relay repartitioned it
  ASSERT_EQ(net.all_envelopes.size(), 15u);
  EXPECT_EQ(net.num_reassembled, 15u);
  EXPECT_GE(net.num_fragments_sent, 15u * 16u);

  // and the payload survived intact at every one of the 16 targets
  ASSERT_EQ(g_handled.size(), 16u);
  std::map<NodeID, std::vector<char>> seen = payload_per_node();
  ASSERT_EQ(seen.size(), 16u);
  for(NodeID n = 0; n < 16; n++) {
    ASSERT_EQ(seen.count(n), 1u) << "node " << n << " was never delivered";
    EXPECT_EQ(seen[n], payload) << "node " << n << " got a corrupted payload";
  }
  for(size_t i = 0; i < g_handled.size(); i++)
    EXPECT_EQ(g_handled[i].sender, 0) << "fragmentation lost the original sender";

  // the shape of the tree is unchanged by fragmentation
  EXPECT_LE(net.max_fan_out(), 3u);
  EXPECT_EQ(net.total_sends(), 15u);
  expect_forward_before_deliver();
}

TEST_F(MulticastForwardTest, FragmentedAndUnfragmentedDeliveriesAgree)
{
  std::vector<char> payload(700);
  for(size_t i = 0; i < payload.size(); i++)
    payload[i] = static_cast<char>((i * 17 + 3) & 0xff);

  McastTestMessage hdr;
  hdr.value = 4242;

  std::map<NodeID, std::vector<char>> whole;
  {
    reset_trace();
    SimMulticastNetwork net(12, 4);
    MulticastTargetSet targets;
    targets.add_range(0, 11);
    net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(),
                  payload.size());
    ASSERT_EQ(g_handled.size(), 12u);
    EXPECT_EQ(net.num_reassembled, 0u);
    whole = payload_per_node();
  }

  std::map<NodeID, std::vector<char>> fragmented;
  {
    reset_trace();
    SimMulticastNetwork net(12, 4);
    net.frag_chunk_size = 100;
    MulticastTargetSet targets;
    targets.add_range(0, 11);
    net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(),
                  payload.size());
    ASSERT_EQ(g_handled.size(), 12u);
    EXPECT_GT(net.num_reassembled, 0u);
    fragmented = payload_per_node();
  }

  EXPECT_EQ(whole, fragmented);
  EXPECT_EQ(whole.size(), 12u);
}

////////////////////////////////////////////////////////////////////////
//
// aggregate remote completion (plan section 7.5)
//

TEST_F(MulticastForwardTest, FireAndForgetKeepsNoAcknowledgementStateOrMetadata)
{
  reset_trace();
  SimMulticastNetwork net(32, 4);
  MulticastTargetSet targets;
  targets.add_range(0, 31);
  McastTestMessage hdr;
  hdr.value = 9;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  ASSERT_EQ(g_handled.size(), 32u);

  // no acknowledgement traffic, no acknowledgement records - not even transiently
  EXPECT_EQ(net.num_acks, 0u);
  EXPECT_EQ(count_trace(TraceEvent::SEND_ACK), 0u);
  EXPECT_EQ(net.peak_pending_completions(), 0u);
  EXPECT_EQ(net.total_pending_completions(), 0u);

  // and no acknowledgement metadata on the wire either
  ASSERT_EQ(net.all_envelopes.size(), 31u);
  for(size_t i = 0; i < net.all_envelopes.size(); i++) {
    const MulticastEnvelopeMessage &env = net.all_envelopes[i].env;
    EXPECT_EQ(env.flags, 0u);
    EXPECT_EQ(env.completion_size, 0u);
    EXPECT_EQ(net.all_envelopes[i].payload.size(), env.target_encoding_size +
                                                       env.original_header_size +
                                                       env.original_payload_size);
  }
}

TEST_F(MulticastForwardTest, RemoteCompletionFiresExactlyOnceAfterEveryTargetHandled)
{
  reset_trace();
  SimMulticastNetwork net(32, 4);
  MulticastTargetSet targets;
  targets.add_range(0, 31);

  CompletionProbe probe;
  McastTestMessage hdr;
  hdr.value = 77;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));

  // the origin retains the callback state under the multicast ID while the tree is
  //  still in flight, and has not fired anything yet
  EXPECT_EQ(net.total_pending_completions(), 1u);
  EXPECT_EQ(probe.invocations, 0u);

  net.run_to_quiescence();

  EXPECT_EQ(g_handled.size(), 32u);
  ASSERT_EQ(probe.invocations, 1u) << "the origin callback must run exactly once";
  EXPECT_EQ(probe.callbacks_alive, 0) << "the completion callback object was leaked";
  EXPECT_EQ(probe.handled_when_invoked, 32u)
      << "the callback ran before every target had handled the message";
  EXPECT_GT(probe.tick, g_last_handled_tick);

  // one acknowledgement per relay, and never two from the same node
  EXPECT_EQ(net.num_acks, net.all_envelopes.size());
  EXPECT_EQ(net.num_acks, 31u);
  for(std::map<NodeID, size_t>::const_iterator it = net.acks_per_node.begin();
      it != net.acks_per_node.end(); ++it)
    EXPECT_EQ(it->second, 1u) << "node " << it->first << " acknowledged twice";

  // every acknowledgement went to the node that had sent that relay its envelope
  std::map<NodeID, NodeID> parent_of;
  for(size_t i = 0; i < net.all_envelopes.size(); i++)
    parent_of[net.all_envelopes[i].to] = net.all_envelopes[i].from;
  size_t acks_seen = 0;
  for(size_t i = 0; i < g_trace.size(); i++) {
    if(g_trace[i].kind != TraceEvent::SEND_ACK)
      continue;
    acks_seen++;
    ASSERT_EQ(parent_of.count(g_trace[i].node), 1u);
    EXPECT_EQ(g_trace[i].peer, parent_of[g_trace[i].node])
        << "node " << g_trace[i].node << " acknowledged the wrong parent";
  }
  EXPECT_EQ(acks_seen, 31u);

  // every acking node had already reclaimed its record when its parent saw the ack
  ASSERT_EQ(net.reclaim_log.size(), 31u);
  for(size_t i = 0; i < net.reclaim_log.size(); i++)
    EXPECT_EQ(net.reclaim_log[i].second, 0u)
        << "node " << net.reclaim_log[i].first
        << " still held state after acknowledging its parent";

  // ... and nothing at all is left anywhere
  EXPECT_GT(net.peak_pending_completions(), 0u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, CompletionTrackedEnvelopesNameTheNodeToAcknowledge)
{
  reset_trace();
  SimMulticastNetwork net(32, 4);
  MulticastTargetSet targets;
  targets.add_range(1, 31);

  CompletionProbe probe;
  McastTestMessage hdr;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));
  net.run_to_quiescence();
  ASSERT_EQ(probe.invocations, 1u);

  ASSERT_FALSE(net.all_envelopes.empty());
  for(size_t i = 0; i < net.all_envelopes.size(); i++) {
    const SimMulticastNetwork::QueuedEnvelope &q = net.all_envelopes[i];
    EXPECT_NE(q.env.flags & MulticastEnvelopeFlags::COMPLETION_TRACKED, 0u);
    ASSERT_GT(q.env.completion_size, 0u);
    EXPECT_EQ(q.payload.size(), q.env.target_encoding_size + q.env.original_header_size +
                                    q.env.original_payload_size + q.env.completion_size);
    // the metadata is a single varint naming the node this subtree acknowledges to,
    //  which is always the node that sent the envelope
    size_t pos = 0;
    uint64_t parent = 0;
    const unsigned char *meta =
        q.payload.data() + q.payload.size() - q.env.completion_size;
    ASSERT_EQ(MulticastWire::read_varint(meta, q.env.completion_size, pos, parent),
              MulticastDecodeStatus::OK);
    EXPECT_EQ(pos, q.env.completion_size);
    EXPECT_EQ(static_cast<NodeID>(parent), q.from);
  }
}

TEST_F(MulticastForwardTest, NoResidualStateAfterACompletionTrackedMulticast)
{
  reset_trace();
  SimMulticastNetwork net(16, 3);
  MulticastTargetSet targets;
  targets.add_range(0, 15);

  for(int round = 0; round < 3; round++) {
    CompletionProbe probe;
    McastTestMessage hdr;
    hdr.value = round;
    net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
                  probe_callback(&probe));
    net.run_to_quiescence();
    EXPECT_EQ(probe.invocations, 1u);
    EXPECT_EQ(probe.callbacks_alive, 0);

    // plan section 2 (final bullet) and 7.1: no per-multicast state may survive, on any
    //  node - not a target plan, not an encoding, not an acknowledgement record
    for(size_t n = 0; n < net.node_completion.size(); n++)
      EXPECT_EQ(net.node_completion[n]->num_pending(), 0u)
          << "node " << n << " retained multicast state after round " << round;
  }
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, RemoteCompletionOnAnEmptyTargetSetFiresImmediately)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets; // empty

  CompletionProbe probe;
  McastTestMessage hdr;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));

  // every one of the zero targets has trivially handled it
  EXPECT_EQ(probe.invocations, 1u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
  EXPECT_EQ(net.peak_pending_completions(), 0u);
  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(net.num_unicasts, 0u);
  EXPECT_EQ(net.num_acks, 0u);
  EXPECT_TRUE(g_handled.empty());

  net.run();
  EXPECT_EQ(probe.invocations, 1u);
  EXPECT_EQ(probe.callbacks_alive, 0);
}

TEST_F(MulticastForwardTest, RemoteCompletionOnASingletonTargetUsesTheUnicastFastPath)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  targets.add(5);

  CompletionProbe probe;
  McastTestMessage hdr;
  hdr.value = 31337;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));

  // still the ordinary unicast fast path - no envelope, no acknowledgement message
  EXPECT_EQ(net.num_unicasts, 1u);
  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(probe.invocations, 0u);
  EXPECT_EQ(net.total_pending_completions(), 1u);

  net.run_to_quiescence();

  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 5);
  EXPECT_EQ(g_handled[0].sender, 0);
  EXPECT_EQ(probe.invocations, 1u);
  EXPECT_EQ(probe.handled_when_invoked, 1u);
  EXPECT_EQ(net.num_acks, 0u) << "the fast path rides the ordinary remote completion";
  EXPECT_EQ(net.total_pending_completions(), 0u);
  EXPECT_EQ(probe.callbacks_alive, 0);
}

TEST_F(MulticastForwardTest, RemoteCompletionWhenTheOriginIsTheOnlyTarget)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  MulticastTargetSet targets;
  targets.add(0);

  CompletionProbe probe;
  McastTestMessage hdr;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));

  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(net.num_unicasts, 0u);
  ASSERT_EQ(g_handled.size(), 1u);
  EXPECT_EQ(g_handled[0].node, 0);
  // the local handler is inline, so the whole thing is already finished
  EXPECT_EQ(probe.invocations, 1u);
  EXPECT_EQ(probe.callbacks_alive, 0);
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, RemoteCompletionWaitsForDeferredHandlers)
{
  reset_trace();
  SimMulticastNetwork net(16, 3);
  MulticastTargetSet targets;
  targets.add_range(0, 15);

  CompletionProbe probe;
  McastDeferredMessage hdr;
  hdr.value = 8888;
  ActiveMessageHandlerTable::MessageID msgid =
      activemsg_handler_table.lookup_message_id<McastDeferredMessage>();
  net.start(0, targets, msgid, &hdr, sizeof(hdr), nullptr, 0, probe_callback(&probe));
  net.run();

  // the whole tree has been transmitted, but nothing has been HANDLED yet, so no
  //  acknowledgement may have been produced and the callback must not have fired
  EXPECT_EQ(net.num_local_deliveries, 16u);
  EXPECT_EQ(net.deferred_queued, 16u);
  EXPECT_TRUE(g_handled.empty());
  EXPECT_EQ(net.num_acks, 0u);
  EXPECT_EQ(probe.invocations, 0u);
  EXPECT_EQ(net.total_pending_completions(), 16u);

  net.run_to_quiescence();

  EXPECT_EQ(g_handled.size(), 16u);
  for(size_t i = 0; i < g_handled.size(); i++)
    EXPECT_EQ(g_handled[i].sender, 0);
  ASSERT_EQ(probe.invocations, 1u);
  EXPECT_EQ(probe.handled_when_invoked, 16u);
  EXPECT_GT(probe.tick, g_last_handled_tick);
  EXPECT_EQ(net.num_acks, 15u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
  EXPECT_EQ(probe.callbacks_alive, 0);
}

TEST_F(MulticastForwardTest, RemoteCompletionWithAFragmentedPayload)
{
  reset_trace();
  SimMulticastNetwork net(16, 4);
  net.frag_chunk_size = 96;
  MulticastTargetSet targets;
  targets.add_range(0, 15);

  std::vector<char> payload(1500);
  for(size_t i = 0; i < payload.size(); i++)
    payload[i] = static_cast<char>((i * 5 + 11) & 0xff);

  CompletionProbe probe;
  McastTestMessage hdr;
  hdr.value = 606;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), payload.data(), payload.size(),
            probe_callback(&probe));
  net.run_to_quiescence();

  EXPECT_GT(net.num_reassembled, 0u);
  ASSERT_EQ(g_handled.size(), 16u);
  std::map<NodeID, std::vector<char>> seen = payload_per_node();
  for(NodeID n = 0; n < 16; n++) {
    ASSERT_EQ(seen.count(n), 1u);
    EXPECT_EQ(seen[n], payload);
  }
  ASSERT_EQ(probe.invocations, 1u);
  EXPECT_EQ(probe.callbacks_alive, 0);
  EXPECT_EQ(probe.handled_when_invoked, 16u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, RemoteCompletionAcrossRadicesOriginsAndShapes)
{
  const NodeID num_nodes = 24;
  const size_t radices[] = {1, 2, 3, 4, 8, 64};
  const NodeID origins[] = {0, 7, 23};

  for(size_t r = 0; r < (sizeof(radices) / sizeof(radices[0])); r++) {
    for(size_t o = 0; o < (sizeof(origins) / sizeof(origins[0])); o++) {
      for(int shape = 0; shape < 4; shape++) {
        MulticastTargetSet targets;
        switch(shape) {
        case 0:
          targets.add_range(0, num_nodes - 1); // everything, origin included
          break;
        case 1:
          for(NodeID n = 0; n < num_nodes; n++) // origin excluded
            if(n != origins[o])
              targets.add(n);
          break;
        case 2:
          for(NodeID n = 1; n < num_nodes; n += 3)
            targets.add(n);
          break;
        case 3:
          targets.add(origins[o] == 11 ? 12 : 11); // singleton
          break;
        }
        SCOPED_TRACE(testing::Message()
                     << "radix=" << radices[r] << " origin=" << origins[o]
                     << " shape=" << shape << " targets=" << targets);
        reset_trace();
        SimMulticastNetwork net(num_nodes, radices[r]);
        CompletionProbe probe;
        McastTestMessage hdr;
        hdr.value = shape;
        net.multicast(origins[o], targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
                      probe_callback(&probe));
        net.run_to_quiescence();

        EXPECT_EQ(g_handled.size(), targets.size());
        EXPECT_EQ(probe.invocations, 1u);
        EXPECT_EQ(probe.callbacks_alive, 0);
        EXPECT_EQ(probe.handled_when_invoked, targets.size());
        EXPECT_EQ(net.total_pending_completions(), 0u);
        EXPECT_LE(net.max_fan_out(), radices[r]);
        // exactly one acknowledgement per envelope, and never more
        EXPECT_EQ(net.num_acks, net.all_envelopes.size());
        for(size_t i = 0; i < net.reclaim_log.size(); i++)
          EXPECT_EQ(net.reclaim_log[i].second, 0u);
      }
    }
  }
}

TEST_F(MulticastForwardTest, RandomizedCompletionTrackedMulticasts)
{
  std::mt19937 rng(20250729);
  const NodeID num_nodes = 40;

  for(int trial = 0; trial < 20; trial++) {
    const size_t radix = 1 + (rng() % 6);
    const NodeID origin = static_cast<NodeID>(rng() % num_nodes);
    const int density = 1 + (rng() % 100);

    MulticastTargetSet targets;
    for(NodeID n = 0; n < num_nodes; n++)
      if(static_cast<int>(rng() % 100) < density)
        targets.add(n);
    if(targets.empty())
      targets.add(static_cast<NodeID>(rng() % num_nodes));

    SCOPED_TRACE(testing::Message() << "trial=" << trial << " radix=" << radix
                                    << " origin=" << origin << " targets=" << targets);
    reset_trace();
    SimMulticastNetwork net(num_nodes, radix);
    CompletionProbe probe;
    McastTestMessage hdr;
    hdr.value = trial;
    net.multicast(origin, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
                  probe_callback(&probe));
    net.run_to_quiescence();

    std::map<NodeID, int> per_node;
    for(size_t i = 0; i < g_handled.size(); i++) {
      EXPECT_EQ(g_handled[i].sender, origin);
      per_node[g_handled[i].node]++;
    }
    EXPECT_EQ(g_handled.size(), targets.size());
    for(MulticastTargetSet::const_iterator it = targets.begin(); it != targets.end();
        ++it)
      EXPECT_EQ(per_node[*it], 1);

    EXPECT_EQ(probe.invocations, 1u);
    EXPECT_EQ(probe.callbacks_alive, 0);
    EXPECT_EQ(probe.handled_when_invoked, targets.size());
    EXPECT_EQ(net.total_pending_completions(), 0u);
    EXPECT_LE(net.max_fan_out(), radix);
    EXPECT_EQ(net.num_acks, net.all_envelopes.size());
  }
}

////////////////////////////////////////////////////////////////////////
//
// completion-related fatal diagnostics (plan section 21.1)
//

TEST_F(MulticastForwardTest, UnknownEnvelopeFlagsAreReportedAndDropped)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastTargetSet slice;
  slice.add_range(2, 5);
  McastTestMessage hdr;
  BuiltEnvelope built =
      build_envelope(slice, 8, 0, 42, 1, hdr, 0x8000u /*not a real flag*/, ByteVec());

  net.cur = 2;
  g_current_node = 2;
  MulticastForwarder::forward(net, 0, built.env, built.payload.data(),
                              built.payload.size(), TimeLimit(), &net.metrics);

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule, "multicast envelope carries unknown flags");
  EXPECT_TRUE(g_handled.empty());
  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, CompletionMetadataWithoutTheFlagIsReportedAndDropped)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastTargetSet slice;
  slice.add_range(2, 5);
  McastTestMessage hdr;
  ByteVec meta;
  put_varint(meta, 0);
  BuiltEnvelope built = build_envelope(slice, 8, 0, 42, 1, hdr, 0 /*no flag*/, meta);

  net.cur = 2;
  g_current_node = 2;
  MulticastForwarder::forward(net, 0, built.env, built.payload.data(),
                              built.payload.size(), TimeLimit(), &net.metrics);

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast envelope carries completion metadata without the completion "
               "flag");
  EXPECT_TRUE(g_handled.empty());
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, MalformedCompletionMetadataIsReportedAndDropped)
{
  MulticastTargetSet slice;
  slice.add_range(2, 5);
  McastTestMessage hdr;

  struct Case {
    const char *what;
    ByteVec meta;
  };
  std::vector<Case> cases;
  {
    // a node ID outside the configured node count
    Case c;
    c.what = "node out of range";
    put_varint(c.meta, 8);
    cases.push_back(c);
  }
  {
    // two varints where exactly one is expected
    Case c;
    c.what = "trailing metadata";
    put_varint(c.meta, 0);
    put_varint(c.meta, 1);
    cases.push_back(c);
  }
  {
    // truncated varint
    Case c;
    c.what = "truncated";
    c.meta.push_back(0x81);
    cases.push_back(c);
  }
  {
    // overlong (noncanonical) varint
    Case c;
    c.what = "overlong";
    c.meta.push_back(0x80);
    c.meta.push_back(0x00);
    cases.push_back(c);
  }

  for(size_t i = 0; i < cases.size(); i++) {
    SCOPED_TRACE(cases[i].what);
    reset_trace();
    SimMulticastNetwork net(8, 4);
    RecordingFatalReporter reporter;
    set_multicast_fatal_reporter(&reporter);

    BuiltEnvelope built =
        build_envelope(slice, 8, 0, 42, 1, hdr,
                       MulticastEnvelopeFlags::COMPLETION_TRACKED, cases[i].meta);
    net.cur = 2;
    g_current_node = 2;
    MulticastForwarder::forward(net, 0, built.env, built.payload.data(),
                                built.payload.size(), TimeLimit(), &net.metrics);

    ASSERT_EQ(reporter.contexts.size(), 1u);
    EXPECT_STREQ(reporter.contexts[0].rule, "malformed multicast completion metadata");
    EXPECT_TRUE(g_handled.empty());
    EXPECT_TRUE(net.all_envelopes.empty());
    EXPECT_EQ(net.total_pending_completions(), 0u);
    set_multicast_fatal_reporter(nullptr);
  }
}

TEST_F(MulticastForwardTest, CompletionParentThatIsNotTheSenderIsReportedAndDropped)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastTargetSet slice;
  slice.add_range(2, 5);
  McastTestMessage hdr;
  ByteVec meta;
  put_varint(meta, 1); // claims node 1 is the parent
  BuiltEnvelope built = build_envelope(slice, 8, 0, 42, 1, hdr,
                                       MulticastEnvelopeFlags::COMPLETION_TRACKED, meta);

  net.cur = 2;
  g_current_node = 2;
  // ... but the envelope actually arrived from node 0
  MulticastForwarder::forward(net, 0, built.env, built.payload.data(),
                              built.payload.size(), TimeLimit(), &net.metrics);

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast completion metadata names a parent that is not the sender");
  EXPECT_TRUE(g_handled.empty());
  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, ARejectedOriginSideMulticastNeitherRunsNorLeaksItsCallback)
{
  reset_trace();
  SimMulticastNetwork net(4, 2);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  MulticastTargetSet targets;
  targets.add(9); // outside the configured node count

  CompletionProbe probe;
  McastTestMessage hdr;
  net.start(0, targets, test_msgid(), &hdr, sizeof(hdr), nullptr, 0,
            probe_callback(&probe));

  ASSERT_EQ(reporter.contexts.size(), 1u);
  // nothing was sent, so nothing "completed" - but the callback must still be reclaimed
  EXPECT_EQ(probe.invocations, 0u);
  EXPECT_EQ(probe.callbacks_alive, 0) << "a rejected multicast leaked its callback";
  EXPECT_TRUE(net.all_envelopes.empty());
  EXPECT_EQ(net.num_unicasts, 0u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
  EXPECT_EQ(net.peak_pending_completions(), 0u);
}

TEST_F(MulticastForwardTest, AcknowledgementForAnUnknownMulticastIsReported)
{
  reset_trace();
  SimMulticastNetwork net(8, 4);
  RecordingFatalReporter reporter;
  set_multicast_fatal_reporter(&reporter);

  net.cur = 0;
  g_current_node = 0;
  MulticastAckMessage ack;
  ack.origin_node = 0;
  ack.multicast_id = 999;
  MulticastForwarder::handle_ack(net, 3, ack);

  ASSERT_EQ(reporter.contexts.size(), 1u);
  EXPECT_STREQ(reporter.contexts[0].rule,
               "multicast acknowledgement does not match any multicast in flight");
  EXPECT_EQ(reporter.contexts[0].multicast_id, 999u);
  EXPECT_EQ(net.total_pending_completions(), 0u);
}

////////////////////////////////////////////////////////////////////////
//
// registration of the acknowledgement message
//

TEST_F(MulticastForwardTest, AckMessageIsRegisteredAndDeliberatelyNotInline)
{
  ActiveMessageHandlerTable::MessageID id =
      activemsg_handler_table.lookup_message_id<MulticastAckMessage>();
  ActiveMessageHandlerTable::HandlerEntry *entry =
      activemsg_handler_table.lookup_message_handler(id);
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->handler, nullptr);
  // handling an acknowledgement can send the next one up the tree, which plan section
  //  22 forbids doing recursively out of an inline handler
  EXPECT_EQ(entry->handler_inline, nullptr);
}

////////////////////////////////////////////////////////////////////////
//
// stage 2d: migrated multicast users and the plan's stage 2 exit criteria
//

namespace {

  ActiveMessageHandlerTable::MessageID shutdown_msgid(void)
  {
    return activemsg_handler_table.lookup_message_id<McastShutdownMessage>();
  }

  // exactly the target set RuntimeImpl::initiate_shutdown() builds: every node except
  //  the shutdown master, which is always node 0
  MulticastTargetSet shutdown_targets(NodeID num_nodes)
  {
    MulticastTargetSet targets;
    if(num_nodes > 1)
      targets.add_range(1, num_nodes - 1);
    return targets;
  }

  // number of messages 'node' transmitted, of any kind
  size_t sends_by(NodeID node)
  {
    size_t count = 0;
    for(size_t i = 0; i < g_trace.size(); i++)
      if((g_trace[i].kind != TraceEvent::HANDLED) && (g_trace[i].node == node))
        count++;
    return count;
  }

  // every slice of a contiguous target set is itself contiguous, so it must be one run
  //  and - once it is big enough for RANGES to beat the tied-size list encodings - must
  //  have been encoded as RANGES (plan section 7.2 and the stage 2 exit criteria)
  void expect_contiguous_slices_use_ranges(const SimMulticastNetwork &net)
  {
    size_t checked = 0;
    for(size_t i = 0; i < net.all_envelopes.size(); i++) {
      const SimMulticastNetwork::QueuedEnvelope &q = net.all_envelopes[i];
      MulticastTargetSet slice;
      ASSERT_EQ(EncodedMulticastTargets::decode(
                    q.payload.data(), q.env.target_encoding_size, net.nodes, slice),
                MulticastDecodeStatus::OK);
      ASSERT_EQ(slice.num_ranges(), 1u)
          << "a slice of a contiguous target set must stay contiguous: " << slice;
      // for 1 or 2 nodes SINGLE/SMALL_INLINE tie with (or beat) RANGES on size and win
      //  the tie-break; from 3 nodes up RANGES is strictly the smallest
      if(slice.size() >= 3) {
        EXPECT_EQ(static_cast<MulticastTargetEncoding>(q.env.target_encoding_kind),
                  MulticastTargetEncoding::RANGES)
            << "slice " << slice << " did not use the range encoding";
        checked++;
      }
    }
    EXPECT_GT(checked, 0u) << "no slice was large enough to exercise RANGES";
  }

}; // namespace

// Plan section 7.6: "Runtime shutdown is a critical forwarding-order test.  Relays must
//  forward before delivering shutdown locally."  This drives the REAL
//  MulticastForwarder over the exact target set RuntimeImpl::initiate_shutdown() builds,
//  with a handler that stops its node the instant it runs.
TEST_F(MulticastForwardTest, RuntimeShutdownRelaysForwardBeforeStoppingThemselves)
{
  const NodeID num_nodes = 64;
  const size_t radix = MULTICAST_DEFAULT_RADIX;
  SimMulticastNetwork net(num_nodes, radix);

  MulticastTargetSet targets = shutdown_targets(num_nodes);
  ASSERT_EQ(targets.size(), 63u);

  McastShutdownMessage hdr;
  hdr.result_code = 42;
  net.multicast(0 /*shutdown master*/, targets, shutdown_msgid(), &hdr, sizeof(hdr));

  // (a) nothing was transmitted by a node that had already shut itself down - had any
  //     relay delivered before forwarding, its whole subtree would be stranded
  EXPECT_EQ(g_sends_from_stopped_nodes, 0u);
  expect_forward_before_deliver();

  // (b) every node except the master shut down exactly once, and each of them saw the
  //     shutdown master (not the relay that handed it the message) as the sender
  ASSERT_EQ(g_handled.size(), 63u);
  std::set<NodeID> shut_down;
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].sender, 0)
        << "node " << g_handled[i].node << " did not see the shutdown master as sender";
    EXPECT_EQ(g_handled[i].value, 42);
    EXPECT_TRUE(shut_down.insert(g_handled[i].node).second)
        << "node " << g_handled[i].node << " was shut down twice";
  }
  EXPECT_EQ(shut_down.size(), 63u);
  EXPECT_EQ(shut_down.count(0), 0u) << "the shutdown master must not shut itself down "
                                       "via the multicast";
  EXPECT_EQ(g_stopped_nodes.size(), 63u);

  // (c) the assertion in (a) is not vacuous: plenty of nodes both relayed and stopped
  size_t relayed_then_stopped = 0;
  for(std::set<NodeID>::const_iterator it = g_stopped_nodes.begin();
      it != g_stopped_nodes.end(); ++it)
    if(sends_by(*it) > 0)
      relayed_then_stopped++;
  EXPECT_GE(relayed_then_stopped, 2u)
      << "no node both forwarded and shut down, so the ordering check proved nothing";

  // (d) the master's fan-out is the radix, not the machine size
  EXPECT_EQ(sends_by(0), radix);
  EXPECT_LE(net.max_fan_out(), radix);
  // exactly one inbound message per target
  EXPECT_EQ(net.total_sends(), 63u);
}

// The same shutdown pattern across machine sizes and radices, including the degenerate
//  two-node case that takes the unicast fast path.
TEST_F(MulticastForwardTest, RuntimeShutdownScalesAcrossMachineSizesAndRadices)
{
  const NodeID sizes[] = {2, 3, 5, 16, 64, 257, 1024};
  const size_t radices[] = {1, 2, 4, 8};

  for(size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
    for(size_t ri = 0; ri < sizeof(radices) / sizeof(radices[0]); ri++) {
      const NodeID num_nodes = sizes[si];
      const size_t radix = radices[ri];
      SCOPED_TRACE(testing::Message() << "nodes=" << num_nodes << " radix=" << radix);
      reset_trace();
      SimMulticastNetwork net(num_nodes, radix);

      MulticastTargetSet targets = shutdown_targets(num_nodes);
      McastShutdownMessage hdr;
      hdr.result_code = 7;
      net.multicast(0, targets, shutdown_msgid(), &hdr, sizeof(hdr));

      EXPECT_EQ(g_sends_from_stopped_nodes, 0u);
      expect_forward_before_deliver();
      EXPECT_EQ(g_handled.size(), static_cast<size_t>(num_nodes - 1));
      EXPECT_EQ(g_stopped_nodes.size(), static_cast<size_t>(num_nodes - 1));
      for(size_t i = 0; i < g_handled.size(); i++)
        EXPECT_EQ(g_handled[i].sender, 0);
      // origin fan-out is bounded by the radix and total traffic is O(M)
      EXPECT_LE(sends_by(0), radix);
      EXPECT_LE(net.max_fan_out(), radix);
      EXPECT_EQ(net.total_sends(), static_cast<size_t>(num_nodes - 1));
    }
  }
}

// Plan section 19, stage 2 exit criteria, all three asserted on one large contiguous
//  multicast: bounded first-hop count, exactly-once delivery with the original sender,
//  and compact range metadata.
TEST_F(MulticastForwardTest, StageTwoExitCriteriaForALargeContiguousTargetSet)
{
  const NodeID num_nodes = 4096;
  const size_t radix = MULTICAST_DEFAULT_RADIX;
  const size_t num_targets = 2047;
  SimMulticastNetwork net(num_nodes, radix);

  // the lower half of a 4096-node machine: contiguous, but not so nearly-everything
  //  that ALL_EXCEPT would legitimately be smaller than a range
  MulticastTargetSet targets;
  targets.add_range(1, static_cast<NodeID>(num_targets));
  ASSERT_EQ(targets.size(), num_targets);
  ASSERT_EQ(targets.num_ranges(), 1u);

  // "contiguous sets use the range encoding" - the whole set first...
  EncodedMulticastTargets whole = EncodedMulticastTargets::encode(targets, num_nodes);
  EXPECT_EQ(whole.kind(), MulticastTargetEncoding::RANGES);
  EXPECT_LE(whole.bytes(), 8u);

  McastTestMessage hdr;
  hdr.value = 0x2D;
  net.multicast(0, targets, test_msgid(), &hdr, sizeof(hdr));

  // "a source sends at most R first-hop messages for a large target set"
  EXPECT_EQ(sends_by(0), radix);
  EXPECT_LE(net.max_fan_out(), radix);

  // "every target receives exactly one delivery with the original sender"
  ASSERT_EQ(g_handled.size(), num_targets);
  std::set<NodeID> seen;
  for(size_t i = 0; i < g_handled.size(); i++) {
    EXPECT_EQ(g_handled[i].sender, 0);
    EXPECT_EQ(g_handled[i].value, 0x2D);
    EXPECT_TRUE(seen.insert(g_handled[i].node).second)
        << "node " << g_handled[i].node << " was delivered twice";
  }
  EXPECT_EQ(seen.size(), num_targets);
  EXPECT_EQ(net.total_sends(), num_targets);

  // ...and every slice on the wire too
  expect_contiguous_slices_use_ranges(net);

  // plan section 23: no reusable multicast state remains
  EXPECT_EQ(net.total_pending_completions(), 0u);
  EXPECT_EQ(net.peak_pending_completions(), 0u);
}

// "Every node except me" is the shape both the shutdown master and the IPC peer sets
//  use.  It is contiguous when the sender is node 0, but the encoder is free to do
//  better than a range there: ALL_EXCEPT names the one excluded node in three bytes.
//  Either way the metadata is compact and independent of the machine size, which is
//  what the stage 2 exit criterion actually asks for.
TEST_F(MulticastForwardTest, AllButTheSenderEncodesCompactlyAtEveryMachineSize)
{
  const NodeID sizes[] = {2, 64, 4096, 1 << 20};
  for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    const NodeID num_nodes = sizes[i];
    SCOPED_TRACE(testing::Message() << "nodes=" << num_nodes);

    MulticastTargetSet targets = shutdown_targets(num_nodes);
    EncodedMulticastTargets enc = EncodedMulticastTargets::encode(targets, num_nodes);
    EXPECT_LE(enc.bytes(), 8u) << "encoding grew with the machine size";
    if(num_nodes > 2)
      EXPECT_EQ(enc.kind(), MulticastTargetEncoding::ALL_EXCEPT);

    MulticastTargetSet round_trip;
    ASSERT_EQ(enc.decode_into(num_nodes, round_trip), MulticastDecodeStatus::OK);
    EXPECT_EQ(round_trip, targets);
  }
}

// The migrated header-only users (runtime shutdown, HIP IPC request/release,
//  MetadataInvalidateMessage) all send a small header and no payload to a set that is
//  typically NOT contiguous - Network::all_peers/shared_peers and a metadata
//  remote-copy set both exclude the local node and can be sparse.
TEST_F(MulticastForwardTest, HeaderOnlyPeerSetMulticastMatchesMigratedCallSites)
{
  const NodeID num_nodes = 96;
  const size_t radix = MULTICAST_DEFAULT_RADIX;
  const NodeID origin = 37;

  // "all peers except me", exactly like HipModule's ipc_peers and
  //  MetadataBase::remote_copies
  MulticastTargetSet peers;
  peers.add_range(0, num_nodes - 1);
  ASSERT_TRUE(peers.remove(origin));

  SimMulticastNetwork net(num_nodes, radix);
  McastTestMessage hdr;
  hdr.value = 0x19C;
  net.multicast(origin, peers, test_msgid(), &hdr, sizeof(hdr));

  ASSERT_EQ(g_handled.size(), static_cast<size_t>(num_nodes - 1));
  std::set<NodeID> seen;
  for(size_t i = 0; i < g_handled.size(); i++) {
    // the HIP and CUDA IPC handlers reply to 'sender', so origin preservation is what
    //  makes the response go back to the node that actually asked
    EXPECT_EQ(g_handled[i].sender, origin);
    EXPECT_TRUE(g_handled[i].payload.empty());
    EXPECT_TRUE(seen.insert(g_handled[i].node).second);
  }
  EXPECT_EQ(seen.count(origin), 0u);
  EXPECT_LE(sends_by(origin), radix);
  EXPECT_LE(net.max_fan_out(), radix);
  EXPECT_EQ(net.total_sends(), static_cast<size_t>(num_nodes - 1));
  expect_forward_before_deliver();
}

// RegionInstanceImpl::send_metadata and CudaModule's IPC broadcast used to chunk their
//  payload at the source against recommended_max_payload(NodeSet, ...).  They now hand
//  the whole blob to the multicast layer and let the envelope be fragmented per hop, so
//  a large copied 1-D payload has to arrive intact at every target either way.
TEST_F(MulticastForwardTest, WholeMetadataBlobReachesEveryEarlyRequestor)
{
  const NodeID num_nodes = 32;
  const size_t radix = MULTICAST_DEFAULT_RADIX;

  std::vector<char> blob(9001);
  for(size_t i = 0; i < blob.size(); i++)
    blob[i] = static_cast<char>((i * 31) & 0xff);

  // sparse "early requestors", including node 0 and the last node
  MulticastTargetSet early_reqs;
  for(NodeID i = 0; i < num_nodes; i += 3)
    early_reqs.add(i);
  early_reqs.add(num_nodes - 1);
  // the sender is never one of its own early requestors
  ASSERT_FALSE(early_reqs.contains(5));

  for(int fragmented = 0; fragmented < 2; fragmented++) {
    SCOPED_TRACE(testing::Message() << "fragmented=" << fragmented);
    reset_trace();
    SimMulticastNetwork net(num_nodes, radix);
    net.frag_chunk_size = (fragmented ? 512 : 0);

    McastTestMessage hdr;
    hdr.value = 0xB10B;
    net.multicast(5, early_reqs, test_msgid(), &hdr, sizeof(hdr), blob.data(),
                  blob.size());

    ASSERT_EQ(g_handled.size(), early_reqs.size());
    std::set<NodeID> seen;
    for(size_t i = 0; i < g_handled.size(); i++) {
      EXPECT_EQ(g_handled[i].sender, 5);
      EXPECT_EQ(g_handled[i].payload.size(), blob.size());
      EXPECT_TRUE(std::equal(blob.begin(), blob.end(), g_handled[i].payload.begin()))
          << "node " << g_handled[i].node << " got a corrupted metadata blob";
      EXPECT_TRUE(seen.insert(g_handled[i].node).second);
    }
    EXPECT_LE(net.max_fan_out(), radix);
    if(fragmented)
      EXPECT_GT(net.num_fragments_sent, 0u);
    else
      EXPECT_EQ(net.num_fragments_sent, 0u);
  }
}

////////////////////////////////////////////////////////////////////////
//
// stage 9: multicast performance acceptance criteria, MEASURED (plan section 23)
//
// The multicast half of plan section 23 is five bounds on one multicast of M targets at
//  radix R.  Every case above asserts them for one shape; this one MEASURES them across
//  a grid of (M, R) and prints the numbers, so that "origin fan-out is at most R" is
//  visible as a table that stops growing rather than as a single passing assertion.
//

TEST_F(MulticastForwardTest, AcceptanceCriteriaScalingTable)
{
  const size_t radices[] = {2, 4, 8};
  const size_t target_counts[] = {1, 2, 8, 64, 256, 1024};

  std::cout << "\nplan section 23: active-message multicast, contiguous target set\n"
               "   R      M | origin_out max_relay_out total_sends deliveries depth | "
               "encoding      bytes | pending acks\n";

  for(size_t r = 0; r < 3; r++) {
    for(size_t t = 0; t < 6; t++) {
      const size_t radix = radices[r];
      const size_t num_targets = target_counts[t];
      // deliberately more nodes than targets, so that the contiguous run 1..M is a
      //  strict SUBSET of the machine and the encoder has to describe it as a range
      //  rather than collapsing it to "everyone except the origin"
      const NodeID num_nodes = static_cast<NodeID>(2 * num_targets + 2);
      SCOPED_TRACE(testing::Message() << "R=" << radix << " M=" << num_targets);

      reset_trace();
      SimMulticastNetwork net(num_nodes, radix);
      MulticastTargetSet targets;
      targets.add_range(1, static_cast<NodeID>(num_targets));

      const EncodedMulticastTargets encoded =
          EncodedMulticastTargets::encode(targets, num_nodes);

      McastTestMessage hdr;
      hdr.value = 0x9009;
      net.multicast(/*origin=*/0, targets, test_msgid(), &hdr, sizeof(hdr));

      const size_t origin_out = net.sends_per_node.count(0) ? net.sends_per_node[0] : 0;
      size_t relay_out = 0;
      for(std::map<NodeID, size_t>::const_iterator it = net.sends_per_node.begin();
          it != net.sends_per_node.end(); ++it) {
        if(it->first != 0) {
          relay_out = std::max(relay_out, it->second);
        }
      }

      std::cout << "  " << std::setw(2) << radix << " " << std::setw(6) << num_targets
                << " | " << std::setw(10) << origin_out << " " << std::setw(13)
                << relay_out << " " << std::setw(11) << net.total_sends() << " "
                << std::setw(10) << g_handled.size() << " " << std::setw(5)
                << net.max_depth_seen << " | " << std::setw(12)
                << multicast_target_encoding_name(encoded.kind()) << " " << std::setw(5)
                << encoded.bytes() << " | " << std::setw(7)
                << net.peak_pending_completions() << " " << std::setw(4) << net.num_acks
                << "\n";

      // "Origin fan-out is at most R" / "relay fan-out is at most R"
      EXPECT_LE(origin_out, radix);
      EXPECT_LE(relay_out, radix);
      EXPECT_LE(net.max_fan_out(), radix);
      // "Total deliveries are exactly M", each with the ORIGIN as its sender
      EXPECT_EQ(g_handled.size(), num_targets);
      std::set<NodeID> seen;
      for(size_t i = 0; i < g_handled.size(); i++) {
        EXPECT_EQ(g_handled[i].sender, 0);
        EXPECT_TRUE(seen.insert(g_handled[i].node).second)
            << "node " << g_handled[i].node << " was delivered twice";
      }
      EXPECT_EQ(seen.size(), num_targets);
      // one wire message per remote target: the tree has exactly M edges
      EXPECT_EQ(net.total_sends(), num_targets);
      // O(log_R M) hops, not O(M)
      EXPECT_LE(net.max_depth_seen, max_possible_depth(num_targets, radix));
      // "A contiguous target set has compact range metadata" - the encoded size must not
      //  grow with M at all, which is only true if the encoder described the run rather
      //  than listing its members
      EXPECT_GT(encoded.bytes(), 0u);
      EXPECT_LE(encoded.bytes(), size_t(16));
      if(num_targets >= 8) {
        EXPECT_EQ(encoded.kind(), MulticastTargetEncoding::RANGES);
      }
      // "No reusable multicast state remains after completion" - a fire-and-forget
      //  multicast never allocates completion state and never acknowledges
      EXPECT_EQ(net.peak_pending_completions(), 0u);
      EXPECT_EQ(net.num_acks, 0u);
      expect_forward_before_deliver();
    }
  }
}
