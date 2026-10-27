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

// Randomized model of the issue-order bookkeeping for remote subgraph
// instantiations (realm/subgraph/subgraph_order.h): several sending nodes,
// instantiations that finish at random times (which moves each sender's
// base), and a network that delivers each node's messages in a random order
// within a bounded window. The owner must handle every node's messages in
// that node's issue order, hold nothing forever, and keep its out-of-order
// window bounded by what was actually in flight. Also covers the reuse of a
// subgraph id by a new owner-side state while a sender keeps counting.

#include <gtest/gtest.h>
#include "realm/subgraph/subgraph_order.h"

#include <algorithm>
#include <map>
#include <random>
#include <vector>

using namespace Realm;

namespace {

  struct Message {
    int sender;
    uint64_t seq;
    int issue_index; // position in the sender's issue order
  };

  // Delivers `msgs` to one tracker per sender in the given order, handling
  // held messages as they become ready, and records the handling order.
  struct Owner {
    std::map<int, RemoteOrderTracker> trackers;
    std::map<int, std::vector<int>> handled; // sender -> issue indices in handling order
    std::vector<Message> storage;            // payloads for held messages
    size_t max_window = 0;

    void deliver(const Message &m)
    {
      RemoteOrderTracker &t = trackers[m.sender];
      if(t.ready(m.seq, true /*record*/)) {
        handled[m.sender].push_back(m.issue_index);
      } else {
        storage.push_back(m);
        t.hold(m.seq, reinterpret_cast<void *>(storage.size())); // 1-based index
      }
      // this may have been the predecessor an earlier arrival waits for
      while(true) {
        uint64_t seq;
        void *p = t.pop_ready(&seq);
        if(!p)
          break;
        const Message &h = storage[reinterpret_cast<size_t>(p) - 1];
        ASSERT_TRUE(t.ready(h.seq, true /*record*/)); // readiness only grows
        handled[h.sender].push_back(h.issue_index);
      }
      max_window = std::max(max_window, t.window());
    }
  };

  // One sender issuing `count` instantiations, one per time step. Message i
  // is issued at time i, reaches the owner at delivery_time[i] >= i (a random
  // network delay, so later messages can overtake earlier ones), and
  // finishes at finish_time[i] > delivery_time[i]: an instantiation can only
  // finish after the owner has handled it, which is the fact the protocol
  // rests on (only the owner triggers the finish event). The sender's view
  // at issue time t is "finished iff finish_time <= t".
  struct Sender {
    RemoteOrderSender<int> order;
    std::vector<int> delivery_time, finish_time;
    int now = 0;

    std::vector<Message> issue(int who, int count, std::mt19937 &rng, int max_delay,
                               int max_run)
    {
      std::vector<Message> out;
      std::uniform_int_distribution<int> delay(0, max_delay), run(0, max_run);
      for(int i = 0; i < count; i++) {
        int d = now + delay(rng);
        delivery_time.push_back(d);
        finish_time.push_back(d + 1 + run(rng));
        Message m;
        m.sender = who;
        m.issue_index = int(finish_time.size()) - 1;
        auto finished = [&](int tok) { return finish_time[tok] <= now; };
        m.seq = order.note(m.issue_index, finished);
        out.push_back(m);
        now++;
      }
      return out;
    }
  };

  // The network: everything in delivery-time order, ties broken at random.
  std::vector<Message> deliver_order(const std::vector<Sender> &senders,
                                     std::vector<Message> msgs, std::mt19937 &rng)
  {
    std::shuffle(msgs.begin(), msgs.end(), rng);
    std::stable_sort(msgs.begin(), msgs.end(), [&](const Message &a, const Message &b) {
      return senders[a.sender].delivery_time[a.issue_index] <
             senders[b.sender].delivery_time[b.issue_index];
    });
    return msgs;
  }

} // namespace

TEST(SubgraphOrder, InOrderDeliveryNeedsNoWindow)
{
  std::mt19937 rng(1);
  std::vector<Sender> s(1);
  std::vector<Message> msgs = s[0].issue(0, 1000, rng, 0 /*no delay*/, 5);
  Owner o;
  for(const Message &m : deliver_order(s, msgs, rng))
    o.deliver(m);
  ASSERT_EQ(o.handled[0].size(), 1000u);
  for(int i = 0; i < 1000; i++)
    EXPECT_EQ(o.handled[0][i], i);
  EXPECT_EQ(o.max_window, 0u);
  EXPECT_EQ(o.trackers[0].num_held(), 0u);
}

TEST(SubgraphOrder, RandomReorderPreservesIssueOrderPerSender)
{
  for(int seed = 1; seed <= 300; seed++) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> senders_d(1, 4), count_d(1, 80), delay_d(0, 12),
        run_d(0, 30);
    const int S = senders_d(rng), count = count_d(rng), max_delay = delay_d(rng),
              max_run = run_d(rng);
    std::vector<Sender> senders(S);
    std::vector<Message> all;
    for(int s = 0; s < S; s++) {
      std::vector<Message> m = senders[s].issue(s, count, rng, max_delay, max_run);
      all.insert(all.end(), m.begin(), m.end());
    }
    Owner o;
    bool any_reordered = false;
    std::vector<int> last_seen(S, -1);
    for(const Message &m : deliver_order(senders, all, rng)) {
      any_reordered = any_reordered || (m.issue_index < last_seen[m.sender]);
      last_seen[m.sender] = std::max(last_seen[m.sender], m.issue_index);
      o.deliver(m);
    }
    for(int s = 0; s < S; s++) {
      ASSERT_EQ(o.handled[s].size(), size_t(count)) << "seed " << seed << " sender " << s;
      for(int i = 0; i < count; i++)
        ASSERT_EQ(o.handled[s][i], i) << "seed " << seed << " sender " << s;
      EXPECT_EQ(o.trackers[s].num_held(), 0u) << "seed " << seed;
      EXPECT_EQ(o.trackers[s].window(), 0u) << "seed " << seed; // all absorbed
    }
    // the window never exceeds one network delay's worth of messages
    EXPECT_LE(o.max_window, size_t(max_delay)) << "seed " << seed;
    (void)any_reordered;
  }
}

TEST(SubgraphOrder, SenderDropsFinishedTokensInBatches)
{
  std::mt19937 rng(7);
  std::vector<Sender> s(1);
  // no delay, no run time: every instantiation finishes right after the
  // next one is issued, so the tracked set never grows past one batch
  std::vector<Message> msgs = s[0].issue(0, 1000, rng, 0, 0);
  for(int i = 0; i < 1000; i++)
    EXPECT_EQ(msgs[i].seq, uint64_t(i)); // continuous numbering
  EXPECT_LE(s[0].order.num_tracked(), 32u);
  std::vector<int> un;
  auto finished = [&](int tok) { return s[0].finish_time[tok] <= s[0].now; };
  s[0].order.unfinished(finished, un);
  EXPECT_EQ(un.size(), 0u);
}

TEST(SubgraphOrder, UnfinishedTokensAreReported)
{
  Sender s;
  s.finish_time = {100, 0, 100}; // tokens 0 and 2 never finish, 1 finishes at once
  auto finished = [&](int tok) { return s.finish_time[tok] <= s.now; };
  EXPECT_EQ(s.order.note(0, finished), 0u);
  s.now = 1;
  EXPECT_EQ(s.order.note(1, finished), 1u);
  s.now = 2;
  EXPECT_EQ(s.order.note(2, finished), 2u);
  std::vector<int> un;
  s.order.unfinished(finished, un);
  ASSERT_EQ(un.size(), 2u);
  EXPECT_EQ(un[0], 0);
  EXPECT_EQ(un[1], 2);
}

TEST(SubgraphOrder, SlotReuseKeepsBothSidesCounting)
{
  // A sender instantiated an earlier subgraph in this slot 37 times; the
  // owner destroyed it and a new subgraph reuses the slot. Neither side
  // resets: the sender numbers on from 37 and the owner's tracker for the
  // slot has recorded 0..36, so the new subgraph's messages are judged in
  // the same numbering whatever the sender believes about its old events.
  std::mt19937 rng(3);
  std::vector<Sender> s(1);
  std::vector<Message> old = s[0].issue(0, 37, rng, 3, 3);
  Owner o;
  for(const Message &m : deliver_order(s, old, rng))
    o.deliver(m);
  ASSERT_EQ(o.handled[0].size(), 37u);
  // the previous subgraph is gone; the tracker stays (as in SubgraphImpl)
  o.handled[0].clear();
  std::vector<Message> fresh = s[0].issue(0, 20, rng, 6, 10);
  for(const Message &m : deliver_order(s, fresh, rng))
    o.deliver(m);
  ASSERT_EQ(o.handled[0].size(), 20u);
  for(int i = 0; i < 20; i++)
    EXPECT_EQ(o.handled[0][i], 37 + i);
  EXPECT_EQ(o.trackers[0].num_held(), 0u);
}

TEST(SubgraphOrder, RecordingWithoutHoldingKeepsTheNumberingConsistent)
{
  // A mode that never holds (CONCURRENT) still records every sequence
  // number, so a later INSTANTIATION_ORDER subgraph in the slot starts
  // consistent with the sender's numbering: 0..4 recorded out of order,
  // then 5 and 6 judged normally.
  RemoteOrderTracker t;
  for(uint64_t q : {3, 0, 4, 1, 2})
    t.record(q);
  EXPECT_EQ(t.next_unhandled(), 5u);
  EXPECT_EQ(t.window(), 0u);
  EXPECT_FALSE(t.ready(6, true));
  EXPECT_TRUE(t.ready(5, true));
  EXPECT_TRUE(t.ready(6, true));
  EXPECT_EQ(t.next_unhandled(), 7u);
}

TEST(SubgraphOrder, HeldMessageReleasedByPredecessor)
{
  RemoteOrderTracker t;
  int payload_b = 2, payload_c = 3;
  // b (seq 1) and c (seq 2) arrive before a (seq 0); c depends on both
  EXPECT_FALSE(t.ready(1, true));
  t.hold(1, &payload_b);
  EXPECT_FALSE(t.ready(2, true));
  t.hold(2, &payload_c);
  EXPECT_EQ(t.pop_ready(), nullptr);
  EXPECT_TRUE(t.ready(0, true));
  uint64_t seq = 99;
  void *p = t.pop_ready(&seq);
  ASSERT_EQ(p, &payload_b); // only b is ready: c still needs b handled
  EXPECT_EQ(seq, 1u);
  EXPECT_EQ(t.pop_ready(), nullptr);
  EXPECT_TRUE(t.ready(seq, true));
  p = t.pop_ready(&seq);
  ASSERT_EQ(p, &payload_c);
  EXPECT_TRUE(t.ready(seq, true));
  EXPECT_EQ(t.num_held(), 0u);
  EXPECT_EQ(t.next_unhandled(), 3u);
}
