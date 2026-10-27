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

// Issue-order bookkeeping for remote subgraph instantiations.
//
// Active messages are not delivered in order (not across message types, and
// not within one type either: a payload large enough for a rendezvous
// protocol reaches the handler after smaller messages sent later). A node's
// issue order for an INSTANTIATION_ORDER subgraph is therefore carried
// explicitly: every instantiate carries its sequence number `seq` among the
// node's instantiates of that subgraph, and `base`, the lowest sequence
// number that was still unfinished when it was sent. Everything below `base`
// had finished, which implies it was handled, so the owner needs only
// [base, seq) to have been handled before this message, and holds an early
// arrival until then. `base` must be exact (not lower than the oldest
// unfinished one): when the owner has since destroyed the subgraph and
// reused its id, a stale lower base would ask the new subgraph for sequence
// numbers it never sees.
//
// Subgraph ids carry no generation, and a node's view of which finish
// events have triggered lags the owner's. So neither side may reset its
// numbering when a subgraph is destroyed or its slot reused: a sender's
// RemoteOrderSender for a slot and the owner's RemoteOrderTracker for
// (slot, sender) live for as long as the runtime and count continuously
// across the slot's lifetimes. (Resetting either side can hold a message
// forever: the owner would wait for a sequence number that belonged to the
// previous subgraph, or treat a renumbered message as already handled.)
//
// These two classes hold no Realm state so that they can be driven by a
// randomized model in tests/unit_tests/subgraph_order_test.cc. The owner
// keeps one RemoteOrderTracker per (slot, sending node), under the
// subgraph's lifecycle lock; a node keeps one RemoteOrderSender per slot it
// has instantiated remotely.

#ifndef REALM_SUBGRAPH_ORDER_H
#define REALM_SUBGRAPH_ORDER_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <set>
#include <utility>
#include <vector>

namespace Realm {

  // Owner side: one sending node's instantiates of one subgraph. Every
  // sequence number below `next` has been handled (or, below a message's
  // `base`, had finished, which implies handled); `handled` holds the ones
  // at or above `next`, i.e. the out-of-order window.
  class RemoteOrderTracker {
  public:
    struct Item {
      uint64_t seq, base;
    };

    // Whether every instantiate in [base, seq) has been handled. If so and
    // `record`, records seq as handled. The caller records in the same
    // critical section as it acts on the message (chains it), so that a
    // later message from the node cannot slip in between.
    bool ready(const Item &it, bool record)
    {
      if(it.base > next) {
        // everything below the base had finished, so it was handled
        next = it.base;
        handled.erase(handled.begin(), handled.lower_bound(next));
        absorb();
      }
      // in order (the common case): nothing to look up
      for(uint64_t q = (it.base > next) ? it.base : next; q < it.seq; q++)
        if(handled.count(q) == 0)
          return false;
      if(record) {
        if(it.seq == next) {
          next++;
          absorb();
        } else if(it.seq > next) {
          handled.insert(it.seq);
        }
      }
      return true;
    }

    // An early arrival, kept until its predecessors have been handled.
    void hold(const Item &it, void *payload) { held.push_back(std::make_pair(it, payload)); }

    // Pops a held message whose predecessors have all been handled, if any
    // (not recorded: the caller handles it, which records it; readiness only
    // grows, so it stays ready).
    void *pop_ready(Item *item = nullptr)
    {
      for(size_t i = 0; i < held.size(); i++) {
        if(ready(held[i].first, false /*record*/)) {
          void *p = held[i].second;
          if(item)
            *item = held[i].first;
          held[i] = held.back();
          held.pop_back();
          return p;
        }
      }
      return nullptr;
    }

    size_t num_held() const { return held.size(); }
    const std::vector<std::pair<Item, void *>> &held_items() const { return held; }
    // size of the out-of-order window (handled sequence numbers above next)
    size_t window() const { return handled.size(); }
    uint64_t next_unhandled() const { return next; }

  private:
    void absorb()
    {
      while(!handled.empty() && (*handled.begin() == next)) {
        handled.erase(handled.begin());
        next++;
      }
    }

    uint64_t next = 0;
    std::set<uint64_t> handled;
    std::vector<std::pair<Item, void *>> held;
  };

  // Sender side: one node's instantiates of one subgraph, in sequence order,
  // each with a token (the finish event) that tells whether it has finished.
  // The finished prefix is popped on every note, so the front is the oldest
  // unfinished one and `base` is exact; a finished entry behind an unfinished
  // one waits its turn. Each entry is popped once: O(1) amortized per note.
  template <typename Token>
  class RemoteOrderSender {
  public:
    // Assigns the next sequence number and computes the base.
    // `finished(token)` must return true once that instantiation has
    // finished (poisoned counts as finished).
    template <typename Finished>
    void note(const Token &token, Finished finished, uint64_t &seq, uint64_t &base)
    {
      seq = next_seq++;
      while(!sent.empty() && finished(sent.front().second))
        sent.pop_front();
      base = sent.empty() ? seq : sent.front().first;
      sent.push_back(std::make_pair(seq, token));
    }

    // The tokens of the instantiations not yet finished, in sequence order.
    template <typename Finished>
    void unfinished(Finished finished, std::vector<Token> &out) const
    {
      for(const auto &e : sent)
        if(!finished(e.second))
          out.push_back(e.second);
    }

    size_t num_tracked() const { return sent.size(); }
    uint64_t next_sequence() const { return next_seq; }

  private:
    uint64_t next_seq = 0;
    std::deque<std::pair<uint64_t, Token>> sent;
  };

} // namespace Realm

#endif
