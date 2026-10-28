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
// explicitly: every instantiate carries its sequence number among the node's
// instantiates of that subgraph slot, and the owner handles a node's
// instantiates in sequence order, holding an early arrival until every
// earlier one has been handled.
//
// Subgraph ids carry no generation, and a node's view of which finish
// events have triggered lags the owner's. So neither side ever resets: a
// sender's RemoteOrderSender for a slot and the owner's RemoteOrderTracker
// for (slot, sending node) live for as long as the runtime, count
// continuously across the slot's lifetimes, and the owner records every
// remote instantiate's sequence number in every concurrency mode (only
// INSTANTIATION_ORDER holds anything). The owner thus knows exactly which of
// a node's instantiates it has handled, which is all the protocol needs; it
// is not told anything about the sender's view of its events.
//
// These two classes hold no Realm state so that they can be driven by a
// randomized model in tests/unit_tests/subgraph_order_test.cc.

#ifndef REALM_SUBGRAPH_ORDER_H
#define REALM_SUBGRAPH_ORDER_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace Realm {

  // Owner side: one sending node's instantiates of one subgraph slot. Every
  // sequence number below `next` has been handled; `handled` holds the ones
  // at or above `next`, i.e. the out-of-order window.
  class RemoteOrderTracker {
  public:
    // Whether every instantiate of the sender below `seq` has been handled.
    // If so and `record`, records seq as handled. The caller records in the
    // same critical section as it acts on the message (chains it), so that
    // a later message from the node cannot slip in between.
    bool ready(uint64_t seq, bool record)
    {
      // in order (the common case): nothing to look up
      for(uint64_t q = next; q < seq; q++)
        if(handled.count(q) == 0)
          return false;
      if(record)
        this->record(seq);
      return true;
    }

    // Records seq as handled (modes that never hold still record, so the
    // numbering stays consistent across a slot's reuse by another mode).
    void record(uint64_t seq)
    {
      if(seq == next) {
        next++;
        while(!handled.empty() && (*handled.begin() == next)) {
          handled.erase(handled.begin());
          next++;
        }
      } else if(seq > next) {
        handled.insert(seq);
      }
    }

    // An early arrival, kept until its predecessors have been handled.
    void hold(uint64_t seq, void *payload)
    {
      held.push_back(std::make_pair(seq, payload));
    }

    // Pops a held message whose predecessors have all been handled, if any
    // (not recorded: the caller handles it, which records it; readiness only
    // grows, so it stays ready).
    void *pop_ready(uint64_t *seq = nullptr)
    {
      for(size_t i = 0; i < held.size(); i++) {
        if(ready(held[i].first, false /*record*/)) {
          void *p = held[i].second;
          if(seq)
            *seq = held[i].first;
          held[i] = held.back();
          held.pop_back();
          return p;
        }
      }
      return nullptr;
    }

    // Whether seq has already been handled: a sender never reuses a number,
    // so a message for which this is true is a protocol violation.
    bool handled_already(uint64_t seq) const
    {
      return (seq < next) || (handled.count(seq) != 0);
    }
    size_t num_held() const { return held.size(); }
    // size of the out-of-order window (handled sequence numbers above next)
    size_t window() const { return handled.size(); }
    uint64_t next_unhandled() const { return next; }

  private:
    uint64_t next = 0;
    std::set<uint64_t> handled;
    std::vector<std::pair<uint64_t, void *>> held;
  };

  // Sender side: one node's instantiates of one subgraph slot, numbered
  // continuously, with a token (the finish event) per instantiate that is
  // still of interest: a remote destroy from this node is sent only once
  // every instantiate it sent before has finished. Finished tokens are
  // dropped in batches, so a node pays O(1) per instantiate on average
  // however long any one instantiation runs.
  template <typename Token>
  class RemoteOrderSender {
  public:
    // Assigns the next sequence number and remembers the token.
    // `finished(token)` must return true once that instantiation has
    // finished (poisoned counts as finished).
    template <typename Finished>
    uint64_t note(const Token &token, Finished finished)
    {
      if(sent.size() >= prune_at) {
        size_t kept = 0;
        for(size_t i = 0; i < sent.size(); i++)
          if(!finished(sent[i]))
            sent[kept++] = sent[i];
        sent.resize(kept);
        prune_at = (kept < 8) ? 16 : 2 * kept;
      }
      sent.push_back(token);
      return next_seq++;
    }

    // The tokens of the instantiations not yet finished.
    template <typename Finished>
    void unfinished(Finished finished, std::vector<Token> &out) const
    {
      for(const Token &t : sent)
        if(!finished(t))
          out.push_back(t);
    }

    size_t num_tracked() const { return sent.size(); }
    uint64_t next_sequence() const { return next_seq; }

  private:
    uint64_t next_seq = 0;
    size_t prune_at = 16;
    std::vector<Token> sent;
  };

} // namespace Realm

#endif
