/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

// Tests for ActiveMessage's chunked SEND path (init_chunked / init_chunked_data /
//  commit_chunked), end to end through a real IncomingMessageManager.
//
// The loopback network reports no payload limit, so ActiveMessage never chunks in
//  the other unit tests.  Here a mock network module with a small, configurable limit
//  is installed as Network::single_network: it records every message ActiveMessage
//  hands it, and the tests deliver those messages to an IncomingMessageManager in
//  whatever order they choose (in order, reversed, interleaved with another
//  message), the way a network that does not order messages could.

#include <gtest/gtest.h>
#include "realm/activemsg.h"
#include "realm/network.h"
#include "realm/threads.h"
#include "realm/timers.h"
#include "realm/serialize.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace Realm;

namespace {

  ////////////////////////////////////////////////////////////////////////
  //
  // a mock network that records what it is asked to send
  //

  struct SentMessage {
    std::vector<NodeID> targets;
    bool multicast{false};
    unsigned short msgid{0};
    std::vector<char> header;
    std::vector<char> payload;
    // remote completions the sender attached, invoked when the test reports the
    //  message handled
    std::shared_ptr<std::vector<char>> remote_comp;
  };

  class MockNetworkModule;

  // must fit in an ActiveMessage's 256 bytes of inline storage
  class MockMessageImpl : public ActiveMessageImpl {
  public:
    MockMessageImpl(MockNetworkModule *_net, const std::vector<NodeID> &_targets,
                    bool _multicast, unsigned short _msgid, size_t _header_size,
                    size_t _max_payload_size, const void *_src_payload_addr)
      : net(_net)
      , msgid(_msgid)
      , multicast(_multicast)
      , header_size(_header_size)
      , src_payload_addr(_src_payload_addr)
      , targets(_targets)
      , hdr(_header_size)
    {
      header_base = hdr.data();
      if(_src_payload_addr != nullptr) {
        payload_base = const_cast<void *>(_src_payload_addr);
      } else {
        payload.resize(_max_payload_size);
        payload_base = payload.data();
      }
      payload_size = _max_payload_size;
    }

    virtual void *add_local_completion(size_t size)
    {
      if(!local_comp)
        local_comp = std::make_shared<std::vector<char>>();
      return grow(*local_comp, size);
    }

    virtual void *add_remote_completion(size_t size)
    {
      if(!remote_comp)
        remote_comp = std::make_shared<std::vector<char>>();
      return grow(*remote_comp, size);
    }

    virtual void commit(size_t act_payload_size);

    virtual void cancel()
    {
      if(local_comp)
        CompletionCallbackBase::destroy_all(local_comp->data(), local_comp->size());
      if(remote_comp)
        CompletionCallbackBase::destroy_all(remote_comp->data(), remote_comp->size());
    }

  protected:
    // completions are placement-new'd into the returned storage, so reserve generously
    //  up front instead of reallocating under live objects
    static void *grow(std::vector<char> &buf, size_t size)
    {
      if(buf.capacity() == 0)
        buf.reserve(4096);
      assert((buf.size() + size) <= buf.capacity());
      size_t ofs = buf.size();
      buf.resize(ofs + size);
      return buf.data() + ofs;
    }

    MockNetworkModule *net;
    unsigned short msgid;
    bool multicast;
    size_t header_size;
    const void *src_payload_addr;
    std::vector<NodeID> targets;
    std::vector<char> hdr;
    std::vector<char> payload;
    std::shared_ptr<std::vector<char>> local_comp;
    std::shared_ptr<std::vector<char>> remote_comp;
  };

  class MockNetworkModule : public NetworkModule {
  public:
    MockNetworkModule()
      : NetworkModule("mock")
    {}

    // payload limit for one message, and the largest header the network can carry
    //  (max_payload_size() is 0 above that, as GASNet-EX's is)
    size_t limit{64};
    size_t max_header{256};
    std::vector<SentMessage> sent;

    virtual size_t max_payload_size(size_t header_size, const void *)
    {
      return (header_size > max_header) ? 0 : limit;
    }

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size)
    {
      assert(storage_size >= sizeof(MockMessageImpl));
      return new(storage_base)
          MockMessageImpl(this, std::vector<NodeID>(1, target), false, msgid,
                          header_size, max_payload_size, src_payload_addr);
    }

    // not virtual overrides: the NodeSet forms of these are slated for removal (PR
    //  465), and without 'override' this file keeps compiling once they are gone
    virtual ActiveMessageImpl *
    create_active_message_impl(const NodeSet &targets, unsigned short msgid,
                               size_t header_size, size_t max_payload_size,
                               const void *src_payload_addr, size_t src_payload_lines,
                               size_t src_payload_line_stride, void *storage_base,
                               size_t storage_size)
    {
      assert(storage_size >= sizeof(MockMessageImpl));
      std::vector<NodeID> v(targets.begin(), targets.end());
      return new(storage_base) MockMessageImpl(this, v, true, msgid, header_size,
                                               max_payload_size, src_payload_addr);
    }

    // the rest is never used by these tests
    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID, unsigned short, size_t, size_t,
                               const LocalAddress &, size_t, size_t,
                               const RemoteAddress &, void *, size_t)
    {
      abort();
    }
    virtual ActiveMessageImpl *create_active_message_impl(NodeID, unsigned short, size_t,
                                                          size_t, const RemoteAddress &,
                                                          void *, size_t)
    {
      abort();
    }
    virtual size_t recommended_max_payload(NodeID, bool, size_t) { return limit; }
    virtual size_t recommended_max_payload(const NodeSet &, bool, size_t)
    {
      return limit;
    }
    virtual size_t recommended_max_payload(NodeID, const RemoteAddress &, bool, size_t)
    {
      return limit;
    }
    virtual size_t recommended_max_payload(NodeID, const void *, size_t, size_t, size_t,
                                           bool, size_t)
    {
      return limit;
    }
    virtual size_t recommended_max_payload(const NodeSet &, const void *, size_t, size_t,
                                           size_t, bool, size_t)
    {
      return limit;
    }
    virtual size_t recommended_max_payload(NodeID, const LocalAddress &, size_t, size_t,
                                           size_t, const RemoteAddress &, bool, size_t)
    {
      return limit;
    }
    virtual void get_shared_peers(NodeSet &) {}
    virtual void attach(RuntimeImpl *, std::vector<NetworkSegment *> &) {}
    virtual void detach(RuntimeImpl *, std::vector<NetworkSegment *> &) {}
    virtual void barrier(void) {}
    virtual void broadcast(NodeID, const void *, void *, size_t) { abort(); }
    virtual void gather(NodeID, const void *, void *, size_t) { abort(); }
    virtual void allgatherv(const char *, size_t, std::vector<char> &,
                            std::vector<size_t> &)
    {
      abort();
    }
    virtual void sample_quiescence_state(QuiescenceState &) { abort(); }
    virtual void quiescence_allreduce_sum(const uint64_t *, uint64_t *, size_t)
    {
      abort();
    }
    virtual MemoryImpl *create_remote_memory(RuntimeImpl *, Memory, size_t, Memory::Kind,
                                             const ByteArray &)
    {
      abort();
    }
    virtual IBMemory *create_remote_ib_memory(RuntimeImpl *, Memory, size_t,
                                              Memory::Kind, const ByteArray &)
    {
      abort();
    }
  };

  void MockMessageImpl::commit(size_t act_payload_size)
  {
    SentMessage m;
    m.targets = targets;
    m.multicast = multicast;
    m.msgid = msgid;
    m.header.assign(hdr.begin(), hdr.end());
    const char *p = static_cast<const char *>(payload_base);
    m.payload.assign(p, p + act_payload_size);
    m.remote_comp = remote_comp;
    net->sent.push_back(std::move(m));
    // everything has been copied, so local completion is immediate
    if(local_comp) {
      CompletionCallbackBase::invoke_all(local_comp->data(), local_comp->size());
      CompletionCallbackBase::destroy_all(local_comp->data(), local_comp->size());
    }
  }

  // the IncomingMessageManager callback a backend would register for a message with
  //  remote completions: data1 points at the SentMessage's completion storage
  void remote_completion_callback(NodeID, IncomingMessageManager::CallbackData data1,
                                  IncomingMessageManager::CallbackData)
  {
    std::vector<char> *comp = reinterpret_cast<std::vector<char> *>(data1);
    CompletionCallbackBase::invoke_all(comp->data(), comp->size());
    CompletionCallbackBase::destroy_all(comp->data(), comp->size());
    comp->clear();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // message types
  //

  struct Received {
    std::string type;
    int seq;
    std::vector<char> payload;
  };
  std::vector<Received> received;

  // queued: no inline handler, so a delivery always goes through the message blocks
  struct QueuedMsg {
    int seq;
    static void handle_message(NodeID, const QueuedMsg &hdr, const void *data,
                               size_t len)
    {
      const char *c = static_cast<const char *>(data);
      received.push_back({"QueuedMsg", hdr.seq, std::vector<char>(c, c + len)});
    }
  };

  // inline: handled in add_incoming_message itself when time allows
  struct InlineMsg {
    int seq;
    static bool handle_inline(NodeID, const InlineMsg &hdr, const void *data, size_t len,
                              TimeLimit)
    {
      const char *c = static_cast<const char *>(data);
      received.push_back({"InlineMsg", hdr.seq, std::vector<char>(c, c + len)});
      return true;
    }
    static void handle_message(NodeID sender, const InlineMsg &hdr, const void *data,
                               size_t len)
    {
      handle_inline(sender, hdr, data, len, TimeLimit());
    }
  };

  // a header too large for the mock network to fragment (see BigHeaderLimit below)
  struct BigHeaderMsg {
    int seq;
    char pad[60];
    static void handle_message(NodeID, const BigHeaderMsg &, const void *, size_t) {}
  };

  ActiveMessageHandlerReg<QueuedMsg> queued_msg_reg;
  ActiveMessageHandlerReg<InlineMsg> inline_msg_reg;
  ActiveMessageHandlerReg<BigHeaderMsg> big_header_msg_reg;

  std::vector<char> pattern(size_t n, int seed)
  {
    std::vector<char> v(n);
    for(size_t i = 0; i < n; i++)
      v[i] = static_cast<char>((i * 31 + seed) & 0xff);
    return v;
  }

  template <typename T>
  const WrappedWithFragInfo<T> &wrapped(const SentMessage &m)
  {
    EXPECT_EQ(m.header.size(), sizeof(WrappedWithFragInfo<T>));
    return *reinterpret_cast<const WrappedWithFragInfo<T> *>(m.header.data());
  }

  class ActMsgChunkedSendTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      saved_network = Network::single_network;
      Network::single_network = &net;
      activemsg_handler_table.construct_handler_table();
      received.clear();
    }

    void TearDown() override { Network::single_network = saved_network; }

    // hands one recorded message to the manager the way a backend would, including
    //  the remote-completion callback and its "handled inline" contract
    void deliver(IncomingMessageManager &mgr, const SentMessage &m, NodeID sender = 1)
    {
      IncomingMessageManager::CallbackFnptr fn = nullptr;
      IncomingMessageManager::CallbackData d1 = 0;
      if(m.remote_comp && !m.remote_comp->empty()) {
        fn = remote_completion_callback;
        d1 = reinterpret_cast<IncomingMessageManager::CallbackData>(m.remote_comp.get());
      }
      bool handled = mgr.add_incoming_message(
          sender, m.msgid, m.header.data(), m.header.size(), PAYLOAD_COPY,
          m.payload.data(), m.payload.size(), PAYLOAD_COPY, fn, d1, 0, TimeLimit());
      // a message handled immediately gets no callback - the backend completes it
      if(handled && (fn != nullptr))
        fn(sender, d1, 0);
    }

    MockNetworkModule net;
    NetworkModule *saved_network{nullptr};
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // tests
  //

  TEST_F(ActMsgChunkedSendTest, PayloadWithinLimitIsOnePlainMessage)
  {
    std::vector<char> data = pattern(net.limit, 1);
    ActiveMessage<QueuedMsg> amsg(1, data.size());
    amsg->seq = 7;
    amsg.add_payload(data.data(), data.size());
    amsg.commit();

    ASSERT_EQ(net.sent.size(), 1u);
    EXPECT_EQ(net.sent[0].msgid, activemsg_handler_table.lookup_message_id<QueuedMsg>());
    EXPECT_EQ(net.sent[0].payload, data);
  }

  TEST_F(ActMsgChunkedSendTest, LargePayloadIsSplitAndReassembled)
  {
    std::vector<char> data = pattern(1000, 2);
    ActiveMessage<QueuedMsg> amsg(1, data.size());
    amsg->seq = 11;
    amsg.add_payload(data.data(), data.size());
    amsg.commit();

    const size_t expected = (data.size() + net.limit - 1) / net.limit;
    ASSERT_EQ(net.sent.size(), expected);
    const uint64_t id = wrapped<QueuedMsg>(net.sent[0]).frag_info.msg_id;
    for(size_t i = 0; i < net.sent.size(); i++) {
      const WrappedWithFragInfo<QueuedMsg> &w = wrapped<QueuedMsg>(net.sent[i]);
      EXPECT_EQ(net.sent[i].msgid,
                activemsg_handler_table.lookup_message_id<WrappedWithFragInfo<QueuedMsg>>());
      EXPECT_EQ(w.frag_info.chunk_id, i);
      EXPECT_EQ(w.frag_info.total_chunks, expected);
      EXPECT_EQ(w.frag_info.msg_id, id);
      EXPECT_EQ(w.user.seq, 11);
      // a single target goes through the unicast path
      EXPECT_FALSE(net.sent[i].multicast);
      EXPECT_LE(net.sent[i].payload.size(), net.limit);
    }

    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, 0 /*dedicated_threads*/, crs);
    // reversed: the network does not have to deliver in order
    for(size_t i = net.sent.size(); i > 0; i--)
      deliver(mgr, net.sent[i - 1]);
    mgr.do_work(TimeLimit());
    mgr.shutdown();

    ASSERT_EQ(received.size(), 1u);
    EXPECT_EQ(received[0].seq, 11);
    EXPECT_EQ(received[0].payload, data);
  }

  // The regression behind A-01: each ActiveMessage<T> instantiation used to number its
  //  chunked messages from 0, and the receiver keyed reassembly on (sender, id) alone,
  //  so the first chunked message of two types collided.  Send one of each type (and
  //  one through a different INLINE_STORAGE, which used to be a third counter), check
  //  their ids differ, and deliver their fragments interleaved.
  TEST_F(ActMsgChunkedSendTest, DifferentTypesDoNotShareReassembly)
  {
    std::vector<char> a = pattern(300, 3), b = pattern(250, 4), c = pattern(200, 5);
    {
      ActiveMessage<QueuedMsg> m(1, a.size());
      m->seq = 1;
      m.add_payload(a.data(), a.size());
      m.commit();
    }
    const size_t na = net.sent.size();
    {
      ActiveMessage<InlineMsg> m(1, b.size());
      m->seq = 2;
      m.add_payload(b.data(), b.size());
      m.commit();
    }
    const size_t nb = net.sent.size() - na;
    {
      ActiveMessage<QueuedMsg, 512> m(1, c.size());
      m->seq = 3;
      m.add_payload(c.data(), c.size());
      m.commit();
    }
    const size_t nc = net.sent.size() - na - nb;

    const uint64_t id_a = wrapped<QueuedMsg>(net.sent[0]).frag_info.msg_id;
    const uint64_t id_b = wrapped<InlineMsg>(net.sent[na]).frag_info.msg_id;
    const uint64_t id_c = wrapped<QueuedMsg>(net.sent[na + nb]).frag_info.msg_id;
    EXPECT_NE(id_a, id_b);
    EXPECT_NE(id_a, id_c);
    EXPECT_NE(id_b, id_c);

    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, 0 /*dedicated_threads*/, crs);
    // round-robin across the three messages
    for(size_t i = 0; i < std::max(na, std::max(nb, nc)); i++) {
      if(i < na)
        deliver(mgr, net.sent[i]);
      if(i < nb)
        deliver(mgr, net.sent[na + i]);
      if(i < nc)
        deliver(mgr, net.sent[na + nb + i]);
    }
    mgr.do_work(TimeLimit());
    mgr.shutdown();

    ASSERT_EQ(received.size(), 3u);
    for(const Received &r : received) {
      if(r.type == "InlineMsg") {
        EXPECT_EQ(r.seq, 2);
        EXPECT_EQ(r.payload, b);
      } else if(r.seq == 1) {
        EXPECT_EQ(r.payload, a);
      } else {
        EXPECT_EQ(r.seq, 3);
        EXPECT_EQ(r.payload, c);
      }
    }
  }

  // A-02 through the send path: a message reassembled above one message block (~1MB),
  //  for a handler that is never run inline
  TEST_F(ActMsgChunkedSendTest, ReassemblyLargerThanOneMessageBlock)
  {
    net.limit = 64 << 10;
    std::vector<char> data = pattern(2 << 20, 6);
    ActiveMessage<QueuedMsg> amsg(1, data.size());
    amsg->seq = 21;
    amsg.add_payload(data.data(), data.size());
    amsg.commit();
    ASSERT_EQ(net.sent.size(), 32u);

    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, 0 /*dedicated_threads*/, crs);
    for(const SentMessage &m : net.sent)
      deliver(mgr, m);
    mgr.do_work(TimeLimit());
    mgr.shutdown();

    ASSERT_EQ(received.size(), 1u);
    EXPECT_EQ(received[0].payload, data);
  }

  // The PAYLOAD_KEEP constructor, serialization and payload_ptr all feed chunked mode
  TEST_F(ActMsgChunkedSendTest, OtherWaysOfProvidingThePayload)
  {
    std::vector<char> keep = pattern(500, 7);
    {
      ActiveMessage<QueuedMsg> m(1, keep.data(), keep.size());
      m->seq = 31;
      m.commit();
    }
    std::vector<int> ints(100);
    for(size_t i = 0; i < ints.size(); i++)
      ints[i] = int(i * 3);
    {
      ActiveMessage<QueuedMsg> m(1, ints.size() * sizeof(int) + 64);
      m->seq = 32;
      EXPECT_TRUE(m << ints);
      EXPECT_TRUE(m << std::string("tail"));
      m.commit();
    }
    std::vector<char> raw = pattern(300, 8);
    {
      ActiveMessage<QueuedMsg> m(1, raw.size());
      m->seq = 33;
      memcpy(m.payload_ptr(100), raw.data(), 100);
      memcpy(m.payload_ptr(200), raw.data() + 100, 200);
      m.commit();
    }

    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, 0 /*dedicated_threads*/, crs);
    for(const SentMessage &m : net.sent)
      deliver(mgr, m);
    mgr.do_work(TimeLimit());
    mgr.shutdown();

    ASSERT_EQ(received.size(), 3u);
    for(const Received &r : received) {
      if(r.seq == 31) {
        EXPECT_EQ(r.payload, keep);
      } else if(r.seq == 32) {
        Serialization::FixedBufferDeserializer fbd(r.payload.data(), r.payload.size());
        std::vector<int> got;
        std::string tail;
        EXPECT_TRUE(fbd >> got);
        EXPECT_TRUE(fbd >> tail);
        EXPECT_EQ(got, ints);
        EXPECT_EQ(tail, "tail");
      } else {
        EXPECT_EQ(r.seq, 33);
        EXPECT_EQ(r.payload, raw);
      }
    }
  }

  // A-05: completions on a message that is chunked.  The local one fires once commit()
  //  has copied everything; the remote one rides chunk 0 and fires once, and only after
  //  the reassembled message has been handled - whichever fragment arrives last.
  TEST_F(ActMsgChunkedSendTest, CompletionsOnAChunkedMessage)
  {
    for(int order = 0; order < 2; order++) {
      for(int use_inline = 0; use_inline < 2; use_inline++) {
        net.sent.clear();
        received.clear();
        std::atomic<int> local{0}, remote{0};
        std::vector<char> data = pattern(400, 9);
        if(use_inline) {
          ActiveMessage<InlineMsg> m(1, data.size());
          m->seq = 41;
          m.add_payload(data.data(), data.size());
          m.add_local_completion([&local]() { local.fetch_add(1); });
          m.add_remote_completion([&remote]() { remote.fetch_add(1); });
          m.commit();
        } else {
          ActiveMessage<QueuedMsg> m(1, data.size());
          m->seq = 41;
          m.add_payload(data.data(), data.size());
          m.add_local_completion([&local]() { local.fetch_add(1); });
          m.add_remote_completion([&remote]() { remote.fetch_add(1); });
          m.commit();
        }
        EXPECT_EQ(local.load(), 1);
        ASSERT_GT(net.sent.size(), 1u);
        // only chunk 0 carries the remote completion
        for(size_t i = 0; i < net.sent.size(); i++)
          EXPECT_EQ(bool(net.sent[i].remote_comp), (i == 0)) << "chunk " << i;

        CoreReservationSet crs(nullptr);
        IncomingMessageManager mgr(2, 0 /*dedicated_threads*/, crs);
        std::vector<size_t> seq(net.sent.size());
        for(size_t i = 0; i < seq.size(); i++)
          seq[i] = (order == 0) ? i : (seq.size() - 1 - i);
        for(size_t i = 0; i < seq.size(); i++) {
          deliver(mgr, net.sent[seq[i]]);
          if(i + 1 < seq.size())
            EXPECT_EQ(remote.load(), 0) << "fired after " << (i + 1) << " fragments";
        }
        // an inline handler has already run; a queued one runs here (do_work insists
        //  on there being something queued)
        if(!use_inline)
          mgr.do_work(TimeLimit());
        mgr.shutdown();

        EXPECT_EQ(remote.load(), 1) << "order " << order << " inline " << use_inline;
        EXPECT_EQ(local.load(), 1);
        ASSERT_EQ(received.size(), 1u);
        EXPECT_EQ(received[0].payload, data);
      }
    }
  }

  TEST_F(ActMsgChunkedSendTest, CancelSendsNothingAndFiresNoCompletion)
  {
    std::atomic<int> local{0}, remote{0};
    std::vector<char> data = pattern(400, 10);
    ActiveMessage<QueuedMsg> m(1, data.size());
    m.add_payload(data.data(), data.size());
    m.add_local_completion([&local]() { local.fetch_add(1); });
    m.add_remote_completion([&remote]() { remote.fetch_add(1); });
    m.cancel();
    EXPECT_TRUE(net.sent.empty());
    EXPECT_EQ(local.load(), 0);
    EXPECT_EQ(remote.load(), 0);
  }

  // NodeSet sends: a single target is sent like a unicast (so MPI can carry a remote
  //  completion), two or more go through the network's multi-target path.  Remove
  //  together with ActiveMessage(NodeSet, ...) (PR 465).
  TEST_F(ActMsgChunkedSendTest, NodeSetTargets)
  {
    std::vector<char> data = pattern(200, 11);
    NodeSet one, two;
    one.add(1);
    two.add(1);
    two.add(2);
    {
      ActiveMessage<QueuedMsg> m(one, data.size());
      m.add_payload(data.data(), data.size());
      m.commit();
    }
    const size_t n1 = net.sent.size();
    for(size_t i = 0; i < n1; i++) {
      EXPECT_FALSE(net.sent[i].multicast);
      EXPECT_EQ(net.sent[i].targets, std::vector<NodeID>(1, 1));
    }
    {
      ActiveMessage<QueuedMsg> m(two, data.size());
      m.add_payload(data.data(), data.size());
      m.commit();
    }
    ASSERT_GT(net.sent.size(), n1);
    for(size_t i = n1; i < net.sent.size(); i++) {
      EXPECT_TRUE(net.sent[i].multicast);
      EXPECT_EQ(net.sent[i].targets.size(), 2u);
    }
  }

  // A-06: a payload above the limit whose fragment header the network cannot carry
  //  fails at the first such send, naming the type, rather than overrunning the
  //  network's header buffer
  TEST_F(ActMsgChunkedSendTest, UnfragmentableHeaderIsReportedByName)
  {
    // the plain header fits, the 16-byte-larger fragment header does not
    net.max_header = sizeof(BigHeaderMsg) + 8;
    ASSERT_GT(sizeof(WrappedWithFragInfo<BigHeaderMsg>), net.max_header);
    std::vector<char> data = pattern(1000, 12);
    EXPECT_DEATH(
        {
          ActiveMessage<BigHeaderMsg> m(1, data.size());
          m.add_payload(data.data(), data.size());
          m.commit();
        },
        // the fatal message names the type, but logging is not configured in the
        //  unit tests, so (like the other death tests here) only the death is checked
        "");
  }

  TEST(ActMsgHandlerTableTest, ConstructingTwiceDoesNotDuplicateEntries)
  {
    activemsg_handler_table.construct_handler_table();
    size_t n = activemsg_handler_table.num_handlers();
    activemsg_handler_table.construct_handler_table();
    EXPECT_EQ(activemsg_handler_table.num_handlers(), n);
  }

} // namespace
