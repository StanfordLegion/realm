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

#include <gtest/gtest.h>
#include "realm/activemsg.h"
#include "realm/timers.h"
#include "realm/threads.h"
#include "realm/activemsg.h"
#include <vector>
#include <atomic>

using namespace Realm;

namespace {

  struct RegularMessage {
    int dummy{0};

    // inline handler that simply records the payload we receive
    static std::vector<std::vector<char>> received_payloads;
    static std::atomic<int> call_count;

    static bool handle_inline(NodeID /*sender*/, const RegularMessage & /*hdr*/,
                              const void *payload, size_t payload_size, TimeLimit /*tl*/)
    {
      const char *c = static_cast<const char *>(payload);
      RegularMessage::received_payloads.emplace_back(c, c + payload_size);
      call_count.fetch_add(1, std::memory_order_relaxed);
      return true; // handled inline, so add_incoming_message returns true
    }

    // Provide a non-inline handler to satisfy ActiveMessageHandlerTable invariant
    static void handle_message(NodeID sender, const RegularMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      (void)sender;
      (void)hdr;
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };
  std::vector<std::vector<char>> RegularMessage::received_payloads;
  std::atomic<int> RegularMessage::call_count{0};

  struct FragmentedMessage {
    FragmentInfo frag_info;
    int dummy{0};

    // inline handler that simply records the payload we receive
    static std::vector<std::vector<char>> received_payloads;
    static std::atomic<int> call_count;

    static bool handle_inline(NodeID /*sender*/, const FragmentedMessage & /*hdr*/,
                              const void *payload, size_t payload_size, TimeLimit /*tl*/)
    {
      const char *c = static_cast<const char *>(payload);
      FragmentedMessage::received_payloads.emplace_back(c, c + payload_size);
      call_count.fetch_add(1, std::memory_order_relaxed);
      return true; // handled inline, so add_incoming_message returns true
    }

    // Provide a non-inline handler to satisfy ActiveMessageHandlerTable invariant
    static void handle_message(NodeID sender, const FragmentedMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      (void)sender;
      (void)hdr;
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };
  std::vector<std::vector<char>> FragmentedMessage::received_payloads;
  std::atomic<int> FragmentedMessage::call_count{0};

  // A fragment-carrying type with NO inline handler, so delivery always goes through
  //  IncomingMessageManager's block-staging path.  Every other fixture in this file
  //  defines handle_inline, which returns before the staging allocator is touched -
  //  which is why an oversized reassembled payload was never exercised here.
  struct OversizedFragMessage {
    FragmentInfo frag_info;
    int dummy{0};

    static std::vector<char> last_payload;
    static std::atomic<int> call_count;

    static void handle_message(NodeID /*sender*/, const OversizedFragMessage & /*hdr*/,
                               const void *payload, size_t payload_size, TimeLimit /*tl*/)
    {
      const char *c = static_cast<const char *>(payload);
      OversizedFragMessage::last_payload.assign(c, c + payload_size);
      call_count.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::vector<char> OversizedFragMessage::last_payload;
  std::atomic<int> OversizedFragMessage::call_count{0};

  static ActiveMessageHandlerReg<OversizedFragMessage> oversized_frag_msg_reg;

  // A second fragment-carrying type, used to show that two types can share a msg_id
  //  without their fragments interfering.
  struct SecondFragMessage {
    FragmentInfo frag_info;
    int dummy{0};

    static std::vector<std::vector<char>> received_payloads;
    static std::atomic<int> call_count;

    static bool handle_inline(NodeID /*sender*/, const SecondFragMessage & /*hdr*/,
                              const void *payload, size_t payload_size, TimeLimit /*tl*/)
    {
      const char *c = static_cast<const char *>(payload);
      SecondFragMessage::received_payloads.emplace_back(c, c + payload_size);
      call_count.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    static void handle_message(NodeID sender, const SecondFragMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };
  std::vector<std::vector<char>> SecondFragMessage::received_payloads;
  std::atomic<int> SecondFragMessage::call_count{0};

  static ActiveMessageHandlerReg<SecondFragMessage> second_frag_msg_reg;

  static ActiveMessageHandlerReg<RegularMessage> reg_msg_reg;
  static ActiveMessageHandlerReg<FragmentedMessage> frag_msg_reg;

  class IncomingMessageManagerTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      FragmentedMessage::received_payloads.clear();
      FragmentedMessage::call_count.store(0);

      activemsg_handler_table.construct_handler_table();
    }
  };

  TEST_F(IncomingMessageManagerTest, RegularAddIncoming)
  {
    CoreReservationSet crs(nullptr);
    const int nodes = 2;
    IncomingMessageManager mgr(nodes, /*dedicated_threads=*/0, crs);

    std::vector<char> full_msg(37);
    for(size_t i = 0; i < full_msg.size(); ++i) {
      full_msg[i] = static_cast<char>(i);
    }

    NodeID sender = 1;
    unsigned short msgid = activemsg_handler_table.lookup_message_id<RegularMessage>();
    RegularMessage hdr;

    bool handled = mgr.add_incoming_message(
        sender, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY, full_msg.data(), full_msg.size(),
        PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
    EXPECT_TRUE(handled);

    ASSERT_EQ(RegularMessage::call_count.load(), 1);

    ASSERT_EQ(RegularMessage::received_payloads.size(), 1u);
    const auto &payload0 = RegularMessage::received_payloads.front();
    EXPECT_EQ(payload0.size(), full_msg.size());
    EXPECT_EQ(payload0, full_msg);
    mgr.shutdown();
  }

  TEST_F(IncomingMessageManagerTest, FragmentReassemblyViaAddIncoming)
  {
    CoreReservationSet crs(nullptr);
    const int nodes = 2;
    IncomingMessageManager mgr(nodes, /*dedicated_threads=*/0, crs);

    // prepare data to send
    const size_t max_payload = 10;
    std::vector<char> full_msg(37);
    for(size_t i = 0; i < full_msg.size(); ++i)
      full_msg[i] = static_cast<char>(i);

    // split into chunks
    size_t total_chunks = (full_msg.size() + max_payload - 1) / max_payload;
    uint64_t msg_id = 0xABCD1234ULL; // arbitrary message id
    NodeID sender = 1;

    unsigned short msgid = activemsg_handler_table.lookup_message_id<FragmentedMessage>();

    size_t offset = 0;
    for(uint32_t chunk_id = 0; chunk_id < total_chunks; ++chunk_id) {
      size_t chunk_size = std::min(max_payload, full_msg.size() - offset);

      FragmentedMessage hdr;
      hdr.frag_info.chunk_id = chunk_id;
      hdr.frag_info.total_chunks = static_cast<uint32_t>(total_chunks);
      hdr.frag_info.msg_id = msg_id;

      const void *payload_ptr = full_msg.data() + offset;

      bool handled = mgr.add_incoming_message(sender, msgid, &hdr, sizeof(hdr),
                                              PAYLOAD_COPY, payload_ptr, chunk_size,
                                              PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
      if(chunk_id < total_chunks - 1) {
        EXPECT_FALSE(handled) << "Intermediate fragment should not be handled.";
      } else {
        EXPECT_TRUE(handled) << "Last fragment should be handled inline.";
      }

      offset += chunk_size;
    }

    // Exactly one inline call should have occurred
    ASSERT_EQ(FragmentedMessage::call_count.load(), 1);

    // The payload passed to the handler must equal the entire original message
    ASSERT_EQ(FragmentedMessage::received_payloads.size(), 1u);
    const auto &payload0 = FragmentedMessage::received_payloads.front();
    EXPECT_EQ(payload0.size(), full_msg.size());
    EXPECT_EQ(payload0, full_msg);
    mgr.shutdown();
  }

  // A reassembled payload larger than one message block cannot be staged out of the
  //  block allocator - blocks are a fixed size and are never grown, so before the
  //  oversized-payload fallback this aborted in MessageBlock::append_message (and, in
  //  a release build, fell through to a null dereference).  The sizes below straddle
  //  and then clear the ~1MB block size.
  TEST_F(IncomingMessageManagerTest, OversizedReassembledPayloadIsDelivered)
  {
    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

    const size_t sizes[] = {1048368, 1048369, 2u << 20, 8u << 20};
    const size_t max_chunk = 64 * 1024;
    unsigned short msgid =
        activemsg_handler_table.lookup_message_id<OversizedFragMessage>();

    for(size_t i = 0; i < (sizeof(sizes) / sizeof(sizes[0])); i++) {
      const size_t total = sizes[i];

      OversizedFragMessage::last_payload.clear();
      OversizedFragMessage::call_count.store(0);

      std::vector<char> full(total);
      for(size_t j = 0; j < total; j++)
        full[j] = static_cast<char>(j & 0xff);

      const uint32_t total_chunks =
          static_cast<uint32_t>((total + max_chunk - 1) / max_chunk);

      size_t offset = 0;
      for(uint32_t chunk_id = 0; chunk_id < total_chunks; chunk_id++) {
        const size_t chunk_size = std::min(max_chunk, total - offset);
        OversizedFragMessage hdr;
        hdr.frag_info = {chunk_id, total_chunks, static_cast<uint64_t>(0x5150 + i)};
        mgr.add_incoming_message(1, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY,
                                 full.data() + offset, chunk_size, PAYLOAD_COPY, nullptr,
                                 0, 0, TimeLimit());
        offset += chunk_size;
      }

      // no inline handler, so the reassembled message was queued rather than run -
      //  drive it from this thread rather than depending on handler threads
      mgr.do_work(TimeLimit());

      EXPECT_EQ(OversizedFragMessage::call_count.load(), 1) << "size " << total;
      ASSERT_EQ(OversizedFragMessage::last_payload.size(), total) << "size " << total;
      EXPECT_EQ(OversizedFragMessage::last_payload, full) << "size " << total;
    }

    mgr.shutdown();
  }


  // Two different chunked message types from one sender used to share a reassembly
  //  key.  next_chunk_message_id's counter was a function-local static inside a member
  //  of the ActiveMessage<T> class template, so every instantiation restarted at zero
  //  and handed out the same ids; the second type's fragments then landed on the first
  //  type's FragmentedMessage, were rejected as duplicates and silently dropped under
  //  NDEBUG, corrupting one message and hanging the other.  Interleave two types that
  //  deliberately claim the SAME msg_id and check both still arrive intact.
  TEST_F(IncomingMessageManagerTest, DistinctTypesSharingAFragmentIdDoNotCollide)
  {
    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

    FragmentedMessage::received_payloads.clear();
    FragmentedMessage::call_count.store(0);
    SecondFragMessage::received_payloads.clear();
    SecondFragMessage::call_count.store(0);

    const NodeID sender = 1;
    const uint64_t shared_msg_id = 0xDEADBEEFULL; // same id for both types, on purpose

    std::vector<char> data_a(30);
    for(size_t i = 0; i < data_a.size(); i++)
      data_a[i] = static_cast<char>('A' + (i % 26));
    std::vector<char> data_b(20);
    for(size_t i = 0; i < data_b.size(); i++)
      data_b[i] = static_cast<char>('a' + (i % 26));

    const size_t chunk_a = 10; // 3 chunks
    const size_t chunk_b = 10; // 2 chunks
    unsigned short msgid_a = activemsg_handler_table.lookup_message_id<FragmentedMessage>();
    unsigned short msgid_b =
        activemsg_handler_table.lookup_message_id<SecondFragMessage>();
    ASSERT_NE(msgid_a, msgid_b);

    auto send_a = [&](uint32_t chunk_id) {
      FragmentedMessage hdr;
      hdr.frag_info = {chunk_id, 3, shared_msg_id};
      return mgr.add_incoming_message(sender, msgid_a, &hdr, sizeof(hdr), PAYLOAD_COPY,
                                      data_a.data() + (chunk_id * chunk_a), chunk_a,
                                      PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
    };
    auto send_b = [&](uint32_t chunk_id) {
      SecondFragMessage hdr;
      hdr.frag_info = {chunk_id, 2, shared_msg_id};
      return mgr.add_incoming_message(sender, msgid_b, &hdr, sizeof(hdr), PAYLOAD_COPY,
                                      data_b.data() + (chunk_id * chunk_b), chunk_b,
                                      PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
    };

    // interleaved: A0 B0 A1 B1(completes B) A2(completes A)
    EXPECT_FALSE(send_a(0));
    EXPECT_FALSE(send_b(0));
    EXPECT_FALSE(send_a(1));
    EXPECT_TRUE(send_b(1));
    EXPECT_TRUE(send_a(2));

    ASSERT_EQ(FragmentedMessage::call_count.load(), 1);
    ASSERT_EQ(SecondFragMessage::call_count.load(), 1);
    ASSERT_EQ(FragmentedMessage::received_payloads.size(), 1u);
    ASSERT_EQ(SecondFragMessage::received_payloads.size(), 1u);
    EXPECT_EQ(FragmentedMessage::received_payloads.front(), data_a);
    EXPECT_EQ(SecondFragMessage::received_payloads.front(), data_b);

    mgr.shutdown();
  }


  class IncomingMessageManagerDeathTest : public IncomingMessageManagerTest {};

  // Realm's transports deliver every message exactly once and nothing retransmits, so
  //  a repeated fragment means a bug upstream.  Absorbing it would also mask two
  //  messages colliding on one reassembly key, which is the failure Fix B addresses.
  TEST_F(IncomingMessageManagerDeathTest, DuplicateFragmentAborts)
  {
    // wrapped in a lambda so the commas below are not parsed as macro arguments
    auto deliver_same_chunk_twice = []() {
      CoreReservationSet crs(nullptr);
      IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

      const char data[] = "chunk";
      unsigned short msgid =
          activemsg_handler_table.lookup_message_id<FragmentedMessage>();
      FragmentedMessage hdr;
      hdr.frag_info = {0, 3, 0x1234ULL};

      mgr.add_incoming_message(1, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY, data,
                               sizeof(data), PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
      // the very same chunk again
      mgr.add_incoming_message(1, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY, data,
                               sizeof(data), PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
    };

    EXPECT_DEATH(deliver_same_chunk_twice(), "");
  }

} // anonymous namespace
