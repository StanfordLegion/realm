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

// Large active message payloads on a real network, between real ranks:
//   - a payload above the network's per-message limit is sent in fragments and
//     reassembled, including above the receiver's ~1MB message block size
//   - the same through a multi-target (NodeSet) send
//   - two message types fragmented at the same time by two threads of one rank, which
//     used to share a reassembly on the receiver and abort it (or, with fragments
//     arriving out of order, corrupt both and lose one)
//   - local and remote completions on a fragmented message, each fired exactly once
//   - a global task registration whose user data is larger than one message block
//
// Needs at least 2 ranks and -ll:cpu 3; with fewer ranks there is no remote target and
//  CTest reports it as skipped.  Networks with no per-message limit (UCX) never fragment, but still
//  carry every payload here in one piece.  Waits on events, not timeouts: a lost
//  message shows up as the test harness's timeout.

#include "realm.h"
#include "realm/activemsg.h"
#include "realm/network.h"
#include "realm/nodeset.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace Realm;

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  SENDER_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 1,
  USERDATA_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 2,
};

// a global registration needs a portable implementation, which Realm derives from the
//  function's exported symbol, so this one has external linkage
void actmsg_chunking_userdata_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor p);

namespace {

  Logger log_app("actmsg_chunking");

  std::atomic<int> g_errors{0};

  // acknowledgements for the phase in progress: the phase's event triggers once the
  //  expected number has arrived
  std::atomic<int> g_acks{0};
  int g_expected_acks = 0;
  UserEvent g_phase_done;

  unsigned char pattern_byte(int seq, size_t i)
  {
    return static_cast<unsigned char>(((i * 31) + seq) & 0xff);
  }

  std::vector<unsigned char> make_payload(size_t bytes, int seq)
  {
    std::vector<unsigned char> v(bytes);
    for(size_t i = 0; i < bytes; i++)
      v[i] = pattern_byte(seq, i);
    return v;
  }

  bool check_payload(int seq, size_t expected, const void *data, size_t len)
  {
    if(len != expected)
      return false;
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for(size_t i = 0; i < len; i++)
      if(p[i] != pattern_byte(seq, i))
        return false;
    return true;
  }

  struct AckMessage {
    int seq;
    int ok;

    static void handle_message(NodeID sender, const AckMessage &args, const void *,
                               size_t)
    {
      if(!args.ok) {
        log_app.error() << "rank " << sender << " received a corrupted payload, seq="
                        << args.seq;
        g_errors.fetch_add(1);
      }
      if((g_acks.fetch_add(1) + 1) == g_expected_acks)
        g_phase_done.trigger();
    }
  };

  void ack(NodeID to, int seq, bool ok)
  {
    ActiveMessage<AckMessage> amsg(to);
    amsg->seq = seq;
    amsg->ok = ok ? 1 : 0;
    amsg.commit();
  }

  // no inline handler, so a reassembled message is always staged in the receiver's
  //  message blocks
  struct BigMessage {
    int seq;
    size_t total;
    NodeID reply_to;

    static void handle_message(NodeID, const BigMessage &args, const void *data,
                               size_t len)
    {
      ack(args.reply_to, args.seq, check_payload(args.seq, args.total, data, len));
    }
  };

  // a second fragmented type, sent concurrently with BigMessage
  struct OtherBigMessage {
    int seq;
    size_t total;
    NodeID reply_to;

    static void handle_message(NodeID, const OtherBigMessage &args, const void *data,
                               size_t len)
    {
      ack(args.reply_to, args.seq, check_payload(args.seq, args.total, data, len));
    }
  };

  ActiveMessageHandlerReg<AckMessage> ack_message_handler;
  ActiveMessageHandlerReg<BigMessage> big_message_handler;
  ActiveMessageHandlerReg<OtherBigMessage> other_big_message_handler;

  void begin_phase(int expected)
  {
    g_acks.store(0);
    g_expected_acks = expected;
    g_phase_done = UserEvent::create_user_event();
  }

  template <typename HDR>
  void send_big(NodeID target, int seq, const std::vector<unsigned char> &buf)
  {
    ActiveMessage<HDR> amsg(target, buf.size());
    amsg->seq = seq;
    amsg->total = buf.size();
    amsg->reply_to = Network::my_node_id;
    amsg.add_payload(buf.data(), buf.size());
    amsg.commit();
  }

  void test_unicast(NodeID target, size_t limit)
  {
    std::vector<size_t> sizes;
    if(limit < (64 << 20)) {
      sizes.push_back(limit + 1);
      sizes.push_back((3 * limit) + 7);
    }
    sizes.push_back(2 << 20); // above one ~1MB message block
    for(size_t i = 0; i < sizes.size(); i++) {
      begin_phase(1);
      send_big<BigMessage>(target, int(10 + i), make_payload(sizes[i], int(10 + i)));
      g_phase_done.wait();
    }
    printf("[actmsg_chunking] unicast: %zu sizes up to %zu bytes\n", sizes.size(),
           sizes.back());
  }

  void test_multicast(NodeID num_nodes)
  {
    NodeSet targets;
    for(NodeID n = 1; n < num_nodes; n++)
      targets.add(n);
    const int seq = 20;
    std::vector<unsigned char> buf = make_payload(1 << 20, seq);
    begin_phase(int(targets.size()));
    ActiveMessage<BigMessage> amsg(targets, buf.size());
    amsg->seq = seq;
    amsg->total = buf.size();
    amsg->reply_to = Network::my_node_id;
    amsg.add_payload(buf.data(), buf.size());
    amsg.commit();
    g_phase_done.wait();
    printf("[actmsg_chunking] multicast to %zu ranks\n", targets.size());
  }

  // two sender tasks on two processors of this rank, released together every round
  std::atomic<int> g_arrivals{0};

  struct SenderArgs {
    int which;
    int rounds;
    NodeID target;
    size_t bytes;
  };

  void sender_task(const void *args, size_t arglen, const void *, size_t, Processor)
  {
    assert(arglen == sizeof(SenderArgs));
    SenderArgs a;
    memcpy(&a, args, sizeof(a));
    for(int r = 0; r < a.rounds; r++) {
      const int seq = 1000 + (2 * r) + a.which;
      std::vector<unsigned char> buf = make_payload(a.bytes, seq);
      g_arrivals.fetch_add(1);
      while(g_arrivals.load() < (2 * (r + 1))) {
      }
      if(a.which == 0)
        send_big<BigMessage>(a.target, seq, buf);
      else
        send_big<OtherBigMessage>(a.target, seq, buf);
    }
  }

  void test_concurrent_types(NodeID target, size_t limit)
  {
    std::vector<Processor> cpus;
    Machine::ProcessorQuery pq(Machine::get_machine());
    pq.only_kind(Processor::LOC_PROC);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
      if(NodeID((*it).address_space()) == Network::my_node_id)
        cpus.push_back(*it);
    if(cpus.size() < 3) {
      printf("[actmsg_chunking] concurrent types: skipped, needs -ll:cpu 3\n");
      return;
    }
    const int rounds = 50;
    const size_t bytes = std::min(size_t(4) * limit, size_t(256) << 10);
    g_arrivals.store(0);
    begin_phase(2 * rounds);
    std::vector<Event> done;
    for(int w = 0; w < 2; w++) {
      SenderArgs a{w, rounds, target, bytes};
      done.push_back(cpus[1 + w].spawn(SENDER_TASK, &a, sizeof(a)));
    }
    Event::merge_events(done).wait();
    g_phase_done.wait();
    printf("[actmsg_chunking] concurrent types: %d rounds of %zu bytes\n", rounds, bytes);
  }

  void test_completions(NodeID target, size_t limit)
  {
    static std::atomic<int> local{0}, remote{0};
    local.store(0);
    remote.store(0);
    const int seq = 30;
    const size_t bytes = std::min(size_t(3) * limit, size_t(512) << 10);
    std::vector<unsigned char> buf = make_payload(bytes, seq);
    begin_phase(1);
    UserEvent local_done = UserEvent::create_user_event();
    UserEvent remote_done = UserEvent::create_user_event();
    {
      ActiveMessage<BigMessage> amsg(target, buf.size());
      amsg->seq = seq;
      amsg->total = buf.size();
      amsg->reply_to = Network::my_node_id;
      amsg.add_payload(buf.data(), buf.size());
      amsg.add_local_completion([local_done]() {
        local.fetch_add(1);
        local_done.trigger();
      });
      amsg.add_remote_completion([remote_done]() {
        remote.fetch_add(1);
        remote_done.trigger();
      });
      amsg.commit();
    }
    g_phase_done.wait();
    local_done.wait();
    remote_done.wait();
    if((local.load() != 1) || (remote.load() != 1)) {
      log_app.error() << "completions: local fired " << local.load() << " times, remote "
                      << remote.load() << " times, expected 1 and 1";
      g_errors.fetch_add(1);
    }
    printf("[actmsg_chunking] completions: local %d, remote %d\n", local.load(),
           remote.load());
  }

  // the user data of a global registration travels in one RegisterTaskMessage per
  //  rank, of whatever size the caller chose
  const size_t USERDATA_BYTES = 2 << 20;
  const int USERDATA_SEQ = 40;

  void test_global_registration(NodeID target)
  {
    std::vector<unsigned char> ud = make_payload(USERDATA_BYTES, USERDATA_SEQ);
    Processor::register_task_by_kind(Processor::LOC_PROC, true /*global*/, USERDATA_TASK,
                                     CodeDescriptor(actmsg_chunking_userdata_task),
                                     ProfilingRequestSet(), ud.data(), ud.size())
        .wait();
    Processor tp = Processor::NO_PROC;
    Machine::ProcessorQuery pq(Machine::get_machine());
    pq.only_kind(Processor::LOC_PROC);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
      if(NodeID((*it).address_space()) == target) {
        tp = *it;
        break;
      }
    assert(tp.exists());
    begin_phase(1);
    NodeID me = Network::my_node_id;
    tp.spawn(USERDATA_TASK, &me, sizeof(me));
    g_phase_done.wait();
    printf("[actmsg_chunking] global registration with %zu bytes of user data\n",
           USERDATA_BYTES);
  }

  void main_task(const void *, size_t, const void *, size_t, Processor p)
  {
    const NodeID num_nodes = Network::max_node_id + 1;
    const size_t limit = Network::max_payload_size(sizeof(BigMessage), nullptr);
    printf("[actmsg_chunking] ranks=%d per-message limit=%zu\n", int(num_nodes), limit);
    fflush(stdout);
    if(num_nodes < 2) {
      printf("[actmsg_chunking] needs at least 2 ranks - nothing to test\n");
      return;
    }
    const NodeID target = 1;
    test_unicast(target, limit);
    test_multicast(num_nodes);
    test_concurrent_types(target, std::min(limit, size_t(64) << 10));
    test_completions(target, std::min(limit, size_t(64) << 10));
    test_global_registration(target);
    printf("[actmsg_chunking] done: errors=%d\n", g_errors.load());
    fflush(stdout);
  }

}; // namespace

void actmsg_chunking_userdata_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor)
{
  NodeID reply_to = *static_cast<const NodeID *>(args);
  ack(reply_to, USERDATA_SEQ,
      check_payload(USERDATA_SEQ, USERDATA_BYTES, userdata, userlen));
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);
  if(Network::max_node_id < 1) {
    printf("[actmsg_chunking] skipped: needs at least 2 ranks\n");
    rt.shutdown();
    int ret = rt.wait_for_shutdown();
    return ret ? ret : 77;
  }

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, SENDER_TASK,
                                   CodeDescriptor(sender_task), ProfilingRequestSet())
      .external_wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  int ret = rt.wait_for_shutdown();
  return ((g_errors.load() != 0) ? 1 : ret);
}
