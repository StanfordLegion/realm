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

// End-to-end coverage for large active-message payloads:
//   - automatic chunking of a payload larger than the network's per-message limit
//   - reassembly of a payload larger than one IncomingMessageManager block
//   - multicast of a chunked payload, including through a relay when ranks > radix
//   - two chunked message types in flight at once (distinct reassembly identities)
//   - local and remote completion callbacks on a chunked message
//
// None of this is reachable from the unit tests: those run on LoopbackNetworkModule,
// whose max_payload_size() is SIZE_MAX, so ActiveMessage never enters chunked mode and
// create_active_message_impl() aborts.  A real network backend is required, which makes
// this the only place the chunked send path is exercised at all.

#include "realm.h"
#include "realm/activemsg.h"
#include "realm/multicast.h"
#include "realm/network.h"
#include "realm/nodeset.h"
#include "realm/serialize.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

using namespace Realm;

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  FETCH_METADATA_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 1
};

namespace {

  Logger log_app("actmsg_large_payload");

  std::atomic<int> g_errors{0};
  std::atomic<int> g_acks{0};
  std::atomic<int> g_local_comp{0};
  std::atomic<int> g_remote_comp{0};

  // payload big enough to clear the ~1MB IncomingMessageManager block size, which is
  //  where an oversized reassembled message used to have nowhere to live
  const size_t OVERSIZED = 2 << 20;

  unsigned char pattern_byte(int seq, size_t i)
  {
    return static_cast<unsigned char>(((i * 31) + seq) & 0xff);
  }

  void fill_pattern(std::vector<unsigned char> &buf, int seq)
  {
    for(size_t i = 0; i < buf.size(); i++)
      buf[i] = pattern_byte(seq, i);
  }

  struct AckMessage {
    int seq;
    int ok;

    static void handle_message(NodeID /*sender*/, const AckMessage &args,
                               const void * /*data*/, size_t /*datalen*/,
                               TimeLimit /*work_until*/)
    {
      if(!args.ok)
        g_errors.fetch_add(1, std::memory_order_relaxed);
      g_acks.fetch_add(1, std::memory_order_relaxed);
    }
  };

  // Verifies a received payload and acknowledges it.  Shared by both big message types.
  template <typename HDR>
  void verify_and_ack(const HDR &args, const void *data, size_t datalen, const char *what)
  {
    bool ok = (datalen == args.total_size);
    if(!ok) {
      log_app.error() << what << " seq=" << args.seq << ": expected " << args.total_size
                      << " bytes, got " << datalen;
    } else {
      const unsigned char *p = static_cast<const unsigned char *>(data);
      for(size_t i = 0; i < datalen; i++)
        if(p[i] != pattern_byte(args.seq, i)) {
          log_app.error() << what << " seq=" << args.seq << ": payload mismatch at byte "
                          << i;
          ok = false;
          break;
        }
    }

    if(args.reply_to == Network::my_node_id) {
      // locally delivered (origin was in its own multicast target set) - there is no
      //  self-addressed active message to send, and on a single-rank run the loopback
      //  network would abort if we tried
      if(!ok)
        g_errors.fetch_add(1, std::memory_order_relaxed);
      g_acks.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    ActiveMessage<AckMessage> amsg(args.reply_to);
    amsg->seq = args.seq;
    amsg->ok = (ok ? 1 : 0);
    amsg.commit();
  }

  // Deliberately no handle_inline: an inline handler returns before the staging
  //  allocator is ever touched, which is what hid the block-size ceiling for so long.
  struct BigMessage {
    int seq;
    size_t total_size;
    NodeID reply_to;

    static void handle_message(NodeID /*sender*/, const BigMessage &args,
                               const void *data, size_t datalen, TimeLimit /*until*/)
    {
      verify_and_ack(args, data, datalen, "BigMessage");
    }
  };

  // A second chunked type, so that two reassemblies can be open on one receiver at the
  //  same time.  Their fragment identities must not collide.
  struct BigMessage2 {
    int seq;
    size_t total_size;
    NodeID reply_to;

    static void handle_message(NodeID /*sender*/, const BigMessage2 &args,
                               const void *data, size_t datalen, TimeLimit /*until*/)
    {
      verify_and_ack(args, data, datalen, "BigMessage2");
    }
  };

  ActiveMessageHandlerReg<AckMessage> ack_message_handler;
  ActiveMessageHandlerReg<BigMessage> big_message_handler;
  ActiveMessageHandlerReg<BigMessage2> big_message2_handler;

  // Spins until 'count' more acknowledgements have arrived since 'base'.  Relative
  //  rather than cumulative, so a phase does not depend on which phases ran before it.
  //  The handlers run on other threads, so this only needs to yield.
  bool wait_for_acks(int base, int count, const char *what)
  {
    const int want = base + count;
    for(int spins = 0; (g_acks.load() < want) && (spins < 60000); spins++)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(g_acks.load() < want) {
      log_app.error() << what << ": timed out waiting for acks ("
                      << (g_acks.load() - base) << " of " << count << ")";
      g_errors.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    return true;
  }

  void test_unicast_chunked(NodeID target)
  {
    const int seq = 1;
    const int base = g_acks.load();
    std::vector<unsigned char> buf(OVERSIZED);
    fill_pattern(buf, seq);

    ActiveMessage<BigMessage> amsg(target, buf.size());
    amsg->seq = seq;
    amsg->total_size = buf.size();
    amsg->reply_to = Network::my_node_id;
    amsg.add_payload(buf.data(), buf.size());
    amsg.commit();

    wait_for_acks(base, 1, "unicast");
  }

  void test_multicast_chunked(const NodeSet &targets, int num_targets)
  {
    const int seq = 2;
    const int base = g_acks.load();
    std::vector<unsigned char> buf(OVERSIZED);
    fill_pattern(buf, seq);

    BigMessage hdr;
    hdr.seq = seq;
    hdr.total_size = buf.size();
    hdr.reply_to = Network::my_node_id;
    multicast_message(MulticastTargetSet(targets), hdr, buf.data(), buf.size());

    wait_for_acks(base, num_targets, "multicast");
  }

  // Two chunked types committed back to back, so both reassemblies are open on the
  //  receiver simultaneously.  They must not be confused for one another.
  void test_concurrent_chunked_types(NodeID target)
  {
    const int seq_a = 3, seq_b = 4;
    const size_t sz = 512 * 1024;
    std::vector<unsigned char> buf_a(sz), buf_b(sz);
    fill_pattern(buf_a, seq_a);
    fill_pattern(buf_b, seq_b);

    const int base = g_acks.load();

    ActiveMessage<BigMessage> a(target, sz);
    a->seq = seq_a;
    a->total_size = sz;
    a->reply_to = Network::my_node_id;
    a.add_payload(buf_a.data(), sz);

    ActiveMessage<BigMessage2> b(target, sz);
    b->seq = seq_b;
    b->total_size = sz;
    b->reply_to = Network::my_node_id;
    b.add_payload(buf_b.data(), sz);

    a.commit();
    b.commit();

    wait_for_acks(base, 2, "concurrent types");
  }

  // Completion callbacks on a payload large enough to be chunked.  Local completion is
  //  satisfied once commit() returns, since every chunk has been copied into a network
  //  buffer by then; remote completion must wait for the target to reassemble and run
  //  the handler.
  void test_chunked_completions(NodeID target)
  {
    const int seq = 5;
    const size_t sz = 512 * 1024;
    std::vector<unsigned char> buf(sz);
    fill_pattern(buf, seq);

    const int base = g_acks.load();
    g_local_comp.store(0);
    g_remote_comp.store(0);

    ActiveMessage<BigMessage> amsg(target, sz);
    amsg->seq = seq;
    amsg->total_size = sz;
    amsg->reply_to = Network::my_node_id;
    amsg.add_payload(buf.data(), sz);
    amsg.add_local_completion([]() { g_local_comp.fetch_add(1); });
    amsg.add_remote_completion([]() { g_remote_comp.fetch_add(1); });
    amsg.commit();

    if(g_local_comp.load() != 1) {
      log_app.error() << "local completion did not fire by the time commit() returned: "
                      << g_local_comp.load();
      g_errors.fetch_add(1, std::memory_order_relaxed);
    }

    if(!wait_for_acks(base, 1, "completions"))
      return;

    for(int spins = 0; (g_remote_comp.load() < 1) && (spins < 60000); spins++)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if(g_remote_comp.load() != 1) {
      log_app.error() << "remote completion fired " << g_remote_comp.load()
                      << " times, expected exactly 1";
      g_errors.fetch_add(1, std::memory_order_relaxed);
    }
    if(g_local_comp.load() != 1) {
      log_app.error() << "local completion fired " << g_local_comp.load()
                      << " times, expected exactly 1";
      g_errors.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Enough one-byte fields that the serialized InstanceLayoutGeneric clears the
  //  IncomingMessageManager block size; the test prints the actual figure so a layout
  //  format change that shrinks it does not silently defeat the point of the test.
  const size_t METADATA_FIELDS = 60000;

  struct FetchMetadataArgs {
    RegionInstance inst;
    UserEvent requested;
    NodeID reply_to;
    size_t expected_fields;
  };

  // Runs on a remote node: requests the instance's metadata while it is still
  //  unavailable, which makes this node an "early requestor" - the set that
  //  RegionInstanceImpl::send_metadata multicasts to once the instance becomes valid.
  void fetch_metadata_task(const void *args, size_t arglen, const void * /*userdata*/,
                           size_t /*userlen*/, Processor p)
  {
    assert(arglen == sizeof(FetchMetadataArgs));
    FetchMetadataArgs fa;
    memcpy(&fa, args, sizeof(fa));

    Event e = fa.inst.fetch_metadata(p);

    // let the origin release the instance now that our request is on its way
    fa.requested.trigger();
    e.wait();

    bool ok = true;
    const InstanceLayoutGeneric *layout = fa.inst.get_layout();
    if(layout == nullptr) {
      log_app.error() << "instance metadata: no layout after fetch_metadata";
      ok = false;
    } else if(layout->fields.size() != fa.expected_fields) {
      log_app.error() << "instance metadata: got " << layout->fields.size()
                      << " fields, expected " << fa.expected_fields;
      ok = false;
    }

    ActiveMessage<AckMessage> amsg(fa.reply_to);
    amsg->seq = 6;
    amsg->ok = (ok ? 1 : 0);
    amsg.commit();
  }

  // The call site that motivated all of this: RegionInstanceImpl::send_metadata()
  //  hands a whole serialized layout to multicast_message() in one go, having dropped
  //  the application-level chunking loop it used to run.  Note this exercises the
  //  early-requestor multicast only if the remote request reaches this node before the
  //  instance becomes valid; if it loses that race the metadata arrives by the ordinary
  //  unicast response path instead.  Either way the metadata must survive intact, which
  //  is what is asserted - and neither path had coverage above 1MB before.
  void test_large_instance_metadata(Processor local_proc, NodeID target)
  {
    const int base = g_acks.load();

    Memory m = Machine::MemoryQuery(Machine::get_machine())
                   .has_affinity_to(local_proc)
                   .only_kind(Memory::SYSTEM_MEM)
                   .first();
    if(!m.exists()) {
      log_app.error() << "instance metadata: no system memory available";
      g_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    Processor remote_proc = Processor::NO_PROC;
    {
      Machine::ProcessorQuery pq(Machine::get_machine());
      pq.only_kind(Processor::LOC_PROC);
      for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
        if(NodeID((*it).address_space()) == target) {
          remote_proc = *it;
          break;
        }
    }
    if(!remote_proc.exists()) {
      log_app.error() << "instance metadata: no processor on node " << target;
      g_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    std::map<FieldID, size_t> field_sizes;
    for(size_t i = 0; i < METADATA_FIELDS; i++)
      field_sizes[FieldID(i)] = 1;

    Rect<1> rect(0, 0);
    UserEvent precond = UserEvent::create_user_event();
    UserEvent requested = UserEvent::create_user_event();

    RegionInstance inst;
    Event ready = RegionInstance::create_instance(inst, m, rect, field_sizes, 0 /*SOA*/,
                                                  ProfilingRequestSet(), precond);

    FetchMetadataArgs fa;
    fa.inst = inst;
    fa.requested = requested;
    fa.reply_to = Network::my_node_id;
    fa.expected_fields = METADATA_FIELDS;
    Event fetched = remote_proc.spawn(FETCH_METADATA_TASK, &fa, sizeof(fa));

    // only release the instance once the remote request is in flight
    requested.wait();
    precond.trigger();
    ready.wait();

    // confirm the premise: this layout really is bigger than one message block
    {
      Serialization::DynamicBufferSerializer dbs(4096);
      const InstanceLayoutGeneric *layout = inst.get_layout();
      assert(layout != nullptr);
      bool ok = (dbs << *layout);
      assert(ok);
      printf("[actmsg_large_payload] instance metadata serializes to %zu bytes\n",
             dbs.bytes_used());
      fflush(stdout);
      if(dbs.bytes_used() <= (1 << 20)) {
        log_app.error() << "instance metadata is only " << dbs.bytes_used()
                        << " bytes - raise METADATA_FIELDS, this no longer tests "
                        << "an oversized payload";
        g_errors.fetch_add(1, std::memory_order_relaxed);
      }
    }

    wait_for_acks(base, 1, "instance metadata");
    fetched.wait();
    inst.destroy();
  }

  void main_task(const void * /*args*/, size_t /*arglen*/, const void * /*userdata*/,
                 size_t /*userlen*/, Processor p)
  {
    const NodeID num_nodes = Network::max_node_id + 1;
    // printf rather than the logger: REALM_LOG_LEVEL is a compile-time floor and is set
    //  above PRINT in some configurations, which would leave this test silent about
    //  whether it did anything.  Failures below use log_app.error(), which survives.
    printf("[actmsg_large_payload] nodes=%d max_payload_size=%zu\n", int(num_nodes),
           Network::max_payload_size(sizeof(BigMessage), nullptr));
    fflush(stdout);

    if(num_nodes < 2) {
      // Single rank: there is no remote target, but a multicast whose target set
      //  contains the origin still exercises reassembly and local delivery with no
      //  network involved at all.
      NodeSet self;
      self.add(Network::my_node_id);
      test_multicast_chunked(self, 1);
    } else {
      const NodeID target = 1;
      test_unicast_chunked(target);

      NodeSet targets;
      int num_targets = 0;
      for(NodeID n = 1; n < num_nodes; n++) {
        targets.add(n);
        num_targets++;
      }
      test_multicast_chunked(targets, num_targets);

      test_concurrent_chunked_types(target);
      test_chunked_completions(target);
      test_large_instance_metadata(p, target);
    }

    printf("[actmsg_large_payload] done: acks=%d errors=%d\n", g_acks.load(),
           g_errors.load());
    fflush(stdout);
  }

}; // namespace

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, FETCH_METADATA_TASK,
      CodeDescriptor(fetch_metadata_task), ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);
  int ret = rt.wait_for_shutdown();

  return ((g_errors.load() != 0) ? 1 : ret);
}
