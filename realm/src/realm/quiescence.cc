/* Copyright 2024 NVIDIA Corporation
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

#include "realm/serialize.h"
#include "realm/activemsg.h"
#include "realm/quiescence.h"
#include "realm/mutex.h"
#include "realm/node_directory.h"
#include <vector>

namespace Realm {

  struct QuiesceReqAM : ControlPlaneMessageTag {
    uint64_t epoch;
    uint64_t round_id;
    NodeID leaving;

    static void handle_message(NodeID sender, const QuiesceReqAM &args, const void *,
                               size_t);
  };

  struct QuiesceRepAM : ControlPlaneMessageTag {
    uint64_t epoch;
    uint64_t round_id;
    uint64_t msg_sent;
    uint64_t msg_recv;
    uint64_t rcomp_sent;
    uint64_t rcomp_recv;
    uint64_t outstanding;

    static void handle_message(NodeID sender, const QuiesceRepAM &args, const void *,
                               size_t);
  };

  struct QuiesceDoneAM : ControlPlaneMessageTag {
    uint64_t epoch;
    uint64_t round_id;
    static void handle_message(NodeID sender, const QuiesceDoneAM &args, const void *,
                               size_t);
  };

  static ActiveMessageHandlerReg<QuiesceReqAM> reg_quiesce_req;
  static ActiveMessageHandlerReg<QuiesceRepAM> reg_quiesce_rep;
  static ActiveMessageHandlerReg<QuiesceDoneAM> reg_quiesce_done;

  struct QuiesceRoundState {
    Mutex mtx;
    NodeSet peers;
    uint64_t epoch;
    std::vector<char> responded;
    uint64_t round_id{0};
    uint64_t sum_msg_sent{0};
    uint64_t sum_msg_recv{0};
    uint64_t sum_rcomp_sent{0};
    uint64_t sum_rcomp_recv{0};
    uint64_t sum_outstanding{0};

    void reset(uint64_t _epoch, uint64_t _round_id, const NodeSet &_peers)
    {
      epoch = _epoch;
      round_id = _round_id;
      peers = _peers;
      responded.assign(peers.size(), 0);
      sum_msg_sent = sum_msg_recv = sum_rcomp_sent = sum_rcomp_recv = sum_outstanding = 0;
    }

    bool mark(NodeID node, const QuiesceRepAM &rep)
    {
      for(NodeID peer : peers) {
        if(node == peer && !responded[peer]) {
          sum_msg_sent += rep.msg_sent;
          sum_msg_recv += rep.msg_recv;
          sum_rcomp_sent += rep.rcomp_sent;
          sum_rcomp_recv += rep.rcomp_recv;
          sum_outstanding += rep.outstanding;
          responded[node] = 1;
          return true;
        }
      }
      return false;
    }

    bool complete() const
    {
      for(NodeID node : peers) {
        if(!responded[node]) {
          return false;
        }
      }
      return true;
    }
  };

  namespace {
    QuiesceRoundState quiesce_state;

    std::atomic<uint64_t> g_next_round{0};
    std::atomic<uint64_t> g_done_epoch{0};
    std::atomic<uint64_t> g_done_round{0};

    NodeDirectory *s_node_dir;
    NetworkModule *s_network;
  } // namespace

  void QuiesceReqAM::handle_message(NodeID sender, const QuiesceReqAM &args, const void *,
                                    size_t)
  {
    if(args.epoch != s_node_dir->cluster_epoch()) {
      return;
    }

    QuiescenceCounters c;
    s_network->collect_quiescence_counters(args.leaving, c);

    ActiveMessage<QuiesceRepAM> am(args.leaving);
    am->epoch = args.epoch;
    am->round_id = args.round_id;
    am->msg_sent = c.msg_sent;
    am->msg_recv = c.msg_recv;
    am->rcomp_sent = c.rcomp_sent;
    am->rcomp_recv = c.rcomp_recv;
    am->outstanding = c.outstanding;
    am.commit();
  }

  void QuiesceRepAM::handle_message(NodeID sender, const QuiesceRepAM &args, const void *,
                                    size_t)
  {
    AutoLock<> al(quiesce_state.mtx);
    if(args.epoch != quiesce_state.epoch || (args.round_id != quiesce_state.round_id)) {
      return;
    }

    quiesce_state.mark(sender, args);
  }

  void QuiesceDoneAM::handle_message(NodeID sender, const QuiesceDoneAM &args,
                                     const void *, size_t)
  {
    g_done_epoch.store(args.epoch, std::memory_order_release);
    g_done_round.store(args.round_id, std::memory_order_release);
  }

  void quiescence_init(NetworkModule *net, NodeDirectory *ndir)
  {
    s_network = net;
    s_node_dir = ndir;
  }

  bool quiescence_exec(NodeID node)
  {
    NodeSet members = s_node_dir->get_members();
    if(members.empty()) {
      return true;
    }

    const NodeID leaving = node;
    const uint64_t epoch = s_node_dir->cluster_epoch();

    assert(!members.contains(leaving));

    {
      AutoLock<> al(quiesce_state.mtx);
      if(quiesce_state.round_id == 0) {

        QuiescenceCounters local_counters;
        if(s_network) {
          s_network->collect_quiescence_counters(leaving, local_counters);
        }

        const uint64_t new_round =
            g_next_round.fetch_add(1, std::memory_order_relaxed) + 1;
        quiesce_state.reset(epoch, new_round, members);

        quiesce_state.sum_outstanding = local_counters.outstanding;

        ActiveMessage<QuiesceReqAM> am(members);
        am->leaving = leaving;
        am->epoch = quiesce_state.epoch;
        am->round_id = quiesce_state.round_id;
        am.commit();
        return false;
      }

      if(epoch != quiesce_state.epoch) {
        quiesce_state.round_id = 0;
        return false;
      }

      if(!quiesce_state.complete()) {
        return false;
      }

      const bool done = (quiesce_state.sum_outstanding == 0);

      // const bool msgs_ok = (quiesce_state.sum_msg_sent == quiesce_state.sum_msg_recv);
      // const bool rcmp_ok = (quiesce_state.sum_rcomp_sent ==
      // quiesce_state.sum_rcomp_recv); const bool outs_ok =
      // (quiesce_state.sum_outstanding == 0); const bool done = msgs_ok && rcmp_ok &&
      // outs_ok;

      if(done) {
        ActiveMessage<QuiesceDoneAM> am(members);
        am->epoch = quiesce_state.epoch;
        am->round_id = quiesce_state.round_id;
        am.commit();
        g_done_epoch.store(quiesce_state.epoch, std::memory_order_release);
        g_done_round.store(quiesce_state.round_id, std::memory_order_release);
      }

      quiesce_state.round_id = 0;
      return done;
    }
    return false;
  }
} // namespace Realm
