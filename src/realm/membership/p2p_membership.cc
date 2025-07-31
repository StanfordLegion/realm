#include "realm/node_directory.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/serialize.h"
#include "realm/activemsg.h"

#include <cstring>
#include <atomic>

using namespace Realm;

struct MembershipP2P {
  std::atomic<int> join_acks{0};
  std::atomic<int> join_acks_total{0};

  std::atomic<int> leave_acks{0};
  std::atomic<int> leave_acks_total{0};

  Mutex subs_mutex;
  NodeSet subscribers;

  Mutex pending_mutex;
  NodeSet pending;

  std::atomic<bool> leaving{false};

  realmMembershipHooks_t hooks;
};

namespace {
  MembershipP2P *membership_state = nullptr;
};

namespace {
  constexpr int DBS_SIZE{4096};

  struct DirectoryPutMessage : ControlPlaneMessageTag {
    static void handle_message(NodeID, const DirectoryPutMessage &msg, const void *data,
                               size_t datalen);
  };

  struct DirectoryFetchMessage : ControlPlaneMessageTag {
    NodeID id;
    static void handle_message(NodeID sender, const DirectoryFetchMessage &msg,
                               const void *data, size_t datalen);
  };

  ActiveMessageHandlerReg<DirectoryFetchMessage> diretory_fetch_msg;
  ActiveMessageHandlerReg<DirectoryPutMessage> directory_put_msg;

  /* ------------------------------------------------------------------ */
  /* Active-message handlers for p2p membership                          */
  /* ------------------------------------------------------------------ */

  struct JoinRequestMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    bool announce_mm{false};
    bool subscribe{true};

    static void handle_message(NodeID sender, const JoinRequestMessage &msg,
                               const void *data, size_t datalen);
  };

  struct JoinAcklMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    int acks{-1};
    bool subscribe{true};

    static void handle_message(NodeID sender, const JoinAcklMessage &msg,
                               const void *data, size_t datalen);
  };

  struct SubscribeReqMessage : ControlPlaneMessageTag {
    bool announce_mm;
    static void handle_message(NodeID sender, const SubscribeReqMessage &msg,
                               const void *data, size_t datalen);
  };

  struct SubscribeAckMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    static void handle_message(NodeID sender, const SubscribeAckMessage &msg,
                               const void *data, size_t datalen);
  };

  struct MemberUpdateMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    NodeID node_id;
    bool announce_mm{false};
    bool subscribe{true};

    static void handle_message(NodeID sender, const MemberUpdateMessage &msg,
                               const void *data, size_t datalen);
  };

  struct LeaveReqMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeaveReqMessage &msg,
                               const void *data, size_t datalen);
  };

  struct LeaveAckMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeaveAckMessage &msg,
                               const void *data, size_t datalen);
  };

  struct MemberRemoveMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    NodeID node_id;

    static void handle_message(NodeID /*seed*/, const MemberRemoveMessage &msg,
                               const void *data, size_t datalen);
  };

  inline void send_join_ack(NodeID dest, Epoch_t epoch, int acks, bool announce_mm)
  {
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(Network::my_node_id, /*include_mm=*/announce_mm,
                                        dbs);

    ActiveMessage<JoinAcklMessage> ack(dest, dbs.bytes_used());
    ack->acks = acks;
    ack->epoch = epoch;
    ack.add_payload(dbs.get_buffer(), dbs.bytes_used());
    ack.commit();
  }

  ActiveMessageHandlerReg<JoinRequestMessage> joinreq_handler;
  ActiveMessageHandlerReg<JoinAcklMessage> joinack_handler;
  ActiveMessageHandlerReg<SubscribeReqMessage> p2p_subreq_handler;
  ActiveMessageHandlerReg<SubscribeAckMessage> p2p_suback_handler;
  ActiveMessageHandlerReg<MemberUpdateMessage> p2p_member_handler;
  ActiveMessageHandlerReg<LeaveReqMessage> leavereq_handler;
  ActiveMessageHandlerReg<LeaveAckMessage> leaveack_handler;
  ActiveMessageHandlerReg<MemberRemoveMessage> member_remove_handler;
} // namespace

class AmProvider : public NodeDirectory::Provider {
public:
  void put(NodeSet peers, const void *blob, size_t bytes, uint64_t) override
  {
    ActiveMessage<DirectoryPutMessage> m(peers, bytes);
    m.add_payload(blob, bytes);
    m.commit();
  }

  void fetch(NodeID id) override
  {
    ActiveMessage<DirectoryFetchMessage> request(id);
    request->id = id;
    request.commit();
  }
};

void DirectoryPutMessage::handle_message(NodeID, const DirectoryPutMessage &,
                                         const void *data, size_t datalen)
{
  Network::node_directory.import_node(data, datalen,
                                      Network::node_directory.cluster_epoch());
}

void DirectoryFetchMessage::handle_message(NodeID sender,
                                           const DirectoryFetchMessage &msg, const void *,
                                           size_t)
{
  assert(msg.id == Network::my_node_id);

  Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
  Network::node_directory.export_node(msg.id, true, dbs);

  ActiveMessage<DirectoryPutMessage> rep(sender, dbs.bytes_used());
  rep.add_payload(dbs.get_buffer(), dbs.bytes_used());
  rep.commit();
}

void JoinRequestMessage::handle_message(NodeID sender, const JoinRequestMessage &msg,
                                        const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);
  assert(membership_state->leaving.load(std::memory_order_acquire) == false);

  if(membership_state->hooks.pre_join) {
    realmNodeMeta_t meta{(int32_t)sender, 0, false};
    membership_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                                     membership_state->hooks.user_arg);
  }

  Network::node_directory.import_node(data, datalen, /*epoch=*/0);
  uint64_t new_epoch = Network::node_directory.cluster_epoch();

  {
    AutoLock<> al(membership_state->subs_mutex);
    if(!membership_state->subscribers.empty()) {
      assert(membership_state->subscribers.contains(Network::my_node_id) == false);
      assert(membership_state->subscribers.contains(sender) == false);
      ActiveMessage<MemberUpdateMessage> am(membership_state->subscribers, datalen);
      am->node_id = sender;
      am->epoch = new_epoch;
      am->announce_mm = msg.announce_mm;
      am->subscribe = msg.subscribe;
      am.add_payload(data, datalen);
      am.commit();
    }
  }

  // TODO: FIX ME - what if ACK is leaving as well? That will deadlock.
  send_join_ack(sender, new_epoch, membership_state->subscribers.size() + 1,
                msg.announce_mm);

  if(msg.subscribe) {
    AutoLock<> al(membership_state->subs_mutex);
    membership_state->subscribers.add(sender);
  }
}

void JoinAcklMessage::handle_message(NodeID sender, const JoinAcklMessage &msg,
                                     const void *data, size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);

  membership_state->join_acks.fetch_add(1, std::memory_order_relaxed);

  if(msg.acks > 0) {
    membership_state->join_acks_total.store(msg.acks, std::memory_order_relaxed);
  }

  if(msg.subscribe) {
    AutoLock<> al(membership_state->subs_mutex);
    membership_state->subscribers.add(sender);
  }

  if(membership_state->join_acks.load() == membership_state->join_acks_total.load()) {
    if(membership_state->hooks.post_join) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, /*announce_mm=*/false};
      membership_state->hooks.post_join(&meta, nullptr, 0, /*joined=*/true,
                                        membership_state->hooks.user_arg);
    }
  }
}

void MemberUpdateMessage::handle_message(NodeID, const MemberUpdateMessage &msg,
                                         const void *data, size_t datalen)
{
  if(msg.epoch <= Network::node_directory.cluster_epoch()) {
    assert(0);
    // send_rejeck_ack();
    return;
  }

  assert(membership_state->leaving.load(std::memory_order_acquire) == false);

  if(membership_state->hooks.pre_join) {
    realmNodeMeta_t meta{(int32_t)msg.node_id, 0, false};
    membership_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                                     membership_state->hooks.user_arg);
  }

  Network::node_directory.import_node(data, datalen, msg.epoch);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.announce_mm);

  if(msg.subscribe) {
    AutoLock<> al(membership_state->subs_mutex);
    membership_state->subscribers.add(msg.node_id);
  }
}

void MemberRemoveMessage::handle_message(NodeID, const MemberRemoveMessage &,
                                         const void *, size_t)
{}

void LeaveReqMessage::handle_message(NodeID sender, const LeaveReqMessage &msg,
                                     const void *, size_t)
{
  assert(sender != Network::my_node_id);
  Network::node_directory.update_epoch(msg.epoch);

  // TODO: thread-safety semantics here needs more thinking

  if(membership_state->leaving.load(std::memory_order_acquire)) {
    AutoLock<> al(membership_state->pending_mutex);
    membership_state->pending.remove(sender);
  } else {
    AutoLock<> al(membership_state->subs_mutex);
    membership_state->subscribers.remove(sender);
  }

  ActiveMessage<LeaveAckMessage> am(sender);
  am.commit();

  if(membership_state->hooks.pre_leave) {
    realmNodeMeta_t meta{(int32_t)sender, 0, false};
    membership_state->hooks.pre_leave(&meta, nullptr, 0, /*left=*/false,
                                      membership_state->hooks.user_arg);
  }

  Network::node_directory.remove_slot(sender);
}

void LeaveAckMessage::handle_message(NodeID /*sender*/ sender, const LeaveAckMessage &,
                                     const void *, size_t)
{
  {
    AutoLock<> al(membership_state->pending_mutex);
    membership_state->pending.remove(sender);
  }

  {
    AutoLock<> al(membership_state->pending_mutex);
    if(membership_state->pending.empty()) {
      assert(membership_state->hooks.post_leave);
      if(membership_state->hooks.post_leave) {
        realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, false};
        membership_state->hooks.post_leave(&meta, nullptr, 0, true,
                                           membership_state->hooks.user_arg);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* SUBSCRIBE request handler (SEED)                                   */
/* ------------------------------------------------------------------ */
void SubscribeReqMessage::handle_message(NodeID sender, const SubscribeReqMessage &msg,
                                         const void *, size_t)
{}

/* ------------------------------------------------------------------ */
/* SUBSCRIBE ACK handler (rookie)                                     */
/* ------------------------------------------------------------------ */
void SubscribeAckMessage::handle_message(NodeID, const SubscribeAckMessage &msg,
                                         const void *data, size_t datalen)
{}

/*static realmStatus_t p2p_subscribe(void *st, realmEvent_t done, bool announce_mm)
{
  return REALM_OK;
}*/

/*static realmStatus_t p2p_destroy(void *s)
{
  delete static_cast<MembershipP2P *>(s);
  return REALM_OK;
}

}*/

namespace {
  realmStatus_t join(void *st, const realmNodeMeta_t *self, realmMembershipHooks_t hooks)
  {
    MembershipP2P *state = static_cast<MembershipP2P *>(st);
    state->hooks = hooks;

    AmProvider *am_provider = new AmProvider();
    Network::node_directory.set_provider(am_provider);
    // state->am_provider = am_provider;

    bool announce_mm = self->announce_mm;
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(self->node_id, announce_mm, dbs);

    if(hooks.pre_join) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      hooks.pre_join(&meta, nullptr, 0, /*joined=*/false, hooks.user_arg);
    }

    if(self->seed_id == NodeDirectory::INVALID_NODE_ID) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      hooks.post_join(&meta, nullptr, 0, /*joined=*/true, hooks.user_arg);
    } else {

      ActiveMessage<JoinRequestMessage> am(self->seed_id, dbs.bytes_used());
      am->epoch = Network::node_directory.cluster_epoch();
      // TODO: That's not how we should subscribe
      am->subscribe = (hooks.post_join != nullptr);
      am->announce_mm = announce_mm;
      am.add_payload(dbs.get_buffer(), dbs.bytes_used());
      am.commit();
    }
    return REALM_OK;
  }

  // TODO: FIX ALL RACE CONDITIONS
  realmStatus_t leave(void *st, const realmNodeMeta_t *self, realmMembershipHooks_t hooks)
  {
    MembershipP2P *state = static_cast<MembershipP2P *>(st);

    state->hooks = hooks;
    state->leave_acks = 0;

    if(hooks.pre_leave) {
      hooks.pre_leave(self, nullptr, 0, /*left=*/false, hooks.user_arg);
    }

    NodeSet members;

    {
      AutoLock<> al(state->pending_mutex);
      assert(state->pending.empty());
      state->pending = state->subscribers;
    }

    state->leaving.store(true, std::memory_order_release);

    {
      AutoLock<> al(state->pending_mutex);
      if(!state->pending.empty()) {
        ActiveMessage<LeaveReqMessage> am(state->pending);
        am->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        am.commit();
      } else {
        if(hooks.post_leave) {
          hooks.post_leave(self, nullptr, 0, /*left=*/true, hooks.user_arg);
        }
      }
    }

    return REALM_OK;
  }

  const realmMembershipOps_t p2p_ops = {
      .join_request = join,
      .leave_request = leave,
  };
} // namespace

realmStatus_t realmMembershipP2PInit(realmMembership_t *out)
{
  MembershipP2P *state = new MembershipP2P();
  membership_state = state;
  return realmMembershipCreate(&p2p_ops, state, out);
}
