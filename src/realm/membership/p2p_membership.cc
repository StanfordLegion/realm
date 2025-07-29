#include "realm/node_directory.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/serialize.h"
#include "realm/activemsg.h"

#include <cstring>

using namespace Realm;

struct MembershipP2P {
  int join_acks{0};
  int join_acks_total{0};

  // Realm::Event sub_done;
  NodeSet subscribers;
  // Mutex sub_mutex;

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

// JoinReq ------------------------------------------------------------------
void JoinRequestMessage::handle_message(NodeID sender, const JoinRequestMessage &msg,
                                        const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);

  Network::node_directory.import_node(data, datalen, /*epoch=*/0);
  uint64_t new_epoch = Network::node_directory.cluster_epoch();

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

  send_join_ack(sender, new_epoch, membership_state->subscribers.size() + 1,
                msg.announce_mm);

  if(msg.subscribe) {
    assert(Network::my_node_id != sender);
    membership_state->subscribers.add(sender);
  }
}

// JoinAck ------------------------------------------------------------------

void JoinAcklMessage::handle_message(NodeID, const JoinAcklMessage &msg, const void *data,
                                     size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);

  membership_state->join_acks++;

  if(msg.acks > 0) {
    membership_state->join_acks_total = msg.acks;
  }

  if(membership_state->join_acks == membership_state->join_acks_total) {
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
  Network::node_directory.import_node(data, datalen, msg.epoch);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.announce_mm);

  if(msg.subscribe) {
    membership_state->subscribers.add(msg.node_id);
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

static realmStatus_t p2p_progress(void *st)
{
  return REALM_OK;
}

}*/

/* ---------- v-table instance -------------------------------- */
/*static const realmMembershipOps_t p2p_ops = {.destroy = p2p_destroy,
                                             .join_request = join,
                                             .progress = p2p_progress,
                                             .epoch = p2p_epoch,
                                             .members = p2p_members};*/

namespace {
  realmStatus_t join(void *st, const realmNodeMeta_t *self, realmMembershipHooks_t hooks)
  {
    AmProvider *am_provider = new AmProvider();

    Network::node_directory.set_provider(am_provider);

    MembershipP2P *state = static_cast<MembershipP2P *>(st);
    state->hooks = hooks;
    // state->am_provider = am_provider;

    bool announce_mm = self->announce_mm;
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(self->node_id, announce_mm, dbs);

    if(hooks.pre_join) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      hooks.pre_join(&meta, nullptr, 0, /*joined=*/false, hooks.user_arg);
    }

    if(Network::my_node_id == 0) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      hooks.post_join(&meta, nullptr, 0, /*joined=*/true, hooks.user_arg);
      return REALM_OK;
    }

    ActiveMessage<JoinRequestMessage> am(self->seed_id, dbs.bytes_used());
    am->announce_mm = announce_mm;
    am->epoch = Network::node_directory.cluster_epoch();

    // TODO: That's not how we should subscribe
    am->subscribe = (hooks.post_join != nullptr);

    am.add_payload(dbs.get_buffer(), dbs.bytes_used());
    am.commit();

    return REALM_OK;
  }

  const realmMembershipOps_t p2p_ops = {
      .join_request = join,
  };
} // namespace

realmStatus_t realmMembershipP2PInit(realmMembership_t *out)
{
  MembershipP2P *state = new MembershipP2P();
  membership_state = state;
  return realmMembershipCreate(&p2p_ops, state, out);
}
