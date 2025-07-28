#include "realm/event.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/node_directory.h"
#include <cstring>
#include "realm/serialize.h"
#include "realm/activemsg.h"
#include "realm/event_impl.h"

#include "realm/runtime_impl.h" // TODO: REMOVE

using namespace Realm;

struct P2PMB {
  Realm::Event join_done;
  int join_acks{0};
  int join_acks_total{0};

  // Realm::Event sub_done;
  NodeSet subscribers;
  // Mutex sub_mutex;

  realmMembershipChangeCB_fn cb_fn{nullptr};
  void *cb_arg{nullptr};
};

namespace {
  P2PMB *p2p_state = nullptr;
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
    // NodeID wanted_id;
    bool lazy_mode{false};
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
    bool lazy_mode;
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
    bool lazy_mode{false};

    static void handle_message(NodeID sender, const MemberUpdateMessage &msg,
                               const void *data, size_t datalen);
  };

  inline void send_join_ack(NodeID dest, Epoch_t epoch, int acks, bool lazy_mode)
  {
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(Network::my_node_id, /*include_mm=*/!lazy_mode,
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

using namespace Realm;

// JoinReq ------------------------------------------------------------------

struct MemberInfo {
  uint64_t epoch{0};
  uint32_t ip{0};
  uint16_t udp_port{0};
  size_t mm_size{0};
};

void JoinRequestMessage::handle_message(NodeID sender, const JoinRequestMessage &msg,
                                        const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);

  Network::node_directory.import_node(data, datalen, /*epoch=*/0);
  uint64_t new_epoch = Network::node_directory.cluster_epoch();

  if(!p2p_state->subscribers.empty()) {
    assert(p2p_state->subscribers.contains(Network::my_node_id) == false);
    assert(p2p_state->subscribers.contains(sender) == false);
    ActiveMessage<MemberUpdateMessage> am(p2p_state->subscribers, datalen);
    am->node_id = sender;
    am->epoch = new_epoch;
    am->lazy_mode = msg.lazy_mode;
    am.add_payload(data, datalen);
    am.commit();
  }

  send_join_ack(sender, new_epoch, p2p_state->subscribers.size() + 1, msg.lazy_mode);

  if(msg.subscribe) {
    assert(Network::my_node_id != sender);
    p2p_state->subscribers.add(sender);
  }
}

// JoinAck ------------------------------------------------------------------

void JoinAcklMessage::handle_message(NodeID, const JoinAcklMessage &msg, const void *data,
                                     size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);

  p2p_state->join_acks++;

  if(msg.acks > 0) {
    p2p_state->join_acks_total = msg.acks;
  }

  if(p2p_state->join_acks == p2p_state->join_acks_total) {
    Network::node_directory.remove_slot(NodeDirectory::UNKNOWN_NODE_ID);
    GenEventImpl::trigger(p2p_state->join_done, false);
    // AutoLock<> al(rt->join_mutex);
    // rt->join_complete = true;
    // rt->join_condvar.broadcast();
  }
}

void MemberUpdateMessage::handle_message(NodeID, const MemberUpdateMessage &msg,
                                         const void *data, size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.lazy_mode);
}

/* ------------------------------------------------------------------ */
/* SUBSCRIBE request handler (SEED)                                   */
/* ------------------------------------------------------------------ */
void SubscribeReqMessage::handle_message(NodeID sender, const SubscribeReqMessage &msg,
                                         const void *, size_t)
{
  /*{
    AutoLock<> al(p2p_state->sub_mutex);
    p2p_state->subscribers.add(sender);
  }

  Serialization::DynamicBufferSerializer dbs(4096);

  if(!msg.lazy_mode) {
    // get_mm(Network::my_node_id, dbs);
  }

  ActiveMessage<SubscribeAckMessage> ack(sender, dbs.bytes_used());
  ack->epoch = Network::node_directory.cluster_epoch();

  if(dbs.bytes_used() > 0) {
    ack.add_payload(dbs.get_buffer(), dbs.bytes_used());
  }

  ack.commit();
  */
}

/* ------------------------------------------------------------------ */
/* SUBSCRIBE ACK handler (rookie)                                     */
/* ------------------------------------------------------------------ */
void SubscribeAckMessage::handle_message(NodeID, const SubscribeAckMessage &msg,
                                         const void *data, size_t datalen)
{
  /*if(datalen > 0) {
    Network::node_directory.complete(Network::my_node_id, msg.epoch, data, datalen);
  }

  if(p2p_state->sub_done.exists()) {
    GenEventImpl::trigger(p2p_state->sub_done, false);
  }*/
}

/*static realmStatus_t p2p_subscribe(void *st, realmEvent_t done, bool lazy_mode)
{
  P2PMB *state = static_cast<P2PMB *>(st);
  state->sub_done = done;

  ActiveMessage<SubscribeReqMessage> sr(NodeDirectory::UNKNOWN_NODE_ID);
  sr->lazy_mode = lazy_mode;
  sr.commit();

  if(Network::my_node_id == NodeDirectory::UNKNOWN_NODE_ID && done) {
    GenEventImpl::trigger(done, false);
  }

  return REALM_OK;
}*/

/*static realmStatus_t p2p_destroy(void *s)
{
  delete static_cast<P2PMB *>(s);
  return REALM_OK;
}

static realmStatus_t p2p_progress(void *st)
{
  auto *state = static_cast<P2PMB *>(st);

  Network::get_module()->poll();

  if(state->join_done && !state->done_fired && runtime_singleton->join_complete) {
    GenEventImpl::trigger(*(state->join_done), false);

    state->done_fired = true;
  }

  return REALM_OK;
}

static realmStatus_t p2p_epoch(void *, uint64_t *e)
{
  *e = Network::node_directory.cluster_epoch();
  return REALM_OK;
}

static realmStatus_t p2p_members(void *, realmNodeMeta_t *buf, size_t *cnt)
{
  NodeSet ns = Network::node_directory.get_members(true);
  if(*cnt < ns.size())
    return REALM_ERR_BAD_ARG;
  size_t i = 0;
  for(NodeID id : ns) {
    const NodeMeta *m = Network::node_directory.lookup(id);
    buf[i++] = *m;
  }
  *cnt = i;
  return REALM_OK;
}*/

/* ---------- v-table instance -------------------------------- */
/*static const realmMembershipOps_t p2p_ops = {.destroy = p2p_destroy,
                                             .join_request = join,
                                             .progress = p2p_progress,
                                             .epoch = p2p_epoch,
                                             .members = p2p_members};*/

namespace {
  realmStatus_t join(void *st, const realmNodeMeta_t *self, realmEvent_t done,
                     uint64_t *epoch_out, bool lazy_mode,
                     realmMembershipChangeCB_fn cb_fn, void *cb_arg)
  {
    AmProvider *am_provider = new AmProvider();
    Network::node_directory.set_provider(am_provider);

    P2PMB *state = static_cast<P2PMB *>(st);
    state->join_done = done;
    state->cb_fn = cb_fn;
    state->cb_arg = cb_arg;
    // state->am_provider = am_provider;

    if(Network::my_node_id == 0) {
      GenEventImpl::trigger(done, false);
      return REALM_OK;
    }

    // assert(self->worker_len > 0);

    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(self->node_id, !lazy_mode, dbs);

    ActiveMessage<JoinRequestMessage> am(self->seed_id, dbs.bytes_used());
    am->epoch = Network::node_directory.cluster_epoch();
    am->lazy_mode = lazy_mode;
    am.add_payload(dbs.get_buffer(), dbs.bytes_used());
    am.commit();

    if(epoch_out) {
      *epoch_out = Network::node_directory.cluster_epoch();
    }

    return REALM_OK;
  }

  const realmMembershipOps_t p2p_ops = {
      .join_request = join,
  };
} // namespace

realmStatus_t realmMembershipP2PInit(realmMembership_t *out)
{
  P2PMB *state = new P2PMB();
  p2p_state = state;
  return realmMembershipCreate(&p2p_ops, state, out);
}
