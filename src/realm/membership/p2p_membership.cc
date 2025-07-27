#include "realm/event.h"
// #include "realm/event_impl.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/node_directory.h"
#include "realm/runtime_impl.h"
#include <cstring>
#include <vector>

using namespace Realm;

struct P2PMB {
  Realm::Event join_done;
  int join_acks{0};
  int join_acks_total{0};

  Realm::Event sub_done;
  NodeSet subscribers;
  // Mutex sub_mutex;

  realmMembershipChangeCB_fn cb_fn{nullptr};
  void *cb_arg{nullptr};
};

namespace {
  P2PMB *p2p_state = nullptr;
};

/* ------------------------------------------------------------------ */
/* Active-message handlers for p2p membership                          */
/* ------------------------------------------------------------------ */

struct JoinRequestMessage : ControlPlaneMessageTag {
  Epoch_t epoch;
  NodeID wanted_id;
  uint32_t ip;
  uint16_t udp_port;
  uint32_t worker_len;
  bool lazy_mode{false};
  bool subscribe{true};

  static void handle_message(NodeID sender, const JoinRequestMessage &msg,
                             const void *data, size_t datalen);
};

struct JoinAcklMessage : ControlPlaneMessageTag {
  Epoch_t epoch;
  uint32_t ip;
  uint16_t udp_port;
  int acks{-1};
  uint32_t worker_len;

  static void handle_message(NodeID sender, const JoinAcklMessage &msg, const void *data,
                             size_t datalen);
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
  // bool is_join{true};
  NodeID node_id;
  uint32_t ip;
  uint16_t udp_port;
  uint32_t worker_len;
  bool lazy_mode{false};

  static void handle_message(NodeID sender, const MemberUpdateMessage &msg,
                             const void *data, size_t datalen);
};

static inline void send_join_ack(NodeID dest, Epoch_t epoch, int acks, bool lazy_mode)
{
  const NodeMeta *self = Network::node_directory.lookup(Network::my_node_id);
  assert(self != nullptr);

  const size_t worker_len = self->worker_address.size();
  const size_t mm_len = lazy_mode ? 0 : self->machine_model.size();

  ActiveMessage<JoinAcklMessage> ack(dest, mm_len + worker_len);
  ack->worker_len = worker_len;
  ack->ip = self->ip;
  ack->udp_port = self->udp_port;
  ack->epoch = epoch;
  ack->acks = acks;

  if(mm_len > 0) {
    ack.add_payload(self->machine_model.data(), mm_len);
  }
  ack.add_payload(self->worker_address.data(), worker_len);
  ack.commit();
}

// JoinReq ------------------------------------------------------------------

struct MemberInfo {
  uint64_t epoch{0};
  uint32_t ip{0};
  uint16_t udp_port{0};
  size_t mm_size{0};
};

void put_mm(NodeID me, MemberInfo minfo, const void *data, size_t datalen)
{
  if(minfo.mm_size > 0) {
    Network::node_directory.complete(me, minfo.epoch, data, minfo.mm_size);
  }

  const size_t addr_len = datalen - minfo.mm_size;
  std::vector<uint8_t> tmp(addr_len);
  std::memcpy(tmp.data(), static_cast<const uint8_t *>(data) + addr_len, addr_len);

  NodeMeta meta;
  meta.epoch = minfo.epoch;
  meta.ip = minfo.ip;
  meta.udp_port = minfo.udp_port;
  meta.worker_address.swap(tmp);
  meta.flags = (minfo.mm_size > 0);
  Network::node_directory.add_slot(me, meta);
}

void JoinRequestMessage::handle_message(NodeID sender, const JoinRequestMessage &msg,
                                        const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);

  Epoch_t new_epoch = Network::node_directory.bump_epoch(Network::my_node_id);
  put_mm(msg.wanted_id, {new_epoch, msg.ip, msg.udp_port, (datalen - msg.worker_len)},
         data, datalen);

  if(!p2p_state->subscribers.empty()) {
    ActiveMessage<MemberUpdateMessage> am(p2p_state->subscribers, datalen);
    am->node_id = msg.wanted_id;
    am->ip = msg.ip;
    am->udp_port = msg.udp_port;
    am->epoch = new_epoch;
    am->lazy_mode = msg.lazy_mode;
    am->worker_len = msg.worker_len;
    am.add_payload(data, datalen);
    am.commit();
  }

  send_join_ack(msg.wanted_id, new_epoch, p2p_state->subscribers.size() + 1,
                msg.lazy_mode);

  if(msg.subscribe) {
    p2p_state->subscribers.add(msg.wanted_id);
  }
}

// JoinAck ------------------------------------------------------------------

void JoinAcklMessage::handle_message(NodeID sender, const JoinAcklMessage &msg,
                                     const void *data, size_t datalen)
{
  put_mm(sender, {msg.epoch, msg.ip, msg.udp_port, (datalen - msg.worker_len)}, data,
         datalen);

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

void MemberUpdateMessage::handle_message(NodeID sender, const MemberUpdateMessage &msg,
                                         const void *data, size_t datalen)
{
  put_mm(msg.node_id, {msg.epoch, msg.ip, msg.udp_port, (datalen - msg.worker_len)}, data,
         datalen);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.lazy_mode);
}

static realmStatus_t join(void *st, const realmNodeMeta_t *self, realmEvent_t done,
                          uint64_t *epoch_out, bool lazy_mode,
                          realmMembershipChangeCB_fn cb_fn, void *cb_arg)
{
  constexpr NodeID seed = NodeDirectory::UNKNOWN_NODE_ID;

  P2PMB *state = static_cast<P2PMB *>(st);
  state->join_done = done;
  state->cb_fn = cb_fn;
  state->cb_arg = cb_arg;

  if(Network::my_node_id == 0) {
    GenEventImpl::trigger(done, false);
    return REALM_OK;
  }

  assert(self->worker_len > 0);

  ActiveMessage<JoinRequestMessage> am(seed, self->mm_len + self->worker_len);
  am->wanted_id = self->node_id;
  am->ip = self->ip;
  am->udp_port = self->udp_port;
  am->worker_len = self->worker_len;
  am->epoch = Network::node_directory.cluster_epoch();
  am->lazy_mode = lazy_mode;

  if(!lazy_mode && self->mm_len > 0) {
    assert(self->mm != nullptr);
    am.add_payload(self->mm, self->mm_len);
  }

  am.add_payload(self->worker, self->worker_len);
  am.commit();

  if(epoch_out) {
    *epoch_out = Network::node_directory.cluster_epoch();
  }

  return REALM_OK;
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

static const realmMembershipOps_t p2p_ops = {
    .join_request = join,
    // subscribe_request can be added later when implemented fully
};

realmStatus_t realmCreateP2PMembershipBackend(realmMembership_t *out)
{
  P2PMB *s = new P2PMB();
  p2p_state = s;
  return realmMembershipCreate(&p2p_ops, s, out);
}

namespace {
  ActiveMessageHandlerReg<JoinRequestMessage> joinreq_handler;
  ActiveMessageHandlerReg<JoinAcklMessage> joinack_handler;
  ActiveMessageHandlerReg<SubscribeReqMessage> p2p_subreq_handler;
  ActiveMessageHandlerReg<SubscribeAckMessage> p2p_suback_handler;
  ActiveMessageHandlerReg<MemberUpdateMessage> p2p_member_handler;
}; // namespace
