#include "realm/event.h"
// #include "realm/event_impl.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/node_directory.h"
#include "realm/runtime_impl.h"
#include "realm/serialize.h"

#include <cstring>
#include <vector>

using namespace Realm;

struct P2PMB {
  Realm::Event join_done;
  // bool done_fired{false};
};

namespace {
  P2PMB *p2p_backend = nullptr;
};

/* ------------------------------------------------------------------ */
/* Active-message handlers for p2p membership                          */
/* ------------------------------------------------------------------ */

struct JoinRequestMessage : ControlPlaneMessageTag {
  Epoch_t epoch;
  NodeID wanted_id;
  uint32_t ip;
  uint16_t udp_port;
  uint32_t payload_bytes;
  bool lazy_mode{false};

  static void handle_message(NodeID sender, const JoinRequestMessage &msg,
                             const void *data, size_t datalen);
};

struct JoinAcklMessage : ControlPlaneMessageTag {
  Epoch_t epoch;
  NodeID assigned_id;
  NodeID seed_id;
  uint32_t ip;
  uint16_t udp_port;
  int acks;
  uint32_t payload_bytes;

  static void handle_message(NodeID sender, const JoinAcklMessage &msg, const void *data,
                             size_t datalen);
};

// JoinReq ------------------------------------------------------------------

void JoinRequestMessage::handle_message(NodeID sender, const JoinRequestMessage &msg,
                                        const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);

  int acks = -1;

  Epoch_t new_epoch = msg.epoch;
  if(sender == msg.wanted_id) {
    new_epoch = Network::node_directory.bump_epoch(Network::my_node_id);
    acks = Network::node_directory.size();
  }

  bool complete = false;
  if(datalen > 0 && datalen > msg.payload_bytes) {
    Network::node_directory.complete(msg.wanted_id, msg.epoch, data,
                                     datalen - msg.payload_bytes);
    complete = true;
  }

  std::vector<uint8_t> tmp(msg.payload_bytes);
  std::memcpy(tmp.data(),
              static_cast<const uint8_t *>(data) + (datalen - msg.payload_bytes),
              msg.payload_bytes);

  NodeMeta meta;
  meta.epoch = new_epoch;
  meta.ip = msg.ip;
  meta.udp_port = msg.udp_port;
  meta.worker_address.swap(tmp);
  meta.flags = complete;
  Network::node_directory.add_slot(msg.wanted_id, meta);

  Serialization::DynamicBufferSerializer dbs(4096);
  size_t bytes = 0;
  if(!msg.lazy_mode) {
    bool ok = serialize_announcement(dbs, &get_runtime()->nodes[Network::my_node_id],
                                     get_runtime()->machine,
                                     Network::get_network(msg.wanted_id));
    assert(ok);
    bytes = dbs.bytes_used();
  }

  const NodeMeta *self = Network::node_directory.lookup(Network::my_node_id);
  assert(self != nullptr);
  const void *blob_ptr = self->worker_address.data();
  size_t blob_size = self->worker_address.size();
  assert(blob_size > 0);

  if(sender == msg.wanted_id) {
    NodeSet multicast = Network::node_directory.get_members();
    multicast.remove(sender);
    multicast.remove(Network::my_node_id);
    if(!multicast.empty()) {

      ActiveMessage<JoinRequestMessage> am(multicast, datalen);
      am->wanted_id = msg.wanted_id;
      am->ip = msg.ip;
      am->udp_port = msg.udp_port;
      am->epoch = new_epoch;
      am->lazy_mode = msg.lazy_mode;
      am->payload_bytes = msg.payload_bytes;
      am.add_payload(data, datalen);
      am.commit();
    }
  }

  ActiveMessage<JoinAcklMessage> am(msg.wanted_id, bytes + blob_size);
  am->assigned_id = msg.wanted_id;
  am->seed_id = Network::my_node_id;
  am->payload_bytes = blob_size;
  am->ip = self->ip;
  am->udp_port = self->udp_port;
  am->epoch = new_epoch;
  am->acks = acks;

  if(bytes > 0) {
    am.add_payload(dbs.get_buffer(), bytes);
  }

  if(blob_size > 0) {
    am.add_payload(blob_ptr, blob_size);
  }

  am.commit();
}

// JoinAck ------------------------------------------------------------------

void JoinAcklMessage::handle_message(NodeID sender, const JoinAcklMessage &msg,
                                     const void *data, size_t datalen)
{
  assert(sender != Network::my_node_id);
  assert(msg.assigned_id != NodeDirectory::UNKNOWN_NODE_ID);

  bool complete = false;

  if(datalen > 0 && datalen > msg.payload_bytes) {
    assert(msg.seed_id == sender);
    Network::node_directory.complete(msg.seed_id, msg.epoch, data,
                                     datalen - msg.payload_bytes);
    complete = true;
  }

  NodeMeta new_meta;
  new_meta.epoch = msg.epoch;
  new_meta.ip = msg.ip;
  new_meta.udp_port = msg.udp_port;
  new_meta.flags = complete;

  const uint8_t *blob =
      static_cast<const uint8_t *>(data) + (datalen - msg.payload_bytes);
  new_meta.worker_address.assign(blob, blob + msg.payload_bytes);
  Network::node_directory.add_slot(msg.seed_id, new_meta);

  RuntimeImpl *rt = runtime_singleton;

  rt->join_acks++;
  if(msg.acks > 0) {
    rt->join_acks_total = msg.acks;
  }

  if(rt->join_acks == rt->join_acks_total) {
    Network::node_directory.remove_slot(NodeDirectory::UNKNOWN_NODE_ID);
    // GenEventImpl::trigger(p2p_backend->join_done, false);
    AutoLock<> al(rt->join_mutex);
    rt->join_complete = true;
    rt->join_condvar.broadcast();
  }
}

/* rookie → seed */
static realmStatus_t p2p_join(void *st, const realmNodeMeta_t *self, realmEvent_t done,
                              uint64_t *epoch_out)
{
  constexpr NodeID seed = NodeDirectory::UNKNOWN_NODE_ID;
  constexpr bool lazy_mode = true;

  P2PMB *state = static_cast<P2PMB *>(st);
  state->join_done = done;
  // state->done_fired = false;
  RuntimeImpl *rt = runtime_singleton;

  CoreModuleConfig *config =
      dynamic_cast<CoreModuleConfig *>(rt->get_module_config("core"));
  assert(config != nullptr);
  Serialization::DynamicBufferSerializer dbs(4096);

  size_t bytes = 0;
  if(!lazy_mode) {
    bool ok = serialize_announcement(dbs, &get_runtime()->nodes[Network::my_node_id],
                                     get_runtime()->machine, Network::get_network(seed));
    assert(ok);
    bytes = dbs.bytes_used();
  }

  const void *blob_ptr = self->worker;
  size_t blob_size = self->worker_len;
  assert(blob_size > 0);

  ActiveMessage<JoinRequestMessage> am(seed, bytes + blob_size);
  am->wanted_id = self->node_id;
  am->ip = self->ip;
  am->udp_port = self->udp_port;
  am->payload_bytes = blob_size;
  am->epoch = Network::node_directory.cluster_epoch();
  am->lazy_mode = lazy_mode;

  if(bytes > 0) {
    am.add_payload(dbs.get_buffer(), bytes);
  }

  if(blob_size > 0) {
    am.add_payload(blob_ptr, blob_size);
  }

  am.commit();

  if(epoch_out) {
    *epoch_out = Network::node_directory.cluster_epoch();
  }

  return REALM_OK;
}

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
                                             .join_request = p2p_join,
                                             .progress = p2p_progress,
                                             .epoch = p2p_epoch,
                                             .members = p2p_members};*/

static const realmMembershipOps_t p2p_ops = {
    .join_request = p2p_join,
};

realmStatus_t realmCreateP2PMembershipBackend(realmMembership_t *out)
{
  P2PMB *s = new P2PMB();
  p2p_backend = s;
  return realmMembershipCreate(&p2p_ops, s, out);
}

namespace {
  ActiveMessageHandlerReg<JoinRequestMessage> p2p_joinreq_handler;
  ActiveMessageHandlerReg<JoinAcklMessage> p2p_joinack_handler;
}; // namespace
