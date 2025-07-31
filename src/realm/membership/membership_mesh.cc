#include "realm/node_directory.h"
#include "realm/membership/membership.h"
#include "realm/network.h"
#include "realm/serialize.h"
#include "realm/activemsg.h"

#include <cstring>
#include <atomic>

using namespace Realm;

struct MembershipMesh {
  std::atomic<int> join_acks{0};
  std::atomic<int> join_acks_total{0};

  Mutex subs_mutex;
  NodeSet subscribers;

  Mutex pending_mutex;
  NodeSet pending;

  std::atomic<bool> leaving{false};

  realmMembershipHooks_t hooks;
};

namespace {
  MembershipMesh *mesh_state = nullptr;
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
  /* Active-message handlers for membership                          */
  /* ------------------------------------------------------------------ */

  struct JoinRequestMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    bool announce_mm{false};

    static void handle_message(NodeID sender, const JoinRequestMessage &msg,
                               const void *data, size_t datalen);
  };

  struct JoinAcklMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    int acks{-1};

    static void handle_message(NodeID sender, const JoinAcklMessage &msg,
                               const void *data, size_t datalen);
  };

  struct MemberUpdateMessage : ControlPlaneMessageTag {
    Epoch_t epoch;
    NodeID node_id;
    bool announce_mm{false};

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
  ActiveMessageHandlerReg<MemberUpdateMessage> p2p_member_handler;
  ActiveMessageHandlerReg<LeaveReqMessage> leavereq_handler;
  ActiveMessageHandlerReg<LeaveAckMessage> leaveack_handler;
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
  assert(mesh_state->leaving.load(std::memory_order_acquire) == false);

  if(mesh_state->hooks.pre_join) {
    realmNodeMeta_t meta{(int32_t)sender, 0, false};
    mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                               mesh_state->hooks.user_arg);
  }

  Network::node_directory.import_node(data, datalen, /*epoch=*/0);
  uint64_t new_epoch = Network::node_directory.cluster_epoch();

  {
    AutoLock<> al(mesh_state->subs_mutex);
    if(!mesh_state->subscribers.empty()) {
      assert(mesh_state->subscribers.contains(Network::my_node_id) == false);
      assert(mesh_state->subscribers.contains(sender) == false);
      ActiveMessage<MemberUpdateMessage> am(mesh_state->subscribers, datalen);
      am->node_id = sender;
      am->epoch = new_epoch;
      am->announce_mm = msg.announce_mm;
      am.add_payload(data, datalen);
      am.commit();
    }
  }

  // TODO: FIX ME - what if ACK is leaving as well? That will deadlock.
  send_join_ack(sender, new_epoch, mesh_state->subscribers.size() + 1, msg.announce_mm);

  if(mesh_state->hooks.filter) {
    realmNodeMeta_t meta{(int32_t)sender, 0, msg.announce_mm};
    if(mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      AutoLock<> al(mesh_state->subs_mutex);
      mesh_state->subscribers.add(sender);
    }
  }
}

void JoinAcklMessage::handle_message(NodeID sender, const JoinAcklMessage &msg,
                                     const void *data, size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);

  mesh_state->join_acks.fetch_add(1, std::memory_order_relaxed);

  if(msg.acks > 0) {
    mesh_state->join_acks_total.store(msg.acks, std::memory_order_relaxed);
  }

  if(mesh_state->hooks.filter) {
    realmNodeMeta_t meta{(int32_t)sender, 0, false};
    if(mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      AutoLock<> al(mesh_state->subs_mutex);
      mesh_state->subscribers.add(sender);
    }
  }

  if(mesh_state->join_acks.load() == mesh_state->join_acks_total.load()) {
    if(mesh_state->hooks.post_join) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, /*announce_mm=*/false};
      mesh_state->hooks.post_join(&meta, nullptr, 0, /*joined=*/true,
                                  mesh_state->hooks.user_arg);
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

  Network::node_directory.import_node(data, datalen, msg.epoch);

  assert(mesh_state->leaving.load(std::memory_order_acquire) == false);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.announce_mm);

  if(mesh_state->hooks.filter) {
    realmNodeMeta_t meta{(int32_t)msg.node_id, 0, msg.announce_mm};
    if(!mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      // TODO: we need import_node to send the ack back at the same time we
      // don't need mm to be installed in case it has been and thus it needs
      // to be removed as we ll.
      Network::node_directory.remove_slot(msg.node_id);
      return;
    }
  }

  if(mesh_state->hooks.pre_join) {
    realmNodeMeta_t meta{(int32_t)msg.node_id, 0, false};
    mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                               mesh_state->hooks.user_arg);
  }

  {
    AutoLock<> al(mesh_state->subs_mutex);
    mesh_state->subscribers.add(msg.node_id);
  }
}

void LeaveReqMessage::handle_message(NodeID sender, const LeaveReqMessage &msg,
                                     const void *, size_t)
{
  assert(sender != Network::my_node_id);
  Network::node_directory.update_epoch(msg.epoch);

  // TODO: thread-safety semantics here needs more thinking

  if(mesh_state->leaving.load(std::memory_order_acquire)) {
    AutoLock<> al(mesh_state->pending_mutex);
    mesh_state->pending.remove(sender);
  } else {
    AutoLock<> al(mesh_state->subs_mutex);
    mesh_state->subscribers.remove(sender);
  }

  ActiveMessage<LeaveAckMessage> am(sender);
  am.commit();

  if(mesh_state->hooks.pre_leave) {
    realmNodeMeta_t meta{(int32_t)sender, 0, false};
    mesh_state->hooks.pre_leave(&meta, nullptr, 0, /*left=*/false,
                                mesh_state->hooks.user_arg);
  }

  Network::node_directory.remove_slot(sender);
}

void LeaveAckMessage::handle_message(NodeID /*sender*/ sender, const LeaveAckMessage &,
                                     const void *, size_t)
{
  {
    AutoLock<> al(mesh_state->pending_mutex);
    mesh_state->pending.remove(sender);
  }

  {
    AutoLock<> al(mesh_state->pending_mutex);
    if(mesh_state->pending.empty()) {
      assert(mesh_state->hooks.post_leave);
      if(mesh_state->hooks.post_leave) {
        realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, false};
        mesh_state->hooks.post_leave(&meta, nullptr, 0, true, mesh_state->hooks.user_arg);
      }
    }
  }
}

namespace {
  realmStatus_t join(void *st, const realmNodeMeta_t *self)
  {
    MembershipMesh *mesh_state = static_cast<MembershipMesh *>(st);
    assert(mesh_state != nullptr);

    AmProvider *am_provider = new AmProvider();
    Network::node_directory.set_provider(am_provider);
    // state->am_provider = am_provider;

    bool announce_mm = self->announce_mm;
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(self->node_id, announce_mm, dbs);

    if(mesh_state->hooks.pre_join) {
      realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                                 mesh_state->hooks.user_arg);
    }

    if(self->seed_id == NodeDirectory::INVALID_NODE_ID) {
      if(mesh_state->hooks.post_join) {
        realmNodeMeta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
        mesh_state->hooks.post_join(&meta, nullptr, 0, /*joined=*/true,
                                    mesh_state->hooks.user_arg);
      }
    } else {

      ActiveMessage<JoinRequestMessage> am(self->seed_id, dbs.bytes_used());
      am->epoch = Network::node_directory.cluster_epoch();
      am->announce_mm = announce_mm;
      am.add_payload(dbs.get_buffer(), dbs.bytes_used());
      am.commit();
    }

    return realmStatus_t::REALM_SUCCESS;
  }

  // TODO: FIX ALL RACE CONDITIONS
  realmStatus_t leave(void *st, const realmNodeMeta_t *self)
  {
    MembershipMesh *mesh_state = static_cast<MembershipMesh *>(st);
    assert(mesh_state != nullptr);

    if(mesh_state->hooks.pre_leave) {
      mesh_state->hooks.pre_leave(self, nullptr, 0, /*left=*/false,
                                  mesh_state->hooks.user_arg);
    }

    NodeSet members;

    {
      AutoLock<> al(mesh_state->pending_mutex);
      assert(mesh_state->pending.empty());
      mesh_state->pending = mesh_state->subscribers;
    }

    mesh_state->leaving.store(true, std::memory_order_release);

    {
      AutoLock<> al(mesh_state->pending_mutex);
      if(!mesh_state->pending.empty()) {
        ActiveMessage<LeaveReqMessage> am(mesh_state->pending);
        am->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        am.commit();
      } else {
        if(mesh_state->hooks.post_leave) {
          mesh_state->hooks.post_leave(self, nullptr, 0, /*left=*/true,
                                       mesh_state->hooks.user_arg);
        }
      }
    }

    return realmStatus_t::REALM_SUCCESS;
  }

  const realmMembershipOps_t operations = {
      .join_request = join,
      .leave_request = leave,
  };
} // namespace

realmStatus_t realmMembershipMeshInit(realmMembership_t *out,
                                      realmMembershipHooks_t hooks)
{
  MembershipMesh *state = new MembershipMesh();
  mesh_state = state;
  mesh_state->hooks = hooks;
  return realmMembershipCreate(&operations, state, out);
}
