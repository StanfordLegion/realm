#include "realm/node_directory.h"
#include "realm/membership/membership.h"
#include "realm/membership/gossip.h"
#include "realm/network.h"
#include "realm/serialize.h"
#include "realm/activemsg.h"

#include "realm/runtime_impl.h"

#include <cstring>
#include <atomic>

using namespace Realm;

class GossipPoller;

namespace MembershioConfig {
  // dtpl - data_plane :)
  bool enable_dtpl_timeouts{true};
  bool enable_ctpl_timeouts{true};
} // namespace MembershioConfig

enum LeavingState
{
  NONE = 0,
  PREPARE = 1,
  COMMIT = 2
};

struct MembershipMesh {
  std::atomic<int> join_acks{0};
  std::atomic<int> join_acks_total{0};

  Mutex subs_mutex;
  NodeSet subscribers;

  Mutex pending_mutex;
  NodeSet pending;

  std::atomic<LeavingState> leaving_state{LeavingState::NONE};

  membership_hooks_t hooks;

  GossipPoller *gossip_poller_{nullptr};
  GossipMonitor gossip_;
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

  struct LeavePrepareMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeavePrepareMessage &msg,
                               const void *data, size_t datalen);
  };

  struct LeavePrepareAckMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeavePrepareAckMessage &msg,
                               const void *data, size_t datalen);
  };

  struct LeavePrepareCommitMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeavePrepareCommitMessage &msg,
                               const void *data, size_t datalen);
  };

  struct LeavePrepareCommitAckMessage : ControlPlaneMessageTag {
    Epoch_t epoch;

    static void handle_message(NodeID sender, const LeavePrepareCommitAckMessage &msg,
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

  ActiveMessageHandlerReg<LeavePrepareMessage> leaveprep_handler;
  ActiveMessageHandlerReg<LeavePrepareAckMessage> leaveprep_ack_handler;

  ActiveMessageHandlerReg<LeavePrepareCommitMessage> leaveprep_commit_handler;
  ActiveMessageHandlerReg<LeavePrepareCommitAckMessage> leaveprep_commit_ack_handler;
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

  if(mesh_state->leaving_state.load(std::memory_order_acquire) != LeavingState::NONE) {
    // send_join_reject
    assert(0); // TODO: TEST ME
    return;
  }

  if(mesh_state->hooks.pre_join) {
    node_meta_t meta{(int32_t)sender, 0, false};
    mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                               mesh_state->hooks.user_arg);
  }

  uint64_t new_epoch = Network::node_directory.import_node(data, datalen, /*epoch=*/0);

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

  send_join_ack(sender, new_epoch, mesh_state->subscribers.size() + 1, msg.announce_mm);

  if(mesh_state->hooks.filter) {
    node_meta_t meta{(int32_t)sender, 0, msg.announce_mm};
    if(mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      AutoLock<> al(mesh_state->subs_mutex);
      mesh_state->subscribers.add(sender);
    }

    mesh_state->gossip_.notify_join(meta);
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
    node_meta_t meta{(int32_t)sender, 0, false};
    if(mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      AutoLock<> al(mesh_state->subs_mutex);
      mesh_state->subscribers.add(sender);
    }

    mesh_state->gossip_.notify_join(meta);
  }

  if(mesh_state->join_acks.load() == mesh_state->join_acks_total.load()) {
    if(mesh_state->hooks.post_join) {
      node_meta_t meta{(int32_t)Network::my_node_id, 0, /*announce_mm=*/false};
      mesh_state->hooks.post_join(&meta, nullptr, 0, /*joined=*/true,
                                  mesh_state->hooks.user_arg);
    }
  }
}

void MemberUpdateMessage::handle_message(NodeID, const MemberUpdateMessage &msg,
                                         const void *data, size_t datalen)
{
  if(msg.epoch <= Network::node_directory.cluster_epoch() ||
     mesh_state->leaving_state.load(std::memory_order_acquire) != LeavingState::NONE) {
    // send_join_rejec();
    assert(0); // TODO: TEST ME
    return;
  }

  Network::node_directory.import_node(data, datalen, msg.epoch);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.announce_mm);

  if(mesh_state->hooks.filter) {
    node_meta_t meta{(int32_t)msg.node_id, 0, msg.announce_mm};
    if(!mesh_state->hooks.filter(&meta, mesh_state->hooks.user_arg)) {
      // TODO: we need import_node to send the ack back at the same time we
      // don't need mm to be installed in case it has been and thus it needs
      // to be removed as we ll.
      Network::node_directory.remove_slot(msg.node_id);
      return;
    }
  }

  if(mesh_state->hooks.pre_join) {
    node_meta_t meta{(int32_t)msg.node_id, 0, false};
    mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                               mesh_state->hooks.user_arg);

    mesh_state->gossip_.notify_join(meta);
  }

  {
    AutoLock<> al(mesh_state->subs_mutex);
    mesh_state->subscribers.add(msg.node_id);
  }
}

void LeavePrepareMessage::handle_message(NodeID sender, const LeavePrepareMessage &msg,
                                         const void * /*data*/, size_t /*datalen*/)
{
  // Stage-A of graceful shutdown.  We do NOT remove the sender yet – that happens
  // in the Commit stage – but we notify higher layers that the peer intends to
  // go away so they can stop scheduling new work there.
  assert(sender != Network::my_node_id);

  if(mesh_state->hooks.pre_leave) {
    node_meta_t meta{(int32_t)sender, 0, /*announce_mm=*/false};
    mesh_state->hooks.pre_leave(&meta, nullptr, 0, /*left=*/false,
                                mesh_state->hooks.user_arg);
  }

  // Record the intention – here we could mark the directory entry as
  // PENDING_REMOVE, but for simplicity we just leave it untouched until commit.
  (void)msg; // msg.epoch is currently unused, kept for future extensions.

  // Acknowledge receipt so the leaver can move to the Commit phase.
  ActiveMessage<LeavePrepareAckMessage> ack(sender);
  ack->epoch = Network::node_directory.cluster_epoch();
  ack.commit();
}

void LeavePrepareAckMessage::handle_message(NodeID /*sender*/ sender,
                                            const LeavePrepareAckMessage & /*msg*/,
                                            const void * /*data*/, size_t /*datalen*/)
{
  // Executed only on the leaving node.  Track which peers have acknowledged the
  // prepare stage.  Once everyone has responded we advance to Commit.
  {
    AutoLock<> al(mesh_state->pending_mutex);
    mesh_state->pending.remove(sender);
    if(mesh_state->pending.empty()) {
      // All peers acknowledged – begin Commit stage.
      mesh_state->pending = mesh_state->subscribers;
      if(!mesh_state->pending.empty()) {
        mesh_state->leaving_state.store(LeavingState::COMMIT, std::memory_order_release);
        ActiveMessage<LeavePrepareCommitMessage> commit(mesh_state->pending);
        commit->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        commit.commit();
      } else {
        // No peers – we can finish immediately.
        if(mesh_state->hooks.post_leave) {
          node_meta_t meta{(int32_t)Network::my_node_id, 0, false};
          mesh_state->hooks.post_leave(&meta, nullptr, 0, /*left=*/true,
                                       mesh_state->hooks.user_arg);
        }
      }
    }
  }
}

void LeavePrepareCommitMessage::handle_message(NodeID sender,
                                               const LeavePrepareCommitMessage &msg,
                                               const void * /*data*/, size_t /*datalen*/)
{
  // Final removal step executed on every remaining peer.
  assert(sender != Network::my_node_id);

  Network::node_directory.update_epoch(msg.epoch);

  // Remove directory entry and subscriber record.
  {
    AutoLock<> al(mesh_state->subs_mutex);
    mesh_state->subscribers.remove(sender);
  }

  if(mesh_state->hooks.post_leave) {
    node_meta_t meta{(int32_t)sender, 0, false};
    mesh_state->hooks.post_leave(&meta, nullptr, 0, /*left=*/true,
                                 mesh_state->hooks.user_arg);
    mesh_state->gossip_.notify_leave(meta);
  }

  ActiveMessage<LeavePrepareCommitAckMessage> ack(sender);
  ack->epoch = Network::node_directory.cluster_epoch();
  ack.commit();

  Network::node_directory.remove_slot(sender);
}

void LeavePrepareCommitAckMessage::handle_message(
    NodeID /*sender*/ sender, const LeavePrepareCommitAckMessage & /*msg*/,
    const void * /*data*/, size_t /*datalen*/)
{
  // Leaving node receives commit acknowledgements; when everyone is done we
  // fire the post_leave hook.
  {
    AutoLock<> al(mesh_state->pending_mutex);
    mesh_state->pending.remove(sender);
    if(mesh_state->pending.empty()) {
      if(mesh_state->hooks.post_leave) {
        node_meta_t meta{(int32_t)Network::my_node_id, 0, false};
        mesh_state->hooks.post_leave(&meta, nullptr, 0, /*left=*/true,
                                     mesh_state->hooks.user_arg);
      }
    }
  }
}

namespace {
  realm_status_t join(void *st, const node_meta_t *self)
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
      node_meta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
      mesh_state->hooks.pre_join(&meta, nullptr, 0, /*joined=*/false,
                                 mesh_state->hooks.user_arg);
    }

    if(self->seed_id == NodeDirectory::INVALID_NODE_ID) {
      if(mesh_state->hooks.post_join) {
        node_meta_t meta{(int32_t)Network::my_node_id, 0, announce_mm};
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

    return realm_status_t::REALM_SUCCESS;
  }

  // TODO: FIX ALL RACE CONDITIONS
  realm_status_t leave(void *st, const node_meta_t *self)
  {
    MembershipMesh *mesh_state = static_cast<MembershipMesh *>(st);
    assert(mesh_state != nullptr);

    mesh_state->leaving_state.store(LeavingState::PREPARE, std::memory_order_release);

    if(mesh_state->hooks.pre_leave) {
      mesh_state->hooks.pre_leave(self, nullptr, 0, /*left=*/false,
                                  mesh_state->hooks.user_arg);
    }

    NodeSet members;

    {
      AutoLock<> al(mesh_state->subs_mutex);
      assert(mesh_state->pending.empty());
      mesh_state->pending = mesh_state->subscribers;
    }

    {
      AutoLock<> al(mesh_state->pending_mutex);
      if(!mesh_state->pending.empty()) {
        ActiveMessage<LeavePrepareMessage> prep(mesh_state->pending);
        prep->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        prep.commit();
      } else {
        // No peers, complete leave immediately.
        if(mesh_state->hooks.post_leave) {
          mesh_state->hooks.post_leave(self, nullptr, 0, /*left=*/true,
                                       mesh_state->hooks.user_arg);
        }
      }
    }

    return realm_status_t::REALM_SUCCESS;
  }

  void mesh_on_peer_failed(NodeID peer, PeerFailureKind kind, void *user_data)
  {
    if(!mesh_state) {
      return;
    }

    {
      AutoLock<> al(mesh_state->subs_mutex);
      mesh_state->subscribers.remove(peer);
    }

    bool empty_members = false;

    {
      AutoLock<> al(mesh_state->pending_mutex);
      mesh_state->pending.remove(peer);
      empty_members = mesh_state->pending.empty();
    }

    if(mesh_state->hooks.pre_leave) {
      node_meta_t meta{(int32_t)peer, 0, false};
      mesh_state->hooks.pre_leave(&meta, nullptr, 0, /*left=*/true,
                                  mesh_state->hooks.user_arg);
    }

    if(mesh_state->hooks.post_leave) {
      node_meta_t meta{(int32_t)peer, 0, false};
      mesh_state->hooks.post_leave(&meta, nullptr, 0, /*left=*/true,
                                   mesh_state->hooks.user_arg);
    }

    Network::node_directory.remove_slot(peer);
    uint64_t epoch = Network::node_directory.bump_epoch(peer);

    LeavingState state = LeavingState::NONE;

    if(empty_members) {

      bool force_leave = false;

      if(mesh_state->leaving_state.load(std::memory_order_acquire) ==
         LeavingState::PREPARE) {
        AutoLock<> al(mesh_state->subs_mutex);
        mesh_state->pending = mesh_state->subscribers;

        if(!mesh_state->pending.empty()) {
          state = LeavingState::COMMIT;
          ActiveMessage<LeavePrepareCommitMessage> commit(mesh_state->pending);
          commit->epoch = epoch;
          commit.commit();
        } else {
          force_leave = true;
        }
      }

      if(force_leave || mesh_state->leaving_state.load(std::memory_order_acquire) ==
                            LeavingState::COMMIT) {
        AutoLock<> al(mesh_state->pending_mutex);
        node_meta_t meta{(int32_t)Network::my_node_id, 0, false};
        mesh_state->hooks.post_leave(&meta, nullptr, 0, /*left=*/true,
                                     mesh_state->hooks.user_arg);
      }
    }

    if(state != LeavingState::NONE) {
      mesh_state->leaving_state.store(state, std::memory_order_release);
    }
  }

  const membership_ops_t operations = {
      .join_request = join,
      .leave_request = leave,
  };
} // namespace

class GossipPoller : public BackgroundWorkItem {
public:
  explicit GossipPoller(GossipMonitor *m)
    : BackgroundWorkItem("gossip")
    , m_(m)
    , cond_(mtx_)
  {}

  void begin_polling() { make_active(); }

  void end_polling()
  {
    AutoLock<> al(mtx_);
    shutdown_.store(true, std::memory_order_release);
    cond_.wait();
  }

  bool do_work(TimeLimit) override
  {
    if(shutdown_.load(std::memory_order_acquire)) {
      AutoLock<> al(mtx_);
      shutdown_.store(false, std::memory_order_release);
      cond_.broadcast();
      return false;
    }

    if(m_) {
      m_->poll(0);
    }
    return true;
  }

private:
  GossipMonitor *m_;
  Mutex mtx_;
  Mutex::CondVar cond_;
  std::atomic<bool> shutdown_{false};
};

realm_status_t membership_mesh_init(membership_handle_t *out, membership_hooks_t hooks)
{
  MembershipMesh *state = new MembershipMesh();
  mesh_state = state;
  mesh_state->hooks = hooks;

  if(MembershioConfig::enable_ctpl_timeouts) {
    mesh_state->gossip_.set_state_callback([&](const node_meta_t &peer, bool alive) {
      if(!alive) {
        mesh_on_peer_failed(peer.node_id, PeerFailureKind::LivenessTimeout, nullptr);
      }
    });

    mesh_state->gossip_.set_backend(make_default_gossip_backend(mesh_state->gossip_));

    {
      node_meta_t meta{(int32_t)Network::my_node_id, 0, false};
      mesh_state->gossip_.start(meta);
    }

    mesh_state->gossip_poller_ = new GossipPoller(&mesh_state->gossip_);
    mesh_state->gossip_poller_->add_to_manager(&get_runtime()->bgwork);
    mesh_state->gossip_poller_->begin_polling();
  }

  if(MembershioConfig::enable_dtpl_timeouts) {
    Network::register_peer_failure_callback(&mesh_on_peer_failed, state);
  }

  return membership_create(&operations, state, out);
}

realm_status_t membership_mesh_destroy(membership_handle_t out)
{
  assert(mesh_state);

  if(mesh_state->gossip_poller_) {
    mesh_state->gossip_poller_->end_polling();
  }

  return membership_delete(out);
}
