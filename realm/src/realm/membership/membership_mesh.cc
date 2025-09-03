#include "realm/node_directory.h"
#include "realm/membership/membership.h"
#include "realm/membership/gossip.h"
#include "realm/quiescence.h"
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
  bool enable_dtpl_timeouts{false};
  bool enable_ctpl_timeouts{true};
} // namespace MembershioConfig

enum LeavingState
{
  NONE = 0,
  PREPARE = 1,
  COMMIT = 2
};

class AmMesh;

namespace {
  AmMesh *am_mesh = nullptr;
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

namespace {
  void mesh_on_peer_failed(NodeID peer, PeerFailureKind kind, void *user_data);
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
  GossipMonitor *m_{nullptr};
  Mutex mtx_;
  Mutex::CondVar cond_;
  std::atomic<bool> shutdown_{false};
};

class AmMesh final : public IMembership {
public:
  AmMesh(const IMembership::Hooks &hooks)
    : hooks(hooks)
  {
    if(MembershioConfig::enable_ctpl_timeouts) {
      gossip_.set_state_callback([&](const NodeInfo &peer, bool alive) {
        if(!alive) {
          mesh_on_peer_failed(peer.node_id, PeerFailureKind::LivenessTimeout, nullptr);
        }
      });
      gossip_.set_backend(make_default_gossip_backend(gossip_));
      gossip_poller_ = new GossipPoller(&gossip_);
      gossip_poller_->add_to_manager(&get_runtime()->bgwork);
    }

    if(MembershioConfig::enable_dtpl_timeouts) {
      Network::register_peer_failure_callback(&mesh_on_peer_failed, nullptr);
    }

    // AmProvider *am_provider = new AmProvider();
    Network::node_directory.set_provider(new AmProvider()); // TODO: release memory
  }

  void start() override
  {
    if(gossip_poller_) {
      gossip_.start(NodeInfo{(int32_t)Network::my_node_id, 0, false});
      gossip_poller_->begin_polling();
    }
  }

  void stop() override
  {
    if(gossip_poller_) {
      gossip_.stop();
      gossip_poller_->end_polling();
    }
  }

  ~AmMesh()
  {
    if(am_mesh == this) {
      am_mesh = nullptr;
    }

    if(gossip_poller_) {
      gossip_.stop();
      delete gossip_poller_;
    }
  }

  bool join(const NodeInfo &self) override
  {
    bool announce_mm = self.announce_mm;
    Serialization::DynamicBufferSerializer dbs(DBS_SIZE);
    Network::node_directory.export_node(self.node_id, announce_mm, dbs);

    if(hooks.pre_join) {
      hooks.pre_join(NodeInfo{(int32_t)Network::my_node_id, 0, announce_mm},
                     /*joined=*/false);
    }

    if(self.seed_id == NodeDirectory::INVALID_NODE_ID) {
      if(hooks.post_join) {
        hooks.post_join(NodeInfo{(int32_t)Network::my_node_id, 0, announce_mm},
                        /*joined=*/true);
      }
    } else {
      ActiveMessage<JoinRequestMessage> am(self.seed_id, dbs.bytes_used());
      am->epoch = Network::node_directory.cluster_epoch();
      am->announce_mm = announce_mm;
      am.add_payload(dbs.get_buffer(), dbs.bytes_used());
      am.commit();
    }
    return true;
  }

  bool leave(const NodeInfo &self) override
  {
    leaving_state.store(LeavingState::PREPARE, std::memory_order_release);

    if(hooks.pre_leave) {
      hooks.pre_leave(self, /*left=*/false);
    }

    NodeSet members;

    {
      AutoLock<> al(subs_mutex);
      assert(pending.empty());
      pending = subscribers;
    }

    {
      AutoLock<> al(pending_mutex);
      if(!pending.empty()) {
        ActiveMessage<LeavePrepareMessage> prep(pending);
        prep->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        prep.commit();
      } else {
        // No peers, complete leave immediately.
        if(hooks.post_leave) {
          hooks.post_leave(self, /*left=*/true);
        }
      }
    }

    return true;
  }

  std::atomic<int> join_acks{0};
  std::atomic<int> join_acks_total{0};
  Mutex subs_mutex;
  NodeSet subscribers;
  Mutex pending_mutex;
  NodeSet pending;
  std::atomic<LeavingState> leaving_state{LeavingState::NONE};
  IMembership::Hooks hooks;
  GossipPoller *gossip_poller_{nullptr};
  GossipMonitor gossip_;
};

namespace {
  void mesh_on_peer_failed(NodeID peer, PeerFailureKind kind, void *user_data)
  {
    assert(am_mesh);

    {
      AutoLock<> al(am_mesh->subs_mutex);
      am_mesh->subscribers.remove(peer);
    }

    bool empty_members = false;

    {
      AutoLock<> al(am_mesh->pending_mutex);
      am_mesh->pending.remove(peer);
      empty_members = am_mesh->pending.empty();
    }

    if(am_mesh->hooks.pre_leave) {
      NodeInfo meta{(int32_t)peer, 0, false};
      am_mesh->hooks.pre_leave(meta, /*left=*/true);
    }

    if(am_mesh->hooks.post_leave) {
      NodeInfo meta{(int32_t)peer, 0, false};
      am_mesh->hooks.post_leave(meta, /*left=*/true);
    }

    Network::node_directory.remove_slot(peer);
    uint64_t epoch = Network::node_directory.bump_epoch(peer);

    LeavingState state = LeavingState::NONE;

    if(empty_members) {
      bool force_leave = false;

      if(am_mesh->leaving_state.load(std::memory_order_acquire) ==
         LeavingState::PREPARE) {
        AutoLock<> al(am_mesh->subs_mutex);
        am_mesh->pending = am_mesh->subscribers;

        if(!am_mesh->pending.empty()) {
          state = LeavingState::COMMIT;
          ActiveMessage<LeavePrepareCommitMessage> commit(am_mesh->pending);
          commit->epoch = epoch;
          commit.commit();
        } else {
          force_leave = true;
        }
      }

      if(force_leave ||
         am_mesh->leaving_state.load(std::memory_order_acquire) == LeavingState::COMMIT) {
        AutoLock<> al(am_mesh->pending_mutex);
        NodeInfo meta{(int32_t)Network::my_node_id, 0, false};
        am_mesh->hooks.post_leave(meta, /*left=*/true);
      }
    }

    if(state != LeavingState::NONE) {
      am_mesh->leaving_state.store(state, std::memory_order_release);
    }

    quiescence_on_peer_failed(peer);
  }
} // namespace

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

  if(am_mesh->leaving_state.load(std::memory_order_acquire) != LeavingState::NONE) {
    // send_join_reject
    assert(0); // TODO: TEST ME
    return;
  }

  if(am_mesh->hooks.pre_join) {
    NodeInfo meta{(int32_t)sender, 0, false};
    am_mesh->hooks.pre_join(meta, /*joined=*/false);
  }

  uint64_t new_epoch = Network::node_directory.import_node(data, datalen, /*epoch=*/0);

  {
    AutoLock<> al(am_mesh->subs_mutex);
    if(!am_mesh->subscribers.empty()) {
      assert(am_mesh->subscribers.contains(Network::my_node_id) == false);
      assert(am_mesh->subscribers.contains(sender) == false);
      ActiveMessage<MemberUpdateMessage> am(am_mesh->subscribers, datalen);
      am->node_id = sender;
      am->epoch = new_epoch;
      am->announce_mm = msg.announce_mm;
      am.add_payload(data, datalen);
      am.commit();
    }
  }

  send_join_ack(sender, new_epoch, am_mesh->subscribers.size() + 1, msg.announce_mm);

  if(am_mesh->hooks.filter) {
    NodeInfo meta{(int32_t)sender, 0, msg.announce_mm};
    if(am_mesh->hooks.filter(meta)) {
      AutoLock<> al(am_mesh->subs_mutex);
      am_mesh->subscribers.add(sender);
    }

    am_mesh->gossip_.notify_join(meta);
  }
}

void JoinAcklMessage::handle_message(NodeID sender, const JoinAcklMessage &msg,
                                     const void *data, size_t datalen)
{
  Network::node_directory.import_node(data, datalen, msg.epoch);
  am_mesh->join_acks.fetch_add(1, std::memory_order_relaxed);

  if(msg.acks > 0) {
    am_mesh->join_acks_total.store(msg.acks, std::memory_order_relaxed);
  }

  if(am_mesh->hooks.filter) {
    NodeInfo meta{(int32_t)sender, 0, false};
    if(am_mesh->hooks.filter(meta)) {
      AutoLock<> al(am_mesh->subs_mutex);
      am_mesh->subscribers.add(sender);
    }

    am_mesh->gossip_.notify_join(meta);
  }

  if(am_mesh->join_acks.load() == am_mesh->join_acks_total.load()) {
    if(am_mesh->hooks.post_join) {
      NodeInfo meta{(int32_t)Network::my_node_id, 0, /*announce_mm=*/false};
      am_mesh->hooks.post_join(meta, /*joined=*/true);
    }
  }
}

void MemberUpdateMessage::handle_message(NodeID, const MemberUpdateMessage &msg,
                                         const void *data, size_t datalen)
{
  if(msg.epoch <= Network::node_directory.cluster_epoch() ||
     am_mesh->leaving_state.load(std::memory_order_acquire) != LeavingState::NONE) {
    // send_join_rejec();
    assert(0); // TODO: TEST ME
    return;
  }

  Network::node_directory.import_node(data, datalen, msg.epoch);
  send_join_ack(msg.node_id, msg.epoch, /*acks=*/-1, msg.announce_mm);

  if(am_mesh->hooks.filter) {
    NodeInfo meta{(int32_t)msg.node_id, 0, msg.announce_mm};
    if(!am_mesh->hooks.filter(meta)) {
      // TODO: we need import_node to send the ack back at the same time we
      // don't need mm to be installed in case it has been and thus it needs
      // to be removed as we ll.
      Network::node_directory.remove_slot(msg.node_id);
      return;
    }
  }

  if(am_mesh->hooks.pre_join) {
    NodeInfo meta{(int32_t)msg.node_id, 0, false};
    am_mesh->hooks.pre_join(meta, /*joined=*/false);

    am_mesh->gossip_.notify_join(meta);
  }

  {
    AutoLock<> al(am_mesh->subs_mutex);
    am_mesh->subscribers.add(msg.node_id);
  }
}

void LeavePrepareMessage::handle_message(NodeID sender, const LeavePrepareMessage &msg,
                                         const void * /*data*/, size_t /*datalen*/)
{
  // Stage-A of graceful shutdown.  We do NOT remove the sender yet – that happens
  // in the Commit stage – but we notify higher layers that the peer intends to
  // go away so they can stop scheduling new work there.
  assert(sender != Network::my_node_id);

  if(am_mesh->hooks.pre_leave) {
    NodeInfo meta{(int32_t)sender, 0, /*announce_mm=*/false};
    am_mesh->hooks.pre_leave(meta, /*left=*/false);
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
    AutoLock<> al(am_mesh->pending_mutex);
    am_mesh->pending.remove(sender);
    if(am_mesh->pending.empty()) {
      // All peers acknowledged – begin Commit stage.
      am_mesh->pending = am_mesh->subscribers;
      if(!am_mesh->pending.empty()) {
        am_mesh->leaving_state.store(LeavingState::COMMIT, std::memory_order_release);
        ActiveMessage<LeavePrepareCommitMessage> commit(am_mesh->pending);
        commit->epoch = Network::node_directory.bump_epoch(Network::my_node_id);
        commit.commit();
      } else {
        // No peers – we can finish immediately.
        if(am_mesh->hooks.post_leave) {
          NodeInfo meta{(int32_t)Network::my_node_id, 0, false};
          am_mesh->hooks.post_leave(meta, /*left=*/true);
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
    AutoLock<> al(am_mesh->subs_mutex);
    am_mesh->subscribers.remove(sender);
  }

  if(am_mesh->hooks.post_leave) {
    NodeInfo meta{(int32_t)sender, 0, false};
    am_mesh->hooks.post_leave(meta, /*left=*/true);
    am_mesh->gossip_.notify_leave(meta);
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
    AutoLock<> al(am_mesh->pending_mutex);
    am_mesh->pending.remove(sender);
    if(am_mesh->pending.empty()) {
      if(am_mesh->hooks.post_leave) {
        NodeInfo meta{(int32_t)Network::my_node_id, 0, false};
        am_mesh->hooks.post_leave(meta, /*left=*/true);
      }
    }
  }
}

std::unique_ptr<IMembership> create_am_mesh(const IMembership::Hooks &hooks)
{
  auto ptr = std::make_unique<AmMesh>(hooks);
  am_mesh = static_cast<AmMesh *>(ptr.get());
  return ptr;
}
