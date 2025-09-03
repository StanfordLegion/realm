#ifndef REALM_HEARTBEAT_H
#define REALM_HEARTBEAT_H

#include "realm/membership/gossip.h"
#include "realm/mutex.h"
#include <unordered_map>
#include <random>

namespace Realm {

  class HeartbeatBackend : public GossipBackend {
  public:
    explicit HeartbeatBackend(GossipMonitor &owner);

    void start(const NodeInfo &self_meta) override;
    void stop() override;
    void poll(uint64_t now_ns = 0) override;
    void notify_join(const NodeInfo &peer) override;
    void notify_leave(const NodeInfo &peer) override;

    void on_beat(NodeID sender, uint64_t ts_ns);
    void on_bye(NodeID sender);

  private:
    /* periodic work */
    void send_probes(uint64_t now_ns);
    void process_timeouts(uint64_t now_ns);

    struct Peer {
      uint64_t last_seen{0};
      bool alive{false};
    };

    /* --- configuration (can be env-overridden) --- */
    uint64_t probe_interval_ns_{500ull * 1000 * 1000};     // e.g. 500ms
    uint64_t dead_timeout_ns_{10ull * 1000 * 1000 * 1000}; // e.g.   5s
    size_t fanout_k_{10};                                  // 0 → log₂N

    /* --- state --- */
    GossipMonitor &monitor_;
    NodeID self_id_{-1};
    std::atomic<bool> running_{false};

    Mutex mtx_;
    std::unordered_map<NodeID, Peer> peers_;

    std::mt19937 rng_;
  };
} // namespace Realm

#endif // REALM_HEARTBEAT_H
