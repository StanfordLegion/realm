#ifndef REALM_GOSSIP_H
#define REALM_GOSSIP_H

#include "realm/network.h"
#include "realm/membership/membership.h"

#include <cstdint>
#include <memory>
#include <functional>

namespace Realm {

  // Pluggable heartbeat/liveness backend interface
  class GossipBackend {
  public:
    virtual ~GossipBackend() = default;
    virtual void start(const NodeInfo &self) = 0;
    virtual void stop() = 0;
    virtual void poll(uint64_t now_ns = 0) = 0;
    virtual void notify_join(const NodeInfo &peer) = 0;
    virtual void notify_leave(const NodeInfo &peer) = 0;
  };

  // Callback type: alive=false means the peer is considered dead/timed-out
  using GossipStateChangeCB = std::function<void(const NodeInfo &peer, bool alive)>;

  class GossipMonitor {
  public:
    GossipMonitor() = default;
    ~GossipMonitor() { }

    GossipMonitor(const GossipMonitor &) = delete;
    GossipMonitor &operator=(const GossipMonitor &) = delete;
    GossipMonitor(GossipMonitor &&) = default;
    GossipMonitor &operator=(GossipMonitor &&) = default;

    void set_backend(std::unique_ptr<GossipBackend> backend)
    {
      backend_ = std::move(backend);
    }

    void set_state_callback(GossipStateChangeCB cb) { state_cb_ = std::move(cb); }

    void start(const NodeInfo &self)
    {
      if(backend_) {
        backend_->start(self);
      }
    }

    void stop()
    {
      if(backend_) {
        backend_->stop();
      }
    }

    void poll(uint64_t now_ns = 0)
    {
      if(backend_) {
        backend_->poll(now_ns);
      }
    }

    void notify_join(const NodeInfo &peer)
    {
      if(backend_) {
        backend_->notify_join(peer);
      }
    }

    void notify_leave(const NodeInfo &peer)
    {
      if(backend_) {
        backend_->notify_leave(peer);
      }
    }

    void on_state_change(const NodeInfo &peer, bool alive)
    {
      if(state_cb_) {
        state_cb_(peer, alive);
      }
    }

  private:
    std::unique_ptr<GossipBackend> backend_;
    GossipStateChangeCB state_cb_;
  };

  std::unique_ptr<GossipBackend> make_default_gossip_backend(GossipMonitor &owner);

} // namespace Realm

#endif // REALM_GOSSIP_H
