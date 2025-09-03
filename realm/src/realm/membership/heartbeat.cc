#include "realm/membership/heartbeat.h"
#include "realm/network.h"
#include "realm/activemsg.h"
#include "realm/timers.h"

using namespace Realm;

namespace {
  size_t ceil_log2(size_t n)
  {
    size_t p = 1, k = 0;
    while(p < n) {
      p <<= 1;
      ++k;
    }
    return k;
  }

  struct Beat : public ControlPlaneMessageTag {
    uint64_t ts_ns{0};
    static void handle_message(NodeID from, const Beat &m, const void *, size_t);
  };

  struct Bye : public ControlPlaneMessageTag {
    uint64_t ts_ns{0};
    static void handle_message(NodeID from, const Bye &, const void *, size_t);
  };

  ActiveMessageHandlerReg<Beat> global_beat_reg;
  ActiveMessageHandlerReg<Bye> global_bye_reg;
  HeartbeatBackend *singleton_ = nullptr;
} // namespace

void Beat::handle_message(NodeID from, const Beat &m, const void *, size_t)
{
  assert(singleton_);

  if(singleton_) {
    singleton_->on_beat(from, m.ts_ns);
  }
}

void Bye::handle_message(NodeID from, const Bye &, const void *, size_t)
{
  assert(singleton_);
  if(singleton_) {
    singleton_->on_bye(from);
  }
}

HeartbeatBackend::HeartbeatBackend(GossipMonitor &owner)
  : monitor_(owner)
  , rng_(std::random_device{}())
/*: probe_interval_ns_(getenv("REALM_GOSSIP_PERIOD_NS")
                         ? strtoull(getenv("REALM_GOSSIP_PERIOD_NS"), nullptr, 10)
                         : 500ull * 1000 * 1000)
, dead_timeout_ns_(getenv("REALM_GOSSIP_DEAD_NS")
                       ? strtoull(getenv("REALM_GOSSIP_DEAD_NS"), nullptr, 10)
                       : 5ull * 1000 * 1000 * 1000)
, fanout_k_(getenv("REALM_GOSSIP_FANOUT")
                ? strtoull(getenv("REALM_GOSSIP_FANOUT"), nullptr, 10)
                : 0)
, monitor_(owner)
, rng_(std::random_device{}())*/
{}

void HeartbeatBackend::start(const NodeInfo &self)
{
  self_id_ = self.node_id;
  running_.store(true, std::memory_order_release);
  singleton_ = this;
}

void HeartbeatBackend::stop()
{
  running_.store(false, std::memory_order_release);
  singleton_ = nullptr;
}

void HeartbeatBackend::notify_join(const NodeInfo &peer)
{
  AutoLock<> al(mtx_);
  peers_[peer.node_id] = {Clock::current_time_in_nanoseconds(), true};
}

void HeartbeatBackend::notify_leave(const NodeInfo &peer)
{
  // best-effort bye
  ActiveMessage<Bye> bye(peer.node_id);
  bye->ts_ns = Clock::current_time_in_nanoseconds();
  bye.commit();

  AutoLock<> al(mtx_);
  peers_.erase(peer.node_id);
}

void HeartbeatBackend::poll(uint64_t now_ns)
{
  if(!running_.load(std::memory_order_acquire)) {
    return;
  }

  if(now_ns == 0) {
    now_ns = Clock::current_time_in_nanoseconds();
  }

  send_probes(now_ns);
  process_timeouts(now_ns);
}

void HeartbeatBackend::send_probes(uint64_t now_ns)
{
  static thread_local uint64_t last = 0;

  if(now_ns - last < probe_interval_ns_) {
    return;
  }

  last = now_ns;

  std::vector<NodeID> live;
  {
    AutoLock<> al(mtx_);
    for(auto &kv : peers_) {
      if(kv.second.alive) {
        live.push_back(kv.first);
      }
    }
  }

  if(live.empty()) {
    return;
  }

  std::shuffle(live.begin(), live.end(), rng_);

  size_t k = fanout_k_ ? fanout_k_ : std::max<size_t>(1, ceil_log2(live.size()));

  if(live.size() > k) {
    live.resize(k);
  }

  for(NodeID peer : live) {
    ActiveMessage<Beat> beat(peer);
    beat->ts_ns = now_ns;
    beat.commit();
  }
}

void HeartbeatBackend::process_timeouts(uint64_t now_ns)
{
  std::vector<NodeID> dead;
  {
    AutoLock<> al(mtx_);
    now_ns = Clock::current_time_in_nanoseconds();
    for(auto &kv : peers_) {
      auto &p = kv.second;
      if(p.alive && now_ns - p.last_seen > dead_timeout_ns_) {
        p.alive = false;
        dead.push_back(kv.first);
      }
    }
  }

  for(NodeID d : dead) {
    NodeInfo meta{static_cast<int32_t>(d), 0, false};
    monitor_.on_state_change(meta, /*alive=*/false);
  }
}

void HeartbeatBackend::on_beat(NodeID sender, uint64_t ts_ns)
{
  const uint64_t now_ns = Clock::current_time_in_nanoseconds();

  bool revived = false;
  {
    AutoLock<> al(mtx_);
    auto &p = peers_[sender];
    revived = !p.alive;
    p.alive = true;
    p.last_seen = now_ns; // ts_ns;
  }

  if(revived) {
    NodeInfo meta{static_cast<int32_t>(sender), 0, false};
    monitor_.on_state_change(meta, /*alive=*/true);
  }
}

void HeartbeatBackend::on_bye(NodeID sender)
{
  {
    AutoLock<> al(mtx_);
    peers_.erase(sender);
  }
  NodeInfo meta{static_cast<int32_t>(sender), 0, false};
  monitor_.on_state_change(meta, /*alive=*/false);
}
