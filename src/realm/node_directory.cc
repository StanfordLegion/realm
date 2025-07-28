#include "realm/node_directory.h"
#include "realm/runtime_impl.h"
#include "realm/machine_impl.h"
#include "realm/logging.h"
// #include "realm/activemsg.h"
#include <atomic>

using namespace Realm;

namespace {

  Logger log_ndir("ndir");

  constexpr int DBS_SIZE{4096};

  struct WireHeader {
    NodeID id;
    Epoch_t epoch;
    uint32_t ip;
    uint16_t port;
    uint8_t flags;
    uint32_t mm_len;
    uint32_t wrk_len;
  } __attribute__((packed));

  template <typename S>
  inline bool serdez(S &s, WireHeader &h)
  {
    bool ok = true;
    ok &= (s & h.id);
    ok &= (s & h.epoch);
    ok &= (s & h.ip);
    ok &= (s & h.port);
    ok &= (s & h.flags);
    ok &= (s & h.mm_len);
    ok &= (s & h.wrk_len);
    return ok;
  }
} // namespace

TYPE_IS_SERIALIZABLE(WireHeader);

// ------------------------------------------------------------------
// public API
// ------------------------------------------------------------------
Event NodeDirectory::request(NodeID id, uint64_t min_epoch)
{
  {
    std::shared_lock sl(mtx_);
    const NodeSlot *s = slot_ro(id);
    // const RuntimeImpl *rt = runtime_singleton;
    // const bool have_blob = !rt->nodes[id].processors.empty();

    // TODO: NEEDS ATTENTION
    // if(s && ((s->brief.epoch >= min_epoch) && have_blob) || s->brief.flags) {
    if(s && s->brief.flags) {
      return Event::NO_EVENT;
    }
  }

  {
    std::scoped_lock lg(pend_mtx_);
    if(auto it = pending_.find(id); it != pending_.end()) {
      return it->second.ev;
    }

    pending_[id] = {UserEvent::create_user_event(), min_epoch};
    assert(provider);
    // send_directory_get(id, min_epoch);
    provider->fetch(id);
    return pending_[id].ev;
  }
}

void NodeDirectory::export_node(NodeID id, bool include_mm,
                                Serialization::DynamicBufferSerializer &dbs)
{
  const NodeMeta *n = lookup(id);
  assert(n);

  WireHeader h;
  h.id = id;
  h.epoch = n->epoch;
  h.ip = n->ip;
  h.port = n->udp_port;
  h.flags = include_mm;
  h.mm_len = include_mm ? n->machine_model.size() : 0;
  h.wrk_len = n->worker_address.size();

  bool ok = (dbs & h);
  assert(ok);

  if(h.mm_len) {
    dbs.append_bytes(n->machine_model.data(), h.mm_len);
  }

  dbs.append_bytes(n->worker_address.data(), h.wrk_len);
}

void NodeDirectory::import_node(const void *blob, size_t bytes, uint64_t epoch)
{
  Serialization::FixedBufferDeserializer dbs(blob, bytes);

  WireHeader h;
  bool ok = (dbs & h);
  assert(ok);

  size_t header_size = sizeof(WireHeader);

  const uint8_t *p = static_cast<const uint8_t *>(blob) + header_size;
  const void *mm = p;
  const void *wrk = p + h.mm_len;

  uint64_t new_epoch = (epoch > 0) ? epoch : bump_epoch(Network::my_node_id);

  if(h.mm_len) {
    complete(h.id, new_epoch, mm, h.mm_len);
  }

  assert(h.id != Network::my_node_id);

  NodeMeta m;
  m.epoch = new_epoch;
  m.ip = h.ip;
  m.udp_port = h.port;
  m.flags = h.flags;
  m.worker_address.assign(static_cast<const uint8_t *>(wrk),
                          static_cast<const uint8_t *>(wrk) + h.wrk_len);
  add_slot(h.id, m);
}

void NodeDirectory::complete(NodeID id, uint64_t epoch, const void *blob, size_t bytes)
{
  {
    std::unique_lock ul(mtx_);
    NodeSlot &slot = slot_rw(id);
    if(slot.brief.flags == 0) {
      slot.brief.flags = 1;
      slot.brief.epoch = std::max(slot.brief.epoch, epoch);
    } else {
      // duplicate – silently ignore
      return;
    }
  }

  update_node_id(id);

  RuntimeImpl *rt = runtime_singleton;
  rt->machine->parse_node_announce_data(id, blob, bytes, /*remote*/ true);
  rt->machine->update_kind_maps();
  rt->machine->enumerate_mem_mem_affinities();

  std::scoped_lock plg(pend_mtx_);
  if(auto it = pending_.find(id); it != pending_.end()) {
    GenEventImpl::trigger(it->second.ev, false);
    pending_.erase(it);
  }
}

void NodeDirectory::add_slot(NodeID id, const NodeMeta &meta)
{
  std::unique_lock ul(mtx_);
  NodeSlot &s = slot_rw(id);
  s.brief = meta;
  if(meta.epoch > s.brief.epoch) {
    s.brief.epoch = meta.epoch;
  }
  update_epoch(meta.epoch);
  update_node_id(id);
}

void NodeDirectory::remove_slot(NodeID id) { erase(id); }

NodeMeta *NodeDirectory::lookup(NodeID id) noexcept
{
  std::shared_lock sl(mtx_);
  auto it = slots_.find(id);
  return (it == slots_.end()) ? nullptr : &it->second.brief;
}

const NodeSlot *NodeDirectory::lookup_slot(NodeID id) const noexcept
{
  std::shared_lock sl(mtx_);
  auto it = slots_.find(id);
  return (it == slots_.end()) ? nullptr : &it->second;
}

// ------------------------------------------------------------------
// helpers / private
// ------------------------------------------------------------------
NodeSlot &NodeDirectory::slot_rw(NodeID id)
{
  return slots_.try_emplace(id).first->second;
}

const NodeSlot *NodeDirectory::slot_ro(NodeID id) const noexcept
{
  if(auto it = slots_.find(id); it != slots_.end()) {
    return &it->second;
  }
  return nullptr;
}

void NodeDirectory::erase(NodeID id)
{
  std::unique_lock ul(mtx_);
  slots_.erase(id);
}

bool NodeDirectory::update_node_id(NodeID id)
{
  if(Network::my_node_id == NodeDirectory::INVALID_NODE_ID) {
    Network::my_node_id = id;
  }

  NodeID cur = max_node_id_.load(std::memory_order_relaxed);
  while(id < cur && !max_node_id_.compare_exchange_weak(
                        cur, id, std::memory_order_release, std::memory_order_relaxed)) {
  }

  if(id > Network::max_node_id) {
    Network::max_node_id = id;
  }

  return id > cur;
}

bool NodeDirectory::update_epoch(uint64_t new_ep)
{
  uint64_t cur = epoch_.load(std::memory_order_relaxed);
  while(new_ep > cur &&
        !epoch_.compare_exchange_weak(cur, new_ep, std::memory_order_release,
                                      std::memory_order_relaxed)) {
  }
  return new_ep > cur;
}

uint64_t NodeDirectory::bump_epoch(NodeID id)
{
  (void)id;
  return epoch_.fetch_add(1) + 1;
}

uint64_t NodeDirectory::cluster_epoch() const noexcept { return epoch_.load(); }

size_t NodeDirectory::size() const noexcept { return slots_.size(); }

NodeSet NodeDirectory::get_members(bool include_self) const
{
  NodeSet set;
  NodeID me = Network::my_node_id;
  std::shared_lock sl(mtx_);
  for(auto &[id, _] : slots_) {
    if(include_self || id != me) {
      set.add(id);
    }
  }
  return set;
}
