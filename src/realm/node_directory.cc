#include "realm/node_directory.h"
#include "realm/runtime_impl.h"
#include "realm/machine_impl.h"
#include "realm/logging.h"
#include <atomic>

using namespace Realm;

static Logger log_ndir("ndir");

// ------------------------------------------------------------------
// wire helpers
// ------------------------------------------------------------------
namespace {
  inline void send_dir_get(NodeID target, uint64_t expect_epoch)
  {
    DirGetRequest req;
    req.id = target;
    req.expect_epoch = expect_epoch;
    ActiveMessage<DirGetRequest> am(target);
    am->id = req.id;
    am->expect_epoch = req.expect_epoch;
    am.commit();
  }
} // namespace

// ------------------------------------------------------------------
// public API
// ------------------------------------------------------------------
Event NodeDirectory::request(NodeID id, uint64_t min_epoch)
{
  // fast path – requires *only* shared-lock
  {
    std::shared_lock sl(mtx_);
    const NodeSlot *s = slot_ro(id);
    const RuntimeImpl *rt = runtime_singleton;
    const bool have_blob = !rt->nodes[id].processors.empty();

    // TODO: NEEDS ATTENTION
    if(s && ((s->brief.epoch >= min_epoch) && have_blob) || s->brief.flags) {
      return Event::NO_EVENT;
    }
  }

  // consolidate duplicate in-flight requests
  {
    std::scoped_lock lg(pend_mtx_);
    if(auto it = pending_.find(id); it != pending_.end()) {
      return it->second.ev;
    }

    pending_[id] = {UserEvent::create_user_event(), min_epoch};
    send_dir_get(id, min_epoch);
    return pending_[id].ev;
  }
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

// ------------------------------------------------------------------
// Active-message handlers
// ------------------------------------------------------------------
void DirGetRequest::handle_message(NodeID sender, const DirGetRequest &req, const void *,
                                   size_t)
{
  RuntimeImpl *rt = runtime_singleton;
  if(req.id != Network::my_node_id) {
    log_ndir.error() << "DirGetRequest mis-routed: id=" << req.id
                     << " dst=" << Network::my_node_id;
    return;
  }

  Realm::Serialization::DynamicBufferSerializer dbs(256);
  if(!serialize_announcement(dbs, &rt->nodes[Network::my_node_id], rt->machine,
                             Network::get_network(sender))) {
    return;
  }

  ActiveMessage<DirGetReply> rep(sender, dbs.bytes_used());
  rep->id = req.id;
  rep->epoch = Network::node_directory.lookup(req.id)->epoch;
  rep.add_payload(dbs.get_buffer(), dbs.bytes_used());
  rep.commit();
}

static ActiveMessageHandlerReg<DirGetRequest> _reg_dir_get_req;

void DirGetReply::handle_message(NodeID, const DirGetReply &rep, const void *payload,
                                 size_t bytes)
{
  Network::node_directory.complete(rep.id, rep.epoch, payload, bytes);
}

static ActiveMessageHandlerReg<DirGetReply> _reg_dir_get_rep;
