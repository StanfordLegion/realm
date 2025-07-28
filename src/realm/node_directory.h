#ifndef NODE_DIRECTORY_H
#define NODE_DIRECTORY_H

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <shared_mutex>

#include "realm/nodeset.h"
#include "realm/event.h"
#include "realm/nodeset.h"

namespace Realm {
  // -----------------------------------------------------------------
  // Small immutable header describing a peer
  // ------------------------------------------------------------------
  struct NodeMeta {
    NodeID id{NodeID(-1)};
    uint64_t epoch{0};
    uint32_t ip{0};
    uint16_t udp_port{0};
    uint8_t flags{0};
    // uint8_t hash[16]{};
    std::vector<uint8_t> worker_address;
    uint32_t dev_index{0};

    std::vector<uint8_t> machine_model;
  };

  struct NodeSlot {
    NodeMeta brief;
    std::atomic<uint8_t> state{0}; // 0 live, 1 retiring, 2 failed
  };

  // TODO:
  // 1. It needs to be fast, concurent, scalable
  // 2. It needs to integrate well with custom-built DHT
  // and other external membership solutions such as etcd, zookeeper
  // 3. Interface and implementation needs to be cleaned up

  class NodeDirectory {
  public:
    [[nodiscard]] Event request(NodeID id, uint64_t min_epoch = 0);
    void complete(NodeID id, uint64_t epoch, const void *blob, size_t bytes);

    void export_node(NodeID id, bool include_mm,
                     Serialization::DynamicBufferSerializer &dbs);
    void import_node(const void *blob, size_t bytes, uint64_t epoch = 0);

    void add_slot(NodeID id, const NodeMeta &meta);
    void remove_slot(NodeID id);
    NodeMeta *lookup(NodeID id) noexcept;
    uint64_t cluster_epoch() const noexcept;

    NodeSet get_members(bool include_self = false) const;

    size_t size() const noexcept;

    class Provider {
    public:
      virtual ~Provider() = default;
      virtual void put(NodeSet peers, const void *blob, size_t bytes,
                       uint64_t ttl_sec = 0) = 0;
      virtual void fetch(NodeID id) = 0;
    };

    void set_provider(Provider *_provider) { provider = _provider; }

    static constexpr NodeID UNKNOWN_NODE_ID{NodeID(-1)};

  private:
    const NodeSlot *lookup_slot(NodeID id) const noexcept;
    bool update_node_id(NodeID id);
    NodeSlot &slot_rw(NodeID id); // creates if absent
    const NodeSlot *slot_ro(NodeID id) const noexcept;

    void erase(NodeID id);

    uint64_t bump_epoch(NodeID id);
    bool update_epoch(uint64_t new_ep);

    // data
    std::atomic<uint64_t> epoch_{1};
    std::atomic<NodeID> max_node_id_{0};
    mutable std::shared_mutex mtx_;
    std::unordered_map<NodeID, NodeSlot> slots_;

    struct Pending {
      UserEvent ev;
      uint64_t min_epoch;
    };
    std::mutex pend_mtx_;
    std::unordered_map<NodeID, Pending> pending_;

    Provider *provider;
  };

} // namespace Realm

#endif
