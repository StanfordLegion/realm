#ifndef NODE_DIRECTORY_H
#define NODE_DIRECTORY_H

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <shared_mutex>

#include "realm/event.h"
#include "realm/nodeset.h"
#include "realm/activemsg.h"

namespace Realm {

  // ------------------------------------------------------------------
  // Active-message op-codes
  // ------------------------------------------------------------------
  struct DirGetRequest : ControlPlaneMessageTag {
    NodeID id; // node we want
    uint64_t expect_epoch;
    static void handle_message(NodeID sender, const DirGetRequest &req, const void *,
                               size_t);
  };

  struct DirGetReply : ControlPlaneMessageTag {
    NodeID id;
    uint64_t epoch;
    static void handle_message(NodeID sender, const DirGetReply &rep, const void *payload,
                               size_t bytes);
  };

  // ------------------------------------------------------------------
  // Small immutable header describing a peer
  // ------------------------------------------------------------------
  struct NodeMeta {
    NodeID id{NodeID(-1)};
    uint64_t epoch{0};
    uint32_t ip{0};
    uint16_t udp_port{0};
    uint8_t flags{0};
    //uint8_t hash[16]{};
    std::vector<uint8_t> worker_address;
    uint32_t dev_index{0};

    std::vector<uint8_t> machine_model;
  };

  struct NodeSlot {
    NodeMeta brief;
    std::atomic<uint8_t> state{0}; // 0 live, 1 retiring, 2 failed
  };

  class NodeDirectory {
  public:
    [[nodiscard]] Event request(NodeID id, uint64_t min_epoch = 0);
    void complete(NodeID id, uint64_t epoch, const void *blob, size_t bytes);

    void add_slot(NodeID id, const NodeMeta &meta);
    void remove_slot(NodeID id);

    NodeMeta *lookup(NodeID id) noexcept;
    const NodeSlot *lookup_slot(NodeID id) const noexcept;

    bool update_node_id(NodeID id);
    bool update_epoch(uint64_t new_ep);
    uint64_t bump_epoch(NodeID id);

    NodeSet get_members(bool include_self = false) const;

    uint64_t cluster_epoch() const noexcept;
    size_t size() const noexcept;

    static constexpr NodeID UNKNOWN_NODE_ID{NodeID(-1)};

  private:
    // helpers
    NodeSlot &slot_rw(NodeID id); // creates if absent
    const NodeSlot *slot_ro(NodeID id) const noexcept;

    void erase(NodeID id);

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
  };

} // namespace Realm

#endif
