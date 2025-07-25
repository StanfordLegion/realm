#ifndef UDP_MODULE_H
#define UDP_MODULE_H

#include "realm/network.h"
#include "realm/activemsg.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/mutex.h"

#include <atomic>
#include <map>
#include <thread>

#include <sys/socket.h>
#include <netinet/in.h>

namespace Realm {

  class UDPMessageImpl;

  class UDPModule : public NetworkModule {
  public:
    explicit UDPModule(RuntimeImpl *runtime);
    virtual ~UDPModule();

    static NetworkModule *create_network_module(RuntimeImpl *runtime, int *argc,
                                                const char ***argv);

    /* ----- NetworkModule overrides ------------------------------------ */
    virtual void get_shared_peers(NodeSet &shared_peers) override {}
    virtual void parse_command_line(RuntimeImpl *runtime,
                                    std::vector<std::string> &cmdline) override;
    virtual void attach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) override;
    virtual void detach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) override;
    virtual void barrier(void) override {}
    virtual void broadcast(NodeID root, const void *val_in, void *val_out,
                           size_t bytes) override;
    virtual void gather(NodeID root, const void *val_in, void *vals_out,
                        size_t bytes) override;
    virtual void allgatherv(const char *val_in, size_t bytes, std::vector<char> &vals_out,
                            std::vector<size_t> &lengths) override;

    virtual size_t sample_messages_received_count(void) override;
    virtual bool check_for_quiescence(size_t sampled) override;

    /* active-message factory – only the first two variants are used by
       the control plane, the others abort for now. */
    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size) override;

    virtual ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size) override;

    virtual ActiveMessageImpl *create_active_message_impl(NodeID target, unsigned short,
                                                          size_t, size_t,
                                                          const RemoteAddress &, void *,
                                                          size_t) override
    {
      abort();
    }

    virtual ActiveMessageImpl *create_active_message_impl(const NodeSet &, unsigned short,
                                                          size_t, size_t, const void *,
                                                          size_t, size_t, void *,
                                                          size_t) override;

    virtual size_t recommended_max_payload(NodeID, bool, size_t) override;
    virtual size_t recommended_max_payload(const NodeSet &, bool, size_t) override;
    virtual size_t recommended_max_payload(NodeID, const RemoteAddress &, bool,
                                           size_t) override;
    virtual size_t recommended_max_payload(NodeID, const void *, size_t, size_t, size_t,
                                           bool, size_t) override;
    virtual size_t recommended_max_payload(const NodeSet &, const void *, size_t, size_t,
                                           size_t, bool, size_t) override;
    virtual size_t recommended_max_payload(NodeID, const LocalAddress &, size_t, size_t,
                                           size_t, const RemoteAddress &, bool,
                                           size_t) override;

    void register_peer(NodeID id, const std::string &ip, uint16_t port);
    void register_peer(NodeID id, uint32_t ip, uint16_t port);

    virtual MemoryImpl *create_remote_memory(RuntimeImpl *, Memory, size_t, Memory::Kind,
                                             const ByteArray &) override
    {
      return nullptr;
    }

    virtual IBMemory *create_remote_ib_memory(RuntimeImpl *, Memory, size_t, Memory::Kind,
                                              const ByteArray &) override
    {
      return nullptr;
    }

  private:
    void init(uint16_t base_port, const std::string &address, NodeID rank_id);

    struct PeerAddr {
      sockaddr_in sin{};
    };

    struct UDPHeader {
      uint32_t src; // NodeID of sender
      uint16_t msgid;
      uint16_t hdr_size;
      uint32_t payload_size;
    } __attribute__((packed));

    void rx_loop();
    void send_datagram(const PeerAddr &peer, const void *data, size_t len);

    static constexpr size_t PAYLOAD_SIZE = 64 * 1024;

    Mutex peer_map_mutex;

    RuntimeImpl *runtime_;
    int sock_fd_{-1};
    std::thread rx_thread_;
    std::atomic<bool> shutting_down_{false};
    std::atomic<size_t> rx_counter_{0};

    std::map<NodeID, PeerAddr> peer_map_;
    Logger log_udp_;
    friend class UDPMessageImpl;

    PeerAddr ensure_peer(NodeID id)
    {
      AutoLock<> al(peer_map_mutex);
      auto it = peer_map_.find(id);
      if(it != peer_map_.end()) {
        return it->second;
      }

      const NodeMeta *nm = Network::node_directory.lookup(id);

      assert(nm != nullptr);
      register_peer(id, nm->ip, nm->udp_port);

      return peer_map_[id];
    }
  };

  /* ------------------------------------------------------------------ */
  class UDPMessageImpl : public ActiveMessageImpl {
  public:
    /* unicast */
    UDPMessageImpl(UDPModule *mod, NodeID tgt, unsigned short msgid, size_t header_size,
                   size_t max_payload, void *storage_base, size_t storage_size);

    /* multicast */
    UDPMessageImpl(UDPModule *mod, const NodeSet &tgts, unsigned short msgid,
                   size_t header_size, size_t max_payload, void *storage_base,
                   size_t storage_size);

    virtual ~UDPMessageImpl() {}

    /* ActiveMessageImpl overrides */
    virtual void *add_local_completion(size_t size) override;
    virtual void *add_remote_completion(size_t size) override;
    virtual void commit(size_t act_payload_size) override;
    virtual void cancel() override;

  private:
    void dispatch(const void *buf, size_t len);

    UDPModule *mod_;
    NodeID target_;
    NodeSet targets_;
    unsigned short msgid_;
    size_t hdr_sz_;
    bool multicast_;
    std::vector<char> local_comp_;
    std::vector<char> remote_comp_;
    std::vector<char> owned_buffer_;
  };

} // namespace Realm

#endif
