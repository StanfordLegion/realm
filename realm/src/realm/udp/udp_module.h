/* Copyright 2024 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UDP_MODULE_H
#define UDP_MODULE_H

#include "realm/network.h"
#include "realm/activemsg.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/mutex.h"
#include "realm/udp/udp_shim.h"

#include <atomic>
#include <map>

#include <sys/socket.h>
#include <netinet/in.h>

#include "realm/bgwork.h"

namespace Realm {

  struct UDPPeerShim;
  class UDPModule;
  class UDPMessageImpl;
  class RxWorker;
  class TxWorker;

  class UDPModule : public NetworkModule {
  public:
    explicit UDPModule(RuntimeImpl *runtime);
    ~UDPModule() override;

    static NetworkModule *create_network_module(RuntimeImpl *runtime, int *argc,
                                                const char ***argv);

    /* ----- NetworkModule overrides ------------------------------------ */
    void get_shared_peers(NodeSet &shared_peers) override {}
    void parse_command_line(RuntimeImpl *runtime,
                            std::vector<std::string> &cmdline) override;

    void attach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments) override;
    void detach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments) override;

    void barrier(void) override {}
    void broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes) override;
    void gather(NodeID root, const void *val_in, void *vals_out, size_t bytes) override;
    void allgatherv(const char *val_in, size_t bytes, std::vector<char> &vals_out,
                    std::vector<size_t> &lengths) override;

    size_t sample_messages_received_count(void) override;

    void collect_quiescence_counters(NodeID node, QuiescenceCounters &out) override;
    bool check_for_quiescence(size_t sampled) override;

    ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size) override;

    ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size) override;

    ActiveMessageImpl *create_active_message_impl(NodeID target, unsigned short, size_t,
                                                  size_t, const RemoteAddress &, void *,
                                                  size_t) override
    {
      abort();
    }

    ActiveMessageImpl *create_active_message_impl(const NodeSet &, unsigned short, size_t,
                                                  size_t, const void *, size_t, size_t,
                                                  void *, size_t) override;

    size_t recommended_max_payload(NodeID, bool, size_t) override;
    size_t recommended_max_payload(const NodeSet &, bool, size_t) override;
    size_t recommended_max_payload(NodeID, const RemoteAddress &, bool, size_t) override;
    size_t recommended_max_payload(NodeID, const void *, size_t, size_t, size_t, bool,
                                   size_t) override;
    size_t recommended_max_payload(const NodeSet &, const void *, size_t, size_t, size_t,
                                   bool, size_t) override;
    size_t recommended_max_payload(NodeID, const LocalAddress &, size_t, size_t, size_t,
                                   const RemoteAddress &, bool, size_t) override;

    void register_peer(NodeID id, uint32_t ip, uint16_t port);
    void delete_remote_ep(NodeID id) override;

    MemoryImpl *create_remote_memory(RuntimeImpl *, Memory, size_t, Memory::Kind,
                                     const ByteArray &) override
    {
      assert(0);
      return nullptr;
    }

    IBMemory *create_remote_ib_memory(RuntimeImpl *, Memory, size_t, Memory::Kind,
                                      const ByteArray &) override
    {
      assert(0);
      return nullptr;
    }

  private:
    void init(uint16_t base_port, const std::string &address, NodeID rank_id);

    struct PeerAddr {
      sockaddr_in sin{};
    };

    struct UDPHeader {
      uint32_t src;
      uint16_t msgid;
      uint16_t hdr_size;
      uint32_t payload_size;
      uint16_t seq;
      uint16_t ack_bits;
    } __attribute__((packed));

    void send_datagram(const PeerAddr &peer, const void *data, size_t len);
    UDPPeerShim *ensure_peer(NodeID id);

    Mutex peer_map_mutex;

    RuntimeImpl *runtime_;
    int sock_fd_{-1};

    RxWorker *rx_worker_;
    TxWorker *tx_worker_;

    std::atomic<bool> shutting_down_{false};
    std::atomic<size_t> rx_counter_{0};

    std::map<NodeID, UDPPeerShim *> peer_map_;

    friend class UDPMessageImpl;
    friend class RxWorker;
    friend class TxWorker;
    friend struct UDPPeerShim;
  };

  struct UDPPeerShim {
    UDPModule::PeerAddr addr;
    UDPShim retransmit;

    explicit UDPPeerShim(UDPModule *owner)
      : retransmit([owner, this](const void *buf, size_t len) {
        owner->send_datagram(addr, buf, len);
      })
    {}
  };

  class RxWorker : public BackgroundWorkItem {
  public:
    RxWorker(UDPModule *owner, int sock);
    ~RxWorker(void) override {};

    bool do_work(TimeLimit work_until) override;
    void begin_polling();
    void end_polling();

  private:
    UDPModule *module;
    int sock_fd;

    std::atomic<size_t> rx_counter_{0};

    Mutex shutdown_mutex;
    atomic<bool> shutdown_flag;
    Mutex::CondVar shutdown_cond;
  };

  class TxWorker : public BackgroundWorkItem {
  public:
    TxWorker(UDPModule *owner);
    ~TxWorker(void) override {};

    bool do_work(TimeLimit work_until) override;
    void begin_polling();
    void end_polling();

  private:
    UDPModule *module;
    Mutex shutdown_mutex;
    atomic<bool> shutdown_flag;
    Mutex::CondVar shutdown_cond;
  };

  /* ------------------------------------------------------------------ */
  class UDPMessageImpl : public ActiveMessageImpl {
  public:
    UDPMessageImpl(UDPModule *mod, NodeID tgt, unsigned short msgid, size_t header_size,
                   size_t max_payload, void *storage_base, size_t storage_size);

    UDPMessageImpl(UDPModule *mod, const NodeSet &tgts, unsigned short msgid,
                   size_t header_size, size_t max_payload, void *storage_base,
                   size_t storage_size);

    ~UDPMessageImpl() override {}

    void *add_local_completion(size_t size) override;
    void *add_remote_completion(size_t size) override;
    void commit(size_t act_payload_size) override;
    void cancel() override;

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
