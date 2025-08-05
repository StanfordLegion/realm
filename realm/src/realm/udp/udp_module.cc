#include "realm/udp/udp_module.h"
#include "realm/activemsg.h"
#include "realm/network.h"
#include "realm/udp/udp_shim.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <netinet/in.h>
#include <netdb.h>

namespace Realm {

  namespace {
    constexpr size_t PAYLOAD_SIZE = 64 * 1024;
    Logger log_udp("udp");
  } // namespace

  /* ------------------------------------------------------------------ */
  /* TxWorker                                                           */
  /* ------------------------------------------------------------------ */
  TxWorker::TxWorker(UDPModule *owner)
    : BackgroundWorkItem("udp rx")
    , module(owner)
    , shutdown_flag(false)
    , shutdown_cond(shutdown_mutex)
  {}

  bool TxWorker::do_work(TimeLimit)
  {
    if(shutdown_flag.load()) {
      AutoLock<> al(shutdown_mutex);
      shutdown_flag.store(false);
      shutdown_cond.broadcast();
      return false;
    }

    bool did_work = true;

    while(!module->shutting_down_.load()) {
      AutoLock<> al(module->peer_map_mutex);
      for(auto &kv : module->peer_map_) {
        kv.second->retransmit.poll(Clock::current_time_in_microseconds());
      }
      break;
    }

    return did_work;
  }

  void TxWorker::begin_polling() { make_active(); }

  void TxWorker::end_polling()
  {
    AutoLock<> al(shutdown_mutex);
    assert(!shutdown_flag.load());
    shutdown_flag.store(true);
    shutdown_cond.wait();
  }

  /* ------------------------------------------------------------------ */
  /* RxWorker                                                          */
  /* ------------------------------------------------------------------ */
  RxWorker::RxWorker(UDPModule *owner, int sock)
    : BackgroundWorkItem("udp rx")
    , module(owner)
    , sock_fd(sock)
    , shutdown_flag(false)
    , shutdown_cond(shutdown_mutex)
  {}

  bool RxWorker::do_work(TimeLimit until)
  {
    uint8_t buf[PAYLOAD_SIZE];

    if(shutdown_flag.load()) {
      AutoLock<> al(shutdown_mutex);
      shutdown_flag.store(false);
      shutdown_cond.broadcast();
      return false;
    }

    bool did_work = true;

    while(!module->shutting_down_.load()) {
      sockaddr_in src{};
      socklen_t slen = sizeof(src);
      ssize_t n = recvfrom(sock_fd, buf, sizeof(buf), 0,
                           reinterpret_cast<sockaddr *>(&src), &slen);

      if(n <= 0) {
        break;
      }

      if(static_cast<size_t>(n) < sizeof(UDPModule::UDPHeader)) {
        assert(0);
        // continue;
      }

      auto *uh = reinterpret_cast<const UDPModule::UDPHeader *>(buf);

      // Always send an ACK for every *data* packet that comes in – whether it is
      // new or a duplicate – so the sender can make progress even if our
      // previous ACK got lost.  Do NOT ACK pure-ACK packets themselves to avoid
      // endless ACK ping-pong.
      const bool is_ack = (uh->msgid == 0) && (uh->payload_size == 0);
      module->register_peer(uh->src, src.sin_addr.s_addr, ntohs(src.sin_port));
      UDPPeerShim *peer = module->ensure_peer(uh->src);

      peer->retransmit.handle_tx(uh->seq, uh->ack_bits);
      if(!is_ack) {
        std::vector<char> packet(buf, buf + n);
        peer->retransmit.handle_rx(uh->seq, packet);
      }

      while(true) {
        auto vec = peer->retransmit.pull();
        if(vec.empty()) {
          break;
        }

        auto *vh = reinterpret_cast<const UDPModule::UDPHeader *>(vec.data());
        const uint8_t *hdr =
            reinterpret_cast<const uint8_t *>(vec.data()) + sizeof(UDPModule::UDPHeader);

        module->runtime_->message_manager->add_incoming_message(
            vh->src, vh->msgid, hdr, vh->hdr_size, PAYLOAD_COPY, hdr + uh->hdr_size,
            vh->payload_size, PAYLOAD_COPY, nullptr, 0, 0, until);

        rx_counter_.fetch_add(1, std::memory_order_relaxed);
        did_work = true;
      }
    }

    return did_work;
  }

  void RxWorker::begin_polling() { make_active(); }

  void RxWorker::end_polling()
  {
    AutoLock<> al(shutdown_mutex);

    assert(!shutdown_flag.load());
    shutdown_flag.store(true);
    shutdown_cond.wait();
  }

  namespace {
    bool parse_address(uint32_t &address, const std::string &host_info)
    {
      if(inet_pton(AF_INET, host_info.c_str(), &address) != 1) {
        struct addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        struct addrinfo *res = nullptr;
        int rc = getaddrinfo(host_info.c_str(), nullptr, &hints, &res);
        if(rc == 0 && res != nullptr) {
          auto *addr = reinterpret_cast<sockaddr_in *>(res->ai_addr);
          address = addr->sin_addr.s_addr;
          freeaddrinfo(res);
        } else {
          log_udp.error() << "UDPModule::init: failed to resolve seed host '" << host_info
                          << "' : " << gai_strerror(rc);
          return false;
        }
      }
      return true;
    }
  } // namespace

  /* ------------------------------------------------------------------ */
  /* UDPModule                                                          */
  /* ------------------------------------------------------------------ */

  UDPModule::UDPModule(RuntimeImpl *rt)
    : NetworkModule("udp")
    , runtime_(rt)
  {}

  UDPModule::~UDPModule()
  {
    shutting_down_.store(true);

    assert(tx_worker_ != nullptr);
    delete tx_worker_;

    assert(rx_worker_ != nullptr);
    delete rx_worker_;

    if(sock_fd_ >= 0) {
      close(sock_fd_);
    }
  }

  NetworkModule *UDPModule::create_network_module(RuntimeImpl *rt, int *, const char ***)
  {
    return new UDPModule(rt);
  }

  void UDPModule::parse_command_line(RuntimeImpl *runtime,
                                     std::vector<std::string> &cmdline)
  {
    NodeID config_rank_id = NodeDirectory::INVALID_NODE_ID;
    int config_self_port = 0;
    std::string config_seed_address;
    CommandLineParser cp;

    cp.add_option_int("-ll:id", config_rank_id);
    cp.add_option_string("-ll:seed", config_seed_address);
    cp.add_option_int("-udp:bind", config_self_port);

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    if(config_self_port == 0) {
      config_self_port = 50000;
    }

    init(static_cast<uint16_t>(config_self_port), config_seed_address, config_rank_id);
  }

  void UDPModule::init(uint16_t base_port, const std::string &seedinfo,
                       NodeID self_rank_id)
  {
    sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd_ >= 0) {
      int flags = fcntl(sock_fd_, F_GETFL, 0);
      if(flags >= 0) {
        fcntl(sock_fd_, F_SETFL, flags | O_NONBLOCK);
      }
    }

    sockaddr_in sin{};
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons(base_port);

    if(bind(sock_fd_, reinterpret_cast<sockaddr *>(&sin), sizeof(sin)) < 0) {
      sin.sin_port = 0;
      if(bind(sock_fd_, reinterpret_cast<sockaddr *>(&sin), sizeof(sin)) < 0) {
        perror("bind");
        abort();
      }
    }

    Network::my_node_id = NodeDirectory::INVALID_NODE_ID;

    {
      sockaddr_in local{};
      socklen_t len = sizeof(local);
      getsockname(sock_fd_, (sockaddr *)&local, &len);

      const std::string local_ip = inet_ntoa(local.sin_addr);
      const uint16_t local_port = ntohs(local.sin_port);

      NodeMeta meta;
      meta.epoch = 1;
      meta.ip = inet_addr(local_ip.c_str());
      meta.udp_port = local_port;
      Network::node_directory.add_slot(self_rank_id, meta);
    }

    assert(self_rank_id != NodeDirectory::INVALID_NODE_ID);

    if(!seedinfo.empty()) {
      size_t colon = seedinfo.find(':');
      std::string host_info = seedinfo.substr(0, colon);
      uint16_t port = atoi(seedinfo.c_str() + colon + 1);

      uint32_t seed_address = 0;
      bool ok = parse_address(seed_address, host_info);
      assert(ok);

      NodeMeta meta;
      meta.epoch = 1;
      meta.ip = seed_address;
      meta.udp_port = port;
      Network::node_directory.add_slot(NodeDirectory::UNKNOWN_NODE_ID, meta);
      register_peer(NodeDirectory::UNKNOWN_NODE_ID, seed_address, port);
    }

    rx_worker_ = new RxWorker(this, sock_fd_);
    rx_worker_->add_to_manager(&runtime_->bgwork);
    rx_worker_->begin_polling();

    tx_worker_ = new TxWorker(this);
    tx_worker_->add_to_manager(&runtime_->bgwork);
    tx_worker_->begin_polling();
  }

  void UDPModule::attach(RuntimeImpl *, std::vector<NetworkSegment *> &) {}

  void UDPModule::detach(RuntimeImpl *, std::vector<NetworkSegment *> &)
  {
    rx_worker_->end_polling();
    tx_worker_->end_polling();
  }

  void UDPModule::broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes)
  {
    assert(0);
    if(root == Network::my_node_id) {
      memcpy(val_out, val_in, bytes);
    }
  }

  void UDPModule::gather(NodeID root, const void *val_in, void *vals_out, size_t bytes)
  {
    assert(0);
    if(root == Network::my_node_id) {
      memcpy(vals_out, val_in, bytes);
    }
  }

  void UDPModule::allgatherv(const char *val_in, size_t bytes,
                             std::vector<char> &vals_out, std::vector<size_t> &lengths)
  {
    assert(0);
    vals_out.assign(val_in, val_in + bytes);
    lengths.assign(1, bytes);
  }

  size_t UDPModule::sample_messages_received_count() { return rx_counter_.exchange(0); }

  bool UDPModule::check_for_quiescence(size_t sampled) { return sampled == 0; }

  /* ---------- AM factory -------------------------------------------- */

  ActiveMessageImpl *UDPModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const void *, size_t, size_t, void *storage_base, size_t storage_size)
  {
    (void)storage_size;
    return new(storage_base) UDPMessageImpl(this, target, msgid, header_size,
                                            max_payload_size, storage_base, storage_size);
  }

  ActiveMessageImpl *
  UDPModule::create_active_message_impl(const NodeSet &targets, unsigned short msgid,
                                        size_t header_size, size_t max_payload_size,
                                        const void *, size_t, size_t, void *storage_base,
                                        size_t storage_size)
  {
    (void)storage_size;
    return new(storage_base) UDPMessageImpl(this, targets, msgid, header_size,
                                            max_payload_size, storage_base, storage_size);
  }

  ActiveMessageImpl *UDPModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const LocalAddress &, size_t, size_t, const RemoteAddress &, void *storage_base,
      size_t storage_size)
  {
    assert(0);
    return create_active_message_impl(target, msgid, header_size, max_payload_size,
                                      nullptr, 0, 0, storage_base, storage_size);
  }

  size_t UDPModule::recommended_max_payload(NodeID, bool, size_t hdr)
  {
    return PAYLOAD_SIZE - sizeof(UDPHeader) - hdr;
  }

  size_t UDPModule::recommended_max_payload(const NodeSet &, bool, size_t hdr)
  {
    return recommended_max_payload(0, false, hdr);
  }

  size_t UDPModule::recommended_max_payload(NodeID, const RemoteAddress &, bool,
                                            size_t hdr)
  {
    return recommended_max_payload(0, false, hdr);
  }

  size_t UDPModule::recommended_max_payload(NodeID, const void *, size_t, size_t, size_t,
                                            bool, size_t hdr)
  {
    return recommended_max_payload(0, false, hdr);
  }

  size_t UDPModule::recommended_max_payload(const NodeSet &, const void *, size_t, size_t,
                                            size_t, bool, size_t hdr)
  {
    return recommended_max_payload(0, false, hdr);
  }

  size_t UDPModule::recommended_max_payload(NodeID, const LocalAddress &, size_t, size_t,
                                            size_t, const RemoteAddress &, bool,
                                            size_t hdr)
  {
    return recommended_max_payload(0, false, hdr);
  }

  void UDPModule::register_peer(NodeID id, uint32_t ip, uint16_t port)
  {
    AutoLock<> al(peer_map_mutex);

    auto it_unknown = peer_map_.find(NodeDirectory::UNKNOWN_NODE_ID);
    if(it_unknown != peer_map_.end() && id != NodeDirectory::UNKNOWN_NODE_ID) {
      peer_map_[id] = it_unknown->second;
      peer_map_.erase(it_unknown);
    }

    UDPPeerShim *peer = nullptr;
    auto it = peer_map_.find(id);
    if(it == peer_map_.end()) {
      peer = peer_map_[id] = new UDPPeerShim(this);
    } else {
      peer = it->second;
    }

    peer->addr.sin.sin_family = AF_INET;
    peer->addr.sin.sin_port = htons(port);
    peer->addr.sin.sin_addr.s_addr = ip;
  }

  void UDPModule::delete_remote_ep(NodeID id)
  {
    AutoLock<> al(peer_map_mutex);
    auto it = peer_map_.find(id);
    if(it != peer_map_.end()) {
      delete it->second;
      peer_map_.erase(it);
    }
  }

  void UDPModule::send_datagram(const PeerAddr &peer, const void *data, size_t len)
  {
    int status = sendto(sock_fd_, data, len, 0,
                        reinterpret_cast<const sockaddr *>(&peer.sin), sizeof(peer.sin));
    if(status < 0) {
      log_udp.error() << "Failed to send datagram size:" << len
                      << " me:" << Network::my_node_id;
    }
  }

  UDPPeerShim *UDPModule::ensure_peer(NodeID id)
  {
    {
      AutoLock<> al(peer_map_mutex);
      auto it = peer_map_.find(id);
      if(it != peer_map_.end()) {
        return it->second;
      }
    }

    const NodeMeta *nm = Network::node_directory.lookup(id);
    if(nm != nullptr) {
      register_peer(id, nm->ip, nm->udp_port);
    } else {
      assert(0);
      peer_map_[id] = new UDPPeerShim(this);
    }
    return peer_map_[id];
  }

  /* ------------------------------------------------------------------ */
  /* UDPMessageImpl                                                     */
  /* ------------------------------------------------------------------ */

  namespace {
    void *reserve(std::vector<char> &vec, size_t size)
    {
      size_t off = vec.size();
      vec.resize(off + ((size + 7) & ~size_t(7))); // 8-byte align
      return vec.data() + off;
    }
  } // namespace

  UDPMessageImpl::UDPMessageImpl(UDPModule *mod, NodeID tgt, unsigned short msgid,
                                 size_t hdr_size, size_t max_payload, void *storage_base,
                                 size_t storage_size)
    : mod_(mod)
    , target_(tgt)
    , msgid_(msgid)
    , hdr_sz_(hdr_size)
    , multicast_(false)
  {
    const size_t obj_sz = sizeof(*this);
    const size_t need_sz = obj_sz + hdr_size + max_payload;

    char *base = nullptr;
    if(storage_size >= need_sz) {
      base = static_cast<char *>(storage_base);
    } else {
      owned_buffer_.resize(hdr_size + max_payload);
      base = owned_buffer_.data() - obj_sz;
    }

    header_base = base + obj_sz;
    payload_base = base + obj_sz + hdr_size;
    payload_size = max_payload;
  }

  UDPMessageImpl::UDPMessageImpl(UDPModule *mod, const NodeSet &tgts,
                                 unsigned short msgid, size_t hdr_size,
                                 size_t max_payload, void *storage_base,
                                 size_t storage_size)
    : mod_(mod)
    , targets_(tgts)
    , msgid_(msgid)
    , hdr_sz_(hdr_size)
    , multicast_(true)
  {
    const size_t obj_sz = sizeof(*this);
    const size_t need_sz = obj_sz + hdr_size + max_payload;

    char *base = nullptr;
    if(storage_size >= need_sz) {
      base = static_cast<char *>(storage_base);
    } else {
      owned_buffer_.resize(hdr_size + max_payload);
      base = owned_buffer_.data() - obj_sz;
    }

    header_base = base + obj_sz;
    payload_base = base + obj_sz + hdr_size;
    payload_size = max_payload;
  }

  void *UDPMessageImpl::add_local_completion(size_t sz)
  {
    return reserve(local_comp_, sz);
  }

  void *UDPMessageImpl::add_remote_completion(size_t sz)
  {
    return reserve(remote_comp_, sz);
  }

  void UDPMessageImpl::commit(size_t act_payload_size)
  {
    // assert(mod_->peer_map_.empty() == false)

    size_t total = sizeof(UDPModule::UDPHeader) + hdr_sz_ + act_payload_size;
    std::vector<char> pkt(total);

    auto *uh = reinterpret_cast<UDPModule::UDPHeader *>(pkt.data());
    uh->src = Network::my_node_id;
    uh->msgid = msgid_;
    uh->hdr_size = hdr_sz_;
    uh->payload_size = act_payload_size;

    memcpy(pkt.data() + sizeof(UDPModule::UDPHeader), header_base, hdr_sz_);
    memcpy(pkt.data() + sizeof(UDPModule::UDPHeader) + hdr_sz_, payload_base,
           act_payload_size);

    if(multicast_) {
      for(NodeSetIterator it(targets_); it != targets_.end(); ++it) {
        auto peer = mod_->ensure_peer(*it);
        peer->retransmit.enqueue(pkt, [](void *h, uint16_t seq, uint16_t ack) {
          auto *uh = reinterpret_cast<UDPModule::UDPHeader *>(h);
          uh->seq = seq;
          uh->ack_bits = ack;
        });
      }
    } else {
      auto peer = mod_->ensure_peer(target_);
      peer->retransmit.enqueue(pkt, [](void *h, uint16_t seq, uint16_t ack) {
        auto *uh = reinterpret_cast<UDPModule::UDPHeader *>(h);
        uh->seq = seq;
        uh->ack_bits = ack;
      });
    }

    CompletionCallbackBase::invoke_all(local_comp_.data(), local_comp_.size());
    CompletionCallbackBase::invoke_all(remote_comp_.data(), remote_comp_.size());
  }

  void UDPMessageImpl::cancel()
  {
    CompletionCallbackBase::destroy_all(local_comp_.data(), local_comp_.size());
    CompletionCallbackBase::destroy_all(remote_comp_.data(), remote_comp_.size());
  }

} // namespace Realm
