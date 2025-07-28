#include "realm/udp/udp_module.h"
#include "realm/activemsg.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <netinet/in.h>

namespace Realm {

  Logger log_udp("udp");

  /* ------------------------------------------------------------------ */
  /* UDPModule                                                          */
  /* ------------------------------------------------------------------ */

  UDPModule::UDPModule(RuntimeImpl *rt)
    : NetworkModule("udp")
    , runtime_(rt)
    , log_udp_("udp")
  {}

  UDPModule::~UDPModule()
  {
    shutting_down_.store(true);
    if(rx_thread_.joinable()) {
      rx_thread_.join();
    }
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
    NodeID config_rank_id = -1;
    int config_udp_port = 0;
    std::string config_udp_address;
    CommandLineParser cp;

    cp.add_option_int("-ll:id", config_rank_id);
    cp.add_option_int("-udp:bind", config_udp_port);
    cp.add_option_string("-udp:seed", config_udp_address);

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    if(config_udp_port == 0) {
      config_udp_port = 50000;
    }

    init(static_cast<uint16_t>(config_udp_port), config_udp_address, config_rank_id);
  }

  void UDPModule::init(uint16_t base_port, const std::string &address,
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
      // fall back to ephemeral if the chosen port is unavailable
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

    assert(self_rank_id != -1);

    if(!address.empty()) {
      size_t colon = address.find(':');
      std::string seed_ip = address.substr(0, colon);
      uint16_t seed_port = atoi(address.c_str() + colon + 1);

      NodeMeta meta;
      meta.epoch = 1;
      meta.ip = inet_addr(seed_ip.c_str());
      meta.udp_port = seed_port;
      Network::node_directory.add_slot(NodeDirectory::UNKNOWN_NODE_ID, meta);

      register_peer(NodeDirectory::UNKNOWN_NODE_ID, seed_ip, seed_port);
    }

    rx_thread_ = std::thread([this]() { rx_loop(); });
  }

  void UDPModule::attach(RuntimeImpl *, std::vector<NetworkSegment *> &) {}

  void UDPModule::detach(RuntimeImpl *, std::vector<NetworkSegment *> &) {}

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

  /* ---------- peer registry ---------------------------------------- */

  void UDPModule::register_peer(NodeID id, const std::string &ip, uint16_t port)
  {
    AutoLock<> al(peer_map_mutex);

    PeerAddr &pa = peer_map_[id];
    pa.sin.sin_family = AF_INET;
    pa.sin.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &pa.sin.sin_addr);
  }

  void UDPModule::register_peer(NodeID id, uint32_t ip, uint16_t port)
  {
    // AutoLock<> al(peer_map_mutex);

    PeerAddr &pa = peer_map_[id];
    pa.sin.sin_family = AF_INET;
    pa.sin.sin_port = htons(port);
    pa.sin.sin_addr.s_addr = ip;
  }

  void UDPModule::send_datagram(const PeerAddr &peer, const void *data, size_t len)
  {
    int status = sendto(sock_fd_, data, len, 0,
                        reinterpret_cast<const sockaddr *>(&peer.sin), sizeof(peer.sin));
    if(status < 0) {
      log_udp.error() << "Failed to send datagram size:" << len;
    }
  }

  /* ---------- rx loop ---------------------------------------------- */

  void UDPModule::rx_loop()
  {
    uint8_t buf[PAYLOAD_SIZE];

    while(!shutting_down_.load()) {
      ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0, nullptr, nullptr);
      if(n <= 0) {
        /* nothing ready – cheap sleep */
        usleep(1000);
        continue;
      }

      if(static_cast<size_t>(n) < sizeof(UDPHeader)) {
        continue; /* malformed */
      }

      auto *uh = reinterpret_cast<const UDPHeader *>(buf);
      const uint8_t *hdr = buf + sizeof(UDPHeader);
      const uint8_t *pl = hdr + uh->hdr_size;

      /* simple sanity */
      if(sizeof(UDPHeader) + uh->hdr_size + uh->payload_size != static_cast<size_t>(n)) {
        continue;
      }

      /* hand to AM manager – copies header/payload */
      runtime_->message_manager->add_incoming_message(
          uh->src, uh->msgid, hdr, uh->hdr_size, PAYLOAD_COPY, pl, uh->payload_size,
          PAYLOAD_COPY, nullptr, 0, 0, TimeLimit::relative(0));

      rx_counter_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /* ------------------------------------------------------------------ */
  /* UDPMessageImpl                                                     */
  /* ------------------------------------------------------------------ */

  static void *reserve(std::vector<char> &vec, size_t size)
  {
    size_t off = vec.size();
    vec.resize(off + ((size + 7) & ~size_t(7))); // 8-byte align
    return vec.data() + off;
  }

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
    // assert(mod_->peer_map_.empty() == false);

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
        const auto &pa = mod_->ensure_peer(*it);
        mod_->send_datagram(pa, pkt.data(), pkt.size());
      }
    } else {
      const auto &pa = mod_->ensure_peer(target_);
      mod_->send_datagram(pa, pkt.data(), pkt.size());
    }

    /* invoke completions synchronously */
    CompletionCallbackBase::invoke_all(local_comp_.data(), local_comp_.size());
    CompletionCallbackBase::invoke_all(remote_comp_.data(), remote_comp_.size());
  }

  void UDPMessageImpl::cancel()
  {
    CompletionCallbackBase::destroy_all(local_comp_.data(), local_comp_.size());
    CompletionCallbackBase::destroy_all(remote_comp_.data(), remote_comp_.size());
  }

} // namespace Realm
