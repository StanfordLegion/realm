/* Copyright 2024 NVIDIA Corporation
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

#ifndef UDP_RELIABLE_H
#define UDP_RELIABLE_H

#include <vector>
#include <deque>
#include <cstdint>
#include <functional>
#include "realm/udp/frag_list.h"

namespace Realm {

  // Simple stop-and-wait + selective-ack sliding-window reliability layer
  // for UDP transport.  This code is transport-agnostic: the owner supplies a
  // callback that actually transmits the byte-vector.
  //
  // Packet format expected by the caller:
  //   | UDPHeader | user_header | payload |
  // where UDPHeader contains `seq` and `ack_bits` fields (see udp_module.h).
  // UDPShim merely fills those two fields and manages retransmission.
  //
  // Lifetime: one UDPShim instance per peer.
  class UDPShim {
  public:
    // Callback type: (const void *buf, size_t len) -> void.
    using TxCallback = std::function<void(const void *, size_t)>;

    explicit UDPShim(TxCallback cb);

    using PatchFn = std::function<void(void *, uint16_t, uint16_t)>;

    // queue a freshly-built packet for transmission – seq/acks are filled in
    // automatically before the first send.
    void enqueue(const std::vector<char> &packet, const PatchFn &patch);

    // Process ACK information from a packet (data or standalone ACK)
    void handle_tx(uint16_t seq, uint16_t ack_bits);

    // Process an incoming data packet and returns true if new in-order data
    // became available (caller should then pull).
    bool handle_rx(uint16_t seq, const std::vector<char> &packet);

    // get next in-order packet if available
    std::vector<char> pull();
    uint16_t get_rx_seq() const { return last_rx_seq; }
    uint16_t get_rx_bitmap() const { return rx_bitmap; }
    bool has_outstanding() const { return !in_flight_entries.empty(); }

    // Retransmit any packet whose ACK is overdue; must be polled periodically.
    void poll(uint64_t now_us);

  private:
    struct TxEntry {
      std::vector<char> pkt;
      uint16_t seq;
      uint64_t ts_last_send;
      int retries;
      std::function<void(void *, uint16_t, uint16_t)> patch;
    };

    // constants
    static constexpr unsigned WINDOW_BITS = 16;   // selective-ack bitmap size
    static constexpr unsigned RETRY_USEC = 50000; // 50 ms
    static constexpr unsigned MAX_RETRIES = 5;

    static inline uint16_t seq_diff(uint16_t a, uint16_t b) { return uint16_t(a - b); }
    static bool in_window(uint16_t seq, uint16_t last_rx);

    void send_now(TxEntry &e);

    TxCallback tx_cb;
    uint16_t next_tx_seq{0};
    uint16_t last_rx_seq{0};
    uint16_t rx_bitmap{0};

    std::deque<TxEntry> in_flight_entries;
    FragList<std::vector<char>> frag_list;
  };

} // namespace Realm

#endif
