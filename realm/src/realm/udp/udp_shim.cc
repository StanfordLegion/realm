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

#include "realm/udp/udp_shim.h"
#include "realm/timers.h"
#include "realm/network.h"
#include <cstring>

namespace {
  struct AckHeader {
    uint32_t src;
    uint16_t msgid;
    uint16_t hdr_size;
    uint32_t payload_size;
    uint16_t seq;
    uint16_t ack_bits;
  } __attribute__((packed));
} // namespace

namespace Realm {

  UDPShim::UDPShim(TxCallback cb)
    : tx_cb(std::move(cb))
  {}

  bool UDPShim::in_window(uint16_t seq, uint16_t last_rx)
  {
    // window covers last_rx .. last_rx-15
    return seq_diff(last_rx, seq) < WINDOW_BITS;
  }

  void UDPShim::send_now(TxEntry &entry)
  {
    if(entry.patch) {
      entry.patch(entry.pkt.data(), entry.seq, rx_bitmap);
    }
    tx_cb(entry.pkt.data(), entry.pkt.size());
    entry.ts_last_send = Clock::current_time_in_microseconds();
    ++entry.retries;
  }

  void UDPShim::enqueue(const std::vector<char> &packet, const PatchFn &patch)
  {
    TxEntry entry{packet, next_tx_seq++, 0, 0, patch};
    send_now(entry);
    in_flight_entries.push_back(std::move(entry));
    if(in_flight_entries.size() > WINDOW_BITS) {
      in_flight_entries.pop_front();
    }
  }

  void UDPShim::handle_tx(uint16_t seq, uint16_t ack_bits)
  {
    while(!in_flight_entries.empty()) {
      uint16_t diff = seq_diff(in_flight_entries.front().seq, seq);
      if(diff == 0 || (diff < WINDOW_BITS && (ack_bits & (1U << (diff - 1))))) {
        in_flight_entries.pop_front();
      } else {
        break;
      }
    }
  }

  bool UDPShim::handle_rx(uint16_t seq, const std::vector<char> &packet)
  {
    last_rx_seq = seq;

    uint16_t diff = seq_diff(seq, last_rx_seq);
    if(diff < WINDOW_BITS) {
      rx_bitmap = uint16_t((rx_bitmap << diff) | 1U);
    } else {
      rx_bitmap = 1U;
    }

    AckHeader ack{};
    ack.src = Network::my_node_id;
    ack.msgid = 0;
    ack.hdr_size = 0;
    ack.payload_size = 0;
    ack.seq = last_rx_seq;
    ack.ack_bits = rx_bitmap;
    tx_cb(&ack, sizeof(ack));

    return frag_list.insert(seq, new std::vector<char>(packet));
  }

  std::vector<char> UDPShim::pull()
  {
    std::vector<char> *vec = frag_list.pull();
    if(vec) {
      std::vector<char> ret = std::move(*vec);
      delete vec;
      return ret;
    }
    return {};
  }

  void UDPShim::poll(uint64_t now_us)
  {
    for(auto &entry : in_flight_entries) {
      if(entry.retries >= int(MAX_RETRIES)) {
        assert(0);
        continue;
      }
      if(now_us - entry.ts_last_send >= RETRY_USEC) {
        assert(entry.patch);
        send_now(entry);
      }
    }
  }
} // namespace Realm
