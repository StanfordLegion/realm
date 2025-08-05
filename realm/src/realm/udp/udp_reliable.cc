#include "realm/udp/udp_shim.h"
#include "realm/timers.h"
#include "realm/network.h"

#include <cstring>

using namespace Realm;

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

UDPShim::UDPShim(TxCallback cb)
  : tx_cb(std::move(cb))
{}

bool UDPShim::in_window(uint16_t seq, uint16_t last_rx)
{
  // window covers last_rx .. last_rx-15
  return seq_diff(last_rx, seq) < WINDOW_BITS;
}

void UDPShim::send_now(TxEntry &e)
{
  if(e.patch) {
    e.patch(e.pkt.data(), e.seq, rx_bitmap);
  }
  tx_cb(e.pkt.data(), e.pkt.size());
  e.ts_last_send = Clock::current_time_in_microseconds();
  ++e.retries;
}

void UDPShim::enqueue(const std::vector<char> &packet, const PatchFn &patch)
{
  TxEntry ent{packet, next_tx_seq++, 0, 0, patch};
  send_now(ent);
  in_flight.push_back(std::move(ent));
  if(in_flight.size() > WINDOW_BITS) {
    in_flight.pop_front();
  }
}

void UDPShim::handle_tx(uint16_t seq, uint16_t ack_bits)
{
  while(!in_flight.empty()) {
    uint16_t diff = seq_diff(in_flight.front().seq, seq);

    if(diff == 0) {
      in_flight.pop_front();
    } else if(diff < WINDOW_BITS && (ack_bits & (1u << (diff - 1)))) {
      in_flight.pop_front();
    } else {
      break;
    }
  }
}

bool UDPShim::handle_rx(uint16_t seq, const std::vector<char> &packet)
{
  uint16_t shift = seq_diff(seq, last_rx_seq);

  if(shift == 0) {
    rx_bitmap |= 1u;
  } else if(shift < WINDOW_BITS) {
    rx_bitmap = uint16_t((rx_bitmap << shift) | 1u);
    last_rx_seq = seq;
  } else {
    rx_bitmap = 1u;
    last_rx_seq = seq;
  }

  // Send ACK reflecting updated rx state
  AckHeader ack{};
  ack.src = Network::my_node_id;
  ack.msgid = 0;
  ack.hdr_size = 0;
  ack.payload_size = 0;
  ack.seq = last_rx_seq;
  ack.ack_bits = rx_bitmap;
  tx_cb(&ack, sizeof(ack));

  return reorder.insert(seq, new std::vector<char>(packet));
}

std::vector<char> UDPShim::pull()
{
  std::vector<char> *vec = reorder.pull();
  if(vec) {
    std::vector<char> ret = std::move(*vec);
    delete vec;
    return ret;
  }
  return {};
}

void UDPShim::poll(uint64_t now_us)
{
  for(auto &e : in_flight) {
    if(e.retries >= int(MAX_RETRIES)) {
      assert(0);
      continue;
    }
    if(now_us - e.ts_last_send >= RETRY_USEC) {
      assert(e.patch);
      send_now(e);
    }
  }
}
