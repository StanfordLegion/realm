/*
 * Copyright 2025 Los Alamos National Laboratory, Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

#include "realm/realm_config.h"

#ifdef REALM_ON_WINDOWS
#define NOMINMAX
#endif

#include "realm/transfer/channel_common.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/channel_disk.h"
#include "realm/transfer/transfer.h"
#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
#include "realm/utils.h"

#include <algorithm>

TYPE_IS_SERIALIZABLE(Realm::XferDesKind);

namespace Realm {

  Logger log_new_dma("new_dma");
  Logger log_request("request");
  Logger log_xd("xd");
  Logger log_xd_ref("xd_ref");

  ////////////////////////////////////////////////////////////////////////
  //
  // class SequenceAssembler
  //

  SequenceAssembler::SequenceAssembler(void)
    : contig_amount_x2(0)
    , first_noncontig((size_t)-1)
    , mutex(0)
  {}

  SequenceAssembler::SequenceAssembler(const SequenceAssembler &copy_from)
    : contig_amount_x2(copy_from.contig_amount_x2)
    , first_noncontig(copy_from.first_noncontig)
    , mutex(0)
    , spans(copy_from.spans)
  {}

  SequenceAssembler::~SequenceAssembler(void)
  {
    if(mutex.load())
      delete mutex.load();
  }

  Mutex *SequenceAssembler::ensure_mutex()
  {
    Mutex *ptr = mutex.load();
    if(ptr)
      return ptr;
    // allocate one and try to install it
    Mutex *new_mutex = new Mutex;
    if(mutex.compare_exchange(ptr, new_mutex)) {
      // succeeded - return the mutex we made
      return new_mutex;
    } else {
      // failed - someone already set the mutex with new_mutex,
      // so just return the mutex
      delete new_mutex;
      return mutex.load();
    }
  }

  bool SequenceAssembler::empty() const { return (contig_amount_x2.load() == 0); }

  void SequenceAssembler::swap(SequenceAssembler &other)
  {
    // NOT thread-safe - taking mutexes won't help
    std::swap(contig_amount_x2, other.contig_amount_x2);
    std::swap(first_noncontig, other.first_noncontig);
    spans.swap(other.spans);
  }

  void SequenceAssembler::import(SequenceAssembler &other) const
  {
    size_t contig_sample_x2 = contig_amount_x2.load_acquire();
    if(contig_sample_x2 > 1)
      other.add_span(0, contig_sample_x2 >> 1);
    if((contig_sample_x2 & 1) != 0) {
      for(std::map<size_t, size_t>::const_iterator it = spans.begin(); it != spans.end();
          ++it)
        other.add_span(it->first, it->second);
    }
  }

  // asks if a span exists - return value is number of bytes from the
  //  start that do
  size_t SequenceAssembler::span_exists(size_t start, size_t count)
  {
    // lock-free case 1: start < contig_amount
    size_t contig_sample_x2 = contig_amount_x2.load_acquire();
    if(start < (contig_sample_x2 >> 1)) {
      size_t max_avail = (contig_sample_x2 >> 1) - start;
      if(count < max_avail)
        return count;
      else
        return max_avail;
    }

    // lock-free case 2a: no noncontig ranges known
    if((contig_sample_x2 & 1) == 0)
      return 0;

    // lock-free case 2b: contig_amount <= start < first_noncontig
    size_t noncontig_sample = first_noncontig.load();
    if(start < noncontig_sample)
      return 0;

    // general case 3: take the lock and look through spans/etc.
    {
      AutoLock<> al(*ensure_mutex());

      // first, recheck the contig_amount, in case both it and the noncontig
      //  counters were bumped in between looking at the two of them
      size_t contig_sample = contig_amount_x2.load_acquire() >> 1;
      if(start < contig_sample) {
        size_t max_avail = contig_sample - start;
        if(count < max_avail)
          return count;
        else
          return max_avail;
      }

      // recheck noncontig as well
      if(start < first_noncontig.load())
        return 0;

      // otherwise find the first span after us and then back up one to find
      //  the one that might contain our 'start'
      std::map<size_t, size_t>::const_iterator it = spans.upper_bound(start);
      // this should never be the first span
      assert(it != spans.begin());
      --it;
      assert(it->first <= start);
      // does this span overlap us?
      if((it->first + it->second) > start) {
        size_t max_avail = it->first + it->second - start;
        while(max_avail < count) {
          // try to get more - return the current 'max_avail' if we fail
          if(++it == spans.end())
            return max_avail; // no more
          if(it->first > (start + max_avail))
            return max_avail; // not contiguous
          max_avail += it->second;
        }
        // got at least as much as we wanted
        return count;
      } else
        return 0;
    }
  }

  // returns the amount by which the contiguous range has been increased
  //  (i.e. from [pos, pos+retval) )
  size_t SequenceAssembler::add_span(size_t pos, size_t count)
  {
    // nothing to do for empty spans
    if(count == 0)
      return 0;

    // fastest case - try to bump the contig amount without a lock, assuming
    //  there's no noncontig spans
    size_t prev_x2 = pos << 1;
    size_t next_x2 = (pos + count) << 1;
    if(contig_amount_x2.compare_exchange(prev_x2, next_x2)) {
      // success - we bumped by exactly 'count'
      return count;
    }

    // second best case - the CAS failed, but only because there are
    //  noncontig spans...  assuming spans aren't getting too out of order
    //  in the common case, we take the mutex and pick up any other spans we
    //  connect with
    if((prev_x2 >> 1) == pos) {
      size_t span_end = pos + count;
      {
        AutoLock<> al(*ensure_mutex());

        size_t new_noncontig = size_t(-1);
        while(!spans.empty()) {
          std::map<size_t, size_t>::iterator it = spans.begin();
          if(it->first == span_end) {
            span_end += it->second;
            spans.erase(it);
          } else {
            // stop here - this is the new first noncontig
            new_noncontig = it->first;
            break;
          }
        }

        // to avoid false negatives in 'span_exists', update contig amount
        //  before we bump first_noncontig
        next_x2 = (span_end << 1) + (spans.empty() ? 0 : 1);
        // this must succeed
        bool ok = contig_amount_x2.compare_exchange(prev_x2, next_x2);
        assert(ok);

        first_noncontig.store(new_noncontig);
      }

      return (span_end - pos);
    }

    // worst case - our span doesn't appear to be contiguous, so we have to
    //  take the mutex and add to the noncontig list (we may end up being
    //  contiguous if we're the first noncontig and things have caught up)
    {
      AutoLock<> al(*ensure_mutex());

      spans[pos] = count;

      if(pos > first_noncontig.load()) {
        // in this case, we also know that spans wasn't empty and somebody
        //  else has already set the LSB of contig_amount_x2
        return 0;
      } else {
        // we need to re-check contig_amount_x2 and make sure the LSB is
        //  set - do both with an atomic OR
        prev_x2 = contig_amount_x2.fetch_or(1);

        if((prev_x2 >> 1) == pos) {
          // we've been caught, so gather up spans and do another bump
          size_t span_end = pos;
          size_t new_noncontig = size_t(-1);
          while(!spans.empty()) {
            std::map<size_t, size_t>::iterator it = spans.begin();
            if(it->first == span_end) {
              span_end += it->second;
              spans.erase(it);
            } else {
              // stop here - this is the new first noncontig
              new_noncontig = it->first;
              break;
            }
          }
          assert(span_end > pos);

          // to avoid false negatives in 'span_exists', update contig amount
          //  before we bump first_noncontig
          next_x2 = (span_end << 1) + (spans.empty() ? 0 : 1);
          // this must succeed (as long as we remember we set the LSB)
          prev_x2 |= 1;
          bool ok = contig_amount_x2.compare_exchange(prev_x2, next_x2);
          assert(ok);

          first_noncontig.store(new_noncontig);

          return (span_end - pos);
        } else {
          // not caught, so no forward progress to report
          return 0;
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ControlPort::Encoder
  //

  ControlPort::Encoder::Encoder()
    : port_shift(0)
    , state(STATE_INIT)
  {}

  ControlPort::Encoder::~Encoder() { assert(state == STATE_DONE); }

  void ControlPort::Encoder::set_port_count(size_t ports)
  {
    assert(state == STATE_INIT);
    // we add one to valid port indices, so we need to encode values in [0,ports]
    port_shift = 1;
    while((ports >> port_shift) > 0) {
      port_shift++;
      assert(port_shift <= 30); // 1B ports will be bad for other reasons too...
    }
    state = STATE_HAVE_PORT_COUNT;
  }

  // encodes some/all of the { count, port, last } packet into the next
  //  32b - returns true if encoding is complete or false if it should
  //  be called again with the same arguments for another 32b packet
  bool ControlPort::Encoder::encode(unsigned &data, size_t count, int port, bool last)
  {
    unsigned port_p1 = port + 1;
    assert((port_p1 >> port_shift) == 0);

    switch(state) {
    case STATE_INIT:
      assert(0 && "encoding control word without known port count");

    case STATE_HAVE_PORT_COUNT:
    {
      // special case - if we're sending a single packet with count=0,last=1,
      //  we don't need to send the port shift first
      if((count == 0) && last) {
        data = 0;
        state = STATE_DONE;
        log_xd.print() << "encode: " << count << " " << port << " " << last;
        return true;
      } else {
        data = port_shift;
        state = STATE_IDLE;
        return false;
      }
    }

    case STATE_IDLE:
    {
      // figure out if we need 1, 2, or 3 chunks for this
      unsigned mid = (count >> (30 - port_shift));
      unsigned hi = ((sizeof(size_t) > 4) ? (count >> (60 - port_shift)) : 0);

      if(hi != 0) {
        // will take three words - send HIGH first
        data = (hi << 2) | CTRL_HIGH;
        state = STATE_SENT_HIGH;
        return false;
      } else if(mid != 0) {
        // will take two words - send MID first
        data = (mid << 2) | CTRL_MID;
        state = STATE_SENT_MID;
        return false;
      } else {
        // fits in a single word
        data = ((count << (port_shift + 2)) | (port_p1 << 2) |
                (last ? CTRL_LO_LAST : CTRL_LO_MORE));
        state = (last ? STATE_DONE : STATE_IDLE);
        // log_xd.print() << "encode: " << count << " " << port << " " << last;
        return true;
      }
    }

    case STATE_SENT_HIGH:
    {
      // since we just sent HIGH, must send MID next
      unsigned mid = (count >> (30 - port_shift));
      data = (mid << 2) | CTRL_MID;
      state = STATE_SENT_MID;
      return false;
    }

    case STATE_SENT_MID:
    {
      // since we just sent MID, send LO to finish
      data = ((count << (port_shift + 2)) | (port_p1 << 2) |
              (last ? CTRL_LO_LAST : CTRL_LO_MORE));
      state = (last ? STATE_DONE : STATE_IDLE);
      // log_xd.print() << "encode: " << count << " " << port << " " << last;
      return true;
    }

    case STATE_DONE:
      assert(0 && "sending after last?");
    }

    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ControlPort::Decoder
  //

  ControlPort::Decoder::Decoder()
    : temp_count(0)
    , port_shift(0)
  {}

  ControlPort::Decoder::~Decoder()
  {
    // shouldn't end with a partial count
    assert(temp_count == 0);
  }

  // decodes the next 32b of packed data, returning true if a complete
  //  { count, port, last } has been received
  bool ControlPort::Decoder::decode(unsigned data, size_t &count, int &port, bool &last)
  {
    if(port_shift == 0) {
      // we haven't received the port shift yet, so it's either this or a 0
      //  meaning there's no data at all
      if(data != 0) {
        port_shift = data;
        return false;
      } else {
        count = 0;
        port = -1;
        last = true;
        // log_xd.print() << "decode: " << count << " " << port << " " << last;
        return true;
      }
    } else {
      // bottom 2 bits tell us the chunk type
      unsigned ctrl = data & 3;

      if(ctrl == CTRL_HIGH) {
        assert(temp_count == 0); // should not be clobbering an existing count
        temp_count = size_t(data >> 2) << (60 - port_shift);
        assert(temp_count != 0); // should not have gotten HIGH with 0 value
        return false;
      } else if(ctrl == CTRL_MID) {
        temp_count |= size_t(data >> 2) << (30 - port_shift);
        assert(temp_count != 0); // must have gotten HIGH or nonzero here
        return false;
      } else {
        // LO means we have a full control packet
        count = temp_count | (data >> (port_shift + 2));
        unsigned port_p1 = (data >> 2) & ((1U << port_shift) - 1);
        port = port_p1 - 1;
        last = (ctrl == CTRL_LO_LAST);
        temp_count = 0;
        // log_xd.print() << "decode: " << count << " " << port << " " << last;
        return true;
      }
    }
  }

  XferDes::XferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                   XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                   const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                   const void *_fill_data, size_t _fill_size)
    : dma_op(_dma_op)
    , xferDes_queue(XferDesQueue::get_singleton())
    , launch_node(_launch_node)
    , iteration_completed(false)
    , bytes_write_pending(0)
    , transfer_completed(false)
    , max_req_size(16 << 20 /*TO REMOVE*/)
    , priority(_priority)
    , guid(_guid)
    , channel(_channel)
    , fill_data(&inline_fill_storage)
    , fill_size(_fill_size)
    , orig_fill_size(_fill_size)
    , progress_counter(0)
    , reference_count(1)
    , nb_update_pre_bytes_total_calls_expected(0)
    , nb_update_pre_bytes_total_calls_received(0)
  {
    input_ports.resize(inputs_info.size());
    int gather_control_port = -1;
    int scatter_control_port = -1;
    for(size_t i = 0; i < inputs_info.size(); i++) {
      XferPort &p = input_ports[i];
      const XferDesPortInfo &ii = inputs_info[i];

      p.mem = get_runtime()->get_memory_impl(ii.mem);
      assert(p.mem != nullptr && "invalid memory handle");
      p.iter = ii.iter;
      if(ii.serdez_id != 0) {
        const CustomSerdezUntyped *op =
            get_runtime()->custom_serdez_table.get(ii.serdez_id, 0);
        assert(op != 0);
        p.serdez_op = op;
      } else {
        p.serdez_op = 0;
      }
      p.peer_guid = ii.peer_guid;
      p.peer_port_idx = ii.peer_port_idx;
      p.indirect_port_idx = ii.indirect_port_idx;
      p.is_indirect_port = false;      // we'll set these below as needed
      p.needs_pbt_update.store(false); // never needed for inputs
      p.local_bytes_total = 0;
      p.local_bytes_cons.store(0);
      p.remote_bytes_total.store(size_t(-1));
      p.ib_offset = ii.ib_offset;
      p.ib_size = ii.ib_size;
      p.addrcursor.set_addrlist(&p.addrlist);
      switch(ii.port_type) {
      case XferDesPortInfo::GATHER_CONTROL_PORT:
        gather_control_port = i;
        break;
      case XferDesPortInfo::SCATTER_CONTROL_PORT:
        scatter_control_port = i;
        break;
      default:
        break;
      }
    }
    // connect up indirect input ports in a second pass
    for(size_t i = 0; i < inputs_info.size(); i++) {
      XferPort &p = input_ports[i];
      if(p.indirect_port_idx >= 0) {
        p.iter->set_indirect_input_port(this, p.indirect_port_idx,
                                        input_ports[p.indirect_port_idx].iter);
        input_ports[p.indirect_port_idx].is_indirect_port = true;
      }
    }
    if(gather_control_port >= 0) {
      input_control.control_port_idx = gather_control_port;
      input_control.current_io_port = 0;
      input_control.remaining_count = 0;
      input_control.eos_received = false;
    } else {
      input_control.control_port_idx = -1;
      input_control.current_io_port = 0;
      input_control.remaining_count = size_t(-1);
      input_control.eos_received = false;
    }

    output_ports.resize(outputs_info.size());
    for(size_t i = 0; i < outputs_info.size(); i++) {
      XferPort &p = output_ports[i];
      const XferDesPortInfo &oi = outputs_info[i];

      p.mem = get_runtime()->get_memory_impl(oi.mem);
      assert(p.mem != nullptr && "invalid memory handle");
      p.iter = oi.iter;
      if(oi.serdez_id != 0) {
        const CustomSerdezUntyped *op =
            get_runtime()->custom_serdez_table.get(oi.serdez_id, 0);
        assert(op != 0);
        p.serdez_op = op;
      } else {
        p.serdez_op = 0;
      }
      p.peer_guid = oi.peer_guid;
      p.peer_port_idx = oi.peer_port_idx;
      p.indirect_port_idx = oi.indirect_port_idx;
      p.is_indirect_port = false; // outputs are never indirections
      if(oi.indirect_port_idx >= 0) {
        p.iter->set_indirect_input_port(this, oi.indirect_port_idx,
                                        inputs_info[oi.indirect_port_idx].iter);
        input_ports[p.indirect_port_idx].is_indirect_port = true;
      }
      // TODO: further refine this to exclude peers that can figure out
      //  the end of a tranfer some othe way
      p.needs_pbt_update.store(oi.peer_guid != XFERDES_NO_GUID);
      p.local_bytes_total = 0;
      p.local_bytes_cons.store(0);
      p.remote_bytes_total.store(size_t(-1));
      p.ib_offset = oi.ib_offset;
      p.ib_size = oi.ib_size;
      p.addrcursor.set_addrlist(&p.addrlist);

      // if we're writing into an IB, the first 'ib_size' byte
      //  locations can be freely written
      if(p.ib_size > 0) {
        p.seq_remote.add_span(0, p.ib_size);
      }
    }

    if(scatter_control_port >= 0) {
      output_control.control_port_idx = scatter_control_port;
      output_control.current_io_port = 0;
      output_control.remaining_count = 0;
      output_control.eos_received = false;
    } else {
      output_control.control_port_idx = -1;
      output_control.current_io_port = 0;
      output_control.remaining_count = size_t(-1);
      output_control.eos_received = false;
    }

    // allocate a larger buffer if needed for fill data
    if(fill_size > ALIGNED_FILL_STORAGE_SIZE) {
      fill_data = malloc(fill_size);
      assert(fill_data);
    }
    if(fill_size > 0) {
      memcpy(fill_data, _fill_data, fill_size);
    }

    nb_update_pre_bytes_total_calls_expected = 0;
    for(size_t i = 0; i < input_ports.size(); i++) {
      if(input_ports[i].peer_guid != XFERDES_NO_GUID) {
        nb_update_pre_bytes_total_calls_expected++;
      }
    }
    log_xd_ref.info("new xd=%llx, update_pre_bytes_total_expected=%u", guid,
                    nb_update_pre_bytes_total_calls_expected);
  }

  XferDes::~XferDes()
  {
    // clear available_reqs
    while(!available_reqs.empty()) {
      available_reqs.pop();
    }
    for(std::vector<XferPort>::const_iterator it = input_ports.begin();
        it != input_ports.end(); ++it) {
      delete it->iter;
    }
    for(std::vector<XferPort>::const_iterator it = output_ports.begin();
        it != output_ports.end(); ++it) {
      delete it->iter;
    }

    if(fill_data != &inline_fill_storage) {
      free(fill_data);
    }
  };

  Event XferDes::request_metadata()
  {
    std::vector<Event> preconditions;
    for(std::vector<XferPort>::iterator it = input_ports.begin(); it != input_ports.end();
        ++it) {
      Event e = it->iter->request_metadata();
      if(!e.has_triggered()) {
        preconditions.push_back(e);
      }
    }
    for(std::vector<XferPort>::iterator it = output_ports.begin();
        it != output_ports.end(); ++it) {
      Event e = it->iter->request_metadata();
      if(!e.has_triggered()) {
        preconditions.push_back(e);
      }
    }
    return Event::merge_events(preconditions);
  }

  void XferDes::mark_completed()
  {
    for(std::vector<XferPort>::const_iterator it = input_ports.begin();
        it != input_ports.end(); ++it) {
      if(it->ib_size > 0) {
        free_intermediate_buffer(it->mem->me, it->ib_offset, it->ib_size);
      }
    }

    // notify owning DmaRequest upon completion of this XferDes
    // printf("complete XD = %lu\n", guid);
    if(launch_node == Network::my_node_id) {
      TransferOperation *op = reinterpret_cast<TransferOperation *>(dma_op);
      op->notify_xd_completion(guid);
    } else {
      TransferOperation *op = reinterpret_cast<TransferOperation *>(dma_op);
      NotifyXferDesCompleteMessage::send_request(launch_node, op, guid);
    }
  }

#define MAX_GEN_REQS 3

  bool support_2d_xfers(XferDesKind kind)
  {
    return (kind == XFER_GPU_TO_FB) || (kind == XFER_GPU_FROM_FB) ||
           (kind == XFER_GPU_IN_FB) || (kind == XFER_GPU_PEER_FB) ||
           (kind == XFER_REMOTE_WRITE) || (kind == XFER_MEM_CPY);
  }

  size_t XferDes::update_control_info(ReadSequenceCache *rseqcache)
  {
    if(iteration_completed.load_acquire())
      return 0;

    // pull control information if we need it
    if(input_control.remaining_count == 0) {
      if(input_control.control_port_idx >= 0) {
        XferPort &icp = input_ports[input_control.control_port_idx];
        size_t avail =
            icp.seq_remote.span_exists(icp.local_bytes_total, 4 * sizeof(unsigned));
        size_t old_lbt = icp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        while(true) {
          if(avail < sizeof(unsigned))
            return 0; // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = icp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = icp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          icp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          if(input_control.decoder.decode(cword, input_control.remaining_count,
                                          input_control.current_io_port,
                                          input_control.eos_received))
            break;
        }

        // can't get here unless we read something, so ack it
        if(rseqcache != 0)
          rseqcache->add_span(input_control.control_port_idx, old_lbt,
                              icp.local_bytes_total - old_lbt);
        else
          update_bytes_read(input_control.control_port_idx, old_lbt,
                            icp.local_bytes_total - old_lbt);

        log_xd.info() << "input control: xd=" << std::hex << guid << std::dec
                      << " port=" << input_control.current_io_port
                      << " count=" << input_control.remaining_count
                      << " done=" << input_control.eos_received;
      }
      // if count is still zero, we're done
      if(input_control.remaining_count == 0) {
        assert(input_control.eos_received);
        begin_completion();
        return 0;
      }
    }

    if(output_control.remaining_count == 0) {
      if(output_control.control_port_idx >= 0) {
        // this looks wrong, but the port that controls the output is
        //  an input port! vvv
        XferPort &ocp = input_ports[output_control.control_port_idx];
        size_t avail =
            ocp.seq_remote.span_exists(ocp.local_bytes_total, 4 * sizeof(unsigned));
        size_t old_lbt = ocp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        while(true) {
          if(avail < sizeof(unsigned))
            return 0; // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = ocp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = ocp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          ocp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          if(output_control.decoder.decode(cword, output_control.remaining_count,
                                           output_control.current_io_port,
                                           output_control.eos_received))
            break;
        }

        // can't get here unless we read something, so ack it
        if(rseqcache != 0)
          rseqcache->add_span(output_control.control_port_idx, old_lbt,
                              ocp.local_bytes_total - old_lbt);
        else
          update_bytes_read(output_control.control_port_idx, old_lbt,
                            ocp.local_bytes_total - old_lbt);

        log_xd.info() << "output control: xd=" << std::hex << guid << std::dec
                      << " port=" << output_control.current_io_port
                      << " count=" << output_control.remaining_count
                      << " done=" << output_control.eos_received;
      }
      // if count is still zero, we're done
      if(output_control.remaining_count == 0) {
        assert(output_control.eos_received);
        begin_completion();
        return 0;
      }
    }

    return std::min(input_control.remaining_count, output_control.remaining_count);
  }

  size_t XferDes::get_addresses(size_t min_xfer_size, ReadSequenceCache *rseqcache)
  {
    const InstanceLayoutPieceBase *in_nonaffine;
    const InstanceLayoutPieceBase *out_nonaffine;
    size_t ret = get_addresses(min_xfer_size, rseqcache, in_nonaffine, out_nonaffine);
    assert(!in_nonaffine && !out_nonaffine);
    return ret;
  }

  size_t XferDes::get_addresses(size_t min_xfer_size, ReadSequenceCache *rseqcache,
                                const InstanceLayoutPieceBase *&in_nonaffine,
                                const InstanceLayoutPieceBase *&out_nonaffine)
  {
    size_t control_count = update_control_info(rseqcache);
    if(control_count == 0) {
      in_nonaffine = 0;
      out_nonaffine = 0;
      return 0;
    }
    if(control_count < min_xfer_size)
      min_xfer_size = control_count;
    size_t max_bytes = control_count;

    // get addresses for the input, if it exists
    if(input_control.current_io_port >= 0) {
      XferPort *in_port = &input_ports[input_control.current_io_port];

      // do we need more addresses?
      size_t read_bytes_avail = in_port->addrlist.bytes_pending();
      if(read_bytes_avail < min_xfer_size) {
        bool flush = in_port->iter->get_addresses(in_port->addrlist, in_nonaffine);
        read_bytes_avail = in_port->addrlist.bytes_pending();
        if(flush) {
          if(read_bytes_avail > 0) {
            // ignore a nonaffine piece as we still have some affine bytes
            in_nonaffine = 0;
          }

          // adjust min size to flush as requested (unless we're non-affine)
          if(!in_nonaffine)
            min_xfer_size = std::min(min_xfer_size, read_bytes_avail);
        }
      } else
        in_nonaffine = 0;

      // if we're not the first in the chain, respect flow control too
      if(in_port->peer_guid != XFERDES_NO_GUID) {
        read_bytes_avail =
            in_port->seq_remote.span_exists(in_port->local_bytes_total, read_bytes_avail);
        size_t pbt_limit =
            (in_port->remote_bytes_total.load_acquire() - in_port->local_bytes_total);
        min_xfer_size = std::min(min_xfer_size, pbt_limit);

        // don't ever expect to be able to read more than half the size of the
        //  incoming intermediate buffer
        if(min_xfer_size > (in_port->ib_size >> 1))
          min_xfer_size = std::max<size_t>(1, (in_port->ib_size >> 1));
      }

      // we'd like to wait until there's `min_xfer_size` bytes available on the
      //  input, but in gather copies with fork-joins in the dataflow, we
      //  can't be guaranteed that's possible, so move whatever we've got,
      //  relying on the upstream producer to be producing it in the largest
      //  chunks it can
      if((read_bytes_avail > 0) && (read_bytes_avail < min_xfer_size))
        min_xfer_size = read_bytes_avail;

      if(!in_nonaffine) {
        max_bytes = std::min(max_bytes, read_bytes_avail);
      }

    } else {
      in_nonaffine = 0;
    }

    // get addresses for the output, if it exists
    if(output_control.current_io_port >= 0) {
      XferPort *out_port = &output_ports[output_control.current_io_port];

      // do we need more addresses?
      size_t write_bytes_avail = out_port->addrlist.bytes_pending();
      if(write_bytes_avail < min_xfer_size) {
        bool flush = out_port->iter->get_addresses(out_port->addrlist, out_nonaffine);
        write_bytes_avail = out_port->addrlist.bytes_pending();

        // TODO(apryakhin@): We add this to handle scatter when both
        // indirection and source are coming from IB and this needs
        // good testing.
        if(out_port->indirect_port_idx >= 0 && write_bytes_avail) {
          min_xfer_size = std::min(write_bytes_avail, min_xfer_size);
        }
        if(flush) {
          if(write_bytes_avail > 0) {
            // ignore a nonaffine piece as we still have some affine bytes
            out_nonaffine = 0;
          }

          // adjust min size to flush as requested (unless we're non-affine)
          if(!out_nonaffine) {
            min_xfer_size = std::min(min_xfer_size, write_bytes_avail);
          }
        }
      } else
        out_nonaffine = 0;

      // if we're not the last in the chain, respect flow control too
      if(out_port->peer_guid != XFERDES_NO_GUID) {
        write_bytes_avail = out_port->seq_remote.span_exists(out_port->local_bytes_total,
                                                             write_bytes_avail);

        // we'd like to wait until there's `min_xfer_size` bytes available on
        //  the output, but if we're landing in an intermediate buffer and need
        //  to wrap around, waiting won't do any good
        if(min_xfer_size > (out_port->ib_size >> 1))
          min_xfer_size = std::max<size_t>(1, (out_port->ib_size >> 1));
      }

      if(!out_nonaffine)
        max_bytes = std::min(max_bytes, write_bytes_avail);
    } else {
      out_nonaffine = 0;
    }

    if(min_xfer_size == 0) {
      // should only happen in the absence of control ports
      assert((input_control.control_port_idx == -1) &&
             (output_control.control_port_idx == -1));
      begin_completion();
      return 0;
    }

    // if we don't have a big enough chunk, wait for more to show up
    if((max_bytes < min_xfer_size) && !in_nonaffine && !out_nonaffine) {
      return 0;
    }

    return max_bytes;
  }

  bool XferDes::record_address_consumption(size_t total_read_bytes,
                                           size_t total_write_bytes)
  {
    bool in_done = false;
    assert(input_control.remaining_count >= total_read_bytes);
    assert(output_control.remaining_count >= total_write_bytes);
    if(input_control.current_io_port >= 0) {
      XferPort *in_port = &input_ports[input_control.current_io_port];

      in_port->local_bytes_total += total_read_bytes;
      in_port->local_bytes_cons.fetch_add(total_read_bytes);

      if(in_port->peer_guid == XFERDES_NO_GUID)
        in_done = ((in_port->addrlist.bytes_pending() == 0) && in_port->iter->done());
      else
        in_done =
            (in_port->local_bytes_total == in_port->remote_bytes_total.load_acquire());
    }

    bool out_done = false;
    if(output_control.current_io_port >= 0) {
      XferPort *out_port = &output_ports[output_control.current_io_port];

      out_port->local_bytes_total += total_write_bytes;
      out_port->local_bytes_cons.fetch_add(total_write_bytes);

      if(out_port->peer_guid == XFERDES_NO_GUID)
        out_done = ((out_port->addrlist.bytes_pending() == 0) && out_port->iter->done());
    }

    input_control.remaining_count -= total_read_bytes;
    output_control.remaining_count -= total_write_bytes;

    // input or output controls override our notion of done-ness
    if(input_control.control_port_idx >= 0)
      in_done = ((input_control.remaining_count == 0) && input_control.eos_received);

    if(output_control.control_port_idx >= 0)
      out_done = ((output_control.remaining_count == 0) && output_control.eos_received);

    if(in_done || out_done) {
      begin_completion();
      return true;
    } else
      return false;
  }

  void XferDes::replicate_fill_data(size_t new_size)
  {
    if(new_size > fill_size) {
#ifdef DEBUG_REALM
      assert((fill_size > 0) && ((new_size % orig_fill_size) == 0));
#endif
      char *new_fill_data;
      if(new_size > ALIGNED_FILL_STORAGE_SIZE) {
        new_fill_data = (char *)malloc(new_size);
        assert(new_fill_data);
        memcpy(new_fill_data, fill_data, fill_size /*old size*/);
      } else {
        // can still fit in the inline storage, so no bootstrap copy needed
        new_fill_data = (char *)&inline_fill_storage;
      }
      do {
        // can't increase by more than 2x per copy
        size_t to_copy = std::min(new_size - fill_size, fill_size);
        memcpy(new_fill_data + fill_size, new_fill_data, to_copy);
        fill_size += to_copy;
      } while(fill_size < new_size);

      // delete old buffer, if it was allocated
      if(fill_data != &inline_fill_storage)
        free(fill_data);

      fill_data = new_fill_data;
    }
  }

  long XferDes::default_get_requests(Request **reqs, long nr, unsigned flags)
  {
    long idx = 0;

    while((idx < nr) && request_available()) {
      // TODO: we really shouldn't even be trying if the iteration
      //   is already done
      if(iteration_completed.load())
        break;

      // pull control information if we need it
      if(input_control.remaining_count == 0) {
        XferPort &icp = input_ports[input_control.control_port_idx];
        size_t avail =
            icp.seq_remote.span_exists(icp.local_bytes_total, 4 * sizeof(unsigned));
        size_t old_lbt = icp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        bool got_packet = false;
        do {
          if(avail < sizeof(unsigned))
            break; // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = icp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = icp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          icp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          got_packet = input_control.decoder.decode(cword, input_control.remaining_count,
                                                    input_control.current_io_port,
                                                    input_control.eos_received);
        } while(!got_packet);

        // can't make further progress if we didn't get a full packet
        if(!got_packet)
          break;

        update_bytes_read(input_control.control_port_idx, old_lbt,
                          icp.local_bytes_total - old_lbt);

        log_xd.info() << "input control: xd=" << std::hex << guid << std::dec
                      << " port=" << input_control.current_io_port
                      << " count=" << input_control.remaining_count
                      << " done=" << input_control.eos_received;
        // if count is still zero, we're done
        if(input_control.remaining_count == 0) {
          assert(input_control.eos_received);
          begin_completion();
          break;
        }
      }
      if(output_control.remaining_count == 0) {
        // this looks wrong, but the port that controls the output is
        //  an input port! vvv
        XferPort &ocp = input_ports[output_control.control_port_idx];
        size_t avail =
            ocp.seq_remote.span_exists(ocp.local_bytes_total, 4 * sizeof(unsigned));
        size_t old_lbt = ocp.local_bytes_total;

        // may take a few chunks of data to get a control packet
        bool got_packet = false;
        do {
          if(avail < sizeof(unsigned))
            break; // no data right now

          TransferIterator::AddressInfo c_info;
          size_t amt = ocp.iter->step(sizeof(unsigned), c_info, 0, false /*!tentative*/);
          assert(amt == sizeof(unsigned));
          const void *srcptr = ocp.mem->get_direct_ptr(c_info.base_offset, amt);
          assert(srcptr != 0);
          unsigned cword;
          memcpy(&cword, srcptr, sizeof(unsigned));

          ocp.local_bytes_total += sizeof(unsigned);
          avail -= sizeof(unsigned);

          got_packet = output_control.decoder.decode(
              cword, output_control.remaining_count, output_control.current_io_port,
              output_control.eos_received);
        } while(!got_packet);

        // can't make further progress if we didn't get a full packet
        if(!got_packet)
          break;

        update_bytes_read(output_control.control_port_idx, old_lbt,
                          ocp.local_bytes_total - old_lbt);

        log_xd.info() << "output control: xd=" << std::hex << guid << std::dec
                      << " port=" << output_control.current_io_port
                      << " count=" << output_control.remaining_count
                      << " done=" << output_control.eos_received;
        // if count is still zero, we're done
        if(output_control.remaining_count == 0) {
          assert(output_control.eos_received);
          begin_completion();
          break;
        }
      }

      XferPort *in_port = ((input_control.current_io_port >= 0)
                               ? &input_ports[input_control.current_io_port]
                               : 0);
      XferPort *out_port = ((output_control.current_io_port >= 0)
                                ? &output_ports[output_control.current_io_port]
                                : 0);

      // special cases for OOR scatter/gather
      if(!in_port) {
        if(!out_port) {
          // no input or output?  just skip the count?
          assert(0);
        } else {
          // no valid input, so no write to the destination -
          //  just step the output transfer iterator if it's a real target
          //  but barf if it's an IB
          assert((out_port->peer_guid == XferDes::XFERDES_NO_GUID) &&
                 !out_port->serdez_op);
          TransferIterator::AddressInfo dummy;
          size_t skip_bytes = out_port->iter->step(
              std::min(input_control.remaining_count, output_control.remaining_count),
              dummy, flags & TransferIterator::DST_FLAGMASK, false /*!tentative*/);
          log_xd.debug() << "skipping " << skip_bytes << " bytes of output";
          assert(skip_bytes > 0);
          input_control.remaining_count -= skip_bytes;
          output_control.remaining_count -= skip_bytes;
          // TODO: pull this code out to a common place?
          if(((input_control.remaining_count == 0) && input_control.eos_received) ||
             ((output_control.remaining_count == 0) && output_control.eos_received)) {
            log_xd.info() << "iteration completed via control port: xd=" << std::hex
                          << guid << std::dec;
            begin_completion();
            break;
          }
          continue; // try again
        }
      } else if(!out_port) {
        // valid input that we need to throw away
        assert(!in_port->serdez_op);
        TransferIterator::AddressInfo dummy;
        // although we're not reading the IB input data ourselves, we need
        //  to wait until it's ready before not-reading it to avoid WAW
        //  races on the producer side
        size_t skip_bytes =
            std::min(input_control.remaining_count, output_control.remaining_count);
        if(in_port->peer_guid != XferDes::XFERDES_NO_GUID) {
          skip_bytes =
              in_port->seq_remote.span_exists(in_port->local_bytes_total, skip_bytes);
          if(skip_bytes == 0)
            break;
        }
        skip_bytes =
            in_port->iter->step(skip_bytes, dummy, flags & TransferIterator::SRC_FLAGMASK,
                                false /*!tentative*/);
        log_xd.debug() << "skipping " << skip_bytes << " bytes of input";
        assert(skip_bytes > 0);
        update_bytes_read(input_control.current_io_port, in_port->local_bytes_total,
                          skip_bytes);
        in_port->local_bytes_total += skip_bytes;
        input_control.remaining_count -= skip_bytes;
        output_control.remaining_count -= skip_bytes;
        // TODO: pull this code out to a common place?
        if(((input_control.remaining_count == 0) && input_control.eos_received) ||
           ((output_control.remaining_count == 0) && output_control.eos_received)) {
          log_xd.info() << "iteration completed via control port: xd=" << std::hex << guid
                        << std::dec;
          begin_completion();
          break;
        }
        continue; // try again
      }

      // there are several variables that can change asynchronously to
      //  the logic here:
      //   pre_bytes_total - the max bytes we'll ever see from the input IB
      //   read_bytes_cons - conservative estimate of bytes we've read
      //   write_bytes_cons - conservative estimate of bytes we've written
      //
      // to avoid all sorts of weird race conditions, sample all three here
      //  and only use them in the code below (exception: atomic increments
      //  of rbc or wbc, for which we adjust the snapshot by the same)
      size_t pbt_snapshot = in_port->remote_bytes_total.load_acquire();
      size_t rbc_snapshot = in_port->local_bytes_cons.load_acquire();
      size_t wbc_snapshot = out_port->local_bytes_cons.load_acquire();

      // normally we detect the end of a transfer after initiating a
      //  request, but empty iterators and filtered streams can cause us
      //  to not realize the transfer is done until we are asking for
      //  the next request (i.e. now)
      if((in_port->peer_guid == XFERDES_NO_GUID)
             ? in_port->iter->done()
             : (in_port->local_bytes_total == pbt_snapshot)) {
        if(in_port->local_bytes_total == 0)
          log_request.info() << "empty xferdes: " << guid;
        // TODO: figure out how to eliminate false positives from these
        //  checks with indirection and/or multiple remote inputs
        begin_completion();
        break;
      }

      TransferIterator::AddressInfo src_info, dst_info;
      size_t read_bytes, write_bytes, read_seq, write_seq;
      size_t write_pad_bytes = 0;
      size_t read_pad_bytes = 0;

      // handle serialization-only and deserialization-only cases
      //  specially, because they have uncertainty in how much data
      //  they write or read
      if(in_port->serdez_op && !out_port->serdez_op) {
        // serialization only - must be into an IB
        assert(in_port->peer_guid == XFERDES_NO_GUID);
        assert(out_port->peer_guid != XFERDES_NO_GUID);

        // when serializing, we don't know how much output space we're
        //  going to consume, so do not step the dst_iter here
        // instead, see what we can get from the source and conservatively
        //  check flow control on the destination and let the stepping
        //  of dst_iter happen in the actual execution of the request

        // if we don't have space to write a single worst-case
        //  element, try again later
        if(out_port->seq_remote.span_exists(wbc_snapshot,
                                            in_port->serdez_op->max_serialized_size) <
           in_port->serdez_op->max_serialized_size)
          break;

        size_t max_bytes = max_req_size;

        size_t src_bytes = in_port->iter->step(max_bytes, src_info,
                                               flags & TransferIterator::SRC_FLAGMASK,
                                               true /*tentative*/);

        size_t num_elems = src_bytes / in_port->serdez_op->sizeof_field_type;
        // no input data?  try again later
        if(num_elems == 0)
          break;
        assert((num_elems * in_port->serdez_op->sizeof_field_type) == src_bytes);
        size_t max_dst_bytes = num_elems * in_port->serdez_op->max_serialized_size;

        // if we have an output control, restrict the max number of
        //  elements
        if(output_control.control_port_idx >= 0) {
          if(num_elems > output_control.remaining_count) {
            log_xd.info() << "scatter/serialize clamp: " << num_elems << " -> "
                          << output_control.remaining_count;
            num_elems = output_control.remaining_count;
          }
        }

        size_t clamp_dst_bytes = num_elems * in_port->serdez_op->max_serialized_size;
        // test for space using our conserative bytes written count
        size_t dst_bytes_avail =
            out_port->seq_remote.span_exists(wbc_snapshot, clamp_dst_bytes);

        if(dst_bytes_avail == max_dst_bytes) {
          // enough space - confirm the source step
          in_port->iter->confirm_step();
        } else {
          // not enough space - figure out how many elements we can
          //  actually take and adjust the source step
          size_t act_elems = dst_bytes_avail / in_port->serdez_op->max_serialized_size;
          // if there was a remainder in the division, get rid of it
          dst_bytes_avail = act_elems * in_port->serdez_op->max_serialized_size;
          size_t new_src_bytes = act_elems * in_port->serdez_op->sizeof_field_type;
          in_port->iter->cancel_step();
          src_bytes = in_port->iter->step(new_src_bytes, src_info,
                                          flags & TransferIterator::SRC_FLAGMASK,
                                          false /*!tentative*/);
          // this can come up shorter than we expect if the source
          //  iterator is 2-D or 3-D - if that happens, re-adjust the
          //  dest bytes again
          if(src_bytes < new_src_bytes) {
            if(src_bytes == 0)
              break;

            num_elems = src_bytes / in_port->serdez_op->sizeof_field_type;
            assert((num_elems * in_port->serdez_op->sizeof_field_type) == src_bytes);

            // no need to recheck seq_next_read
            dst_bytes_avail = num_elems * in_port->serdez_op->max_serialized_size;
          }
        }

        // since the dst_iter will be stepped later, the dst_info is a
        //  don't care, so copy the source so that lines/planes/etc match
        //  up
        dst_info = src_info;

        read_seq = in_port->local_bytes_total;
        read_bytes = src_bytes;
        in_port->local_bytes_total += src_bytes;

        write_seq = 0; // filled in later
        write_bytes = dst_bytes_avail;
        out_port->local_bytes_cons.fetch_add(dst_bytes_avail);
        wbc_snapshot += dst_bytes_avail;
      } else if(!in_port->serdez_op && out_port->serdez_op) {
        // deserialization only - must be from an IB
        assert(in_port->peer_guid != XFERDES_NO_GUID);
        assert(out_port->peer_guid == XFERDES_NO_GUID);

        // when deserializing, we don't know how much input data we need
        //  for each element, so do not step the src_iter here
        //  instead, see what the destination wants
        // if the transfer is still in progress (i.e. pre_bytes_total
        //  hasn't been set), we have to be conservative about how many
        //  elements we can get from partial data

        // input data is done only if we know the limit AND we have all
        //  the remaining bytes (if any) up to that limit
        bool input_data_done = ((pbt_snapshot != size_t(-1)) &&
                                ((rbc_snapshot >= pbt_snapshot) ||
                                 (in_port->seq_remote.span_exists(
                                      rbc_snapshot, pbt_snapshot - rbc_snapshot) ==
                                  (pbt_snapshot - rbc_snapshot))));
        // if we're using an input control and it's not at the end of the
        //  stream, the above checks may not be precise
        if((input_control.control_port_idx >= 0) && !input_control.eos_received)
          input_data_done = false;

        // this done-ness overrides many checks based on the conservative
        //  out_port->serdez_op->max_serialized_size
        if(!input_data_done) {
          // if we don't have enough input data for a single worst-case
          //  element, try again later
          if((in_port->seq_remote.span_exists(rbc_snapshot,
                                              out_port->serdez_op->max_serialized_size) <
              out_port->serdez_op->max_serialized_size)) {
            break;
          }
        }

        size_t max_bytes = max_req_size;

        size_t dst_bytes = out_port->iter->step(max_bytes, dst_info,
                                                flags & TransferIterator::DST_FLAGMASK,
                                                !input_data_done);

        size_t num_elems = dst_bytes / out_port->serdez_op->sizeof_field_type;
        if(num_elems == 0)
          break;
        assert((num_elems * out_port->serdez_op->sizeof_field_type) == dst_bytes);
        size_t max_src_bytes = num_elems * out_port->serdez_op->max_serialized_size;
        // if we have an input control, restrict the max number of
        //  elements
        if(input_control.control_port_idx >= 0) {
          if(num_elems > input_control.remaining_count) {
            log_xd.info() << "gather/deserialize clamp: " << num_elems << " -> "
                          << input_control.remaining_count;
            num_elems = input_control.remaining_count;
          }
        }

        size_t clamp_src_bytes = num_elems * out_port->serdez_op->max_serialized_size;
        size_t src_bytes_avail;
        if(input_data_done) {
          // we're certainty to have all the remaining data, so keep
          //  the limit at max_src_bytes - we won't actually overshoot
          //  (unless the serialized data is corrupted)
          src_bytes_avail = max_src_bytes;
        } else {
          // test for space using our conserative bytes read count
          src_bytes_avail =
              in_port->seq_remote.span_exists(rbc_snapshot, clamp_src_bytes);

          if(src_bytes_avail == max_src_bytes) {
            // enough space - confirm the dest step
            out_port->iter->confirm_step();
          } else {
            log_request.info() << "pred limits deserialize: " << max_src_bytes << " -> "
                               << src_bytes_avail;
            // not enough space - figure out how many elements we can
            //  actually read and adjust the dest step
            size_t act_elems = src_bytes_avail / out_port->serdez_op->max_serialized_size;
            // if there was a remainder in the division, get rid of it
            src_bytes_avail = act_elems * out_port->serdez_op->max_serialized_size;
            size_t new_dst_bytes = act_elems * out_port->serdez_op->sizeof_field_type;
            out_port->iter->cancel_step();
            dst_bytes = out_port->iter->step(new_dst_bytes, dst_info,
                                             flags & TransferIterator::SRC_FLAGMASK,
                                             false /*!tentative*/);
            // this can come up shorter than we expect if the destination
            //  iterator is 2-D or 3-D - if that happens, re-adjust the
            //  source bytes again
            if(dst_bytes < new_dst_bytes) {
              if(dst_bytes == 0)
                break;

              num_elems = dst_bytes / out_port->serdez_op->sizeof_field_type;
              assert((num_elems * out_port->serdez_op->sizeof_field_type) == dst_bytes);

              // no need to recheck seq_pre_write
              src_bytes_avail = num_elems * out_port->serdez_op->max_serialized_size;
            }
          }
        }

        // since the src_iter will be stepped later, the src_info is a
        //  don't care, so copy the source so that lines/planes/etc match
        //  up
        src_info = dst_info;

        read_seq = 0; // filled in later
        read_bytes = src_bytes_avail;
        in_port->local_bytes_cons.fetch_add(src_bytes_avail);
        rbc_snapshot += src_bytes_avail;

        write_seq = out_port->local_bytes_total;
        write_bytes = dst_bytes;
        out_port->local_bytes_total += dst_bytes;
        out_port->local_bytes_cons.store(
            out_port->local_bytes_total); // completion detection uses this
      } else {
        // either no serialization or simultaneous serdez

        // limit transfer based on the max request size, or the largest
        //  amount of data allowed by the control port(s)
        size_t max_bytes =
            std::min(size_t(max_req_size), std::min(input_control.remaining_count,
                                                    output_control.remaining_count));

        // if we're not the first in the chain, and we know the total bytes
        //  written by the predecessor, don't exceed that
        if(in_port->peer_guid != XFERDES_NO_GUID) {
          size_t pre_max = pbt_snapshot - in_port->local_bytes_total;
          if(pre_max == 0) {
            // should not happen with snapshots
            assert(0);
            // due to unsynchronized updates to pre_bytes_total, this path
            //  can happen for an empty transfer reading from an intermediate
            //  buffer - handle it by looping around and letting the check
            //  at the top of the loop notice it the second time around
            if(in_port->local_bytes_total == 0)
              continue;
            // otherwise, this shouldn't happen - we should detect this case
            //  on the the transfer of those last bytes
            assert(0);
            begin_completion();
            break;
          }
          if(pre_max < max_bytes) {
            log_request.info() << "pred limits xfer: " << max_bytes << " -> " << pre_max;
            max_bytes = pre_max;
          }

          // further limit by bytes we've actually received
          max_bytes =
              in_port->seq_remote.span_exists(in_port->local_bytes_total, max_bytes);
          if(max_bytes == 0) {
            // TODO: put this XD to sleep until we do have data
            break;
          }
        }

        if(out_port->peer_guid != XFERDES_NO_GUID) {
          // if we're writing to an intermediate buffer, make sure to not
          //  overwrite previously written data that has not been read yet
          max_bytes =
              out_port->seq_remote.span_exists(out_port->local_bytes_total, max_bytes);
          if(max_bytes == 0) {
            // TODO: put this XD to sleep until we do have data
            break;
          }
        }

        // tentatively get as much as we can from the source iterator
        size_t src_bytes = in_port->iter->step(max_bytes, src_info,
                                               flags & TransferIterator::SRC_FLAGMASK,
                                               true /*tentative*/);
        if(src_bytes == 0) {
          // not enough space for even one element
          // TODO: put this XD to sleep until we do have data
          break;
        }

        // destination step must be tentative for an non-IB source or
        //  target that might collapse dimensions differently
        bool dimension_mismatch_possible = (((in_port->peer_guid == XFERDES_NO_GUID) ||
                                             (out_port->peer_guid == XFERDES_NO_GUID)) &&
                                            ((flags & TransferIterator::LINES_OK) != 0));

        size_t dst_bytes = out_port->iter->step(src_bytes, dst_info,
                                                flags & TransferIterator::DST_FLAGMASK,
                                                dimension_mismatch_possible);
        if(dst_bytes == 0) {
          // not enough space for even one element

          // if this happens when the input is an IB, the output is not,
          //  and the input doesn't seem to be limited by max_bytes, this
          //  is (probably?) the case that requires padding on the input
          //  side
          if((in_port->peer_guid != XFERDES_NO_GUID) &&
             (out_port->peer_guid == XFERDES_NO_GUID) && (src_bytes < max_bytes)) {
            log_xd.info() << "padding input buffer by " << src_bytes << " bytes";
            src_info.bytes_per_chunk = 0;
            src_info.num_lines = 1;
            src_info.num_planes = 1;
            dst_info.bytes_per_chunk = 0;
            dst_info.num_lines = 1;
            dst_info.num_planes = 1;
            read_pad_bytes = src_bytes;
            src_bytes = 0;
            dimension_mismatch_possible = false;
            // src iterator will be confirmed below
            // in_port->iter->confirm_step();
            // dst didn't actually take a step, so we don't need to cancel it
          } else {
            in_port->iter->cancel_step();
            // TODO: put this XD to sleep until we do have data
            break;
          }
        }

        // does source now need to be shrunk?
        if(dst_bytes < src_bytes) {
          // cancel the src step and try to just step by dst_bytes
          in_port->iter->cancel_step();
          // this step must still be tentative if a dimension mismatch is
          //  posisble
          src_bytes = in_port->iter->step(dst_bytes, src_info,
                                          flags & TransferIterator::SRC_FLAGMASK,
                                          dimension_mismatch_possible);
          if(src_bytes == 0) {
            // corner case that should occur only with a destination
            //  intermediate buffer - no transfer, but pad to boundary
            //  destination wants as long as we're not being limited by
            //  max_bytes
            assert((in_port->peer_guid == XFERDES_NO_GUID) &&
                   (out_port->peer_guid != XFERDES_NO_GUID));
            if(dst_bytes < max_bytes) {
              log_xd.info() << "padding output buffer by " << dst_bytes << " bytes";
              src_info.bytes_per_chunk = 0;
              src_info.num_lines = 1;
              src_info.num_planes = 1;
              dst_info.bytes_per_chunk = 0;
              dst_info.num_lines = 1;
              dst_info.num_planes = 1;
              write_pad_bytes = dst_bytes;
              dst_bytes = 0;
              dimension_mismatch_possible = false;
              // src didn't actually take a step, so we don't need to cancel it
              out_port->iter->confirm_step();
            } else {
              // retry later
              // src didn't actually take a step, so we don't need to cancel it
              out_port->iter->cancel_step();
              break;
            }
          }
          // a mismatch is still possible if the source is 2+D and the
          //  destination wants to stop mid-span
          if(src_bytes < dst_bytes) {
            assert(dimension_mismatch_possible);
            out_port->iter->cancel_step();
            dst_bytes = out_port->iter->step(src_bytes, dst_info,
                                             flags & TransferIterator::DST_FLAGMASK,
                                             true /*tentative*/);
          }
          // byte counts now must match
          assert(src_bytes == dst_bytes);
        } else {
          // in the absense of dimension mismatches, it's safe now to confirm
          //  the source step
          if(!dimension_mismatch_possible)
            in_port->iter->confirm_step();
        }

        // when 2D transfers are allowed, it is possible that the
        // bytes_per_chunk don't match, and we need to add an extra
        //  dimension to one side or the other
        // NOTE: this transformation can cause the dimensionality of the
        //  transfer to grow.  Allow this to happen and detect it at the
        //  end.
        if(!dimension_mismatch_possible) {
          assert(src_info.bytes_per_chunk == dst_info.bytes_per_chunk);
          assert(src_info.num_lines == 1);
          assert(src_info.num_planes == 1);
          assert(dst_info.num_lines == 1);
          assert(dst_info.num_planes == 1);
        } else {
          // track how much of src and/or dst is "lost" into a 4th
          //  dimension
          size_t src_4d_factor = 1;
          size_t dst_4d_factor = 1;
          if(src_info.bytes_per_chunk < dst_info.bytes_per_chunk) {
            size_t ratio = dst_info.bytes_per_chunk / src_info.bytes_per_chunk;
            assert((src_info.bytes_per_chunk * ratio) == dst_info.bytes_per_chunk);
            dst_4d_factor *= dst_info.num_planes; // existing planes lost
            dst_info.num_planes = dst_info.num_lines;
            dst_info.plane_stride = dst_info.line_stride;
            dst_info.num_lines = ratio;
            dst_info.line_stride = src_info.bytes_per_chunk;
            dst_info.bytes_per_chunk = src_info.bytes_per_chunk;
          }
          if(dst_info.bytes_per_chunk < src_info.bytes_per_chunk) {
            size_t ratio = src_info.bytes_per_chunk / dst_info.bytes_per_chunk;
            assert((dst_info.bytes_per_chunk * ratio) == src_info.bytes_per_chunk);
            src_4d_factor *= src_info.num_planes; // existing planes lost
            src_info.num_planes = src_info.num_lines;
            src_info.plane_stride = src_info.line_stride;
            src_info.num_lines = ratio;
            src_info.line_stride = dst_info.bytes_per_chunk;
            src_info.bytes_per_chunk = dst_info.bytes_per_chunk;
          }

          // similarly, if the number of lines doesn't match, we need to promote
          //  one of the requests from 2D to 3D
          if(src_info.num_lines < dst_info.num_lines) {
            size_t ratio = dst_info.num_lines / src_info.num_lines;
            assert((src_info.num_lines * ratio) == dst_info.num_lines);
            dst_4d_factor *= dst_info.num_planes; // existing planes lost
            dst_info.num_planes = ratio;
            dst_info.plane_stride = dst_info.line_stride * src_info.num_lines;
            dst_info.num_lines = src_info.num_lines;
          }
          if(dst_info.num_lines < src_info.num_lines) {
            size_t ratio = src_info.num_lines / dst_info.num_lines;
            assert((dst_info.num_lines * ratio) == src_info.num_lines);
            src_4d_factor *= src_info.num_planes; // existing planes lost
            src_info.num_planes = ratio;
            src_info.plane_stride = src_info.line_stride * dst_info.num_lines;
            src_info.num_lines = dst_info.num_lines;
          }

          // sanity-checks: src/dst should match on lines/planes and we
          //  shouldn't have multiple planes if we don't have multiple lines
          assert(src_info.num_lines == dst_info.num_lines);
          assert((src_info.num_planes * src_4d_factor) ==
                 (dst_info.num_planes * dst_4d_factor));
          assert((src_info.num_lines > 1) || (src_info.num_planes == 1));
          assert((dst_info.num_lines > 1) || (dst_info.num_planes == 1));

          // only do as many planes as both src and dst can manage
          if(src_info.num_planes > dst_info.num_planes)
            src_info.num_planes = dst_info.num_planes;
          else
            dst_info.num_planes = src_info.num_planes;

          // if 3D isn't allowed, set num_planes back to 1
          if((flags & TransferIterator::PLANES_OK) == 0) {
            src_info.num_planes = 1;
            dst_info.num_planes = 1;
          }

          // now figure out how many bytes we're actually able to move and
          //  if it's less than what we got from the iterators, try again
          size_t act_bytes =
              (src_info.bytes_per_chunk * src_info.num_lines * src_info.num_planes);
          if(act_bytes == src_bytes) {
            // things match up - confirm the steps
            in_port->iter->confirm_step();
            out_port->iter->confirm_step();
          } else {
            // log_request.info() << "dimension mismatch! " << act_bytes << " < " <<
            // src_bytes << " (" << bytes_total << ")";
            TransferIterator::AddressInfo dummy_info;
            in_port->iter->cancel_step();
            src_bytes = in_port->iter->step(act_bytes, dummy_info,
                                            flags & TransferIterator::SRC_FLAGMASK,
                                            false /*!tentative*/);
            assert(src_bytes == act_bytes);
            out_port->iter->cancel_step();
            dst_bytes = out_port->iter->step(act_bytes, dummy_info,
                                             flags & TransferIterator::DST_FLAGMASK,
                                             false /*!tentative*/);
            assert(dst_bytes == act_bytes);
          }
        }

        size_t act_bytes =
            (src_info.bytes_per_chunk * src_info.num_lines * src_info.num_planes);
        read_seq = in_port->local_bytes_total;
        read_bytes = act_bytes + read_pad_bytes;

        // update bytes read unless we're using indirection
        if(in_port->indirect_port_idx < 0)
          in_port->local_bytes_total += read_bytes;

        write_seq = out_port->local_bytes_total;
        write_bytes = act_bytes + write_pad_bytes;
        out_port->local_bytes_total += write_bytes;
        out_port->local_bytes_cons.store(
            out_port->local_bytes_total); // completion detection uses this
      }

      Request *new_req = dequeue_request();
      new_req->src_port_idx = input_control.current_io_port;
      new_req->dst_port_idx = output_control.current_io_port;
      new_req->read_seq_pos = read_seq;
      new_req->read_seq_count = read_bytes;
      new_req->write_seq_pos = write_seq;
      new_req->write_seq_count = write_bytes;
      new_req->dim =
          ((src_info.num_planes == 1)
               ? ((src_info.num_lines == 1) ? Request::DIM_1D : Request::DIM_2D)
               : Request::DIM_3D);
      new_req->src_off = src_info.base_offset;
      new_req->dst_off = dst_info.base_offset;
      new_req->nbytes = src_info.bytes_per_chunk;
      new_req->nlines = src_info.num_lines;
      new_req->src_str = src_info.line_stride;
      new_req->dst_str = dst_info.line_stride;
      new_req->nplanes = src_info.num_planes;
      new_req->src_pstr = src_info.plane_stride;
      new_req->dst_pstr = dst_info.plane_stride;

      // we can actually hit the end of an intermediate buffer input
      //  even if our initial pbt_snapshot was (size_t)-1 because
      //  we use the asynchronously-updated seq_pre_write, so if
      //  we think we might be done, go ahead and resample here if
      //  we still have -1
      if((in_port->peer_guid != XFERDES_NO_GUID) && (pbt_snapshot == (size_t)-1))
        pbt_snapshot = in_port->remote_bytes_total.load_acquire();

      // if we have control ports, they tell us when we're done
      if((input_control.control_port_idx >= 0) ||
         (output_control.control_port_idx >= 0)) {
        // update control port counts, which may also flag a completed iteration
        size_t input_count = read_bytes - read_pad_bytes;
        size_t output_count = write_bytes - write_pad_bytes;
        // if we're serializing or deserializing, we count in elements,
        //  not bytes
        if(in_port->serdez_op != 0) {
          // serializing impacts output size
          assert((output_count % in_port->serdez_op->max_serialized_size) == 0);
          output_count /= in_port->serdez_op->max_serialized_size;
        }
        if(out_port->serdez_op != 0) {
          // and deserializing impacts input size
          assert((input_count % out_port->serdez_op->max_serialized_size) == 0);
          input_count /= out_port->serdez_op->max_serialized_size;
        }
        assert(input_control.remaining_count >= input_count);
        assert(output_control.remaining_count >= output_count);
        input_control.remaining_count -= input_count;
        output_control.remaining_count -= output_count;
        if(((input_control.remaining_count == 0) && input_control.eos_received) ||
           ((output_control.remaining_count == 0) && output_control.eos_received)) {
          log_xd.info() << "iteration completed via control port: xd=" << std::hex << guid
                        << std::dec;
          begin_completion();
        }
      } else {
        // otherwise, we go by our iterators
        if(in_port->iter->done() || out_port->iter->done() ||
           (in_port->local_bytes_total == pbt_snapshot)) {
          assert(!iteration_completed.load());
          begin_completion();

          // TODO: figure out how to eliminate false positives from these
          //  checks with indirection and/or multiple remote inputs
#if 0
	      // non-ib iterators should end at the same time
	      assert((in_port->peer_guid != XFERDES_NO_GUID) || in_port->iter->done());
	      assert((out_port->peer_guid != XFERDES_NO_GUID) || out_port->iter->done());
#endif

          if(!in_port->serdez_op && out_port->serdez_op) {
            // ok to be over, due to the conservative nature of
            //  deserialization reads
            assert((rbc_snapshot >= pbt_snapshot) || (pbt_snapshot == size_t(-1)));
          } else {
            // TODO: this check is now too aggressive because the previous
            //  xd doesn't necessarily know when it's emitting its last
            //  data, which means the update of local_bytes_total might
            //  be delayed
#if 0
		assert((in_port->peer_guid == XFERDES_NO_GUID) ||
		       (pbt_snapshot == in_port->local_bytes_total));
#endif
          }
        }
      }

      switch(new_req->dim) {
      case Request::DIM_1D:
      {
        log_request.info() << "request: guid=" << std::hex << guid << std::dec
                           << " ofs=" << new_req->src_off << "->" << new_req->dst_off
                           << " len=" << new_req->nbytes;
        break;
      }
      case Request::DIM_2D:
      {
        log_request.info() << "request: guid=" << std::hex << guid << std::dec
                           << " ofs=" << new_req->src_off << "->" << new_req->dst_off
                           << " len=" << new_req->nbytes << " lines=" << new_req->nlines
                           << "(" << new_req->src_str << "," << new_req->dst_str << ")";
        break;
      }
      case Request::DIM_3D:
      {
        log_request.info() << "request: guid=" << std::hex << guid << std::dec
                           << " ofs=" << new_req->src_off << "->" << new_req->dst_off
                           << " len=" << new_req->nbytes << " lines=" << new_req->nlines
                           << "(" << new_req->src_str << "," << new_req->dst_str << ")"
                           << " planes=" << new_req->nplanes << "(" << new_req->src_pstr
                           << "," << new_req->dst_pstr << ")";
        break;
      }
      }
      reqs[idx++] = new_req;
    }
    return idx;
  }

  void XferDes::begin_completion()
  {
#ifdef DEBUG_REALM
    // shouldn't be called more than once
    assert(!iteration_completed.load());
#endif
    iteration_completed.store_release(true);

    // give all output channels a chance to indicate completion and determine
    //  the total number of bytes we've written
    size_t total_bytes_written = 0;
    for(size_t i = 0; i < output_ports.size(); i++) {
      total_bytes_written += output_ports[i].local_bytes_cons.load();
      update_bytes_write(i, output_ports[i].local_bytes_total, 0);

      // see if we still need to send the total bytes
      if(output_ports[i].needs_pbt_update.load() &&
         (output_ports[i].local_bytes_total == output_ports[i].local_bytes_cons.load())) {
#ifdef DEBUG_REALM
        assert(output_ports[i].peer_guid != XFERDES_NO_GUID);
#endif
        // exchange sets the flag to false and tells us previous value
        if(output_ports[i].needs_pbt_update.exchange(false))
          xferDes_queue->update_pre_bytes_total(output_ports[i].peer_guid,
                                                output_ports[i].peer_port_idx,
                                                output_ports[i].local_bytes_total);
      }
    }

    // bytes pending is total minus however many writes have already
    //  finished - if that's all of them, we can mark full transfer completion
    int64_t prev = bytes_write_pending.fetch_add(total_bytes_written);
    int64_t pending = prev + total_bytes_written;
    log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                  << " total_bytes=" << total_bytes_written << " pending=" << pending;
    assert(pending >= 0);
    if(pending == 0)
      transfer_completed.store_release(true);
  }

  void XferDes::update_bytes_read(int port_idx, size_t offset, size_t size)
  {
    XferPort *in_port = &input_ports[port_idx];
    size_t inc_amt = in_port->seq_local.add_span(offset, size);
    log_xd.info() << "bytes_read: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") " << offset << "+" << size << " -> " << inc_amt;
    if(in_port->peer_guid != XFERDES_NO_GUID) {
      if(inc_amt > 0) {
        // we're actually telling the previous XD which offsets are ok to
        //  overwrite, so adjust our offset by our (circular) IB size
        xferDes_queue->update_next_bytes_read(in_port->peer_guid, in_port->peer_port_idx,
                                              offset + in_port->ib_size, inc_amt);
      } else {
        // TODO: mode to send non-contiguous updates?
      }
    }
  }

  void XferDes::update_bytes_write(int port_idx, size_t offset, size_t size)
  {
    XferPort *out_port = &output_ports[port_idx];
    size_t inc_amt = out_port->seq_local.add_span(offset, size);
    log_xd.info() << "bytes_write: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") " << offset << "+" << size << " -> " << inc_amt;

    if(out_port->peer_guid != XFERDES_NO_GUID) {
      // update bytes total if needed (and available)
      if(out_port->needs_pbt_update.load() && iteration_completed.load_acquire() &&
         (out_port->local_bytes_total == out_port->local_bytes_cons.load())) {
        // exchange sets the flag to false and tells us previous value
        if(out_port->needs_pbt_update.exchange(false))
          xferDes_queue->update_pre_bytes_total(
              out_port->peer_guid, out_port->peer_port_idx, out_port->local_bytes_total);
      }
      // we can skip an update if this was empty
      if(inc_amt > 0) {
        xferDes_queue->update_pre_bytes_write(out_port->peer_guid,
                                              out_port->peer_port_idx, offset, inc_amt);
      } else {
        // TODO: mode to send non-contiguous updates?
      }
    }

    // subtract bytes written from the pending count - if that causes it to
    //  go to zero, we can mark the transfer completed and update progress
    //  in case the xd is just waiting for that
    // NOTE: as soon as we set `transfer_completed`, the other references
    //  to this xd may be removed, so do this last, and hold a reference of
    //  our own long enough to call update_progress
    if(inc_amt > 0) {
      int64_t prev = bytes_write_pending.fetch_sub(inc_amt);
      if(prev > 0)
        log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                      << " remaining=" << (prev - inc_amt);
      if(inc_amt == static_cast<size_t>(prev)) {
        add_reference();
        transfer_completed.store_release(true);
        update_progress();
        remove_reference();
      }
    }
  }

  void XferDes::update_pre_bytes_write(int port_idx, size_t offset, size_t size)
  {
    XferPort *in_port = &input_ports[port_idx];

    size_t inc_amt = in_port->seq_remote.add_span(offset, size);
    log_xd.info() << "pre_write: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") " << offset << "+" << size << " -> " << inc_amt << " ("
                  << in_port->remote_bytes_total.load() << ")";
    // if we got new data at the current pointer OR if we now know the
    //  total incoming bytes, update progress
    if(inc_amt > 0)
      update_progress();
  }

  void XferDes::update_pre_bytes_total(int port_idx, size_t pre_bytes_total)
  {
    XferPort *in_port = &input_ports[port_idx];

    // should always be exchanging -1 -> (not -1)
#ifdef DEBUG_REALM
    size_t oldval =
#endif
        in_port->remote_bytes_total.exchange(pre_bytes_total);
#ifdef DEBUG_REALM
    assert((oldval == size_t(-1)) && (pre_bytes_total != size_t(-1)));
#endif
    log_xd.info() << "pre_total: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") = " << pre_bytes_total;
    // this may unblock an xd that has consumed all input but didn't
    //  realize there was no more
    update_progress();
  }

  void XferDes::update_next_bytes_read(int port_idx, size_t offset, size_t size)
  {
    XferPort *out_port = &output_ports[port_idx];

    size_t inc_amt = out_port->seq_remote.add_span(offset, size);
    log_xd.info() << "next_read: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") " << offset << "+" << size << " -> " << inc_amt;
    // if we got new room at the current pointer (and we're still
    //  iterating), update progress
    if((inc_amt > 0) && !iteration_completed.load())
      update_progress();
  }

  void XferDes::notify_request_read_done(Request *req)
  {
    default_notify_request_read_done(req);
  }

  void XferDes::notify_request_write_done(Request *req)
  {
    default_notify_request_write_done(req);
  }

  void XferDes::flush() {}

  void XferDes::default_notify_request_read_done(Request *req)
  {
    req->is_read_done = true;
    update_bytes_read(req->src_port_idx, req->read_seq_pos, req->read_seq_count);
  }

  void XferDes::default_notify_request_write_done(Request *req)
  {
    req->is_write_done = true;
    // calling update_bytes_write can cause the transfer descriptor to
    //  be destroyed, so enqueue the request first, and cache the values
    //  we need
    int dst_port_idx = req->dst_port_idx;
    size_t write_seq_pos = req->write_seq_pos;
    size_t write_seq_count = req->write_seq_count;
    update_bytes_write(dst_port_idx, write_seq_pos, write_seq_count);
    enqueue_request(req);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillXferDes
  //

  MemfillXferDes::MemfillXferDes(uintptr_t _dma_op, Channel *_channel,
                                 NodeID _launch_node, XferDesID _guid,
                                 const std::vector<XferDesPortInfo> &inputs_info,
                                 const std::vector<XferDesPortInfo> &outputs_info,
                                 int _priority, const void *_fill_data, size_t _fill_size,
                                 size_t _fill_total)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, _fill_data, _fill_size)
  {
    kind = XFER_MEM_FILL;

    // no direct input data for us, but we know how much data to produce
    //  (in case the output is an intermediate buffer)
    assert(input_control.control_port_idx == -1);
    input_control.current_io_port = -1;
    input_control.remaining_count = _fill_total;
    input_control.eos_received = true;
  }

  long MemfillXferDes::get_requests(Request **requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }

  bool MemfillXferDes::request_available()
  {
    // unused
    assert(0);
    return false;
  }

  Request *MemfillXferDes::dequeue_request()
  {
    // unused
    assert(0);
    return 0;
  }

  void MemfillXferDes::enqueue_request(Request *req)
  {
    // unused
    assert(0);
  }

  bool MemfillXferDes::progress_xd(MemfillChannel *channel, TimeLimit work_until)
  {
    bool did_work = false;
    ReadSequenceCache rseqcache(this, 2 << 20);
    WriteSequenceCache wseqcache(this, 2 << 20);

    while(true) {
      size_t min_xfer_size = 4096; // TODO: make controllable
      size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
      if(max_bytes == 0)
        break;

      XferPort *out_port = 0;
      size_t out_span_start = 0;
      if(output_control.current_io_port >= 0) {
        out_port = &output_ports[output_control.current_io_port];
        out_span_start = out_port->local_bytes_total;
      }

      size_t total_bytes = 0;
      if(out_port != 0) {
        // input and output both exist - transfer what we can
        log_xd.info() << "memfill chunk: min=" << min_xfer_size << " max=" << max_bytes;

        uintptr_t out_base =
            reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

        while(total_bytes < max_bytes) {
          AddressListCursor &out_alc = out_port->addrcursor;

          uintptr_t out_offset = out_alc.get_offset();

          // the reported dim is reduced for partially consumed address
          //  ranges - whatever we get can be assumed to be regular
          int out_dim = out_alc.get_dim();

          size_t bytes = 0;
          size_t bytes_left = max_bytes - total_bytes;
          // memfills don't need to be particularly big to achieve
          //  peak efficiency, so trim to something that takes
          //  10's of us to be responsive to the time limit
          // NOTE: have to be a little careful and make sure the limit
          //  is a multiple of the fill size - we'll make it a power-of-2
          const size_t TARGET_CHUNK_SIZE = 256 << 10; // 256KB
          if(bytes_left > TARGET_CHUNK_SIZE) {
            size_t max_chunk = fill_size;
            while(max_chunk < TARGET_CHUNK_SIZE)
              max_chunk <<= 1;
            bytes_left = std::min(bytes_left, max_chunk);
          }

          if(out_dim > 0) {
            size_t ocount = out_alc.remaining(0);

            // contig bytes is always the first dimension
            size_t contig_bytes = std::min(ocount, bytes_left);

            // catch simple 1D case first
            if((contig_bytes == bytes_left) ||
               ((contig_bytes == ocount) && (out_dim == 1))) {
              bytes = contig_bytes;
              memset_1d(out_base + out_offset, contig_bytes, fill_data, fill_size);
              out_alc.advance(0, bytes);
            } else {
              // grow to a 2D fill
              ocount = out_alc.remaining(1);
              uintptr_t out_lstride = out_alc.get_stride(1);

              size_t lines = std::min(ocount, bytes_left / contig_bytes);

              bytes = contig_bytes * lines;
              memset_2d(out_base + out_offset, out_lstride, contig_bytes, lines,
                        fill_data, fill_size);
              out_alc.advance(1, lines);
            }
          } else {
            // scatter adddress list
            assert(0);
          }

#ifdef DEBUG_REALM
          assert(bytes <= bytes_left);
#endif
          total_bytes += bytes;

          // stop if it's been too long, but make sure we do at least the
          //  minimum number of bytes
          if((total_bytes >= min_xfer_size) && work_until.is_expired())
            break;
        }
      } else {
        // fill with no output, so just count the bytes
        total_bytes = max_bytes;
      }

      // mem fill is always immediate, so handle both skip and copy with
      //  the same code
      wseqcache.add_span(output_control.current_io_port, out_span_start, total_bytes);
      out_span_start += total_bytes;

      bool done = record_address_consumption(total_bytes, total_bytes);

      did_work = true;

      if(done || work_until.is_expired())
        break;
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillXferDes
  //

  MemreduceXferDes::MemreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                                     NodeID _launch_node, XferDesID _guid,
                                     const std::vector<XferDesPortInfo> &inputs_info,
                                     const std::vector<XferDesPortInfo> &outputs_info,
                                     int _priority, XferDesRedopInfo _redop_info)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, 0, 0)
    , redop_info(_redop_info)
  {
    kind = XFER_MEM_CPY;
    redop = get_runtime()->reduce_op_table.get(redop_info.id, 0);
    assert(redop);
  }

  long MemreduceXferDes::get_requests(Request **requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }

  bool MemreduceXferDes::progress_xd(MemreduceChannel *channel, TimeLimit work_until)
  {
    bool did_work = false;
    ReadSequenceCache rseqcache(this, 2 << 20); // flush after 2MB
    WriteSequenceCache wseqcache(this, 2 << 20);

    const size_t in_elem_size = redop->sizeof_rhs;
    const size_t out_elem_size =
        (redop_info.is_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
    assert(redop_info.in_place); // TODO: support for out-of-place reduces

    while(true) {
      size_t min_xfer_size = 4096; // TODO: make controllable
      size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
      if(max_bytes == 0)
        break;

      XferPort *in_port = 0, *out_port = 0;
      size_t in_span_start = 0, out_span_start = 0;
      if(input_control.current_io_port >= 0) {
        in_port = &input_ports[input_control.current_io_port];
        in_span_start = in_port->local_bytes_total;
      }
      if(output_control.current_io_port >= 0) {
        out_port = &output_ports[output_control.current_io_port];
        out_span_start = out_port->local_bytes_total;
      }

      // have to count in terms of elements, which requires redoing some math
      //  if in/out sizes do not match
      size_t max_elems;
      if(in_elem_size == out_elem_size) {
        max_elems = max_bytes / in_elem_size;
      } else {
        max_elems = std::min(input_control.remaining_count / in_elem_size,
                             output_control.remaining_count / out_elem_size);
        if(in_port != 0) {
          max_elems =
              std::min(max_elems, in_port->addrlist.bytes_pending() / in_elem_size);
          if(in_port->peer_guid != XFERDES_NO_GUID) {
            size_t read_bytes_avail = in_port->seq_remote.span_exists(
                in_port->local_bytes_total, (max_elems * in_elem_size));
            max_elems = std::min(max_elems, (read_bytes_avail / in_elem_size));
          }
        }
        if(out_port != 0) {
          max_elems =
              std::min(max_elems, out_port->addrlist.bytes_pending() / out_elem_size);
          // no support for reducing into an intermediate buffer
          assert(out_port->peer_guid == XFERDES_NO_GUID);
        }
      }

      size_t total_elems = 0;
      if(in_port != 0) {
        if(out_port != 0) {
          // input and output both exist - transfer what we can
          log_xd.info() << "memreduce chunk: min=" << min_xfer_size
                        << " max_elems=" << max_elems;

          uintptr_t in_base =
              reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
          uintptr_t out_base =
              reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

          while(total_elems < max_elems) {
            AddressListCursor &in_alc = in_port->addrcursor;
            AddressListCursor &out_alc = out_port->addrcursor;

            uintptr_t in_offset = in_alc.get_offset();
            uintptr_t out_offset = out_alc.get_offset();

            // the reported dim is reduced for partially consumed address
            //  ranges - whatever we get can be assumed to be regular
            int in_dim = in_alc.get_dim();
            int out_dim = out_alc.get_dim();

            // the current reduction op interface can reduce multiple elements
            //  with a fixed address stride, which looks to us like either
            //  1D (stride = elem_size), or 2D with 1 elem/line

            size_t icount = in_alc.remaining(0) / in_elem_size;
            size_t ocount = out_alc.remaining(0) / out_elem_size;
            size_t istride, ostride;
            if((in_dim > 1) && (icount == 1)) {
              in_dim = 2;
              icount = in_alc.remaining(1);
              istride = in_alc.get_stride(1);
            } else {
              in_dim = 1;
              istride = in_elem_size;
            }
            if((out_dim > 1) && (ocount == 1)) {
              out_dim = 2;
              ocount = out_alc.remaining(1);
              ostride = out_alc.get_stride(1);
            } else {
              out_dim = 1;
              ostride = out_elem_size;
            }

            size_t elems_left = max_elems - total_elems;
            size_t elems = std::min(std::min(icount, ocount), elems_left);
            assert(elems > 0);

            void *out_ptr = reinterpret_cast<void *>(out_base + out_offset);
            const void *in_ptr = reinterpret_cast<const void *>(in_base + in_offset);
            if(redop_info.is_fold) {
              if(redop_info.is_exclusive)
                (redop->cpu_fold_excl_fn)(out_ptr, ostride, in_ptr, istride, elems,
                                          redop->userdata);
              else
                (redop->cpu_fold_nonexcl_fn)(out_ptr, ostride, in_ptr, istride, elems,
                                             redop->userdata);
            } else {
              if(redop_info.is_exclusive)
                (redop->cpu_apply_excl_fn)(out_ptr, ostride, in_ptr, istride, elems,
                                           redop->userdata);
              else
                (redop->cpu_apply_nonexcl_fn)(out_ptr, ostride, in_ptr, istride, elems,
                                              redop->userdata);
            }

            in_alc.advance(in_dim - 1, elems * ((in_dim == 1) ? in_elem_size : 1));
            out_alc.advance(out_dim - 1, elems * ((out_dim == 1) ? out_elem_size : 1));

#ifdef DEBUG_REALM
            assert(elems <= elems_left);
#endif
            total_elems += elems;

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if(((total_elems * in_elem_size) >= min_xfer_size) && work_until.is_expired())
              break;
          }
        } else {
          // input but no output, so skip input bytes
          total_elems = max_elems;
          in_port->addrcursor.skip_bytes(total_elems * in_elem_size);
        }
      } else {
        if(out_port != 0) {
          // output but no input, so skip output bytes
          total_elems = max_elems;
          out_port->addrcursor.skip_bytes(total_elems * out_elem_size);
        } else {
          // skipping both input and output is possible for simultaneous
          //  gather+scatter
          total_elems = max_elems;
        }
      }

      // memcpy is always immediate, so handle both skip and copy with the
      //  same code
      rseqcache.add_span(input_control.current_io_port, in_span_start,
                         total_elems * in_elem_size);
      in_span_start += total_elems * in_elem_size;
      wseqcache.add_span(output_control.current_io_port, out_span_start,
                         total_elems * out_elem_size);
      out_span_start += total_elems * out_elem_size;

      bool done = record_address_consumption(total_elems * in_elem_size,
                                             total_elems * out_elem_size);

      did_work = true;

      if(done || work_until.is_expired())
        break;
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }

  GASNetXferDes::GASNetXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                               XferDesID _guid,
                               const std::vector<XferDesPortInfo> &inputs_info,
                               const std::vector<XferDesPortInfo> &outputs_info,
                               int _priority)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, 0, 0)
  {
    if((inputs_info.size() >= 1) &&
       (input_ports[0].mem->kind == MemoryImpl::MKIND_GLOBAL)) {
      kind = XFER_GASNET_READ;
    } else if((outputs_info.size() >= 1) &&
              (output_ports[0].mem->kind == MemoryImpl::MKIND_GLOBAL)) {
      kind = XFER_GASNET_WRITE;
    } else {
      assert(0 && "neither source nor dest of GASNetXferDes is gasnet!?");
    }
    const int max_nr = 10; // FIXME
    gasnet_reqs = (GASNetRequest *)calloc(max_nr, sizeof(GASNetRequest));
    for(int i = 0; i < max_nr; i++) {
      gasnet_reqs[i].xd = this;
      available_reqs.push(&gasnet_reqs[i]);
    }
  }

  long GASNetXferDes::get_requests(Request **requests, long nr)
  {
    GASNetRequest **reqs = (GASNetRequest **)requests;
    long new_nr = default_get_requests(requests, nr);
    switch(kind) {
    case XFER_GASNET_READ:
    {
      for(long i = 0; i < new_nr; i++) {
        reqs[i]->gas_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
        // reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
        reqs[i]->mem_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(
            reqs[i]->dst_off, reqs[i]->nbytes);
        assert(reqs[i]->mem_base != 0);
      }
      break;
    }
    case XFER_GASNET_WRITE:
    {
      for(long i = 0; i < new_nr; i++) {
        // reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
        reqs[i]->mem_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(
            reqs[i]->src_off, reqs[i]->nbytes);
        assert(reqs[i]->mem_base != 0);
        reqs[i]->gas_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
      }
      break;
    }
    default:
      assert(0);
    }
    return new_nr;
  }

  bool GASNetXferDes::progress_xd(GASNetChannel *channel, TimeLimit work_until)
  {
    Request *rq;
    bool did_work = false;
    do {
      long count = get_requests(&rq, 1);
      if(count > 0) {
        channel->submit(&rq, count);
        did_work = true;
      } else
        break;
    } while(!work_until.is_expired());

    return did_work;
  }

  void GASNetXferDes::notify_request_read_done(Request *req)
  {
    default_notify_request_read_done(req);
  }

  void GASNetXferDes::notify_request_write_done(Request *req)
  {
    default_notify_request_write_done(req);
  }

  void GASNetXferDes::flush() {}

  RemoteWriteXferDes::RemoteWriteXferDes(uintptr_t _dma_op, Channel *_channel,
                                         NodeID _launch_node, XferDesID _guid,
                                         const std::vector<XferDesPortInfo> &inputs_info,
                                         const std::vector<XferDesPortInfo> &outputs_info,
                                         int _priority)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, 0, 0)
  {
    kind = XFER_REMOTE_WRITE;
    requests = 0;
  }

  long RemoteWriteXferDes::get_requests(Request **requests, long nr)
  {
    xd_lock.lock();
    RemoteWriteRequest **reqs = (RemoteWriteRequest **)requests;
    // remote writes allow 2D on source, but not destination
    unsigned flags = TransferIterator::SRC_LINES_OK;
    long new_nr = default_get_requests(requests, nr, flags);
    for(long i = 0; i < new_nr; i++) {
      // reqs[i]->src_base = (char*)(src_buf_base + reqs[i]->src_off);
      reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(
          reqs[i]->src_off, reqs[i]->nbytes);
      assert(reqs[i]->src_base != 0);
      // RemoteMemory *remote = checked_cast<RemoteMemory
      // *>(output_ports[reqs[i]->dst_port_idx].mem); reqs[i]->dst_base =
      // static_cast<char *>(remote->get_remote_addr(reqs[i]->dst_off));
      // assert(reqs[i]->dst_base != 0);
    }
    xd_lock.unlock();
    return new_nr;
  }

  // callbacks for updating read/write spans
  class ReadBytesUpdater {
  public:
    ReadBytesUpdater(XferDes *_xd, int _port_idx, size_t _offset, size_t _size)
      : xd(_xd)
      , port_idx(_port_idx)
      , offset(_offset)
      , size(_size)
    {}

    void operator()() const
    {
      xd->update_bytes_read(port_idx, offset, size);
      xd->remove_reference();
    }

  protected:
    XferDes *xd;
    int port_idx;
    size_t offset, size;
  };

  class WriteBytesUpdater {
  public:
    WriteBytesUpdater(XferDes *_xd, int _port_idx, size_t _offset, size_t _size)
      : xd(_xd)
      , port_idx(_port_idx)
      , offset(_offset)
      , size(_size)
    {}

    void operator()() const { xd->update_bytes_write(port_idx, offset, size); }

  protected:
    XferDes *xd;
    int port_idx;
    size_t offset, size;
  };

  bool RemoteWriteXferDes::progress_xd(RemoteWriteChannel *channel, TimeLimit work_until)
  {
    bool did_work = false;
    // immediate acks for reads happen when we assemble or skip input,
    //  while immediate acks for writes happen only if we skip output
    ReadSequenceCache rseqcache(this);
    WriteSequenceCache wseqcache(this);

    const size_t MAX_ASSEMBLY_SIZE = 4096;
    while(true) {
      size_t min_xfer_size = 4096; // TODO: make controllable
      size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
      if(max_bytes == 0)
        break;

      XferPort *in_port = 0, *out_port = 0;
      size_t in_span_start = 0, out_span_start = 0;
      if(input_control.current_io_port >= 0) {
        in_port = &input_ports[input_control.current_io_port];
        in_span_start = in_port->local_bytes_total;
      }
      if(output_control.current_io_port >= 0) {
        out_port = &output_ports[output_control.current_io_port];
        out_span_start = out_port->local_bytes_total;
      }

      size_t total_bytes = 0;
      if(in_port != 0) {
        if(out_port != 0) {
          // input and output both exist - transfer what we can
          log_xd.info() << "remote write chunk: min=" << min_xfer_size
                        << " max=" << max_bytes;

          while(total_bytes < max_bytes) {
            AddressListCursor &in_alc = in_port->addrcursor;
            AddressListCursor &out_alc = out_port->addrcursor;
            int in_dim = in_alc.get_dim();
            int out_dim = out_alc.get_dim();
            size_t icount = in_alc.remaining(0);
            size_t ocount = out_alc.remaining(0);

            size_t bytes = 0;
            size_t bytes_left = max_bytes - total_bytes;

            // look at the output first, because that controls the message
            //  size
            size_t dst_1d_maxbytes = ((out_dim > 0) ? std::min(bytes_left, ocount) : 0);
            size_t dst_2d_maxbytes =
                (((out_dim > 1) && (ocount <= (MAX_ASSEMBLY_SIZE / 2)))
                     ? (ocount *
                        std::min(MAX_ASSEMBLY_SIZE / ocount, out_alc.remaining(1)))
                     : 0);
            // would have to scan forward through the dst address list to
            //  get the exact number of bytes that we can fit into
            //  MAX_ASSEMBLY_SIZE after considering address info overhead,
            //  but this is a last resort anyway, so just use a probably-
            //  pessimistic estimate;
            size_t dst_sc_maxbytes = std::min(bytes_left, MAX_ASSEMBLY_SIZE / 4);
            // TODO: actually implement 2d and sc
            dst_2d_maxbytes = 0;
            dst_sc_maxbytes = 0;

            // favor 1d >> 2d >> sc
            if((dst_1d_maxbytes >= dst_2d_maxbytes) &&
               (dst_1d_maxbytes >= dst_sc_maxbytes)) {
              // 1D target
              NodeID dst_node = ID(out_port->mem->me).memory_owner_node();
              RemoteAddress dst_buf;
              bool ok = out_port->mem->get_remote_addr(out_alc.get_offset(), dst_buf);
              assert(ok);

              // now look at the input
              LocalAddress src_buf;
              ok = in_port->mem->get_local_addr(in_alc.get_offset(), src_buf);
              assert(ok);
              size_t src_1d_maxbytes = 0;
              if(in_dim > 0) {
                size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(
                    dst_node, src_buf, icount, 1, 0, dst_buf, true /*w/ congestion*/);
                src_1d_maxbytes = std::min({dst_1d_maxbytes, icount, rec_bytes});
              }

              size_t src_2d_maxbytes = 0;
              // TODO: permit if source memory is cpu-accessible?
#ifdef ALLOW_RDMA_SOURCE_2D
              if(in_dim > 1) {
                size_t lines = in_alc.remaining(1);
                size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(
                    dst_node, src_buf, icount, lines, in_alc.get_stride(1), dst_buf,
                    true /*w/ congestion*/);
                // round the recommendation down to a multiple of the line size
                rec_bytes -= (rec_bytes % icount);
                src_2d_maxbytes = std::min({dst_1d_maxbytes, icount * lines, rec_bytes});
              }
#endif
              size_t src_ga_maxbytes = 0;
              // TODO: permit if source memory is cpu-accessible?
#ifdef ALLOW_RDMA_GATHER
              {
                // a gather will assemble into a buffer provided by the network
                size_t rec_bytes = ActiveMessage<Write1DMessage>::recommended_max_payload(
                    dst_node, dst_buf, true /*w/ congestion*/);
                src_ga_maxbytes = std::min({dst_1d_maxbytes, bytes_left, rec_bytes});
              }
#endif

              // source also favors 1d >> 2d >> gather
              if((src_1d_maxbytes >= src_2d_maxbytes) &&
                 (src_1d_maxbytes >= src_ga_maxbytes)) {
                // TODO: if congestion is telling us not to send anything
                //  at all, it'd be better to sleep until the network
                //  says it's reasonable to try again - this approach
                //  will effectively spinwait (but at least guarantees to
                //  intersperse it with calls to the network progress
                //  work item)
                if(src_1d_maxbytes == 0)
                  break;

                // 1D source
                bytes = src_1d_maxbytes;
                // log_xd.info() << "remote write 1d: guid=" << guid
                //              << " src=" << src_buf << " dst=" << dst_buf
                //              << " bytes=" << bytes;
                ActiveMessage<Write1DMessage> amsg(dst_node, src_buf, bytes, dst_buf);
                amsg->next_xd_guid = out_port->peer_guid;
                amsg->next_port_idx = out_port->peer_port_idx;
                amsg->span_start = out_span_start;

                // reads aren't consumed until local completion, but
                //  only ask if we have a previous xd that's going to
                //  care
                if(in_port->peer_guid != XFERDES_NO_GUID) {
                  // a ReadBytesUpdater holds a reference to the xd
                  add_reference();
                  amsg.add_local_completion(ReadBytesUpdater(
                      this, input_control.current_io_port, in_span_start, bytes));
                }
                in_span_start += bytes;
                // the write isn't complete until it's ack'd by the target
                amsg.add_remote_completion(WriteBytesUpdater(
                    this, output_control.current_io_port, out_span_start, bytes));
                out_span_start += bytes;

                amsg.commit();
                in_alc.advance(0, bytes);
                out_alc.advance(0, bytes);
              } else if(src_2d_maxbytes >= src_ga_maxbytes) {
                // 2D source
                size_t bytes_per_line = icount;
                size_t lines = src_2d_maxbytes / icount;
                bytes = bytes_per_line * lines;
                assert(bytes == src_2d_maxbytes);
                size_t src_stride = in_alc.get_stride(1);
                // log_xd.info() << "remote write 2d: guid=" << guid
                //              << " src=" << src_buf << " dst=" << dst_buf
                //              << " bytes=" << bytes << " lines=" << lines
                //              << " stride=" << src_stride;
                ActiveMessage<Write1DMessage> amsg(dst_node, src_buf, bytes_per_line,
                                                   lines, src_stride, dst_buf);
                amsg->next_xd_guid = out_port->peer_guid;
                amsg->next_port_idx = out_port->peer_port_idx;
                amsg->span_start = out_span_start;

                // reads aren't consumed until local completion, but
                //  only ask if we have a previous xd that's going to
                //  care
                if(in_port->peer_guid != XFERDES_NO_GUID) {
                  // a ReadBytesUpdater holds a reference to the xd
                  add_reference();
                  amsg.add_local_completion(ReadBytesUpdater(
                      this, input_control.current_io_port, in_span_start, bytes));
                }
                in_span_start += bytes;
                // the write isn't complete until it's ack'd by the target
                amsg.add_remote_completion(WriteBytesUpdater(
                    this, output_control.current_io_port, out_span_start, bytes));
                out_span_start += bytes;

                amsg.commit();
                in_alc.advance(1, lines);
                out_alc.advance(0, bytes);
              } else {
                // gather: assemble data
                bytes = src_ga_maxbytes;
                ActiveMessage<Write1DMessage> amsg(dst_node, bytes, dst_buf);
                amsg->next_xd_guid = out_port->peer_guid;
                amsg->next_port_idx = out_port->peer_port_idx;
                amsg->span_start = out_span_start;

                size_t todo = bytes;
                while(true) {
                  if(in_dim > 0) {
                    if((icount >= todo / 2) || (in_dim == 1)) {
                      size_t chunk = std::min(todo, icount);
                      uintptr_t src = reinterpret_cast<uintptr_t>(
                          in_port->mem->get_direct_ptr(in_alc.get_offset(), chunk));
                      uintptr_t dst =
                          reinterpret_cast<uintptr_t>(amsg.payload_ptr(chunk));
                      memcpy_1d(dst, src, chunk);
                      in_alc.advance(0, chunk);
                      todo -= chunk;
                    } else {
                      size_t lines = std::min(todo / icount, in_alc.remaining(1));

                      if(((icount * lines) >= todo / 2) || (in_dim == 2)) {
                        uintptr_t src = reinterpret_cast<uintptr_t>(
                            in_port->mem->get_direct_ptr(in_alc.get_offset(), icount));
                        uintptr_t dst =
                            reinterpret_cast<uintptr_t>(amsg.payload_ptr(icount * lines));
                        memcpy_2d(dst, icount /*lstride*/, src, in_alc.get_stride(1),
                                  icount, lines);
                        in_alc.advance(1, lines);
                        todo -= icount * lines;
                      } else {
                        size_t planes =
                            std::min(todo / (icount * lines), in_alc.remaining(2));
                        uintptr_t src = reinterpret_cast<uintptr_t>(
                            in_port->mem->get_direct_ptr(in_alc.get_offset(), icount));
                        uintptr_t dst = reinterpret_cast<uintptr_t>(
                            amsg.payload_ptr(icount * lines * planes));
                        memcpy_3d(dst, icount /*lstride*/, (icount * lines) /*pstride*/,
                                  src, in_alc.get_stride(1), in_alc.get_stride(2), icount,
                                  lines, planes);
                        in_alc.advance(2, planes);
                        todo -= icount * lines * planes;
                      }
                    }
                  } else {
                    assert(0);
                  }

                  if(todo == 0)
                    break;

                  // read next entry
                  in_dim = in_alc.get_dim();
                  icount = in_alc.remaining(0);
                }

                // the write isn't complete until it's ack'd by the target
                amsg.add_remote_completion(WriteBytesUpdater(
                    this, output_control.current_io_port, out_span_start, bytes));
                out_span_start += bytes;

                // assembly complete - send message
                amsg.commit();

                // we made a copy of input data, so "read" is complete
                rseqcache.add_span(input_control.current_io_port, in_span_start, bytes);
                in_span_start += bytes;

                out_alc.advance(0, bytes);
              }
            } else if(dst_2d_maxbytes >= dst_sc_maxbytes) {
              // 2D target
              assert(0);
            } else {
              // scatter target
              assert(0);
            }

#ifdef DEBUG_REALM
            assert((bytes > 0) && (bytes <= bytes_left));
#endif
            total_bytes += bytes;

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired())
              break;
          }
        } else {
          // input but no output, so skip input bytes
          total_bytes = max_bytes;
          in_port->addrcursor.skip_bytes(total_bytes);
          rseqcache.add_span(input_control.current_io_port, in_span_start, total_bytes);
          in_span_start += total_bytes;
        }
      } else {
        if(out_port != 0) {
          // output but no input, so skip output bytes
          total_bytes = max_bytes;
          out_port->addrcursor.skip_bytes(total_bytes);
          wseqcache.add_span(output_control.current_io_port, out_span_start, total_bytes);
          out_span_start += total_bytes;
        } else {
          // skipping both input and output is possible for simultaneous
          //  gather+scatter
          total_bytes = max_bytes;
        }
      }

      bool done = record_address_consumption(total_bytes, total_bytes);

      did_work = true;

      if(done || work_until.is_expired())
        break;
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }

  void RemoteWriteXferDes::notify_request_read_done(Request *req)
  {
    xd_lock.lock();
    default_notify_request_read_done(req);
    xd_lock.unlock();
  }

  void RemoteWriteXferDes::notify_request_write_done(Request *req)
  {
    xd_lock.lock();
    default_notify_request_write_done(req);
    xd_lock.unlock();
  }

  void RemoteWriteXferDes::flush()
  {
    // xd_lock.lock();
    // xd_lock.unlock();
  }

  // doesn't do pre_bytes_write updates, since the remote write message
  //  takes care of it with lower latency (except for zero-byte
  //  termination updates)
  void RemoteWriteXferDes::update_bytes_write(int port_idx, size_t offset, size_t size)
  {
    XferPort *out_port = &output_ports[port_idx];
    size_t inc_amt = out_port->seq_local.add_span(offset, size);
    log_xd.info() << "bytes_write: " << std::hex << guid << std::dec << "(" << port_idx
                  << ") " << offset << "+" << size << " -> " << inc_amt;

    // pre_bytes_write update was handled in the remote AM handler
    if(out_port->peer_guid != XFERDES_NO_GUID) {
      // update bytes total if needed (and available)
      if(out_port->needs_pbt_update.load() && iteration_completed.load_acquire() &&
         (out_port->local_bytes_total == out_port->local_bytes_cons.load())) {
        // exchange sets the flag to false and tells us previous value
        if(out_port->needs_pbt_update.exchange(false))
          xferDes_queue->update_pre_bytes_total(
              out_port->peer_guid, out_port->peer_port_idx, out_port->local_bytes_total);
      }
    }

    // subtract bytes written from the pending count - if that causes it to
    //  go to zero, we can mark the transfer completed and update progress
    //  in case the xd is just waiting for that
    // NOTE: as soon as we set `transfer_completed`, the other references
    //  to this xd may be removed, so do this last, and hold a reference of
    //  our own long enough to call update_progress
    if(inc_amt > 0) {
      int64_t prev = bytes_write_pending.fetch_sub(inc_amt);
      if(prev > 0)
        log_xd.info() << "completion: xd=" << std::hex << guid << std::dec
                      << " remaining=" << (prev - inc_amt);
      if(inc_amt == static_cast<size_t>(prev)) {
        add_reference();
        transfer_completed.store_release(true);
        update_progress();
        remove_reference();
      }
    }
  }

  /*static*/
  void RemoteWriteXferDes::Write1DMessage::handle_message(
      NodeID sender, const RemoteWriteXferDes::Write1DMessage &args, const void *data,
      size_t datalen)
  {
    // assert data copy is in right position
    // assert(data == args.dst_buf);

    log_xd.info() << "remote write recieved: next=" << args.next_xd_guid
                  << " start=" << args.span_start << " size=" << datalen;

    // if requested, notify (probably-local) next XD
    if(args.next_xd_guid != XferDes::XFERDES_NO_GUID)
      XferDesQueue::get_singleton()->update_pre_bytes_write(
          args.next_xd_guid, args.next_port_idx, args.span_start, datalen);
  }

  /*static*/ bool RemoteWriteXferDes::Write1DMessage::handle_inline(
      NodeID sender, const RemoteWriteXferDes::Write1DMessage &args, const void *data,
      size_t datalen, TimeLimit work_until)
  {
    handle_message(sender, args, data, datalen);
    return true;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Channel::SupportedPath
  //

  Channel::SupportedPath &Channel::SupportedPath::set_max_dim(int src_and_dst_dim)
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->max_src_dim = p->max_dst_dim = src_and_dst_dim;
    return *this;
  }

  Channel::SupportedPath &Channel::SupportedPath::set_max_dim(int src_dim, int dst_dim)
  {
    for(SupportedPath *p = this; p; p = p->chain) {
      p->max_src_dim = src_dim;
      p->max_dst_dim = dst_dim;
    }
    return *this;
  }

  Channel::SupportedPath &Channel::SupportedPath::allow_redops()
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->redops_allowed = true;
    return *this;
  }

  Channel::SupportedPath &Channel::SupportedPath::allow_serdez()
  {
    for(SupportedPath *p = this; p; p = p->chain)
      p->serdez_allowed = true;
    return *this;
  }

  void Channel::SupportedPath::populate_memory_bitmask(
      span<const Memory> mems, NodeID node, Channel::SupportedPath::MemBitmask &bitmask)
  {
    bitmask.node = node;

    for(size_t i = 0; i < SupportedPath::MemBitmask::BITMASK_SIZE; i++)
      bitmask.mems[i] = bitmask.ib_mems[i] = 0;

    for(size_t i = 0; i < mems.size(); i++)
      if(mems[i].exists() && (NodeID(ID(mems[i]).memory_owner_node()) == node)) {
        if(ID(mems[i]).is_memory())
          bitmask.mems[ID(mems[i]).memory_mem_idx() >> 6] |=
              (uint64_t(1) << (ID(mems[i]).memory_mem_idx() & 63));
        else if(ID(mems[i]).is_ib_memory())
          bitmask.ib_mems[ID(mems[i]).memory_mem_idx() >> 6] |=
              (uint64_t(1) << (ID(mems[i]).memory_mem_idx() & 63));
      }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ChannelCopyInfo
  //

  std::ostream &operator<<(std::ostream &os, const ChannelCopyInfo &info)
  {
    os << "ChannelCopyInfo { "
       << "src_mem: " << info.src_mem << ", "
       << "dst_mem: " << info.dst_mem << ", "
       << "ind_mem: " << info.ind_mem << ", "
       << "num_spaces: " << info.num_spaces << ", "
       << "is_scatter: " << info.is_scatter << ", "
       << "is_ranges: " << info.is_ranges << ", "
       << "is_direct: " << info.is_direct << ", "
       << "oor_possible: " << info.oor_possible << ", "
       << "addr_size: " << info.addr_size << " }";
    return os;
  }

  bool operator==(const ChannelCopyInfo &lhs, const ChannelCopyInfo &rhs)
  {
    return lhs.src_mem == rhs.src_mem && lhs.dst_mem == rhs.dst_mem &&
           lhs.ind_mem == rhs.ind_mem && lhs.num_spaces == rhs.num_spaces &&
           lhs.is_scatter == rhs.is_scatter && lhs.is_ranges == rhs.is_ranges &&
           lhs.is_direct == rhs.is_direct && lhs.oor_possible == rhs.oor_possible &&
           lhs.addr_size == rhs.addr_size;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Channel
  //

  std::ostream &operator<<(std::ostream &os, const Channel::SupportedPath &p)
  {
    switch(p.src_type) {
    case Channel::SupportedPath::SPECIFIC_MEMORY:
    {
      os << "src=" << p.src_mem;
      break;
    }
    case Channel::SupportedPath::LOCAL_KIND:
    {
      os << "src=" << p.src_kind << "(lcl)";
      break;
    }
    case Channel::SupportedPath::GLOBAL_KIND:
    {
      os << "src=" << p.src_kind << "(gbl)";
      break;
    }
    case Channel::SupportedPath::LOCAL_RDMA:
    {
      os << "src=rdma(lcl)";
      break;
    }
    case Channel::SupportedPath::REMOTE_RDMA:
    {
      os << "src=rdma(rem)";
      break;
    }
    case Channel::SupportedPath::MEMORY_BITMASK:
    {
      os << "src=" << p.src_bitmask.node << '/';
      bool first = true;
      for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
        for(int j = 0; j < 64; j++)
          if((p.src_bitmask.mems[i] & (uint64_t(1) << j)) != 0) {
            if(!first)
              os << ",";
            first = false;
            os << (64 * i + j);
          }
      if(first)
        os << '-';
      os << '/';
      first = true;
      for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
        for(int j = 0; j < 64; j++)
          if((p.src_bitmask.ib_mems[i] & (uint64_t(1) << j)) != 0) {
            if(!first)
              os << ",";
            first = false;
            os << (64 * i + j);
          }
      if(first)
        os << '-';
      break;
    }
    default:
      assert(0);
    }
    switch(p.dst_type) {
    case Channel::SupportedPath::SPECIFIC_MEMORY:
    {
      os << " dst=" << p.dst_mem;
      break;
    }
    case Channel::SupportedPath::LOCAL_KIND:
    {
      os << " dst=" << p.dst_kind << "(lcl)";
      break;
    }
    case Channel::SupportedPath::GLOBAL_KIND:
    {
      os << " dst=" << p.dst_kind << "(gbl)";
      break;
    }
    case Channel::SupportedPath::LOCAL_RDMA:
    {
      os << " dst=rdma(lcl)";
      break;
    }
    case Channel::SupportedPath::REMOTE_RDMA:
    {
      os << " dst=rdma(rem)";
      break;
    }
    case Channel::SupportedPath::MEMORY_BITMASK:
    {
      os << " dst=" << p.dst_bitmask.node << '/';
      bool first = true;
      for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
        for(int j = 0; j < 64; j++)
          if((p.dst_bitmask.mems[i] & (uint64_t(1) << j)) != 0) {
            if(!first)
              os << ",";
            first = false;
            os << (64 * i + j);
          }
      if(first)
        os << '-';
      os << '/';
      first = true;
      for(int i = 0; i < Channel::SupportedPath::MemBitmask::BITMASK_SIZE; i++)
        for(int j = 0; j < 64; j++)
          if((p.dst_bitmask.ib_mems[i] & (uint64_t(1) << j)) != 0) {
            if(!first)
              os << ",";
            first = false;
            os << (64 * i + j);
          }
      if(first)
        os << '-';
      break;
    }
    default:
      assert(0);
    }
    os << " bw=" << p.bandwidth << " lat=" << p.latency;
    if(p.serdez_allowed)
      os << " serdez";
    if(p.redops_allowed)
      os << " redop";
    return os;
  }

  RemoteChannelInfo *Channel::construct_remote_info() const
  {
    std::vector<Memory> indirect_memories;
    for(NodeID n = 0; n < Network::max_node_id + 1; n++) {
      Node &node = get_runtime()->nodes[n];
      for(MemoryImpl *impl : node.memories) {
        if(supports_indirection_memory(impl->me)) {
          indirect_memories.push_back(impl->me);
        }
      }
      for(IBMemory *impl : node.ib_memories) {
        if(supports_indirection_memory(impl->me)) {
          indirect_memories.push_back(impl->me);
        }
      }
    }
    return new SimpleRemoteChannelInfo(node, kind, reinterpret_cast<uintptr_t>(this),
                                       paths, indirect_memories);
  }

  void Channel::print(std::ostream &os) const
  {
    os << "channel{ node=" << node << " kind=" << kind << " paths=[";
    if(!paths.empty()) {
      for(std::vector<SupportedPath>::const_iterator it = paths.begin();
          it != paths.end(); ++it)
        os << "\n    " << *it;
      os << "\n";
    }
    os << "] }";
  }

  const std::vector<Channel::SupportedPath> &Channel::get_paths(void) const
  {
    return paths;
  }

  uint64_t Channel::supports_path(
      ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
      CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
      const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
      XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/, unsigned *lat_ret /*= 0*/)
  {
    if(!supports_redop(redop_id)) {
      return 0;
    }

    Memory src_mem = channel_copy_info.src_mem;
    Memory dst_mem = channel_copy_info.dst_mem;
    // If we don't support the indirection memory, then no need to check the paths.
    if((channel_copy_info.ind_mem != Memory::NO_MEMORY) &&
       !supports_indirection_memory(channel_copy_info.ind_mem)) {
      return 0;
    }
    for(std::vector<SupportedPath>::const_iterator it = paths.begin(); it != paths.end();
        ++it) {
      if(!it->serdez_allowed && ((src_serdez_id != 0) || (dst_serdez_id != 0)))
        continue;
      if(!it->redops_allowed && (redop_id != 0))
        continue;

      bool src_ok = false;
      switch(it->src_type) {
      case SupportedPath::SPECIFIC_MEMORY:
      {
        src_ok = (src_mem == it->src_mem);
        break;
      }
      case SupportedPath::LOCAL_KIND:
      {
        src_ok = (src_mem.exists() && (src_mem.kind() == it->src_kind) &&
                  (NodeID(ID(src_mem).memory_owner_node()) == node));
        break;
      }
      case SupportedPath::GLOBAL_KIND:
      {
        src_ok = (src_mem.exists() && (src_mem.kind() == it->src_kind));
        break;
      }
      case SupportedPath::LOCAL_RDMA:
      {
        if(src_mem.exists() && (NodeID(ID(src_mem).memory_owner_node()) == node)) {
          MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
          assert(src_impl != nullptr && "invalid memory handle");
          // detection of rdma-ness depends on whether memory is
          //  local/remote to us, not the channel
          if(NodeID(ID(src_mem).memory_owner_node()) == Network::my_node_id) {
            src_ok = (src_impl->get_rdma_info(Network::single_network) != nullptr);
          } else {
            RemoteAddress dummy;
            src_ok = src_impl->get_remote_addr(0, dummy);
          }
        }
        break;
      }
      case SupportedPath::REMOTE_RDMA:
      {
        if(src_mem.exists() && (NodeID(ID(src_mem).memory_owner_node()) != node)) {
          MemoryImpl *src_impl = get_runtime()->get_memory_impl(src_mem);
          assert(src_impl != nullptr && "invalid memory handle");
          // detection of rdma-ness depends on whether memory is
          //  local/remote to us, not the channel
          if(NodeID(ID(src_mem).memory_owner_node()) == Network::my_node_id) {
            src_ok = (src_impl->get_rdma_info(Network::single_network) != nullptr);
          } else {
            RemoteAddress dummy;
            src_ok = src_impl->get_remote_addr(0, dummy);
          }
        }
        break;
      }
      case SupportedPath::MEMORY_BITMASK:
      {
        ID src_id(src_mem);
        if(NodeID(src_id.memory_owner_node()) == it->src_bitmask.node) {
          if(src_id.is_memory())
            src_ok = ((it->src_bitmask.mems[src_id.memory_mem_idx() >> 6] &
                       (uint64_t(1) << (src_id.memory_mem_idx() & 63))) != 0);
          else if(src_id.is_ib_memory())
            src_ok = ((it->src_bitmask.ib_mems[src_id.memory_mem_idx() >> 6] &
                       (uint64_t(1) << (src_id.memory_mem_idx() & 63))) != 0);
          else
            src_ok = false; // consider asserting on a non-memory ID?
        } else
          src_ok = false;
        break;
      }
      }
      if(!src_ok)
        continue;

      bool dst_ok = false;
      switch(it->dst_type) {
      case SupportedPath::SPECIFIC_MEMORY:
      {
        dst_ok = (dst_mem == it->dst_mem);
        break;
      }
      case SupportedPath::LOCAL_KIND:
      {
        dst_ok = ((dst_mem.kind() == it->dst_kind) &&
                  (NodeID(ID(dst_mem).memory_owner_node()) == node));
        break;
      }
      case SupportedPath::GLOBAL_KIND:
      {
        dst_ok = (dst_mem.kind() == it->dst_kind);
        break;
      }
      case SupportedPath::LOCAL_RDMA:
      {
        if(NodeID(ID(dst_mem).memory_owner_node()) == node) {
          MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
          assert(dst_impl != nullptr && "invalid memory handle");
          // detection of rdma-ness depends on whether memory is
          //  local/remote to us, not the channel
          if(NodeID(ID(dst_mem).memory_owner_node()) == Network::my_node_id) {
            dst_ok = (dst_impl->get_rdma_info(Network::single_network) != nullptr);
          } else {
            RemoteAddress dummy;
            dst_ok = dst_impl->get_remote_addr(0, dummy);
          }
        }
        break;
      }
      case SupportedPath::REMOTE_RDMA:
      {
        if(NodeID(ID(dst_mem).memory_owner_node()) != node) {
          MemoryImpl *dst_impl = get_runtime()->get_memory_impl(dst_mem);
          assert(dst_impl != nullptr && "invalid memory handle");
          // detection of rdma-ness depends on whether memory is
          //  local/remote to us, not the channel
          if(NodeID(ID(dst_mem).memory_owner_node()) == Network::my_node_id) {
            dst_ok = (dst_impl->get_rdma_info(Network::single_network) != nullptr);
          } else {
            RemoteAddress dummy;
            dst_ok = dst_impl->get_remote_addr(0, dummy);
          }
        }
        break;
      }
      case SupportedPath::MEMORY_BITMASK:
      {
        ID dst_id(dst_mem);
        if(NodeID(dst_id.memory_owner_node()) == it->dst_bitmask.node) {
          if(dst_id.is_memory())
            dst_ok = ((it->dst_bitmask.mems[dst_id.memory_mem_idx() >> 6] &
                       (uint64_t(1) << (dst_id.memory_mem_idx() & 63))) != 0);
          else if(dst_id.is_ib_memory())
            dst_ok = ((it->dst_bitmask.ib_mems[dst_id.memory_mem_idx() >> 6] &
                       (uint64_t(1) << (dst_id.memory_mem_idx() & 63))) != 0);
          else
            dst_ok = false; // consider asserting on a non-memory ID?
        } else
          dst_ok = false;
        break;
      }
      }
      if(!dst_ok)
        continue;

      // match
      if(kind_ret) {
        *kind_ret = it->xd_kind;
      }
      if(bw_ret) {
        *bw_ret = it->bandwidth;
      }
      if(lat_ret) {
        *lat_ret = it->latency;
      }

      // estimate transfer time
      uint64_t xfer_time = uint64_t(total_bytes) * 1000 / it->bandwidth;
      size_t frags = 1;
      if(src_frags) {
        frags = std::max(
            frags,
            (*src_frags)[std::min<size_t>(src_frags->size() - 1, it->max_src_dim)]);
      }
      if(dst_frags) {
        frags = std::max(
            frags,
            (*dst_frags)[std::min<size_t>(dst_frags->size() - 1, it->max_dst_dim)]);
      }
      xfer_time += uint64_t(frags) * it->frag_overhead;

      // make sure returned value is strictly positive
      return std::max<uint64_t>(xfer_time, 1);
    }

    return 0;
  }

  bool Channel::supports_indirection_memory(Memory memory) const
  {
    ID id_mem(memory);
    // TODO: This should just be a query on the memory if it's mapped and accessible
    // locally by a CPU on this node.
    if(node == Network::my_node_id) {
      // This check is only valid for local channels.
      switch(memory.kind()) {
      case Memory::GPU_DYNAMIC_MEM:
      case Memory::GPU_FB_MEM:
        return false;
      default:
      {
        if(NodeID(id_mem.memory_owner_node()) == node) {
          // Local memories are always accessible
          return true;
        } else {
          // Check if we have a remote memory mapping (HACK: should check the
          // MemoryImpl)
          return get_runtime()->remote_shared_memory_mappings.count(memory.id) > 0;
        }
      }
      }
    } else {
      assert(0 && "Should not be called on remote channels!");
    }
    return false;
  }

  static Memory find_sysmem_ib_memory(NodeID node)
  {
    Node &n = get_runtime()->nodes[node];
    for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
        it != n.ib_memories.end(); ++it) {
      switch((*it)->lowlevel_kind) {
      case Memory::SYSTEM_MEM:
      case Memory::REGDMA_MEM:
      case Memory::SOCKET_MEM:
      case Memory::Z_COPY_MEM:
        return (*it)->me;
      default:
        break;
      }
    }
    log_new_dma.fatal() << "no sysmem ib memory on node:" << node;
    abort();
    return Memory::NO_MEMORY;
  }

  Memory Channel::suggest_ib_memories() const { return find_sysmem_ib_memory(node); }

  Memory Channel::suggest_ib_memories_for_node(NodeID node_id) const
  {
    return find_sysmem_ib_memory(node_id);
  }

  // sometimes we need to return a reference to a SupportedPath that won't
  //  actually be added to a channel
  Channel::SupportedPath dummy_supported_path;

  Channel::SupportedPath &Channel::add_path(span<const Memory> src_mems,
                                            span<const Memory> dst_mems,
                                            unsigned bandwidth, unsigned latency,
                                            unsigned frag_overhead, XferDesKind xd_kind)
  {
    NodeSet src_nodes;
    for(size_t i = 0; i < src_mems.size(); i++)
      if(src_mems[i].exists())
        src_nodes.add(ID(src_mems[i]).memory_owner_node());
      else
        src_nodes.add(Network::max_node_id + 1); // src fill placeholder

    NodeSet dst_nodes;
    for(size_t i = 0; i < dst_mems.size(); i++)
      if(dst_mems[i].exists())
        dst_nodes.add(ID(dst_mems[i]).memory_owner_node());

    if(src_nodes.empty() || dst_nodes.empty()) {
      // don't actually add a path
      return dummy_supported_path;
    }

    size_t num_new = src_nodes.size() * dst_nodes.size();
    size_t first_idx = paths.size();
    paths.resize(first_idx + num_new);

    SupportedPath *cur_sp = &paths[first_idx];
    NodeSetIterator src_iter = src_nodes.begin();
    NodeSetIterator dst_iter = dst_nodes.begin();

    while(true) {
      if(src_mems.size() == 1) {
        cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->src_mem = src_mems[0];
      } else if(*src_iter > Network::max_node_id) {
        cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->src_mem = Memory::NO_MEMORY; // src fill
      } else {
        cur_sp->src_type = SupportedPath::MEMORY_BITMASK;
        cur_sp->populate_memory_bitmask(src_mems, *src_iter, cur_sp->src_bitmask);
      }

      if(dst_mems.size() == 1) {
        cur_sp->dst_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->dst_mem = dst_mems[0];
      } else {
        cur_sp->dst_type = SupportedPath::MEMORY_BITMASK;
        cur_sp->populate_memory_bitmask(dst_mems, *dst_iter, cur_sp->dst_bitmask);
      }

      cur_sp->bandwidth = bandwidth;
      cur_sp->latency = latency;
      cur_sp->frag_overhead = frag_overhead;
      cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
      cur_sp->redops_allowed = false;                // default
      cur_sp->serdez_allowed = false;                // default
      cur_sp->xd_kind = xd_kind;

      // bump iterators, wrapping dst if not done with src
      ++dst_iter;
      if(dst_iter == dst_nodes.end()) {
        ++src_iter;
        if(src_iter == src_nodes.end()) {
          // end of chain and of loop
          cur_sp->chain = 0;
          break;
        }
        dst_iter = dst_nodes.begin();
      }
      // not end of chain, so connect to next before bumping current pointer
      cur_sp->chain = cur_sp + 1;
      ++cur_sp;
    }
#ifdef DEBUG_REALM
    assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
    // return reference to beginning of chain
    return paths[first_idx];
  }

  Channel::SupportedPath &Channel::add_path(span<const Memory> src_mems,
                                            Memory::Kind dst_kind, bool dst_global,
                                            unsigned bandwidth, unsigned latency,
                                            unsigned frag_overhead, XferDesKind xd_kind)
  {
    NodeSet src_nodes;
    for(size_t i = 0; i < src_mems.size(); i++)
      if(src_mems[i].exists())
        src_nodes.add(ID(src_mems[i]).memory_owner_node());
      else
        src_nodes.add(Network::max_node_id + 1); // src fill placeholder

    if(src_nodes.empty()) {
      // don't actually add a path
      return dummy_supported_path;
    }

    size_t num_new = src_nodes.size();
    size_t first_idx = paths.size();
    paths.resize(first_idx + num_new);

    SupportedPath *cur_sp = &paths[first_idx];
    NodeSetIterator src_iter = src_nodes.begin();

    while(true) {
      if(src_mems.size() == 1) {
        cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->src_mem = src_mems[0];
      } else if(*src_iter > Network::max_node_id) {
        cur_sp->src_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->src_mem = Memory::NO_MEMORY; // src fill
      } else {
        cur_sp->src_type = SupportedPath::MEMORY_BITMASK;
        cur_sp->populate_memory_bitmask(src_mems, *src_iter, cur_sp->src_bitmask);
      }

      cur_sp->dst_type =
          (dst_global ? SupportedPath::GLOBAL_KIND : SupportedPath::LOCAL_KIND);
      cur_sp->dst_kind = dst_kind;

      cur_sp->bandwidth = bandwidth;
      cur_sp->latency = latency;
      cur_sp->frag_overhead = frag_overhead;
      cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
      cur_sp->redops_allowed = false;                // default
      cur_sp->serdez_allowed = false;                // default
      cur_sp->xd_kind = xd_kind;

      ++src_iter;
      if(src_iter == src_nodes.end()) {
        // end of chain and of loop
        cur_sp->chain = 0;
        break;
      }

      // not end of chain, so connect to next before bumping current pointer
      cur_sp->chain = cur_sp + 1;
      ++cur_sp;
    }
#ifdef DEBUG_REALM
    assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
    // return reference to beginning of chain
    return paths[first_idx];
  }

  Channel::SupportedPath &Channel::add_path(Memory::Kind src_kind, bool src_global,
                                            span<const Memory> dst_mems,
                                            unsigned bandwidth, unsigned latency,
                                            unsigned frag_overhead, XferDesKind xd_kind)
  {
    NodeSet dst_nodes;
    for(size_t i = 0; i < dst_mems.size(); i++)
      if(dst_mems[i].exists())
        dst_nodes.add(ID(dst_mems[i]).memory_owner_node());

    if(dst_nodes.empty()) {
      // don't actually add a path
      return dummy_supported_path;
    }

    size_t num_new = dst_nodes.size();
    size_t first_idx = paths.size();
    paths.resize(first_idx + num_new);

    SupportedPath *cur_sp = &paths[first_idx];
    NodeSetIterator dst_iter = dst_nodes.begin();

    while(true) {
      cur_sp->src_type =
          (src_global ? SupportedPath::GLOBAL_KIND : SupportedPath::LOCAL_KIND);
      cur_sp->src_kind = src_kind;

      if(dst_mems.size() == 1) {
        cur_sp->dst_type = SupportedPath::SPECIFIC_MEMORY;
        cur_sp->dst_mem = dst_mems[0];
      } else {
        cur_sp->dst_type = SupportedPath::MEMORY_BITMASK;
        cur_sp->populate_memory_bitmask(dst_mems, *dst_iter, cur_sp->dst_bitmask);
      }

      cur_sp->bandwidth = bandwidth;
      cur_sp->latency = latency;
      cur_sp->frag_overhead = frag_overhead;
      cur_sp->max_src_dim = cur_sp->max_dst_dim = 1; // default
      cur_sp->redops_allowed = false;                // default
      cur_sp->serdez_allowed = false;                // default
      cur_sp->xd_kind = xd_kind;

      ++dst_iter;
      if(dst_iter == dst_nodes.end()) {
        // end of chain and of loop
        cur_sp->chain = 0;
        break;
      }

      // not end of chain, so connect to next before bumping current pointer
      cur_sp->chain = cur_sp + 1;
      ++cur_sp;
    }
#ifdef DEBUG_REALM
    assert(cur_sp == (paths.data() + paths.size() - 1));
#endif
    // return reference to beginning of chain
    return paths[first_idx];
  }

  Channel::SupportedPath &Channel::add_path(Memory::Kind src_kind, bool src_global,
                                            Memory::Kind dst_kind, bool dst_global,
                                            unsigned bandwidth, unsigned latency,
                                            unsigned frag_overhead, XferDesKind xd_kind)
  {
    size_t idx = paths.size();
    paths.resize(idx + 1);
    SupportedPath &p = paths[idx];
    p.chain = 0;

    p.src_type = (src_global ? SupportedPath::GLOBAL_KIND : SupportedPath::LOCAL_KIND);
    p.src_kind = src_kind;
    p.dst_type = (dst_global ? SupportedPath::GLOBAL_KIND : SupportedPath::LOCAL_KIND);
    p.dst_kind = dst_kind;
    p.bandwidth = bandwidth;
    p.latency = latency;
    p.frag_overhead = frag_overhead;
    p.max_src_dim = p.max_dst_dim = 1; // default
    p.redops_allowed = false;          // default
    p.serdez_allowed = false;          // default
    p.xd_kind = xd_kind;
    return p;
  }

  // TODO: allow rdma path to limit by kind?
  Channel::SupportedPath &Channel::add_path(bool local_loopback, unsigned bandwidth,
                                            unsigned latency, unsigned frag_overhead,
                                            XferDesKind xd_kind)
  {
    size_t idx = paths.size();
    paths.resize(idx + 1);
    SupportedPath &p = paths[idx];
    p.chain = 0;

    p.src_type = SupportedPath::LOCAL_RDMA;
    p.dst_type =
        (local_loopback ? SupportedPath::LOCAL_RDMA : SupportedPath::REMOTE_RDMA);
    p.bandwidth = bandwidth;
    p.latency = latency;
    p.frag_overhead = frag_overhead;
    p.max_src_dim = p.max_dst_dim = 1; // default
    p.redops_allowed = false;          // default
    p.serdez_allowed = false;          // default
    p.xd_kind = xd_kind;
    return p;
  }

  void Channel::update_channel_state(void)
  {
    assert(has_redop_path == false);
    assert(has_non_redop_path == false);

    for(const SupportedPath &path : paths) {
      if(path.redops_allowed) {
        has_redop_path = true;
      } else {
        has_non_redop_path = true;
      }

      // the channel has both redop and non-redop path
      // early return
      if(has_redop_path && has_non_redop_path) {
        break;
      }
    }
  }

  bool Channel::supports_redop(ReductionOpID redop_id) const
  {
    if(redop_id == 0) {
      return true;
    }

    return has_redop_path;
  }

  long Channel::progress_xd(XferDes *xd, long max_nr)
  {
    const long MAX_NR = 8;
    Request *requests[MAX_NR];
    long nr_got = xd->get_requests(requests, std::min(max_nr, MAX_NR));
    if(nr_got == 0)
      return 0;
    long nr_submitted = submit(requests, nr_got);
    assert(nr_got == nr_submitted);
    return nr_submitted;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleXferDesFactory
  //

  SimpleXferDesFactory::SimpleXferDesFactory(uintptr_t _channel)
    : channel(_channel)
  {}

  bool SimpleXferDesFactory::needs_release() { return false; }

  void SimpleXferDesFactory::create_xfer_des(
      uintptr_t dma_op, NodeID launch_node, NodeID target_node, XferDesID guid,
      const std::vector<XferDesPortInfo> &inputs_info,
      const std::vector<XferDesPortInfo> &outputs_info, int priority,
      XferDesRedopInfo redop_info, const void *fill_data, size_t fill_size,
      size_t fill_total)
  {
    if(target_node == Network::my_node_id) {
      // local creation
      // assert(!inst.exists());
      LocalChannel *c = reinterpret_cast<LocalChannel *>(channel);
      XferDes *xd =
          c->create_xfer_des(dma_op, launch_node, guid, inputs_info, outputs_info,
                             priority, redop_info, fill_data, fill_size, fill_total);

      c->enqueue_ready_xd(xd);
    } else {
      // remote creation
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = ((bcs << inputs_info) && (bcs << outputs_info) && (bcs << priority) &&
                   (bcs << redop_info) && (bcs << fill_total));
        if(ok && (fill_size > 0))
          ok = bcs.append_bytes(fill_data, fill_size);
        assert(ok);
      }
      size_t req_size = bcs.bytes_used();
      ActiveMessage<SimpleXferDesCreateMessage> amsg(target_node, req_size);
      // amsg->inst = inst;
      amsg->launch_node = launch_node;
      amsg->guid = guid;
      amsg->dma_op = dma_op;
      amsg->channel = channel;
      {
        bool ok = ((amsg << inputs_info) && (amsg << outputs_info) &&
                   (amsg << priority) && (amsg << redop_info) && (amsg << fill_total));
        if(ok && (fill_size > 0))
          amsg.add_payload(fill_data, fill_size);
        assert(ok);
      }
      amsg.commit();

      // normally ownership of input and output iterators would be taken
      //  by the local XferDes we create, but here we sent a copy, so delete
      //  the originals
      for(std::vector<XferDesPortInfo>::const_iterator it = inputs_info.begin();
          it != inputs_info.end(); ++it)
        delete it->iter;

      for(std::vector<XferDesPortInfo>::const_iterator it = outputs_info.begin();
          it != outputs_info.end(); ++it)
        delete it->iter;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleXferDesCreateMessage
  //

  /*static*/ void
  SimpleXferDesCreateMessage::handle_message(NodeID sender,
                                             const SimpleXferDesCreateMessage &args,
                                             const void *msgdata, size_t msglen)
  {
    std::vector<XferDesPortInfo> inputs_info, outputs_info;
    int priority = 0;
    XferDesRedopInfo redop_info;
    size_t fill_total = 0;

    Realm::Serialization::FixedBufferDeserializer fbd(msgdata, msglen);

    bool ok = ((fbd >> inputs_info) && (fbd >> outputs_info) && (fbd >> priority) &&
               (fbd >> redop_info) && (fbd >> fill_total));
    assert(ok);
    const void *fill_data;
    size_t fill_size;
    if(fbd.bytes_left() == 0) {
      fill_data = 0;
      fill_size = 0;
    } else {
      fill_size = fbd.bytes_left();
      fill_data = fbd.peek_bytes(fill_size);
    }

    // assert(!args.inst.exists());
    LocalChannel *c = reinterpret_cast<LocalChannel *>(args.channel);
    XferDes *xd = c->create_xfer_des(args.dma_op, args.launch_node, args.guid,
                                     inputs_info, outputs_info, priority, redop_info,
                                     fill_data, fill_size, fill_total);

    c->enqueue_ready_xd(xd);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class NotifyXferDesCompleteMessage
  //

  /*static*/ void
  NotifyXferDesCompleteMessage::handle_message(NodeID sender,
                                               const NotifyXferDesCompleteMessage &args,
                                               const void *data, size_t datalen)
  {
    args.op->notify_xd_completion(args.xd_id);
  }

  /*static*/ void NotifyXferDesCompleteMessage::send_request(NodeID target,
                                                             TransferOperation *op,
                                                             XferDesID xd_id)
  {
    ActiveMessage<NotifyXferDesCompleteMessage> amsg(target);
    amsg->op = op;
    amsg->xd_id = xd_id;
    amsg.commit();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalChannel
  //

  LocalChannel::LocalChannel(XferDesKind _kind)
    : Channel(_kind)
    , factory_singleton(reinterpret_cast<uintptr_t>(this))
  {}

  XferDesFactory *LocalChannel::get_factory() { return &factory_singleton; }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SimpleRemoteChannelInfo
  //

  SimpleRemoteChannelInfo::SimpleRemoteChannelInfo() {}

  SimpleRemoteChannelInfo::SimpleRemoteChannelInfo(
      NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
      const std::vector<Channel::SupportedPath> &_paths,
      const std::vector<Memory> &_indirection_memories)
    : owner(_owner)
    , kind(_kind)
    , remote_ptr(_remote_ptr)
    , paths(_paths)
    , indirect_memories(_indirection_memories)
  {}

  SimpleRemoteChannelInfo::SimpleRemoteChannelInfo(
      NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
      const std::vector<Channel::SupportedPath> &_paths)
    : owner(_owner)
    , kind(_kind)
    , remote_ptr(_remote_ptr)
    , paths(_paths)
  {}

  RemoteChannel *SimpleRemoteChannelInfo::create_remote_channel()
  {
    RemoteChannel *rc = new RemoteChannel(remote_ptr, indirect_memories);
    rc->node = owner;
    rc->kind = kind;
    rc->paths.swap(paths);
    return rc;
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      SimpleRemoteChannelInfo>
      SimpleRemoteChannelInfo::serdez_subclass;

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteChannel
  //

  RemoteChannel::RemoteChannel(uintptr_t _remote_ptr,
                               const std::vector<Memory> &_indirect_memories)
    : Channel(XFER_NONE)
    , remote_ptr(_remote_ptr)
    , factory_singleton(_remote_ptr)
    , indirect_memories(_indirect_memories.begin(), _indirect_memories.end())
  {}
  RemoteChannel::RemoteChannel(uintptr_t _remote_ptr)
    : Channel(XFER_NONE)
    , remote_ptr(_remote_ptr)
    , factory_singleton(_remote_ptr)
  {}

  void RemoteChannel::shutdown() {}

  uintptr_t RemoteChannel::get_remote_ptr() const { return remote_ptr; }

  XferDesFactory *RemoteChannel::get_factory() { return &factory_singleton; }

  void RemoteChannel::register_redop(ReductionOpID redop_id)
  {
    RWLock::AutoWriterLock al(mutex);
    (void)supported_redops.insert(redop_id);
  }

  bool RemoteChannel::supports_redop(ReductionOpID redop_id) const
  {
    if(redop_id == 0) {
      return has_non_redop_path;
    }

    if(has_redop_path) {
      RWLock::AutoReaderLock al(mutex);
      return supported_redops.count(redop_id) != 0;
    }

    return false;
  }

  long RemoteChannel::submit(Request **requests, long nr)
  {
    assert(0);
    return 0;
  }

  void RemoteChannel::pull() { assert(0); }

  long RemoteChannel::available()
  {
    assert(0);
    return 0;
  }

  uint64_t RemoteChannel::supports_path(
      ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
      CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
      const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
      XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/, unsigned *lat_ret /*= 0*/)
  {
    // simultaneous serialization/deserialization not
    //  allowed anywhere right now
    if((src_serdez_id != 0) && (dst_serdez_id != 0)) {
      return 0;
    }

    // fall through to normal checks
    return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                  redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                  bw_ret, lat_ret);
  }

  bool RemoteChannel::supports_indirection_memory(Memory mem) const
  {
    return indirect_memories.count(mem) > 0;
  }

  static void enumerate_remote_shared_mems(std::vector<Memory> &mems)
  {
    RuntimeImpl *runtime = get_runtime();
    size_t idx = 0;
    mems.resize(runtime->remote_shared_memory_mappings.size(), Memory::NO_MEMORY);
    for(std::unordered_map<realm_id_t, SharedMemoryInfo>::iterator it =
            runtime->remote_shared_memory_mappings.begin();
        it != runtime->remote_shared_memory_mappings.end(); ++it) {
      Memory m;
      m.id = it->first;
      mems[idx++] = m;
    }
  }

  /*static*/ void enumerate_local_cpu_memories(std::vector<Memory> &mems)
  {
    Node &n = get_runtime()->nodes[Network::my_node_id];

    for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
        it != n.memories.end(); ++it) {
      if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
         ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
         ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
         ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
         ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM)) {
        mems.push_back((*it)->me);
      }
    }

    for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
        it != n.ib_memories.end(); ++it) {
      if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
         ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
         ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
         ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
         ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM)) {
        mems.push_back((*it)->me);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemfillChannel
  //

  MemfillChannel::MemfillChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<MemfillChannel, MemfillXferDes>(bgwork, XFER_MEM_FILL,
                                                       "memfill channel")
  {
    unsigned bw = 128000;         // HACK - estimate at 128 GB/s
    unsigned latency = 100;       // HACK - estimate at 100ns
    unsigned frag_overhead = 100; // HACK - estimate at 100ns

    // all local cpu memories are valid dests
    std::vector<Memory> local_cpu_mems;
    enumerate_local_cpu_memories(local_cpu_mems);
    std::vector<Memory> remote_shared_mems;
    enumerate_remote_shared_mems(remote_shared_mems);

    add_path(Memory::NO_MEMORY, local_cpu_mems, bw, latency, frag_overhead, XFER_MEM_FILL)
        .set_max_dim(3);

    if(remote_shared_mems.size() > 0) {
      add_path(Memory::NO_MEMORY, remote_shared_mems, bw, latency, frag_overhead,
               XFER_MEM_FILL)
          .set_max_dim(3);
    }

    xdq.add_to_manager(bgwork);
  }

  MemfillChannel::~MemfillChannel() {}

  XferDes *
  MemfillChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                  const std::vector<XferDesPortInfo> &inputs_info,
                                  const std::vector<XferDesPortInfo> &outputs_info,
                                  int priority, XferDesRedopInfo redop_info,
                                  const void *fill_data, size_t fill_size,
                                  size_t fill_total)
  {
    assert(redop_info.id == 0); // TODO: add support
    assert(fill_size > 0);
    return new MemfillXferDes(dma_op, this, launch_node, guid, inputs_info, outputs_info,
                              priority, fill_data, fill_size, fill_total);
  }

  long MemfillChannel::submit(Request **requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemreduceChannel
  //

  MemreduceChannel::MemreduceChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<MemreduceChannel, MemreduceXferDes>(bgwork, XFER_MEM_CPY,
                                                           "memreduce channel")
  {
    unsigned bw = 1000;           // HACK - estimate at 1 GB/s
    unsigned latency = 100;       // HACK - estimate at 100ns
    unsigned frag_overhead = 100; // HACK - estimate at 100ns

    // all local cpu memories are valid sources and dests
    std::vector<Memory> local_cpu_mems;
    enumerate_local_cpu_memories(local_cpu_mems);
    std::vector<Memory> remote_shared_mems;
    enumerate_remote_shared_mems(remote_shared_mems);

    add_path(local_cpu_mems, local_cpu_mems, bw, latency, frag_overhead, XFER_MEM_CPY)
        .set_max_dim(3)
        .allow_redops();

    if(remote_shared_mems.size() > 0) {
      add_path(local_cpu_mems, remote_shared_mems, bw, latency, frag_overhead,
               XFER_MEM_CPY)
          .set_max_dim(3)
          .allow_redops();
    }

    xdq.add_to_manager(bgwork);
  }

  bool MemreduceChannel::supports_redop(ReductionOpID redop_id) const
  {
    return redop_id != 0;
  }

  XferDes *
  MemreduceChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                    const std::vector<XferDesPortInfo> &inputs_info,
                                    const std::vector<XferDesPortInfo> &outputs_info,
                                    int priority, XferDesRedopInfo redop_info,
                                    const void *fill_data, size_t fill_size,
                                    size_t fill_total)
  {
    assert(redop_info.id != 0); // redop is required
    assert(fill_size == 0);
    return new MemreduceXferDes(dma_op, this, launch_node, guid, inputs_info,
                                outputs_info, priority, redop_info);
  }

  long MemreduceChannel::submit(Request **requests, long nr)
  {
    // unused
    assert(0);
    return 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class GASNetChannel
  //

  // TODO: deprecate this channel/memory entirely
  GASNetChannel::GASNetChannel(BackgroundWorkManager *bgwork, XferDesKind _kind)
    : SingleXDQChannel<GASNetChannel, GASNetXferDes>(
          bgwork, _kind, stringbuilder() << "gasnet channel (kind= " << _kind << ")")
  {
    unsigned bw = 1000;            // HACK - estimate at 1 GB/s
    unsigned latency = 5000;       // HACK - estimate at 5 us
    unsigned frag_overhead = 1000; // HACK - estimate at 1 us

    // all local cpu memories are valid sources/dests
    std::vector<Memory> local_cpu_mems;
    enumerate_local_cpu_memories(local_cpu_mems);

    if(_kind == XFER_GASNET_READ)
      add_path(Memory::GLOBAL_MEM, true, local_cpu_mems, bw, latency, frag_overhead,
               XFER_GASNET_READ);
    else
      add_path(local_cpu_mems, Memory::GLOBAL_MEM, true, bw, latency, frag_overhead,
               XFER_GASNET_WRITE);
  }

  GASNetChannel::~GASNetChannel() {}

  XferDes *
  GASNetChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                 const std::vector<XferDesPortInfo> &inputs_info,
                                 const std::vector<XferDesPortInfo> &outputs_info,
                                 int priority, XferDesRedopInfo redop_info,
                                 const void *fill_data, size_t fill_size,
                                 size_t fill_total)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    return new GASNetXferDes(dma_op, this, launch_node, guid, inputs_info, outputs_info,
                             priority);
  }

  long GASNetChannel::submit(Request **requests, long nr)
  {
    for(long i = 0; i < nr; i++) {
      GASNetRequest *req = (GASNetRequest *)requests[i];
      // no serdez support
      assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
      assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);
      switch(kind) {
      case XFER_GASNET_READ:
      {
        req->xd->input_ports[req->src_port_idx].mem->get_bytes(
            req->gas_off, req->mem_base, req->nbytes);
        break;
      }
      case XFER_GASNET_WRITE:
      {
        req->xd->output_ports[req->dst_port_idx].mem->put_bytes(
            req->gas_off, req->mem_base, req->nbytes);
        break;
      }
      default:
        assert(0);
      }
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }
    return nr;
  }

  RemoteWriteChannel::RemoteWriteChannel(BackgroundWorkManager *bgwork)
    : SingleXDQChannel<RemoteWriteChannel, RemoteWriteXferDes>(bgwork, XFER_REMOTE_WRITE,
                                                               "remote write channel")
  {
    unsigned bw = 5000;            // HACK - estimate at 5 GB/s
    unsigned latency = 2000;       // HACK - estimate at 2 us
    unsigned frag_overhead = 1000; // HACK - estimate at 1 us
    // any combination of SYSTEM/REGDMA/Z_COPY/SOCKET_MEM
    // for(size_t i = 0; i < num_cpu_mem_kinds; i++)
    //   add_path(cpu_mem_kinds[i], false,
    // 	   Memory::REGDMA_MEM, true,
    // 	   bw, latency, false, false, XFER_REMOTE_WRITE);
    add_path(false /*!local_loopback*/, bw, latency, frag_overhead, XFER_REMOTE_WRITE);
    // TODO: permit 2d sources?
  }

  RemoteWriteChannel::~RemoteWriteChannel() {}

  XferDes *RemoteWriteChannel::create_xfer_des(
      uintptr_t dma_op, NodeID launch_node, XferDesID guid,
      const std::vector<XferDesPortInfo> &inputs_info,
      const std::vector<XferDesPortInfo> &outputs_info, int priority,
      XferDesRedopInfo redop_info, const void *fill_data, size_t fill_size,
      size_t fill_total)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    return new RemoteWriteXferDes(dma_op, this, launch_node, guid, inputs_info,
                                  outputs_info, priority);
  }

  long RemoteWriteChannel::submit(Request **requests, long nr)
  {
    // should not be reached
    assert(0);
    return nr;
  }

  /*static*/ void XferDesDestroyMessage::handle_message(NodeID sender,
                                                        const XferDesDestroyMessage &args,
                                                        const void *msgdata,
                                                        size_t msglen)
  {
    XferDesQueue::get_singleton()->destroy_xferDes(args.guid);
  }

  /*static*/ void
  UpdateBytesTotalMessage::handle_message(NodeID sender,
                                          const UpdateBytesTotalMessage &args,
                                          const void *msgdata, size_t msglen)
  {
    XferDesQueue::get_singleton()->update_pre_bytes_total(args.guid, args.port_idx,
                                                          args.pre_bytes_total);
  }

  /*static*/ void
  UpdateBytesWriteMessage::handle_message(NodeID sender,
                                          const UpdateBytesWriteMessage &args,
                                          const void *msgdata, size_t msglen)
  {
    XferDesQueue::get_singleton()->update_pre_bytes_write(
        args.guid, args.port_idx, args.span_start, args.span_size);
  }

  /*static*/ void
  UpdateBytesReadMessage::handle_message(NodeID sender,
                                         const UpdateBytesReadMessage &args,
                                         const void *msgdata, size_t msglen)
  {
    XferDesQueue::get_singleton()->update_next_bytes_read(
        args.guid, args.port_idx, args.span_start, args.span_size);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesPlaceholder
  //

  XferDesPlaceholder::XferDesPlaceholder()
    : refcount(1)
    , xd(0)
    , nb_update_pre_bytes_total_calls_received(0)
  {
    for(int i = 0; i < INLINE_PORTS; i++)
      inline_bytes_total[i] = ~size_t(0);
  }

  XferDesPlaceholder::~XferDesPlaceholder() {}

  void XferDesPlaceholder::add_reference() { refcount.fetch_add_acqrel(1); }

  void XferDesPlaceholder::remove_reference()
  {
    unsigned prev = refcount.fetch_sub_acqrel(1);
    // if this is the last reference to a placeholder that was assigned an
    //  xd (the unassigned case should only happen on an insertion race),
    //  propagate our progress info to the xd
    if((prev == 1) && xd) {
      bool updated = false;
      for(int i = 0; i < INLINE_PORTS; i++) {
        if(inline_bytes_total[i] != ~size_t(0)) {
          xd->update_pre_bytes_total(i, inline_bytes_total[i]);
          updated = true;
        }
        if(!inline_pre_write[i].empty()) {
          inline_pre_write[i].import(xd->input_ports[i].seq_remote);
          updated = true;
        }
      }
      for(std::map<int, size_t>::const_iterator it = extra_bytes_total.begin();
          it != extra_bytes_total.end(); ++it) {
        xd->update_pre_bytes_total(it->first, it->second);
        updated = true;
      }
      for(std::map<int, SequenceAssembler>::const_iterator it = extra_pre_write.begin();
          it != extra_pre_write.end(); ++it) {
        it->second.import(xd->input_ports[it->first].seq_remote);
        updated = true;
      }
      if(updated)
        xd->update_progress();
      xd->remove_reference();
    }

    if(prev == 1)
      delete this;
  }

  void XferDesPlaceholder::update_pre_bytes_write(int port_idx, size_t span_start,
                                                  size_t span_size)
  {
    if(port_idx < INLINE_PORTS) {
      inline_pre_write[port_idx].add_span(span_start, span_size);
    } else {
      // need a mutex around getting the reference to the SequenceAssembler
      SequenceAssembler *sa;
      {
        AutoLock<> al(extra_mutex);
        sa = &extra_pre_write[port_idx];
      }
      sa->add_span(span_start, span_size);
    }
  }

  void XferDesPlaceholder::update_pre_bytes_total(int port_idx, size_t pre_bytes_total)
  {
    if(port_idx < INLINE_PORTS) {
      inline_bytes_total[port_idx] = pre_bytes_total;
    } else {
      AutoLock<> al(extra_mutex);
      extra_bytes_total[port_idx] = pre_bytes_total;
    }
  }

  void XferDesPlaceholder::set_real_xd(XferDes *_xd)
  {
    // remember the xd and add a reference to it - actual updates will
    //  happen once we're destroyed
    xd = _xd;
    xd->add_reference();
    xd->nb_update_pre_bytes_total_calls_received.fetch_add_acqrel(
        nb_update_pre_bytes_total_calls_received.load_acquire());
  }

  void XferDesPlaceholder::add_update_pre_bytes_total_received(void)
  {
    nb_update_pre_bytes_total_calls_received.fetch_add_acqrel(1);
  }

  unsigned XferDesPlaceholder::get_update_pre_bytes_total_received(void)
  {
    return nb_update_pre_bytes_total_calls_received.load_acquire();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class XferDesQueue
  //

  /*static*/ XferDesQueue *XferDesQueue::get_singleton()
  {
    // we use a single queue for all xferDes
    static XferDesQueue xferDes_queue;
    return &xferDes_queue;
  }

  void XferDesQueue::update_pre_bytes_write(XferDesID xd_guid, int port_idx,
                                            size_t span_start, size_t span_size)
  {
    NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
    if(execution_node == Network::my_node_id) {
      XferDes *xd = 0;
      XferDesPlaceholder *ph = 0;
      {
        AutoLock<> al(guid_lock);
        std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
        if(it != guid_to_xd.end()) {
          if((it->second & 1) == 0) {
            // is a real xd - add a reference before we release the lock
            xd = reinterpret_cast<XferDes *>(it->second);
            xd->add_reference();
          } else {
            // is a placeholder - add a reference before we release lock
            ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
            ph->add_reference();
          }
        }
      }
      // if we got neither, create a new placeholder and try to add it,
      //  coping with the case where we lose to another insertion
      if(!xd && !ph) {
        XferDesPlaceholder *new_ph = new XferDesPlaceholder;
        {
          AutoLock<> al(guid_lock);
          std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
          if(it != guid_to_xd.end()) {
            if((it->second & 1) == 0) {
              // is a real xd - add a reference before we release the lock
              xd = reinterpret_cast<XferDes *>(it->second);
              xd->add_reference();
            } else {
              // is a placeholder - add a reference before we release lock
              ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
              ph->add_reference();
            }
          } else {
            guid_to_xd.insert(
                std::make_pair(xd_guid, reinterpret_cast<uintptr_t>(new_ph) + 1));
            ph = new_ph;
            new_ph->add_reference(); // table keeps the original reference
          }
        }
        // if we didn't install our placeholder, remove the reference so it
        //  goes away
        if(ph != new_ph)
          new_ph->remove_reference();
      }
      // now we can update the xd or the placeholder and then release the
      //  reference we kept
      if(xd) {
        xd->update_pre_bytes_write(port_idx, span_start, span_size);
        xd->remove_reference();
      } else {
        ph->update_pre_bytes_write(port_idx, span_start, span_size);
        ph->remove_reference();
      }
    } else {
      // send a active message to remote node
      // this can happen if we have a non-network path (e.g. ipc) to another rank
      UpdateBytesWriteMessage::send_request(execution_node, xd_guid, port_idx, span_start,
                                            span_size);
    }
  }

  void XferDesQueue::update_pre_bytes_total(XferDesID xd_guid, int port_idx,
                                            size_t pre_bytes_total)
  {
    NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
    if(execution_node == Network::my_node_id) {
      XferDes *xd = 0;
      XferDesPlaceholder *ph = 0;
      {
        AutoLock<> al(guid_lock);
        std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
        if(it != guid_to_xd.end()) {
          if((it->second & 1) == 0) {
            // is a real xd - add a reference before we release the lock
            xd = reinterpret_cast<XferDes *>(it->second);
            xd->add_reference();
            xd->add_update_pre_bytes_total_received();
            log_xd_ref.info(
                "xd=%llx, add_ref refcount=%u, update_pre_bytes_total_received=%u",
                xd_guid, xd->reference_count.load_acquire(),
                xd->nb_update_pre_bytes_total_calls_received.load_acquire());
          } else {
            // is a placeholder - add a reference before we release lock
            ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
            ph->add_reference();
            ph->add_update_pre_bytes_total_received();
            log_xd_ref.info("xd=%llx, placeholder, update_pre_bytes_total_received=%u",
                            xd_guid, ph->get_update_pre_bytes_total_received());
          }
        }
      }
      // if we got neither, create a new placeholder and try to add it,
      //  coping with the case where we lose to another insertion
      if(!xd && !ph) {
        XferDesPlaceholder *new_ph = new XferDesPlaceholder;
        {
          AutoLock<> al(guid_lock);
          std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
          if(it != guid_to_xd.end()) {
            if((it->second & 1) == 0) {
              // is a real xd - add a reference before we release the lock
              xd = reinterpret_cast<XferDes *>(it->second);
              xd->add_reference();
              xd->add_update_pre_bytes_total_received();
              log_xd_ref.info(
                  "xd=%llx, 2nd, add_ref refcount=%u, update_pre_bytes_total_received=%u",
                  xd_guid, xd->reference_count.load_acquire(),
                  xd->nb_update_pre_bytes_total_calls_received.load_acquire());
            } else {
              // is a placeholder - add a reference before we release lock
              ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
              ph->add_reference();
              ph->add_update_pre_bytes_total_received();
              log_xd_ref.info(
                  "xd=%llx, 2nd, placeholder, update_pre_bytes_total_received=%u",
                  xd_guid, ph->get_update_pre_bytes_total_received());
            }
          } else {
            guid_to_xd.insert(
                std::make_pair(xd_guid, reinterpret_cast<uintptr_t>(new_ph) + 1));
            ph = new_ph;
            new_ph->add_reference(); // table keeps the original reference
            new_ph->add_update_pre_bytes_total_received();
            log_xd_ref.info(
                "xd=%llx, new placeholder, update_pre_bytes_total_received=%u", xd_guid,
                ph->get_update_pre_bytes_total_received());
          }
        }
        // if we didn't install our placeholder, remove the reference so it
        //  goes away
        if(ph != new_ph)
          new_ph->remove_reference();
      }
      // now we can update the xd or the placeholder and then release the
      //  reference we kept
      if(xd) {
        xd->update_pre_bytes_total(port_idx, pre_bytes_total);
        xd->remove_reference();
      } else {
        ph->update_pre_bytes_total(port_idx, pre_bytes_total);
        ph->remove_reference();
      }
    } else {
      // send an active message to remote node
      ActiveMessage<UpdateBytesTotalMessage> amsg(execution_node);
      amsg->guid = xd_guid;
      amsg->port_idx = port_idx;
      amsg->pre_bytes_total = pre_bytes_total;
      amsg.commit();
    }
  }

  void XferDesQueue::update_next_bytes_read(XferDesID xd_guid, int port_idx,
                                            size_t span_start, size_t span_size)
  {
    NodeID execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
    if(execution_node == Network::my_node_id) {
      XferDes *xd = 0;
      {
        AutoLock<> al(guid_lock);
        std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd_guid);
        if(it != guid_to_xd.end()) {
          if((it->second & 1) == 0) {
            // is a real xd - add a reference before we release the lock
            xd = reinterpret_cast<XferDes *>(it->second);
            xd->add_reference();
            log_xd_ref.info("xd=%llx, after add_ref refcount=%u", xd_guid,
                            xd->reference_count.load_acquire());
          } else {
            // should never be a placeholder!
            assert(0);
          }
        } else {
          // ok if we don't find it - upstream xd's can be destroyed before
          //  the downstream xd has stopped updating it
        }
      }
      if(xd) {
        xd->update_next_bytes_read(port_idx, span_start, span_size);
        log_xd_ref.info("xd=%llx, before rm_ref refcount=%u", xd_guid,
                        xd->reference_count.load_acquire());
        xd->remove_reference();
      }
    } else {
      // send a active message to remote node
      UpdateBytesReadMessage::send_request(execution_node, xd_guid, port_idx, span_start,
                                           span_size);
    }
  }

  void XferDesQueue::destroy_xferDes(XferDesID guid)
  {
    XferDes *xd = 0;
    {
      AutoLock<> al(guid_lock);
      std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(guid);
      if(it != guid_to_xd.end()) {
        if((it->second & 1) == 0) {
          // remember xd but remove from table (stealing table's reference)
          xd = reinterpret_cast<XferDes *>(it->second);
          guid_to_xd.erase(it);
          log_xd_ref.info(
              "destroy xd=%llx, update_pre_bytes_total_received=%u, expected=%u", guid,
              xd->nb_update_pre_bytes_total_calls_received.load_acquire(),
              xd->nb_update_pre_bytes_total_calls_expected);
        } else {
          // should never be a placeholder!
          assert(0);
        }
      } else {
        // should always be present!
        assert(0);
      }
    }
    // just remove table's reference (actual destruction may be delayed
    //   if some other thread is still poking it)
    xd->remove_reference();
  }

  bool XferDesQueue::enqueue_xferDes_local(XferDes *xd, bool add_to_queue /*= true*/)
  {
    Event wait_on = xd->request_metadata();
    if(!wait_on.has_triggered()) {
      log_new_dma.info() << "xd metadata wait: xd=" << xd->guid << " ready=" << wait_on;
      xd->deferred_enqueue.defer(this, xd, wait_on);
      return false;
    }

    // insert ourselves in the table, replacing a placeholder if present
    XferDesPlaceholder *ph = 0;
    {
      AutoLock<> al(guid_lock);
      std::map<XferDesID, uintptr_t>::iterator it = guid_to_xd.find(xd->guid);
      if(it != guid_to_xd.end()) {
        if((it->second & 1) == 0) {
          // should never be a real xd!
          assert(0);
          guid_to_xd.erase(it);
        } else {
          // remember placeholder (stealing table's reference)
          ph = reinterpret_cast<XferDesPlaceholder *>(it->second - 1);
          // put xd in, donating the initial reference to the table
          it->second = reinterpret_cast<uintptr_t>(xd);
          log_xd_ref.info("xd=%llx, swap placeholder, refcount=%u", xd->guid,
                          xd->reference_count.load_acquire());
        }
      } else {
        guid_to_xd.insert(std::make_pair(xd->guid, reinterpret_cast<uintptr_t>(xd)));
        log_xd_ref.info("xd=%llx, new xd, refcount=%u", xd->guid,
                        xd->reference_count.load_acquire());
      }
    }
    if(ph) {
      // tell placeholder about real xd and have it update it once there
      //  are no other concurrent updates
      ph->set_real_xd(xd);
      ph->remove_reference();
    }

    if(!add_to_queue)
      return true;
    assert(0);

    return true;
  }

  void XferDes::DeferredXDEnqueue::defer(XferDesQueue *_xferDes_queue, XferDes *_xd,
                                         Event wait_on)
  {
    xferDes_queue = _xferDes_queue;
    xd = _xd;
    Realm::EventImpl::add_waiter(wait_on, this);
  }

  void XferDes::DeferredXDEnqueue::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // TODO: handle poisoning
    assert(!poisoned);
    log_new_dma.info() << "xd metadata ready: xd=" << xd->guid;
    xd->channel->enqueue_ready_xd(xd);
    // xferDes_queue->enqueue_xferDes_local(xd);
  }

  void XferDes::DeferredXDEnqueue::print(std::ostream &os) const
  {
    os << "deferred xd enqueue: xd=" << xd->guid;
  }

  Event XferDes::DeferredXDEnqueue::get_finish_event(void) const
  {
    // TODO: would be nice to provide dma op's finish event here
    return Event::NO_EVENT;
  }

  void destroy_xfer_des(XferDesID _guid)
  {
    log_new_dma.info("Destroy XferDes: id(" IDFMT ")", _guid);
    NodeID execution_node = _guid >> (XferDesQueue::NODE_BITS + XferDesQueue::INDEX_BITS);
    if(execution_node == Network::my_node_id) {
      XferDesQueue::get_singleton()->destroy_xferDes(_guid);
    } else {
      XferDesDestroyMessage::send_request(execution_node, _guid);
    }
  }

  ActiveMessageHandlerReg<SimpleXferDesCreateMessage>
      simple_xfer_des_create_message_handler;
  ActiveMessageHandlerReg<NotifyXferDesCompleteMessage> notify_xfer_des_complete_handler;
  ActiveMessageHandlerReg<XferDesDestroyMessage> xfer_des_destroy_message_handler;
  ActiveMessageHandlerReg<UpdateBytesTotalMessage> update_bytes_total_message_handler;
  ActiveMessageHandlerReg<UpdateBytesWriteMessage> update_bytes_write_message_handler;
  ActiveMessageHandlerReg<UpdateBytesReadMessage> update_bytes_read_message_handler;
  ActiveMessageHandlerReg<RemoteWriteXferDes::Write1DMessage>
      remote_write_1d_message_handler;

}; // namespace Realm
