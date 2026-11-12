/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
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

// INCLDUED FROM profiling.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/profiling.h"
#include "realm/serialize.h"

TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurementID);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationTimeline);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationTimelineGPU);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationEventWaits::WaitInterval);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationMemoryUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationProcessorUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationFinishEvent);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::InstanceAllocResult);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::InstanceMemoryUsage);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::InstanceTimeline);
template <Realm::ProfilingMeasurementID _ID>
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::CachePerfCounters<_ID>);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::IPCPerfCounters);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::TLBPerfCounters);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::BranchPredictionPerfCounters);
TYPE_IS_SERIALIZABLE(Realm::ProfilingMeasurements::OperationCopyInfo::RequestType);

#include "realm/timers.h"

namespace Realm {

  namespace ProfilingMeasurements {

    TYPE_IS_SERIALIZABLE(OperationStatus::Result);

    template <typename S>
    bool serdez(S &serdez, const OperationStatus &s)
    {
      return ((serdez & s.result) && (serdez & s.error_code) &&
              (serdez & s.error_details));
    }

    template <typename S>
    bool serdez(S &serdez, const OperationAbnormalStatus &s)
    {
      return ((serdez & s.result) && (serdez & s.error_code) &&
              (serdez & s.error_details));
    }

    TYPE_IS_SERIALIZABLE(InstanceStatus::Result);

    template <typename S>
    bool serdez(S &serdez, const InstanceStatus &s)
    {
      return ((serdez & s.result) && (serdez & s.error_code) &&
              (serdez & s.error_details));
    }

    template <typename S>
    bool serdez(S &serdez, const InstanceAbnormalStatus &s)
    {
      return ((serdez & s.result) && (serdez & s.error_code) &&
              (serdez & s.error_details));
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationBacktrace
    //

    template <typename S>
    bool serdez(S &serdez, const OperationBacktrace &b)
    {
      return ((serdez & b.pcs) && (serdez & b.symbols));
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationTimeLineGPU
    //
    inline bool OperationTimelineGPU::is_valid(void) const
    {
      return ((start_time != INVALID_TIMESTAMP) && (end_time != INVALID_TIMESTAMP));
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationTimeLine
    //

    inline void OperationTimeline::record_create_time(void)
    {
      create_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_ready_time(void)
    {
      ready_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_start_time(void)
    {
      start_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_end_time(void)
    {
      end_time = Clock::current_time_in_nanoseconds();
    }

    inline void OperationTimeline::record_complete_time(void)
    {
      complete_time = Clock::current_time_in_nanoseconds();
    }

    inline bool OperationTimeline::is_valid(void) const
    {
      return ((create_time != INVALID_TIMESTAMP) && (ready_time != INVALID_TIMESTAMP) &&
              (start_time != INVALID_TIMESTAMP) && (end_time != INVALID_TIMESTAMP) &&
              (complete_time != INVALID_TIMESTAMP));
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationEventWaits
    //

    template <typename S>
    bool serdez(S &serdez, const OperationEventWaits &w)
    {
      return (serdez & w.intervals);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationEventWaits::WaitInterval
    //

    inline void OperationEventWaits::WaitInterval::record_wait_start(void)
    {
      wait_start = Clock::current_time_in_nanoseconds();
    }

    inline void OperationEventWaits::WaitInterval::record_wait_ready(void)
    {
      wait_ready = Clock::current_time_in_nanoseconds();
    }

    inline void OperationEventWaits::WaitInterval::record_wait_end(void)
    {
      wait_end = Clock::current_time_in_nanoseconds();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct InstanceTimeLine
    //

    inline void InstanceTimeline::record_create_time(void)
    {
      create_time = Clock::current_time_in_nanoseconds();
    }

    inline void InstanceTimeline::record_ready_time(void)
    {
      ready_time = Clock::current_time_in_nanoseconds();
    }

    inline void InstanceTimeline::record_delete_time(void)
    {
      delete_time = Clock::current_time_in_nanoseconds();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationCopyInfo
    //

    template <typename S>
    bool serialize(S &ser, const OperationCopyInfo &c)
    {
      bool success = false;
      success = (ser & c.inst_info.size());
      if(!success)
        return false;
      for(size_t i = 0; i < c.inst_info.size(); i++) {
        success = (ser & c.inst_info[i].src_insts) && (ser & c.inst_info[i].dst_insts) &&
                  (ser & c.inst_info[i].src_indirection_inst) &&
                  (ser & c.inst_info[i].dst_indirection_inst) &&
                  (ser & c.inst_info[i].src_fields) &&
                  (ser & c.inst_info[i].dst_fields) &&
                  (ser & c.inst_info[i].src_indirection_field) &&
                  (ser & c.inst_info[i].dst_indirection_field) &&
                  (ser & c.inst_info[i].request_type) && (ser & c.inst_info[i].num_hops);
      }
      return success;
    }

    template <typename S>
    bool deserialize(S &dez, OperationCopyInfo &c)
    {
      bool success = false;
      size_t len = 0;
      success = (dez & len);
      if(!success)
        return false;
      c.inst_info.resize(len);
      for(size_t i = 0; i < c.inst_info.size(); i++) {
        success = (dez & c.inst_info[i].src_insts) && (dez & c.inst_info[i].dst_insts) &&
                  (dez & c.inst_info[i].src_indirection_inst) &&
                  (dez & c.inst_info[i].dst_indirection_inst) &&
                  (dez & c.inst_info[i].src_fields) &&
                  (dez & c.inst_info[i].dst_fields) &&
                  (dez & c.inst_info[i].src_indirection_field) &&
                  (dez & c.inst_info[i].dst_indirection_field) &&
                  (dez & c.inst_info[i].request_type) && (dez & c.inst_info[i].num_hops);
      }
      return success;
    }

  }; // namespace ProfilingMeasurements

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingMeasurementSet
  //

  inline size_t ProfilingMeasurementSet::size(void) const
  {
    size_t count = high_ids.size();
    for(unsigned long long bits = low_bits; bits != 0; bits &= bits - 1)
      count++;
    return count;
  }

  inline bool ProfilingMeasurementSet::contains(ProfilingMeasurementID id) const
  {
    if((id >= 0) && (id < LOW_LIMIT))
      return ((low_bits >> id) & 1ULL) != 0;
    return std::binary_search(high_ids.begin(), high_ids.end(), id);
  }

  inline void ProfilingMeasurementSet::insert(ProfilingMeasurementID id)
  {
    if((id >= 0) && (id < LOW_LIMIT)) {
      low_bits |= (1ULL << id);
      return;
    }
    std::vector<ProfilingMeasurementID>::iterator it =
        std::lower_bound(high_ids.begin(), high_ids.end(), id);
    if((it == high_ids.end()) || (*it != id))
      high_ids.insert(it, id);
  }

  inline void ProfilingMeasurementSet::insert_all(const ProfilingMeasurementSet &other)
  {
    low_bits |= other.low_bits;
    for(ProfilingMeasurementID id : other.high_ids)
      insert(id);
  }

  inline void ProfilingMeasurementSet::clear(void)
  {
    low_bits = 0;
    high_ids.clear();
  }

  template <typename F>
  inline void ProfilingMeasurementSet::for_each(F f) const
  {
    // (any negative IDs sort before the bitmask's)
    std::vector<ProfilingMeasurementID>::const_iterator it = high_ids.begin();
    for(; (it != high_ids.end()) && (*it < 0); ++it)
      f(*it);
    int id = 0;
    for(unsigned long long bits = low_bits; bits != 0; bits >>= 1, id++)
      if((bits & 1ULL) != 0)
        f(static_cast<ProfilingMeasurementID>(id));
    for(; it != high_ids.end(); ++it)
      f(*it);
  }

  // same wire format as a std::set of the IDs
  template <typename S>
  bool serialize(S &s, const ProfilingMeasurementSet &ms)
  {
    size_t len = ms.size();
    if(!(s << len))
      return false;
    bool ok = true;
    ms.for_each([&](ProfilingMeasurementID id) {
      if(ok && !(s << id))
        ok = false;
    });
    return ok;
  }

  template <typename S>
  bool deserialize(S &s, ProfilingMeasurementSet &ms)
  {
    size_t len;
    if(!(s >> len))
      return false;
    ms.clear();
    for(size_t i = 0; i < len; i++) {
      ProfilingMeasurementID id;
      if(!(s >> id))
        return false;
      ms.insert(id);
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequest
  //

  template <typename T>
  ProfilingRequest &ProfilingRequest::add_measurement(void)
  {
    // SJT: the typecast here is a NOP, but somehow it avoids weird linker errors
    requested_measurements.insert(static_cast<ProfilingMeasurementID>(T::ID));
    return *this;
  }

  inline ProfilingRequest &
  ProfilingRequest::add_measurement(ProfilingMeasurementID measurement_id)
  {
    requested_measurements.insert(measurement_id);
    return *this;
  }

  inline ProfilingRequest &ProfilingRequest::add_measurements(
      const std::set<ProfilingMeasurementID> &measurement_ids)
  {
    for(ProfilingMeasurementID id : measurement_ids)
      requested_measurements.insert(id);
    return *this;
  }

  template <typename S>
  bool serialize(S &s, const ProfilingRequest &pr)
  {
    return ((s << pr.response_proc) && (s << pr.response_task_id) && (s << pr.priority) &&
            (s << pr.report_if_empty) && (s << pr.user_data) &&
            (s << pr.requested_measurements) && (s << pr.batched_delivery));
  }

  template <typename S>
  /*static*/ ProfilingRequest *ProfilingRequest::deserialize_new(S &s)
  {
    // have to get fields of the reqeuest in order to build it
    Processor p;
    Processor::TaskFuncID fid;
    int priority;
    bool report_if_empty;
    if(!(s >> p))
      return 0;
    if(!(s >> fid))
      return 0;
    if(!(s >> priority))
      return 0;
    if(!(s >> report_if_empty))
      return 0;
    ProfilingRequest *pr = new ProfilingRequest(p, fid, priority, report_if_empty);
    if(!(s >> pr->user_data) || !(s >> pr->requested_measurements) ||
       !(s >> pr->batched_delivery)) {
      delete pr;
      return 0;
    }
    return pr;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingMeasurementCollection
  //

  template <typename T>
  bool ProfilingMeasurementCollection::wants_measurement(void) const
  {
    return wanted.contains(static_cast<ProfilingMeasurementID>(T::ID));
  }

  template <typename T>
  void
  ProfilingMeasurementCollection::add_measurement(const T &data,
                                                  bool send_complete_responses /*= true*/)
  {
    const ProfilingMeasurementID id = static_cast<ProfilingMeasurementID>(T::ID);
    if(!wanted.contains(id)) {
      // caller probably should have asked if we wanted this before measuring it...
      return;
    }

    // no duplicates for now
    assert(find_measurement(id) == nullptr);

    // serialize the data onto the end of our buffer
    Serialization::ByteCountSerializer bcs;
#ifndef NDEBUG
    bool ok =
#endif
        bcs << data;
    assert(ok);
    const size_t size = bcs.bytes_used();
    const size_t offset = this->data.size();
    if(this->data.capacity() == 0)
      this->data.reserve(256);
    this->data.resize(offset + size);
    Serialization::FixedBufferSerializer fbs(this->data.data() + offset, size);
#ifndef NDEBUG
    ok =
#endif
        fbs << data;
    assert(ok);
    measurements.push_back(Measurement{id, offset, size});
    // (were there a duplicate, the first one recorded would be sent)

    // update the number of remaining measurements for each profiling request that wanted
    //  this - if the count hits zero, we can either send the request immediately or mark
    //  that we want to later
    for(PendingRequest &pr : requests) {
      if(!pr.pending || !pr.request->requested_measurements.contains(id))
        continue;
      assert(pr.measurements_left > 0);
      if(--pr.measurements_left == 0) {
        if(send_complete_responses) {
          pr.pending = false;
          send_response(*pr.request);
        } else {
          completed_requests_present = true;
        }
      }
    }

    // while we're here, if we're allowed to send responses, see if there are any deferred
    // ones
    if(send_complete_responses && completed_requests_present) {
      for(PendingRequest &pr : requests) {
        if(pr.pending && (pr.measurements_left == 0)) {
          pr.pending = false;
          send_response(*pr.request);
        }
      }
      completed_requests_present = false;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequestSet
  //

  template <typename S>
  bool serialize(S &s, const ProfilingRequestSet &prs)
  {
    size_t len = prs.requests.size();
    if(!(s << len))
      return false;
    for(size_t i = 0; i < len; i++)
      if(!(s << *prs.requests[i]))
        return false;
    return true;
  }

  template <typename S>
  bool deserialize(S &s, ProfilingRequestSet &prs)
  {
    size_t len;
    if(!(s >> len))
      return false;
    prs.clear(); // erase any existing data cleanly
    prs.requests.reserve(len);
    for(size_t i = 0; i < len; i++) {
      ProfilingRequest *pr = ProfilingRequest::deserialize_new(s);
      if(!pr)
        return false;
      prs.requests.push_back(pr);
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingResponse
  //

  template <typename T>
  inline bool ProfilingResponse::has_measurement(void) const
  {
    int offset, size; // not actually used
    return find_id(static_cast<int>(T::ID), offset, size);
  }

  template <typename T>
  inline T *ProfilingResponse::get_measurement(void) const
  {
    int offset, size;
    if(find_id(static_cast<int>(T::ID), offset, size)) {
      Serialization::FixedBufferDeserializer fbd(data + offset, size);
      T *m = new T;
#ifndef NDEBUG
      bool ok =
#endif
          fbd >> *m;
      assert(ok);
      return m;
    } else
      return 0;
  }

  template <typename T>
  inline bool ProfilingResponse::get_measurement(T &result) const
  {
    int offset, size;
    if(find_id(static_cast<int>(T::ID), offset, size)) {
      Serialization::FixedBufferDeserializer fbd(data + offset, size);
      return (fbd >> result);
    } else
      return false;
  }

}; // namespace Realm
