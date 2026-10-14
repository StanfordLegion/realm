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

    inline realm_status_t OperationBacktrace::to_c(uintptr_t *result, size_t *result_count) const
    {
      realm_status_t status = REALM_SUCCESS;
      if(result != nullptr) {
        if (pcs.size() > *result_count) {
          status = REALM_PROFILING_ERROR_INVALID_BUFFER;
        } else {
          memcpy(result, pcs.data(), pcs.size() * sizeof(uintptr_t));
        }
      }
      *result_count = pcs.size();
      return status;
    }

    inline realm_status_t OperationBacktrace::to_c(realm_profiling_operation_backtrace_symbol_t *result, size_t *result_count) const
    {
      *result_count = symbols.size();
      if(result != nullptr) {
        realm_profiling_operation_backtrace_symbol_t *symbol_result =
            reinterpret_cast<realm_profiling_operation_backtrace_symbol_t *>(
                result);
        for(size_t i = 0; i < symbols.size(); i++) {
          if(symbols[i].size() > PMID_OP_BACKTRACE_SYMBOLS_MAX_LENGTH) {
            return REALM_PROFILING_ERROR_INVALID_BUFFER;
          }
          memcpy(symbol_result[i].symbol, symbols[i].c_str(), symbols[i].size());
          symbol_result[i].symbol[symbols[i].size()] = '\0';
        }
      }
      return REALM_SUCCESS;
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationTimeLineGPU
    //
    inline bool OperationTimelineGPU::is_valid(void) const
    {
      return ((start_time != INVALID_TIMESTAMP) && (end_time != INVALID_TIMESTAMP));
    }

    inline realm_status_t OperationTimelineGPU::to_c(realm_profiling_operation_timeline_gpu_t *result) const
    {
      if(result) {
        result->start_time = start_time;
        result->end_time = end_time;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
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

    inline realm_status_t OperationTimeline::to_c(realm_profiling_operation_timeline_t *result) const
    {
      if(result) {
        result->create_time = create_time;
        result->ready_time = ready_time;
        result->start_time = start_time;
        result->end_time = end_time;
        result->complete_time = complete_time;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
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

    inline realm_status_t OperationEventWaits::to_c(realm_profiling_operation_event_wait_interval_t *result, size_t *result_count) const
    {
      realm_status_t status = REALM_SUCCESS;
      if(result != nullptr) {
        if (intervals.size() > *result_count) {
          status = REALM_PROFILING_ERROR_INVALID_BUFFER;
        } else {
          for(size_t i = 0; i < intervals.size(); i++) {
            const Realm::ProfilingMeasurements::OperationEventWaits::WaitInterval
                &wait_interval = intervals[i];
            result[i].wait_start = wait_interval.wait_start;
            result[i].wait_ready = wait_interval.wait_ready;
            result[i].wait_end = wait_interval.wait_end;
            result[i].wait_event = wait_interval.wait_event;
          }
        }
      }
      *result_count = intervals.size();
      return status;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationProcessorUsage
    //

    inline realm_status_t OperationProcessorUsage::to_c(realm_profiling_operation_processor_usage_t *result) const
    {
      if(result) {
        result->proc = proc.id;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct OperationMemoryUsage
    //

    inline realm_status_t OperationMemoryUsage::to_c(realm_profiling_operation_memory_usage_t *result) const
    {
      if(result) {
        result->source = source.id;
        result->target = target.id;
        result->size = size;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
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

    inline realm_status_t InstanceTimeline::to_c(realm_profiling_instance_timeline_t *result) const
    {
      if(result) {
        result->instance = instance.id;
        result->create_time = create_time;
        result->ready_time = ready_time;
        result->delete_time = delete_time;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct InstanceMemoryUsage
    //

    inline realm_status_t InstanceMemoryUsage::to_c(realm_profiling_instance_memory_usage_t *result) const
    {
      if(result) {
        result->instance = instance.id;
        result->memory = memory.id;
        result->bytes = bytes;
        return REALM_SUCCESS;
      }
      return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
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

    inline realm_status_t OperationCopyInfo::to_c(realm_profiling_operation_copy_info_t *result, size_t *result_count) const
    {
      realm_status_t status = REALM_SUCCESS;
      if(result != nullptr) {
        if (inst_info.size() > *result_count) {
          status = REALM_PROFILING_ERROR_INVALID_BUFFER;
        } else {
          for(size_t i = 0; i < inst_info.size(); i++) {
            result[i].src_indirection_inst = inst_info[i].src_indirection_inst.id;
            result[i].dst_indirection_inst = inst_info[i].dst_indirection_inst.id;
            result[i].src_indirection_field = inst_info[i].src_indirection_field;
            result[i].dst_indirection_field = inst_info[i].dst_indirection_field;
            result[i].request_type =
                static_cast<realm_profiling_operation_copy_info_request_type_t>(
                    inst_info[i].request_type);
            result[i].num_hops = inst_info[i].num_hops;
          }
        }
      }
      *result_count = inst_info.size();
      return status;
    }

    inline realm_status_t OperationCopyInfo::to_c(realm_region_instance_t *result, size_t *result_count, const void *data) const
    {
      realm_profiling_measurement_id_t measurement_id =
          *(static_cast<const realm_profiling_measurement_id_t *>(data));
      bool is_dst = measurement_id >= PMID_OP_COPY_INFO_DST_INST;
      int index = measurement_id - (is_dst ? PMID_OP_COPY_INFO_DST_INST
                                          : PMID_OP_COPY_INFO_SRC_INST);
      if((index < 0) || (static_cast<size_t>(index) >= inst_info.size())) {
        return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
      }
      const Realm::ProfilingMeasurements::OperationCopyInfo::InstInfo &inst_info_item =
          inst_info[index];
      const std::vector<Realm::RegionInstance> &inst_vec =
          (is_dst ? inst_info_item.dst_insts : inst_info_item.src_insts);

      realm_status_t status = REALM_SUCCESS;
      if(result != nullptr) {
        if (inst_vec.size() > *result_count) {
          status = REALM_PROFILING_ERROR_INVALID_BUFFER;
        } else {
          memcpy(result, inst_vec.data(), inst_vec.size() * sizeof(inst_vec[0]));
        }
      }
      *result_count = inst_vec.size();
      return status;
    }

    inline realm_status_t OperationCopyInfo::to_c(realm_field_id_t *result, size_t *result_count, const void *data) const
    {
      realm_profiling_measurement_id_t measurement_id =
          *(static_cast<const realm_profiling_measurement_id_t *>(data));
      bool is_dst = measurement_id >= PMID_OP_COPY_INFO_DST_FIELD;
      int index = measurement_id - (is_dst ? PMID_OP_COPY_INFO_DST_FIELD
                                          : PMID_OP_COPY_INFO_SRC_FIELD);

      if((index < 0) || (static_cast<size_t>(index) >= inst_info.size())) {
        return REALM_PROFILING_ERROR_INVALID_MEASUREMENT;
      }

      const Realm::ProfilingMeasurements::OperationCopyInfo::InstInfo &inst_info_item =
          inst_info[index];
      const std::vector<Realm::FieldID> &field_vec =
          (is_dst ? inst_info_item.dst_fields : inst_info_item.src_fields);

      realm_status_t status = REALM_SUCCESS;
      if(result != nullptr) {
        if (field_vec.size() > *result_count) {
          status = REALM_PROFILING_ERROR_INVALID_BUFFER;
        } else {
          memcpy(result, field_vec.data(), field_vec.size() * sizeof(field_vec[0]));
        }
      }
      *result_count = field_vec.size();
      return status;
    }

  }; // namespace ProfilingMeasurements

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
    requested_measurements.insert(measurement_ids.begin(), measurement_ids.end());
    return *this;
  }

  inline ProfilingRequest &ProfilingRequest::add_measurements(const ProfilingMeasurementID *measurement_ids, size_t num_measurements)
  {
    static_assert(sizeof(::realm_profiling_measurement_id_t) ==
                         sizeof(ProfilingMeasurementID),
                         "C and C++ profiling measurement enums must have the same size");

    static_assert(std::is_same<
          std::underlying_type<realm_profiling_measurement_id_t>::type,
          std::underlying_type<ProfilingMeasurementID>::type>::value,
        "Underlying enum types must match");
    requested_measurements.insert(measurement_ids, measurement_ids + num_measurements);
    return *this;
  }

  template <typename S>
  bool serialize(S &s, const ProfilingRequest &pr)
  {
    return ((s << pr.response_proc) && (s << pr.response_task_id) && (s << pr.priority) &&
            (s << pr.report_if_empty) && (s << pr.user_data) &&
            (s << pr.requested_measurements));
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
    if(!(s >> pr->user_data) || !(s >> pr->requested_measurements)) {
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
    return requested_measurements.count(static_cast<ProfilingMeasurementID>(T::ID));
  }

  template <typename T>
  void
  ProfilingMeasurementCollection::add_measurement(const T &data,
                                                  bool send_complete_responses /*= true*/)
  {
    std::map<ProfilingMeasurementID,
             std::vector<const ProfilingRequest *>>::const_iterator it =
        requested_measurements.find(static_cast<ProfilingMeasurementID>(T::ID));
    if(it == requested_measurements.end()) {
      // caller probably should have asked if we wanted this before measuring it...
      return;
    }

    // no duplicates for now
    assert(measurements.count(static_cast<ProfilingMeasurementID>(T::ID)) == 0);

    // serialize the data
    Serialization::DynamicBufferSerializer dbs(128);
#ifndef NDEBUG
    bool ok =
#endif
        dbs << data;
    assert(ok);

    // measurement data is stored in a ByteArray
    ByteArray &md = measurements[static_cast<ProfilingMeasurementID>(T::ID)];
    ByteArray b = dbs.detach_bytearray(-1); // no trimming
    md.swap(b);                             // avoids a copy

    // update the number of remaining measurements for each profiling request that wanted
    // this
    //  if the count hits zero, we can either send the request immediately or mark that we
    //  want to later
    for(std::vector<const ProfilingRequest *>::const_iterator it2 = it->second.begin();
        it2 != it->second.end(); it2++) {
      std::map<const ProfilingRequest *, int>::iterator it3 =
          measurements_left.find(*it2);
      assert(it3 != measurements_left.end());
      it3->second--;
      if(it3->second == 0) {
        if(send_complete_responses) {
          measurements_left.erase(it3);
          send_response(**it2);
        } else {
          completed_requests_present = true;
        }
      }
    }

    // while we're here, if we're allowed to send responses, see if there are any deferred
    // ones
    if(send_complete_responses && completed_requests_present) {
      std::map<const ProfilingRequest *, int>::iterator it = measurements_left.begin();
      while(it != measurements_left.end()) {
        if(it->second > 0) {
          it++;
          continue;
        }

        // make a copy of the iterator so we can increment it
        std::map<const ProfilingRequest *, int>::iterator old = it;
        it++;
        send_response(*(old->first));
        measurements_left.erase(old);
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
