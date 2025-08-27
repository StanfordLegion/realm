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

#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

class CProfilingTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImpl>();
    runtime_impl->init(1);
  }

  void TearDown() override { runtime_impl->finalize(); }

  std::unique_ptr<MockRuntimeImpl> runtime_impl{nullptr};
};

// use task spawn to test the ProfilingRequestSet

TEST_F(CProfilingTest, ProfilingRequestSetInvalidProcessor)
{
  realm_profiling_measurement_id_t measurement_ids[2];
  measurement_ids[0] = realm_profiling_measurement_id_t::PMID_OP_TIMELINE;
  measurement_ids[1] = realm_profiling_measurement_id_t::PMID_OP_STATUS;

  realm_profiling_request_t request[1];
  request[0].response_proc = REALM_NO_PROC;
  request[0].response_task_id = 0;
  request[0].priority = 0;
  request[0].report_if_empty = 0;
  request[0].measurement_ids = measurement_ids;
  request[0].num_measurements = 2;
  request[0].payload = NULL;
  request[0].payload_size = 0;

  realm_processor_t proc = ID::make_processor(0, 0).convert<Realm::Processor>();
  realm_runtime_t runtime = *runtime_impl;
  realm_event_t event;
  constexpr realm_task_func_id_t task_id = 0;
  realm_status_t status = realm_processor_spawn(runtime, proc, task_id, nullptr, 0,
                                                request, 1, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CProfilingTest, ProfilingRequestSetInvalidMeasurement)
{
  realm_processor_t proc = ID::make_processor(0, 0).convert<Realm::Processor>();
  realm_profiling_request_t request[1];
  request[0].response_proc = proc;
  request[0].response_task_id = 0;
  request[0].priority = 0;
  request[0].report_if_empty = 0;
  request[0].measurement_ids = nullptr;
  request[0].num_measurements = 0;
  request[0].payload = NULL;
  request[0].payload_size = 0;

  realm_runtime_t runtime = *runtime_impl;
  realm_event_t event;
  constexpr realm_task_func_id_t task_id = 0;
  realm_status_t status = realm_processor_spawn(runtime, proc, task_id, nullptr, 0,
                                                request, 1, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_MEASUREMENT);
}

static void test_task_func(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, realm_processor_t proc_id)
{}

struct ResponseUserdata {
  bool *response_task_func_called;
  bool *args_empty;
};

static void response_task_func(const void *args, size_t arglen, const void *userdata,
                               size_t userlen, realm_processor_t proc_id)
{
  ProfilingResponse resp(args, arglen);
  const ResponseUserdata *result =
      static_cast<const ResponseUserdata *>(resp.user_data());
  *result->response_task_func_called = true;
  *result->args_empty = (arglen == 0);
}

// enable it once we get the spawn and register task working with unit tests
TEST_F(CProfilingTest, DISABLED_ProfilingRequestSetResponseTaskSpawned)
{
  // TODO: setup processor for task spawn
  realm_processor_t proc;
  realm_profiling_measurement_id_t measurement_ids[2];
  measurement_ids[0] = realm_profiling_measurement_id_t::PMID_OP_TIMELINE;
  measurement_ids[1] = realm_profiling_measurement_id_t::PMID_OP_STATUS;

  bool response_task_func_called = false;
  bool args_empty = false;
  ResponseUserdata userdata;
  userdata.response_task_func_called = &response_task_func_called;
  userdata.args_empty = &args_empty;

  realm_profiling_request_t request[1];
  request[0].response_proc = ID::make_processor(0, 0).convert<Realm::Processor>();
  request[0].response_task_id = 0;
  request[0].priority = 0;
  request[0].report_if_empty = 0;
  request[0].measurement_ids = measurement_ids;
  request[0].num_measurements = 2;
  request[0].payload = &userdata;
  request[0].payload_size = sizeof(ResponseUserdata);

  realm_runtime_t runtime = *runtime_impl;
  realm_event_t event;
  constexpr realm_task_func_id_t test_task_id = 0;
  constexpr realm_task_func_id_t response_task_id = 1;
  ASSERT_REALM(realm_processor_register_task_by_kind(
      nullptr, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, test_task_id, test_task_func,
      nullptr, 0, &event));

  ASSERT_REALM(realm_processor_register_task_by_kind(
      nullptr, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, response_task_id,
      response_task_func, nullptr, 0, &event));

  realm_status_t status = realm_processor_spawn(runtime, proc, test_task_id, nullptr, 0,
                                                request, 1, REALM_NO_EVENT, 0, &event);
  EXPECT_EQ(response_task_func_called, true);
  EXPECT_NE(args_empty, true);
}

class MockProfilingMeasurementCollection : public Realm::ProfilingMeasurementCollection {
public:
  // The requests inside ProfilingRequestSet is protected, so we need to pass them as a
  // vector here
  void send_responses_to_buffer(const std::vector<ProfilingRequest *> &requests,
                                std::vector<std::vector<char>> &buffers) const
  {
    for(std::vector<ProfilingRequest *>::const_iterator it = requests.begin();
        it != requests.end(); it++) {
      // only send responses for things that we didn't send eagerly on completion
      if(measurements_left.count(*it) == 0)
        continue;

      std::vector<char> buffer;
      dump_to_buffer(**it, buffer);
      buffers.push_back(buffer);
    }
  }
};

// We test the failed case first, then we test the success case for each measurement type

TEST_F(CProfilingTest, ProfilingResponseHasMeasurementNullResponse)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      nullptr, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_RESPONSE);
}

TEST_F(CProfilingTest, ProfilingResponseHasMeasurementNullData)
{
  realm_profiling_response_t response;
  response.data = nullptr;
  response.data_size = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_RESPONSE);
}

TEST_F(CProfilingTest, ProfilingResponseHasMeasurementZeroDataSize)
{
  realm_profiling_response_t response;
  response.data = nullptr;
  response.data_size = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_RESPONSE);
}

TEST_F(CProfilingTest, ProfilingResponseHasMeasurementInvalidMeasurement)
{
  // Arrange
  // create a fake reponse buffer
  std::vector<int> buffer{
      1, static_cast<int>(Realm::ProfilingMeasurementID::PMID_OP_TIMELINE), 0, 0};
  realm_profiling_response_t response;
  response.data = buffer.data();
  response.data_size = buffer.size() * sizeof(int);

  // Act and Assert
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_REALM_LAST, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_MEASUREMENT);
}

TEST_F(CProfilingTest, ProfilingResponseHasMeasurementNoMeasurement)
{
  // Arrange
  // create a fake reponse buffer
  std::vector<int> buffer{
      1, static_cast<int>(Realm::ProfilingMeasurementID::PMID_OP_TIMELINE), 0, 0};
  realm_profiling_response_t response;
  response.data = buffer.data();
  response.data_size = buffer.size() * sizeof(int);

  // Act and Assert
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_STATUS, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_MEASUREMENT);
}

TEST_F(CProfilingTest, ProfilingResponseGetMeasurementNullResponse)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      nullptr, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_RESPONSE);
}

// now we test the success case for each measurement type

class CProfilingTest_MeasurementBase : public CProfilingTest {
protected:
  void SetUp() override { CProfilingTest::SetUp(); }

  void TearDown() override { CProfilingTest::TearDown(); }

  template <typename T>
  void setup_measurement(T &measurement)
  {
    MockProfilingMeasurementCollection measurements;
    ProfilingRequestSet prs;
    ProfilingRequest &pr = prs.add_request(
        Realm::Processor(ID::make_processor(0, 0).convert<Realm::Processor>()), 0, 0, 0,
        false);
    pr.add_measurement(T::ID);
    measurements.import_requests(prs);
    measurements.add_measurement(measurement, false);
    std::vector<ProfilingRequest *> requests = {&pr};
    measurements.send_responses_to_buffer(requests, buffers);
    response.data = buffers[0].data();
    response.data_size = buffers[0].size();
  }

  realm_profiling_response_t response;
  std::vector<std::vector<char>> buffers;
};

// OperationBacktrace

class CProfilingTest_OperationBacktrace : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    backtrace.pcs.push_back(0x1234);
    backtrace.pcs.push_back(0x5678);
    backtrace.symbols.push_back("symbol");
    backtrace.symbols.push_back("symbol2");
    CProfilingTest_MeasurementBase::setup_measurement(backtrace);
  }

  ProfilingMeasurements::OperationBacktrace backtrace;
};

TEST_F(CProfilingTest_OperationBacktrace, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_PCS, nullptr,
      nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktrace)
{
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE, nullptr,
      &result_count);
  // should be invalid measurement because we need to use PMID_OP_BACKTRACE_PCS or
  // PMID_OP_BACKTRACE_SYMBOLS instead of PMID_OP_BACKTRACE
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_MEASUREMENT);
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktracePCsSmallBuffer)
{
  size_t result_count = backtrace.pcs.size() - 1;
  uintptr_t pcs[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_PCS, pcs,
      &result_count);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_BUFFER);
  EXPECT_EQ(result_count, backtrace.pcs.size());
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktracePCs)
{
  size_t result_count = backtrace.pcs.size();
  uintptr_t pcs[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_PCS, pcs,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, backtrace.pcs.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(pcs[i], backtrace.pcs[i]);
  }
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktracePCsCount)
{
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_PCS, nullptr,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, backtrace.pcs.size());
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktraceSymbols)
{
  size_t result_count = backtrace.symbols.size();
  realm_profiling_operation_backtrace_symbol_t symbols[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_SYMBOLS, symbols,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, backtrace.symbols.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(symbols[i].symbol, backtrace.symbols[i]);
  }
}

TEST_F(CProfilingTest_OperationBacktrace,
       ProfilingResponseGetMeasurementOperationBacktraceSymbolsCount)
{
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_BACKTRACE_SYMBOLS, nullptr,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, backtrace.symbols.size());
}

// OperationTimeline

class CProfilingTest_OperationTimeline : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    timeline.create_time = 1;
    timeline.ready_time = 2;
    timeline.start_time = 3;
    timeline.end_time = 4;
    timeline.complete_time = 5;
    CProfilingTest_MeasurementBase::setup_measurement(timeline);
  }

  ProfilingMeasurements::OperationTimeline timeline;
};

TEST_F(CProfilingTest_OperationTimeline, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationTimeline, ProfilingResponseGetMeasurementOperationTimeline)
{
  realm_profiling_operation_timeline_t op_timeline_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE, &op_timeline_result,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(op_timeline_result.create_time, timeline.create_time);
  EXPECT_EQ(op_timeline_result.ready_time, timeline.ready_time);
  EXPECT_EQ(op_timeline_result.start_time, timeline.start_time);
  EXPECT_EQ(op_timeline_result.end_time, timeline.end_time);
  EXPECT_EQ(op_timeline_result.complete_time, timeline.complete_time);
}

// OperationTimelineGPU

class CProfilingTest_OperationTimelineGPU : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    timeline.start_time = 1;
    timeline.end_time = 2;
    CProfilingTest_MeasurementBase::setup_measurement(timeline);
  }

  ProfilingMeasurements::OperationTimelineGPU timeline;
};

TEST_F(CProfilingTest_OperationTimelineGPU, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE_GPU, nullptr,
      nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationTimelineGPU,
       ProfilingResponseGetMeasurementOperationTimelineGPU)
{
  realm_profiling_operation_timeline_gpu_t op_timeline_gpu_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_TIMELINE_GPU,
      &op_timeline_gpu_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(op_timeline_gpu_result.start_time, timeline.start_time);
  EXPECT_EQ(op_timeline_gpu_result.end_time, timeline.end_time);
}

// OperationEventWaits

class CProfilingTest_OperationEventWaits : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    event_waits.intervals.push_back(
        ProfilingMeasurements::OperationEventWaits::WaitInterval{
            1, 2, 3, ID::make_event(0, 0, 0).convert<Realm::Event>()});
    event_waits.intervals.push_back(
        ProfilingMeasurements::OperationEventWaits::WaitInterval{
            4, 5, 6, ID::make_event(0, 0, 1).convert<Realm::Event>()});
    CProfilingTest_MeasurementBase::setup_measurement(event_waits);
  }

  ProfilingMeasurements::OperationEventWaits event_waits;
};

TEST_F(CProfilingTest_OperationEventWaits, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_EVENT_WAITS, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationEventWaits,
       ProfilingResponseGetMeasurementOperationEventWaits)
{
  size_t result_count = event_waits.intervals.size();
  realm_profiling_operation_event_wait_interval_t op_event_waits_result[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_EVENT_WAITS,
      op_event_waits_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, event_waits.intervals.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(op_event_waits_result[i].wait_start, event_waits.intervals[i].wait_start);
    EXPECT_EQ(op_event_waits_result[i].wait_ready, event_waits.intervals[i].wait_ready);
    EXPECT_EQ(op_event_waits_result[i].wait_end, event_waits.intervals[i].wait_end);
    EXPECT_EQ(op_event_waits_result[i].wait_event, event_waits.intervals[i].wait_event);
  }
}

TEST_F(CProfilingTest_OperationEventWaits,
       ProfilingResponseGetMeasurementOperationEventWaitsSmallBuffer)
{
  size_t result_count = event_waits.intervals.size() - 1;
  realm_profiling_operation_event_wait_interval_t op_event_waits_result[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_EVENT_WAITS,
      op_event_waits_result, &result_count);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_BUFFER);
  EXPECT_EQ(result_count, event_waits.intervals.size());
}

TEST_F(CProfilingTest_OperationEventWaits,
       ProfilingResponseGetMeasurementOperationEventWaitsNullResult)
{
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_EVENT_WAITS, nullptr,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, event_waits.intervals.size());
}

// OperationProcessorUsage

class CProfilingTest_OperationProcessorUsage : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    processor_usage.proc = ID::make_processor(0, 0).convert<Realm::Processor>();
    CProfilingTest_MeasurementBase::setup_measurement(processor_usage);
  }

  void TearDown() override { CProfilingTest_MeasurementBase::TearDown(); }

  ProfilingMeasurements::OperationProcessorUsage processor_usage;
};

TEST_F(CProfilingTest_OperationProcessorUsage, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_PROC_USAGE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationProcessorUsage,
       ProfilingResponseGetMeasurementOperationProcessorUsage)
{
  realm_profiling_operation_processor_usage_t op_processor_usage_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_PROC_USAGE,
      &op_processor_usage_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(op_processor_usage_result.proc, processor_usage.proc.id);
}

// OperationMemoryUsage

class CProfilingTest_OperationMemoryUsage : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    memory_usage.source = ID::make_memory(0, 0).convert<Realm::Memory>();
    memory_usage.target = ID::make_memory(0, 1).convert<Realm::Memory>();
    memory_usage.size = 123;
    CProfilingTest_MeasurementBase::setup_measurement(memory_usage);
  }

  ProfilingMeasurements::OperationMemoryUsage memory_usage;
};

TEST_F(CProfilingTest_OperationMemoryUsage, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_MEM_USAGE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_OperationMemoryUsage,
       ProfilingResponseGetMeasurementOperationMemoryUsage)
{
  realm_profiling_operation_memory_usage_t op_memory_usage_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_MEM_USAGE,
      &op_memory_usage_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(op_memory_usage_result.source, memory_usage.source.id);
  EXPECT_EQ(op_memory_usage_result.target, memory_usage.target.id);
  EXPECT_EQ(op_memory_usage_result.size, memory_usage.size);
}

// InstanceTimeline

class CProfilingTest_InstanceTimeline : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    instance_timeline.instance =
        ID::make_instance(0, 0, 0, 0).convert<Realm::RegionInstance>();
    instance_timeline.create_time = 1;
    instance_timeline.ready_time = 2;
    instance_timeline.delete_time = 3;
    CProfilingTest_MeasurementBase::setup_measurement(instance_timeline);
  }

  ProfilingMeasurements::InstanceTimeline instance_timeline;
};

TEST_F(CProfilingTest_InstanceTimeline, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_INST_TIMELINE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_InstanceTimeline, ProfilingResponseGetMeasurementInstanceTimeline)
{
  realm_profiling_instance_timeline_t inst_timeline_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_INST_TIMELINE,
      &inst_timeline_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(inst_timeline_result.instance, instance_timeline.instance.id);
  EXPECT_EQ(inst_timeline_result.create_time, instance_timeline.create_time);
  EXPECT_EQ(inst_timeline_result.ready_time, instance_timeline.ready_time);
  EXPECT_EQ(inst_timeline_result.delete_time, instance_timeline.delete_time);
}

// InstanceMemoryUsage

class CProfilingTest_InstanceMemoryUsage : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    instance_memory_usage.instance =
        ID::make_instance(0, 0, 0, 0).convert<Realm::RegionInstance>();
    instance_memory_usage.memory = ID::make_memory(0, 0).convert<Realm::Memory>();
    instance_memory_usage.bytes = 123;
    CProfilingTest_MeasurementBase::setup_measurement(instance_memory_usage);
  }

  ProfilingMeasurements::InstanceMemoryUsage instance_memory_usage;
};

TEST_F(CProfilingTest_InstanceMemoryUsage, ProfilingResponseHasMeasurementSuccess)
{
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_INST_MEM_USAGE, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CProfilingTest_InstanceMemoryUsage,
       ProfilingResponseGetMeasurementInstanceMemoryUsage)
{
  realm_profiling_instance_memory_usage_t inst_mem_usage_result;
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_INST_MEM_USAGE,
      &inst_mem_usage_result, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(inst_mem_usage_result.instance, instance_memory_usage.instance.id);
  EXPECT_EQ(inst_mem_usage_result.memory, instance_memory_usage.memory.id);
  EXPECT_EQ(inst_mem_usage_result.bytes, instance_memory_usage.bytes);
}

// OperationCopyInfo

class CProfilingTest_CopyInfo : public CProfilingTest_MeasurementBase {
protected:
  void SetUp() override
  {
    CProfilingTest_MeasurementBase::SetUp();
    ProfilingMeasurements::OperationCopyInfo::InstInfo inst_info = {
        {ID::make_instance(0, 0, 0, 0).convert<Realm::RegionInstance>(),
         ID::make_instance(0, 0, 0, 1).convert<Realm::RegionInstance>()},
        {ID::make_instance(0, 0, 0, 2).convert<Realm::RegionInstance>(),
         ID::make_instance(0, 0, 0, 3).convert<Realm::RegionInstance>()},
        ID::make_instance(0, 0, 0, 4).convert<Realm::RegionInstance>(),
        ID::make_instance(0, 0, 0, 5).convert<Realm::RegionInstance>(),
        {0, 1},
        {2, 3},
        123,
        456,
        ProfilingMeasurements::OperationCopyInfo::RequestType::COPY};
    copy_info.inst_info.push_back(inst_info);
    CProfilingTest_MeasurementBase::setup_measurement(copy_info);
  }

  ProfilingMeasurements::OperationCopyInfo copy_info;
};

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoMeasurementCount)
{
  size_t result_count = 0;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_COPY_INFO, nullptr,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info.size());
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoSmallBuffer)
{
  size_t result_count = copy_info.inst_info[0].src_insts.size() - 1;
  realm_region_instance_t insts[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, MAKE_OP_COPY_INFO_SRC_INST_ID(0), insts, &result_count);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_BUFFER);
  EXPECT_EQ(result_count, copy_info.inst_info[0].src_insts.size());
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoSrcInsts)
{
  size_t result_count = copy_info.inst_info[0].src_insts.size();
  realm_region_instance_t insts[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, MAKE_OP_COPY_INFO_SRC_INST_ID(0), insts, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info[0].src_insts.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(insts[i], copy_info.inst_info[0].src_insts[i].id);
  }
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoDstInsts)
{
  size_t result_count = copy_info.inst_info[0].dst_insts.size();
  realm_region_instance_t insts[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, MAKE_OP_COPY_INFO_DST_INST_ID(0), insts, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info[0].dst_insts.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(insts[i], copy_info.inst_info[0].dst_insts[i].id);
  }
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoSrcFields)
{
  size_t result_count = copy_info.inst_info[0].src_fields.size();
  realm_field_id_t field_ids[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, MAKE_OP_COPY_INFO_SRC_FIELD_ID(0), field_ids, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info[0].src_fields.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(field_ids[i], copy_info.inst_info[0].src_fields[i]);
  }
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoDstFields)
{
  size_t result_count = copy_info.inst_info[0].dst_fields.size();
  realm_field_id_t field_ids[result_count];
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, MAKE_OP_COPY_INFO_DST_FIELD_ID(0), field_ids, &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info[0].dst_fields.size());
  for(size_t i = 0; i < result_count; i++) {
    EXPECT_EQ(field_ids[i], copy_info.inst_info[0].dst_fields[i]);
  }
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoOther)
{
  size_t result_count = copy_info.inst_info.size();
  realm_profiling_operation_copy_info_t inst_info_result;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_COPY_INFO, &inst_info_result,
      &result_count);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(result_count, copy_info.inst_info.size());
  EXPECT_EQ(inst_info_result.src_indirection_inst,
            copy_info.inst_info[0].src_indirection_inst.id);
  EXPECT_EQ(inst_info_result.dst_indirection_inst,
            copy_info.inst_info[0].dst_indirection_inst.id);
  EXPECT_EQ(inst_info_result.src_indirection_field,
            copy_info.inst_info[0].src_indirection_field);
  EXPECT_EQ(inst_info_result.dst_indirection_field,
            copy_info.inst_info[0].dst_indirection_field);
  EXPECT_EQ(inst_info_result.request_type, copy_info.inst_info[0].request_type);
  EXPECT_EQ(inst_info_result.num_hops, copy_info.inst_info[0].num_hops);
}

TEST_F(CProfilingTest_CopyInfo, GetCopyInfoOtherSmallBuffer)
{
  size_t result_count = copy_info.inst_info.size() - 1;
  realm_profiling_operation_copy_info_t inst_info_result;
  realm_status_t status = realm_profiling_response_get_measurement(
      &response, realm_profiling_measurement_id_t::PMID_OP_COPY_INFO, &inst_info_result,
      &result_count);
  EXPECT_EQ(status, REALM_PROFILING_ERROR_INVALID_BUFFER);
  EXPECT_EQ(result_count, copy_info.inst_info.size());
}