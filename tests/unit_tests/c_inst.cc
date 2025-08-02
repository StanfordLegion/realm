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
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

class MockRuntimeImplInst : public MockRuntimeImplMachineModel {
public:
  MockRuntimeImplInst(void)
    : MockRuntimeImplMachineModel()
  {}

  void init(int num_nodes)
  {
    MockRuntimeImplMachineModel::init(num_nodes);
    repl_heap.init(16 << 20, 1 /*chunks*/);
    local_event_free_list = new LocalEventTableAllocator::FreeList(local_events, 0);
  }

  void finalize(void)
  {
    delete local_event_free_list;
    local_event_free_list = nullptr;
    repl_heap.cleanup();
    MockRuntimeImplMachineModel::finalize();
  }
};

class CInstBaseTest {
protected:
  void initialize(int num_nodes)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplInst>();
    runtime_impl->init(num_nodes);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImplInst> runtime_impl{nullptr};
};

// test realm_region_instance_create and realm_region_instance_destroy

class CInstCreateDestroyTest : public CInstBaseTest, public ::testing::Test {
protected:
  void SetUp() override { CInstBaseTest::initialize(1); }

  void TearDown() override { CInstBaseTest::finalize(); }
};

TEST_F(CInstCreateDestroyTest, CreateNullRuntime)
{
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params{
      .memory = REALM_NO_MEM,
      .lower_bound = nullptr,
      .upper_bound = nullptr,
      .num_dims = 0,
      .field_ids = nullptr,
  };
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(nullptr, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CInstCreateDestroyTest, CreateNullParams)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, nullptr, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS);
}

TEST_F(CInstCreateDestroyTest, CreateInvalidMemory)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  params.memory = REALM_NO_MEM;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CInstCreateDestroyTest, CreateInvalidLowerBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int upper_bound[1] = {10};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = nullptr;
  params.upper_bound = upper_bound;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateDestroyTest, CreateInvalidUpperBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int lower_bound[1] = {10};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = lower_bound;
  params.upper_bound = nullptr;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateDestroyTest, CreateZeroDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 0;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateDestroyTest, CreateOverMaxDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = REALM_MAX_DIM + 1;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateDestroyTest, CreateNullFieldIds)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.field_ids = nullptr;
  params.field_sizes = field_sizes;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCreateDestroyTest, CreateNullFieldSizes)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  int field_ids[1] = {0};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.field_ids = field_ids;
  params.field_sizes = nullptr;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCreateDestroyTest, CreateZeroField)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.num_fields = 0;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

// UBSAN will report an error if the coord_type is not REALM_COORD_TYPE_LONG_LONG or
// REALM_COORD_TYPE_INT, so we need to skip this test under UBSAN.
#ifndef UBSAN_ENABLED
TEST_F(CInstCreateDestroyTest, CreateInvalidCoordType)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  int field_ids[1] = {0};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.num_fields = 1;
  params.field_ids = field_ids;
  params.field_sizes = field_sizes;
  params.coord_type = static_cast<realm_coord_type_t>(REALM_COORD_TYPE_NUM + 1);
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_COORD_TYPE);
}
#endif

TEST_F(CInstCreateDestroyTest, CreateNullEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int bound[1] = {10};
  int field_ids[1] = {0};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.num_fields = 1;
  params.field_ids = field_ids;
  params.field_sizes = field_sizes;
  params.coord_type = REALM_COORD_TYPE_INT;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, nullptr);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_EVENT);
}

TEST_F(CInstCreateDestroyTest, CreateSuccess)
{
  // add a processor and a memory
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024, 0}},
          {{0, 0, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = lower_bound;
  params.upper_bound = upper_bound;
  params.coord_type = REALM_COORD_TYPE_INT;
  params.num_dims = 1;
  params.num_fields = 1;
  params.field_ids = field_ids;
  params.field_sizes = field_sizes;
  params.external_resource = nullptr;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(RegionInstance(inst).exists(), true);

  ASSERT_REALM(realm_region_instance_destroy(runtime, inst, event));
}

TEST_F(CInstCreateDestroyTest, DestroySuccess)
{
  // add a processor and a memory
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024, 0}},
          {{0, 0, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = lower_bound;
  params.upper_bound = upper_bound;
  params.coord_type = REALM_COORD_TYPE_INT;
  params.num_dims = 1;
  params.num_fields = 1;
  params.field_ids = field_ids;
  params.field_sizes = field_sizes;
  params.external_resource = nullptr;
  realm_event_t event;
  ASSERT_REALM(realm_region_instance_create(runtime, &params, nullptr, REALM_NO_EVENT,
                                            &inst, &event));

  realm_status_t status = realm_region_instance_destroy(runtime, inst, event);
  EXPECT_EQ(status, REALM_SUCCESS);
}

// TODO: currently, we poison the event if allocation failed, but it needs a refactor of
// GenEventImpl::create_genevent() and GenEventImpl::trigger;
TEST_F(CInstCreateDestroyTest, DISABLED_CreateFailedTooLarge)
{
  // add a processor and a memory
  runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024, 0}},
          {{0, 0, 1000, 1}}});

  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_region_instance_create_params_t params;
  int lower_bound[1] = {0};
  int upper_bound[1] = {2000};
  int field_ids[1] = {0};
  size_t field_sizes[1] = {sizeof(int)};
  params.memory = ID::make_memory(0, 0).convert<Memory>();
  params.lower_bound = lower_bound;
  params.upper_bound = upper_bound;
  params.coord_type = REALM_COORD_TYPE_INT;
  params.num_dims = 1;
  params.num_fields = 1;
  params.field_ids = field_ids;
  params.field_sizes = field_sizes;
  params.external_resource = nullptr;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(RegionInstance(inst).exists(), true);

  ASSERT_REALM(realm_region_instance_destroy(runtime, inst, event));
}
