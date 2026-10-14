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

// MSVC doesn't support the C++17 aligned_alloc
#if defined(_WIN32)
#define ALIGNED_ALLOC(s, a) _aligned_malloc(s, a)
#define ALIGNED_FREE(p) _aligned_free(p)
#else
#define ALIGNED_ALLOC(s, a) aligned_alloc(s, a)
#define ALIGNED_FREE(p) free(p)
#endif

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

class CExternalInstBaseTest {
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

class CCreateInstFromExternalInstanceResourceTest : public CExternalInstBaseTest,
                                                    public ::testing::Test {
protected:
  void SetUp() override
  {
    CExternalInstBaseTest::initialize(1);

    // add a processor and a memory
    runtime_impl->setup_mock_proc_mems(
        MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
            {{0, Processor::Kind::LOC_PROC, 0}},
            {{0, Memory::Kind::SYSTEM_MEM, 1024, 0}},
            {{0, 0, 1000, 1}}});
  }

  void TearDown() override
  {
    realm_runtime_t runtime = *runtime_impl;
    CExternalInstBaseTest::finalize();
  }
};

TEST_F(CCreateInstFromExternalInstanceResourceTest, InvalidExternalResource)
{
  realm_external_resource_t external_resource;
  external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_MAX; // invalid resource type
  realm_region_instance_t inst;
  realm_runtime_t runtime = *runtime_impl;
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
  params.external_resource = &external_resource;
  realm_event_t event;
  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr, 0,
                                                       REALM_NO_EVENT, &inst, &event);
  EXPECT_EQ(status, REALM_EXTERNAL_RESOURCE_ERROR_INVALID_TYPE);
}

TEST_F(CCreateInstFromExternalInstanceResourceTest, InvalidBase)
{
  realm_external_resource_t external_resource;
  external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY;
  external_resource.resource.system_memory.base = nullptr;
  external_resource.resource.system_memory.size = 1024;
  external_resource.resource.system_memory.read_only = 0;
  MockMemoryImpl *mem_impl =
      static_cast<MockMemoryImpl *>(runtime_impl->nodes[0].memories[0]);
  realm_region_instance_t inst;
  realm_runtime_t runtime = *runtime_impl;
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
  params.external_resource = &external_resource;
  realm_event_t event;
  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr, 0,
                                                       REALM_NO_EVENT, &inst, &event);
  EXPECT_EQ(status, REALM_EXTERNAL_RESOURCE_ERROR_INVALID_BASE);
}

TEST_F(CCreateInstFromExternalInstanceResourceTest, InvalidSize)
{
  const size_t MIN_ALIGN = 32;
  const size_t EXTERNAL_MEM_SIZE = 1024;
  void *external_mem = ALIGNED_ALLOC(MIN_ALIGN, EXTERNAL_MEM_SIZE);
  ASSERT_TRUE(external_mem != nullptr);
  realm_external_resource_t external_resource;
  external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY;
  external_resource.resource.system_memory.base = external_mem;
  external_resource.resource.system_memory.size = 0;
  external_resource.resource.system_memory.read_only = 0;
  MockMemoryImpl *mem_impl =
      static_cast<MockMemoryImpl *>(runtime_impl->nodes[0].memories[0]);
  realm_region_instance_t inst;
  realm_runtime_t runtime = *runtime_impl;
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
  params.external_resource = &external_resource;
  realm_event_t event;
  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr, 0,
                                                       REALM_NO_EVENT, &inst, &event);
  EXPECT_EQ(status, REALM_EXTERNAL_RESOURCE_ERROR_INVALID_SIZE);
  ALIGNED_FREE(external_mem);
}

TEST_F(CCreateInstFromExternalInstanceResourceTest, CreateSuccess)
{
  const size_t MIN_ALIGN = 32;
  const size_t EXTERNAL_MEM_SIZE = 1024;
  void *external_mem = ALIGNED_ALLOC(MIN_ALIGN, EXTERNAL_MEM_SIZE);
  ASSERT_TRUE(external_mem != nullptr);
  realm_external_resource_t external_resource;
  external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY;
  external_resource.resource.system_memory.base = external_mem;
  external_resource.resource.system_memory.size = EXTERNAL_MEM_SIZE;
  external_resource.resource.system_memory.read_only = 0;
  MockMemoryImpl *mem_impl =
      static_cast<MockMemoryImpl *>(runtime_impl->nodes[0].memories[0]);
  realm_region_instance_t inst;
  realm_runtime_t runtime = *runtime_impl;
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
  params.external_resource = &external_resource;
  realm_event_t event;
  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr, 0,
                                                       REALM_NO_EVENT, &inst, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(RegionInstance(inst).exists(), true);
  ASSERT_REALM(realm_region_instance_destroy(runtime, inst, REALM_NO_EVENT));
  ALIGNED_FREE(external_mem);
}

class CExternalInstGenerateExternalResourceInfoTest : public CExternalInstBaseTest,
                                                      public ::testing::Test {
protected:
  void SetUp() override
  {
    CExternalInstBaseTest::initialize(1);

    // add a processor and a memory
    runtime_impl->setup_mock_proc_mems(
        MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
            {{0, Processor::Kind::LOC_PROC, 0}},
            {{0, Memory::Kind::SYSTEM_MEM, 1024, 0}},
            {{0, 0, 1000, 1}}});

    realm_runtime_t runtime = *runtime_impl;
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
    ASSERT_REALM(realm_region_instance_create(runtime, &params, nullptr, 0,
                                              REALM_NO_EVENT, &inst, &event));
  }

  void TearDown() override
  {
    realm_runtime_t runtime = *runtime_impl;
    ASSERT_REALM(realm_region_instance_destroy(runtime, inst, REALM_NO_EVENT));
    CExternalInstBaseTest::finalize();
  }

  realm_region_instance_t inst;
};

TEST_F(CExternalInstGenerateExternalResourceInfoTest, NullRuntime)
{
  realm_runtime_t runtime = nullptr;
  realm_external_resource_t external_resource;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  realm_index_space_t ispace;
  ispace.coord_type = REALM_COORD_TYPE_INT;
  ispace.num_dims = 1;
  ispace.lower_bound = lower_bound;
  ispace.upper_bound = upper_bound;
  realm_status_t status = realm_region_instance_generate_external_resource_info(
      runtime, inst, &ispace, field_ids, 1, 0, &external_resource);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CExternalInstGenerateExternalResourceInfoTest, NullIndexSpace)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_resource_t external_resource;
  int field_ids[1] = {0};
  realm_status_t status = realm_region_instance_generate_external_resource_info(
      runtime, inst, nullptr, field_ids, 1, 0, &external_resource);
  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CExternalInstGenerateExternalResourceInfoTest, NullInstance)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_resource_t external_resource;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  realm_index_space_t ispace;
  ispace.coord_type = REALM_COORD_TYPE_INT;
  ispace.num_dims = 1;
  ispace.lower_bound = lower_bound;
  ispace.upper_bound = upper_bound;
  realm_status_t status = realm_region_instance_generate_external_resource_info(
      runtime, REALM_NO_INST, &ispace, field_ids, 1, 0, &external_resource);
  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE);
}

TEST_F(CExternalInstGenerateExternalResourceInfoTest, NullResource)
{
  realm_runtime_t runtime = *runtime_impl;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  realm_index_space_t ispace;
  ispace.coord_type = REALM_COORD_TYPE_INT;
  ispace.num_dims = 1;
  ispace.lower_bound = lower_bound;
  ispace.upper_bound = upper_bound;
  realm_status_t status = realm_region_instance_generate_external_resource_info(
      runtime, inst, &ispace, field_ids, 1, 0, nullptr);
  EXPECT_EQ(status, REALM_EXTERNAL_RESOURCE_ERROR_INVALID_RESOURCE);
}

TEST_F(CExternalInstGenerateExternalResourceInfoTest, GenerateSuccess)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_resource_t external_resource;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  int field_ids[1] = {0};
  realm_index_space_t ispace;
  ispace.coord_type = REALM_COORD_TYPE_INT;
  ispace.num_dims = 1;
  ispace.lower_bound = lower_bound;
  ispace.upper_bound = upper_bound;
  realm_status_t status = realm_region_instance_generate_external_resource_info(
      runtime, inst, &ispace, field_ids, 1, 0, &external_resource);
  EXPECT_EQ(status, REALM_SUCCESS);
  MockMemoryImpl *mem_impl =
      static_cast<MockMemoryImpl *>(runtime_impl->nodes[0].memories[0]);
  EXPECT_EQ(external_resource.resource.system_memory.base,
            reinterpret_cast<void *>(mem_impl->buffer.data()));
  EXPECT_EQ(external_resource.resource.system_memory.size, mem_impl->buffer.size());
  EXPECT_EQ(external_resource.resource.system_memory.read_only, 0);
}