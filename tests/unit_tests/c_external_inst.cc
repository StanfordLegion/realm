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

class CExternalInstanceBaseTest {
protected:
  void initialize(int num_nodes)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplMachineModel>();
    runtime_impl->init(num_nodes);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl{nullptr};
};

// test realm_external_instance_resource_create and
// realm_external_instance_resource_destroy

class CExternalInstanceResourceCreateDestroyTest : public CExternalInstanceBaseTest,
                                                   public ::testing::Test {
protected:
  void SetUp() override { CExternalInstanceBaseTest::initialize(1); }

  void TearDown() override { CExternalInstanceBaseTest::finalize(); }
};

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateNullRuntime)
{
  realm_external_instance_resource_t resource;
  realm_external_instance_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;

  realm_status_t status =
      realm_external_instance_resource_create(nullptr, &params, &resource);

  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateNullParams)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, nullptr, &resource);

  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_PARAMS);
}

// UBSAN will report an error if the type is not
// REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY, so we need to skip this test under
// UBSAN.
#ifndef UBSAN_ENABLED
TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateInvalidType)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_instance_resource_create_params_t params;
  params.type = static_cast<realm_external_instance_resource_type_t>(
      REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_NUM + 1);

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);

  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_TYPE);
}
#endif

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateSystemMemoryInvalidBase)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = nullptr;
  params.size = 1024;
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_BASE);
}

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateSystemMemoryInvalidSize)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  int dummy = 0;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = reinterpret_cast<const void *>(&dummy);
  params.size = 0;
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_SIZE);
}

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateSystemMemoryNormal)
{
  int dummy;
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = reinterpret_cast<const void *>(&dummy);
  params.size = sizeof(int);
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_external_instance_resource_destroy(runtime, resource));
}

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateSystemMemoryReadOnly)
{
  size_t size = sizeof(int) * 1024;
  void *base = malloc(size);
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 1;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_external_instance_resource_destroy(runtime, resource));
  free(base);
}

TEST_F(CExternalInstanceResourceCreateDestroyTest, CreateSystemMemoryDestroy)
{
  size_t size = sizeof(int) * 1024;
  void *base = malloc(size);
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 1;
  ASSERT_REALM(realm_external_instance_resource_create(runtime, &params, &resource));

  realm_status_t status = realm_external_instance_resource_destroy(runtime, resource);
  EXPECT_EQ(status, REALM_SUCCESS);
  free(base);
}

// TODO: we need core module to make this test work
TEST_F(CExternalInstanceResourceCreateDestroyTest,
       DISABLED_CreateSystemMemorySuggestedMemory)
{
  size_t size = sizeof(int) * 1024;
  void *base = malloc(size);
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_system_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_SYSTEM_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 0;
  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_SUCCESS);

  realm_memory_t memory;
  ASSERT_REALM(
      realm_external_instance_resource_suggested_memory(runtime, resource, &memory));
  EXPECT_EQ(memory, REALM_NO_MEM);

  ASSERT_REALM(realm_external_instance_resource_destroy(runtime, resource));
  free(base);
}

// test realm_region_instance_create with external instance resource

class CExternalInstanceCreateTest : public CExternalInstanceBaseTest,
                                    public ::testing::Test {
protected:
  void SetUp() override { CExternalInstanceBaseTest::initialize(1); }

  void TearDown() override { CExternalInstanceBaseTest::finalize(); }
};

class REALM_PUBLIC_API MockExternalInstanceResource : public ExternalInstanceResource {
public:
  MockExternalInstanceResource(uintptr_t _base, size_t _size_in_bytes,
                               bool _read_only = false)
    : ExternalInstanceResource(REALM_HASH_TOKEN(MockExternalInstanceResource))
    , base(_base)
    , size_in_bytes(_size_in_bytes)
    , read_only(_read_only)
  {}

  virtual bool satisfies(const InstanceLayoutGeneric &layout) const { return true; }

  // returns the suggested memory in which this resource should be created
  Memory suggested_memory() const { return Memory::NO_MEMORY; }

  virtual ExternalInstanceResource *clone(void) const
  {
    return new MockExternalInstanceResource(base, size_in_bytes, read_only);
  }

protected:
  virtual void print(std::ostream &os) const {}

public:
  uintptr_t base;
  size_t size_in_bytes;
  bool read_only;
};

TEST_F(CExternalInstanceCreateTest, CreateInvalidResource)
{
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
  MockExternalInstanceResource *resource = new MockExternalInstanceResource(0, 1024);
  params.external_resource =
      reinterpret_cast<realm_external_instance_resource_t>(resource); // invalid resource
  realm_event_t event;

  realm_status_t status = realm_region_instance_create(runtime, &params, nullptr,
                                                       REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_TYPE);
  delete resource;
}
