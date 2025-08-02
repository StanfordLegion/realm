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

class CExternalInstanceCudaBaseTest {
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

// test realm_region_instance_create and realm_region_instance_destroy

class CExternalInstanceCudaCreateDestroyTest : public CExternalInstanceCudaBaseTest,
                                               public ::testing::Test {
protected:
  void SetUp() override { CExternalInstanceCudaBaseTest::initialize(1); }

  void TearDown() override { CExternalInstanceCudaBaseTest::finalize(); }
};

TEST_F(CExternalInstanceCudaCreateDestroyTest, CreateCUDAMemoryInvalidBase)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_cuda_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_CUDA_MEMORY;
  params.cuda_device_id = 0;
  params.base = nullptr;
  params.size = 1024;
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_BASE);
}

TEST_F(CExternalInstanceCudaCreateDestroyTest, CreateCUDAMemoryInvalidSize)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_cuda_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_CUDA_MEMORY;
  params.cuda_device_id = 0;
  params.base = reinterpret_cast<const void *>(0x1234); // use a fake address
  params.size = 0;
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_EXTERNAL_INSTANCE_RESOURCE_ERROR_INVALID_SIZE);
}

TEST_F(CExternalInstanceCudaCreateDestroyTest, CreateCUDAMemoryNormal)
{
  size_t size = sizeof(int) * 1024;
  void *base = reinterpret_cast<void *>(0x1234);
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_cuda_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_CUDA_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 0;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_external_instance_resource_destroy(runtime, resource));
}

TEST_F(CExternalInstanceCudaCreateDestroyTest, CreateCUDAMemoryReadOnly)
{
  size_t size = sizeof(int) * 1024;
  void *base = reinterpret_cast<void *>(0x1234);
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_cuda_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_CUDA_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 1;

  realm_status_t status =
      realm_external_instance_resource_create(runtime, &params, &resource);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_external_instance_resource_destroy(runtime, resource));
}

TEST_F(CExternalInstanceCudaCreateDestroyTest, CreateCUDAMemoryDestroy)
{
  size_t size = sizeof(int) * 1024;
  void *base = reinterpret_cast<void *>(0x1234);
  ;
  realm_runtime_t runtime = *runtime_impl;
  realm_external_instance_resource_t resource;
  realm_external_cuda_memory_resource_create_params_t params;
  params.type = REALM_EXTERNAL_INSTANCE_RESOURCE_TYPE_CUDA_MEMORY;
  params.base = base;
  params.size = size;
  params.read_only = 1;
  ASSERT_REALM(realm_external_instance_resource_create(runtime, &params, &resource));

  realm_status_t status = realm_external_instance_resource_destroy(runtime, resource);
  EXPECT_EQ(status, REALM_SUCCESS);
}
