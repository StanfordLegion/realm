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
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

class CRuntimeAttrTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImpl>();
    runtime_impl->init(2);
    num_nodes = runtime_impl->num_nodes;
    my_node_id = runtime_impl->my_node_id;
  }

  void TearDown() override { runtime_impl->finalize(); }

  std::unique_ptr<MockRuntimeImpl> runtime_impl{nullptr};
  realm_address_space_t num_nodes = 0;
  realm_address_space_t my_node_id = 0;
};

TEST_F(CRuntimeAttrTest, NullRuntime)
{
  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_ADDRESS_SPACE};
  uint64_t values[1];
  realm_status_t status = realm_runtime_get_attributes(nullptr, attrs, values, 1);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CRuntimeAttrTest, InvalidAttribute)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_MAX};
  uint64_t values[1];
  realm_status_t status = realm_runtime_get_attributes(runtime, attrs, values, 1);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_INVALID_ATTRIBUTE);
}

TEST_F(CRuntimeAttrTest, ZeroAttributes)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_MAX};
  uint64_t values[1];
  realm_status_t status = realm_runtime_get_attributes(runtime, attrs, values, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CRuntimeAttrTest, GetAttributesAddressSpace)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_ADDRESS_SPACE};
  uint64_t values[1];

  realm_status_t status = realm_runtime_get_attributes(runtime, attrs, values, 1);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(values[0], num_nodes);
}

TEST_F(CRuntimeAttrTest, GetAttributesLocalAddressSpace)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE};
  uint64_t values[1];

  realm_status_t status = realm_runtime_get_attributes(runtime, attrs, values, 1);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(values[0], my_node_id);
}