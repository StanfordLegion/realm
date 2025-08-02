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
#include "realm/transfer/transfer.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <gtest/gtest.h>

#include <type_traits> // for std::integral_constant

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

// test realm_region_instance_copy

class CInstCopyTest : public ::testing::Test {
protected:
  void SetUp() override { initialize(1); }

  void TearDown() override { finalize(); }

  void initialize(int num_nodes)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImpl>();
    runtime_impl->init(num_nodes);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImpl> runtime_impl{nullptr};
};

TEST_F(CInstCopyTest, CopyNullRuntime)
{
  realm_region_instance_copy_params_t params = {nullptr, nullptr, 0, nullptr, nullptr, 0};
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(nullptr, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CInstCopyTest, CopyNullParams)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, nullptr, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS);
}

TEST_F(CInstCopyTest, CopyNullSrcs)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t dsts;
  params.srcs = nullptr;
  params.dsts = &dsts;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCopyTest, CopyNullDsts)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs;
  params.srcs = &srcs;
  params.dsts = nullptr;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCopyTest, CopySrcsInvalidInstance)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {REALM_NO_INST, 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE);
}

TEST_F(CInstCopyTest, CopyDstsInvalidInstance)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {REALM_NO_INST, 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE);
}

TEST_F(CInstCopyTest, CopySrcsInvalidFieldSize)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 0};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCopyTest, CopyDstsInvalidFieldSize)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 0};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCopyTest, CopyZeroNumFields)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 0;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCopyTest, CopyInvalidLowerBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  params.lower_bound = nullptr;
  int upper_bound[1] = {10};
  params.upper_bound = upper_bound;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCopyTest, CopyInvalidUpperBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  int lower_bound[1] = {10};
  params.lower_bound = lower_bound;
  params.upper_bound = nullptr;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCopyTest, CopyZeroDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  int bound[1] = {10};
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 0;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCopyTest, CopyOverMaxDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  int bound[1] = {10};
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = REALM_MAX_DIM + 1;
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

// UBSAN will report an error if the coord_type is not REALM_COORD_TYPE_LONG_LONG or
// REALM_COORD_TYPE_INT, so we need to skip this test under UBSAN.
#ifndef UBSAN_ENABLED
TEST_F(CInstCopyTest, CopyInvalidCoordType)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_copy_params_t params;
  realm_copy_src_dst_field_t srcs = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  realm_copy_src_dst_field_t dsts = {
      ID::make_instance(0, 0, 0, 0).convert<RegionInstance>(), 0, 10};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = 1;
  int bound[1] = {10};
  params.lower_bound = bound;
  params.upper_bound = bound;
  params.num_dims = 1;
  params.coord_type = static_cast<realm_coord_type_t>(REALM_COORD_TYPE_NUM + 1);
  realm_event_t event;

  realm_status_t status =
      realm_region_instance_copy(runtime, &params, nullptr, REALM_NO_EVENT, 0, &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_COORD_TYPE);
}
#endif

// Template mock machinery
template <int N>
std::function<Realm::Event(
    const IndexSpace<N, int> &idx_space, const std::vector<CopySrcDstField> &srcs,
    const std::vector<CopySrcDstField> &dsts,
    const std::vector<const typename CopyIndirection<N, int>::Base *> &indirects,
    const Realm::ProfilingRequestSet &reqs, Event wait_on, int priority)>
    dynamic_mock_impl_N;

template <int N>
Realm::Event mock_copy_redirect_N(
    const IndexSpace<N, int> &is, const std::vector<CopySrcDstField> &srcs,
    const std::vector<CopySrcDstField> &dsts,
    const std::vector<const typename CopyIndirection<N, int>::Base *> &indirects,
    const Realm::ProfilingRequestSet &reqs, Event wait_on, int priority)
{
  return dynamic_mock_impl_N<N>(is, srcs, dsts, indirects, reqs, wait_on, priority);
}

// Explicit instantiations for CopyImplRouter
template <>
CopyImplFn<1, int> CopyImplRouter<1, int>::impl;
template <>
CopyImplFn<2, int> CopyImplRouter<2, int>::impl;
template <>
CopyImplFn<3, int> CopyImplRouter<3, int>::impl;

// Wrap each dimension as a type
using DimTypes =
    ::testing::Types<std::integral_constant<int, 1>, std::integral_constant<int, 2>,
                     std::integral_constant<int, 3>>;

template <typename DimWrapper>
class CInstCopyTestN : public CInstCopyTest {
protected:
  static constexpr int N = DimWrapper::value;
};

TYPED_TEST_SUITE(CInstCopyTestN, DimTypes);

TYPED_TEST(CInstCopyTestN, CopySuccess)
{
  constexpr int N = TestFixture::N;

  realm_runtime_t runtime = *this->runtime_impl;
  realm_region_instance_copy_params_t params;

  Realm::RegionInstance inst = ID::make_instance(0, 0, 0, 0).convert<RegionInstance>();
  realm_field_id_t field_id = 0;
  size_t field_size = 10;
  size_t num_fields = 1;

  realm_copy_src_dst_field_t srcs = {inst, field_id, field_size};
  realm_copy_src_dst_field_t dsts = {inst, field_id, field_size};
  params.srcs = &srcs;
  params.dsts = &dsts;
  params.num_fields = num_fields;

  int lower_bound[N], upper_bound[N];
  for(int i = 0; i < N; ++i) {
    lower_bound[i] = 10;
    upper_bound[i] = 10;
  }

  params.lower_bound = lower_bound;
  params.upper_bound = upper_bound;
  params.num_dims = N;
  params.coord_type = REALM_COORD_TYPE_INT;

  realm_event_t event;
  int priority = 0;
  realm_event_t wait_on = REALM_NO_EVENT;

  bool mock_was_called_successfully = false;

  dynamic_mock_impl_N<N> =
      [&](const IndexSpace<N, int> &idx_space,
          const std::vector<CopySrcDstField> &srcs_received,
          const std::vector<CopySrcDstField> &dsts_received,
          const std::vector<const typename CopyIndirection<N, int>::Base *>
              &indirects_received,
          const Realm::ProfilingRequestSet &reqs_received, Event wait_on_received,
          int priority_received) {
        if(srcs_received.size() != num_fields || dsts_received.size() != num_fields)
          return Event::NO_EVENT;

        for(size_t i = 0; i < num_fields; ++i) {
          if(srcs_received[i].field_id != field_id ||
             srcs_received[i].size != field_size ||
             dsts_received[i].field_id != field_id || dsts_received[i].size != field_size)
            return Event::NO_EVENT;
        }

        if(priority_received != priority || wait_on_received.id != wait_on)
          return Event::NO_EVENT;

        for(int i = 0; i < N; ++i) {
          if(idx_space.bounds.lo[i] != lower_bound[i] ||
             idx_space.bounds.hi[i] != upper_bound[i])
            return Event::NO_EVENT;
        }

        mock_was_called_successfully = true;
        return Event::NO_EVENT;
      };

  CopyImplRouter<N, int>::impl = &mock_copy_redirect_N<N>;

  realm_status_t status = realm_region_instance_copy(runtime, &params, nullptr,
                                                     REALM_NO_EVENT, priority, &event);

  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_TRUE(mock_was_called_successfully);
}