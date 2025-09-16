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

class CInstLayoutBaseTest {
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

class CInstCreateFromInstanceLayoutTest : public CInstLayoutBaseTest,
                                          public ::testing::Test {
protected:
  void SetUp() override { CInstLayoutBaseTest::initialize(1); }

  void TearDown() override { CInstLayoutBaseTest::finalize(); }
};

TEST_F(CInstCreateFromInstanceLayoutTest, CreateNullRuntime)
{
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      nullptr, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateInvalidMemory)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateInvalidLowerBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = nullptr;
  space.upper_bound = upper_bound;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateInvalidUpperBound)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = nullptr;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateZeroDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = 0;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateOverMaxDim)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = REALM_MAX_DIM + 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_REGION_INSTANCE_ERROR_INVALID_DIMS);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateInvalidFieldLayouts)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  instance_layout.field_layouts = nullptr;
  instance_layout.num_fields = 0;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;
  realm_event_t event;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      &event);

  EXPECT_EQ(status, REALM_INSTANCE_LAYOUT_ERROR_INVALID_FIELDS);
}

TEST_F(CInstCreateFromInstanceLayoutTest, CreateNullEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_region_instance_t inst;
  realm_instance_layout_t instance_layout;
  int lower_bound[1] = {0};
  int upper_bound[1] = {9};
  realm_index_space_t space;
  space.lower_bound = lower_bound;
  space.upper_bound = upper_bound;
  space.num_dims = 1;
  space.coord_type = REALM_COORD_TYPE_INT;
  realm_field_layout_t field_layout;
  field_layout.field_id = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = sizeof(int);
  instance_layout.field_layouts = &field_layout;
  instance_layout.num_fields = 1;
  instance_layout.alignment_reqd = 32;
  instance_layout.space = space;
  int dim_order[1] = {0};
  instance_layout.dim_order = dim_order;

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, REALM_NO_MEM, nullptr, nullptr, REALM_NO_EVENT, &inst,
      nullptr);
}

struct InstanceLayoutConfig {
  int num_dims;
  std::vector<int> lower_bound;
  std::vector<int> upper_bound;
  int alignment_reqd;
  std::vector<realm_field_layout_t> fields;
  std::vector<int> dim_order;
};

template <typename T>
class CInstCreateFromInstanceLayoutTestT : public CInstCreateFromInstanceLayoutTest {
protected:
  template <int Dim>
  realm_instance_layout_t make_instance_layout(const InstanceLayoutConfig &cfg)
  {
    assert(cfg.num_dims == Dim); // sanity check

    realm_instance_layout_t layout;
    layout.num_fields = cfg.fields.size();
    layout.field_layouts = const_cast<realm_field_layout_t *>(cfg.fields.data());
    layout.alignment_reqd = cfg.alignment_reqd;
    layout.dim_order = const_cast<int *>(cfg.dim_order.data());

    layout.space.num_dims = cfg.num_dims;
    layout.space.coord_type = REALM_COORD_TYPE_INT;
    layout.space.lower_bound = const_cast<int *>(cfg.lower_bound.data());
    layout.space.upper_bound = const_cast<int *>(cfg.upper_bound.data());

    return layout;
  }
};

struct LayoutCase1D {
  static constexpr int DIM = 1;
  static const InstanceLayoutConfig value;
};
const InstanceLayoutConfig LayoutCase1D::value = {
    1, {0}, {9}, 32, {{0, sizeof(int), 0}, {1, sizeof(int), sizeof(int) * 10}}, {0}};

struct LayoutCase2D {
  static constexpr int DIM = 2;
  static const InstanceLayoutConfig value;
};
const InstanceLayoutConfig LayoutCase2D::value = {
    2,
    {0, 0},
    {4, 4},
    32,
    {{0, sizeof(float), 0}, {1, sizeof(float), sizeof(float) * 25}},
    {0, 1}};

using LayoutConfigs = ::testing::Types<LayoutCase1D, LayoutCase2D>;
TYPED_TEST_SUITE(CInstCreateFromInstanceLayoutTestT, LayoutConfigs);

TYPED_TEST(CInstCreateFromInstanceLayoutTestT, CreateSuccess)
{
  const InstanceLayoutConfig &cfg = TypeParam::value;
  constexpr int DIM = TypeParam::DIM;

  this->runtime_impl->setup_mock_proc_mems(
      MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded{
          {{0, Processor::Kind::LOC_PROC, 0}},
          {{0, Memory::Kind::SYSTEM_MEM, 1024 * 1024, 0}},
          {{0, 0, 1000, 1}}});

  realm_runtime_t runtime = *this->runtime_impl;
  realm_region_instance_t inst;
  realm_event_t event;

  auto instance_layout = this->template make_instance_layout<DIM>(cfg);

  realm_status_t status = realm_region_instance_create_from_instance_layout(
      runtime, &instance_layout, ID::make_memory(0, 0).convert<Memory>(), nullptr,
      nullptr, REALM_NO_EVENT, &inst, &event);

  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(RegionInstance(inst).exists(), true);

  RegionInstanceImpl *r_impl = this->runtime_impl->get_instance_impl(inst);
  const InstanceLayout<DIM> *instance_layout_cxx =
      dynamic_cast<const InstanceLayout<DIM> *>(r_impl->metadata.layout);
  ASSERT_TRUE(instance_layout_cxx != nullptr);
  EXPECT_EQ(instance_layout_cxx->alignment_reqd, instance_layout.alignment_reqd);

  Rect<DIM, int> bounds = instance_layout_cxx->space.bounds;
  for(int d = 0; d < DIM; d++) {
    EXPECT_EQ(bounds.lo[d], cfg.lower_bound[d]);
    EXPECT_EQ(bounds.hi[d], cfg.upper_bound[d]);
  }

  EXPECT_EQ(instance_layout_cxx->fields.size(), cfg.fields.size());
  for(auto &field : instance_layout_cxx->fields) {
    EXPECT_EQ(field.first, cfg.fields[field.first].field_id);
    EXPECT_EQ(field.second.size_in_bytes, cfg.fields[field.first].size_in_bytes);
  }

  ASSERT_REALM(realm_region_instance_destroy(runtime, inst, event));
}
