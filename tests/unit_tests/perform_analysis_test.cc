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

#include "realm/transfer/transfer.h"
#include "realm/timers.h"
#include <gtest/gtest.h>

using namespace Realm;

// Minimal TransferDomain implementation that avoids runtime dependencies.
class MockTransferDomain : public TransferDomain {
public:
  TransferDomain *clone() const override { return new MockTransferDomain; }
  Event request_metadata() override { return Event::NO_EVENT; }
  bool empty() const override { return false; }
  size_t volume() const override { return 10; }
  void choose_dim_order(std::vector<int> &dim_order,
                        const std::vector<CopySrcDstField> &srcs,
                        const std::vector<CopySrcDstField> &dsts,
                        const std::vector<IndirectionInfo *> &indirects,
                        bool force_fortran_order, size_t max_stride) const override
  {
    dim_order.clear();
    dim_order.push_back(0);
  }
  void count_fragments(RegionInstance inst, const std::vector<int> &dim_order,
                       const std::vector<FieldID> &fields,
                       const std::vector<size_t> &fld_sizes,
                       std::vector<size_t> &fragments) const override
  {
    fragments.clear();
  }
  TransferIterator *create_iterator(RegionInstance inst,
                                    const std::vector<int> &dim_order,
                                    const std::vector<FieldID> &fields,
                                    const std::vector<size_t> &fld_offsets,
                                    const std::vector<size_t> &fld_sizes,
                                    bool idindexed_fields = false) const override
  {
    return nullptr;
  }
  TransferIterator *create_iterator(RegionInstance inst, RegionInstance peer,
                                    const std::vector<FieldID> &fields,
                                    const std::vector<size_t> &fld_offsets,
                                    const std::vector<size_t> &fld_sizes) const override
  {
    return nullptr;
  }
  void print(std::ostream &os) const override { os << "MockTransferDomain"; }
};

// Test subclass that exposes TransferDesc's protected members for testing.
// Uses the TestTag constructor to bypass check_analysis_preconditions (which
// requires the runtime).
class TestableTransferDesc : public TransferDesc {
public:
  TestableTransferDesc(TransferDomain *domain)
    : TransferDesc(TestTag{}, domain)
  {}

  using TransferDesc::perform_analysis;

  // Accessors for protected state
  bool get_analysis_complete() const { return analysis_complete.load(); }
  bool get_analysis_init_done() const { return analysis_init_done; }
  size_t get_analysis_field_idx() const { return analysis_field_idx; }
  size_t get_dim_order_size() const { return dim_order.size(); }
  const int *get_dim_order_data() const { return dim_order.data(); }
  size_t get_src_fields_size() const { return src_fields.size(); }
  size_t get_dst_fields_size() const { return dst_fields.size(); }

  // Set up dummy src/dst field pairs. Uses NO_INST so that choose_dim_order
  // skips the preferred_dim_order call (which requires the runtime). The
  // loop body won't execute in timeout tests since is_expired() fires
  // before any per-field work.
  void add_dummy_fields(size_t num_fields)
  {
    for(size_t i = 0; i < num_fields; i++) {
      CopySrcDstField src;
      src.set_field(RegionInstance::NO_INST, FieldID(i), /*size=*/8);
      CopySrcDstField dst;
      dst.set_field(RegionInstance::NO_INST, FieldID(i), /*size=*/8);
      srcs.push_back(src);
      dsts.push_back(dst);
    }
  }
};

// Test that perform_analysis returns true immediately for an empty domain.
TEST(PerformAnalysisTest, EmptyDomainCompletesImmediately)
{
  class MockEmptyDomain : public MockTransferDomain {
  public:
    bool empty() const override { return true; }
    size_t volume() const override { return 0; }
  };

  TestableTransferDesc *desc = new TestableTransferDesc(new MockEmptyDomain);

  bool completed = desc->perform_analysis(TimeLimit());
  EXPECT_TRUE(completed);
  EXPECT_TRUE(desc->get_analysis_complete());

  desc->remove_reference();
}

// Test that perform_analysis with an immediately-expired TimeLimit returns
// false (timed out) without completing the analysis.
TEST(PerformAnalysisTest, ExpiredTimeLimitCausesTimeout)
{
  const size_t num_fields = 5;
  TestableTransferDesc *desc = new TestableTransferDesc(new MockTransferDomain);
  desc->add_dummy_fields(num_fields);

  // Call perform_analysis with an already-expired time limit.
  // The init phase runs (it doesn't check the time limit), but the loop
  // should immediately detect the expired limit and return false.
  bool completed = desc->perform_analysis(TimeLimit::relative(0));

  EXPECT_FALSE(completed);
  // Init should have completed
  EXPECT_TRUE(desc->get_analysis_init_done());
  // But the field loop should not have progressed
  EXPECT_EQ(desc->get_analysis_field_idx(), 0u);
  // analysis_complete should still be false
  EXPECT_FALSE(desc->get_analysis_complete());

  desc->remove_reference();
}

// Test that after a timeout, calling perform_analysis again with an expired
// TimeLimit preserves the init state (doesn't redo it).
TEST(PerformAnalysisTest, InitStatePreservedAcrossTimeoutCalls)
{
  const size_t num_fields = 3;
  TestableTransferDesc *desc = new TestableTransferDesc(new MockTransferDomain);
  desc->add_dummy_fields(num_fields);

  // First call: times out immediately
  bool completed = desc->perform_analysis(TimeLimit::relative(0));
  ASSERT_FALSE(completed);
  ASSERT_TRUE(desc->get_analysis_init_done());

  // Verify that dim_order was set during init (1D domain -> single entry)
  EXPECT_EQ(desc->get_dim_order_size(), 1u);
  // Verify src/dst fields were resized during init
  EXPECT_EQ(desc->get_src_fields_size(), num_fields);
  EXPECT_EQ(desc->get_dst_fields_size(), num_fields);

  // Second call: also times out, but init should not be redone.
  // Capture dim_order pointer to verify it's the same vector (not reallocated).
  const int *dim_order_data = desc->get_dim_order_data();
  completed = desc->perform_analysis(TimeLimit::relative(0));
  EXPECT_FALSE(completed);
  EXPECT_TRUE(desc->get_analysis_init_done());
  // dim_order should not have been modified (init was skipped)
  EXPECT_EQ(desc->get_dim_order_data(), dim_order_data);

  desc->remove_reference();
}
