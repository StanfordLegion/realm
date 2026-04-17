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
// Used to replace the empty domain after construction so that perform_analysis
// takes the non-empty path and enters the main field loop.
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

// PerformAnalysisTest is a friend of TransferDesc, allowing direct access to
// protected members for testing the incremental analysis behavior.
class PerformAnalysisTest : public ::testing::Test {
protected:
  // Helper to construct a TransferDesc using the test-only constructor.
  // This bypasses TransferDomain::construct and check_analysis_preconditions,
  // avoiding runtime dependencies.
  static TransferDesc *create_desc(TransferDomain *domain)
  {
    return new TransferDesc(TransferDesc::TestTag{}, domain);
  }

  // Set up dummy src/dst field pairs on a TransferDesc. Uses NO_INST so that
  // choose_dim_order skips the preferred_dim_order call (which requires
  // the runtime). The loop body won't execute in timeout tests since
  // is_expired() fires before any per-field work.
  static void setup_dummy_fields(TransferDesc *desc, size_t num_fields)
  {
    for(size_t i = 0; i < num_fields; i++) {
      CopySrcDstField src;
      src.set_field(RegionInstance::NO_INST, FieldID(i), /*size=*/8);
      CopySrcDstField dst;
      dst.set_field(RegionInstance::NO_INST, FieldID(i), /*size=*/8);
      desc->srcs.push_back(src);
      desc->dsts.push_back(dst);
    }
  }

  // Accessors for protected members (friendship is not inherited by TEST_F
  // subclasses, so all access must go through PerformAnalysisTest methods).
  static bool call_perform_analysis(TransferDesc *desc, TimeLimit work_until)
  {
    return desc->perform_analysis(work_until);
  }
  static bool get_analysis_complete(TransferDesc *desc)
  {
    return desc->analysis_complete.load();
  }
  static bool get_analysis_init_done(TransferDesc *desc)
  {
    return desc->analysis_init_done;
  }
  static size_t get_analysis_field_idx(TransferDesc *desc)
  {
    return desc->analysis_field_idx;
  }
  static size_t get_dim_order_size(TransferDesc *desc) { return desc->dim_order.size(); }
  static const int *get_dim_order_data(TransferDesc *desc)
  {
    return desc->dim_order.data();
  }
  static size_t get_src_fields_size(TransferDesc *desc)
  {
    return desc->src_fields.size();
  }
  static size_t get_dst_fields_size(TransferDesc *desc)
  {
    return desc->dst_fields.size();
  }
};

// Test that perform_analysis returns true immediately for an empty domain.
TEST_F(PerformAnalysisTest, EmptyDomainCompletesImmediately)
{
  // MockEmptyDomain returns empty() == true
  class MockEmptyDomain : public MockTransferDomain {
  public:
    bool empty() const override { return true; }
    size_t volume() const override { return 0; }
  };

  TransferDesc *desc = create_desc(new MockEmptyDomain);

  bool completed = call_perform_analysis(desc, TimeLimit());
  EXPECT_TRUE(completed);
  EXPECT_TRUE(get_analysis_complete(desc));

  desc->remove_reference();
}

// Test that perform_analysis with an immediately-expired TimeLimit returns
// false (timed out) without completing the analysis.
TEST_F(PerformAnalysisTest, ExpiredTimeLimitCausesTimeout)
{
  const size_t num_fields = 5;
  TransferDesc *desc = create_desc(new MockTransferDomain);
  setup_dummy_fields(desc, num_fields);

  // Call perform_analysis with an already-expired time limit.
  // The init phase runs (it doesn't check the time limit), but the loop
  // should immediately detect the expired limit and return false.
  bool completed = call_perform_analysis(desc, TimeLimit::relative(0));

  EXPECT_FALSE(completed);
  // Init should have completed
  EXPECT_TRUE(get_analysis_init_done(desc));
  // But the field loop should not have progressed
  EXPECT_EQ(get_analysis_field_idx(desc), 0u);
  // analysis_complete should still be false
  EXPECT_FALSE(get_analysis_complete(desc));

  desc->remove_reference();
}

// Test that after a timeout, calling perform_analysis again with an expired
// TimeLimit preserves the init state (doesn't redo it).
TEST_F(PerformAnalysisTest, InitStatePreservedAcrossTimeoutCalls)
{
  const size_t num_fields = 3;
  TransferDesc *desc = create_desc(new MockTransferDomain);
  setup_dummy_fields(desc, num_fields);

  // First call: times out immediately
  bool completed = call_perform_analysis(desc, TimeLimit::relative(0));
  ASSERT_FALSE(completed);
  ASSERT_TRUE(get_analysis_init_done(desc));

  // Verify that dim_order was set during init (1D domain -> single entry)
  EXPECT_EQ(get_dim_order_size(desc), 1u);
  // Verify src/dst fields were resized during init
  EXPECT_EQ(get_src_fields_size(desc), num_fields);
  EXPECT_EQ(get_dst_fields_size(desc), num_fields);

  // Second call: also times out, but init should not be redone.
  // Capture dim_order pointer to verify it's the same vector (not reallocated).
  const int *dim_order_data = get_dim_order_data(desc);
  completed = call_perform_analysis(desc, TimeLimit::relative(0));
  EXPECT_FALSE(completed);
  EXPECT_TRUE(get_analysis_init_done(desc));
  // dim_order should not have been modified (init was skipped)
  EXPECT_EQ(get_dim_order_data(desc), dim_order_data);

  desc->remove_reference();
}
