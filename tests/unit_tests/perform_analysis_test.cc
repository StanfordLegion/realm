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
                                    const std::vector<size_t> &fld_sizes) const override
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
  // Helper to construct a TransferDesc with an empty domain and no fields.
  // The constructor calls check_analysis_preconditions -> perform_analysis,
  // which returns immediately for an empty domain (no runtime needed).
  static TransferDesc *create_empty_desc()
  {
    IndexSpace<1, int> empty_is = IndexSpace<1, int>::make_empty();
    std::vector<CopySrcDstField> srcs;
    std::vector<CopySrcDstField> dsts;
    std::vector<const CopyIndirection<1, int>::Base *> indirects;
    ProfilingRequestSet prs;
    TransferDesc *desc = new TransferDesc(empty_is, srcs, dsts, indirects, prs);
    return desc;
  }

  // Reset a TransferDesc's analysis state so perform_analysis can be called
  // again with a non-empty domain and dummy fields.
  static void reset_for_nonempty_analysis(TransferDesc *desc, size_t num_fields)
  {
    // Replace the empty domain with a non-empty mock domain
    delete desc->domain;
    desc->domain = new MockTransferDomain;

    // Reset analysis state
    desc->analysis_complete.store(false);
    desc->analysis_successful = false;
    desc->analysis_init_done = false;
    desc->analysis_field_idx = 0;
    desc->analysis_fld_start = 0;
    desc->analysis_fill_ofs = 0;
    desc->analysis_field_done.clear();

    // Reset graph
    desc->graph.xd_nodes.clear();
    desc->graph.ib_edges.clear();
    desc->graph.ib_alloc_order.clear();

    // Reset fields
    desc->dim_order.clear();
    desc->src_fields.clear();
    desc->dst_fields.clear();
    if(desc->fill_data) {
      free(desc->fill_data);
      desc->fill_data = nullptr;
    }
    desc->fill_size = 0;

    // Set up dummy src/dst field pairs. Use NO_INST so that
    // choose_dim_order skips the preferred_dim_order call (which requires
    // the runtime). The loop body won't execute in timeout tests since
    // is_expired() fires before any per-field work.
    desc->srcs.clear();
    desc->dsts.clear();
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
  TransferDesc *desc = create_empty_desc();

  // The constructor already ran perform_analysis, which completed for the
  // empty domain case.
  EXPECT_TRUE(get_analysis_complete(desc));

  desc->remove_reference();
}

// Test that perform_analysis with an immediately-expired TimeLimit returns
// false (timed out) without completing the analysis.
TEST_F(PerformAnalysisTest, ExpiredTimeLimitCausesTimeout)
{
  TransferDesc *desc = create_empty_desc();
  ASSERT_TRUE(get_analysis_complete(desc));

  // Reset state for a non-empty analysis with multiple fields
  const size_t num_fields = 5;
  reset_for_nonempty_analysis(desc, num_fields);
  ASSERT_FALSE(get_analysis_complete(desc));
  ASSERT_FALSE(get_analysis_init_done(desc));

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
  TransferDesc *desc = create_empty_desc();

  const size_t num_fields = 3;
  reset_for_nonempty_analysis(desc, num_fields);

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
