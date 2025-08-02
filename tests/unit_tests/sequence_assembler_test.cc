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

#include "realm/transfer/channel.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

TEST(SequenceAssemblerTest, EmptySpan)
{
  SequenceAssembler assembler;
  EXPECT_TRUE(assembler.empty());
  EXPECT_EQ(assembler.add_span(0, 0), 0);
  EXPECT_EQ(assembler.span_exists(0, 0), 0);
}

TEST(SequenceAssemblerTest, SingleSpan)
{
  SequenceAssembler assembler;
  EXPECT_TRUE(assembler.empty());

  const size_t count = 16;
  const size_t start = 4;

  EXPECT_EQ(assembler.add_span(0, count), count);

  // exists
  EXPECT_EQ(assembler.span_exists(0, count - 1), count - 1);
  EXPECT_EQ(assembler.span_exists(start, count - 1), count - start);
  EXPECT_EQ(assembler.span_exists(0, count), count);
  EXPECT_EQ(assembler.span_exists(0, count + 1), count);
  EXPECT_EQ(assembler.span_exists(0, count * 2), count);

  // doesn't exist
  EXPECT_EQ(assembler.span_exists(0, 0), 0);
  EXPECT_EQ(assembler.span_exists(count, count * 2), 0);
}

// TODO: we need more test cases as not all branches are coverd in
// `add_span` and `span_exists` calls.
TEST(SequenceAssemblerTest, NocontigOutOfOrderSpans)
{
  SequenceAssembler assembler;

  const size_t beg_span_a = 0;
  const size_t count_a = 16;
  const size_t beg_span_b = 18;
  const size_t count_b = 4;

  EXPECT_EQ(assembler.add_span(beg_span_b, count_b), 0);
  EXPECT_EQ(assembler.add_span(beg_span_a, count_a), count_a - beg_span_a);

  EXPECT_EQ(assembler.span_exists(beg_span_a, count_a), count_a);
  EXPECT_EQ(assembler.span_exists(beg_span_b, count_b), count_b);
  EXPECT_EQ(assembler.span_exists(4, count_b), 4);

  EXPECT_EQ(assembler.add_span(beg_span_b + count_b + 1, 1), 0);

  EXPECT_FALSE(assembler.empty());
}

TEST(SequenceAssemblerTest, ContigSpans)
{
  SequenceAssembler assembler;

  const size_t beg_span_a = 0;
  const size_t count_a = 16;
  const size_t beg_span_b = 16;
  const size_t count_b = 24;

  EXPECT_EQ(assembler.add_span(beg_span_a, count_a), count_a);
  EXPECT_EQ(assembler.add_span(beg_span_b, count_b), count_b);

  EXPECT_EQ(assembler.span_exists(beg_span_a, count_a), count_a);
  EXPECT_EQ(assembler.span_exists(beg_span_b, count_b), count_b);
  EXPECT_FALSE(assembler.empty());
}
