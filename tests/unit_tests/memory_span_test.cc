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

#include "realm/transfer/memory_span.h"
#include <gtest/gtest.h>

using namespace Realm;

namespace {

  constexpr size_t kStride = 8;
  constexpr size_t kBytes = 1024;

  TEST(MemorySpanTests, SingleFieldBasic) {
    // Single field, 1D span
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0x1000, fields, 1, extents, strides);

    EXPECT_EQ(spans.size(), 1);
    EXPECT_EQ(spans[0].field_ids.size(), 1);
    EXPECT_EQ(spans[0].field_ids[0], 100);
    EXPECT_EQ(spans[0].total_bytes(), kBytes);

    SpanIterator it(&spans);
    EXPECT_FALSE(it.done());
    EXPECT_EQ(it.current_field(), 100);
    EXPECT_EQ(it.offset(), 0x1000);
    EXPECT_EQ(it.remaining(0), kBytes);
    EXPECT_EQ(it.dim(), 1);
  }

  TEST(MemorySpanTests, SingleFieldAdvance) {
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0, fields, 1, extents, strides);

    SpanIterator it(&spans);
    
    // Advance partway
    it.advance(0, 128);
    EXPECT_EQ(it.offset(), 128);
    EXPECT_EQ(it.remaining(0), kBytes - 128);
    EXPECT_FALSE(it.done());

    // Advance more
    it.advance(0, 128);
    EXPECT_EQ(it.offset(), 256);
    EXPECT_EQ(it.remaining(0), kBytes - 256);

    // Advance to end
    it.advance(0, kBytes - 256);
    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, MultiFieldSingleSpan) {
    // Multiple fields in one span (contiguous case)
    std::vector<FieldID> fields = {10, 11, 12, 13};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0x1000, fields, 1, extents, strides);

    EXPECT_EQ(spans.size(), 1);
    EXPECT_EQ(spans[0].field_ids.size(), 4);
    EXPECT_EQ(spans[0].total_bytes(), kBytes * 4);

    SpanIterator it(&spans);
    
    // Start at first field
    EXPECT_EQ(it.current_field(), 10);
    EXPECT_EQ(it.remaining_fields(), 4);
    EXPECT_EQ(it.offset(), 0x1000);

    // Advance through geometry for first field
    it.advance(0, 128);
    EXPECT_EQ(it.current_field(), 10);
    EXPECT_EQ(it.remaining(0), kBytes - 128);

    // Advance more
    it.advance(0, 128);
    EXPECT_EQ(it.remaining(0), kBytes - 256);

    // Complete first field
    it.advance(0, kBytes - 256);
    // Should auto-advance to next field
    EXPECT_FALSE(it.done());
    EXPECT_EQ(it.current_field(), 11);
    EXPECT_EQ(it.remaining_fields(), 3);
    EXPECT_EQ(it.remaining(0), kBytes);
  }

  TEST(MemorySpanTests, MultiFieldAdvanceAll) {
    std::vector<FieldID> fields = {7, 8, 9};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0, fields, 1, extents, strides);

    SpanIterator it(&spans);

    // Advance through all fields at once
    for (size_t f = 0; f < 3; f++) {
      EXPECT_EQ(it.current_field(), FieldID(7 + f));
      it.advance(0, kBytes);
    }

    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, TwoDimensionalSingle) {
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {1024, 100};  // 1024 bytes × 100 lines
    size_t strides[] = {1, 1024};

    SpanList spans;
    spans.append(0, fields, 2, extents, strides);

    SpanIterator it(&spans);
    EXPECT_EQ(it.dim(), 2);
    EXPECT_EQ(it.remaining(0), 1024);
    EXPECT_EQ(it.remaining(1), 100);
    EXPECT_EQ(it.stride(1), 1024);

    // Advance one line
    it.advance(0, 1024);
    EXPECT_EQ(it.remaining(0), 1024);
    EXPECT_EQ(it.remaining(1), 99);

    // Advance rest of lines
    it.advance(1, 99);
    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, TwoDimensionalMultiField) {
    std::vector<FieldID> fields = {10, 11, 12};
    uint32_t extents[] = {1024, 100};
    size_t strides[] = {1, 1024};

    SpanList spans;
    spans.append(0, fields, 2, extents, strides);

    SpanIterator it(&spans);
    
    size_t total_bytes = 1024 * 100 * 3;
    EXPECT_EQ(spans.total_bytes(), total_bytes);

    // Process first field
    EXPECT_EQ(it.current_field(), 10);
    it.advance(1, 100);  // Advance all lines
    
    // Should move to second field
    EXPECT_FALSE(it.done());
    EXPECT_EQ(it.current_field(), 11);
    EXPECT_EQ(it.remaining(1), 100);
  }

  TEST(MemorySpanTests, MultipleSpansSeparateFields) {
    // Three spans, one field each (non-contiguous case)
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    std::vector<FieldID> fields1 = {10};
    std::vector<FieldID> fields2 = {20};
    std::vector<FieldID> fields3 = {30};
    spans.append(0x1000, fields1, 1, extents, strides);
    spans.append(0x5000, fields2, 1, extents, strides);
    spans.append(0x9000, fields3, 1, extents, strides);

    EXPECT_EQ(spans.size(), 3);
    EXPECT_EQ(spans.total_bytes(), kBytes * 3);

    SpanIterator it(&spans);

    // First span
    EXPECT_EQ(it.current_field(), 10);
    EXPECT_EQ(it.offset(), 0x1000);
    it.advance(0, kBytes);

    // Second span
    EXPECT_FALSE(it.done());
    EXPECT_EQ(it.current_field(), 20);
    EXPECT_EQ(it.offset(), 0x5000);
    it.advance(0, kBytes);

    // Third span
    EXPECT_FALSE(it.done());
    EXPECT_EQ(it.current_field(), 30);
    EXPECT_EQ(it.offset(), 0x9000);
    it.advance(0, kBytes);

    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, ThreeDimensional) {
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {64, 8, 2};  // 64 bytes × 8 × 2
    size_t strides[] = {1, 64, 512};

    SpanList spans;
    spans.append(0, fields, 3, extents, strides);

    SpanIterator it(&spans);
    EXPECT_EQ(it.dim(), 3);
    EXPECT_EQ(it.remaining(0), 64);
    EXPECT_EQ(it.remaining(1), 8);
    EXPECT_EQ(it.remaining(2), 2);

    // Advance through highest dimension
    it.advance(2, 2);
    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, SkipBytes) {
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0, fields, 1, extents, strides);

    SpanIterator it(&spans);
    
    // Skip some bytes
    it.skip_bytes(128);
    EXPECT_EQ(it.offset(), 128);
    EXPECT_EQ(it.remaining(0), kBytes - 128);

    // Skip more
    it.skip_bytes(128);
    EXPECT_EQ(it.offset(), 256);
  }

  TEST(MemorySpanTests, AdvanceFieldsExplicit) {
    std::vector<FieldID> fields = {10, 11, 12, 13};
    uint32_t extents[] = {kBytes};
    size_t strides[] = {1};

    SpanList spans;
    spans.append(0, fields, 1, extents, strides);

    SpanIterator it(&spans);
    
    // At first field
    EXPECT_EQ(it.current_field(), 10);
    EXPECT_EQ(it.current_field_index(), 0);

    // Advance to skip geometry for 2 fields
    it.advance(0, kBytes);  // Complete field 10
    EXPECT_EQ(it.current_field(), 11);
    
    it.advance(0, kBytes);  // Complete field 11
    EXPECT_EQ(it.current_field(), 12);
    EXPECT_EQ(it.remaining_fields(), 2);
  }

  TEST(MemorySpanTests, IsContiguous) {
    // Contiguous span
    std::vector<FieldID> fields = {100};
    uint32_t extents1[] = {64, 100};
    size_t strides1[] = {1, 64};

    Span span1;
    span1.base_offset = 0;
    span1.field_ids = fields;
    span1.num_dims = 2;
    span1.extents[0] = extents1[0];
    span1.extents[1] = extents1[1];
    span1.strides[0] = strides1[0];
    span1.strides[1] = strides1[1];

    EXPECT_TRUE(span1.is_contiguous());

    // Non-contiguous span
    Span span2;
    span2.base_offset = 0;
    span2.field_ids = fields;
    span2.num_dims = 2;
    span2.extents[0] = 64;
    span2.extents[1] = 100;
    span2.strides[0] = 1;
    span2.strides[1] = 128;  // Stride doesn't match (has padding)

    EXPECT_FALSE(span2.is_contiguous());
  }

  TEST(MemorySpanTests, EmptySpanList) {
    SpanList spans;
    EXPECT_TRUE(spans.empty());
    EXPECT_EQ(spans.size(), 0);
    EXPECT_EQ(spans.total_bytes(), 0);

    SpanIterator it(&spans);
    EXPECT_TRUE(it.done());
  }

  TEST(MemorySpanTests, PartialAdvanceTwoDim) {
    std::vector<FieldID> fields = {100};
    uint32_t extents[] = {64, 10};  // 64 bytes × 10 lines
    size_t strides[] = {1, 64};

    SpanList spans;
    spans.append(0, fields, 2, extents, strides);

    SpanIterator it(&spans);

    // Advance half a line
    it.advance(0, 32);
    EXPECT_EQ(it.remaining(0), 32);
    EXPECT_EQ(it.remaining(1), 10);
    EXPECT_EQ(it.offset(), 32);

    // Complete the line
    it.advance(0, 32);
    EXPECT_EQ(it.remaining(0), 64);
    EXPECT_EQ(it.remaining(1), 9);
    EXPECT_EQ(it.offset(), 64);
  }

  TEST(MemorySpanTests, AdvanceMultiLevelOverflow)
  {
    // Test advancing by an amount that overflows multiple times
    // 2D: 10 bytes per line, 5 lines = 50 bytes total
    SpanList spans;
    uint32_t extents[] = {10, 5};
    size_t strides[] = {1, 10};
    spans.append(0x1000, {100}, 2, extents, strides);

    SpanIterator it(&spans);
    EXPECT_EQ(it.dim(), 2);
    EXPECT_EQ(it.remaining(0), 10);
    EXPECT_EQ(it.remaining(1), 5);

    // Advance by 25 bytes in dimension 0
    // Should overflow twice: 25 / 10 = 2 carries, 25 % 10 = 5 remainder
    // Expected: pos[0] = 5, pos[1] = 2
    it.advance(0, 25);
    
    EXPECT_EQ(it.remaining(0), 5);  // 10 - 5 = 5 remaining in dim 0
    EXPECT_EQ(it.remaining(1), 3);  // 5 - 2 = 3 remaining in dim 1
    EXPECT_EQ(it.offset(), 0x1000 + 2 * 10 + 5);  // base + 2 lines + 5 bytes
    EXPECT_FALSE(it.done());

    // Advance by another 25 bytes to complete the span
    // Current: pos[0]=5, pos[1]=2 (consumed 25 bytes, 25 remaining)
    it.advance(0, 25);
    EXPECT_TRUE(it.done());
  }

} // namespace

