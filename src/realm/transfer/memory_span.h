/*
 * Copyright 2025 Los Alamos National Laboratory, Stanford University, NVIDIA Corporation
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

#ifndef MEMORY_SPAN_H
#define MEMORY_SPAN_H

#include "realm/realm_config.h"
#include "realm/indexspace.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace Realm {

  // =================================================================================================
  //                                             Span
  // =================================================================================================
  
  struct Span {
    size_t base_offset;              // Base memory offset
    std::vector<FieldID> field_ids;  // Field ID(s) - one or more
    uint8_t num_dims;                // 1-4 (extended from 3 to 4 for complex layouts)
    uint32_t extents[4];             // Elements per dimension
    size_t strides[4];               // Byte stride per dimension
    
    // Computed properties
    size_t total_bytes() const;
    bool is_contiguous() const;
  };

  // =================================================================================================
  //                                           SpanList
  // =================================================================================================
  
  class SpanIterator;  // Forward declaration
  
  class SpanList {
    std::vector<Span> spans_;
    
  public:
    SpanList() = default;
    
    // Append span (works for single or multiple fields)
    void append(const Span& span);
    
    // Convenience builder
    void append(size_t base, const std::vector<FieldID>& fields, uint8_t dims,
                const uint32_t* extents, const size_t* strides);
    
    // Access
    size_t size() const { return spans_.size(); }
    const Span& operator[](size_t idx) const { return spans_[idx]; }
    bool empty() const { return spans_.empty(); }
    
    // Compute total bytes pending across all spans
    size_t total_bytes() const;
    size_t bytes_pending() const { return total_bytes(); }  // Total bytes (no iterator)
  };

  // =================================================================================================
  //                                         SpanIterator
  // =================================================================================================
  
  class SpanIterator {
    const SpanList* list_ = nullptr;
    size_t span_idx_ = 0;
    size_t field_idx_ = 0;          // Current field within span
    uint32_t pos_[4] = {};
    size_t bytes_consumed_ = 0;     // Total bytes consumed so far
    
  public:
    SpanIterator() = default;
    explicit SpanIterator(const SpanList* list);
    
    // Geometry queries
    size_t offset() const;          // Offset for current field at current position
    size_t stride(int dim) const;
    size_t remaining(int dim) const;
    int dim() const;
    
    // Field queries
    FieldID current_field() const;
    size_t remaining_fields() const;  // Fields left in current span
    
    // Progress tracking - dynamically accounts for list growth!
    size_t bytes_pending() const {
      return list_ ? (list_->total_bytes() - bytes_consumed_) : 0;
    }
    
    // Navigation
    void advance(int dim, size_t count);
    void advance_fields(size_t num_fields);  // Move to next field(s)
    void skip_bytes(size_t bytes);           // Helper for compatibility
    bool done() const;
    
    // Access to current state (for debugging/testing)
    size_t current_span_index() const { return span_idx_; }
    size_t current_field_index() const { return field_idx_; }
    const uint32_t* position() const { return pos_; }
  };

} // namespace Realm

#endif // MEMORY_SPAN_H

