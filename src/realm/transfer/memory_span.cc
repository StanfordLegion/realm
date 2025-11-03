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

#include "realm/transfer/memory_span.h"

#include <cassert>

namespace Realm {

  // =================================================================================================
  //                                             Span
  // =================================================================================================
  
  size_t Span::total_bytes() const {
    if (num_dims == 0) {
      return 0;
    }
    
    // Calculate bytes for geometry
    size_t bytes = extents[0];
    for (int d = 1; d < num_dims; d++) {
      bytes *= extents[d];
    }
    
    // Multiply by number of fields
    return bytes * field_ids.size();
  }
  
  bool Span::is_contiguous() const {
    if (num_dims == 0) {
      return true;
    }
    
    // Check if strides match expected contiguous layout
    size_t expected_stride = 1;
    for (int d = 0; d < num_dims; d++) {
      if (strides[d] != expected_stride) {
        return false;
      }
      expected_stride *= extents[d];
    }
    return true;
  }

  // =================================================================================================
  //                                           SpanList
  // =================================================================================================
  
  void SpanList::append(const Span& span) {
    spans_.push_back(span);
  }
  
  void SpanList::append(size_t base, const std::vector<FieldID>& fields, uint8_t dims,
                        const uint32_t* extents, const size_t* strides) {
    Span span;
    span.base_offset = base;
    span.field_ids = fields;
    span.num_dims = dims;
    
    for (int d = 0; d < dims && d < 3; d++) {
      span.extents[d] = extents[d];
      span.strides[d] = strides[d];
    }
    // Zero out unused dimensions
    for (int d = dims; d < 3; d++) {
      span.extents[d] = 0;
      span.strides[d] = 0;
    }
    
    spans_.push_back(span);
  }
  
  size_t SpanList::total_bytes() const {
    size_t total = 0;
    for (const auto& span : spans_) {
      total += span.total_bytes();
    }
    return total;
  }
  
  // =================================================================================================
  //                                         SpanIterator
  // =================================================================================================
  
  SpanIterator::SpanIterator(const SpanList* list)
    : list_(list)
    , bytes_consumed_(0)
  {
    memset(pos_, 0, sizeof(pos_));
  }
  
  size_t SpanIterator::offset() const {
    if (!list_ || done()) {
      return 0;
    }
    
    const Span& s = (*list_)[span_idx_];
    
    size_t offset = s.base_offset;
    
    // Add position within geometry
    for (int d = 0; d < s.num_dims; d++) {
      offset += pos_[d] * s.strides[d];
    }
    
    // If multiple fields, add field offset
    if (s.field_ids.size() > 1 && field_idx_ > 0) {
      size_t bytes_per_field = s.total_bytes() / s.field_ids.size();
      offset += field_idx_ * bytes_per_field;
    }
    
    return offset;
  }
  
  size_t SpanIterator::stride(int dim) const {
    if (done()) {
      return 0;
    }
    
    const Span& s = (*list_)[span_idx_];
    assert(dim >= 0 && dim < s.num_dims);
    return s.strides[dim];
  }
  
  size_t SpanIterator::remaining(int dim) const {
    if (done()) {
      return 0;
    }
    
    const Span& s = (*list_)[span_idx_];
    assert(dim >= 0 && dim < s.num_dims);
    return s.extents[dim] - pos_[dim];
  }
  
  int SpanIterator::dim() const {
    if (done()) {
      return 0;
    }
    
    const Span& s = (*list_)[span_idx_];
    return s.num_dims;
  }
  
  FieldID SpanIterator::current_field() const {
    if (done()) {
      return FieldID(-1);
    }
    
    const Span& s = (*list_)[span_idx_];
    assert(field_idx_ < s.field_ids.size());
    return s.field_ids[field_idx_];
  }
  
  size_t SpanIterator::remaining_fields() const {
    if (done()) {
      return 0;
    }
    
    const Span& s = (*list_)[span_idx_];
    return s.field_ids.size() - field_idx_;
  }
  
  void SpanIterator::advance(int dim, size_t count) {
    if (done()) {
      return;
    }
    
    const Span& s = (*list_)[span_idx_];
    
    size_t bytes_to_consume = count;
    for (int d = 0; d < dim; d++) {
      bytes_to_consume *= s.extents[d];
    }
    
    // Update bytes_consumed
    bytes_consumed_ += bytes_to_consume;
    
    pos_[dim] += count;
    
    // Carry to higher dimensions - handle multi-level overflow
    bool geometry_exhausted = false;
    for (int d = dim; d < s.num_dims; d++) {
      if (pos_[d] >= s.extents[d]) {
        if (d + 1 < s.num_dims) {
          // Calculate how many times we overflow this dimension
          size_t carries = pos_[d] / s.extents[d];
          pos_[d] = pos_[d] % s.extents[d];  // Remainder stays in this dimension
          pos_[d + 1] += carries;             // Propagate carries to next dimension
          // Continue loop to handle cascading carries
        } else {
          // Reached the end of the highest dimension
          geometry_exhausted = true;
          break;
        }
      } else {
        break;  // No more carries needed
      }
    }
    
    // Check if geometry exhausted
    if (geometry_exhausted) {
      // Move to next field if multi-field span
      if (field_idx_ + 1 < s.field_ids.size()) {
        field_idx_++;
        memset(pos_, 0, sizeof(pos_));
      } else {
        // Move to next span
        span_idx_++;
        field_idx_ = 0;
        memset(pos_, 0, sizeof(pos_));
      }
    }
  }
  
  void SpanIterator::advance_fields(size_t num_fields) {
    if (done()) {
      return;
    }
    
    const Span& s = (*list_)[span_idx_];
    
    // Calculate bytes per field for current span
    size_t elements_per_field = s.extents[0];
    for (int d = 1; d < s.num_dims; d++) {
      elements_per_field *= s.extents[d];
    }
    
    // Increment bytes consumed for fields we're skipping in current span
    size_t fields_in_current_span = std::min(num_fields, s.field_ids.size() - field_idx_);
    bytes_consumed_ += fields_in_current_span * elements_per_field;
    
    field_idx_ += num_fields;
    
    // If we've exhausted fields in current span, move to next span
    while (field_idx_ >= s.field_ids.size() && !done()) {
      field_idx_ -= s.field_ids.size();
      span_idx_++;
      memset(pos_, 0, sizeof(pos_));
      
      if (!done() && field_idx_ >= (*list_)[span_idx_].field_ids.size()) {
        // Continue to next span if still have fields to skip
        // Increment bytes consumed for skipped fields in next span
        const Span& next_s = (*list_)[span_idx_];
        size_t next_elements_per_field = next_s.extents[0];
        for (int d = 1; d < next_s.num_dims; d++) {
          next_elements_per_field *= next_s.extents[d];
        }
        size_t next_fields_skipped = std::min(field_idx_, next_s.field_ids.size());
        bytes_consumed_ += next_fields_skipped * next_elements_per_field;
        continue;
      }
      break;
    }
  }
  
  void SpanIterator::skip_bytes(size_t bytes) {
    while (bytes > 0 && !done()) {
      int d = dim();
      
      if (d == 0) {
        // No geometry to skip through
        return;
      }
      
      size_t chunk = remaining(0);
      if (chunk <= bytes) {
        // Skip entire remaining chunk in dimension 0
        advance(0, chunk);
        bytes -= chunk;
      } else {
        // Skip partial chunk
        advance(0, bytes);
        bytes = 0;
      }
    }
  }
  
  bool SpanIterator::done() const {
    return !list_ || span_idx_ >= list_->size();
  }

} // namespace Realm

