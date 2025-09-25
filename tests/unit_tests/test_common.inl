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

#include "test_common.h"

namespace Realm {

  template <int N, typename T>
  static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
                                             const std::vector<FieldID> &field_ids,
                                             const std::vector<size_t> &field_sizes)
  {
    InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();
    assert(field_ids.size() == field_sizes.size() &&
           "Given field ids and sizes must match");

    inst_layout->piece_lists.resize(field_ids.size());
    inst_layout->bytes_used = 0;
    inst_layout->space = bounds;

    for(size_t i = 0; i < field_ids.size(); i++) {
      InstanceLayoutGeneric::FieldLayout field_layout;
      field_layout.list_idx = i;
      field_layout.rel_offset = 0;
      field_layout.size_in_bytes = field_sizes[i];

      AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
      affine_piece->bounds = bounds;
      affine_piece->offset = 0;
      affine_piece->strides[0] = field_sizes[i];
      size_t mult = affine_piece->strides[0];
      for(int d = 1; d < N; d++) {
        affine_piece->strides[d] = (bounds.hi[d - 1] - bounds.lo[d - 1] + 1) * mult;
        mult *= (bounds.hi[d - 1] - bounds.lo[d - 1] + 1);
      }

      inst_layout->fields[field_ids[i]] = field_layout;
      inst_layout->piece_lists[i].pieces.push_back(affine_piece);
      inst_layout->bytes_used += bounds.volume() * field_sizes[i];
    }

    return inst_layout;
  }

  template <int N, typename T>
  RegionInstanceImpl *create_inst(MemoryImpl *mem_impl, Rect<N, T> bounds,
                                  const std::vector<FieldID> &field_ids,
                                  const std::vector<size_t> &field_sizes)
  {
    RegionInstance inst = ID::make_instance(0, 0, 0, 0).convert<RegionInstance>();
    InstanceLayout<N, T> *inst_layout = create_layout(bounds, field_ids, field_sizes);
    RegionInstanceImpl *impl = new RegionInstanceImpl(nullptr, inst, mem_impl);
    impl->metadata.layout = inst_layout;
    impl->metadata.inst_offset = 0;
    return impl;
  }

  template <typename Func, size_t... Is>
  void dispatch_for_dimension(int dim, Func &&func, std::index_sequence<Is...>)
  {
    (
        [&] {
          if(dim == static_cast<int>(Is + 1)) {
            func(std::integral_constant<int, Is + 1>{});
          }
        }(),
        ...);
  }

} // namespace Realm
