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

// sparsity maps for Realm

// nop, but helps IDEs
#include "realm/inst_layout.h"
#include "realm/sparsity.h"

#include "realm/serialize.h"

TEMPLATE_TYPE_IS_SERIALIZABLE2(int N, typename T, Realm::SparsityMap<N, T>);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  template <int N, typename T>
  inline bool SparsityMap<N, T>::operator<(const SparsityMap<N, T> &rhs) const
  {
    return id < rhs.id;
  }

  template <int N, typename T>
  inline bool SparsityMap<N, T>::operator==(const SparsityMap<N, T> &rhs) const
  {
    return id == rhs.id;
  }

  template <int N, typename T>
  inline bool SparsityMap<N, T>::operator!=(const SparsityMap<N, T> &rhs) const
  {
    return id != rhs.id;
  }

  template <int N, typename T>
  REALM_CUDA_HD inline bool SparsityMap<N, T>::exists(void) const
  {
    return id != 0;
  }

  template <int N, typename T>
  REALM_PUBLIC_API inline std::ostream &operator<<(std::ostream &os, SparsityMap<N, T> s)
  {
    return os << std::hex << s.id << std::dec;
  }

  template <int N, typename T>
  REALM_PUBLIC_API inline std::ostream &operator<<(std::ostream &os,
                                                   const SparsityMapEntry<N, T> &entry)
  {
    os << entry.bounds;
    if(entry.sparsity.id)
      os << ",sparsity=" << std::hex << entry.sparsity.id << std::dec;
    if(entry.bitmap)
      os << ",bitmap=" << entry.bitmap;
    return os;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapPublicImpl<N,T>

  template <int N, typename T>
  inline bool SparsityMapPublicImpl<N, T>::is_valid(bool precise /*= true*/)
  {
    return (precise ? entries_valid.load_acquire() : approx_valid.load_acquire());
  }

  template <int N, typename T>
  inline const span<SparsityMapEntry<N, T>> SparsityMapPublicImpl<N, T>::get_entries(void)
  {
    if(!entries_valid.load_acquire())
      REALM_ASSERT(0, "get_entries called on sparsity map without valid data");
    if(from_gpu) {
      return span<SparsityMapEntry<N, T>>(
          reinterpret_cast<SparsityMapEntry<N, T> *>(entries_instance.pointer_untyped(
              0, num_entries * sizeof(SparsityMapEntry<N, T>))),
          num_entries);
    } else {
      return span<SparsityMapEntry<N, T>>(entries.data(), entries.size());
    }
  }

  template <int N, typename T>
  inline const span<Rect<N, T>> SparsityMapPublicImpl<N, T>::get_approx_rects(void)
  {
    if(!approx_valid.load_acquire())
      REALM_ASSERT(0, "get_approx_rects called on sparsity map without valid data");
    if(from_gpu) {
      return span<Rect<N, T>>(
          reinterpret_cast<Rect<N, T> *>(
              approx_instance.pointer_untyped(0, num_approx * sizeof(Rect<N, T>))),
          num_approx);
    } else {
      return span<Rect<N, T>>(approx_rects.data(), approx_rects.size());
    }
  }

}; // namespace Realm
