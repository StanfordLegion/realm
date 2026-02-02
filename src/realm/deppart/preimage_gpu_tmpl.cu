/* Copyright 2024 Stanford University, NVIDIA Corporation
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

// per‐dimension instantiator for the GPU version of
// ImageMicroOp<…>::gpu_populate_bitmasks_ptrs

#define REALM_TEMPLATES_ONLY
#include "realm/deppart/preimage_gpu_kernels.hpp"
#include "realm/deppart/preimage_gpu_impl.hpp"

#ifndef INST_N1
  #error "INST_N1 must be defined before including preimage_gpu_tmpl.cu"
#endif
#ifndef INST_N2
  #error "INST_N2 must be defined before including preimage_gpu_tmpl.cu"
#endif

// same set of T1,T2 pairs you use on the CPU side:
#define FOREACH_TT(__func__)       \
  __func__(int,    int)            \
  __func__(int,    unsigned)       \
  __func__(int,    long long)      \
  __func__(unsigned,int)           \
  __func__(unsigned,unsigned)      \
  __func__(unsigned,long long)     \
  __func__(long long, int)         \
  __func__(long long, unsigned)    \
  __func__(long long, long long)

#define FOREACH_T(__func__)       \
  __func__(int)            \
  __func__(unsigned)       \
  __func__(long long)

namespace Realm {
  #define N1 INST_N1
  #define N2 INST_N2

  // Replace MyBitmask with whatever bitmask‐type you actually use
  // (it must have an `as_vector.rects` member that your code touches).
  //
  // This explicitly instantiates:
  //   template void
  //   ImageMicroOp<N1,T1,N2,T2>::gpu_populate_bitmasks_ptrs<MyBitmask>(
  //     std::map<int,MyBitmask*>&);
  //
  #define DO_DOUBLE(T1,T2) \
    template class GPUPreimageMicroOp<N1,T1,N2,T2>; \
    template class PreimageMicroOp<N1,T1,N2,T2>;

  FOREACH_TT(DO_DOUBLE)

  #undef DO_DOUBLE
  #undef N1
  #undef N2

} // namespace Realm