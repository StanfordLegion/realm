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


#include "realm/deppart/partitions_gpu_impl.hpp"
#include "realm/deppart/inst_helper.h"

namespace Realm {
  #define DOIT(N,T) \
    template class GPUMicroOp<N, T>;

  FOREACH_NT(DOIT)

} // namespace Realm