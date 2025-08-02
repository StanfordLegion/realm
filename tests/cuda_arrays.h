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

#ifndef CUDA_ARRAYS_H
#define CUDA_ARRAYS_H

#include "realm.h"

#include <surface_types.h>

#define CHECK_CUDART(cmd)                                                                \
  do {                                                                                   \
    cudaError_t ret = (cmd);                                                             \
    if(ret != cudaSuccess) {                                                             \
      fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret));     \
      assert(0);                                                                         \
      exit(1);                                                                           \
    }                                                                                    \
  } while(0)

void smooth_kernel(Realm::Rect<1> extent, float alpha, cudaSurfaceObject_t surf_in,
                   cudaSurfaceObject_t surf_out);
void smooth_kernel(Realm::Rect<2> extent, float alpha, cudaSurfaceObject_t surf_in,
                   cudaSurfaceObject_t surf_out);
void smooth_kernel(Realm::Rect<3> extent, float alpha, cudaSurfaceObject_t surf_in,
                   cudaSurfaceObject_t surf_out);

#endif
