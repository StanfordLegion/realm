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

#include "realm.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#ifdef REALM_USE_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

// Compute load for gpu_copy_streams.  Every thread runs a chain of `spin`
// dependent floating-point operations and stores the result, so a block lasts for
// a time proportional to `spin` and a grid of many blocks keeps blocks waiting for
// a place on the GPU.  Threads of different blocks may store to the same element;
// the values are never read.
__global__ void load_kernel(float *buffer, size_t elements, size_t spin)
{
  size_t idx = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) % elements;
  float x = float(threadIdx.x + 1) * 0.00001f;
  for(size_t i = 0; i < spin; i++) {
    x = x * 0.999999f + 0.000001f;
  }
  buffer[idx] = x;
}

#ifdef REALM_USE_CUDA
void launch_load_kernel(float *buffer, size_t elements, size_t blocks, size_t threads,
                        size_t spin, cudaStream_t stream)
#else
void launch_load_kernel(float *buffer, size_t elements, size_t blocks, size_t threads,
                        size_t spin, hipStream_t stream)
#endif
{
  if(blocks == 0 || threads == 0 || blocks > UINT_MAX || threads > UINT_MAX) {
    fprintf(stderr, "invalid load kernel grid/block dimensions\n");
    std::exit(1);
  }
  load_kernel<<<blocks, threads, 0, stream>>>(buffer, elements, spin);
#ifdef REALM_USE_CUDA
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr, "load kernel launch failed: %s\n", cudaGetErrorString(error));
    std::exit(1);
  }
#else
  hipError_t error = hipGetLastError();
  if(error != hipSuccess) {
    fprintf(stderr, "load kernel launch failed: %s\n", hipGetErrorString(error));
    std::exit(1);
  }
#endif
}
