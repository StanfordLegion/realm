/* Copyright 2025 Stanford University, NVIDIA Corporation
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

// Kernels for the GPU subgraph tests: a kernel that waits a while and then
// writes a value, so that tests can tell whether the host ran ahead of the
// device and whether dependents saw the device's writes.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ static long long global_timer_ns()
{
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// After spin_ns nanoseconds, *dst = (src ? *src : 0) + add.
__global__ void spin_add_kernel(int *dst, const int *src, int add, long long spin_ns)
{
  const long long start = global_timer_ns();
  while(global_timer_ns() - start < spin_ns)
    ;
  const int v = src ? *src : 0;
  *dst = v + add;
}

extern "C" void subgraph_gpu_spin_add(void *stream, int *dst, const int *src, int add,
                                      long long spin_ns)
{
  spin_add_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(dst, src, add, spin_ns);
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    abort();
  }
}
