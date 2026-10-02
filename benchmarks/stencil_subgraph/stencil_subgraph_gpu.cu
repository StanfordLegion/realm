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

#include "stencil_subgraph.h"
#include "realm/cuda/cuda_module.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace Realm;

constexpr int64_t TX = 16;
constexpr int64_t TY = 16;

static void check_launch(const char *what)
{
  cudaError_t err = cudaPeekAtLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "%s: kernel launch failed: %s\n", what, cudaGetErrorString(err));
    abort();
  }
}

__global__ void stencil_kernel(AffineAccessor<float, 2> input,
                               AffineAccessor<float, 2> output, int64_t nx, int64_t ny,
                               Rect<2> bounds)
{
  int64_t i = bounds.lo[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t j = bounds.lo[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  if(!bounds.contains(Point<2>(i, j)))
    return;
  float center = input[Point<2>(i, j)];
  float north = (i > 0) ? input[Point<2>(i - 1, j)] : 0.0f;
  float south = (i < nx - 1) ? input[Point<2>(i + 1, j)] : 0.0f;
  float west = (j > 0) ? input[Point<2>(i, j - 1)] : 0.0f;
  float east = (j < ny - 1) ? input[Point<2>(i, j + 1)] : 0.0f;
  output[Point<2>(i, j)] = (center + north + south + west + east) / 5.f;
}

__global__ void increment_kernel(AffineAccessor<float, 2> input,
                                 AffineAccessor<float, 2> output, Rect<2> bounds)
{
  int64_t i = bounds.lo[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t j = bounds.lo[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  if(!bounds.contains(Point<2>(i, j)))
    return;
  output[Point<2>(i, j)] = input[Point<2>(i, j)] + 1.f;
}

// Both tasks put all their work on the task's stream and are registered
// with DeferredEffectsProperty.
void stencil_task_gpu(const void *_args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  const StencilArgs *args = static_cast<const StencilArgs *>(_args);
  AffineAccessor<float, 2> input(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> output(args->buffer, FID_OUTPUT);
  Rect<2> bounds = args->local_space.bounds;
  int64_t blkx = (bounds.hi[0] - bounds.lo[0] + TX) / TX;
  int64_t blky = (bounds.hi[1] - bounds.lo[1] + TY) / TY;
  Cuda::set_task_ctxsync_required(false);
  cudaStream_t stream = Cuda::get_task_cuda_stream();
  stencil_kernel<<<dim3(blkx, blky, 1), dim3(TX, TY, 1), 0, stream>>>(
      input, output, args->hx, args->hy, bounds);
  check_launch("stencil");
}

void increment_task_gpu(const void *_args, size_t arglen, const void *userdata,
                        size_t userlen, Processor p)
{
  const IncrementArgs *args = static_cast<const IncrementArgs *>(_args);
  AffineAccessor<float, 2> input(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> output(args->buffer, FID_OUTPUT);
  Rect<2> bounds = args->local_space.bounds;
  int64_t blkx = (bounds.hi[0] - bounds.lo[0] + TX) / TX;
  int64_t blky = (bounds.hi[1] - bounds.lo[1] + TY) / TY;
  Cuda::set_task_ctxsync_required(false);
  cudaStream_t stream = Cuda::get_task_cuda_stream();
  increment_kernel<<<dim3(blkx, blky, 1), dim3(TX, TY, 1), 0, stream>>>(output, input,
                                                                        bounds);
  check_launch("increment");
}
