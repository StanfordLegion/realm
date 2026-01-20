#include "realm.h"
#include "realm/cuda/cuda_module.h"

#include "stencil_subgraph.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace Realm;

constexpr int64_t TX = 16;
constexpr int64_t TY = 16;

__global__
void stencil_kernel(AffineAccessor<float, 2> input, AffineAccessor<float, 2> output, int64_t nx, int64_t ny, Rect<2> bounds) {
  auto i = bounds.lo[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  auto j = bounds.lo[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  if (!bounds.contains({i, j}))
    return;
  float center = input[{i, j}];
  float north = i > 0 ? input[{i - 1, j}] : 0.0f;
  float south = i < nx - 1 ? input[{i + 1, j}] : 0.0f;
  float west = j > 0 ? input[{i, j - 1}] : 0.0f;
  float east = j < ny - 1 ? input[{i, j + 1}] : 0.0f;
  output[{i, j}] = (center + north + south + west + east) / 5.f;
}

void stencil_task_gpu(const void *_args, size_t arglen,
                  const void *userdata, size_t userlen, Processor p) {
  StencilArgs* args = (StencilArgs*)(_args);
  AffineAccessor<float, 2> finput_acc(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> foutput_acc(args->buffer, FID_OUTPUT);
  auto bounds = args->local_space.bounds;

  // Launch a 16x16 thread block grid.
  int64_t blkx = (bounds.hi[0] - bounds.lo[0] + TX - 1) / TX;
  int64_t blky = (bounds.hi[1] - bounds.lo[1] + TY - 1) / TY;
  Cuda::set_task_ctxsync_required(false);
  auto stream = Cuda::get_task_cuda_stream();
  dim3 blockShape(TX, TY, 1);
  dim3 gridShape(blkx, blky, 1);
  stencil_kernel<<<gridShape, blockShape, 0, stream>>>(finput_acc, foutput_acc, args->hx, args->hy, bounds);
  gpuErrchk( cudaPeekAtLastError() );
}

__global__
void increment_kernel(AffineAccessor<float, 2> input, AffineAccessor<float, 2> output, Rect<2> bounds) {
  auto i = bounds.lo[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  auto j = bounds.lo[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  if (!bounds.contains({i, j}))
    return;
  output[{i, j}] = input[{i, j}] + 1.f;
}

void increment_task_gpu(const void *_args, size_t arglen,
                  const void *userdata, size_t userlen, Processor p) {
  IncrementArgs* args = (IncrementArgs*)(_args);
  AffineAccessor<float, 2> finput_acc(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> foutput_acc(args->buffer, FID_OUTPUT);
  auto bounds = args->local_space.bounds;

  int64_t blkx = (bounds.hi[0] - bounds.lo[0] + TX - 1) / TX;
  int64_t blky = (bounds.hi[1] - bounds.lo[1] + TY - 1) / TY;
  Cuda::set_task_ctxsync_required(false);
  auto stream = Cuda::get_task_cuda_stream();
  dim3 blockShape(TX, TY, 1);
  dim3 gridShape(blkx, blky, 1);
  increment_kernel<<<gridShape, blockShape, 0, stream>>>(foutput_acc, finput_acc, bounds);
  gpuErrchk( cudaPeekAtLastError() );
}
