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

#include <assert.h>
#include <stdio.h>

#include <iostream>

#include "realm/cuda/cuda_memcpy.h"
#include "realm/point.h"
#include "realm/sparsity.h"

/*using namespace Realm;
using namespace Realm::PieceLookup;

template <int N, typename COORD_T>
static __device__ inline uintptr_t execute_lookup(const Instruction *ip,
                                                  Realm::Point<N, COORD_T> point,
                                                  uintptr_t field_base, size_t field_size)
{
  while(true) {
    switch(ip->opcode()) {

    case Opcodes::OP_SPLIT1:
    {
      const auto *sp = static_cast<const SplitPlane<N, COORD_T> *>(ip);
      ip = sp->next(point);
      break;
    }

    case Opcodes::OP_AFFINE_PIECE:
    {
      const auto *ap = static_cast<const AffinePiece<N, COORD_T> *>(ip);

      size_t lin = 0;
#pragma unroll
      for(int d = 0; d < N; d++)
        lin += static_cast<size_t>(point[d] - ap->bounds.lo[d]) *
               static_cast<size_t>(ap->strides[d] / field_size);

      return field_base + ap->base + lin * field_size;
    }

    default:
      return 0; // unsupported opcode => error
    }
  }
}*/

// The general formula for a linearized index is the following:
// I = \sum_{i=0}^{N} v_i (\prod_{j=0}^{i-1} D_j)
// This implies that the general form for a coordinate is:
// v_i = mod(div, D_j) where div is the floored dividend of all the dimensions D
// of the earlier dimensions, e.g. for 3D:
// I = z * D_y * D_x + y * D_x + x
// x = mod(I, D_x)                      = I % D_x
// y = mod(div(I, D_x), D_y)            = (I / D_x) % D_y
// z = mod(div(div((I,D_x),D_y), D_z)   = ((I / D_x) / D_y) % D_z

template <size_t N, typename Offset_t = size_t>
static __device__ inline void index_to_coords(Offset_t *coords, Offset_t index,
                                              const Offset_t *extents)
{
  size_t div = index;
#pragma unroll
  for(int i = 0; i < N - 1; i++) {
    size_t div_tmp = div / extents[i];
    coords[i] = div - div_tmp * extents[i];
    div = div_tmp;
  }
  coords[N - 1] = div;
}

template <size_t N, typename Offset_t = size_t>
static __device__ inline size_t coords_to_index(Offset_t *coords, const Offset_t *strides)
{
  size_t i = 0;
  size_t vol = 1;
  int d = 0;

#pragma unroll
  for(; d < N - 1; d++) {
    i += vol * coords[d];
    vol *= strides[d];
  }

  i += vol * coords[d];

  return i;
}

template <typename T, typename Offset_t = size_t>
static __device__ inline void
memcpy_kernel_transpose(Realm::Cuda::MemcpyTransposeInfo<Offset_t> info, T *tile)
{
  __restrict__ T *out_base = reinterpret_cast<T *>(info.dst);
  __restrict__ T *in_base = reinterpret_cast<T *>(info.src);
  const Offset_t tile_size = info.tile_size;

  const Offset_t tidx = threadIdx.x % tile_size;
  const Offset_t tidy = (threadIdx.x / tile_size) % tile_size;

  const Offset_t grid_dimx = ((info.extents[2] + tile_size - 1) / tile_size);
  const Offset_t grid_dimy = ((info.extents[1] + tile_size - 1) / tile_size);

  const Offset_t chunks = info.extents[0] / sizeof(T);

  const Offset_t src_size_dim0 = info.src_strides[0] / sizeof(T);
  const Offset_t src_size_dim1 = info.src_strides[1] / sizeof(T);

  const Offset_t dst_size_dim1 = info.dst_strides[1] / sizeof(T);
  const Offset_t dst_size_dim0 = info.dst_strides[0] / sizeof(T);

  for(Offset_t block = blockIdx.x; block < grid_dimx * grid_dimy; block += gridDim.x) {
    Offset_t block_idx = block % grid_dimx;
    Offset_t block_idy = block / grid_dimx;

    Offset_t gx_tile_idx = block_idx * tile_size * chunks + tidx;
    Offset_t gy_tile_idx = block_idy * tile_size + tidy;

    __syncthreads();

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(gx_tile_idx + block_offset < info.extents[2] * chunks &&
         gy_tile_idx < info.extents[1]) {
        Offset_t in_tile_idx = tidx + (tile_size + 1) * tidy * chunks;

        Offset_t xe = gx_tile_idx + block_offset;
        Offset_t chunk_idx = xe / chunks;
        Offset_t chunk_rem = xe % chunks;

        tile[in_tile_idx + block_offset] =
            in_base[chunk_idx * src_size_dim1 + gy_tile_idx * src_size_dim0 + chunk_rem];
      }
    }

    __syncthreads();

    gx_tile_idx = block_idy * tile_size * chunks + tidx;
    gy_tile_idx = block_idx * tile_size + tidy;

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(gx_tile_idx + block_offset < info.extents[1] * chunks &&
         gy_tile_idx < info.extents[2]) {
        Offset_t out_tile_idx =
            (tidy + (tile_size + 1) * ((tidx + block_offset) / chunks)) * chunks +
            (tidx + block_offset) % chunks;

        Offset_t xe = gx_tile_idx + block_offset;
        Offset_t chunk_idx = xe / chunks;
        Offset_t chunk_rem = xe % chunks;

        out_base[chunk_idx * dst_size_dim0 + gy_tile_idx * dst_size_dim1 + chunk_rem] =
            tile[out_tile_idx];
      }
    }
  }
}

#define MAX_UNROLL (1)

template <typename T, size_t N, typename Offset_t = size_t>
static __device__ inline void
memcpy_affine_batch(Realm::Cuda::AffineCopyPair<N, Offset_t> *info, size_t nrects,
                    size_t start_offset = 0)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x - start_offset;
  const unsigned grid_stride = gridDim.x * blockDim.x;

  for(size_t rect = 0; rect < nrects; rect++) {
    Realm::Cuda::AffineCopyPair<N, Offset_t> &current_info = info[rect];
    const Offset_t vol = current_info.volume;
    __restrict__ T *dst = reinterpret_cast<T *>(current_info.dst.addr);
    __restrict__ T *src = reinterpret_cast<T *>(current_info.src.addr);

    while(offset < vol) {
      T tmp[MAX_UNROLL];
      unsigned i;

#pragma unroll
      for(i = 0; i < MAX_UNROLL; i++) {
        Offset_t src_coords[N];
        if((offset + i * grid_stride) >= vol) {
          break;
        }
        index_to_coords<N, Offset_t>(src_coords, offset + i * grid_stride,
                                     current_info.extents);
        const size_t src_idx =
            coords_to_index<N, Offset_t>(src_coords, current_info.src.strides);
        tmp[i] = src[src_idx];
      }
      for(unsigned j = 0; j < i; j++) {
        Offset_t dst_coords[N];

        index_to_coords<N, Offset_t>(dst_coords, (offset + j * grid_stride),
                                     current_info.extents);

        const size_t dst_idx =
            coords_to_index<N, Offset_t>(dst_coords, current_info.dst.strides);
        dst[dst_idx] = tmp[j];
      }

      offset += i * grid_stride;
    }

    // Skip this rectangle as it's covered by another thread
    // This can split the warp, and it may not coalesce again unless we sync them
    offset -= vol;
  }
}

template <int N, typename COORD_T, typename Offset_t>
static __device__ inline unsigned
locate_rectangle(const Realm::SparsityMapEntry<N, COORD_T> *__restrict__ drects,
                 Offset_t num_rects, Offset_t global_lin, Offset_t &local_lin)
{
  Offset_t lo = 0;
  Offset_t hi = num_rects;

  while(lo < hi) {
    Offset_t mid = (lo + hi) >> 1;
    Offset_t start = drects[mid].prefix_sum;
    Offset_t end = start + drects[mid].bounds.volume();

    if(global_lin < start) {
      hi = mid;
    } else if(global_lin >= end) {
      lo = mid + 1;
    } else {
      local_lin = global_lin - start;
      return mid;
    }
  }

  local_lin = 0;
  return num_rects;
}

template <int N, typename COORD_T, typename DATA_T, typename Offset_t = size_t>
static __device__ inline void
memcpy_indirect_points(Realm::Cuda::MemcpyIndirectInfo<3, Offset_t> info)
{
  //--------------------------------------------------------------------
  // 0. flat thread id and stride
  //--------------------------------------------------------------------
  const Offset_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const Offset_t grid_step = gridDim.x * blockDim.x;

  //--------------------------------------------------------------------
  // 1. handy aliases (all in registers)
  //--------------------------------------------------------------------
  const Offset_t vec = info.field_size / sizeof(DATA_T);

  const COORD_T *__restrict__ ind_base = reinterpret_cast<const COORD_T *>(info.ind_base);
  const Realm::SparsityMapEntry<N, COORD_T> *__restrict__ drects =
      reinterpret_cast<const Realm::SparsityMapEntry<N, COORD_T> *>(info.domain_base);

  const auto src_pcs = (info.src_pieces_ptr);
  const auto dst_pcs = (info.dst_pieces_ptr);

  //--------------------------------------------------------------------
  // 2. thread-local linear index in the *whole* domain
  //--------------------------------------------------------------------
  Offset_t lin = info.point_pos + tid;
  Offset_t gidx = info.point_pos + tid;
  Offset_t curr_rect_vol = drects[0].bounds.volume();

  // TODO: PREFIX SUM
  //--------------------------------------------------------------------
  // 3. locate the rectangle that owns ‘lin’
  //--------------------------------------------------------------------
  // unsigned r = 0;
  /*while((r < info.domain_rects) && (lin >= curr_rect_vol)) {
    lin -= curr_rect_vol;
    r++;
    if(r < info.domain_rects) {
      curr_rect_vol = drects[r].bounds.volume();
    }
  }*/

  Offset_t local_lin = 0;
  unsigned r =
      locate_rectangle<N, COORD_T, Offset_t>(drects, info.domain_rects, gidx, local_lin);
  lin = local_lin;

  while((r < info.domain_rects) && (gidx < info.volume)) {
    const Realm::Rect<N, COORD_T> &rb = drects[r].bounds;

    Offset_t ind_linear_idx = 0;
#pragma unroll
    for(int d = 0; d < N; d++) {
      ind_linear_idx += rb.lo[d] * info.ind_strides[d];
    }

    Offset_t tmp = lin;
    Offset_t ind_lin_ofs = 0;
#pragma unroll
    for(int d = 0; d < N; d++) {
      COORD_T len = rb.hi[d] - rb.lo[d] + 1;
      COORD_T coord_in_rect = tmp % len;
      tmp /= len;
      ind_lin_ofs += Offset_t(coord_in_rect) * info.ind_strides[d];
    }

    Realm::Point<N, COORD_T> src_pt;
#pragma unroll
    for(int d = 0; d < N; d++) {
      src_pt[d] = ind_base[(ind_linear_idx + ind_lin_ofs) * N + d];
      // src_pt[d] = ind_base[(ind_linear_idx + lin) * N + d];
    }

    Realm::Point<N, COORD_T> dst_pt;
    tmp = lin;
#pragma unroll
    for(int d = 0; d < N; d++) {
      const COORD_T len = rb.hi[d] - rb.lo[d] + 1;
      dst_pt[d] = rb.lo[d] + (tmp % len);
      tmp /= len;
    }

    auto inside = [&](const auto &pc, const Realm::Point<N, COORD_T> &q) {
#pragma unroll
      for(int d = 0; d < N; d++) {
        if((q[d] < pc.lo[d]) || (q[d] > pc.hi[d])) {
          return false;
        }
        return true;
      }
    };

    int32_t src_pidx = -1;
    int32_t dst_pidx = -1;

    // TODO: ACCELERATE ME
    for(int32_t i = 0; i < info.num_src_pieces && src_pidx == -1; i++) {
      if(inside(src_pcs[i], src_pt)) {
        src_pidx = i;
      }
    }

    // TODO: ACCELERATE ME
    for(int32_t j = 0; j < info.num_dst_pieces && dst_pidx == -1; j++) {
      if(inside(dst_pcs[j], dst_pt)) {
        dst_pidx = j;
      }
    }

    if(src_pidx == -1 || dst_pidx == -1) {
      return;
    }

    //---------------- byte address calculations ----------------------
    Offset_t src_lin = 0, dst_lin = 0;
#pragma unroll
    for(int d = 0; d < N; d++) {
      src_lin +=
          Offset_t(src_pt[d] - src_pcs[src_pidx].lo[d]) * src_pcs[src_pidx].strides[d];
      dst_lin +=
          Offset_t(dst_pt[d] - dst_pcs[dst_pidx].lo[d]) * dst_pcs[dst_pidx].strides[d];
    }

    /*uintptr_t src_byte = execute_lookup<N>(info.src_instruction, src_pt,
                                           src_pcs[src_pidx].base, info.field_size);
    uintptr_t dst_byte = execute_lookup<N>(info.dst_instruction, dst_pt,
                                           dst_pcs[src_pidx].base, info.field_size);

    __restrict__ DATA_T *src = reinterpret_cast<DATA_T *>(src_byte);
    __restrict__ DATA_T *dst = reinterpret_cast<DATA_T *>(dst_byte);*/

    __restrict__ DATA_T *src =
        reinterpret_cast<DATA_T *>(src_pcs[src_pidx].base + src_lin * info.field_size);
    __restrict__ DATA_T *dst =
        reinterpret_cast<DATA_T *>(dst_pcs[dst_pidx].base + dst_lin * info.field_size);

    for(Offset_t v = 0; v < vec; v++) {
      dst[v] = src[v];
    }

    gidx += grid_step;
    lin += grid_step;

    r = locate_rectangle<N, COORD_T, Offset_t>(drects, info.domain_rects, gidx,
                                               local_lin);
    lin = local_lin;

    /*while((r < info.domain_rects) && (lin >= curr_rect_vol)) {
      lin -= curr_rect_vol;
      r++;
      if(r < info.domain_rects) {
        curr_rect_vol = drects[r].bounds.volume();
      }
    }*/
  }
}

template <int N, typename T, typename Offset_t = size_t>
static __device__ inline void
memfill_affine_batch(const Realm::Cuda::AffineFillInfo<N, Offset_t> &info)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned grid_stride = gridDim.x * blockDim.x;
  T fill_value = *reinterpret_cast<const T *>(info.fill_value);
  for(size_t rect = 0; rect < info.num_rects; rect++) {
    const Realm::Cuda::AffineFillRect<N, Offset_t> &current_info = info.subrects[rect];
    const Offset_t vol = current_info.volume;
    __restrict__ T *addr = reinterpret_cast<T *>(current_info.addr);
    while(offset < vol) {
      unsigned i = 0;
#pragma unroll
      for(i = 0; i < MAX_UNROLL; i++) {
        Offset_t coords[N];
        if((offset + i * grid_stride) >= vol) {
          break;
        }
        index_to_coords<N, Offset_t>(coords, offset + i * grid_stride,
                                     current_info.extents);
        const size_t idx = coords_to_index<N, Offset_t>(coords, current_info.strides);
        addr[idx] = fill_value;
      }
      offset += i * grid_stride;
    }
    // Skip this rectangle as it's covered by another thread
    // This can split the warp, and it may not coalesce again unless we sync them
    offset -= vol;
  }
}

#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_batch##name(        \
      Realm::Cuda::AffineCopyInfo<dim, offt> info)                                       \
  {                                                                                      \
    memcpy_affine_batch<type, dim, offt>(info.subrects, info.num_rects);                 \
  }

#define FILL_TEMPLATE_INST(type, dim, offt, name)                                        \
  extern "C" __global__ void fill_affine_batch##name(                                    \
      Realm::Cuda::AffineFillInfo<dim, offt> info)                                       \
  {                                                                                      \
    memfill_affine_batch<dim, type, offt>(info);                                         \
  }

#define FILL_LARGE_TEMPLATE_INST(type, dim, offt, name)                                  \
  extern "C" __global__ void fill_affine_large##name(                                    \
      Realm::Cuda::AffineLargeFillInfo<dim, offt> info)                                  \
  {}

#define MEMCPY_TRANSPOSE_TEMPLATE_INST(type, offt, name)                                 \
  extern "C" __global__ __launch_bounds__(1024) void memcpy_transpose##name(             \
      Realm::Cuda::MemcpyTransposeInfo<offt> info)                                       \
  {                                                                                      \
    extern __shared__ type tile_shared_##name[];                                         \
    memcpy_kernel_transpose<type, offt>(info, tile_shared_##name);                       \
  }

#define MEMCPY_INDIRECT_TEMPLATE_INST(addr_type, data_type, dim, offt, name)             \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_indirect##name(            \
      Realm::Cuda::MemcpyIndirectInfo<3, offt> info)                                     \
  {                                                                                      \
    memcpy_indirect_points<dim, addr_type, data_type, offt>(info);                       \
  }

#define INST_TEMPLATES(type, sz, dim, off)                                               \
  MEMCPY_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                      \
  FILL_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                        \
  FILL_LARGE_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                  \
  MEMCPY_INDIRECT_TEMPLATE_INST(int, type, dim, off, dim##D_##sz##32)                    \
  MEMCPY_INDIRECT_TEMPLATE_INST(long long, type, dim, off, dim##D_##sz##64)

#define INST_TEMPLATES_FOR_TYPES(dim, off)                                               \
  INST_TEMPLATES(unsigned char, 8, dim, off)                                             \
  INST_TEMPLATES(unsigned short, 16, dim, off)                                           \
  INST_TEMPLATES(unsigned int, 32, dim, off)                                             \
  INST_TEMPLATES(unsigned long long, 64, dim, off)                                       \
  INST_TEMPLATES(uint4, 128, dim, off)

#define INST_TEMPLATES_FOR_DIMS()                                                        \
  INST_TEMPLATES_FOR_TYPES(1, size_t)                                                    \
  INST_TEMPLATES_FOR_TYPES(2, size_t)                                                    \
  INST_TEMPLATES_FOR_TYPES(3, size_t)

INST_TEMPLATES_FOR_DIMS()

MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned char, size_t, 8)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned short, size_t, 16)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned int, size_t, 32)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned long long, size_t, 64)
MEMCPY_TRANSPOSE_TEMPLATE_INST(uint4, size_t, 128)
