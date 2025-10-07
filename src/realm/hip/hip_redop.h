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
#ifndef REALM_HIP_REDOP_H
#define REALM_HIP_REDOP_H

#include "realm/realm_config.h"
#include "hip_reduc.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifdef REALM_USE_HIP
#include <hip/hip_runtime.h>
#endif
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace Realm {
  namespace Hip {
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
    static __device__ inline size_t
    coords_to_index(Offset_t *coords, const Offset_t *strides, size_t elem_size)
    {
      size_t i = 0;
      size_t vol = 1;
      int d = 0;
      coords[0] = coords[0] * elem_size;
#pragma unroll
      for(; d < N - 1; d++) {
        i += vol * coords[d];
        vol *= strides[d];
      }

      i += vol * coords[d];
      i = i / elem_size;
      return i;
    }

    template <size_t N, typename Offset_t = size_t>
    static __device__ inline size_t
    coords_to_index_trans(Offset_t *coords, const Offset_t *strides, size_t elem_size)
    {
      size_t i = 0;
      i = coords[1] * strides[0] + coords[2] * strides[1] + coords[0];
      return i;
    }

    // the ability to add CUDA kernels to a reduction op is only available
    //  when using a compiler that understands CUDA
    namespace ReductionKernels {
      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
                                       uintptr_t rhs_base, uintptr_t rhs_stride,
                                       size_t count, REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x) {
          redop.template apply_hip<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(lhs_base + idx * lhs_stride),
              *reinterpret_cast<const typename REDOP::RHS *>(rhs_base +
                                                             idx * rhs_stride));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(uintptr_t rhs1_base, uintptr_t rhs1_stride,
                                      uintptr_t rhs2_base, uintptr_t rhs2_stride,
                                      size_t count, REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x)
          redop.template fold_hip<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(rhs1_base + idx * rhs1_stride),
              *reinterpret_cast<const typename REDOP::RHS *>(rhs2_base +
                                                             idx * rhs2_stride));
      }
    }; // namespace ReductionKernels

    namespace ReductionKernelsAdv {
      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(Realm::Hip::AffineReducInfo<3> info, REDOP redop)
      {
        size_t nrects = info.num_rects;
        size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t rect = 0; rect < nrects; rect++) {
          Realm::Hip::AffineReducPair<3> &current_info = info.subrects[rect];
          size_t vol = current_info.volume / sizeof(typename REDOP::LHS);
          typename REDOP::RHS *dst =
              reinterpret_cast<typename REDOP::RHS *>(current_info.dst.addr);
          typename REDOP::RHS *src =
              reinterpret_cast<typename REDOP::RHS *>(current_info.src.addr);
          for(; offset < vol; offset += blockDim.x * gridDim.x) {
            size_t src_coords[3];
            index_to_coords<3, size_t>(src_coords, offset, current_info.extents);
            const size_t src_idx = coords_to_index<3, size_t>(
                src_coords, current_info.src.strides, sizeof(typename REDOP::RHS));
            size_t dst_coords[3];
            index_to_coords<3, size_t>(dst_coords, offset, current_info.extents);
            const size_t dst_idx = coords_to_index<3, size_t>(
                dst_coords, current_info.dst.strides, sizeof(typename REDOP::LHS));
            redop.template fold_hip<EXCL>(
                *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx]),
                *reinterpret_cast<typename REDOP::RHS *>(&src[src_idx]));
          }
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(Realm::Hip::AffineReducInfo<3> info, REDOP redop)
      {
        size_t nrects = info.num_rects;
        size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t rect = 0; rect < nrects; rect++) {
          Realm::Hip::AffineReducPair<3> &current_info = info.subrects[rect];
          size_t vol = current_info.volume / sizeof(typename REDOP::LHS);
          typename REDOP::LHS *dst =
              reinterpret_cast<typename REDOP::LHS *>(current_info.dst.addr);

          typename REDOP::RHS *src =
              reinterpret_cast<typename REDOP::RHS *>(current_info.src.addr);

          for(; offset < vol; offset += blockDim.x * gridDim.x) {
            size_t src_coords[3];
            index_to_coords<3, size_t>(src_coords, offset, current_info.extents);
            const size_t src_idx = coords_to_index<3, size_t>(
                src_coords, current_info.src.strides, sizeof(typename REDOP::RHS));

            size_t dst_coords[3];
            index_to_coords<3, size_t>(dst_coords, offset, current_info.extents);
            const size_t dst_idx = coords_to_index<3, size_t>(
                dst_coords, current_info.dst.strides, sizeof(typename REDOP::LHS));
            redop.template apply_hip<EXCL>(
                *reinterpret_cast<typename REDOP::LHS *>(&dst[dst_idx]),
                *reinterpret_cast<typename REDOP::RHS *>(&src[src_idx]));
          }
        }
      }
    }; // namespace ReductionKernelsAdv

    namespace ReductionKernelsAdvTranspose {
      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(Realm::Hip::MemReducInfo<size_t> current_info,
                                      REDOP redop)
      {
        size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        size_t vol = current_info.volume / sizeof(typename REDOP::LHS);
        typename REDOP::RHS *dst =
            reinterpret_cast<typename REDOP::RHS *>(current_info.dst);
        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src);

        for(; offset < vol; offset += blockDim.x * gridDim.x) {
          size_t src_coords[3];
          index_to_coords<3, size_t>(src_coords, offset, current_info.extents);
          const size_t src_idx = coords_to_index_trans<3, size_t>(
              src_coords, current_info.src_strides, sizeof(typename REDOP::RHS));

          size_t dst_coords[3];
          index_to_coords<3, size_t>(dst_coords, offset, current_info.extents);
          const size_t dst_idx = coords_to_index_trans<3, size_t>(
              dst_coords, current_info.dst_strides, sizeof(typename REDOP::LHS));
          redop.template fold_hip<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx]),
              *reinterpret_cast<typename REDOP::RHS *>(&src[src_idx]));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(Realm::Hip::MemReducInfo<size_t> current_info,
                                       REDOP redop)
      {
        size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        size_t vol = current_info.volume / sizeof(typename REDOP::LHS);

        typename REDOP::LHS *dst =
            reinterpret_cast<typename REDOP::LHS *>(current_info.dst);

        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src);

        for(; offset < vol; offset += blockDim.x * gridDim.x) {
          size_t src_coords[3];
          index_to_coords<3, size_t>(src_coords, offset, current_info.extents);
          const size_t src_idx = coords_to_index_trans<3, size_t>(
              src_coords, current_info.src_strides, sizeof(typename REDOP::RHS));

          size_t dst_coords[3];
          index_to_coords<3, size_t>(dst_coords, offset, current_info.extents);
          const size_t dst_idx = coords_to_index_trans<3, size_t>(
              dst_coords, current_info.dst_strides, sizeof(typename REDOP::LHS));
          redop.template apply_hip<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(&dst[dst_idx]),
              *reinterpret_cast<typename REDOP::RHS *>(&src[src_idx]));
        }
      }
    }; // namespace ReductionKernelsAdvTranspose

    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_hip_redop_kernels_adv(T *redop)
    {
      redop->hip_apply_excl_fn_adv =
          reinterpret_cast<void *>(&ReductionKernelsAdv::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn_adv =
          reinterpret_cast<void *>(&ReductionKernelsAdv::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn_adv =
          reinterpret_cast<void *>(&ReductionKernelsAdv::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn_adv =
          reinterpret_cast<void *>(&ReductionKernelsAdv::fold_hip_kernel<REDOP, false>);
      redop->hip_apply_excl_fn_tran_adv = reinterpret_cast<void *>(
          &ReductionKernelsAdvTranspose::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn_tran_adv = reinterpret_cast<void *>(
          &ReductionKernelsAdvTranspose::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn_tran_adv = reinterpret_cast<void *>(
          &ReductionKernelsAdvTranspose::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn_tran_adv = reinterpret_cast<void *>(
          &ReductionKernelsAdvTranspose::fold_hip_kernel<REDOP, false>);
    }

    // this helper adds the appropriate kernels for REDOP to a ReductionOpUntyped,
    // although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_hip_redop_kernels(T *redop)
    {
      // store the host proxy function pointer, as it's the same for all
      // devices - translation to actual hip functions happens later
      redop->hip_apply_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, false>);
      add_hip_redop_kernels_adv<REDOP, T>(redop);
    }
  }; // namespace Hip
};   // namespace Realm
#endif
#endif
