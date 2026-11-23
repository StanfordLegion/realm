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

#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdio.h>

#include <iostream>

#include "realm/hip/hip_memcpy.h"
#include "realm/point.h"

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
static __device__ inline void memcpy_kernel_transpose(
    Realm::Hip::MemcpyTransposeInfo<Offset_t> info, T* tile) {
  T* __restrict__ out_base = reinterpret_cast<T *>(info.dst);
  T* __restrict__ in_base = reinterpret_cast<T *>(info.src);
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
      if(gx_tile_idx + block_offset < info.extents[2] * chunks && gy_tile_idx < info.extents[1]) {
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
      if(gx_tile_idx + block_offset < info.extents[1] * chunks && gy_tile_idx < info.extents[2]) {
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

#if defined(__GFX9__)
// Affine copies on gfx9 (CDNA: MI100, MI200, MI300), whose waves are 64 lanes wide.
//
// Each wave copies whole tiles of a rectangle: 64 elements of one row, or, for rows
// narrower than 64 elements, as many whole rows as fit in 64 lanes.  The waves of the
// grid take turns at the tiles, as the threads of a grid-stride loop take turns at
// elements, and the tile index left over after a rectangle carries into the next one.
// Which tile a wave copies, and the tile's base addresses, are the same for the whole
// wave and live in scalar registers; each lane keeps only its 32-bit byte offsets within
// a tile and its data.  Loads and stores are buffer instructions, which take the scalar
// base and the 32-bit offset directly, so the copy loop does no per-element address
// arithmetic on the vector unit (about 2 vector instructions per KiB copied, against 95
// for the per-element kernel this replaced), and the kernel needs 8 vector registers.
//
// The kernel allocates HIP_COPY_KERNEL_VGPRS vector registers per wave for copies within
// GPU memory and HIP_HOST_COPY_KERNEL_VGPRS for copies that read or write host memory
// (hip_memcpy.h); anything above 8 is claimed but unused.  A copy wave small enough to
// fit in the registers an application's kernel leaves unused runs beside that kernel
// instead of taking a place it would otherwise use.  For copies within GPU memory that
// helps the application.  For host copies it hurts it: their loads or stores take
// microseconds, a CU's vector memory instructions share one path, and the application's
// memory instructions on the same CU wait behind them.  On MI250X, beside host->device
// copies, a task kernel of 248 vector registers took 1.45x as long with 8-register copy
// waves (which fit beside it on every CU) as with 24-register ones (which do not), and
// the difference vanished when the task kernel issued no memory instructions.  The
// per-element kernel used 24, so host copies keep that.  This only keeps their
// placement as it was: 24-register waves fit beside a 232-register task kernel and slow
// it as much.
//
// Requirements, which GPU::launch_batch_affine_kernel and the DMA path meet:
//  * blocks of 256 threads (four waves);
//  * fewer than 2^21 waves in the grid (the setup divides by the float reciprocal);
//  * at least one rectangle;
//  * fewer than 2^32 elements in a rectangle (tile counts are 32-bit).

// Makes the kernel allocate VGPRS vector registers per wave by claiming register
// VGPRS - 1 in an inline-asm clobber (the copy itself uses 8).
template <unsigned VGPRS>
struct CopyVgprs {
  static_assert(VGPRS == 8, "copy kernel VGPRs must be a multiple of 8 from 8 to 128");
  static __device__ inline void claim() {}
};
#define COPY_VGPRS(n, reg)                                                               \
  template <>                                                                            \
  struct CopyVgprs<n> {                                                                  \
    static __device__ inline void claim()                                                \
    {                                                                                    \
      asm volatile("; claim " #n " VGPRs" ::: #reg);                                     \
    }                                                                                    \
  };
COPY_VGPRS(16, v15)
COPY_VGPRS(24, v23)
COPY_VGPRS(32, v31)
COPY_VGPRS(40, v39)
COPY_VGPRS(48, v47)
COPY_VGPRS(56, v55)
COPY_VGPRS(64, v63)
COPY_VGPRS(72, v71)
COPY_VGPRS(80, v79)
COPY_VGPRS(88, v87)
COPY_VGPRS(96, v95)
COPY_VGPRS(104, v103)
COPY_VGPRS(112, v111)
COPY_VGPRS(120, v119)
COPY_VGPRS(128, v127)
#undef COPY_VGPRS

typedef unsigned int copy_v2u __attribute__((ext_vector_type(2)));
typedef unsigned int copy_v4u __attribute__((ext_vector_type(4)));

// A raw buffer resource at `base` whose range is the largest a 32-bit offset reaches, so
// that the range check never fails.  (The compiler does not use global_load's
// scalar-base form for a uniform base plus a 32-bit offset; it builds a 64-bit address
// in vector registers for every access.)  0x00020000 selects the 32-bit data format,
// which untyped buffer access on gfx9 requires.
static __device__ inline __amdgpu_buffer_rsrc_t copy_rsrc(uint64_t base)
{
  return __builtin_amdgcn_make_buffer_rsrc(reinterpret_cast<void *>(base), 0, 0xffffffffu,
                                           0x00020000);
}

// Loads and stores of one S-byte element at byte offset `off` from `base`
template <size_t S>
struct CopyWord;
#define COPY_WORD(bytes, word_type, bits)                                                \
  template <>                                                                            \
  struct CopyWord<bytes> {                                                               \
    typedef word_type type;                                                              \
    static __device__ inline type load(uint64_t base, unsigned off)                      \
    {                                                                                    \
      return __builtin_amdgcn_raw_buffer_load_b##bits(copy_rsrc(base), off, 0, 0);       \
    }                                                                                    \
    static __device__ inline void store(uint64_t base, unsigned off, type v)             \
    {                                                                                    \
      __builtin_amdgcn_raw_buffer_store_b##bits(v, copy_rsrc(base), off, 0, 0);          \
    }                                                                                    \
  };
COPY_WORD(1, unsigned char, 8)
COPY_WORD(2, unsigned short, 16)
COPY_WORD(4, unsigned int, 32)
COPY_WORD(8, copy_v2u, 64)
COPY_WORD(16, copy_v4u, 128)
#undef COPY_WORD

// floor(a / b) for a below 2^21 and any b >= 1, without a correction step: the
// fractional part of (a + 0.5) / b is between 0.5 / b and 1 - 0.5 / b, and the float
// reciprocal's error (with the product's rounding, under 2^-22 relative) moves the
// product by less than (a + 0.5) * 2^-22 / b < 0.5 / b.
static __device__ inline unsigned copy_div21(unsigned a, unsigned b)
{
  return unsigned((float(a) + 0.5f) * __builtin_amdgcn_rcpf(float(b)));
}
// The same for wave-uniform a and b, with the quotient moved to a scalar register.  The
// conversion from float is a vector instruction, and without the move the compiler
// computes everything derived from the quotient (the tile's base addresses, then the
// buffer resources, in a loop over the lanes) on the vector unit.
// __builtin_amdgcn_readfirstlane does not do this: it is removed for a value already
// known to be uniform.
static __device__ inline unsigned copy_udiv21(unsigned a, unsigned b)
{
  unsigned q;
  asm volatile("v_readfirstlane_b32 %0, %1" : "=s"(q) : "v"(copy_div21(a, b)));
  return q;
}

// A 1D rectangle of `vol` elements: tile t is elements [64t, 64t + 64).  Returns the
// wave's tile index in the next rectangle.
template <size_t S>
static __device__ inline unsigned copy_rect_1d(uint64_t src, uint64_t dst, uint64_t vol,
                                               unsigned t, unsigned nw, unsigned lane)
{
  // fewer than 2^32 tiles: vol is below 2^38
  const unsigned ntiles = unsigned((vol + 63) >> 6), nfull = unsigned(vol >> 6);
  const unsigned off = lane * S;
  const uint64_t step = uint64_t(nw) * (64 * S);
  uint64_t sb = src + uint64_t(t) * (64 * S), db = dst + uint64_t(t) * (64 * S);
  for(; t < nfull; t += nw) {
    CopyWord<S>::store(db, off, CopyWord<S>::load(sb, off));
    sb += step;
    db += step;
  }
  if((t < ntiles) && (lane < unsigned(vol & 63)))
    CopyWord<S>::store(db, off, CopyWord<S>::load(sb, off));
  return (t < ntiles) ? (t + nw - ntiles) : (t - ntiles);
}

// The current tile of a wave in a 2D/3D rectangle: its column and row of tiles within
// the plane, and its base addresses
struct CopyTile {
  unsigned xt, yt;
  uint64_t sb, db;
};

// One rectangle; t is the wave's first tile in it and nw the number of waves in the grid
// (both below 2^21).  Returns the wave's tile index in the next rectangle.
template <size_t S, size_t N, typename Offset_t>
static __device__ inline unsigned
copy_rect(const Realm::Hip::AffineCopyPair<N, Offset_t> &ci, unsigned t, unsigned nw,
          unsigned lane)
{
  constexpr unsigned LOG2S =
      (S == 16) ? 4 : ((S == 8) ? 3 : ((S == 4) ? 2 : ((S == 2) ? 1 : 0)));
  const uint64_t e0 = ci.extents[0];
  // row pitches (elements) and plane pitches (rows)
  uint64_t e1 = 1, e2 = 1, rs = 0, rd = 0, sp = 0, dp = 0;
  if constexpr(N > 1) {
    e1 = ci.extents[1];
    rs = ci.src.strides[0];
    rd = ci.dst.strides[0];
  }
  if constexpr(N > 2) {
    e2 = ci.extents[2];
    sp = ci.src.strides[1];
    dp = ci.dst.strides[1];
  }
  if((e1 == 1) && (e2 == 1))
    return copy_rect_1d<S>(ci.src.addr, ci.dst.addr, e0, t, nw, lane);

  // The tile is w elements of each of rr rows: either one row (rr == 1) or whole rows
  // (w == e0), so its valid lanes are always a prefix.  Several rows per tile only while
  // a lane's offset in the tile stays below 2^32 bytes (both pitches below 2^26 bytes).
  // 64-bit tests are split into 32-bit halves: LLVM turns (x >> k) == 0 into an unsigned
  // less-than, which the scalar unit cannot do in 64 bits.
  const unsigned w_e0 = unsigned(e0), w_e1 = unsigned(e1), w_e2 = unsigned(e2);
  unsigned w, rr, tx, ty;
  if((unsigned(e0 >> 32) | (unsigned(e0) >> 6)) != 0) { // e0 >= 64
    w = 64;
    rr = 1;
    tx = unsigned((e0 + 63) >> 6);
    ty = w_e1;
  } else {
    w = w_e0;
    tx = 1;
    const uint64_t rsd = rs | rd;
    rr = ((unsigned(rsd >> 32) | (unsigned(rsd) >> (26 - LOG2S))) == 0)
             ? copy_udiv21(64, w)
             : 1u;
    ty = (rr == 1) ? w_e1
                   : ((w_e1 < (1u << 20)) ? copy_udiv21(w_e1 + rr - 1, rr)
                                          : (w_e1 + rr - 1) / rr);
  }
  const unsigned ntiles = tx * ty * w_e2;
  if(t >= ntiles)
    return t - ntiles;

  // the wave's first tile
  const unsigned q = (tx == 1) ? t : copy_udiv21(t, tx);
  const unsigned z = copy_udiv21(q, ty);
  CopyTile k;
  k.xt = t - q * tx;
  k.yt = q - z * ty;
  const uint64_t ps = rs * sp, pd = rd * dp; // plane pitches, elements
  const uint64_t rrs = uint64_t(rr) * rs, rrd = uint64_t(rr) * rd;
  k.sb = ci.src.addr + ((z * ps + k.yt * rrs + uint64_t(k.xt) * w) << LOG2S);
  k.db = ci.dst.addr + ((z * pd + k.yt * rrd + uint64_t(k.xt) * w) << LOG2S);
  // this lane's place in every tile
  const unsigned r = copy_div21(lane, w), c = lane - r * w;
  const unsigned soff = (r * unsigned(rs) + c) << LOG2S;
  const unsigned doff = (r * unsigned(rd) + c) << LOG2S;
  // valid lanes of a tile: min(rows left, rr) rows of min(columns left, w) elements
  auto limit = [&](const CopyTile &m) {
    const unsigned cols = w_e0 - m.xt * w, rows = w_e1 - m.yt * rr;
    return ((rows < rr) ? rows : rr) * ((cols < w) ? cols : w);
  };
  auto copy_tile = [&](const CopyTile &m) {
    const unsigned lim = limit(m);
    if(lim == 64)
      CopyWord<S>::store(m.db, doff, CopyWord<S>::load(m.sb, soff));
    else if(lane < lim)
      CopyWord<S>::store(m.db, doff, CopyWord<S>::load(m.sb, soff));
  };
  if(t + nw >= ntiles) {
    // the only tile of this wave in this rectangle (any small copy): no step needed
    copy_tile(k);
    return t + nw - ntiles;
  }

  // The grid's step in tile coordinates, applied with carries, so that the loop does no
  // division.  as/ad: the step's address increment; bs/bd: a column wrap's (next row of
  // tiles); cs/cd: a row wrap's (next plane).
  const unsigned nq = (tx == 1) ? nw : copy_udiv21(nw, tx);
  const unsigned sz = copy_udiv21(nq, ty);
  const unsigned sx = nw - nq * tx, sy = nq - sz * ty;
  const uint64_t as = (sz * ps + sy * rrs + uint64_t(sx) * w) << LOG2S;
  const uint64_t ad = (sz * pd + sy * rrd + uint64_t(sx) * w) << LOG2S;
  const uint64_t bs = (rrs - uint64_t(tx) * w) << LOG2S;
  const uint64_t bd = (rrd - uint64_t(tx) * w) << LOG2S;
  const uint64_t cs = (ps - uint64_t(ty) * rrs) << LOG2S;
  const uint64_t cd = (pd - uint64_t(ty) * rrd) << LOG2S;
  for(; t < ntiles; t += nw) {
    copy_tile(k);
    k.xt += sx;
    k.yt += sy;
    k.sb += as;
    k.db += ad;
    if(k.xt >= tx) {
      k.xt -= tx;
      k.yt += 1;
      k.sb += bs;
      k.db += bd;
    }
    if(k.yt >= ty) {
      k.yt -= ty;
      k.sb += cs;
      k.db += cd;
    }
  }
  return t - ntiles;
}

template <size_t S, size_t N, typename Offset_t, unsigned VGPRS>
static __device__ inline void
memcpy_affine_tiles(const Realm::Hip::AffineCopyInfo<N, Offset_t> &info)
{
  CopyVgprs<VGPRS>::claim();
  const unsigned nrects = info.num_rects;
  const unsigned nw = gridDim.x * 4;
  const unsigned lane = __builtin_amdgcn_workitem_id_x() & 63;
  unsigned t = __builtin_amdgcn_workgroup_id_x() * 4 +
               __builtin_amdgcn_readfirstlane(__builtin_amdgcn_workitem_id_x() >> 6);
  // The first rectangle is copied before num_rects is tested, so that its descriptor and
  // num_rects are fetched together: kernel arguments are in host memory, and a chain of
  // dependent reads of them costs about a microsecond on a small copy.
  const Realm::Hip::AffineCopyPair<N, Offset_t> first = info.subrects[0];
  t = copy_rect<S>(first, t, nw, lane);
  // The lane number and wave count are made opaque at each further rectangle, so that the
  // compiler does not hoist values derived from them (the lane's offset, the wave count
  // as a float) out of this loop and keep them all live in vector registers.
  unsigned l = lane, n = nw;
  for(unsigned i = 1; i < nrects; i++) {
    asm volatile("" : "+v"(l), "+s"(n));
    t = copy_rect<S>(info.subrects[i], t, n, l);
  }
}
#else
// Affine copies on other targets (whose buffer resources and wave widths the kernel
// above does not handle): one element per thread in a grid-stride loop.

template <typename T, size_t N, typename Offset_t = size_t>
static __device__ inline void
memcpy_affine_batch(Realm::Hip::AffineCopyPair<N, Offset_t> *info,
                    size_t nrects, size_t start_offset = 0)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x - start_offset;
  const unsigned grid_stride = gridDim.x * blockDim.x;

  for(size_t rect = 0; rect < nrects; rect++) {
    Realm::Hip::AffineCopyPair<N, Offset_t> &current_info = info[rect];
    const Offset_t vol = current_info.volume;
    T *__restrict__ dst = reinterpret_cast<T *>(current_info.dst.addr);
    T *__restrict__ src = reinterpret_cast<T *>(current_info.src.addr);

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

        index_to_coords<N, Offset_t>(dst_coords,
                                     (offset + j * grid_stride),
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
#endif

/*
 * Scatter/gather points using indirection from/to dense buffer.
 * General assumptions:
 * 1. The src_ind/dst_ind buffer is dense and always has the same size
 * as the src/dst buffer (depending whether we are doing scatter or gather).
 * 2. src_ind_/dst_ind are accessed in a linear fashion with the base
 * type Point<N, Offset_t> per indirection element.
 * 3. src_ind/dst_ind do not have to be sorted but it should be
 * considered for coalesced access.
 *
 * TODO: Consider handling ranges where src_ind/dst_ind
 * contain Rect<N, Offset_t> instead of Point<N Offset_t>.
 *
 * */

template <int N, typename T, typename DT, typename Offset_t = size_t>
static __device__ inline void
memcpy_indirect_points(Realm::Hip::MemcpyIndirectInfo<3, Offset_t> info)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  T *__restrict__ dst_ind_base = reinterpret_cast<T *>(info.dst_ind_addr);
  T *__restrict__ src_ind_base = reinterpret_cast<T *>(info.src_ind_addr);

  Offset_t chunks = info.field_size / sizeof(DT);

  for(; offset < info.volume; offset += blockDim.x * gridDim.x) {
    Offset_t src_index = offset;
    if(info.src_ind_addr != 0) {
      Offset_t index = 0;
#pragma unroll
      for(int i = 0; i < N; i++) {
        index += src_ind_base[offset * N + i] * info.src_strides[i];
      }
      src_index = index;
    }

    Offset_t dst_index = offset;
    if(info.dst_ind_addr != 0) {
      Offset_t index = 0;
#pragma unroll
      for(int i = 0; i < N; i++) {
        index += dst_ind_base[offset * N + i] * info.dst_strides[i];
      }

      dst_index = index;
    }

    DT *__restrict__ dst =
        reinterpret_cast<DT *>(info.dst_addr + dst_index * info.field_size);
    DT *__restrict__ src =
        reinterpret_cast<DT *>(info.src_addr + src_index * info.field_size);
    for(Offset_t chunk_idx = 0; chunk_idx < chunks; chunk_idx++) {
      dst[chunk_idx] = src[chunk_idx];
    }
  }
}

template <int N, typename T, typename Offset_t = size_t>
static __device__ inline void
memfill_affine_batch(const Realm::Hip::AffineFillInfo<N, Offset_t>& info)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned grid_stride = gridDim.x * blockDim.x;
  T fill_value = *reinterpret_cast<const T *>(info.fill_value);
  for(size_t rect = 0; rect < info.num_rects; rect++) {
    const Realm::Hip::AffineFillRect<N, Offset_t> &current_info = info.subrects[rect];
    const Offset_t vol = current_info.volume;
    T* __restrict__ addr = reinterpret_cast<T *>(current_info.addr);
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

// memcpy_affine_batch* copy within GPU memory (one GPU, between GPUs, or to another
// process's GPU memory), memcpy_affine_host_batch* read or write host memory; on gfx9
// they differ only in the vector registers they allocate (see above).
#if defined(__GFX9__)
#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_batch##name(        \
      Realm::Hip::AffineCopyInfo<dim, offt> info)                                        \
  {                                                                                      \
    memcpy_affine_tiles<sizeof(type), dim, offt, HIP_COPY_KERNEL_VGPRS>(info);           \
  }                                                                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_host_batch##name(   \
      Realm::Hip::AffineCopyInfo<dim, offt> info)                                        \
  {                                                                                      \
    memcpy_affine_tiles<sizeof(type), dim, offt, HIP_HOST_COPY_KERNEL_VGPRS>(info);      \
  }
#else
#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_batch##name(        \
      Realm::Hip::AffineCopyInfo<dim, offt> info)                                        \
  {                                                                                      \
    memcpy_affine_batch<type, dim, offt>(info.subrects, info.num_rects);                 \
  }                                                                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_host_batch##name(   \
      Realm::Hip::AffineCopyInfo<dim, offt> info)                                        \
  {                                                                                      \
    memcpy_affine_batch<type, dim, offt>(info.subrects, info.num_rects);                 \
  }
#endif

#define FILL_TEMPLATE_INST(type, dim, offt, name)                                        \
  extern "C" __global__ void fill_affine_batch##name(                                    \
      Realm::Hip::AffineFillInfo<dim, offt> info)                                        \
  {                                                                                      \
    memfill_affine_batch<dim, type, offt>(info);                                         \
  }

#define FILL_LARGE_TEMPLATE_INST(type, dim, offt, name)                                  \
  extern "C" __global__ void fill_affine_large##name(                                    \
      Realm::Hip::AffineLargeFillInfo<dim, offt> info)                                   \
  {}

#define MEMCPY_TRANSPOSE_TEMPLATE_INST(type, offt, name)                                 \
  extern "C" __global__ __launch_bounds__(1024) void memcpy_transpose##name(             \
      Realm::Hip::MemcpyTransposeInfo<offt> info)                                        \
  {                                                                                      \
    extern __shared__ type tile_shared_##name[];                                         \
    memcpy_kernel_transpose<type, offt>(info, tile_shared_##name);                       \
  }

#define MEMCPY_INDIRECT_TEMPLATE_INST(addr_type, data_type, dim, offt, name)             \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_indirect##name(            \
      Realm::Hip::MemcpyIndirectInfo<3, offt> info)                                     \
  {                                                                                      \
    memcpy_indirect_points<dim, addr_type, data_type, offt>(info);                       \
  }

#define INST_TEMPLATES(type, sz, dim, off)                                               \
  MEMCPY_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                      \
  FILL_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                        \
  FILL_LARGE_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                  \
  MEMCPY_INDIRECT_TEMPLATE_INST(int, type, dim, off, dim##D_##sz##32)                    \
  MEMCPY_INDIRECT_TEMPLATE_INST(long long, type, dim, off, dim##D_##sz##64)

#define INST_TEMPLATES_FOR_TYPES(dim, off)                                     \
  INST_TEMPLATES(unsigned char, 8, dim, off)                                   \
  INST_TEMPLATES(unsigned short, 16, dim, off)                                 \
  INST_TEMPLATES(unsigned int, 32, dim, off)                                   \
  INST_TEMPLATES(unsigned long long, 64, dim, off)                             \
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
