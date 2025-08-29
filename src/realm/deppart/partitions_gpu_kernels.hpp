#pragma once
#include "realm/deppart/partitions.h"

namespace Realm {

template<int N, typename T>
__global__ void build_coord_key(T*        d_keys,
                                const PointDesc<N,T>* d_pts,
                                size_t            M,
                                int               dim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_pts[i].point[dim];
}

template<int N, typename T>
__global__ void build_src_key(size_t*        d_keys,
                              const PointDesc<N,T>* d_pts,
                              size_t            M) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_pts[i].src_idx;
}

  template<int N, typename T>
  __global__
  void points_to_rects(const PointDesc<N,T>* pts,
                       RectDesc<N,T>*        rects,
                       size_t                M)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= M) return;
  rects[i].src_idx = pts[i].src_idx;
  rects[i].rect.lo     = pts[i].point;
  rects[i].rect.hi     = pts[i].point;
}

// 1) mark breaks on RectDesc array at pass d
//NOTE: ONLY WORKS IF WE STARTED WITH SINGLETONS
template<int N, typename T>
__global__
void mark_breaks_dim(const RectDesc<N,T>* in,
                       uint8_t*              brk,
                       size_t                M,
                       int                   d)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= M) return;
  if(i == 0) { brk[0] = 1; return; }

  const auto &p = in[i].rect, &q = in[i-1].rect;
  bool split = (in[i].src_idx != in[i-1].src_idx);

  // more‐significant dims 0..d-1 must match lo
#pragma unroll
  for(int k = 0; k < d && !split; ++k)
    if(p.lo[k] != q.lo[k] || p.hi[k] != q.hi[k]) split = true;

  // already‐processed dims d+1..N-1 must match [lo,hi]
#pragma unroll
  for(int k = d+1; k < N && !split; ++k)
    if((p.lo[k] != q.lo[k]) || (p.hi[k] != q.hi[k]))
      split = true;

  // current dim d must equal or advance by +1 in lo
  if(!split && (p.lo[d] != (q.hi[d] + 1)) && (p.lo[d] != q.lo[d]))
    split = true;

  brk[i] = split ? 1 : 0;
}

// Write output rectangles for RLE
//Starts write lo, ends write hi, everyone else no-ops
template<int N, typename T>
__global__
void init_rects_dim(const RectDesc<N,T>* in,
                    const uint8_t*        brk,
                    const size_t*         gid,
                    RectDesc<N,T>*        out,
                    size_t                M,
                    int                   d)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= M) return;

  bool is_end = (i == M-1) || (gid[i+1] != gid[i]);
  if (!brk[i] && !is_end) return;

  size_t g = gid[i] - 1;  // zero-based rectangle index
  const Rect<N, T> &r = in[i].rect;
  out[g].src_idx = in[i].src_idx;

  #pragma unroll
  for(int k = 0; k < N; ++k) {
    if (brk[i]) {
        out[g].rect.lo[k] = r.lo[k];
    }
    if (is_end) {
        out[g].rect.hi[k] = r.hi[k];
    }
  }
}

//Convert RectDesc to sparsity output and determine [d_start[i], d_end[i]) for each src i
template<int N, typename T>
__global__
void build_final_output(const RectDesc<N,T>* d_rects,
                              SparsityMapEntry<N,T>* d_entries_out,
                              Rect<N,T>* d_rects_out,
                              size_t* d_starts,
                              size_t* d_ends,
                              size_t numRects) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRects) return;
  d_rects_out[idx] = d_rects[idx].rect;
  d_entries_out[idx].bounds = d_rects[idx].rect;
  d_entries_out[idx].sparsity.id = 0;
  d_entries_out[idx].bitmap = 0;

  //Checks if we're the first value for a given src
  if (idx == 0 || d_rects[idx].src_idx != d_rects[idx-1].src_idx) {
    d_starts[d_rects[idx].src_idx] = idx;
  }

  //Checks if we're the last value for a given src
  if (idx == numRects-1 || d_rects[idx].src_idx != d_rects[idx+1].src_idx) {
    d_ends[d_rects[idx].src_idx] = idx+1;
  }
}

}