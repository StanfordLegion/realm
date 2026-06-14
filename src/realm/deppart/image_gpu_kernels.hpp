#pragma once
#include "realm/deppart/image.h"

namespace Realm {

//Device helper to check parent space for membership
//TODO: if expensive, may benefit from BVH
template<int N, typename T>
__device__ bool image_isInIndexSpace(
    const Point<N,T>& p,
    const Rect<N,T>*  parent_entries,
    size_t              numRects)
{
  // for each rectangle, check all dims…
  for(size_t i = 0; i < numRects; ++i) {
    const auto &r = parent_entries[i];
    bool inside = true;
    #pragma unroll
    for(int d = 0; d < N; ++d) {
      if(p[d] < r.lo[d] || p[d] > r.hi[d]) {
        inside = false;
        break;
      }
    }
    if(inside) return true;
  }
  return false;
}

//Count + emit to chase pointers and check for membership in parent space
template <
  int N, typename T,
  int N2, typename T2
>
__global__
void image_gpuPopulateBitmasksPtrsKernel(
  AffineAccessor<Point<N,T>,N2,T2> *accessors,
  RectDesc<N2,T2>* rects,
  Rect<N,T>* parent_entries,
  size_t* prefix,
  uint32_t *inst_offsets,
  uint32_t *d_inst_prefix,
  size_t numPoints,
  size_t numRects,
  size_t num_insts,
  size_t numParentRects,
  uint32_t* d_inst_counters,
  PointDesc<N,T> *d_points
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;
  size_t low = 0, high = numRects;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (prefix[mid+1] <= idx) low = mid + 1;
    else                      high = mid;
  }
  size_t r = low;
  bool found = false;
  size_t inst_idx;
  for (inst_idx = 0; inst_idx < num_insts; ++inst_idx) {
    if (inst_offsets[inst_idx] <= r && inst_offsets[inst_idx+1] > r) {
      found = true;
      break;
    }
  }
  assert(found);
  size_t offset = idx - prefix[r];
  Point<N2, T2> p;
  for (int k = N2-1; k >= 0; --k) {
    size_t dim = rects[r].rect.hi[k] + 1 - rects[r].rect.lo[k];
    p[k]  = rects[r].rect.lo[k] + (offset % dim);
    offset /= dim;
  }
  Point<N,T> ptr = accessors[inst_idx].read(p);
  if (image_isInIndexSpace<N,T>(ptr, parent_entries, numParentRects)) {
    uint32_t local = atomicAdd(&d_inst_counters[inst_idx], 1);
    if (d_points != nullptr) {
      uint32_t out_idx = d_inst_prefix[inst_idx] + local;
      PointDesc<N,T> point_desc;
      point_desc.src_idx = rects[r].src_idx;
      point_desc.point = ptr;
      d_points[out_idx] = point_desc;
    }
  }
  
}

template<int N, typename T>
__device__ bool image_rectOverlapsIndexSpace(
    const Rect<N,T>& r,
    const Rect<N,T>* parent_entries,
    size_t numRects)
{
  for(size_t i = 0; i < numRects; ++i) {
    if(r.overlaps(parent_entries[i])) return true;
  }
  return false;
}

template <
  int N, typename T,
  int N2, typename T2
>
__global__
void image_gpuApproxPtrsKernel(
  AffineAccessor<Point<N,T>,N2,T2> accessor,
  Rect<N2,T2>* rects,
  Rect<N,T>* parent_entries,
  size_t* prefix,
  size_t numPoints,
  size_t numRects,
  size_t numParentRects,
  uint32_t* d_counter,
  PointDesc<N,T> *d_points
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;
  size_t low = 0, high = numRects;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (prefix[mid+1] <= idx) low = mid + 1;
    else                      high = mid;
  }
  size_t r = low;
  size_t offset = idx - prefix[r];
  Point<N2, T2> p;
  for (int k = N2-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }
  Point<N,T> ptr = accessor.read(p);
  if (image_isInIndexSpace<N,T>(ptr, parent_entries, numParentRects)) {
    uint32_t local = atomicAdd(d_counter, 1);
    if (d_points != nullptr) {
      PointDesc<N,T> point_desc;
      point_desc.src_idx = 0;
      point_desc.point = ptr;
      d_points[local] = point_desc;
    }
  }
}

template <
  int N, typename T,
  int N2, typename T2
>
__global__
void image_gpuApproxRngsKernel(
  AffineAccessor<Rect<N,T>,N2,T2> accessor,
  Rect<N2,T2>* rects,
  Rect<N,T>* parent_entries,
  size_t* prefix,
  size_t numPoints,
  size_t numRects,
  size_t numParentRects,
  uint32_t* d_counter,
  RectDesc<N,T> *d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;
  size_t low = 0, high = numRects;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (prefix[mid+1] <= idx) low = mid + 1;
    else                      high = mid;
  }
  size_t r = low;
  size_t offset = idx - prefix[r];
  Point<N2, T2> p;
  for (int k = N2-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }
  Rect<N,T> rng = accessor.read(p);
  if (image_rectOverlapsIndexSpace<N,T>(rng, parent_entries, numParentRects)) {
    uint32_t local = atomicAdd(d_counter, 1);
    if (d_rects != nullptr) {
      RectDesc<N,T> rect_desc;
      rect_desc.src_idx = 0;
      rect_desc.rect = rng;
      d_rects[local] = rect_desc;
    }
  }
}

template<int N, typename T>
__global__
void image_buildGapKeys1d(const RectDesc<N,T> *rects,
                          T *gap_keys,
                          size_t *gap_indices,
                          size_t num_gaps)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= num_gaps) return;
  gap_keys[idx] = rects[idx + 1].rect.lo[0] - rects[idx].rect.hi[0];
  gap_indices[idx] = idx + 1;
}

template<typename T>
__global__
void image_copyTopGapIndices(const T *sorted_gap_indices,
                             T *kept_gap_indices,
                             T num_gaps,
                             T num_kept)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= num_kept) return;
  kept_gap_indices[idx] = sorted_gap_indices[num_gaps - num_kept + idx];
}

template<int N, typename T>
__global__
void image_emitApproxRects1d(const RectDesc<N,T> *rects,
                             const size_t *kept_gap_indices,
                             size_t num_kept,
                             size_t num_rects,
                             RectDesc<N,T> *out)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx > num_kept) return;

  out[idx].src_idx = 0;
  if(idx == 0) {
    out[idx].rect.lo = rects[0].rect.lo;
  } else {
    out[idx].rect.lo = rects[kept_gap_indices[idx - 1]].rect.lo;
  }

  if(idx == num_kept) {
    out[idx].rect.hi = rects[num_rects - 1].rect.hi;
  } else {
    out[idx].rect.hi = rects[kept_gap_indices[idx] - 1].rect.hi;
  }
}

template<int N, typename T>
__global__
void image_rectDescsToRects(const RectDesc<N,T> *in,
                            Rect<N,T> *out,
                            size_t count)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= count) return;
  out[idx] = in[idx].rect;
}

//Same as image_intersect_input, but for output rectangles and parent entries
//rather than input rectangles and parent rectangles
  template <int N, typename T>
__global__ void image_intersect_output(
  const Rect<N,T>* d_parent_entries,
  const RectDesc<N,T>* d_output_rngs,
  const uint32_t* d_src_prefix,
  size_t numParentRects,
  size_t numOutputRects,
  uint32_t* d_src_counters,
  RectDesc<N,T>* d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParentRects * numOutputRects) return;
  size_t idx_x = idx % numParentRects;
  size_t idx_y = idx / numParentRects;
  const auto parent_entry = d_parent_entries[idx_x];
  const auto output_entry = d_output_rngs[idx_y];
  RectDesc<N,T> rect_output;
  rect_output.rect = parent_entry.intersection(output_entry.rect);
  if (!rect_output.rect.empty()) {
    uint32_t local = atomicAdd(&d_src_counters[output_entry.src_idx], 1);
    if (d_rects != nullptr) {
      rect_output.src_idx = output_entry.src_idx;
      size_t out_idx = d_src_prefix[output_entry.src_idx] + local;
      d_rects[out_idx] = rect_output;
    }
  }
}

//Single pass function to chase pointers to rectangles.
  template <
  int N, typename T,
  int N2, typename T2
>
__global__
void image_gpuPopulateBitmasksRngsKernel(
  AffineAccessor<Rect<N,T>,N2,T2> *accessors,
  RectDesc<N2,T2>* rects,
  size_t* prefix,
  uint32_t *inst_offsets,
  size_t numPoints,
  size_t numRects,
  size_t num_insts,
  RectDesc<N,T> *d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;
  size_t low = 0, high = numRects;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (prefix[mid+1] <= idx) low = mid + 1;
    else                      high = mid;
  }
  size_t r = low;
  bool found = false;
  size_t inst_idx;
  for (inst_idx = 0; inst_idx < num_insts; ++inst_idx) {
    if (inst_offsets[inst_idx] <= r && inst_offsets[inst_idx+1] > r) {
      found = true;
      break;
    }
  }
  assert(found);
  size_t offset = idx - prefix[r];
  Point<N2, T2> p;
  for (int k = N2-1; k >= 0; --k) {
    size_t dim = rects[r].rect.hi[k] + 1 - rects[r].rect.lo[k];
    p[k]  = rects[r].rect.lo[k] + (offset % dim);
    offset /= dim;
  }
  Rect<N,T> rng = accessors[inst_idx].read(p);
  RectDesc<N,T> rect_desc;
  rect_desc.src_idx = rects[r].src_idx;
  rect_desc.rect = rng;
  d_rects[idx] = rect_desc;
}

} // namespace Realm
