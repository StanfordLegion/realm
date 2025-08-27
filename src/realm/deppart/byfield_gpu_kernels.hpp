#pragma once
#include "realm/deppart/byfield.h"

namespace Realm {

// Intersect all instance rectangles with all parent rectangles in parallel.
// Used for both count and emit depending on whether the output array is null.

template <int N, typename T>
__global__ void intersect_input_rects(
  const SparsityMapEntry<N,T>* d_inst_entries,
  const SparsityMapEntry<N,T>* d_parent_entries,
  const size_t *d_inst_offsets,
  const uint32_t *d_inst_prefix,
  size_t numInstRects,
  size_t numParentRects,
  size_t numInsts,
  uint32_t *d_inst_counters,
  Rect<N,T>* d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numInstRects * numParentRects) return;
  size_t idx_x = idx % numParentRects;
  size_t idx_y = idx / numParentRects;
  assert(idx_x < numParentRects);
  assert(idx_y < numInstRects);
  const SparsityMapEntry<N, T> source_entry = d_parent_entries[idx_x];
  const SparsityMapEntry<N, T> inst_entry = d_inst_entries[idx_y];
  Rect<N,T> rect_output = inst_entry.bounds.intersection(source_entry.bounds);
  if (rect_output.empty()) {
    return;
  }
  size_t low = 0, high = numInsts;
  while (low < high) {
      size_t mid = (low + high) >> 1;
      if (d_inst_offsets[mid+1] <= idx_y) low = mid + 1;
      else                                 high = mid;
  }
  size_t inst_idx = low;
  uint32_t local = atomicAdd(&d_inst_counters[inst_idx], 1);
  if (d_rects != nullptr) {
    // If d_rects is not null, we write the output rect
    uint32_t out_idx = d_inst_prefix[inst_idx] + local;
    d_rects[out_idx] = rect_output;
  }
}


template <
  int N, typename T, typename FT
>
__global__
void byfield_gpuPopulateBitmasksKernel(
  AffineAccessor<FT,N,T>* accessors,
  Rect<N,T>* rects,
  size_t* prefix,
  uint32_t* inst_prefix,
  FT* d_colors,
  size_t numPoints,
  size_t numColors,
  size_t numRects,
  size_t num_insts,
  PointDesc<N,T> *d_points
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPoints) return;

  // Binary search to find which rectangle this point belongs to
  size_t low = 0, high = numRects;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (prefix[mid+1] <= idx) low = mid + 1;
    else                      high = mid;
  }
  size_t r = low;

  // Binary search to find which instance this rectangle belongs to
  low = 0, high = num_insts;
  while (low < high) {
      size_t mid = (low + high) >> 1;
      if (inst_prefix[mid+1] <= r) low = mid + 1;
      else                         high = mid;
  }
  size_t inst_idx = low;

  // Now we know which rectangle we're in, figure out the point coordinates
  size_t offset = idx - prefix[r];
  Point<N, T> p;
  for (int k = N-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }

  // Read the field value at that point
  FT ptr = accessors[inst_idx].read(p);

  // Find our color's idx and write output
  PointDesc<N,T> point_desc;
  point_desc.point = p;
  for (size_t i = 0; i < numColors; ++i) {
    if (ptr == d_colors[i]) {
      point_desc.src_idx = i;
      break;
    }
  }
  d_points[idx] = point_desc;
}

} // namespace Realm