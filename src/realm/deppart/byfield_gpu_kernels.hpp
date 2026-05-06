#pragma once
#include "realm/deppart/byfield.h"
#include "realm/deppart/partitions_gpu_kernels.hpp"

namespace Realm {


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

  // Binary search to find which rectangle this point belongs to.
  uint32_t r = bsearch(prefix, numRects, idx);

  // Binary search to find which instance this rectangle belongs to.
  size_t inst_idx = bsearch(inst_prefix, num_insts, r);

  // Now we know which rectangle we're in, figure out the point coordinates.
  size_t offset = idx - prefix[r];
  Point<N, T> p;
  for (int k = N-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }

  // Read the field value at that point.
  FT ptr = accessors[inst_idx].read(p);

  // Find our color's idx and write output.
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