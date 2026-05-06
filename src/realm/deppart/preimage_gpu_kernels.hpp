#pragma once
#include "realm/deppart/preimage.h"

namespace Realm {


template <int N, typename T>
__global__ void preimage_build_morton_codes(
  const Rect<N,T>* d_targets_entries,
  const size_t* d_offsets_rects,
  const Rect<N,T>* d_global_bounds,
  size_t total_rects,
  size_t num_targets,
  uint64_t* d_morton_codes,
  uint64_t* d_indices,
  uint64_t* d_targets_indices) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_rects) return;
  const auto &entry = d_targets_entries[idx];
  d_morton_codes[idx] = bvh_morton_code(entry, *d_global_bounds);
  d_indices[idx] = idx;
  size_t low = 0, high = num_targets;
    while (low < high) {
    size_t mid = (low + high) >> 1;
    if (d_offsets_rects[mid+1] <= idx) low = mid + 1;
    else                                 high = mid;
  }
  d_targets_indices[idx] = low;
}

//
// 2) Initialize leaf boxes
//
template<int N, typename T>
__global__
void preimage_init_leaf_boxes_kernel(
    const Rect<N,T> *rects,    // [G] all flattened Rects
    const uint64_t    *leafIdx, // [n] maps leaf→orig Rect index
    size_t total_rects,
    Rect<N,T> *boxes)                 // [(2n−1)]
{
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  if (k >= total_rects) return;

  size_t orig = leafIdx[k];
  boxes[k + total_rects - 1] = rects[orig];
}

  template<int N, typename T, int N2, typename T2, typename Q>
__device__ void preimage_queryBVH(
    const Rect<N2,T2> *boxes,
    const int*      childLeft,
    const int*      childRight,
    const uint64_t* leafIdx,
    const size_t*   targets_indices,
    int            root,
    size_t          numTargetRects,
    const Q&        in_query,
    Point<N, T> out_point,
    uint32_t* d_targets_prefix,
    uint32_t* d_target_counters,
    PointDesc<N,T> *d_points)
{
  constexpr int MAX_STACK = 64; // max stack size for BVH traversal
  int stack[MAX_STACK];
  int sp = 0;

  // start at the root
  stack[sp++] = -1;
  int node = root;
  do
  {

    int left = childLeft[node];
    int right = childRight[node];

    bool overlapL;
    bool overlapR;

    if constexpr (std::is_same_v<Q, Rect<N2,T2>>) {
      overlapL = boxes[left].overlaps(in_query);
      overlapR = boxes[right].overlaps(in_query);
    } else {
      static_assert(std::is_same_v<Q, Point<N2,T2>>,
                    "Q must be Rect<N2,T2> or Point<N2,T2>");
      overlapL = boxes[left].contains(in_query);
      overlapR = boxes[right].contains(in_query);
    }


    if (overlapL && left >= numTargetRects - 1) {
      // left child is a leaf
      uint64_t rect_idx = leafIdx[left - (numTargetRects - 1)];
      size_t target_idx = targets_indices[rect_idx];
      uint32_t local = atomicAdd(&d_target_counters[target_idx], 1);
      if (d_points != nullptr) {
        PointDesc<N,T> point_desc;
        point_desc.src_idx = target_idx;
        point_desc.point = out_point;
        uint32_t out_idx = d_targets_prefix[target_idx] + local;
        d_points[out_idx] = point_desc;
      }
    }
    if (overlapR && right >= numTargetRects - 1) {
      uint64_t rect_idx = leafIdx[right - (numTargetRects - 1)];
      size_t target_idx = targets_indices[rect_idx];
      uint32_t local = atomicAdd(&d_target_counters[target_idx], 1);
      if (d_points != nullptr) {
        PointDesc<N,T> point_desc;
        point_desc.src_idx = target_idx;
        point_desc.point = out_point;
        uint32_t out_idx = d_targets_prefix[target_idx] + local;
        d_points[out_idx] = point_desc;
      }
    }

    bool traverseL = overlapL && left < numTargetRects - 1;
    bool traverseR = overlapR && right < numTargetRects - 1;

    if (!traverseL && !traverseR) {
      node = stack[--sp];
    } else {
      node = (traverseL ? left : right);
      if (traverseL && traverseR) {
        stack[sp++] = right;
      }
    }
  } while (node != -1);
}

template <
  int N, typename T,
  int N2, typename T2, typename Q
>
__global__
void preimage_gpuPopulateBitmasksPtrsKernel(
  AffineAccessor<Q,N,T> *accessors,
  Rect<N,T>* rects,
  size_t* prefix,
  uint32_t* inst_offsets,
  int root,
  int *childLeft,
  int *childRight,
  uint64_t *indices,
  uint64_t *targets_indices,
  Rect<N2,T2> *boxes,
  size_t numPoints,
  size_t numRects,
  size_t numInsts,
  size_t numTargetRects,
  uint32_t* d_targets_prefix,
  uint32_t* d_target_counters,
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
  low = 0, high = numInsts;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (inst_offsets[mid+1] <= r) low = mid + 1;
    else                          high = mid;
  }
  size_t inst_idx = low;
  size_t offset = idx - prefix[r];
  Point<N, T> p;
  for (int k = N-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }
  Q ptr = accessors[inst_idx].read(p);
  preimage_queryBVH(boxes, childLeft, childRight, indices, targets_indices, root, numTargetRects, ptr, p, d_targets_prefix, d_target_counters, d_points);
}

template <
  int N, typename T,
  int N2, typename T2, typename Q
>
__global__
void preimage_dense_populate_bitmasks_kernel(
  AffineAccessor<Q,N,T>* accessors,
  Rect<N,T>* rects,
  size_t* prefix,
  uint32_t* inst_offsets,
  Rect<N2,T2>* targets_entries,
  size_t* target_offsets,
  size_t numPoints,
  size_t numRects,
  size_t numInsts,
  size_t numTargets,
  uint32_t *d_targets_prefix,
  uint32_t *d_target_counters,
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
  low = 0, high = numInsts;
  while (low < high) {
    size_t mid = (low + high) >> 1;
    if (inst_offsets[mid+1] <= r) low = mid + 1;
    else                          high = mid;
  }
  size_t inst_idx = low;
  size_t offset = idx - prefix[r];
  Point<N, T> p;
  for (int k = N-1; k >= 0; --k) {
    size_t dim = rects[r].hi[k] + 1 - rects[r].lo[k];
    p[k]  = rects[r].lo[k] + (offset % dim);
    offset /= dim;
  }
  Q ptr = accessors[inst_idx].read(p);
  for (size_t i = 0; i < numTargets; i++) {
    bool inside = false;
    for (size_t j = target_offsets[i]; j < target_offsets[i+1]; j++) {
      if constexpr (std::is_same_v<Q, Rect<N2,T2>>) {
        if (targets_entries[j].overlaps(ptr)) {
          inside = true;
          break;
        }
      } else {
        static_assert(std::is_same_v<Q, Point<N2,T2>>,
                      "Q must be Rect<N2,T2> or Point<N2,T2>");
        if (targets_entries[j].contains(ptr)) {
          inside = true;
          break;
        }
      }
    }
    if (inside) {
      uint32_t local = atomicAdd(&d_target_counters[i], 1);
      if (d_points != nullptr) {
        PointDesc<N,T> point_desc;
        point_desc.src_idx = i;
        point_desc.point = p;
        uint32_t out_idx = d_targets_prefix[i] + local;
        d_points[out_idx] = point_desc;
      }
    }
  }
}

} // namespace Realm
