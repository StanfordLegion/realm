#pragma once
#include "realm/deppart/partitions.h"

namespace Realm {

template <typename T>
__device__ __forceinline__ size_t bsearch(const T* arr, size_t len, T val) {
  size_t low = 0, high = len;
  while (low < high) {
    size_t mid = low + ((high - low) >> 1);
    if (arr[mid + 1] <= val)
      low = mid + 1;
    else
      high = mid;
  }
  return low;
}

// Intersect all instance rectangles with all parent rectangles in parallel.
// Used for both count and emit depending on whether the output array is null.

template <int N, typename T, typename out_t>
__global__ void intersect_input_rects(
  const SparsityMapEntry<N,T>* d_lhs_entries,
  const SparsityMapEntry<N,T>* d_rhs_entries,
  const size_t *d_lhs_offsets,
  const uint32_t *d_lhs_prefix,
  const size_t* d_rhs_offsets,
  size_t numLHSRects,
  size_t numRHSRects,
  size_t numLHSChildren,
  size_t numRHSChildren,
  uint32_t *d_lhs_counters,
  out_t* d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numLHSRects * numRHSRects) return;
  size_t idx_x = idx % numRHSRects;
  size_t idx_y = idx / numRHSRects;
  assert(idx_x < numRHSRects);
  assert(idx_y < numLHSRects);
  const SparsityMapEntry<N, T> rhs_entry = d_rhs_entries[idx_x];
  const SparsityMapEntry<N, T> lhs_entry = d_lhs_entries[idx_y];
  Rect<N,T> rect_output = lhs_entry.bounds.intersection(rhs_entry.bounds);
  if (rect_output.empty()) {
    return;
  }
  size_t lhs_idx = bsearch(d_lhs_offsets, numLHSRects, idx_y);
  uint32_t local = atomicAdd(&d_lhs_counters[lhs_idx], 1);
  if (d_rects != nullptr) {
    // If d_rects is not null, we write the output rect
    uint32_t out_idx = d_lhs_prefix[lhs_idx] + local;
    if constexpr (std::is_same_v<out_t, RectDesc<N, T>>) {
      d_rects[out_idx].src_idx = bsearch(d_rhs_offsets, numRHSChildren, idx_x);
      d_rects[out_idx].rect = rect_output;
    } else {
      d_rects[out_idx] = rect_output;
    }
  }
}

template <int N, typename T>
__device__ __forceinline__ uint64_t bvh_morton_code(const Rect<N,T>& rect,
                            const Rect<N,T>& globalBounds) {
  // bits per axis (floor)
  constexpr int bits     = 64 / N;
  constexpr uint64_t maxQ = (bits == 64 ? ~0ULL
                                       : (1ULL << bits) - 1);

  uint64_t coords[N];
#pragma unroll
  for(int d = 0; d < N; ++d) {
    // 1) compute centroid in dimension d
    float center = 0.5f * (float(rect.lo[d]) + float(rect.hi[d]) + 1.0f);

    // 2) normalize into [0,1] using globalBounds
    float span = float(globalBounds.hi[d] + 1 - globalBounds.lo[d]);
    float norm = (center - float(globalBounds.lo[d])) / span;

    // 3) quantize to [0 … maxQ]
    uint64_t q = uint64_t(norm * float(maxQ) + 0.5f);
    coords[d] = (q > maxQ ? maxQ : q);
  }

  // 4) interleave bits MSB→LSB across all dims
  uint64_t code = 0;
  for(int b = bits - 1; b >= 0; --b) {
#pragma unroll
    for(int d = 0; d < N; ++d) {
      code = (code << 1) | ((coords[d] >> b) & 1ULL);
    }
  }

  return code;
}

template <int N, typename T>
__global__ void bvh_build_morton_codes(
  const SparsityMapEntry<N,T>* d_targets_entries,
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
  d_morton_codes[idx] = bvh_morton_code(entry.bounds, *d_global_bounds);
  d_indices[idx] = idx;
  if (d_offsets_rects != nullptr) {
    d_targets_indices[idx] = bsearch(d_offsets_rects, num_targets, idx);
  }
}

  __global__
void bvh_build_radix_tree_kernel(
    const uint64_t *morton,    // [n]
    const uint64_t *leafIdx,   // [n]  (unused here but kept for symmetry)
    int n,
    int *childLeft,            // [2n−1]
    int *childRight,           // [2n−1]
    int *parent);               // [2n−1], pre‐initialized to −1

__global__
void bvh_build_root_kernel(
    int *root,
    int *parent,
    size_t total_rects);

template<int N, typename T>
__global__
void bvh_init_leaf_boxes_kernel(
    const SparsityMapEntry<N,T> *rects,    // [G] all flattened Rects
    const uint64_t    *leafIdx, // [n] maps leaf→orig Rect index
    size_t total_rects,
    Rect<N,T> *boxes)                 // [(2n−1)]
{
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  if (k >= total_rects) return;

  size_t orig = leafIdx[k];
  boxes[k + total_rects - 1] = rects[orig].bounds;
}

template<int N, typename T>
__global__
void bvh_merge_internal_boxes_kernel(
    size_t total_rects,
    const int *childLeft,      // [(2n−1)]
    const int *childRight,     // [(2n−1)]
    const int *parent,         // [(2n−1)]
    Rect<N,T> *boxes,                 // [(2n−1)×N]
    int *visitCount)           // [(2n−1)] initialized to zero
{
  int leaf = blockIdx.x*blockDim.x + threadIdx.x;
  if (leaf >= total_rects) return;

  int cur = leaf + total_rects - 1;
  int p   = parent[cur];

  while(p >= 0) {
    // increment visit count; the second arrival merges
    int prev = atomicAdd(&visitCount[p], 1);
    if (prev == 1) {
      // both children ready, do the merge
      int c0 = childLeft[p], c1 = childRight[p];
      boxes[p] = boxes[c0].union_bbox(boxes[c1]);
      // climb
      cur = p;
      p   = parent[cur];
    } else {
      // first child arrived, wait for sibling
      break;
    }
  }
}

template <int N, typename T, typename out_t>
__global__
void query_input_bvh(
  SparsityMapEntry<N, T>* queries,
  size_t* d_query_offsets,
  int root,
  int *childLeft,
  int *childRight,
  uint64_t *indices,
  uint64_t *labels,
  Rect<N,T> *boxes,
  size_t numQueries,
  size_t numBoxes,
  size_t numLHSChildren,
  uint32_t* d_inst_prefix,
  uint32_t* d_inst_counters,
  out_t *d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numQueries) return;
  Rect<N, T> in_rect = queries[idx].bounds;
  size_t lhs_idx = bsearch(d_query_offsets, numLHSChildren, idx);

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

    bool overlapL = boxes[left].overlaps(in_rect);
    bool overlapR = boxes[right].overlaps(in_rect);

    if (overlapL && left >= numBoxes - 1) {
      uint64_t rect_idx = indices[left - (numBoxes - 1)];
      uint32_t local = atomicAdd(&d_inst_counters[lhs_idx], 1);
      if (d_rects != nullptr) {
        uint32_t out_idx = d_inst_prefix[lhs_idx] + local;
        Rect<N, T> out_rect = boxes[left].intersection(in_rect);
        if constexpr (std::is_same_v<out_t, RectDesc<N, T>>) {
          d_rects[out_idx].rect = out_rect;
          d_rects[out_idx].src_idx = labels[rect_idx];
        } else {
          d_rects[out_idx] = out_rect;
        }
      }
    }
    if (overlapR && right >= numBoxes - 1) {
      uint64_t rect_idx = indices[right - (numBoxes - 1)];
      uint32_t local = atomicAdd(&d_inst_counters[lhs_idx], 1);
      if (d_rects != nullptr) {
        uint32_t out_idx = d_inst_prefix[lhs_idx] + local;
        Rect<N, T> out_rect = boxes[right].intersection(in_rect);
        if constexpr (std::is_same_v<out_t, RectDesc<N, T>>) {
          d_rects[out_idx].rect = out_rect;
          d_rects[out_idx].src_idx = labels[rect_idx];
        } else {
          d_rects[out_idx] = out_rect;
        }
      }
    }

    bool traverseL = overlapL && left < numBoxes - 1;
    bool traverseR = overlapR && right < numBoxes - 1;

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