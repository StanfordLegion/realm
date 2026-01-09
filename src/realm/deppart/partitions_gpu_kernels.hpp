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

template<typename T>
__global__ void subtract_const(
  T* d_data,
  size_t num_elems,
  T value
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elems) return;
  d_data[idx] = d_data[idx] <= value ? 0 : d_data[idx] - value;
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
  size_t lhs_idx = bsearch(d_lhs_offsets, numLHSChildren, idx_y);
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
struct CornerDesc {
    uint32_t src_idx;
    T        coord[N];
    int32_t  delta;

    // Equality for ReduceByKey: compare key fields only (src_idx, coords)
    __host__ __device__ __forceinline__
    bool operator==(const CornerDesc& rhs) const {
      if (src_idx != rhs.src_idx) return false;
      for (int d = 0; d < N; ++d)
        if (coord[d] != rhs.coord[d]) return false;
      return true;
    }
};

template<int N, typename T>
__global__ void mark_endpoints(const RectDesc<N,T>* d_rects,
                                size_t            M,
                                int               dim,
                                uint32_t*       d_src_keys,
                                T*       d_crd_keys) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  d_src_keys[2*i] = d_rects[i].src_idx;
  d_src_keys[2*i+1] = d_rects[i].src_idx;
  d_crd_keys[2*i] = d_rects[i].rect.lo[dim];
  d_crd_keys[2*i+1] = d_rects[i].rect.hi[dim] + 1;
}

template<typename T>
__global__ void mark_heads(const uint32_t* d_src_keys,
                                  const T* d_crd_keys,
                                  size_t            M,
                                  uint8_t* d_heads) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  if (i==0) d_heads[0] = 1;
  else {
    d_heads[i] = d_src_keys[i] != d_src_keys[i-1] || d_crd_keys[i] != d_crd_keys[i-1];
  }
}

template<typename T>
__global__ void seg_boundaries(const uint8_t* d_flags,
                              const T* d_exc_sum,
                              size_t            M,
                              size_t *d_starts,
                              size_t *d_ends) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  if (d_flags[i]) {
    d_starts[d_exc_sum[i]-1] = i;
  }
  if (i== M-1 || d_flags[i+1]) {
    d_ends[d_exc_sum[i]-1] = i + 1;
  }
}

template<typename T>
__global__ void scatter_unique(const uint32_t* d_src_keys,
                                const T* d_crd_keys,
                                const size_t* d_output,
                                const uint8_t* d_heads,
                                size_t            M,
                                size_t *d_starts,
                                size_t *d_ends,
                                T* d_boundaries) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  size_t u = d_output[i] - (d_heads[i] ? 0 : 1);
  d_boundaries[u] = d_crd_keys[i];
  if (i == 0 || d_src_keys[i] != d_src_keys[i-1]) {
    d_starts[d_src_keys[i]] = u;
  }
  if (i== M-1 || d_src_keys[i] != d_src_keys[i+1]) {
    d_ends[d_src_keys[i]] = u + 1;
  }
}

template<int N, typename T>
__global__ void mark_deltas_heads(const CornerDesc<N, T>* d_corners,
                                size_t            M,
                                int dim,
                                uint8_t* d_heads,
                                DeltaFlag* d_deltas) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  uint8_t head = 1;
  if (i>0) {
    head = 0;
    for (int j = 0; j < N; j++) {
      if (j== dim) continue;
      if (d_corners[i].coord[j] != d_corners[i-1].coord[j]) {
        head = 1;
        break;
      }
    }
    head = head || d_corners[i].src_idx != d_corners[i-1].src_idx;
  }
  d_heads[i] = head;
  d_deltas[i].delta = d_corners[i].delta;
  d_deltas[i].head = head;
}

// For each segment and each boundary, determine whether to emit a new subsegment
template<int N, typename T>
__global__ void count_segments(const DeltaFlag* d_delta_flags,
                                const size_t *d_segment_starts,
                                const size_t *d_segment_ends,
                                const size_t *d_boundary_starts,
                                const size_t *d_boundary_ends,
                                const CornerDesc<N, T>* d_corners,
                                const T* d_boundaries,
                                size_t num_boundaries,
                                size_t num_segments,
                                int dim,
                                uint32_t *seg_counters) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= num_segments * num_boundaries) return;
  size_t bnd_idx = i % num_boundaries;
  size_t seg_idx = i / num_boundaries;
  int my_src = d_corners[d_segment_starts[seg_idx]].src_idx;

  //No boundaries for this src
  if (d_boundary_starts[my_src]>= d_boundary_ends[my_src]) return;

  //This boundary is not a subsegment start for this segment's src
  if (bnd_idx < d_boundary_starts[my_src] || bnd_idx >= d_boundary_ends[my_src]-1) return;

  //Binary search the segment to find the first subsegment whose start is > boundary
  size_t low = d_segment_starts[seg_idx];
  size_t high = d_segment_ends[seg_idx];
  while (low < high) {
    int mid = (low + high) / 2;
    if (d_corners[mid].coord[dim] <= d_boundaries[bnd_idx]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  //The prefix sum for this boundary within this segment is the delta of the corner just before it (if any)
  int my_delta = (low == d_segment_starts[seg_idx] ? 0 : d_delta_flags[low-1].delta);

  //We emit if it's non-zero, and strengthen the requirement to > 0 for dim 0.
  if (my_delta != 0 && (dim !=0 || my_delta > 0)) {
    atomicAdd(&seg_counters[seg_idx], 1);
  }
}

//Do the same computation as above, but this time emit the actual subsegment
template<int N, typename T>
__global__ void write_segments(const DeltaFlag* d_delta_flags,
                                const size_t *d_segment_starts,
                                const size_t *d_segment_ends,
                                const size_t *d_boundary_starts,
                                const size_t *d_boundary_ends,
                                const CornerDesc<N, T>* d_corners,
                                const T* d_boundaries,
                                const uint32_t *seg_offsets,
                                size_t num_boundaries,
                                size_t num_segments,
                                int dim,
                                uint32_t *seg_counters,
                                CornerDesc<N, T>* d_out_corners) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= num_segments * num_boundaries) return;
  size_t bnd_idx = i % num_boundaries;
  size_t seg_idx = i / num_boundaries;
  int my_src = d_corners[d_segment_starts[seg_idx]].src_idx;
  if (d_boundary_starts[my_src]>= d_boundary_ends[my_src]) return;
  if (bnd_idx < d_boundary_starts[my_src] || bnd_idx >= d_boundary_ends[my_src]-1) return;
  size_t low = d_segment_starts[seg_idx];
  size_t high = d_segment_ends[seg_idx];
  while (low < high) {
    int mid = (low + high) / 2;
    if (d_corners[mid].coord[dim] <= d_boundaries[bnd_idx]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  int my_delta = (low == d_segment_starts[seg_idx] ? 0 : d_delta_flags[low-1].delta);

  //To emit, we keep everything the same except the current dim - set that to the boundary value
  if (my_delta != 0 && (dim !=0 || my_delta > 0)) {
    uint32_t my_idx = seg_offsets[seg_idx] + atomicAdd(&seg_counters[seg_idx], 1);
    CornerDesc<N, T> my_corner = d_corners[low-1];
    my_corner.coord[dim] = d_boundaries[bnd_idx];
    my_corner.delta = my_delta;
    d_out_corners[my_idx] = my_corner;
  }
}

//Again, do the same computation as above, but this time emit the actual rectangle
template<int N, typename T>
__global__ void write_segments(const DeltaFlag* d_delta_flags,
                                const size_t *d_segment_starts,
                                const size_t *d_segment_ends,
                                size_t **d_boundary_starts,
                                size_t **d_boundary_ends,
                                const CornerDesc<N, T>* d_corners,
                                T** d_boundaries,
                                const uint32_t *seg_offsets,
                                size_t num_boundaries,
                                size_t num_segments,
                                uint32_t *seg_counters,
                                RectDesc<N, T>* d_out_rects) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= num_segments * num_boundaries) return;
  size_t bnd_idx = i % num_boundaries;
  size_t seg_idx = i / num_boundaries;
  int my_src = d_corners[d_segment_starts[seg_idx]].src_idx;
  if (d_boundary_starts[0][my_src]>= d_boundary_ends[0][my_src]) return;
  if (bnd_idx < d_boundary_starts[0][my_src] || bnd_idx >= d_boundary_ends[0][my_src]-1) return;

  size_t low = d_segment_starts[seg_idx];
  size_t high = d_segment_ends[seg_idx];
  while (low < high) {
    int mid = (low + high) / 2;
    if (d_corners[mid].coord[0] <= d_boundaries[0][bnd_idx]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  int my_delta = (low == d_segment_starts[seg_idx] ? 0 : d_delta_flags[low-1].delta);
  if (my_delta==0) return;
  int my_corner_idx = low - 1;
  uint32_t my_idx = seg_offsets[seg_idx] + atomicAdd(&seg_counters[seg_idx], 1);
  RectDesc<N, T> my_output;
  my_output.src_idx = my_src;
  my_output.rect.lo[0] = d_boundaries[0][bnd_idx];

  //Remember we marked each boundary as hi+1, so need to revert
  my_output.rect.hi[0] = d_boundaries[0][bnd_idx+1] - 1;

  //For every other dimension, map segment -> rect by finding the two boundaries that surround the segment's corner
  for (int d = 1; d < N; d++) {
    low = d_boundary_starts[d][my_src];
    high = d_boundary_ends[d][my_src];
    while (low < high) {
      int mid = (low + high) / 2;
      if (d_boundaries[d][mid] <= d_corners[my_corner_idx].coord[d]) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    my_output.rect.lo[d] = d_boundaries[d][low-1];
    my_output.rect.hi[d] = d_boundaries[d][low] - 1;
  }
  d_out_rects[my_idx] = my_output;
}

  template<int N, typename T>
  __global__ void populate_corners(const RectDesc<N, T>* __restrict__ d_rects,
                                   size_t M,
                                   CornerDesc<N, T>* __restrict__ d_corners)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;

  const auto& r = d_rects[i];            // assumes r.rect.lo[d], r.rect.hi[d], r.src_idx
  const uint32_t src = r.src_idx;

  const size_t corners_per_rect = size_t(1) << N;
  const size_t base = i * corners_per_rect;

  // emit 2^N corners. Each 1 in the mask -> use hi[d]+1, each 0 -> use lo[d]
  for (unsigned mask = 0; mask < corners_per_rect; ++mask) {
    CornerDesc<N,T> c;
    c.src_idx = src;
    // sign = +1 for even popcount(mask), -1 for odd
    c.delta = (__popc(mask) & 1) ? -1 : +1;

    #pragma unroll
    for (int d = 0; d < N; ++d) {
      const T lo   = r.rect.lo[d];
      const T hip1 = r.rect.hi[d] + T(1);   // half-open (hi+1)
      c.coord[d]   = ( (mask & (1u << d)) ? hip1 : lo );
    }

    d_corners[base + mask] = c;
  }
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
__global__ void build_coord_key(T*        d_keys,
                                const CornerDesc<N,T>* d_corners,
                                size_t            M,
                                int               dim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_corners[i].coord[dim];
}

template<int N, typename T>
__global__ void get_delta(int32_t*        d_deltas,
                                const CornerDesc<N,T>* d_corners,
                                size_t            M) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_deltas[i] = d_corners[i].delta;
}

template<int N, typename T>
__global__ void set_delta(const int32_t*        d_deltas,
                                CornerDesc<N,T>* d_corners,
                                size_t            M) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_corners[i].delta = d_deltas[i];
}


  template<int N, typename T>
__global__ void build_lo_key(T*        d_keys,
                                const RectDesc<N,T>* d_rects,
                                size_t            M,
                                int               dim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_rects[i].rect.lo[dim];
}

  template<int N, typename T>
__global__ void build_hi_key(T*        d_keys,
                                const RectDesc<N,T>* d_rects,
                                size_t            M,
                                int               dim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_rects[i].rect.hi[dim];
}

  template<int N, typename T>
__global__ void build_hi_flag(HiFlag<T>*        d_flags,
                              const RectDesc<N,T>* d_rects,
                              size_t            M,
                              int               dim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= M) return;
  d_flags[i].hi = d_rects[i].rect.hi[dim];
  d_flags[i].head = i==0 || d_rects[i].src_idx != d_rects[i-1].src_idx;
}

  template<int N, typename T>
__global__ void build_src_key(size_t*        d_keys,
                              const RectDesc<N,T>* d_rects,
                              size_t            M) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_rects[i].src_idx;
}

  template<int N, typename T>
__global__ void build_src_key(size_t*        d_keys,
                              const CornerDesc<N, T> *d_corners,
                              size_t            M) {
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < M) d_keys[i] = d_corners[i].src_idx;
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
// Starts a new rectangle if src or lo/hi in any dimension but d doesn't match,
// or if dim d doesn't match or advance by +1
//NOTE: ONLY WORKS IF WE STARTED WITH DISJOINT RECTANGLES
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

  // more‐significant dims 0..d-1 must match [lo,hi]
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

//1) Mark breaks for 1D rectangle merge - if low > hi + 1, must start new rect
  template<int N, typename T>
__global__
void mark_breaks_dim(const HiFlag<T>* hi_flag_in,
                     const HiFlag<T>* hi_flag_out,
                     const RectDesc<N,T>* in,
                     uint8_t*              brk,
                     size_t                M,
                     int                   d)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= M) return;
  brk[i] = hi_flag_in[i].head || in[i].rect.lo[d] > hi_flag_out[i].hi + 1;
}

// 2) Write output rectangles for ND disjoint rects RLE
// Starts write lo, ends write hi, everyone else no-ops
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

  // 2) Write output rectangles for 1D rects RLE
  // Starts write lo, ends write max(hi, prefix max hi) because the max was exclusive
  template<int N, typename T>
  __global__
  void init_rects_dim(const RectDesc<N,T>* in,
                      const HiFlag<T> *hi_flag_out,
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

  size_t g = gid[i] - 1;  // zero-based
  const auto &r = in[i].rect;
  out[g].src_idx = in[i].src_idx;

  // copy dims ≠ d
#pragma unroll
  for(int k = 0; k < N; ++k) {
    if (brk[i]) {
      out[g].rect.lo[k] = r.lo[k];
    }
    if (k != d || (brk[i] && is_end)) {
      out[g].rect.hi[k] = r.hi[k];
    } else if (is_end) {
      out[g].rect.hi[k] = r.hi[k] > hi_flag_out[i].hi ? r.hi[k] : hi_flag_out[i].hi;
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

} // namespace Realm