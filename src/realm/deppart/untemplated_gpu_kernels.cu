#include "realm/deppart/partitions.h"

namespace Realm {

__device__ __forceinline__
int bvh_common_prefix(const uint64_t *morton, const uint64_t *leafIdx, int i, int j, int n) {
  if (j < 0 || j >= n) return -1;
  uint64_t x = morton[i] ^ morton[j];
  uint64_t y = leafIdx[i] ^ leafIdx[j];
  if (x == 0) {
    return 64 + __clzll(y);
  }
  return __clzll(x);
}

__global__
void bvh_build_radix_tree_kernel(
    const uint64_t *morton,    // [n]
    const uint64_t *leafIdx,   // [n]  (unused here but kept for symmetry)
    int n,
    int *childLeft,            // [2n−1]
    int *childRight,           // [2n−1]
    int *parent)               // [2n−1], pre‐initialized to −1
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int i = idx;
  if (i >= n-1) return;            // we only build n−1 internal nodes

  int left, right;
  int dL = bvh_common_prefix(morton, leafIdx, i, i-1, n);
  int dR = bvh_common_prefix(morton, leafIdx, i, i+1, n);
  int d  = (dR > dL ? +1 : -1);
  int deltaMin = (dR > dL ? dL : dR);

  // 3) find j by exponential + binary search
  int l_max = 2;
  int delta = -1;
  int i_tmp = i + d * l_max;
  if (0 <= i_tmp && i_tmp < n) {
    delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
  }
  while (delta > deltaMin) {
    l_max <<= 1;
    i_tmp = i + d * l_max;
    delta = -1;
    if (0 <= i_tmp && i_tmp < n) {
      delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
    }
  }
  int l = 0;
  int t = (l_max) >> 1;
  while (t > 0) {
    i_tmp = i + d*(l + t);
    delta = -1;
    if (0 <= i_tmp && i_tmp < n) {
      delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
    }
    if (delta > deltaMin) {
      l += t;
    }
    t >>= 1;
  }
  if (d < 0) {
    right = i;
    left = i + d*l;
  } else {
    left = i;
    right = i + d*l;
  }

  int gamma;
  if (morton[left] == morton[right] && leafIdx[left] == leafIdx[right]) {
    gamma = (left+right) >> 1;
  } else {
    int deltaNode = bvh_common_prefix(morton, leafIdx, left, right, n);
    int split = left;
    int stride = right - left;
    do {
      stride = (stride + 1) >> 1;
      int middle = split + stride;
      if (middle < right) {
        int delta = bvh_common_prefix(morton, leafIdx, left, middle, n);
        if (delta > deltaNode) {
          split = middle;
        }
      }
    } while (stride > 1);
    gamma = split;
  }

  int left_node = gamma;
  int right_node = gamma + 1;
  if (left == gamma) {
    left_node += n-1;
  }
  if (right == gamma + 1) {
    right_node += n-1;
  }

  childLeft [idx] = left_node;
  childRight[idx] = right_node;
  parent[left_node]  = idx;
  parent[right_node] = idx;
}

__global__
void bvh_build_root_kernel(
    int *root,
    int *parent,
    size_t total_rects) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid >= 2 * total_rects - 1) return;
  if (parent[tid] == -1) {
    *root = tid;
  }
}

}