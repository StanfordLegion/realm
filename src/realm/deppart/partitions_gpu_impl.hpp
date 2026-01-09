#pragma once
#include "deppart_config.h"
#include "partitions.h"
#ifdef REALM_USE_NVTX
#include "realm/nvtx.h"
#endif
#include "realm/cuda/cuda_internal.h"
#include "realm/deppart/partitions_gpu_kernels.hpp"
#include <cub/cub.cuh>

//CUDA ERROR CHECKING MACROS

#define CUDA_CHECK(call, stream)                                                \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " '" #call "' failed with "                                 \
                << cudaGetErrorString(err) << " (" << err << ")\n";            \
      assert(false);                                                  \
    }                                                                           \
  } while (0)

#define KERNEL_CHECK(stream)                                                    \
  do {                                                                          \
    cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "Kernel launch failed at " << __FILE__ << ":" << __LINE__   \
                << ": " << cudaGetErrorString(err) << "\n";                    \
      assert(false);                                                \
    }                                                                        \
  } while (0)

#define THREADS_PER_BLOCK 256

#define COMPUTE_GRID(num_items) \
  (((num_items) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)


//NVTX macros to only add ranges if defined.
#ifdef REALM_USE_NVTX

  #define NVTX_CAT(a,b)  a##b

  #define NVTX_DEPPART(message) \
  nvtxScopedRange NVTX_CAT(nvtx_, message)("cuda", #message, 0)

#else

  #define NVTX_DEPPART(message) do { } while (0)

#endif

namespace Realm {

  // Used by cub::DeviceReduce to compute bad GPU approximation.
  template<int N, typename T>
  struct UnionRectOp {
    __host__ __device__
    Rect<N,T> operator()(const Rect<N,T>& a,
                         const Rect<N,T>& b) const {
      Rect<N,T> r;
      for(int d=0; d<N; d++){
        r.lo[d] = a.lo[d] <  b.lo[d] ? a.lo[d] : b.lo[d];
        r.hi[d] = a.hi[d] > b.hi[d] ? a.hi[d] : b.hi[d];
      }
      return r;
    }
  };

  // Used to compute prefix sum by volume for an array of Rects or RectDescs.
  template <int N, typename T, typename out_t>
  struct RectVolumeOp {
    __device__ __forceinline__
    size_t operator()(const out_t& r) const {
      if constexpr (std::is_same_v<Rect<N, T>, out_t>) {
        return r.volume();
      } else {
        return r.rect.volume();
      }
    }
  };

  // Finds a memory of the specified kind. Returns true on success, false otherwise.
  inline bool find_memory(Memory &output, Memory::Kind kind)
  {
    bool found = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(auto& memory : all_memories) {
      if(memory.kind() == kind) {
        output = memory;
        found = true;
        break;
      }
    }
    return found;
  }

  //Given a list of spaces, compacts them all into one collapsed_space
  template<int N, typename T>
  template<typename space_t>
  void GPUMicroOp<N,T>::collapse_multi_space(const std::vector<space_t>& spaces, collapsed_space<N, T> &out_space, Arena &my_arena, cudaStream_t stream)
  {

    char *val = std::getenv("SHATTER_SIZE");  // or any env var
    int shatter_size = 1; //default
    if (val) {
      shatter_size = atoi(val);
    }
    // We need space_offsets to preserve which space each rectangle came from
    std::vector<size_t> space_offsets(spaces.size() + 1);

    // Determine size of allocation for combined rects.
    out_space.num_entries = 0;

    for (size_t i = 0; i < spaces.size(); ++i) {
      space_offsets[i] = out_space.num_entries;
      IndexSpace<N,T> my_space;
      if constexpr (std::is_same_v<space_t, IndexSpace<N,T>>) {
        my_space = spaces[i];
      } else {
        my_space = spaces[i].index_space;
      }
      if (my_space.dense()) {
        if constexpr (std::is_same_v<space_t, IndexSpace<N,T>>) {
          out_space.num_entries += 1;
        } else {
          out_space.num_entries += shatter_size;
        }
      } else {
        out_space.num_entries += my_space.sparsity.impl()->get_entries().size();
      }
    }
    space_offsets[spaces.size()] = out_space.num_entries;

    //We copy into one contiguous host buffer, then copy to device
    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM));


    RegionInstance h_instance = realm_malloc(out_space.num_entries * sizeof(SparsityMapEntry<N,T>), sysmem);
    SparsityMapEntry<N, T>* h_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(h_instance, 0).base);

    if (my_arena.capacity()==0) {
      out_space.entries_buffer = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(h_instance, 0).base);
    } else {
      out_space.entries_buffer = my_arena.alloc<SparsityMapEntry<N,T> >(out_space.num_entries);
    }


    //Now we fill the host array with all rectangles
    size_t pos = 0;
    for (size_t i = 0; i < spaces.size(); ++i) {
      IndexSpace<N,T> my_space;
      if constexpr (std::is_same_v<space_t, IndexSpace<N,T>>) {
        my_space = spaces[i];
      } else {
        my_space = spaces[i].index_space;
      }
      if (my_space.dense()) {
        if constexpr (std::is_same_v<space_t, IndexSpace<N,T>>) {
          SparsityMapEntry<N,T> entry;
          entry.bounds = my_space.bounds;
          memcpy(h_entries + pos, &entry, sizeof(SparsityMapEntry<N,T>));
          ++pos;
        } else {
          std::vector<SparsityMapEntry<N,T> > tmp(shatter_size);
          int ppt = (my_space.bounds.hi[0] - my_space.bounds.lo[0]+1) / shatter_size;
          for (int i = 0; i < shatter_size; ++i) {
            Rect<N,T> new_rect = my_space.bounds;
            new_rect.lo[0] = my_space.bounds.lo[0] + i * ppt;
            new_rect.hi[0] = (i == shatter_size - 1) ? my_space.bounds.hi[0] : (new_rect.lo[0] + ppt - 1);
            SparsityMapEntry<N,T> entry;
            entry.bounds = new_rect;
            entry.sparsity.id = 0;
            entry.bitmap = 0;
            tmp[i] = entry;
          }
          memcpy(h_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>));
          pos += shatter_size;
        }
      } else {
        span<SparsityMapEntry<N, T>> tmp = my_space.sparsity.impl()->get_entries();
        memcpy(h_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>));
        pos += tmp.size();
      }
    }

    //Now we copy our entries and offsets to the device
    CUDA_CHECK(cudaMemcpyAsync(out_space.offsets, space_offsets.data(), (spaces.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
    if (my_arena.capacity() != 0) {
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, h_entries, out_space.num_entries * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      h_instance.destroy();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

  }

  // Only real work here is getting dense/sparse into a single collapsed_space.
  template<int N, typename T>
  void GPUMicroOp<N,T>::collapse_parent_space(const IndexSpace<N, T>& parent_space, collapsed_space<N, T> &out_space, Arena &my_arena, cudaStream_t stream)
  {
    if (parent_space.dense()) {
      SparsityMapEntry<N,T> entry;
      entry.bounds = parent_space.bounds;
      out_space.entries_buffer = my_arena.alloc<SparsityMapEntry<N, T>>(1);
      out_space.num_entries = 1;
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    } else {
      span<SparsityMapEntry<N, T>> tmp =  parent_space.sparsity.impl()->get_entries();
      out_space.num_entries = tmp.size();
      out_space.entries_buffer = my_arena.alloc<SparsityMapEntry<N, T>>(tmp.size());
      out_space.bounds = parent_space.bounds;
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    }
    out_space.offsets = nullptr;
    out_space.num_children = 1;
  }

  // Given a collapsed space, builds a (potentially marked) bvh over that space.
  // Based on Tero Karras' Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
  template<int N, typename T>
  void GPUMicroOp<N, T>::build_bvh(const collapsed_space<N, T> &space, BVH<N, T> &result, Arena &my_arena, cudaStream_t stream)
  {

      //We want to keep the entire BVH that we return in one instance for convenience.
      size_t indices_instance_size = space.num_entries * sizeof(uint64_t);
      size_t labels_instance_size = space.offsets == nullptr ? 0 : space.num_entries * sizeof(size_t);
      size_t boxes_instance_size =  (2*space.num_entries - 1) * sizeof(Rect<N, T>);
      size_t child_instance_size = (2*space.num_entries - 1) * sizeof(int);

      size_t total_instance_size = indices_instance_size + labels_instance_size + boxes_instance_size + 2 * child_instance_size;
      char* bvh_ptr = my_arena.alloc<char>(total_instance_size);

      result.num_leaves = space.num_entries;

      size_t curr_idx = 0;
      result.indices = reinterpret_cast<uint64_t*>(bvh_ptr + curr_idx);
      curr_idx += indices_instance_size;
      result.labels = space.offsets == nullptr ? nullptr : reinterpret_cast<size_t*>(bvh_ptr + curr_idx);
      curr_idx += labels_instance_size;
      result.boxes = reinterpret_cast<Rect<N,T>*>(bvh_ptr + curr_idx);
      curr_idx += boxes_instance_size;
      result.childLeft = reinterpret_cast<int*>(bvh_ptr + curr_idx);
      curr_idx += child_instance_size;
      result.childRight = reinterpret_cast<int*>(bvh_ptr + curr_idx);

      size_t prev = my_arena.mark();

      // Bounds used for morton code computation.
      Rect<N,T>* d_global_bounds = my_arena.alloc<Rect<N,T>>(1);
      CUDA_CHECK(cudaMemcpyAsync(d_global_bounds, &space.bounds, sizeof(Rect<N,T>), cudaMemcpyHostToDevice, stream), stream);

      // These are intermediate instances we'll destroy before returning.
      char* d_morton_visit = my_arena.alloc<char>(2 * space.num_entries * max(sizeof(uint64_t), sizeof(int)));
      uint64_t* d_morton_codes = reinterpret_cast<uint64_t*>(d_morton_visit);

      size_t intermed = my_arena.mark();

      uint64_t* d_indices_in = my_arena.alloc<uint64_t>(space.num_entries);

      // We compute morton codes for each leaf and sort, labeling if necessary.
      bvh_build_morton_codes<N, T><<<COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.entries_buffer, space.offsets, d_global_bounds, space.num_entries, space.num_children, d_morton_codes, d_indices_in, result.labels);
      KERNEL_CHECK(stream);

      uint64_t* d_morton_codes_out = d_morton_codes + space.num_entries;
      uint64_t* d_indices_out = result.indices;

      void *bvh_temp = nullptr;
      size_t bvh_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices_in,
                                      d_indices_out, space.num_entries, 0, 64, stream);
      bvh_temp = reinterpret_cast<void*>(my_arena.alloc<char>(bvh_temp_bytes));
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices_in,
                                      d_indices_out, space.num_entries, 0, 64, stream);

      std::swap(d_morton_codes, d_morton_codes_out);

      my_arena.rollback(intermed);


      // Another temporary instance.
      int* d_parent = my_arena.alloc<int>(2*space.num_entries - 1);
      CUDA_CHECK(cudaMemsetAsync(d_parent, -1, (2*space.num_entries - 1) * sizeof(int), stream), stream);

      // Here's where we actually build the BVH
      int n = (int) space.num_entries;
      bvh_build_radix_tree_kernel<<< COMPUTE_GRID(space.num_entries - 1), THREADS_PER_BLOCK, 0, stream>>>(d_morton_codes, result.indices, n, result.childLeft, result.childRight, d_parent);
      KERNEL_CHECK(stream);

      // Figure out which node didn't get its parent set.
      int* d_root = my_arena.alloc<int>(1);

      CUDA_CHECK(cudaMemsetAsync(d_root, -1, sizeof(int), stream), stream);

      bvh_build_root_kernel<<< COMPUTE_GRID(2 * space.num_entries - 1), THREADS_PER_BLOCK, 0, stream>>>(d_root, d_parent, space.num_entries);
      KERNEL_CHECK(stream);

      CUDA_CHECK(cudaMemcpyAsync(&result.root, d_root, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      // Now we materialize the tree into something the client can query.
      bvh_init_leaf_boxes_kernel<N, T><<<COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.entries_buffer, result.indices, space.num_entries, result.boxes);
      KERNEL_CHECK(stream);

      int* d_visitCount = reinterpret_cast<int*>(d_morton_visit);
      CUDA_CHECK(cudaMemsetAsync(d_visitCount, 0, (2*space.num_entries - 1) * sizeof(int), stream), stream);

      bvh_merge_internal_boxes_kernel < N, T ><<< COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.num_entries, result.childLeft, result.childRight, d_parent, result.boxes, d_visitCount);
      KERNEL_CHECK(stream);

      // Cleanup.
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      my_arena.rollback(prev);

  }

  // Intersects two collapsed spaces, where lhs is always instances and rhs is either parent or sources/targets.
  // If rhs is sources/targets, we mark the intersected rectangles by where they came from.
  // If the intersection is costly, we accelerate with a BVH.
  template<int N, typename T>
  template<typename out_t>
  void GPUMicroOp<N,T>::construct_input_rectlist(const collapsed_space<N, T> &lhs, const collapsed_space<N, T> &rhs, out_t* &d_valid_rects, size_t& out_size, uint32_t* counters, uint32_t* out_offsets, Arena &my_arena, cudaStream_t stream)
  {

    CUDA_CHECK(cudaMemsetAsync(counters, 0, (lhs.num_children) * sizeof(uint32_t), stream), stream);

    BVH<N, T> my_bvh;
    bool bvh_valid = rhs.num_children < rhs.num_entries;
    if (bvh_valid) {
      build_bvh(rhs, my_bvh, my_arena, stream);
    }

    // First pass: figure out how many rectangles survive intersection.
    if (!bvh_valid) {
      intersect_input_rects<N, T, out_t><<<COMPUTE_GRID(lhs.num_entries * rhs.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs.entries_buffer, rhs.entries_buffer, lhs.offsets, nullptr, rhs.offsets, lhs.num_entries, rhs.num_entries, lhs.num_children, rhs.num_children, counters, nullptr);
    } else {
      query_input_bvh<N, T, out_t><<<COMPUTE_GRID(lhs.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs.entries_buffer, lhs.offsets, my_bvh.root, my_bvh.childLeft, my_bvh.childRight, my_bvh.indices, my_bvh.labels, my_bvh.boxes, lhs.num_entries, my_bvh.num_leaves, lhs.num_children, nullptr, counters, nullptr);
    }
    KERNEL_CHECK(stream);


    // Prefix sum over instances (small enough to keep on host).
    std::vector<uint32_t> h_inst_counters(lhs.num_children+1);
    h_inst_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_inst_counters.data()+1, counters, lhs.num_children * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 0; i < lhs.num_children; ++i) {
      h_inst_counters[i+1] += h_inst_counters[i];
    }

    out_size = h_inst_counters[lhs.num_children];

    if (out_size==0) {
      return;
    }

    //Moving on...
    my_arena.flip_parity();

    // Non-empty rectangles from the intersection.
    d_valid_rects = my_arena.alloc<out_t>(out_size);

    // Where each instance should start writing its rectangles.
    CUDA_CHECK(cudaMemcpyAsync(out_offsets, h_inst_counters.data(), (lhs.num_children + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

    // Reset counters.
    CUDA_CHECK(cudaMemsetAsync(counters, 0, lhs.num_children * sizeof(uint32_t), stream), stream);

    // Second pass: recompute intersection, but this time write to output.
    if (!bvh_valid) {
      intersect_input_rects<N, T, out_t><<<COMPUTE_GRID(lhs.num_entries * rhs.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs.entries_buffer, rhs.entries_buffer, lhs.offsets, out_offsets, rhs.offsets, lhs.num_entries, rhs.num_entries, lhs.num_children, rhs.num_children, counters, d_valid_rects);
    } else {
      query_input_bvh<N, T, out_t><<<COMPUTE_GRID(lhs.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs.entries_buffer, lhs.offsets, my_bvh.root, my_bvh.childLeft, my_bvh.childRight, my_bvh.indices, my_bvh.labels, my_bvh.boxes, lhs.num_entries, my_bvh.num_leaves, lhs.num_children, out_offsets, counters, d_valid_rects);
    }
    KERNEL_CHECK(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
  }

  // Prefix sum an array of Rects or RectDescs by volume.
  template<int N, typename T>
  template<typename out_t>
  void GPUMicroOp<N, T>::volume_prefix_sum(const out_t* d_rects, size_t total_rects, size_t* &d_prefix_rects, size_t& num_pts, Arena &my_arena, cudaStream_t stream)
  {
    d_prefix_rects = my_arena.alloc<size_t>(total_rects+1);
    CUDA_CHECK(cudaMemsetAsync(d_prefix_rects, 0, sizeof(size_t), stream), stream);

    size_t prev = my_arena.mark();

    // Build the CUB transform‐iterator.
    using VolIter = cub::TransformInputIterator<
                      size_t,              // output type
                      RectVolumeOp<N,T,out_t>,            // functor
                      const out_t*     // underlying input iterator
                    >;
    VolIter d_volumes(d_rects, RectVolumeOp<N,T,out_t>());

    void*   d_temp = nullptr;
    size_t rect_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        /* d_temp_storage */  nullptr,
        /* temp_bytes */      rect_temp_bytes,
        /* d_in */            d_volumes,
        /* d_out */           d_prefix_rects + 1,   // shift by one so prefix[1]..prefix[n]
        /* num_items */       total_rects, stream);

    d_temp = reinterpret_cast<void*>(my_arena.alloc<char>(rect_temp_bytes));
    cub::DeviceScan::InclusiveSum(
        /* d_temp_storage */  d_temp,
        /* temp_bytes */      rect_temp_bytes,
        /* d_in */            d_volumes,
        /* d_out */           d_prefix_rects + 1,
        /* num_items */       total_rects, stream);


    //Number of points across all rectangles (also our total output count).
    CUDA_CHECK(cudaMemcpyAsync(&num_pts, &d_prefix_rects[total_rects], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    my_arena.rollback(prev);
  }

  template<typename T>
  struct SegmentedMax {
    __device__ __forceinline__
    HiFlag<T> operator()(HiFlag<T> a, HiFlag<T> b) const {
      // if b.head==1, start new segment at b; otherwise merge with running max
      return b.head
        ? b
      : HiFlag<T>{ a.hi > b.hi ? a.hi : b.hi , a.head };
    }
  };

  struct SegmentedSum {
    __device__ __forceinline__
    DeltaFlag operator()(DeltaFlag a, DeltaFlag b) const {
      // if b.head==1, start new segment at b; otherwise merge with running max
      return b.head
        ? b
      : DeltaFlag{ a.delta + b.delta , a.head };
    }
  };

  struct CustomSum
  {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
      return b+a;
    }
  };


 /*
  *  Input: An array of rectangles (potentially overlapping) with associated
  *  src indices, where all the rectangles with a given src idx together represent an exact covering
  *  of the partitioning output for that index.
  *  Output: A disjoint, coalesced array of rectangles sorted by src idx that it then sends off
  *  to the send output function, which constructs the final sparsity map.
  *  Approach: The difficult part is constructing a disjoint covering. To do so, collect all the corners from all the
  *  rectangles as the unique "boundaries" for each dimension and mark them with the parity for the number of dimensions
  *  in which they are the hi+1 coord (we add 1 to make intervals half-open). This means that if you prefix sum in each dimension,
  *  for any given rectangle anything internal will sum to 1, and anything external will sum to 0.  To understand the intuition,
  *  see the illustration below for the rectangle [(0,0), (2,2)]
  *  Corners: (0,0), (0,3), (3,0), (3,3)
  *  Parities: 0 hi-> +1, 1 hi -> -1, 1 hi -> -1, 2 hi -> +1
  *  Computation:
  *  Initial Markings
  *    0  1  2  3  4 ...
  *  0 +1      -1
  *  1
  *  2
  *  3 -1      +1
  *  4
  *  ...
  *  Prefix sum by Y
  *    0  1  2  3  4 ...
  *  0 +1      -1
  *  1  1      -1
  *  2  1      -1
  *  3  0       0
  *  4  0       0
  *  ...
  *  Prefix sum by X
  *    0   1  2  3  4 ...
  *  0 +1  1  1  0  0 ...
  *  1  1  1  1  0  0 ...
  *  2  1  1  1  0  0 ...
  *  3  0  0  0  0  0 ...
  *  4  0  0  0  0  0 ...
  *  ...
  *  Note that all the points in the rectangle end up labeled 1, and all the points outside labeled 0. In the actual computation, we use segments
  *  rather than points, where a segment accounts for all points between two consecutive boundaries. Because a prefix sum is a linear operator, when
  *  we extend the computation above to multiple overlapping rectangles, you end up with included segments labeled with a count of how many rectangles include them,
  *  and excluded segments labeled with 0. Thus, for the last dimension, we emit all segments with sums > 0 as disjoint output rectangles. We can then dump these
  *  into the sort + coalesce pipeline.
  */
  template<int N, typename T>
  template<typename Container, typename IndexFn, typename MapFn>
  void GPUMicroOp<N,T>::complete_rect_pipeline(RectDesc<N, T>* d_rects, size_t total_rects, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {

    //1D case is much simpler
    if (N==1) {
      this->complete1d_pipeline(d_rects, total_rects, d_out_rects, out_rects, my_arena, ctr, getIndex, getMap);
      return;
    }
    NVTX_DEPPART(complete_rect_pipeline);
    cudaStream_t stream = Cuda::get_task_cuda_stream();

    Memory my_mem;
    bool found = find_memory(my_mem, Memory::GPU_FB_MEM);
    assert(found);

    RegionInstance srcs_instance = this->realm_malloc(4*total_rects*sizeof(int32_t), my_mem);
    RegionInstance crds_instance = this->realm_malloc(4*total_rects*sizeof(T), my_mem);
    RegionInstance heads_instance = this->realm_malloc(2*total_rects * sizeof(uint8_t), my_mem);
    RegionInstance sum_instance = this->realm_malloc(2*total_rects * sizeof(size_t), my_mem);

    RegionInstance B_src_inst[N];
    RegionInstance B_coord_inst[N];

    size_t *B_starts[N];
    size_t *B_ends[N];

    T* B_coord[N];
    size_t B_size[N];

    RegionInstance B_ptrs_instance = this->realm_malloc(2 * N * sizeof(size_t*), my_mem);
    size_t** B_start_ptrs = reinterpret_cast<size_t**>(AffineAccessor<char,1>(B_ptrs_instance, 0).base);
    size_t** B_end_ptrs = reinterpret_cast<size_t**>(AffineAccessor<char,1>(B_ptrs_instance, 0).base) + N;

    RegionInstance B_coord_ptrs_instance = this->realm_malloc(N * sizeof(T*), my_mem);
    T** B_coord_ptrs = reinterpret_cast<T**>(AffineAccessor<char,1>(B_coord_ptrs_instance, 0).base);
    
    int threads_per_block = 256;
    size_t grid_size = (total_rects + threads_per_block - 1) / threads_per_block;

    RegionInstance tmp_instance;
    size_t orig_tmp = 0;
    void *tmp_storage = nullptr;

    //Our first step is to find all the unique "boundaries" in each dimension (lo coord or hi+1 coord)
    {
      NVTX_DEPPART(mark_endpoints);
      for (int d = 0; d < N; ++d) {

        //We need the coordinates to be sorted by our curent dim and separated by src idx
        grid_size = (total_rects + threads_per_block - 1) / threads_per_block;
        uint32_t* d_srcs_in = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(srcs_instance, 0).base);
        uint32_t* d_srcs_out = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(srcs_instance, 0).base) + 2* total_rects;
        T* d_coord_keys_in = reinterpret_cast<T*>(AffineAccessor<char,1>(crds_instance,0).base);
        T* d_coord_keys_out = reinterpret_cast<T*>(AffineAccessor<char,1>(crds_instance,0).base) + 2 * total_rects;
        mark_endpoints<<<grid_size, threads_per_block, 0, stream>>>(d_rects, total_rects, d, d_srcs_in, d_coord_keys_in);
        KERNEL_CHECK(stream);
        size_t temp_bytes;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                            d_coord_keys_in, d_coord_keys_out,
                                            d_srcs_in, d_srcs_out,
                                            2 * total_rects, 0, 8*sizeof(T), stream);
        if (temp_bytes > orig_tmp) {
          if (orig_tmp > 0) {
            tmp_instance.destroy();
          }
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                  d_coord_keys_in, d_coord_keys_out,
                                                  d_srcs_in, d_srcs_out,
                                                  2 * total_rects, 0, 8*sizeof(T), stream);
        std::swap(d_srcs_in, d_srcs_out);
        std::swap(d_coord_keys_in, d_coord_keys_out);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                            d_srcs_in, d_srcs_out,
                                            d_coord_keys_in, d_coord_keys_out,
                                            2 * total_rects, 0, 8*sizeof(uint32_t), stream);
        if (temp_bytes > orig_tmp) {
          if (orig_tmp > 0) {
            tmp_instance.destroy();
          }
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                            d_srcs_in, d_srcs_out,
                                            d_coord_keys_in, d_coord_keys_out,
                                            2 * total_rects, 0, 8*sizeof(uint32_t), stream);

        //Now mark the unique keys
        grid_size = (2*total_rects + threads_per_block - 1) / threads_per_block;
        uint8_t * d_heads = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(heads_instance, 0).base);
        size_t *d_output = reinterpret_cast<size_t *>(AffineAccessor<char,1>(sum_instance, 0).base);
        mark_heads<<<grid_size, threads_per_block, 0, stream>>>(d_srcs_out, d_coord_keys_out, 2 * total_rects, d_heads);
        KERNEL_CHECK(stream);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_heads, d_output, 2 * total_rects, stream);
        if (temp_bytes > orig_tmp) {
          if (orig_tmp > 0) {
            tmp_instance.destroy();
          }
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_heads, d_output, 2 * total_rects, stream);

        size_t num_unique;
        uint8_t last_bit;
        CUDA_CHECK(cudaMemcpyAsync(&num_unique, &d_output[2*total_rects-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaMemcpyAsync(&last_bit, &d_heads[2*total_rects-1], sizeof(uint8_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        num_unique += last_bit;

        //Collect all the data we'll need later for this dimension - starts/ends by src, unique boundaries, unique boundaries count
        B_coord_inst[d] = this->realm_malloc(num_unique * sizeof(T), my_mem);
        B_src_inst[d] = this->realm_malloc(2*ctr.size() * sizeof(size_t), my_mem);
        B_starts[d] = reinterpret_cast<size_t*>(AffineAccessor<char,1>(B_src_inst[d], 0).base);
        B_ends[d] = reinterpret_cast<size_t*>(AffineAccessor<char,1>(B_src_inst[d], 0).base) + ctr.size();
        B_coord[d] = reinterpret_cast<T*>(AffineAccessor<char,1>(B_coord_inst[d], 0).base);
        B_size[d] = num_unique;
        CUDA_CHECK(cudaMemsetAsync(B_starts[d], 0, ctr.size() * sizeof(size_t), stream), stream);
        CUDA_CHECK(cudaMemsetAsync(B_ends[d], 0, ctr.size() * sizeof(size_t), stream), stream);
        scatter_unique<<<grid_size, threads_per_block, 0, stream>>>(d_srcs_out, d_coord_keys_out, d_output, d_heads, 2 * total_rects, B_starts[d], B_ends[d], B_coord[d]);
        KERNEL_CHECK(stream);
        std::vector<size_t> d_starts_host(ctr.size()), d_ends_host(ctr.size());
        CUDA_CHECK(cudaMemcpyAsync(d_starts_host.data(), B_starts[d], ctr.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaMemcpyAsync(d_ends_host.data(), B_ends[d], ctr.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        for (size_t i = 1; i < ctr.size(); i++) {
          if (d_starts_host[i] < d_ends_host[i-1]) {
            d_starts_host[i] = d_ends_host[i-1];
            d_ends_host[i] = d_ends_host[i-1];
          }
        }
        CUDA_CHECK(cudaMemcpyAsync(B_starts[d], d_starts_host.data(), ctr.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
        CUDA_CHECK(cudaMemcpyAsync(B_ends[d], d_ends_host.data(), ctr.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }
    srcs_instance.destroy();
    crds_instance.destroy();
    heads_instance.destroy();
    sum_instance.destroy();


    //We need the arrays themselves on the device
    CUDA_CHECK(cudaMemcpyAsync(B_coord_ptrs, B_coord, N * sizeof(T*), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(B_start_ptrs, B_starts, N * sizeof(size_t*), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(B_end_ptrs, B_ends, N * sizeof(size_t*), cudaMemcpyHostToDevice, stream), stream);

    //Next up, we generate all the corners of all the rectangles and mark them by parity
    size_t num_corners = (1 << N);
    RegionInstance corners_instance = this->realm_malloc(2 * num_corners * total_rects * sizeof(CornerDesc<N, T>), my_mem);
    CornerDesc<N, T>* d_corners_in = reinterpret_cast<CornerDesc<N, T>*>(AffineAccessor<char,1>(corners_instance, 0).base);
    CornerDesc<N, T>* d_corners_out = reinterpret_cast<CornerDesc<N, T>*>(AffineAccessor<char,1>(corners_instance, 0).base) + num_corners * total_rects;

    populate_corners<<<grid_size, threads_per_block, 0, stream>>>(d_rects, total_rects, d_corners_in);
    KERNEL_CHECK(stream);


    // We have a LOT of bookkeeping to do
    std::set<Event> RLE_alloc_events;

    size_t alloc_size_1 = std::max({sizeof(size_t), sizeof(T), sizeof(int32_t), sizeof(DeltaFlag)});

    RegionInstance shared_instance = this->realm_malloc(2 * num_corners * total_rects * alloc_size_1, my_mem);

    RegionInstance flags_instance = this->realm_malloc(num_corners * total_rects * sizeof(uint8_t), my_mem);

    RegionInstance exc_sum_instance = this->realm_malloc(num_corners * total_rects * sizeof(size_t), my_mem);

    size_t per_elem_size = 2*alloc_size_1 + sizeof(uint8_t) + sizeof(size_t);

    size_t* d_src_keys_in = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base);
    size_t* d_src_keys_out = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_corners * total_rects;
    T* d_coord_keys_in = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base);
    T* d_coord_keys_out = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_corners * total_rects;
    int32_t* d_deltas = reinterpret_cast<int32_t*>(AffineAccessor<char,1>(shared_instance, 0).base);
    int32_t* d_deltas_out = reinterpret_cast<int32_t*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_corners * total_rects;
    DeltaFlag* d_delta_flags_in = reinterpret_cast<DeltaFlag*>(AffineAccessor<char,1>(shared_instance, 0).base);
    DeltaFlag* d_delta_flags_out = reinterpret_cast<DeltaFlag*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_corners * total_rects;
    uint8_t* d_flags = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(flags_instance, 0).base);
    size_t* d_exc_sum = reinterpret_cast<size_t*>(AffineAccessor<char,1>(exc_sum_instance, 0).base);

    RegionInstance seg_bound_instance;
    size_t* seg_starts;
    size_t* seg_ends;

    RegionInstance seg_counters;
    uint32_t* d_seg_counters;

    RegionInstance seg_counters_out;
    uint32_t* d_seg_counters_out;

    grid_size = (num_corners * total_rects + threads_per_block - 1) / threads_per_block;

    //We need to reduce duplicate corners by their parity, so we sort to get duplicates next to each other and then reduce by key
    {
      NVTX_DEPPART(sort_corners);
      for (int dim = 0; dim < N; dim++) {
        build_coord_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_corners_in, num_corners * total_rects, dim);
        KERNEL_CHECK(stream);
        size_t temp_bytes;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                            d_coord_keys_in, d_coord_keys_out,
                                            d_corners_in, d_corners_out,
                                            num_corners * total_rects, 0, 8*sizeof(T), stream);
        if (temp_bytes > orig_tmp) {
          tmp_instance.destroy();
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                        d_coord_keys_in, d_coord_keys_out,
                                                        d_corners_in, d_corners_out,
                                                        num_corners * total_rects, 0, 8*sizeof(T), stream);

        std::swap(d_corners_in, d_corners_out);

      }
    }

    size_t temp_bytes;
    build_src_key<<<grid_size, threads_per_block, 0, stream>>>(d_src_keys_in, d_corners_in, num_corners * total_rects);
    KERNEL_CHECK(stream);
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                            d_src_keys_in, d_src_keys_out,
                                            d_corners_in, d_corners_out,
                                            num_corners * total_rects, 0, 8*sizeof(size_t), stream);
    if (temp_bytes > orig_tmp) {
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      orig_tmp = temp_bytes;
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
    }
    cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                    d_src_keys_in, d_src_keys_out,
                                                    d_corners_in, d_corners_out,
                                                    num_corners * total_rects, 0, 8*sizeof(size_t), stream);

    std::swap(d_corners_in, d_corners_out);
    get_delta<<<grid_size, threads_per_block, 0, stream>>>(d_deltas, d_corners_in, num_corners * total_rects);
    KERNEL_CHECK(stream);

    RegionInstance num_runs_instance = this->realm_malloc(sizeof(int), my_mem);
    int* d_num_runs = reinterpret_cast<int*>(AffineAccessor<char,1>(num_runs_instance, 0).base);

    //See above, we have custom equality and reduction operators for CornerDesc
    CustomSum red_op;
    cub::DeviceReduce::ReduceByKey(
        nullptr, temp_bytes,
        d_corners_in, d_corners_out,
        d_deltas, d_deltas_out,
        d_num_runs,
        red_op,
        /*num_items=*/(int) (num_corners * total_rects),
         /*stream=*/stream);

    if (temp_bytes > orig_tmp) {
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      orig_tmp = temp_bytes;
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
    }
    cub::DeviceReduce::ReduceByKey(
        tmp_storage, temp_bytes,
        d_corners_in, d_corners_out,
        d_deltas, d_deltas_out,
        d_num_runs,
        red_op,
        /*num_items=*/(int) (num_corners * total_rects),
         /*stream=*/stream);

    int num_unique_corners;
    CUDA_CHECK(cudaMemcpyAsync(&num_unique_corners, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    num_runs_instance.destroy();

    grid_size = (num_unique_corners + threads_per_block - 1) / threads_per_block;
    set_delta<<<grid_size, threads_per_block, 0, stream>>>(d_deltas_out, d_corners_out, num_unique_corners);
    KERNEL_CHECK(stream);

    std::swap(d_corners_out, d_corners_in);

    size_t num_intermediate = num_unique_corners;
    size_t num_segments;

    //This is where the real work is done. In each dimension, we do a segmented prefix sum of the parity markings keyed on (src idx, {every dim but d}) for all active segments.
    // Then, for each unique boundary b in dim d, for each segment s keyed on (src idx, {every dim but d}), we evaluate s's prefix sum value at b. If nonzero, we emit a segment
    // for s between b and the next boundary in d with all the other coords set to s's coords. These become the active segments for the next pass. In the last pass (d = 0), rather
    // than emitting segments, we emit rectangles for all segments with nonzero prefix sums (in fact they must also be nonnegative - recall the model is > 0 for included, 0 for excluded
    // by the end).
    {
      NVTX_DEPPART(collapse_higher_dims);
      for (int d = N-1; d >= 0; d--) {
        grid_size = (num_intermediate + threads_per_block - 1) / threads_per_block;

        //Our least significant sort is by d.
        build_coord_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_corners_in, num_intermediate, d);
        KERNEL_CHECK(stream);
        size_t temp_bytes;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                            d_coord_keys_in, d_coord_keys_out,
                                            d_corners_in, d_corners_out,
                                            num_intermediate, 0, 8*sizeof(T), stream);
        if (temp_bytes > orig_tmp) {
          tmp_instance.destroy();
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                        d_coord_keys_in, d_coord_keys_out,
                                                        d_corners_in, d_corners_out,
                                                        num_intermediate, 0, 8*sizeof(T), stream);

        std::swap(d_corners_in, d_corners_out);

        //We need to key segments on every dimension but d and src idx, so we do a series of stable sorts to get there
        for (int dim = 0; dim < N; dim++) {
          if (dim == d) {
            continue;
          }
          build_coord_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_corners_in, num_intermediate, dim);
          KERNEL_CHECK(stream);
          size_t temp_bytes;
          cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                              d_coord_keys_in, d_coord_keys_out,
                                              d_corners_in, d_corners_out,
                                              num_intermediate, 0, 8*sizeof(T), stream);
          if (temp_bytes > orig_tmp) {
            tmp_instance.destroy();
            tmp_instance = this->realm_malloc(temp_bytes, my_mem);
            orig_tmp = temp_bytes;
            tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
          }
          cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                          d_coord_keys_in, d_coord_keys_out,
                                                          d_corners_in, d_corners_out,
                                                          num_intermediate, 0, 8*sizeof(T), stream);

          std::swap(d_corners_in, d_corners_out);

        }

        build_src_key<<<grid_size, threads_per_block, 0, stream>>>(d_src_keys_in, d_corners_in, num_intermediate);
        KERNEL_CHECK(stream);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                                d_src_keys_in, d_src_keys_out,
                                                d_corners_in, d_corners_out,
                                                num_intermediate, 0, 8*sizeof(size_t), stream);
        if (temp_bytes > orig_tmp) {
          tmp_instance.destroy();
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                        d_src_keys_in, d_src_keys_out,
                                                        d_corners_in, d_corners_out,
                                                        num_intermediate, 0, 8*sizeof(size_t), stream);

        std::swap(d_corners_in, d_corners_out);

        //This serves 2 purposes
        // 1) Our segmented prefix sum needs to know where to start and stop
        // 2) We need to know how many unique segments (keyed on (src_idx, {every dimension but d}) we have
        mark_deltas_heads<<<grid_size, threads_per_block, 0, stream>>>(d_corners_in, num_intermediate, d, d_flags, d_delta_flags_in);
        KERNEL_CHECK(stream);

        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_flags, d_exc_sum, num_intermediate, stream);
        if (temp_bytes > orig_tmp) {
          if (orig_tmp > 0) {
            tmp_instance.destroy();
          }
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceScan::InclusiveSum(tmp_storage, temp_bytes, d_flags, d_exc_sum, num_intermediate, stream);

        CUDA_CHECK(cudaMemcpyAsync(&num_segments, &d_exc_sum[num_intermediate-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        //Mark the beginning and end of each segment for our kernel to use in binary search
        seg_bound_instance = this->realm_malloc(2 * num_segments * sizeof(size_t), my_mem);
        seg_starts = reinterpret_cast<size_t*>(AffineAccessor<char,1>(seg_bound_instance, 0).base);
        seg_ends = reinterpret_cast<size_t*>(AffineAccessor<char,1>(seg_bound_instance, 0).base) + num_segments;

        seg_boundaries<<<grid_size, threads_per_block, 0, stream>>>(d_flags, d_exc_sum, num_intermediate, seg_starts, seg_ends);
        KERNEL_CHECK(stream);

        //Segmented prefix sum using our flags constructed above
        cub::DeviceScan::InclusiveScan(
          /*d_temp=*/    nullptr,
          /*bytes=*/     temp_bytes,
          /*in=*/        d_delta_flags_in,
          /*out=*/       d_delta_flags_out,
          /*op=*/        SegmentedSum(),
          /*num_items=*/ num_intermediate,
          /*stream=*/    stream
        );

        if (temp_bytes > orig_tmp) {
          tmp_instance.destroy();
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }

        cub::DeviceScan::InclusiveScan(
          /*d_temp=*/    tmp_storage,
          /*bytes=*/     temp_bytes,
          /*in=*/        d_delta_flags_in,
          /*out=*/       d_delta_flags_out,
          /*op=*/        SegmentedSum(),
          /*num_items=*/ num_intermediate,
          /*stream=*/    stream
        );

        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        //Per usual, we do a count + emit pass to track active segments and limit memory usage. If the evaluated prefix sum for a boundary within a segment
        //is 0, we can skip it because it won't contribute anything to future sums and also won't be emitted.
        seg_counters = this->realm_malloc(num_segments * sizeof(uint32_t), my_mem);
        d_seg_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(seg_counters, 0).base);
        CUDA_CHECK(cudaMemsetAsync(d_seg_counters, 0, num_segments * sizeof(uint32_t), stream), stream);

        grid_size = ((num_segments*B_size[d]) + threads_per_block - 1) / threads_per_block;
        count_segments<<<grid_size, threads_per_block, 0, stream>>>(d_delta_flags_out, seg_starts, seg_ends, B_starts[d], B_ends[d], d_corners_in, B_coord[d], B_size[d], num_segments, d, d_seg_counters);
        KERNEL_CHECK(stream);

        seg_counters_out = this->realm_malloc(num_segments * sizeof(uint32_t), my_mem);
        d_seg_counters_out = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(seg_counters_out, 0).base);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_seg_counters, d_seg_counters_out, num_segments, stream);
        if (temp_bytes > orig_tmp) {
          if (orig_tmp > 0) {
            tmp_instance.destroy();
          }
          tmp_instance = this->realm_malloc(temp_bytes, my_mem);
          orig_tmp = temp_bytes;
          tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
        }
        cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_seg_counters, d_seg_counters_out, num_segments, stream);

        uint32_t next_round;
        uint32_t last_count;
        CUDA_CHECK(cudaMemcpyAsync(&next_round, &d_seg_counters_out[num_segments-1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaMemcpyAsync(&last_count, &d_seg_counters[num_segments-1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        next_round += last_count;
        if (out_rects > 0 && (next_round + last_count) * per_elem_size > out_rects) {
          shared_instance.destroy();
          flags_instance.destroy();
          exc_sum_instance.destroy();
          seg_bound_instance.destroy();
          seg_counters.destroy();
          seg_counters_out.destroy();
          corners_instance.destroy();
          out_rects = std::numeric_limits<size_t>::max();
          return;
        }

        num_intermediate = next_round;

        //In this case we exit out to emit rectangles rather than segments
        if (d==0) {
          break;
        }

        RegionInstance next_corners_instance = this->realm_malloc(2 * next_round * sizeof(CornerDesc<N, T>), my_mem);
        CornerDesc<N, T>* d_next_corners = reinterpret_cast<CornerDesc<N, T>*>(AffineAccessor<char,1>(next_corners_instance, 0).base);
        CUDA_CHECK(cudaMemsetAsync(d_seg_counters, 0, num_segments*sizeof(uint32_t), stream), stream);

        write_segments<<<grid_size, threads_per_block, 0, stream>>>(d_delta_flags_out, seg_starts, seg_ends, B_starts[d], B_ends[d], d_corners_in, B_coord[d], d_seg_counters_out, B_size[d], num_segments, d, d_seg_counters, d_next_corners);
        KERNEL_CHECK(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        corners_instance.destroy();
        corners_instance = next_corners_instance;
        d_corners_in = d_next_corners;
        d_corners_out = d_next_corners + next_round;

        //The segment count in each iter is not monotonic, so we have to realloc each time

        shared_instance.destroy();
        flags_instance.destroy();
        exc_sum_instance.destroy();
        seg_bound_instance.destroy();
        seg_counters.destroy();
        seg_counters_out.destroy();

        shared_instance = this->realm_malloc(2 * num_intermediate * alloc_size_1, my_mem);
        flags_instance = this->realm_malloc(num_intermediate * sizeof(uint8_t), my_mem);
        exc_sum_instance = this->realm_malloc(num_intermediate * sizeof(size_t), my_mem);

        d_src_keys_in = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base);
        d_src_keys_out = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;
        d_coord_keys_in = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base);
        d_coord_keys_out = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;
        d_deltas = reinterpret_cast<int32_t*>(AffineAccessor<char,1>(shared_instance, 0).base);
        d_deltas_out = reinterpret_cast<int32_t*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;

        d_flags = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(flags_instance, 0).base);
        d_exc_sum = reinterpret_cast<size_t*>(AffineAccessor<char,1>(exc_sum_instance, 0).base);
        d_delta_flags_in = reinterpret_cast<DeltaFlag*>(AffineAccessor<char,1>(shared_instance, 0).base);
        d_delta_flags_out = reinterpret_cast<DeltaFlag*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;

      }
    }


    //For our last dim, we emit rectangles rather than segments. These rectangles are a disjoint, precise covering of the original set.
    RegionInstance rects_out_instance = this->realm_malloc(2 * num_intermediate * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N,T>* d_rects_out = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(rects_out_instance, 0).base);
    RectDesc<N, T>* d_rects_in = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(rects_out_instance, 0).base) + num_intermediate;
    CUDA_CHECK(cudaMemsetAsync(d_seg_counters, 0, num_segments*sizeof(uint32_t), stream), stream);

    write_segments<<<grid_size, threads_per_block, 0, stream>>>(d_delta_flags_out, seg_starts, seg_ends, B_start_ptrs, B_end_ptrs, d_corners_in, B_coord_ptrs, d_seg_counters_out, B_size[0], num_segments, d_seg_counters, d_rects_out);
    KERNEL_CHECK(stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    //Don't need these anymore
    flags_instance.destroy();
    exc_sum_instance.destroy();
    seg_bound_instance.destroy();
    seg_counters.destroy();
    seg_counters_out.destroy();
    corners_instance.destroy();
    for (int d = 0; d < N; d++) {
      B_coord_inst[d].destroy();
      B_src_inst[d].destroy();
    }
    B_ptrs_instance.destroy();
    B_coord_ptrs_instance.destroy();

    std::swap(d_rects_out, d_rects_in);

    shared_instance.destroy();
    size_t alloc_size_2 = max(sizeof(size_t), sizeof(T));

    shared_instance = this->realm_malloc(2 * num_intermediate * alloc_size_2, my_mem);

    d_src_keys_in = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base);
    d_src_keys_out = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;
    d_coord_keys_in = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base);
    d_coord_keys_out = reinterpret_cast<T*>(AffineAccessor<char,1>(shared_instance, 0).base) + num_intermediate;

    RegionInstance break_points_instance = this->realm_malloc(num_intermediate * sizeof(uint8_t), my_mem);
    uint8_t* break_points = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(break_points_instance, 0).base);

    size_t* group_ids = reinterpret_cast<size_t*>(AffineAccessor<char,1>(shared_instance, 0).base);

    //Now that we have disjoint rectangles, we can do our usual sort and coalesce pass
    size_t last = INT_MAX;
    {
      NVTX_DEPPART(compact_disjoint_rects);
      while (last > num_intermediate) {
        last = num_intermediate;

        bool done = false;
        for (int dim = 1; !done; dim++) {
          if (dim == N) {
            dim = 0; // wrap around to 0
            done = true;
          }
          grid_size = (num_intermediate + threads_per_block - 1) / threads_per_block;

          build_lo_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_rects_in, num_intermediate, dim);
          KERNEL_CHECK(stream);
          size_t temp_bytes;
          cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                              d_coord_keys_in, d_coord_keys_out,
                                              d_rects_in, d_rects_out,
                                              num_intermediate, 0, 8*sizeof(T), stream);
          if (temp_bytes > orig_tmp) {
            tmp_instance.destroy();
            tmp_instance = this->realm_malloc(temp_bytes, my_mem);
            orig_tmp = temp_bytes;
            tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
          }
          cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                          d_coord_keys_in, d_coord_keys_out,
                                                          d_rects_in, d_rects_out,
                                                          num_intermediate, 0, 8*sizeof(T), stream);

          std::swap(d_rects_in, d_rects_out);
          for (int d = 0; d < N; d++) {
            if (d == dim) {
              continue;
            }
            build_hi_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_rects_in, num_intermediate, d);
            KERNEL_CHECK(stream);
            size_t temp_bytes;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                                d_coord_keys_in, d_coord_keys_out,
                                                d_rects_in, d_rects_out,
                                                num_intermediate, 0, 8*sizeof(T), stream);
            if (temp_bytes > orig_tmp) {
              tmp_instance.destroy();
              tmp_instance = this->realm_malloc(temp_bytes, my_mem);
              orig_tmp = temp_bytes;
              tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
            }
            cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                            d_coord_keys_in, d_coord_keys_out,
                                                            d_rects_in, d_rects_out,
                                                            num_intermediate, 0, 8*sizeof(T), stream);

            std::swap(d_rects_in, d_rects_out);
            build_lo_key<<<grid_size, threads_per_block, 0, stream>>>(d_coord_keys_in, d_rects_in, num_intermediate, d);
            KERNEL_CHECK(stream);
            cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                                d_coord_keys_in, d_coord_keys_out,
                                                d_rects_in, d_rects_out,
                                                num_intermediate, 0, 8*sizeof(T), stream);
            if (temp_bytes > orig_tmp) {
              tmp_instance.destroy();
              tmp_instance = this->realm_malloc(temp_bytes, my_mem);
              orig_tmp = temp_bytes;
              tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
            }
            cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                            d_coord_keys_in, d_coord_keys_out,
                                                            d_rects_in, d_rects_out,
                                                            num_intermediate, 0, 8*sizeof(T), stream);

            std::swap(d_rects_in, d_rects_out);

          }

          build_src_key<<<grid_size, threads_per_block, 0, stream>>>(d_src_keys_in, d_rects_in, num_intermediate);
          KERNEL_CHECK(stream);
          cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                                  d_src_keys_in, d_src_keys_out,
                                                  d_rects_in, d_rects_out,
                                                  num_intermediate, 0, 8*sizeof(size_t), stream);
          if (temp_bytes > orig_tmp) {
            tmp_instance.destroy();
            tmp_instance = this->realm_malloc(temp_bytes, my_mem);
            orig_tmp = temp_bytes;
            tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
          }
          cub::DeviceRadixSort::SortPairs(tmp_storage, temp_bytes,
                                                          d_src_keys_in, d_src_keys_out,
                                                          d_rects_in, d_rects_out,
                                                          num_intermediate, 0, 8*sizeof(size_t), stream);

          std::swap(d_rects_in, d_rects_out);

          mark_breaks_dim<<<grid_size, threads_per_block, 0, stream>>>(d_rects_in, break_points, num_intermediate, dim);
          KERNEL_CHECK(stream);

          cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, break_points, group_ids, num_intermediate, stream);

          if (temp_bytes > orig_tmp) {
            tmp_instance.destroy();
            tmp_instance = this->realm_malloc(temp_bytes, my_mem);
            orig_tmp = temp_bytes;
            tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
          }

          cub::DeviceScan::InclusiveSum(tmp_storage, temp_bytes, break_points, group_ids, num_intermediate, stream);

          size_t last_grp;
          CUDA_CHECK(cudaMemcpyAsync(&last_grp, &group_ids[num_intermediate-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);

          init_rects_dim<<<grid_size, threads_per_block, 0, stream>>>(d_rects_in, break_points, group_ids, d_rects_out, num_intermediate, dim);
          KERNEL_CHECK(stream);

          num_intermediate = last_grp;
          std::swap(d_rects_in, d_rects_out);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }

    heads_instance.destroy();
    shared_instance.destroy();
    tmp_instance.destroy();

    //And... we're done
    if (out_rects > 0) {
      d_out_rects = d_rects_in;
      out_rects = num_intermediate;
    } else {
      this->send_output(d_rects_in, num_intermediate, my_arena, ctr, getIndex, getMap);
      rects_out_instance.destroy();
    }

  }

  /*
 *  Input: An array of 1D rectangles (potentially overlapping) with associated
 *  src indices, where all the rectangles with a given src idx together represent an exact covering
 *  of the partitioning output for that index.
 *  Output: A disjoint, coalesced array of rectangles sorted by src idx that it then sends off
 *  to the send output function, which constructs the final sparsity map.
 *  Approach: The canonical 1D rectangle merge, in parallel. Sort the rectangles by (src_idx, lo). Then
 *  prefix max by hi segmented by src_idx to find overlapping rectangles. Then, RLE by starting a new rectangle
 *  when in a new src or lo > current max hi and merging otherwise.
 */
  template<int N, typename T>
  template<typename Container, typename IndexFn, typename MapFn>
  void GPUMicroOp<N,T>::complete1d_pipeline(RectDesc<N, T>* d_rects, size_t total_rects, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {

    NVTX_DEPPART(complete1d_pipeline);
    cudaStream_t stream = Cuda::get_task_cuda_stream();

    RectDesc<N,T>* d_rects_in = d_rects;

    size_t bytes_T   = total_rects * sizeof(T);
    size_t bytes_S   = total_rects * sizeof(size_t);
    size_t bytes_HF  = total_rects * sizeof(HiFlag<T>);
    size_t max_bytes = std::max({bytes_T, bytes_HF, bytes_S});

    char* aux_ptr = my_arena.alloc<char>(2 * max_bytes);

    uint8_t* break_points = my_arena.alloc<uint8_t>(total_rects);
    size_t* group_ids = my_arena.alloc<size_t>(total_rects);

    T* d_keys_in = reinterpret_cast<T*>(aux_ptr);
    T* d_keys_out = reinterpret_cast<T*>(aux_ptr + max_bytes);

    size_t* d_src_keys_in = reinterpret_cast<size_t*>(aux_ptr);
    size_t* d_src_keys_out = reinterpret_cast<size_t*>(aux_ptr + max_bytes);

    HiFlag<T>* d_hi_flags_in = reinterpret_cast<HiFlag<T>*>(aux_ptr);
    HiFlag<T>* d_hi_flags_out = reinterpret_cast<HiFlag<T>*>(aux_ptr + max_bytes);

    size_t num_intermediate = total_rects;

    const size_t prev = my_arena.mark();
    RectDesc<N, T>* d_rects_out = my_arena.alloc<RectDesc<N, T>>(total_rects);

    size_t t1=0, t2 = 0, t3 = 0, t4 = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, t1,
    d_keys_in, d_keys_out, d_rects_in, d_rects_out, num_intermediate,
    0, 8*sizeof(T), stream);
    // exclusive scan
    cub::DeviceScan::ExclusiveScan(nullptr, t2,
      d_hi_flags_in, d_hi_flags_out,
      SegmentedMax<T>(), HiFlag<T>{std::numeric_limits<T>::min(), 0},
      num_intermediate, stream);
    // inclusive sum
    cub::DeviceScan::InclusiveSum(nullptr, t3,
      break_points, group_ids,
      num_intermediate, stream);

    cub::DeviceRadixSort::SortPairs(nullptr, t4, d_src_keys_in, d_src_keys_out, d_rects_in, d_rects_out, num_intermediate, 0, 8*sizeof(size_t), stream);

    size_t temp_bytes = std::max({t1, t2, t3, t4});
    size_t use_bytes = temp_bytes;
    void *temp_storage = my_arena.alloc<char>(temp_bytes);

    int threads_per_block = 256;
    size_t grid_size = (num_intermediate + threads_per_block - 1) / threads_per_block;

    //Sort the rectangles keyed by (src, lo)
    {
      NVTX_DEPPART(sort_rects);

      build_lo_key<<<grid_size, threads_per_block, 0, stream>>>(d_keys_in, d_rects_in, num_intermediate, 0);
      KERNEL_CHECK(stream);
      cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_keys_in, d_keys_out, d_rects_in, d_rects_out, num_intermediate, 0, 8*sizeof(T), stream);
      std::swap(d_rects_in, d_rects_out);

      build_src_key<<<grid_size, threads_per_block, 0, stream>>>(d_src_keys_in, d_rects_in, num_intermediate);
      KERNEL_CHECK(stream);

      use_bytes = temp_bytes;
      cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_src_keys_in, d_src_keys_out, d_rects_in, d_rects_out, num_intermediate, 0, 8*sizeof(size_t), stream);
      std::swap(d_rects_in, d_rects_out);
    }

    //Prefix max by hi segmented by src, then RLE to merge.
    {
      NVTX_DEPPART(run_length_encode);
      build_hi_flag<<<grid_size, threads_per_block, 0, stream>>>(d_hi_flags_in, d_rects_in, num_intermediate, 0);
      KERNEL_CHECK(stream);


      use_bytes = temp_bytes;
      cub::DeviceScan::ExclusiveScan(
        /*d_temp=*/    temp_storage,
        /*bytes=*/     use_bytes,
        /*in=*/        d_hi_flags_in,
        /*out=*/       d_hi_flags_out,
        /*op=*/        SegmentedMax<T>(),
                       HiFlag<T>{std::numeric_limits<T>::min(), 0},
        /*num_items=*/ num_intermediate,
        /*stream=*/    stream
      );

      threads_per_block = 256;
      grid_size = (num_intermediate + threads_per_block - 1) / threads_per_block;
      mark_breaks_dim<<<grid_size, threads_per_block, 0, stream>>>(d_hi_flags_in, d_hi_flags_out, d_rects_in, break_points, num_intermediate, 0);
      KERNEL_CHECK(stream);
      use_bytes = temp_bytes;
      cub::DeviceScan::InclusiveSum(temp_storage, use_bytes, break_points, group_ids, num_intermediate, stream);

      size_t last_grp;
      CUDA_CHECK(cudaMemcpyAsync(&last_grp, &group_ids[num_intermediate-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      my_arena.rollback(prev);
      my_arena.flip_parity();
      assert(my_arena.get_parity());
      my_arena.reset(true);
      d_rects_out = my_arena.alloc<RectDesc<N,T>>(last_grp);
      my_arena.commit(true);

      init_rects_dim<<<grid_size, threads_per_block, 0, stream>>>(d_rects_in, d_hi_flags_out, break_points, group_ids, d_rects_out, num_intermediate, 0);
      KERNEL_CHECK(stream);

      num_intermediate = last_grp;
      std::swap(d_rects_in, d_rects_out);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }

    if (out_rects > 0) {
      d_out_rects = d_rects_in;
      out_rects = num_intermediate;
    } else {
      this->send_output(d_rects_in, num_intermediate, my_arena, ctr, getIndex, getMap);
    }
  }

   /*
  *  Input: An array of points (potentially with duplicates) with associated
  *  src indices, where all the points with a given src idx together represent an exact covering
  *  of the partitioning output for that index.
  *  Output: A disjoint, coalesced array of rectangles sorted by src idx that it then sends off
  *  to the send output function, which constructs the final sparsity map.
  *  Approach: Sort the points by (x0,x1,...,xN-1,src) (right is MSB). Convert them to singleton rects.
  *  Run-length encode along each dimension (N-1...0).
  */
  template<int N, typename T>
  template<typename Container, typename IndexFn, typename MapFn>
  void GPUMicroOp<N,T>::complete_pipeline(PointDesc<N, T>* d_points, size_t total_pts, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {

    NVTX_DEPPART(complete_pipeline);

    size_t prev = my_arena.mark();

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    size_t bytes_T   = total_pts * sizeof(T);
    size_t bytes_S   = total_pts * sizeof(size_t);
    size_t bytes_R  =  total_pts * sizeof(RectDesc<N,T>);
    size_t bytes_p = total_pts * sizeof(PointDesc<N,T>);
    size_t max_aux_bytes = std::max({bytes_T, bytes_S, bytes_R});
    size_t max_pg_bytes = std::max({bytes_p, bytes_S});


    // Instance shared by coordinate keys, source keys, and rectangle outputs
    char* aux_ptr = my_arena.alloc<char>(2 * max_aux_bytes);

    //Instance shared by group ids (RLE) and intermediate points in sorting
    char* pg_ptr = my_arena.alloc<char>(max_pg_bytes);

    uint8_t* break_points = my_arena.alloc<uint8_t>(total_pts);

    T* d_keys_in = reinterpret_cast<T*>(aux_ptr);
    T* d_keys_out = reinterpret_cast<T*>(aux_ptr + max_aux_bytes);

    PointDesc<N,T>* d_points_in = d_points;
    PointDesc<N,T>* d_points_out = reinterpret_cast<PointDesc<N,T>*>(pg_ptr);

    size_t* group_ids = reinterpret_cast<size_t*>(pg_ptr);

    RectDesc<N,T>* d_rects_in = reinterpret_cast<RectDesc<N,T>*>(aux_ptr);
    RectDesc<N, T> *d_rects_out = reinterpret_cast<RectDesc<N,T>*>(aux_ptr + max_aux_bytes);

    size_t* d_src_keys_in = reinterpret_cast<size_t*>(aux_ptr);
    size_t* d_src_keys_out = reinterpret_cast<size_t*>(aux_ptr + max_aux_bytes);

    size_t t1=0, t2=0, t3=0;
    cub::DeviceRadixSort::SortPairs(nullptr, t1, d_keys_in, d_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(T), stream);
    cub::DeviceRadixSort::SortPairs(nullptr, t2, d_src_keys_in, d_src_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(size_t), stream);
    cub::DeviceScan::InclusiveSum(nullptr, t3, break_points, group_ids, total_pts, stream);

    //Temporary storage instance shared by CUB operations.
    size_t temp_bytes = std::max({t1, t2, t3});
    void *temp_storage = my_arena.alloc<char>(temp_bytes);


    //Sort along each dimension from LSB to MSB (0 to N-1)
    size_t use_bytes = temp_bytes;

    {
      NVTX_DEPPART(sort_valid_points);
      for (int dim = 0; dim < N; ++dim) {
        build_coord_key<<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_keys_in, d_points_in, total_pts, dim);
        KERNEL_CHECK(stream);
        cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_keys_in, d_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(T), stream);
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_points_in, d_points_out);
      }

      //Sort by source index now to keep individual partitions separate
      build_src_key<<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_src_keys_in, d_points_in, total_pts);
      KERNEL_CHECK(stream);
      use_bytes = temp_bytes;
      cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_src_keys_in, d_src_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(size_t), stream);
    }


    points_to_rects<<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_points_out, d_rects_in, total_pts);
    KERNEL_CHECK(stream);

    size_t num_intermediate = total_pts;

    {
      NVTX_DEPPART(run_length_encode);

      for (int dim = N-1; dim >= 0; --dim) {

        // Step 1: Mark rectangle starts
        // e.g. [1, 2, 4, 5, 6, 8] -> [1, 0, 1, 0, 0, 1]
        mark_breaks_dim<<<COMPUTE_GRID(num_intermediate), THREADS_PER_BLOCK, 0, stream>>>(d_rects_in, break_points, num_intermediate, dim);
        KERNEL_CHECK(stream);

        // Step 2: Inclusive scan of break points to get group ids
        // e.g. [1, 0, 1, 0, 0, 1] -> [1, 1, 2, 2, 2, 3]
        use_bytes = temp_bytes;
        cub::DeviceScan::InclusiveSum(temp_storage, use_bytes, break_points, group_ids, num_intermediate, stream);

        //Determine new number of intermediate rectangles
        size_t last_grp;
        CUDA_CHECK(cudaMemcpyAsync(&last_grp, &group_ids[num_intermediate-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        //Step 3: Write output rectangles, where rect starts write lo and rect ends write hi
        init_rects_dim<<<COMPUTE_GRID(num_intermediate), THREADS_PER_BLOCK, 0, stream>>>(d_rects_in, break_points, group_ids, d_rects_out, num_intermediate, dim);
        KERNEL_CHECK(stream);

        num_intermediate = last_grp;
        std::swap(d_rects_in, d_rects_out);
      }
      my_arena.rollback(prev);
      d_out_rects = my_arena.alloc<RectDesc<N, T>>(num_intermediate);
      CUDA_CHECK(cudaMemcpyAsync(d_out_rects, d_rects_in, num_intermediate * sizeof(RectDesc<N,T>), cudaMemcpyDeviceToDevice, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }

    if (out_rects==1) {
      out_rects = num_intermediate;
    } else {
      this->send_output(d_rects_in, num_intermediate, my_arena, ctr, getIndex, getMap);
      my_arena.rollback(prev);
    }
  }

  /*
   *  Input: An array of disjoint rectangles sorted by src idx.
   *  Output: Fills the sparsity output for each src with a host region instance
   *  containing the entries/approx entries and calls gpu_finalize on the SparsityMapImpl.
   *  Approach: Segments the rectangles by their src idx and copies them back to the host,
   */

  template<int N, typename T>
  template<typename Container, typename IndexFn, typename MapFn>
  void GPUMicroOp<N,T>::send_output(RectDesc<N, T>* d_rects, size_t total_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {
    NVTX_DEPPART(send_output);

    size_t prev = my_arena.mark();

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    std::set<Event> output_allocs;

    SparsityMapEntry<N,T>* final_entries = my_arena.alloc<SparsityMapEntry<N,T>>(total_rects);
    Rect<N,T>* final_rects = my_arena.alloc<Rect<N,T>>(total_rects);

    size_t* d_starts = my_arena.alloc<size_t>(2 * ctr.size());
    size_t* d_ends = d_starts + ctr.size();

    CUDA_CHECK(cudaMemsetAsync(d_starts, 0, ctr.size()*sizeof(size_t),stream), stream);
    CUDA_CHECK(cudaMemsetAsync(d_ends, 0, ctr.size()*sizeof(size_t),stream), stream);


    //Convert RectDesc to SparsityMapEntry and determine where each src's rectangles start and end.
    build_final_output<<<COMPUTE_GRID(total_rects), THREADS_PER_BLOCK, 0, stream>>>(d_rects, final_entries, final_rects, d_starts, d_ends, total_rects);
    KERNEL_CHECK(stream);


    //Copy starts and ends back to host and handle empty partitions
    std::vector<size_t> d_starts_host(ctr.size()), d_ends_host(ctr.size());
    CUDA_CHECK(cudaMemcpyAsync(d_starts_host.data(), d_starts, ctr.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(d_ends_host.data(), d_ends, ctr.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 1; i < ctr.size(); i++) {
      if (d_starts_host[i] < d_ends_host[i-1]) {
        d_starts_host[i] = d_ends_host[i-1];
        d_ends_host[i] = d_ends_host[i-1];
      }
    }

    if (!this->exclusive) {
      for (auto const& elem : ctr) {
        size_t idx = getIndex(elem);
        auto mapOpj = getMap(elem);
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(mapOpj);
        if (d_ends_host[idx] > d_starts_host[idx]) {
          size_t end = d_ends_host[idx];
          size_t start = d_starts_host[idx];
          std::vector<Rect<N, T>> h_rects(end - start);
          CUDA_CHECK(cudaMemcpyAsync(h_rects.data(), final_rects + start, (end - start) * sizeof(Rect<N,T>), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          impl->contribute_dense_rect_list(h_rects, true);
        } else {
          impl->contribute_nothing();
        }
      }
    } else {
      Memory sysmem;
      assert(find_memory(sysmem, Memory::SYSTEM_MEM));

      //Use provided lambdas to iterate over sparsity output container (map or vector)
      for (auto const& elem : ctr) {
        size_t idx = getIndex(elem);
        auto mapOpj = getMap(elem);
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(mapOpj);
        if (d_ends_host[idx] > d_starts_host[idx]) {
          size_t end = d_ends_host[idx];
          size_t start = d_starts_host[idx];
          RegionInstance entries = this->realm_malloc((end - start) * sizeof(SparsityMapEntry<N,T>), sysmem);
          SparsityMapEntry<N, T> *h_entries = reinterpret_cast<SparsityMapEntry<N, T> *>(AffineAccessor<char, 1>(entries, 0).base);
          CUDA_CHECK(cudaMemcpyAsync(h_entries, final_entries + start, (end - start) * sizeof(SparsityMapEntry<N,T>), cudaMemcpyDeviceToHost, stream), stream);

          Rect<N,T> *approx_rects;
          size_t num_approx;
          if (end - start <= ((size_t) DeppartConfig::cfg_max_rects_in_approximation)) {
            approx_rects = final_rects + start;
            num_approx = end - start;
          } else {
            //TODO: Maybe add a better GPU approx here when given more rectangles
            //Use CUB to compute a bad approx on the GPU (union of all rectangles)
            approx_rects = my_arena.alloc<Rect<N,T>>(1);
            num_approx = 1;
            void*  d_temp   = nullptr;
            size_t temp_sz  = 0;
            Rect<N, T> identity_rect;
            for(int d=0; d<N; d++){
              identity_rect.lo[d] =  std::numeric_limits<T>::max();
              identity_rect.hi[d] =  std::numeric_limits<T>::min();
            }
            cub::DeviceReduce::Reduce(
              d_temp, temp_sz,
              final_rects + start,
              approx_rects,
              (end - start),
              UnionRectOp<N,T>(),
              identity_rect,
              stream
            );
            d_temp = reinterpret_cast<void*>(my_arena.alloc<char>(temp_sz));
            cub::DeviceReduce::Reduce(
              d_temp, temp_sz,
              final_rects + start,
              approx_rects,
              end - start,
              UnionRectOp<N,T>(),
              identity_rect,
              stream
            );
            CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          }
          RegionInstance approx_entries = this->realm_malloc(num_approx * sizeof(Rect<N,T>), sysmem);
          SparsityMapEntry<N, T> *h_approx_entries = reinterpret_cast<SparsityMapEntry<N, T> *>(AffineAccessor<char, 1>(approx_entries, 0).base);
          CUDA_CHECK(cudaMemcpyAsync(h_approx_entries, approx_rects, num_approx * sizeof(Rect<N,T>), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          impl->set_instance(entries, end - start);
          impl->set_approx_instance(approx_entries, num_approx);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (auto const& elem : ctr) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(getMap(elem));
        impl->gpu_finalize();
      }
    }
    my_arena.rollback(prev);
  }


}