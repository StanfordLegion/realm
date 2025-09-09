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

  //Given a list of instances, compacts them all into one collapsed_space
  template<int N, typename T>
  template<typename FT>
  void GPUMicroOp<N,T>::collapse_inst_space(const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT> >& field_data, RegionInstance& out_instance, collapsed_space<N, T> &out_space, Memory my_mem, cudaStream_t stream)
  {
    // We need inst_offsets to preserve which instance each rectangle came from
    // so that we know where to read from after intersecting.
    std::vector<size_t> inst_offsets(field_data.size() + 1);

    // Determine size of allocation for combined inst_rects.
    out_space.num_entries = 0;

    for (size_t i = 0; i < field_data.size(); ++i) {
      inst_offsets[i] = out_space.num_entries;
      if (field_data[i].index_space.dense()) {
        out_space.num_entries += 1;
      } else {
        out_space.num_entries += field_data[i].index_space.sparsity.impl()->get_entries().size();
      }
    }
    inst_offsets[field_data.size()] = out_space.num_entries;

    //We copy into one contiguous host buffer, then copy to device
    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM));

    RegionInstance h_instance = realm_malloc(out_space.num_entries * sizeof(SparsityMapEntry<N,T>), sysmem);
    SparsityMapEntry<N, T>* h_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(h_instance, 0).base);

    out_instance = realm_malloc(out_space.num_entries * sizeof(SparsityMapEntry<N,T>), my_mem);
    out_space.entries_buffer = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);

    //Now we fill the host array with all instance rectangles
    size_t inst_pos = 0;
    for (size_t i = 0; i < field_data.size(); ++i) {
      const IndexSpace<N,T> &inst_space = field_data[i].index_space;
      if (inst_space.dense()) {
        SparsityMapEntry<N,T> entry;
        entry.bounds = inst_space.bounds;
        memcpy(h_entries + inst_pos, &entry, sizeof(SparsityMapEntry<N,T>));
        ++inst_pos;
      } else {
        span<SparsityMapEntry<N, T>> tmp = inst_space.sparsity.impl()->get_entries();
        memcpy(h_entries + inst_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>));
        inst_pos += tmp.size();
      }
    }

    //Now we copy our entries and offsets to the device
    CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, h_entries, out_space.num_entries * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(out_space.offsets, inst_offsets.data(), (field_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    h_instance.destroy();
  }

  // Only real work here is getting dense/sparse into a single collapsed_space.
  template<int N, typename T>
  void GPUMicroOp<N,T>::collapse_parent_space(const IndexSpace<N, T>& parent_space, RegionInstance& out_instance, collapsed_space<N, T> &out_space, Memory my_mem, cudaStream_t stream)
  {
    if (parent_space.dense()) {
      SparsityMapEntry<N,T> entry;
      entry.bounds = parent_space.bounds;
      out_instance = realm_malloc(sizeof(SparsityMapEntry<N,T>), my_mem);
      out_space.entries_buffer = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);
      out_space.num_entries = 1;
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    } else {
      span<SparsityMapEntry<N, T>> tmp =  parent_space.sparsity.impl()->get_entries();
      out_space.num_entries = tmp.size();
      out_instance = realm_malloc(tmp.size() * sizeof(SparsityMapEntry<N,T>), my_mem);
      out_space.entries_buffer = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);
      out_space.bounds = parent_space.bounds;
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    }
    out_space.offsets = nullptr;
    out_space.num_children = 1;
  }

  // Given a collapsed space, builds a (potentially marked) bvh over that space.
  // Based on Tero Karras' Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
  template<int N, typename T>
  void GPUMicroOp<N, T>::build_bvh(const collapsed_space<N, T> &space, RegionInstance &bvh_instance, BVH<N, T> &result, Memory my_mem, cudaStream_t stream)
  {

      // Bounds used for morton code computation.
      RegionInstance global_bounds_instance = realm_malloc(sizeof(Rect<N,T>), my_mem);
      Rect<N,T>* d_global_bounds = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(global_bounds_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_global_bounds, &space.bounds, sizeof(Rect<N,T>), cudaMemcpyHostToDevice, stream),
             stream);

      //We want to keep the entire BVH that we return in one instance for convenience.
      size_t indices_instance_size = space.num_entries * sizeof(uint64_t);
      size_t labels_instance_size = space.offsets == nullptr ? 0 : space.num_entries * sizeof(size_t);
      size_t boxes_instance_size =  (2*space.num_entries - 1) * sizeof(Rect<N, T>);
      size_t child_instance_size = (2*space.num_entries - 1) * sizeof(int);

      size_t total_instance_size = indices_instance_size + labels_instance_size + boxes_instance_size + 2 * child_instance_size;
      bvh_instance = realm_malloc(total_instance_size, my_mem);

      size_t curr_idx = 0;
      result.indices = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(bvh_instance, 0).base + curr_idx);
      curr_idx += indices_instance_size;
      result.labels = space.offsets == nullptr ? nullptr : reinterpret_cast<size_t*>(AffineAccessor<char,1>(bvh_instance, 0).base + curr_idx);
      curr_idx += labels_instance_size;
      result.boxes = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(bvh_instance, 0).base + curr_idx);
      curr_idx += boxes_instance_size;
      result.childLeft = reinterpret_cast<int*>(AffineAccessor<char,1>(bvh_instance, 0).base + curr_idx);
      curr_idx += child_instance_size;
      result.childRight = reinterpret_cast<int*>(AffineAccessor<char,1>(bvh_instance, 0).base + curr_idx);

      // These are intermediate instances we'll destroy before returning.
      RegionInstance morton_visit_instance = realm_malloc(2 * space.num_entries * max(sizeof(uint64_t), sizeof(int)), my_mem);
      uint64_t* d_morton_codes = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_visit_instance, 0).base);

      RegionInstance indices_tmp_instance = realm_malloc(space.num_entries * sizeof(uint64_t), my_mem);
      uint64_t* d_indices_in = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_tmp_instance, 0).base);

      // We compute morton codes for each leaf and sort, labeling if necessary.
      bvh_build_morton_codes<N, T><<<COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.entries_buffer, space.offsets, d_global_bounds, space.num_entries, space.num_children, d_morton_codes, d_indices_in, result.labels);
      KERNEL_CHECK(stream);

      uint64_t* d_morton_codes_out = d_morton_codes + space.num_entries;
      uint64_t* d_indices_out = result.indices;

      void *bvh_temp = nullptr;
      size_t bvh_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices_in,
                                      d_indices_out, space.num_entries, 0, 64, stream);
      RegionInstance bvh_temp_instance = realm_malloc(bvh_temp_bytes, my_mem);
      bvh_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(bvh_temp_instance, 0).base);
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices_in,
                                      d_indices_out, space.num_entries, 0, 64, stream);

      std::swap(d_morton_codes, d_morton_codes_out);


      // Another temporary instance.
      RegionInstance parent_instance = realm_malloc((2*space.num_entries - 1) * sizeof(int), my_mem);
      int* d_parent = reinterpret_cast<int*>(AffineAccessor<char,1>(parent_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_parent, -1, (2*space.num_entries - 1) * sizeof(int), stream), stream);

      // Here's where we actually build the BVH
      int n = (int) space.num_entries;
      bvh_build_radix_tree_kernel<<< COMPUTE_GRID(space.num_entries - 1), THREADS_PER_BLOCK, 0, stream>>>(d_morton_codes, result.indices, n, result.childLeft, result.childRight, d_parent);
      KERNEL_CHECK(stream);

      // Figure out which node didn't get its parent set.
      RegionInstance root_instance = realm_malloc(sizeof(int), my_mem);
      int* d_root = reinterpret_cast<int*>(AffineAccessor<char,1>(root_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_root, -1, sizeof(int), stream), stream);

      bvh_build_root_kernel<<< COMPUTE_GRID(2 * space.num_entries - 1), THREADS_PER_BLOCK, 0, stream>>>(d_root, d_parent, space.num_entries);
      KERNEL_CHECK(stream);

      CUDA_CHECK(cudaMemcpyAsync(&result.root, d_root, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      // Now we materialize the tree into something the client can query.
      bvh_init_leaf_boxes_kernel<N, T><<<COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.entries_buffer, result.indices, space.num_entries, result.boxes);
      KERNEL_CHECK(stream);

      int* d_visitCount = reinterpret_cast<int*>(AffineAccessor<char,1>(morton_visit_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_visitCount, 0, (2*space.num_entries - 1) * sizeof(int), stream), stream);

      bvh_merge_internal_boxes_kernel < N, T ><<< COMPUTE_GRID(space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(space.num_entries, result.childLeft, result.childRight, d_parent, result.boxes, d_visitCount);
      KERNEL_CHECK(stream);

      // Cleanup.
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      root_instance.destroy();
      parent_instance.destroy();
      bvh_temp_instance.destroy();
      morton_visit_instance.destroy();
      indices_tmp_instance.destroy();
      global_bounds_instance.destroy();

  }

  // Intersects two collapsed spaces, where lhs is always instances and rhs is either parent or soruces/targets.
  // If rhs is sources/targets, we mark the intersected rectangles by where they came from.
  // If the intersection is costly, we accelerate with a BVH.
  template<int N, typename T>
  template<typename out_t>
  void GPUMicroOp<N,T>::construct_input_rectlist(const collapsed_space<N, T> &lhs, const collapsed_space<N, T> &rhs, RegionInstance &out_instance,  size_t& out_size, uint32_t* counters, uint32_t* out_offsets, Memory my_mem, cudaStream_t stream)
  {

    CUDA_CHECK(cudaMemsetAsync(counters, 0, (lhs.num_children) * sizeof(uint32_t), stream), stream);

    BVH<N, T> my_bvh;
    RegionInstance bvh_instance;
    bool bvh_valid = rhs.num_children > rhs.num_entries;
    if (bvh_valid) {
      build_bvh(rhs, bvh_instance, my_bvh, my_mem, stream);
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

    out_instance = realm_malloc(out_size * sizeof(out_t), my_mem);

    // Where each instance should start writing its rectangles.
    CUDA_CHECK(cudaMemcpyAsync(out_offsets, h_inst_counters.data(), (lhs.num_children + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

    // Non-empty rectangles from the intersection.
    out_t* d_valid_rects = reinterpret_cast<out_t*>(AffineAccessor<char,1>(out_instance, 0).base);

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
    if (bvh_valid) {
      bvh_instance.destroy();
    }
  }

  // Prefix sum an array of Rects or RectDescs by volume.
  template<int N, typename T>
  template<typename out_t>
  void GPUMicroOp<N, T>::volume_prefix_sum(const out_t* d_rects, size_t total_rects, RegionInstance &out_instance, size_t& num_pts, Memory my_mem, cudaStream_t stream)
  {
    out_instance = realm_malloc((total_rects + 1) * sizeof(size_t), my_mem);
    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(out_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_prefix_rects, 0, sizeof(size_t), stream), stream);

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

    RegionInstance temp_instance = realm_malloc(rect_temp_bytes, my_mem);
    d_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_instance, 0).base);
    cub::DeviceScan::InclusiveSum(
        /* d_temp_storage */  d_temp,
        /* temp_bytes */      rect_temp_bytes,
        /* d_in */            d_volumes,
        /* d_out */           d_prefix_rects + 1,
        /* num_items */       total_rects, stream);


    //Number of points across all rectangles (also our total output count).
    CUDA_CHECK(cudaMemcpyAsync(&num_pts, &d_prefix_rects[total_rects], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    temp_instance.destroy();
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
  void GPUMicroOp<N,T>::complete_pipeline(PointDesc<N, T>* d_points, size_t total_pts, Memory my_mem, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {

    NVTX_DEPPART(complete_pipeline);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    size_t bytes_T   = total_pts * sizeof(T);
    size_t bytes_S   = total_pts * sizeof(size_t);
    size_t bytes_R  =  total_pts * sizeof(RectDesc<N,T>);
    size_t bytes_p = total_pts * sizeof(PointDesc<N,T>);
    size_t max_aux_bytes = std::max({bytes_T, bytes_S, bytes_R});
    size_t max_pg_bytes = std::max({bytes_p, bytes_S});


    // Instance shared by coordinate keys, source keys, and rectangle outputs
    RegionInstance aux_instance = this->realm_malloc(2 * max_aux_bytes, my_mem);

    //Instance shared by group ids (RLE) and intermediate points in sorting
    RegionInstance pg_instance = this->realm_malloc(max_pg_bytes, my_mem);

    RegionInstance break_points_instance = this->realm_malloc(total_pts * sizeof(uint8_t), my_mem);

    const uintptr_t aux_in = AffineAccessor<char,1>(aux_instance, 0).base;
    const uintptr_t aux_out = aux_in + max_aux_bytes;

    T* d_keys_in = reinterpret_cast<T*>(aux_in);
    T* d_keys_out = reinterpret_cast<T*>(aux_out);

    PointDesc<N,T>* d_points_in = d_points;
    PointDesc<N,T>* d_points_out = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(pg_instance, 0).base);

    uint8_t* break_points = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(break_points_instance, 0).base);

    size_t* group_ids = reinterpret_cast<size_t*>(AffineAccessor<char,1>(pg_instance, 0).base);

    RectDesc<N,T>* d_rects_in = reinterpret_cast<RectDesc<N,T>*>(aux_in);
    RectDesc<N, T> *d_rects_out = reinterpret_cast<RectDesc<N,T>*>(aux_out);

    size_t* d_src_keys_in = reinterpret_cast<size_t*>(aux_in);
    size_t* d_src_keys_out = reinterpret_cast<size_t*>(aux_out);

    size_t t1=0, t2=0, t3=0;
    cub::DeviceRadixSort::SortPairs(nullptr, t1, d_keys_in, d_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(T), stream);
    cub::DeviceRadixSort::SortPairs(nullptr, t2, d_src_keys_in, d_src_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(size_t), stream);
    cub::DeviceScan::InclusiveSum(nullptr, t3, break_points, group_ids, total_pts, stream);

    //Temporary storage instance shared by CUB operations.
    size_t temp_bytes = std::max({t1, t2, t3});
    RegionInstance temp_storage_instance = this->realm_malloc(temp_bytes, my_mem);
    void *temp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_storage_instance, 0).base);


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

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      pg_instance.destroy();
      temp_storage_instance.destroy();
      break_points_instance.destroy();
    }

    this->send_output(d_rects_in, num_intermediate, my_mem, ctr, getIndex, getMap);

    aux_instance.destroy();
  }

  /*
   *  Input: An array of disjoint rectangles sorted by src idx.
   *  Output: Fills the sparsity output for each src with a host region instance
   *  containing the entries/approx entries and calls gpu_finalize on the SparsityMapImpl.
   *  Approach: Segments the rectangles by their src idx and copies them back to the host,
   */

  template<int N, typename T>
  template<typename Container, typename IndexFn, typename MapFn>
  void GPUMicroOp<N,T>::send_output(RectDesc<N, T>* d_rects, size_t total_rects, Memory my_mem, const Container& ctr, IndexFn getIndex, MapFn getMap)
  {
    NVTX_DEPPART(send_output);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    std::set<Event> output_allocs;

    RegionInstance final_entries_instance = this->realm_malloc(total_rects * sizeof(SparsityMapEntry<N,T>), my_mem);

    RegionInstance final_rects_instance = this->realm_malloc(total_rects * sizeof(Rect<N,T>), my_mem);

    RegionInstance boundaries_instance = this->realm_malloc(2 * ctr.size() * sizeof(size_t), my_mem);

    SparsityMapEntry<N,T>* final_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(final_entries_instance, 0).base);
    Rect<N,T>* final_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(final_rects_instance, 0).base);

    size_t* d_starts = reinterpret_cast<size_t*>(AffineAccessor<char,1>(boundaries_instance, 0).base);
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
        RegionInstance approx_rects_instance;
        size_t num_approx;
        if (end - start <= ((size_t) DeppartConfig::cfg_max_rects_in_approximation)) {
          approx_rects = final_rects + start;
          num_approx = end - start;
        } else {
          //TODO: Maybe add a better GPU approx here when given more rectangles
          //Use CUB to compute a bad approx on the GPU (union of all rectangles)
          approx_rects_instance = this->realm_malloc(sizeof(Rect<N,T>), my_mem);
          approx_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(approx_rects_instance, 0).base);
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
          RegionInstance temp_rects_instance = this->realm_malloc(temp_sz * sizeof(Rect<N,T>), my_mem);
          d_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_rects_instance, 0).base);
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
          temp_rects_instance.destroy();
        }
        RegionInstance approx_entries = this->realm_malloc(num_approx * sizeof(Rect<N,T>), sysmem);
        SparsityMapEntry<N, T> *h_approx_entries = reinterpret_cast<SparsityMapEntry<N, T> *>(AffineAccessor<char, 1>(approx_entries, 0).base);
        CUDA_CHECK(cudaMemcpyAsync(h_approx_entries, approx_rects, num_approx * sizeof(Rect<N,T>), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        impl->set_instance(entries, end - start);
        impl->set_approx_instance(approx_entries, num_approx);
        if (end - start > ((size_t) DeppartConfig::cfg_max_rects_in_approximation)) {
          approx_rects_instance.destroy();
        }
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (auto const& elem : ctr) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(getMap(elem));
      impl->gpu_finalize();
    }
    final_entries_instance.destroy();
    final_rects_instance.destroy();
    boundaries_instance.destroy();
  }


}