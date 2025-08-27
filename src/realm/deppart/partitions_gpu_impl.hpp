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

namespace Realm {

  //Used by cub::DeviceReduce to compute bad GPU approximation
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

    nvtx_range_push("cuda", "sort valid points");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);

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

    uintptr_t aux_in = AffineAccessor<char,1>(aux_instance, 0).base;
    uintptr_t aux_out = aux_in + max_aux_bytes;

    T* d_keys_in = reinterpret_cast<T*>(aux_in);
    T* d_keys_out = reinterpret_cast<T*>(aux_out);

    PointDesc<N,T>* d_points_in = d_points;
    PointDesc<N,T>* d_points_out = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(pg_instance, 0).base);

    RegionInstance break_points_instance = this->realm_malloc(total_pts * sizeof(uint8_t), my_mem);
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
    int threads_per_block = 256;
    size_t grid_size = (total_pts + threads_per_block - 1) / threads_per_block;
    size_t use_bytes = temp_bytes;
    for (int dim = 0; dim < N; ++dim) {
      build_coord_key<<<grid_size, threads_per_block, 0, stream>>>(d_keys_in, d_points_in, total_pts, dim);
      KERNEL_CHECK(stream);
      cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_keys_in, d_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(T), stream);
      std::swap(d_keys_in, d_keys_out);
      std::swap(d_points_in, d_points_out);
    }

    //Sort by source index now to keep individual partitions separate
    build_src_key<<<grid_size, threads_per_block, 0, stream>>>(d_src_keys_in, d_points_in, total_pts);
    KERNEL_CHECK(stream);
    use_bytes = temp_bytes;
    cub::DeviceRadixSort::SortPairs(temp_storage, use_bytes, d_src_keys_in, d_src_keys_out, d_points_in, d_points_out, total_pts, 0, 8*sizeof(size_t), stream);

    nvtx_range_pop();

    nvtx_range_push("cuda", "Run length encode");


    points_to_rects<<<grid_size, threads_per_block, 0, stream>>>(d_points_out, d_rects_in, total_pts);
    KERNEL_CHECK(stream);

    size_t num_intermediate = total_pts;


    for (int dim = N-1; dim >= 0; --dim) {

      // Step 1: Mark rectangle starts
      // e.g. [1, 2, 4, 5, 6, 8] -> [1, 0, 1, 0, 0, 1]
      grid_size = (num_intermediate + threads_per_block - 1) / threads_per_block;
      mark_breaks_dim<<<grid_size, threads_per_block, 0, stream>>>(d_rects_in, break_points, num_intermediate, dim);
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
      init_rects_dim<<<grid_size, threads_per_block, 0, stream>>>(d_rects_in, break_points, group_ids, d_rects_out, num_intermediate, dim);
      KERNEL_CHECK(stream);

      num_intermediate = last_grp;
      std::swap(d_rects_in, d_rects_out);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    pg_instance.destroy();
    temp_storage_instance.destroy();
    break_points_instance.destroy();

    nvtx_range_pop();

    this->send_output(d_rects_in, num_intermediate, my_mem, ctr, getIndex, getMap);

    nvtx_range_push("cuda", "free");
    cudaStreamDestroy(stream);
    aux_instance.destroy();
    nvtx_range_pop();
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
    nvtx_range_push("cuda", "build final output");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);

    RegionInstance final_entries_instance = this->realm_malloc(total_rects * sizeof(SparsityMapEntry<N,T>), my_mem);
    SparsityMapEntry<N,T>* final_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(final_entries_instance, 0).base);

    RegionInstance final_rects_instance = this->realm_malloc(total_rects * sizeof(Rect<N,T>), my_mem);
    Rect<N,T>* final_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(final_rects_instance, 0).base);

    RegionInstance starts_instance = this->realm_malloc(ctr.size() * sizeof(size_t), my_mem);
    size_t* d_starts = reinterpret_cast<size_t*>(AffineAccessor<char,1>(starts_instance, 0).base);

    RegionInstance ends_instance = this->realm_malloc(ctr.size() * sizeof(size_t), my_mem);
    size_t* d_ends = reinterpret_cast<size_t*>(AffineAccessor<char,1>(ends_instance, 0).base);

    CUDA_CHECK(cudaMemsetAsync(d_starts, 0, ctr.size()*sizeof(size_t),stream), stream);
    CUDA_CHECK(cudaMemsetAsync(d_ends, 0, ctr.size()*sizeof(size_t),stream), stream);


    //Convert RectDesc to SparsityMapEntry and determine where eeach src's rectangles start and end.
    int threads_per_block = 256;
    size_t grid_size = (total_rects + threads_per_block - 1) / threads_per_block;
    build_final_output<<<grid_size, threads_per_block, 0, stream>>>(d_rects, final_entries, final_rects, d_starts, d_ends, total_rects);
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

    //Find system memory
    Memory sysmem;
    bool found_sysmem = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(auto& memory : all_memories) {
      if(memory.kind() == Memory::SYSTEM_MEM) {
        sysmem = memory;
        found_sysmem = true;
        break;
      }
    }
    assert(found_sysmem);


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
          approx_rects_instance = realm_malloc(sizeof(Rect<N,T>), my_mem);
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
          RegionInstance temp_rects_instance = realm_malloc(temp_sz * sizeof(Rect<N,T>), my_mem);
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

    nvtx_range_pop();
    nvtx_range_push("cuda", "free");
    cudaStreamDestroy(stream);
    final_entries_instance.destroy();
    final_rects_instance.destroy();
    starts_instance.destroy();
    ends_instance.destroy();
    nvtx_range_pop();
  }


}