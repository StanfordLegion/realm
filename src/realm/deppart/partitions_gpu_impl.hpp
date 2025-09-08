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


//NVTX macros to only add ranges if defined
#ifdef REALM_USE_NVTX

  #define NVTX_CAT(a,b)  a##b

  #define NVTX_DEPPART(message) \
  nvtxScopedRange NVTX_CAT(nvtx_, message)("cuda", #message, 0)

#else

  #define NVTX_DEPPART(message) do { } while (0)

#endif

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

  template<int N, typename T>
  struct BVH {
    Rect<N,T>* boxes;
    int root;
    size_t num_leaves;
    uint64_t* indices;
    uint64_t* labels;
    int* childLeft;
    int* childRight;
  };

  inline bool find_sysmem(Memory &sysmem)
  {
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
    return found_sysmem;
  }

  template<int N, typename T>
  template<typename FT>
  void GPUMicroOp<N,T>::collapse_inst_space(const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT> >& field_data, RegionInstance& out_instance, collapsed_space<N, T> &out_space, Memory my_mem, cudaStream_t stream)
  {
    // We need inst_offsets to preserve which instance each rectangle came from
    // so that we know where to read from after intersecting.
    std::vector<size_t> inst_offsets(field_data.size() + 1);

    //Determine size of allocation for combined inst_rects and offset entries
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

    Memory sysmem;
    assert(find_sysmem(sysmem));

    RegionInstance h_instance = realm_malloc(out_space.num_entries * sizeof(SparsityMapEntry<N,T>), sysmem);
    SparsityMapEntry<N, T>* h_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(h_instance, 0).base);

    out_instance = realm_malloc(out_space.num_entries * sizeof(SparsityMapEntry<N,T>), my_mem);
    out_space.entries_buffer = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);

    //Now we fill the device array with all instance rectangles
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
    CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, h_entries, out_space.num_entries * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(out_space.offsets, inst_offsets.data(), (field_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    h_instance.destroy();
  }

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
      CUDA_CHECK(cudaMemcpyAsync(out_space.entries_buffer, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    }
    out_space.offsets = nullptr;
    out_space.num_children = 1;
  }

  //template<int N, typename T>
  //template<typename out_t>
  //void GPUMicroOp<N,T>::construct_input_rectlist(const collapsed_space<N, T> &lhs, const collapsed_space<N, T> &rhs, RegionInstance &out_instance,  size_t& out_size, uint32_t* out_offsets, Memory my_mem)
  //{
//
  //}

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
    int threads_per_block = 256;
    size_t grid_size = (total_pts + threads_per_block - 1) / threads_per_block;
    size_t use_bytes = temp_bytes;

    {
      NVTX_DEPPART(sort_valid_points);
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
    }


    points_to_rects<<<grid_size, threads_per_block, 0, stream>>>(d_points_out, d_rects_in, total_pts);
    KERNEL_CHECK(stream);

    size_t num_intermediate = total_pts;

    {
      NVTX_DEPPART(run_length_encode);

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

    Memory sysmem;
    assert(find_sysmem(sysmem));

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