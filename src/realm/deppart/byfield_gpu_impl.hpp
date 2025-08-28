#pragma once
#include "realm/deppart/byfield.h"
#include "realm/deppart/byfield_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include "realm/nvtx.h"

namespace Realm {

//Used to compute prefix sum by volume for an array of rects
template <int N, typename T>
struct RectFieldVolumeOp {
  __device__ __forceinline__
  size_t operator()(const Rect<N,T>& r) const {
    return r.volume();
  }
};

/*
 *  Input (stored in MicroOp): Array of field instances, a parent index space, and a list of colors
 *  Output: A list of (potentially overlapping) points in original instances ∩ parent index space marked with their color,
 *  which it then sends off to complete_pipeline.
 *  Approach: Intersect all instance rectangles with parent rectangles in parallel. For surviving rectangles, use
 *  prefix sum + binary search to iterate over these in parallel and mark each point with its color.
 */
template <int N, typename T, typename FT>
void GPUByFieldMicroOp<N,T,FT>::gpu_populate_bitmasks()
{

    NVTX_DEPPART(byfield_gpu);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);

    Memory my_mem = field_data[0].inst.get_location();


    // We need inst_offsets to preserve which instance each rectangle came from
    // so that we know where to read from after intersecting.
    std::vector<size_t> inst_offsets(field_data.size() + 1);

    //Determine size of allocation for combined inst_rects and offset entries
    size_t inst_size = 0;


    for (size_t i = 0; i < field_data.size(); ++i) {
      inst_offsets[i] = inst_size;
      if (field_data[i].index_space.dense()) {
        inst_size += 1;
      } else {
        inst_size += field_data[i].index_space.sparsity.impl()->get_entries().size();
      }
    }
    inst_offsets[field_data.size()] = inst_size;


    RegionInstance inst_entries_instance;
    this->realm_malloc(inst_entries_instance, inst_size * sizeof(SparsityMapEntry<N,T>), my_mem).wait();
    SparsityMapEntry<N,T>* d_inst_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(inst_entries_instance, 0).base);

    //Now we fill the device array with all instance rectangles
    size_t inst_pos = 0;
    for (size_t i = 0; i < field_data.size(); ++i) {
      const IndexSpace<N,T> &inst_space = field_data[i].index_space;
      if (inst_space.dense()) {
        SparsityMapEntry<N,T> entry;
        entry.bounds = inst_space.bounds;
        CUDA_CHECK(cudaMemcpyAsync(d_inst_entries + inst_pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
        ++inst_pos;
      } else {
        auto tmp = inst_space.sparsity.impl()->get_entries();
        CUDA_CHECK(cudaMemcpyAsync(d_inst_entries + inst_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
        inst_pos += tmp.size();
      }
    }

    //Copy the parent entries to the device as well
    span<SparsityMapEntry<N,T>> parent_entries;
    std::vector<SparsityMapEntry<N,T>> parent_entries_vec;
    if (parent_space.dense()) {
      SparsityMapEntry<N,T> entry;
      entry.bounds = parent_space.bounds;
      parent_entries_vec = {entry};
      parent_entries = span<SparsityMapEntry<N,T>>(parent_entries_vec.data(), 1);
    } else {
      parent_entries = parent_space.sparsity.impl()->get_entries();
    }

    RegionInstance parent_entries_instance;
    Event parent_alloc_event = this->realm_malloc(parent_entries_instance, parent_entries.size() * sizeof(SparsityMapEntry<N,T>), my_mem);

    RegionInstance inst_counters_instance;
    Event counters_alloc_event = this->realm_malloc(inst_counters_instance, (field_data.size()) * sizeof(uint32_t), my_mem);

    RegionInstance inst_offsets_instance;
    Event inst_offsets_alloc_event = this->realm_malloc(inst_offsets_instance, (field_data.size() + 1) * sizeof(size_t), my_mem);

    parent_alloc_event.wait();
    SparsityMapEntry<N, T>* d_parent_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(parent_entries_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_parent_entries, parent_entries.data(), parent_entries.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);

    //This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle
    counters_alloc_event.wait();
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (field_data.size()) * sizeof(uint32_t), stream), stream);


    //This is used to track which instance each rectangle came from
    inst_offsets_alloc_event.wait();
    size_t* d_inst_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_offsets, inst_offsets.data(), (field_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);


    //First pass: figure out how many rectangles survive intersection
    int flattened_threads = 256;
    int grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, nullptr, inst_size, parent_entries.size(), field_data.size(), d_inst_counters, nullptr);
    KERNEL_CHECK(stream);


    //Prefix sum over instances (small enough to keep on host)
    std::vector<uint32_t> h_inst_counters(field_data.size()+1);
    h_inst_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_inst_counters.data()+1, d_inst_counters, field_data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 0; i < field_data.size(); ++i) {
      h_inst_counters[i+1] += h_inst_counters[i];
    }

    size_t num_valid_rects = h_inst_counters[field_data.size()];

    if (num_valid_rects == 0) {
      for (std::pair<const FT, SparsityMap<N, T>> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it.second);
        impl->gpu_finalize();
      }
      inst_offsets_instance.destroy();
      inst_counters_instance.destroy();
      inst_entries_instance.destroy();
      parent_entries_instance.destroy();
      cudaStreamDestroy(stream);
      return;
    }


    RegionInstance inst_prefix_instance;
    Event inst_prefix_alloc_event = this->realm_malloc(inst_prefix_instance, (field_data.size() + 1) * sizeof(uint32_t), my_mem);
    RegionInstance valid_rects_instance;
    Event valid_rects_alloc_event = this->realm_malloc(valid_rects_instance, num_valid_rects * sizeof(Rect<N,T>), my_mem);
    RegionInstance prefix_rects_instance;;
    Event vol_prefix_alloc_event = this->realm_malloc(prefix_rects_instance, (num_valid_rects + 1) * sizeof(size_t), my_mem);

    //Where each instance should start writing its rectangles
    inst_prefix_alloc_event.wait();
    uint32_t* d_inst_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_prefix_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_prefix, h_inst_counters.data(), (field_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    //Non-empty rectangles from the intersection
    valid_rects_alloc_event.wait();
    Rect<N,T>* d_valid_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(valid_rects_instance, 0).base);

    //Reset counters
    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (field_data.size()) * sizeof(uint32_t), stream), stream);

    //Second pass: recompute intersection, but this time write to output
    grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, d_inst_prefix, inst_size, parent_entries.size(), field_data.size(), d_inst_counters, d_valid_rects);
    KERNEL_CHECK(stream);


    // Prefix sum the valid rectangles by volume
    vol_prefix_alloc_event.wait();
    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_prefix_rects, 0, sizeof(size_t), stream), stream);

    // Build the CUB transform‐iterator
    using VolIter = cub::TransformInputIterator<
                      size_t,              // output type
                      RectFieldVolumeOp<N,T>,            // functor
                      Rect<N,T>*     // underlying input iterator
                    >;
    VolIter d_volumes(d_valid_rects, RectFieldVolumeOp<N,T>());

    void*   d_temp = nullptr;
    size_t rect_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        /* d_temp_storage */  nullptr,
        /* temp_bytes */      rect_temp_bytes,
        /* d_in */            d_volumes,
        /* d_out */           d_prefix_rects + 1,   // shift by one so prefix[1]..prefix[n]
        /* num_items */       num_valid_rects, stream);

    RegionInstance temp_instance;
    this->realm_malloc(temp_instance, rect_temp_bytes, my_mem).wait();
    d_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_instance, 0).base);
    cub::DeviceScan::InclusiveSum(
        /* d_temp_storage */  d_temp,
        /* temp_bytes */      rect_temp_bytes,
        /* d_in */            d_volumes,
        /* d_out */           d_prefix_rects + 1,
        /* num_items */       num_valid_rects, stream);


    //Number of points across all rectangles (also our total output count)
    size_t total_pts;
    CUDA_CHECK(cudaMemcpyAsync(&total_pts, &d_prefix_rects[num_valid_rects], sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    inst_entries_instance.destroy();
    parent_entries_instance.destroy();
    inst_offsets_instance.destroy();
    temp_instance.destroy();
    inst_counters_instance.destroy();

    RegionInstance points_instance;
    this->realm_malloc(points_instance, total_pts * sizeof(PointDesc<N,T>), my_mem).wait();
    PointDesc<N,T>* d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

    FT* d_colors;
    RegionInstance colors_instance;

    //Memcpying a boolean vector breaks things for some reason so we have this disgusting workaround
    if constexpr(std::is_same_v<FT,bool>) {
      std::vector<uint8_t> flat_colors(colors.size());
      for (size_t i = 0; i < colors.size(); i++) {
        flat_colors[i] = colors[i] ? 1 : 0;
      }
      this->realm_malloc(colors_instance, total_pts * sizeof(PointDesc<N,T>), my_mem).wait();
      uint8_t* d_flat_colors = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(colors_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_flat_colors, flat_colors.data(), colors.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream), stream);
      d_colors = reinterpret_cast<FT*>(d_flat_colors);
    } else {
      this->realm_malloc(colors_instance, colors.size() * sizeof(FT), my_mem).wait();
      d_colors = reinterpret_cast<FT*>(AffineAccessor<char,1>(colors_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_colors, colors.data(), colors.size() * sizeof(FT), cudaMemcpyHostToDevice, stream), stream);
    }



    // We need to pass the accessors to the GPU so it can read field values
    std::vector<AffineAccessor<FT,N,T>> h_accessors(field_data.size());
    for (size_t i = 0; i < field_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<FT,N,T>(field_data[i].inst, field_data[i].field_offset);
    }

    RegionInstance accessors_instance;
    this->realm_malloc(accessors_instance, field_data.size() * sizeof(AffineAccessor<FT,N,T>), my_mem).wait();
    AffineAccessor<FT,N,T>* d_accessors = reinterpret_cast<AffineAccessor<FT,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(), field_data.size() * sizeof(AffineAccessor<FT,N,T>), cudaMemcpyHostToDevice, stream), stream);


    //This is where the work is actually done - each thread figures out which points to read, reads it, marks a PointDesc with its color, and writes it out
    int num_blocks = (total_pts + flattened_threads - 1) / flattened_threads;
    byfield_gpuPopulateBitmasksKernel<N,T,FT><<<num_blocks, flattened_threads, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_colors, total_pts, colors.size(), num_valid_rects, field_data.size(), d_points);
    KERNEL_CHECK(stream);

  // Map colors to their output index to match send output iterator
  std::map<FT, size_t> color_indices;
  for (size_t i = 0; i < colors.size(); i++) {
    color_indices[colors[i]] = i;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream), stream);
  cudaStreamDestroy(stream);
  valid_rects_instance.destroy();
  prefix_rects_instance.destroy();
  colors_instance.destroy();
  accessors_instance.destroy();
  inst_prefix_instance.destroy();

  // Ship off the points for final processing

  this->complete_pipeline(d_points, total_pts, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& kv){
                            // elem is a SparsityMap<N,T> from the vector
                            return color_indices.at(kv.first);
                         },
    /* getMap: */         [&](auto const& kv){
                          // return the SparsityMap key itself
                          return kv.second;
                       });

    points_instance.destroy();
}
}
