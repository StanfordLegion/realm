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
void GPUByFieldMicroOp<N,T,FT>::execute()
{

    NVTX_DEPPART(byfield_gpu);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    Memory my_mem = field_data[0].inst.get_location();

    collapsed_space<N, T> inst_space;

    //This is used to track which instance each rectangle came from
    RegionInstance inst_offsets_instance = this->realm_malloc((field_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = field_data.size();

    RegionInstance inst_entries_instance;

    GPUMicroOp<N, T>::collapse_inst_space(field_data, inst_entries_instance, inst_space, my_mem, stream);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    GPUMicroOp<N, T>::collapse_parent_space(parent_space, parent_entries_instance, collapsed_parent, my_mem, stream);

    RegionInstance inst_counters_instance = this->realm_malloc((2*field_data.size() + 1) * sizeof(uint32_t), my_mem);

    //This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
    uint32_t* d_inst_prefix = d_inst_counters + field_data.size();
    RegionInstance out_instance;
    size_t num_valid_rects;

    GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space, collapsed_parent, out_instance, num_valid_rects, d_inst_counters, d_inst_prefix, my_mem, stream);
    Rect<N, T>* d_valid_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);

    if (num_valid_rects == 0) {
      for (std::pair<const FT, SparsityMap<N, T>> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it.second);
        impl->gpu_finalize();
      }
      inst_offsets_instance.destroy();
      inst_counters_instance.destroy();
      inst_entries_instance.destroy();
      parent_entries_instance.destroy();
      return;
    }

    RegionInstance prefix_rects_instance = this->realm_malloc((num_valid_rects + 1) * sizeof(size_t), my_mem);

    // Prefix sum the valid rectangles by volume
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

    RegionInstance temp_instance = this->realm_malloc(rect_temp_bytes, my_mem);
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

    RegionInstance points_instance = this->realm_malloc(total_pts * sizeof(PointDesc<N,T>), my_mem);
    PointDesc<N,T>* d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

    FT* d_colors;
    RegionInstance colors_instance;

    //Memcpying a boolean vector breaks things for some reason so we have this disgusting workaround
    if constexpr(std::is_same_v<FT,bool>) {
      std::vector<uint8_t> flat_colors(colors.size());
      for (size_t i = 0; i < colors.size(); i++) {
        flat_colors[i] = colors[i] ? 1 : 0;
      }
      colors_instance = this->realm_malloc(total_pts * sizeof(PointDesc<N,T>), my_mem);
      uint8_t* d_flat_colors = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(colors_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_flat_colors, flat_colors.data(), colors.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream), stream);
      d_colors = reinterpret_cast<FT*>(d_flat_colors);
    } else {
      colors_instance = this->realm_malloc(colors.size() * sizeof(FT), my_mem);
      d_colors = reinterpret_cast<FT*>(AffineAccessor<char,1>(colors_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_colors, colors.data(), colors.size() * sizeof(FT), cudaMemcpyHostToDevice, stream), stream);
    }



    // We need to pass the accessors to the GPU so it can read field values
    std::vector<AffineAccessor<FT,N,T>> h_accessors(field_data.size());
    for (size_t i = 0; i < field_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<FT,N,T>(field_data[i].inst, field_data[i].field_offset);
    }

    RegionInstance accessors_instance = this->realm_malloc(field_data.size() * sizeof(AffineAccessor<FT,N,T>), my_mem);
    AffineAccessor<FT,N,T>* d_accessors = reinterpret_cast<AffineAccessor<FT,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(), field_data.size() * sizeof(AffineAccessor<FT,N,T>), cudaMemcpyHostToDevice, stream), stream);


    //This is where the work is actually done - each thread figures out which points to read, reads it, marks a PointDesc with its color, and writes it out
    int threads_per_block = 256;
    int num_blocks = (total_pts + threads_per_block - 1) / threads_per_block;
    byfield_gpuPopulateBitmasksKernel<N,T,FT><<<num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_colors, total_pts, colors.size(), num_valid_rects, field_data.size(), d_points);
    KERNEL_CHECK(stream);

  // Map colors to their output index to match send output iterator
  std::map<FT, size_t> color_indices;
  for (size_t i = 0; i < colors.size(); i++) {
    color_indices[colors[i]] = i;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream), stream);
  prefix_rects_instance.destroy();
  colors_instance.destroy();
  accessors_instance.destroy();
  out_instance.destroy();
  inst_counters_instance.destroy();

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
