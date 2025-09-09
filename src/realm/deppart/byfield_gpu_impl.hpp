#pragma once
#include "realm/deppart/byfield.h"
#include "realm/deppart/byfield_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include "realm/nvtx.h"

namespace Realm {

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

  // For profiling.
  NVTX_DEPPART(byfield_gpu);

  cudaStream_t stream = Cuda::get_task_cuda_stream();

  Memory my_mem = field_data[0].inst.get_location();

  collapsed_space<N, T> inst_space;

  // We combine all of our instances into one to batch work, tracking the offsets between instances.
  RegionInstance inst_offsets_instance = this->realm_malloc((field_data.size() + 1) * sizeof(size_t), my_mem);
  inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
  inst_space.num_children = field_data.size();

  RegionInstance inst_entries_instance;

  GPUMicroOp<N, T>::collapse_inst_space(field_data, inst_entries_instance, inst_space, my_mem, stream);

  RegionInstance parent_entries_instance;
  collapsed_space<N, T> collapsed_parent;

  // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
  GPUMicroOp<N, T>::collapse_parent_space(parent_space, parent_entries_instance, collapsed_parent, my_mem, stream);


  // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
  // to figure out where to write each rectangle.
  RegionInstance inst_counters_instance = this->realm_malloc((2*field_data.size() + 1) * sizeof(uint32_t), my_mem);
  uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);

  // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
  // to track which instance each rectangle came from in the populate phase.
  uint32_t* d_inst_prefix = d_inst_counters + field_data.size();
  RegionInstance out_instance;
  size_t num_valid_rects;

  // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
  GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space, collapsed_parent, out_instance, num_valid_rects, d_inst_counters, d_inst_prefix, my_mem, stream);
  inst_entries_instance.destroy();
  parent_entries_instance.destroy();
  inst_offsets_instance.destroy();

  // Early out if we don't have any rectangles.
  if (num_valid_rects == 0) {
    for (std::pair<const FT, SparsityMap<N, T>> it : sparsity_outputs) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it.second);
      impl->gpu_finalize();
    }
    out_instance.destroy();
    inst_counters_instance.destroy();
    return;
  }

  Rect<N, T>* d_valid_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(out_instance, 0).base);

  // Prefix sum the valid rectangles by volume.
  RegionInstance prefix_rects_instance;
  size_t total_pts;

  GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, prefix_rects_instance, total_pts, my_mem, stream);

  size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);

  // Now we have everything we need to actually populate our outputs.
  RegionInstance points_instance = this->realm_malloc(total_pts * sizeof(PointDesc<N,T>), my_mem);
  PointDesc<N,T>* d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

  FT* d_colors;
  RegionInstance colors_instance;

  // Memcpying a boolean vector breaks things for some reason so we have this disgusting workaround.
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

  Memory zcpy_mem;
  assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));

  // We need to pass the accessors to the GPU so it can read field values.
  RegionInstance accessors_instance = this->realm_malloc(field_data.size() * sizeof(AffineAccessor<FT,N,T>), zcpy_mem);
  AffineAccessor<FT,N,T>* d_accessors = reinterpret_cast<AffineAccessor<FT,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
  for (size_t i = 0; i < field_data.size(); ++i) {
    d_accessors[i] = AffineAccessor<FT,N,T>(field_data[i].inst, field_data[i].field_offset);
  }


  // This is where the work is actually done - each thread figures out which points to read, reads it, marks a PointDesc with its color, and writes it out.
  byfield_gpuPopulateBitmasksKernel<N,T,FT><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_colors, total_pts, colors.size(), num_valid_rects, field_data.size(), d_points);
  KERNEL_CHECK(stream);

  // Map colors to their output index to match send output iterator.
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

  // Ship off the points for final processing.

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
