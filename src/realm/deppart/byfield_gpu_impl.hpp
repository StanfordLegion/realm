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

  Cuda::AutoGPUContext agc(this->gpu);

  // For profiling.
  NVTX_DEPPART(byfield_gpu);

  CUstream stream = this->stream->get_stream();

  collapsed_space<N, T> inst_space;

  size_t tile_size = field_data[0].scratch_buffer.get_layout()->bytes_used;

  //std::cout << "Using tile size of " << tile_size << " bytes." << std::endl;

  Arena buffer_arena(field_data[0].scratch_buffer);

  inst_space.offsets = buffer_arena.alloc<size_t>(field_data.size() + 1);
  inst_space.num_children = field_data.size();

  Arena sys_arena;
  GPUMicroOp<N, T>::collapse_multi_space(field_data, inst_space, sys_arena, stream);

  collapsed_space<N, T> collapsed_parent;
  collapsed_parent.offsets = buffer_arena.alloc<size_t>(2);
  collapsed_parent.num_children = 1;
  std::vector<IndexSpace<N, T>> parent_spaces = {parent_space};

  // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
  GPUMicroOp<N, T>::collapse_multi_space(parent_spaces, collapsed_parent, buffer_arena, stream);

  // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
  // to figure out where to write each rectangle.
  uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2*field_data.size() + 1);

  // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
  // to track which instance each rectangle came from in the populate phase.
  uint32_t* d_inst_prefix = d_inst_counters + field_data.size();
  size_t num_valid_rects = 0;
  Rect<N, T>* d_valid_rects;

  FT* d_colors;


  // Memcpying a boolean vector breaks things for some reason so we have this disgusting workaround.
  if constexpr(std::is_same_v<FT,bool>) {
    std::vector<uint8_t> flat_colors(colors.size());
    for (size_t i = 0; i < colors.size(); i++) {
      flat_colors[i] = colors[i] ? 1 : 0;
    }
    uint8_t* d_flat_colors = buffer_arena.alloc<uint8_t>(colors.size());
    CUDA_CHECK(cudaMemcpyAsync(d_flat_colors, flat_colors.data(), colors.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream), stream);
    d_colors = reinterpret_cast<FT*>(d_flat_colors);
  } else {
    d_colors = buffer_arena.alloc<FT>(colors.size());
    CUDA_CHECK(cudaMemcpyAsync(d_colors, colors.data(), colors.size() * sizeof(FT), cudaMemcpyHostToDevice, stream), stream);
  }


  Memory zcpy_mem;
  assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM, buffer_arena.location));

  // We need to pass the accessors to the GPU so it can read field values.
  RegionInstance accessors_instance = this->realm_malloc(field_data.size() * sizeof(AffineAccessor<FT,N,T>), zcpy_mem);
  AffineAccessor<FT,N,T>* d_accessors = reinterpret_cast<AffineAccessor<FT,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
  for (size_t i = 0; i < field_data.size(); ++i) {
    d_accessors[i] = AffineAccessor<FT,N,T>(field_data[i].inst, field_data[i].field_offset);
  }

  buffer_arena.commit(false);

  // Map colors to their output index to match send output iterator.
  std::map<FT, size_t> color_indices;
  for (size_t i = 0; i < colors.size(); i++) {
    color_indices[colors[i]] = i;
  }

  Memory sysmem;
  assert(find_memory(sysmem, Memory::SYSTEM_MEM, buffer_arena.location));

  size_t num_output = 0;
  RectDesc<N, T>* output_start = nullptr;
  size_t num_completed = 0;
  size_t curr_tile = tile_size / 2;
  int count = 0;
  if (count) {}
  bool host_fallback = false;
  std::vector<RegionInstance> h_instances(colors.size(), RegionInstance::NO_INST);
  std::vector<size_t> entry_counts(colors.size(), 0);
  while (num_completed < inst_space.num_entries) {
    try {
      //std::cout << "Byfield iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_entries << " entries." << std::endl;
      buffer_arena.start();
      if (num_completed + curr_tile > inst_space.num_entries) {
        curr_tile = inst_space.num_entries - num_completed;
      }

      collapsed_space<N, T> inst_space_tile = inst_space;
      inst_space_tile.num_entries = curr_tile;
      inst_space_tile.entries_buffer = buffer_arena.alloc<SparsityMapEntry<N,T>>(curr_tile);
      CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.entries_buffer, inst_space.entries_buffer + num_completed, curr_tile * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);

      // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
      GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space_tile, collapsed_parent, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);


      // Early out if we don't have any rectangles.
      if (num_valid_rects == 0) {
        num_completed += curr_tile;
        curr_tile = tile_size / 2;
        subtract_const<<<COMPUTE_GRID(field_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, field_data.size()+1, curr_tile);
        KERNEL_CHECK(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        continue;
      }


      // Prefix sum the valid rectangles by volume.
      size_t total_pts;
      size_t* d_prefix_rects;

      GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

      // Now we have everything we need to actually populate our outputs.
      buffer_arena.flip_parity();
      assert(!buffer_arena.get_parity());

      PointDesc<N,T>* d_points = buffer_arena.alloc<PointDesc<N,T>>(total_pts);

      // This is where the work is actually done - each thread figures out which points to read, reads it, marks a PointDesc with its color, and writes it out.
      byfield_gpuPopulateBitmasksKernel<N,T,FT><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_colors, total_pts, colors.size(), num_valid_rects, field_data.size(), d_points);
      KERNEL_CHECK(stream);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      // Ship off the points for final processing.
      size_t num_new_rects = (num_output == 0) ? 1 : 2;
      assert(!buffer_arena.get_parity());
      RectDesc<N, T>* d_new_rects;
      this->complete_pipeline(d_points, total_pts, d_new_rects, num_new_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& kv){
                                // elem is a SparsityMap<N,T> from the vector
                                return color_indices.at(kv.first);
                             },
        /* getMap: */         [&](auto const& kv){
                              // return the SparsityMap key itself
                              return kv.second;
                           });

      if (host_fallback) {
        this->split_output(d_new_rects, num_new_rects, h_instances, entry_counts, buffer_arena);
      }

      if (num_output==0 || host_fallback) {
        num_output = num_new_rects;
        output_start = d_new_rects;
        num_completed += curr_tile;
        subtract_const<<<COMPUTE_GRID(field_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, field_data.size()+1, curr_tile);
        KERNEL_CHECK(stream);
        curr_tile = tile_size / 2;
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        continue;
      }

      //Otherwise we merge with existing rectangles
      RectDesc<N, T>* d_old_rects = buffer_arena.alloc<RectDesc<N, T>>(num_output);
      assert(d_old_rects == d_new_rects + num_new_rects);
      CUDA_CHECK(cudaMemcpyAsync(d_old_rects, output_start, num_output * sizeof(RectDesc<N,T>), cudaMemcpyDeviceToDevice, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      size_t num_final_rects = 1;
      //Send it off for processing
      this->complete_rect_pipeline(d_new_rects, num_output + num_new_rects, output_start, num_final_rects, buffer_arena,
      /* the Container: */  sparsity_outputs,
      /* getIndex: */       [&](auto const& kv){
                              // elem is a SparsityMap<N,T> from the vector
                              return color_indices.at(kv.first);
                           },
      /* getMap: */         [&](auto const& kv){
                            // return the SparsityMap key itself
                            return kv.second;
                         });
      num_completed += curr_tile;
      num_output = num_final_rects;
      subtract_const<<<COMPUTE_GRID(field_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, field_data.size()+1, curr_tile);
      KERNEL_CHECK(stream);
      curr_tile = tile_size / 2;
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    } catch (arena_oom&) {
      //std::cout << "Caught arena_oom, reducing tile size from " << curr_tile << " to " << curr_tile / 2 << std::endl;
      curr_tile /= 2;
      if (curr_tile == 0) {
        if (host_fallback) {
          GPUMicroOp<N, T>::shatter_rects(inst_space, num_completed, stream);
          curr_tile = 1;
        } else {
          host_fallback = true;
          if (num_output > 0) {
            this->split_output(output_start, num_output, h_instances, entry_counts, buffer_arena);
          }
          curr_tile = tile_size / 2;
        }
      }
    }
  }

  if (num_output == 0) {
    for (std::pair<const FT, SparsityMap<N, T>> it : sparsity_outputs) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it.second);
      if (this->exclusive) {
        impl->gpu_finalize();
      } else {
        impl->contribute_nothing();
      }
    }
    return;
  }

  if (!host_fallback) {
    try {
      this->send_output(output_start, num_output, buffer_arena, sparsity_outputs,
        /* getIndex: */       [&](auto const& kv){
                                // elem is a SparsityMap<N,T> from the vector
                                return color_indices.at(kv.first);
                             },
        /* getMap: */         [&](auto const& kv){
                              // return the SparsityMap key itself
                              return kv.second;
                           });
    } catch (arena_oom&) {
      this->split_output(output_start, num_output, h_instances, entry_counts, buffer_arena);
      host_fallback = true;
    }
  }

  if (host_fallback) {
    for (std::pair<const FT, SparsityMap<N, T>> it : sparsity_outputs) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it.second);
      if (this->exclusive) {
        impl->set_contributor_count(1);
      }
      size_t idx = color_indices.at(it.first);
      if (entry_counts[idx] > 0) {
        Rect<N, T>* h_rects = reinterpret_cast<Rect<N,T> *>(AffineAccessor<char,1>(h_instances[idx], 0).base);
        span<Rect<N, T>> h_rects_span(h_rects, entry_counts[idx]);
        impl->contribute_dense_rect_list(h_rects_span, true);
        h_instances[idx].destroy();
      } else {
        impl->contribute_nothing();
      }
    }
  }

}
}
