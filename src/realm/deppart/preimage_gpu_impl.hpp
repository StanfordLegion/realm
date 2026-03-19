#pragma once
#include "realm/deppart/preimage.h"
#include "realm/deppart/preimage_gpu_kernels.hpp"
#include "realm/deppart/byfield_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include <sstream>
#include "realm/nvtx.h"

namespace Realm {

  template<int N, typename T, int N2, typename T2>
  void GPUPreimageMicroOp<N, T, N2, T2>::gpu_populate_ranges() {
    if (targets.size() == 0) {
      assert(sparsity_outputs.empty());
      return;
    }

    RegionInstance buffer = domain_transform.range_data[0].scratch_buffer;

    size_t tile_size = buffer.get_layout()->bytes_used;
    //std::cout << "Using tile size of " << tile_size << " bytes." << std::endl;
    Arena buffer_arena(buffer);

    NVTX_DEPPART(gpu_preimage_range);

    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM, buffer_arena.location));

    CUstream stream = this->stream->get_stream();

    collapsed_space<N, T> inst_space;

    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    inst_space.offsets = buffer_arena.alloc<size_t>(domain_transform.range_data.size() + 1);
    inst_space.num_children = domain_transform.range_data.size();

    Arena sys_arena;
    GPUMicroOp<N, T>::collapse_multi_space(domain_transform.range_data, inst_space, sys_arena, stream);

    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);


    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2 * domain_transform.range_data.size() + 1);

    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.range_data.size();

    collapsed_space<N2, T2> target_space;
    target_space.offsets = buffer_arena.alloc<size_t>(targets.size() + 1);
    target_space.num_children = targets.size();

    GPUMicroOp<N2, T2>::collapse_multi_space(targets, target_space, buffer_arena, stream);

    std::vector<AffineAccessor<Rect<N2,T2>,N,T>> h_accessors(domain_transform.range_data.size());
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Rect<N2,T2>,N,T>(domain_transform.range_data[i].inst, domain_transform.range_data[i].field_offset);
    }
    AffineAccessor<Rect<N2,T2>,N,T>* d_accessors =
        buffer_arena.alloc<AffineAccessor<Rect<N2,T2>,N,T>>(domain_transform.range_data.size());
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(),
                               domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N2,T2>,N,T>),
                               cudaMemcpyHostToDevice, stream), stream);

    uint32_t* d_target_counters = buffer_arena.alloc<uint32_t>(2*targets.size() + 1);
    uint32_t* d_targets_prefix = d_target_counters + targets.size();
    CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

    buffer_arena.commit(false);

    size_t num_output = 0;
    RectDesc<N, T>* output_start = nullptr;
    size_t num_completed = 0;
    size_t curr_tile = tile_size / 2;
    int count = 0;
    if (count) {}
    bool host_fallback = false;
    std::vector<Rect<N, T>*> host_rect_buffers(targets.size(), nullptr);
    std::vector<size_t> entry_counts(targets.size(), 0);
    while (num_completed < inst_space.num_entries) {
      try {

        //std::cout << "Preimage iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_entries << " entries." << std::endl;
        buffer_arena.start();
        if (num_completed + curr_tile > inst_space.num_entries) {
          curr_tile = inst_space.num_entries - num_completed;
        }

        collapsed_space<N, T> inst_space_tile = inst_space;
        inst_space_tile.num_entries = curr_tile;
        inst_space_tile.entries_buffer = buffer_arena.alloc<SparsityMapEntry<N,T>>(curr_tile);
        CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.entries_buffer, inst_space.entries_buffer + num_completed, curr_tile * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);

        size_t num_valid_rects;
        Rect<N, T>* d_valid_rects;
        // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
        GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space_tile, collapsed_parent, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);

        if (num_valid_rects == 0) {
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        // Prefix sum the valid rectangles by volume.
        size_t total_pts;
        size_t* d_prefix_rects;
        GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

        PointDesc<N,T>* d_points;
        size_t num_valid_points;

        CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

        if (target_space.num_entries > targets.size()) {

          BVH<N2, T2> preimage_bvh;
          GPUMicroOp<N2, T2>::build_bvh(target_space, preimage_bvh, buffer_arena, stream);

          preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
           preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.range_data.size(), preimage_bvh.num_leaves, nullptr, d_target_counters, nullptr);
          KERNEL_CHECK(stream);

          std::vector<uint32_t> h_target_counters(targets.size()+1);
          h_target_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_target_counters.data()+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          for (size_t i = 0; i < targets.size(); ++i) {
            h_target_counters[i+1] += h_target_counters[i];
          }

          num_valid_points = h_target_counters[targets.size()];

          if (num_valid_points == 0) {
            num_completed += curr_tile;
            subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
            KERNEL_CHECK(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream), stream);
            curr_tile = tile_size / 2;
            continue;
          }

          CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

          buffer_arena.flip_parity();
          d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

          CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

          preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
           preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.range_data.size(), preimage_bvh.num_leaves, d_targets_prefix, d_target_counters, d_points);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        } else {
          preimage_dense_populate_bitmasks_kernel< N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
           num_valid_rects, domain_transform.range_data.size(), targets.size(), nullptr, d_target_counters, nullptr);
          KERNEL_CHECK(stream);

          std::vector<uint32_t> h_target_counters(targets.size()+1);
          h_target_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_target_counters.data()+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          for (size_t i = 0; i < targets.size(); ++i) {
            h_target_counters[i+1] += h_target_counters[i];
          }

          num_valid_points = h_target_counters[targets.size()];

          if (num_valid_points == 0) {
            num_completed += curr_tile;
            subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
            KERNEL_CHECK(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream), stream);
            curr_tile = tile_size / 2;
            continue;
          }

          CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

          buffer_arena.flip_parity();
          d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

          CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

          preimage_dense_populate_bitmasks_kernel< N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
           num_valid_rects, domain_transform.range_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        }

        buffer_arena.flip_parity();
        buffer_arena.flip_parity();
        d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

        size_t num_new_rects = num_output == 0 ? 1 : 2;
        assert(!buffer_arena.get_parity());
        RectDesc<N, T>* d_new_rects;

        this->complete_pipeline(d_points, num_valid_points, d_new_rects, num_new_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });

        if (host_fallback) {
          this->split_output(d_new_rects, num_new_rects, host_rect_buffers, entry_counts, buffer_arena);
        }

        if (num_output==0 || host_fallback) {
          num_output = num_new_rects;
          num_completed += curr_tile;
          output_start = d_new_rects;
          subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          curr_tile = tile_size / 2;
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          continue;
        }

        RectDesc<N, T>* d_old_rects = buffer_arena.alloc<RectDesc<N, T>>(num_output);
        assert(d_old_rects == d_new_rects + num_new_rects);
        CUDA_CHECK(cudaMemcpyAsync(d_old_rects, output_start, num_output * sizeof(RectDesc<N,T>), cudaMemcpyDeviceToDevice, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        size_t num_final_rects = 1;

        //Send it off for processing
        this->complete_rect_pipeline(d_new_rects, num_output + num_new_rects, output_start, num_final_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });
        num_completed += curr_tile;
        num_output = num_final_rects;
        subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
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
              this->split_output(output_start, num_output, host_rect_buffers, entry_counts, buffer_arena);
            }
            curr_tile = tile_size / 2;
          }
        }
      }
    }
    if (num_output == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
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
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });
      } catch (arena_oom&) {
        this->split_output(output_start, num_output, host_rect_buffers, entry_counts, buffer_arena);
        host_fallback = true;
      }
    }

    if (host_fallback) {
      for (size_t idx = 0; idx < sparsity_outputs.size(); ++idx) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(sparsity_outputs[idx]);
        if (this->exclusive) {
          impl->set_contributor_count(1);
        }
        if (entry_counts[idx] > 0) {
          span<Rect<N, T>> h_rects_span(host_rect_buffers[idx], entry_counts[idx]);
          impl->contribute_dense_rect_list(h_rects_span, true);
          deppart_host_free(host_rect_buffers[idx]);
        } else {
          impl->contribute_nothing();
        }
      }
    }
  }

  template<int N, typename T, int N2, typename T2>
  void GPUPreimageMicroOp<N, T, N2, T2>::gpu_populate_bitmasks() {
    if (targets.size() == 0) {
      assert(sparsity_outputs.empty());
      return;
    }

    RegionInstance buffer = domain_transform.ptr_data[0].scratch_buffer;

    size_t tile_size = buffer.get_layout()->bytes_used;
    //std::cout << "Using tile size of " << tile_size << " bytes." << std::endl;
    Arena buffer_arena(buffer);

    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM, buffer_arena.location));

    CUstream stream = this->stream->get_stream();

    NVTX_DEPPART(gpu_preimage);

    collapsed_space<N, T> inst_space;

    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    inst_space.offsets = buffer_arena.alloc<size_t>(domain_transform.ptr_data.size() + 1);
    inst_space.num_children = domain_transform.ptr_data.size();

    Arena sys_arena;
    GPUMicroOp<N, T>::collapse_multi_space(domain_transform.ptr_data, inst_space, sys_arena, stream);

    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);


    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2 * domain_transform.ptr_data.size() + 1);

    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.ptr_data.size();

    collapsed_space<N2, T2> target_space;
    target_space.offsets = buffer_arena.alloc<size_t>(targets.size() + 1);
    target_space.num_children = targets.size();

    GPUMicroOp<N2, T2>::collapse_multi_space(targets, target_space, buffer_arena, stream);

    std::vector<AffineAccessor<Point<N2,T2>,N,T>> h_accessors(domain_transform.ptr_data.size());
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Point<N2,T2>,N,T>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }
    AffineAccessor<Point<N2,T2>,N,T>* d_accessors =
        buffer_arena.alloc<AffineAccessor<Point<N2,T2>,N,T>>(domain_transform.ptr_data.size());
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(),
                               domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N2,T2>,N,T>),
                               cudaMemcpyHostToDevice, stream), stream);

    uint32_t* d_target_counters = buffer_arena.alloc<uint32_t>(2*targets.size() + 1);
    uint32_t* d_targets_prefix = d_target_counters + targets.size();
    CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

    buffer_arena.commit(false);

    size_t num_output = 0;
    RectDesc<N, T>* output_start = nullptr;
    size_t num_completed = 0;
    size_t curr_tile = tile_size / 2;
    int count = 0;
    if (count) {}
    bool host_fallback = false;
    std::vector<Rect<N, T>*> host_rect_buffers(targets.size(), nullptr);
    std::vector<size_t> entry_counts(targets.size(), 0);
    while (num_completed < inst_space.num_entries) {
      try {

        //std::cout << "Preimage iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_entries << " entries." << std::endl;
        buffer_arena.start();
        if (num_completed + curr_tile > inst_space.num_entries) {
          curr_tile = inst_space.num_entries - num_completed;
        }

        collapsed_space<N, T> inst_space_tile = inst_space;
        inst_space_tile.num_entries = curr_tile;
        inst_space_tile.entries_buffer = buffer_arena.alloc<SparsityMapEntry<N,T>>(curr_tile);
        CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.entries_buffer, inst_space.entries_buffer + num_completed, curr_tile * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);

        size_t num_valid_rects;
        Rect<N, T>* d_valid_rects;
        // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
        GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space_tile, collapsed_parent, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);

        if (num_valid_rects == 0) {
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        // Prefix sum the valid rectangles by volume.
        size_t total_pts;
        size_t* d_prefix_rects;
        GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

        PointDesc<N,T>* d_points;
        size_t num_valid_points;

        CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

        if (target_space.num_entries > targets.size()) {

          BVH<N2, T2> preimage_bvh;
          GPUMicroOp<N2, T2>::build_bvh(target_space, preimage_bvh, buffer_arena, stream);

          preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
           preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.ptr_data.size(), preimage_bvh.num_leaves, nullptr, d_target_counters, nullptr);
          KERNEL_CHECK(stream);

          std::vector<uint32_t> h_target_counters(targets.size()+1);
          h_target_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_target_counters.data()+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          for (size_t i = 0; i < targets.size(); ++i) {
            h_target_counters[i+1] += h_target_counters[i];
          }

          num_valid_points = h_target_counters[targets.size()];

          if (num_valid_points == 0) {
            num_completed += curr_tile;
            subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
            KERNEL_CHECK(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream), stream);
            curr_tile = tile_size / 2;
            continue;
          }

          CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

          buffer_arena.flip_parity();
          d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

          CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

          preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
           preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.ptr_data.size(), preimage_bvh.num_leaves, d_targets_prefix, d_target_counters, d_points);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        } else {
          preimage_dense_populate_bitmasks_kernel< N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
           num_valid_rects, domain_transform.ptr_data.size(), targets.size(), nullptr, d_target_counters, nullptr);
          KERNEL_CHECK(stream);

          std::vector<uint32_t> h_target_counters(targets.size()+1);
          h_target_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_target_counters.data()+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          for (size_t i = 0; i < targets.size(); ++i) {
            h_target_counters[i+1] += h_target_counters[i];
          }

          num_valid_points = h_target_counters[targets.size()];

          if (num_valid_points == 0) {
            num_completed += curr_tile;
            subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
            KERNEL_CHECK(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream), stream);
            curr_tile = tile_size / 2;
            continue;
          }

          CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

          buffer_arena.flip_parity();
          d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

          CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

          preimage_dense_populate_bitmasks_kernel< N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
           num_valid_rects, domain_transform.ptr_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        }

        buffer_arena.flip_parity();
        buffer_arena.flip_parity();
        d_points = buffer_arena.alloc<PointDesc<N, T>>(num_valid_points);

        size_t num_new_rects = num_output == 0 ? 1 : 2;
        assert(!buffer_arena.get_parity());
        RectDesc<N, T>* d_new_rects;

        this->complete_pipeline(d_points, num_valid_points, d_new_rects, num_new_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });

        if (host_fallback) {
          this->split_output(d_new_rects, num_new_rects, host_rect_buffers, entry_counts, buffer_arena);
        }

        if (num_output==0 || host_fallback) {
          num_output = num_new_rects;
          num_completed += curr_tile;
          output_start = d_new_rects;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          curr_tile = tile_size / 2;
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          continue;
        }

        RectDesc<N, T>* d_old_rects = buffer_arena.alloc<RectDesc<N, T>>(num_output);
        assert(d_old_rects == d_new_rects + num_new_rects);
        CUDA_CHECK(cudaMemcpyAsync(d_old_rects, output_start, num_output * sizeof(RectDesc<N,T>), cudaMemcpyDeviceToDevice, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        size_t num_final_rects = 1;

        //Send it off for processing
        this->complete_rect_pipeline(d_new_rects, num_output + num_new_rects, output_start, num_final_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });
        num_completed += curr_tile;
        num_output = num_final_rects;
        subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
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
              this->split_output(output_start, num_output, host_rect_buffers, entry_counts, buffer_arena);
            }
            curr_tile = tile_size / 2;
          }
        }
      }
    }
    if (num_output == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
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
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });
      } catch (arena_oom&) {
        this->split_output(output_start, num_output, host_rect_buffers, entry_counts, buffer_arena);
        host_fallback = true;
      }
    }

    if (host_fallback) {
      for (size_t idx = 0; idx < sparsity_outputs.size(); ++idx) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(sparsity_outputs[idx]);
        if (this->exclusive) {
          impl->set_contributor_count(1);
        }
        if (entry_counts[idx] > 0) {
          span<Rect<N, T>> h_rects_span(host_rect_buffers[idx], entry_counts[idx]);
          impl->contribute_dense_rect_list(h_rects_span, true);
          deppart_host_free(host_rect_buffers[idx]);
        } else {
          impl->contribute_nothing();
        }
      }
    }
  }
}
