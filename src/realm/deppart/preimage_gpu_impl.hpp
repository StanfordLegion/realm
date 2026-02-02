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
      return;
    }

    Memory my_mem = domain_transform.range_data[0].inst.get_location();

    const char* val = std::getenv("TILE_SIZE");  // or any env var
    size_t tile_size = 100000000; //default
    if (val) {
      tile_size = atoi(val);
    }

    RegionInstance fixed_buffer = this->realm_malloc(tile_size, my_mem);
    Arena buffer_arena(reinterpret_cast<void *>(AffineAccessor<char, 1>(fixed_buffer, 0).base), tile_size);

    NVTX_DEPPART(gpu_preimage);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    collapsed_space<N, T> inst_space;

    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.range_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = domain_transform.range_data.size();

    RegionInstance inst_entries_instance;

    GPUMicroOp<N, T>::collapse_multi_space(domain_transform.range_data, inst_space, buffer_arena, stream);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);


    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    RegionInstance inst_counters_instance = this->realm_malloc((2*domain_transform.range_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);

    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.range_data.size();
    RegionInstance out_instance;
    size_t num_valid_rects;

    Rect<N, T>* d_valid_rects;

    // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
    GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space, collapsed_parent, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);
    inst_entries_instance.destroy();
    parent_entries_instance.destroy();
    inst_offsets_instance.destroy();

    if (num_valid_rects == 0) {
      for (auto it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        if (this->exclusive) {
          impl->gpu_finalize();
        } else {
          impl->contribute_nothing();
        }
      }
      out_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    // Prefix sum the valid rectangles by volume.
    RegionInstance prefix_rects_instance;
    size_t total_pts;

    size_t* d_prefix_rects;
    GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

    nvtx_range_push("cuda", "build target entries");

    collapsed_space<N2, T2> target_space;
    RegionInstance offsets_instance = this->realm_malloc((targets.size()+1) * sizeof(size_t), my_mem);
    target_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    target_space.num_children = targets.size();

    RegionInstance targets_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(targets, target_space, buffer_arena, stream);

    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N2,T2>,N,T>), zcpy_mem);
    AffineAccessor<Rect<N2,T2>,N,T>* d_accessors = reinterpret_cast<AffineAccessor<Rect<N2,T2>,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Rect<N2,T2>,N,T>(domain_transform.range_data[i].inst, domain_transform.range_data[i].field_offset);
    }

    RegionInstance points_instance;
    PointDesc<N,T>* d_points;
    size_t num_valid_points;

    RegionInstance target_counters_instance = this->realm_malloc((2*targets.size()+1) * sizeof(uint32_t), my_mem);
    uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
    uint32_t* d_targets_prefix = d_target_counters + targets.size();
    CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

    if (target_space.num_entries > targets.size()) {
      BVH<N2, T2> preimage_bvh;
      RegionInstance bvh_instance;
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
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          if (this->exclusive) {
            impl->gpu_finalize();
          } else {
            impl->contribute_nothing();
          }
        }
        target_counters_instance.destroy();
        accessors_instance.destroy();
        targets_entries_instance.destroy();
        offsets_instance.destroy();
        prefix_rects_instance.destroy();
        out_instance.destroy();
        inst_counters_instance.destroy();
        bvh_instance.destroy();
        return;
      }

      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
       preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.range_data.size(), preimage_bvh.num_leaves, d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      bvh_instance.destroy();
    } else {
      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
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
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          if (this->exclusive) {
            impl->gpu_finalize();
          } else {
            impl->contribute_nothing();
          }
        }
        target_counters_instance.destroy();
        accessors_instance.destroy();
        targets_entries_instance.destroy();
        offsets_instance.destroy();
        prefix_rects_instance.destroy();
        out_instance.destroy();
        inst_counters_instance.destroy();
        return;
      }

      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
       num_valid_rects, domain_transform.range_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }

    target_counters_instance.destroy();
    accessors_instance.destroy();
    targets_entries_instance.destroy();
    offsets_instance.destroy();
    prefix_rects_instance.destroy();
    out_instance.destroy();
    inst_counters_instance.destroy();

    size_t out_rects = 0;
    RectDesc<N, T>* trash;
    this->complete_pipeline(d_points, num_valid_points, trash, out_rects, buffer_arena,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

    points_instance.destroy();
  }

  template<int N, typename T, int N2, typename T2>
  void GPUPreimageMicroOp<N, T, N2, T2>::gpu_populate_bitmasks() {
    if (targets.size() == 0) {
      return;
    }

    Memory my_mem = domain_transform.ptr_data[0].inst.get_location();

    const char* val = std::getenv("TILE_SIZE");  // or any env var
    size_t tile_size = 100000000; //default
    if (val) {
      tile_size = atoi(val);
    }

    RegionInstance fixed_buffer = this->realm_malloc(tile_size, my_mem);
    Arena buffer_arena(reinterpret_cast<void *>(AffineAccessor<char, 1>(fixed_buffer, 0).base), tile_size);

    NVTX_DEPPART(gpu_preimage);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    collapsed_space<N, T> inst_space;

    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.ptr_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = domain_transform.ptr_data.size();

    RegionInstance inst_entries_instance;

    GPUMicroOp<N, T>::collapse_multi_space(domain_transform.ptr_data, inst_space, buffer_arena, stream);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);


    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    RegionInstance inst_counters_instance = this->realm_malloc((2*domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);

    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.ptr_data.size();
    RegionInstance out_instance;
    size_t num_valid_rects;

    Rect<N, T>* d_valid_rects;
    // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
    GPUMicroOp<N, T>::template construct_input_rectlist<Rect<N, T>>(inst_space, collapsed_parent, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);
    inst_entries_instance.destroy();
    parent_entries_instance.destroy();
    inst_offsets_instance.destroy();

    if (num_valid_rects == 0) {
      for (auto it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        if (this->exclusive) {
          impl->gpu_finalize();
        } else {
          impl->contribute_nothing();
        }
      }
      out_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    // Prefix sum the valid rectangles by volume.
    RegionInstance prefix_rects_instance;
    size_t total_pts;

    size_t* d_prefix_rects;
    GPUMicroOp<N, T>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

    nvtx_range_push("cuda", "build target entries");

    collapsed_space<N2, T2> target_space;
    RegionInstance offsets_instance = this->realm_malloc((targets.size()+1) * sizeof(size_t), my_mem);
    target_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    target_space.num_children = targets.size();

    RegionInstance targets_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(targets, target_space, buffer_arena, stream);

    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N2,T2>,N,T>), zcpy_mem);
    AffineAccessor<Point<N2,T2>,N,T>* d_accessors = reinterpret_cast<AffineAccessor<Point<N2,T2>,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Point<N2,T2>,N,T>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }

    RegionInstance points_instance;
    PointDesc<N,T>* d_points;
    size_t num_valid_points;

    RegionInstance target_counters_instance = this->realm_malloc((2*targets.size()+1) * sizeof(uint32_t), my_mem);
    uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
    uint32_t* d_targets_prefix = d_target_counters + targets.size();
    CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

    if (target_space.num_entries > targets.size()) {
      BVH<N2, T2> preimage_bvh;
      RegionInstance bvh_instance;
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
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          if (this->exclusive) {
            impl->gpu_finalize();
          } else {
            impl->contribute_nothing();
          }
        }
        target_counters_instance.destroy();
        accessors_instance.destroy();
        targets_entries_instance.destroy();
        offsets_instance.destroy();
        prefix_rects_instance.destroy();
        out_instance.destroy();
        inst_counters_instance.destroy();
        bvh_instance.destroy();
        return;
      }

      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, preimage_bvh.root, preimage_bvh.childLeft, preimage_bvh.childRight, preimage_bvh.indices,
       preimage_bvh.labels, preimage_bvh.boxes, total_pts, num_valid_rects, domain_transform.ptr_data.size(), preimage_bvh.num_leaves, d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      bvh_instance.destroy();
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
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          if (this->exclusive) {
            impl->gpu_finalize();
          } else {
            impl->contribute_nothing();
          }
        }
        target_counters_instance.destroy();
        accessors_instance.destroy();
        targets_entries_instance.destroy();
        offsets_instance.destroy();
        prefix_rects_instance.destroy();
        out_instance.destroy();
        inst_counters_instance.destroy();
        return;
      }

      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters.data(), (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_dense_populate_bitmasks_kernel< N, T, N2, T2 ><<< COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, target_space.entries_buffer, target_space.offsets, total_pts,
       num_valid_rects, domain_transform.ptr_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    }

    target_counters_instance.destroy();
    accessors_instance.destroy();
    targets_entries_instance.destroy();
    offsets_instance.destroy();
    prefix_rects_instance.destroy();
    out_instance.destroy();
    inst_counters_instance.destroy();

    size_t out_rects = 0;
    RectDesc<N, T>* trash;
    this->complete_pipeline(d_points, num_valid_points, trash, out_rects, buffer_arena,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

    points_instance.destroy();
  }
}