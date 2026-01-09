#pragma once
#include "realm/deppart/image.h"
#include "realm/deppart/image_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include "realm/nvtx.h"

namespace Realm {

//TODO: INTERSECTING INPUT/OUTPUT RECTS CAN BE DONE WITH BVH IF BECOME EXPENSIVE

template <int N2, typename T2>
struct RectDescVolumeOp {
  __device__ __forceinline__
  size_t operator()(const RectDesc<N2,T2>& rd) const {
    return rd.rect.volume();
  }
};

template <int N2, typename T2>
struct SparsityMapEntryVolumeOp {
  __device__ __forceinline__
  size_t operator()(const SparsityMapEntry<N2,T2>& entry) const {
    return entry.bounds.volume();
  }
};

  /*
   *  Input (stored in MicroOp): Array of field instances, a parent index space, and a list of source index spaces
   *  Output: A list of (potentially overlapping) rectangles that result from chasing all the pointers in the source index spaces
   *  through the provided instances and emitting only those that intersect the parent index space labeled by which source they came from,
   *  which are then sent off to complete_rect_pipeline.
   *  Approach: Intersect all instance rectangles with source rectangles in parallel. Prefix sum + binary search to iterate over these in
   *  parallel and chase all the pointers in the source rectangles to their corresponding rectangle. Finally, intersect the output rectangles
   *  with the parent rectangles in parallel.
   */
template <int N, typename T, int N2, typename T2>
void GPUImageMicroOp<N,T,N2,T2>::gpu_populate_rngs()
{

    if (sources.size() == 0) {
      return;
    }

    NVTX_DEPPART(gpu_image);

    Memory my_mem = domain_transform.range_data[0].inst.get_location();

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    const char* val = std::getenv("TILE_SIZE");  // or any env var
    size_t tile_size = 100000000; //default
    if (val) {
      tile_size = atoi(val);
    }

    RegionInstance fixed_buffer = this->realm_malloc(tile_size, my_mem);
    Arena buffer_arena(reinterpret_cast<void *>(AffineAccessor<char, 1>(fixed_buffer, 0).base), tile_size);

    collapsed_space<N2, T2> src_space;
    RegionInstance offsets_instance = this->realm_malloc((sources.size()+1) * sizeof(size_t), my_mem);
    src_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    src_space.num_children = sources.size();

    GPUMicroOp<N2, T2>::collapse_multi_space(sources, src_space, buffer_arena, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.range_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = domain_transform.range_data.size();
  
    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.range_data, inst_space, buffer_arena, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    RegionInstance inst_counters_instance = this->realm_malloc((2*domain_transform.range_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.range_data.size();
    RegionInstance valid_rects_instance;
    size_t num_valid_rects;
    RectDesc<N2, T2>* d_valid_rects;
  
    // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
    GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space, src_space, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);
    inst_offsets_instance.destroy();

    if (num_valid_rects == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
            SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
            if (this->exclusive) {
              impl->gpu_finalize();
            } else {
              impl->contribute_nothing();
            }
      }
      valid_rects_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    // Prefix sum the valid rectangles by volume.
    size_t* d_prefix_rects;
    size_t total_pts;

    GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

    RegionInstance rngs_instance = this->realm_malloc(total_pts * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N,T>* d_rngs = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(rngs_instance, 0).base);


    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N,T>,N2,T2>), zcpy_mem);
    AffineAccessor<Rect<N,T>,N2,T2>* d_accessors = reinterpret_cast<AffineAccessor<Rect<N,T>,N2,T2>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Rect<N,T>,N2,T2>(domain_transform.range_data[i].inst, domain_transform.range_data[i].field_offset);
    }

    image_gpuPopulateBitmasksRngsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, total_pts, num_valid_rects, domain_transform.range_data.size(), d_rngs);
    KERNEL_CHECK(stream);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);
  

    RegionInstance src_counters_instance = this->realm_malloc(sources.size() * sizeof(uint32_t), my_mem);
    uint32_t* d_src_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(src_counters_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);


    //Finally, we do another two pass count + emit to intersect with the parent rectangles
    image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_entries * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.entries_buffer, d_rngs, nullptr, collapsed_parent.num_entries, total_pts, d_src_counters, nullptr);
    KERNEL_CHECK(stream);

    std::vector<uint32_t> h_src_counters(sources.size()+1);
    h_src_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_src_counters.data()+1, d_src_counters, sources.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    valid_rects_instance.destroy();
    accessors_instance.destroy();

    for (size_t i = 0; i < sources.size(); ++i) {
      h_src_counters[i+1] += h_src_counters[i];
    }

    size_t num_valid_output = h_src_counters[sources.size()];

    if (num_valid_output == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        if (this->exclusive) {
          impl->gpu_finalize();
        } else {
          impl->contribute_nothing();
        }
      }
      parent_entries_instance.destroy();
      src_counters_instance.destroy();
      rngs_instance.destroy();
      return;
    }


    RegionInstance valid_intersect_instance = this->realm_malloc(num_valid_output * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N,T>* d_valid_intersect = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(valid_intersect_instance, 0).base);

    RegionInstance src_prefix_instance = this->realm_malloc((sources.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_src_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(src_prefix_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_src_prefix, h_src_counters.data(), (sources.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);

    image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_entries * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.entries_buffer, d_rngs, d_src_prefix, collapsed_parent.num_entries, total_pts, d_src_counters, d_valid_intersect);
    KERNEL_CHECK(stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    src_prefix_instance.destroy();
    parent_entries_instance.destroy();
    src_counters_instance.destroy();
    rngs_instance.destroy();

    size_t out_rects = 0;
    RectDesc<N, T>* trash;
    this->complete_rect_pipeline(d_valid_intersect, num_valid_output, trash, out_rects, buffer_arena,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

  valid_intersect_instance.destroy();

}

  /*
   *  Input (stored in MicroOp): Array of field instances, a parent index space, and a list of source index spaces
   *  Output: A list of (potentially overlapping) points that result from chasing all the pointers in the source index spaces
   *  through the provided instances and emitting only points in the parent index space labeled by which source they came from,
   *  which are then sent off to complete_pipeline.
   *  Approach: Intersect all instance rectangles with source rectangles in parallel. Prefix sum + binary search to iterate over these in
   *  parallel and chase all the pointers in the source rectangles to their corresponding point. Here, the pointer chasing is also a count + emit,
   *  where only points that are in the parent index space are counted/emitted.
   */
template <int N, typename T, int N2, typename T2>
void GPUImageMicroOp<N,T,N2,T2>::gpu_populate_ptrs()
{
    if (sources.size() == 0) {
      return;
    }

    NVTX_DEPPART(gpu_image);

    Memory sysmem;
    find_memory(sysmem, Memory::SYSTEM_MEM);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    size_t tile_size = fixed_buffer_size;
    RegionInstance fixed_buffer = buffer;
    Arena buffer_arena(reinterpret_cast<void *>(AffineAccessor<char, 1>(fixed_buffer, 0).base), tile_size);

    collapsed_space<N2, T2> src_space;
    src_space.offsets = buffer_arena.alloc<size_t>(sources.size()+1);
    src_space.num_children = sources.size();

    GPUMicroOp<N2, T2>::collapse_multi_space(sources, src_space, buffer_arena, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    inst_space.offsets = buffer_arena.alloc<size_t>(domain_transform.ptr_data.size()+1);
    inst_space.num_children = domain_transform.ptr_data.size();

    Arena no;
    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.ptr_data, inst_space, no, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2*domain_transform.ptr_data.size()+1);

  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.ptr_data.size();
    size_t num_valid_rects = tile_size;

    //Uniform for all tiles
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);

    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N,T>,N2,T2>), zcpy_mem);
    AffineAccessor<Point<N,T>,N2,T2>* d_accessors = reinterpret_cast<AffineAccessor<Point<N,T>,N2,T2>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Point<N,T>,N2,T2>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }

    uint32_t* d_prefix_points = buffer_arena.alloc<uint32_t>(domain_transform.ptr_data.size()+1);

    buffer_arena.commit(false);

    size_t left = buffer_arena.used();

    //Here we iterate over the tiles
    size_t num_output = 0;
    RectDesc<N, T>* output_start = nullptr;
    size_t num_completed = 0;
    size_t curr_tile = tile_size / 2;
    int count = 0;
    while (num_completed < inst_space.num_entries) {
      try {
        std::cout << "Tile iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_entries << " entries." << std::endl;
        buffer_arena.start();
        std::cout << "Amount Used: " << buffer_arena.used() << std::endl;
        std::cout << "Expected Amount Used: " << left + num_output * sizeof(RectDesc<N,T>) << std::endl;
        if (num_completed + curr_tile > inst_space.num_entries) {
          curr_tile = inst_space.num_entries - num_completed;
        }
        collapsed_space<N2, T2> inst_space_tile = inst_space;
        inst_space_tile.num_entries = curr_tile;
        inst_space_tile.entries_buffer = buffer_arena.alloc<SparsityMapEntry<N2,T2>>(curr_tile);
        CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.entries_buffer, inst_space.entries_buffer + num_completed, curr_tile * sizeof(SparsityMapEntry<N2,T2>), cudaMemcpyHostToDevice, stream), stream);

        RectDesc<N2, T2>* d_valid_rects;
        GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space_tile, src_space, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);

        if (num_valid_rects == std::numeric_limits<size_t>::max()) {
          curr_tile /= 2;
          continue;
        }

        if (num_valid_rects == 0) {
          num_completed += curr_tile;
          curr_tile = tile_size / 2;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          continue;
        }

        // Prefix sum the valid rectangles by volume.
        size_t* d_prefix_rects;
        size_t total_pts;

        GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

        CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

        //We do a two pass count + emit to chase all the pointers in parallel and check for membership in the parent index space
        image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.entries_buffer, d_prefix_rects, d_inst_prefix, nullptr, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_entries, d_inst_counters, nullptr);
        KERNEL_CHECK(stream);

        std::vector<uint32_t> h_inst_counters(domain_transform.ptr_data.size()+1);
        h_inst_counters[0] = 0; // prefix sum starts at 0
        CUDA_CHECK(cudaMemcpyAsync(h_inst_counters.data()+1, d_inst_counters, domain_transform.ptr_data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
          h_inst_counters[i+1] += h_inst_counters[i];
        }

        size_t num_valid_points = h_inst_counters[domain_transform.ptr_data.size()];

        if (num_valid_points == 0) {
          num_completed += curr_tile;
          curr_tile = tile_size / 2;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          continue;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_prefix_points, h_inst_counters.data(), (domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

        buffer_arena.flip_parity();
        PointDesc<N,T>* d_valid_points = buffer_arena.alloc<PointDesc<N,T>>(num_valid_points);

        CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

        image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.entries_buffer, d_prefix_rects, d_inst_prefix, d_prefix_points, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_entries, d_inst_counters, d_valid_points);
        KERNEL_CHECK(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);


        size_t num_new_rects = 1;
        assert(!buffer_arena.get_parity());
        RectDesc<N, T>* d_new_rects;

        //Send it off for processing
        this->complete_pipeline(d_valid_points, num_valid_points, d_new_rects, num_new_rects, buffer_arena,
        /* the Container: */  sparsity_outputs,
        /* getIndex: */       [&](auto const& elem){
                                // elem is a SparsityMap<N,T> from the vector
                                return size_t(&elem - sparsity_outputs.data());
                             },
        /* getMap: */         [&](auto const& elem){
                              // return the SparsityMap key itself
                              return elem;
                           });

        if (num_output==0) {
          buffer_arena.flip_parity();
          buffer_arena.reset(true);
          output_start = buffer_arena.alloc<RectDesc<N, T>>(num_new_rects);
          buffer_arena.commit(true);
          CUDA_CHECK(cudaMemcpyAsync(output_start, d_new_rects, num_new_rects * sizeof(RectDesc<N,T>), cudaMemcpyDeviceToDevice, stream), stream);
          num_output = num_new_rects;
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
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
        curr_tile = tile_size / 2;
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      }
      catch (arena_oom&) {
        std::cout << "Caught arena_oom, reducing tile size from " << curr_tile << " to " << curr_tile / 2 << std::endl;
        std::cout << buffer_arena.used() << " bytes used in arena." << std::endl;
        curr_tile /= 2;
        if (curr_tile == 0) {
          throw;
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
    this->send_output(output_start, num_output, buffer_arena, sparsity_outputs,
      /* getIndex: */       [&](auto const& elem){
                              // elem is a SparsityMap<N,T> from the vector
                              return size_t(&elem - sparsity_outputs.data());
                           },
      /* getMap: */         [&](auto const& elem){
                            // return the SparsityMap key itself
                            return elem;
                         });
}
}