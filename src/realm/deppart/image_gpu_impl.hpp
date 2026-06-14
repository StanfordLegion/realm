#pragma once
#include "realm/deppart/image.h"
#include "realm/deppart/image_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include "realm/nvtx.h"

namespace Realm {

extern Logger log_part;

//TODO: INTERSECTING INPUT/OUTPUT RECTS CAN BE DONE WITH BVH IF BECOME EXPENSIVE

template <int N2, typename T2>
struct RectDescVolumeOp {
  __device__ __forceinline__
  size_t operator()(const RectDesc<N2,T2>& rd) const {
    return rd.rect.volume();
  }
};

template <int N, typename T, int N2, typename T2>
void GPUApproxImageMicroOp<N,T,N2,T2>::send_approx_output(
    const std::vector<Rect<N,T> > &approx_rects)
{
  if(approx_output_receiver != nullptr) {
    approx_output_receiver->provide_approx_image(approx_output_index,
                                                 approx_rects.empty() ? nullptr : approx_rects.data(),
                                                 approx_rects.size());
  }
}

template <int N, typename T, int N2, typename T2>
void GPUApproxImageMicroOp<N,T,N2,T2>::approximate_rects(
    RectDesc<N,T> *d_rects, size_t num_rects, std::vector<Rect<N,T> > &approx_rects,
    Arena &buffer_arena)
{
  if(num_rects == 0) return;

  CUstream stream = this->stream->get_stream();
  const size_t max_rects =
      std::max<size_t>(1, DeppartConfig::cfg_max_rects_in_approximation);

  RectDesc<N,T> *d_merged = nullptr;
  size_t num_merged = 1;
  (void)d_merged;
  (void)num_merged;

  if constexpr (N == 1) {
    std::vector<int> dummy;
    this->complete1d_pipeline(d_rects, num_rects, d_merged, num_merged, buffer_arena,
      dummy,
      [&](auto const& elem) { return size_t(&elem - dummy.data()); },
      [&](auto const& elem) { return SparsityMap<N,T>(); });

    RectDesc<N,T> *d_approx = d_merged;
    size_t num_approx = num_merged;
    if(num_merged > max_rects) {
      size_t num_gaps = num_merged - 1;
      size_t num_kept = max_rects - 1;
      T *d_gap_keys_in = buffer_arena.alloc<T>(num_gaps);
      T *d_gap_keys_out = buffer_arena.alloc<T>(num_gaps);
      size_t *d_gap_indices_in = buffer_arena.alloc<size_t>(num_gaps);
      size_t *d_gap_indices_out = buffer_arena.alloc<size_t>(num_gaps);

      image_buildGapKeys1d<N,T><<<COMPUTE_GRID(num_gaps), THREADS_PER_BLOCK, 0, stream>>>(
          d_merged, d_gap_keys_in, d_gap_indices_in, num_gaps);
      KERNEL_CHECK(stream);

      size_t temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                      d_gap_keys_in, d_gap_keys_out,
                                      d_gap_indices_in, d_gap_indices_out,
                                      num_gaps, 0, 8 * sizeof(T), stream);
      void *temp_storage = buffer_arena.alloc<char>(temp_bytes);
      cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes,
                                      d_gap_keys_in, d_gap_keys_out,
                                      d_gap_indices_in, d_gap_indices_out,
                                      num_gaps, 0, 8 * sizeof(T), stream);

      size_t *d_kept_indices_in = buffer_arena.alloc<size_t>(num_kept);
      size_t *d_kept_indices_out = buffer_arena.alloc<size_t>(num_kept);
      image_copyTopGapIndices<size_t><<<COMPUTE_GRID(num_kept), THREADS_PER_BLOCK, 0, stream>>>(
          d_gap_indices_out, d_kept_indices_in, num_gaps, num_kept);
      KERNEL_CHECK(stream);

      temp_bytes = 0;
      cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes,
                                     d_kept_indices_in, d_kept_indices_out,
                                     num_kept, 0, 8 * sizeof(size_t), stream);
      temp_storage = buffer_arena.alloc<char>(temp_bytes);
      cub::DeviceRadixSort::SortKeys(temp_storage, temp_bytes,
                                     d_kept_indices_in, d_kept_indices_out,
                                     num_kept, 0, 8 * sizeof(size_t), stream);

      d_approx = buffer_arena.alloc<RectDesc<N,T> >(max_rects);
      image_emitApproxRects1d<N,T><<<COMPUTE_GRID(max_rects), THREADS_PER_BLOCK, 0, stream>>>(
          d_merged, d_kept_indices_out, num_kept, num_merged, d_approx);
      KERNEL_CHECK(stream);
      num_approx = max_rects;
    }

    std::vector<RectDesc<N,T> > h_rect_descs(num_approx);
    CUDA_CHECK(cudaMemcpyAsync(h_rect_descs.data(), d_approx,
                               num_approx * sizeof(RectDesc<N,T>),
                               cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for(size_t i = 0; i < num_approx; i++)
      approx_rects.push_back(h_rect_descs[i].rect);
  } else {
    Rect<N,T> identity_rect;
    for(int d = 0; d < N; d++) {
      identity_rect.lo[d] = std::numeric_limits<T>::max();
      identity_rect.hi[d] = std::numeric_limits<T>::min();
    }

    Rect<N,T> *d_plain_rects = buffer_arena.alloc<Rect<N,T> >(num_rects);
    image_rectDescsToRects<N,T><<<COMPUTE_GRID(num_rects), THREADS_PER_BLOCK, 0, stream>>>(
        d_rects, d_plain_rects, num_rects);
    KERNEL_CHECK(stream);

    Rect<N,T> *d_bbox = buffer_arena.alloc<Rect<N,T> >(1);
    void *temp_storage = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Reduce(temp_storage, temp_bytes,
                              d_plain_rects, d_bbox, num_rects,
                              UnionRectOp<N,T>(), identity_rect, stream);
    temp_storage = buffer_arena.alloc<char>(temp_bytes);
    cub::DeviceReduce::Reduce(temp_storage, temp_bytes,
                              d_plain_rects, d_bbox, num_rects,
                              UnionRectOp<N,T>(), identity_rect, stream);
    Rect<N,T> h_bbox;
    CUDA_CHECK(cudaMemcpyAsync(&h_bbox, d_bbox, sizeof(Rect<N,T>),
                               cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    approx_rects.push_back(h_bbox);
  }
}

template <int N, typename T, int N2, typename T2>
void GPUApproxImageMicroOp<N,T,N2,T2>::approximate_points(
    PointDesc<N,T> *d_points, size_t num_points, std::vector<Rect<N,T> > &approx_rects,
    Arena &buffer_arena)
{
  if(num_points == 0) return;

  CUstream stream = this->stream->get_stream();
  RectDesc<N,T> *d_rects = buffer_arena.alloc<RectDesc<N,T> >(num_points);
  points_to_rects<N,T><<<COMPUTE_GRID(num_points), THREADS_PER_BLOCK, 0, stream>>>(
      d_points, d_rects, num_points);
  KERNEL_CHECK(stream);

  approximate_rects(d_rects, num_points, approx_rects, buffer_arena);
}

template <int N, typename T, int N2, typename T2>
void GPUApproxImageMicroOp<N,T,N2,T2>::gpu_populate_ptrs()
{
  NVTX_DEPPART(gpu_approx_image);

  RegionInstance buffer = scratch_buffer;
  size_t tile_size = validated_scratch_bytes();
  Arena buffer_arena(buffer);
  CUstream stream = this->stream->get_stream();

  FieldDataDescriptor<IndexSpace<N2,T2>, Point<N,T> > field_data;
  field_data.index_space = inst_space;
  field_data.inst = inst;
  field_data.field_offset = field_offset;
  field_data.scratch_buffer = scratch_buffer;
  std::vector<FieldDataDescriptor<IndexSpace<N2,T2>, Point<N,T> > > fields(1, field_data);

  collapsed_space<N2,T2> collapsed_inst;
  collapsed_inst.offsets = buffer_arena.alloc<size_t>(2);
  collapsed_inst.num_children = 1;
  Arena sys_arena;
  GPUMicroOp<N2,T2>::collapse_multi_space(fields, collapsed_inst, sys_arena, stream);

  collapsed_space<N,T> collapsed_parent;
  GPUMicroOp<N,T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);

  AffineAccessor<Point<N,T>,N2,T2> accessor(inst, field_offset);
  uint32_t *d_counter = buffer_arena.alloc<uint32_t>(1);

  buffer_arena.commit(false);

  DenseRectangleList<N,T> final_rects(DeppartConfig::cfg_max_rects_in_approximation);
  Rect<N,T> nd_bbox = Rect<N,T>::make_empty();
  bool nd_bbox_valid = false;
  (void)nd_bbox;
  (void)nd_bbox_valid;
  size_t num_completed = 0;
  size_t curr_tile = std::max<size_t>(1, tile_size / 2);

  while(num_completed < collapsed_inst.num_rects) {
    try {
      buffer_arena.start();
      if(num_completed + curr_tile > collapsed_inst.num_rects)
        curr_tile = collapsed_inst.num_rects - num_completed;

      Rect<N2,T2> *d_tile_rects = buffer_arena.alloc<Rect<N2,T2> >(curr_tile);
      CUDA_CHECK(cudaMemcpyAsync(d_tile_rects, collapsed_inst.rects_buffer + num_completed,
                                 curr_tile * sizeof(Rect<N2,T2>),
                                 cudaMemcpyHostToDevice, stream), stream);

      size_t *d_prefix_rects = nullptr;
      size_t total_pts = 0;
      GPUMicroOp<N2,T2>::volume_prefix_sum(d_tile_rects, curr_tile, d_prefix_rects,
                                           total_pts, buffer_arena, stream);
      if(total_pts == 0) {
        num_completed += curr_tile;
        curr_tile = std::max<size_t>(1, tile_size / 2);
        continue;
      }

      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t), stream), stream);
      image_gpuApproxPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(
          accessor, d_tile_rects, collapsed_parent.rects_buffer, d_prefix_rects,
          total_pts, curr_tile, collapsed_parent.num_rects, d_counter, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_count = 0;
      CUDA_CHECK(cudaMemcpyAsync(&h_count, d_counter, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      if(h_count == 0) {
        num_completed += curr_tile;
        curr_tile = std::max<size_t>(1, tile_size / 2);
        continue;
      }

      PointDesc<N,T> *d_points = buffer_arena.alloc<PointDesc<N,T> >(h_count);
      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t), stream), stream);
      image_gpuApproxPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(
          accessor, d_tile_rects, collapsed_parent.rects_buffer, d_prefix_rects,
          total_pts, curr_tile, collapsed_parent.num_rects, d_counter, d_points);
      KERNEL_CHECK(stream);

      std::vector<Rect<N,T> > tile_rects;
      approximate_points(d_points, h_count, tile_rects, buffer_arena);
      if constexpr (N == 1) {
        for(size_t i = 0; i < tile_rects.size(); i++)
          final_rects.add_rect(tile_rects[i]);
      } else {
        for(size_t i = 0; i < tile_rects.size(); i++) {
          if(!nd_bbox_valid) {
            nd_bbox = tile_rects[i];
            nd_bbox_valid = true;
          } else
            nd_bbox = nd_bbox.union_bbox(tile_rects[i]);
        }
      }

      num_completed += curr_tile;
      curr_tile = std::max<size_t>(1, tile_size / 2);
    } catch(arena_oom&) {
      curr_tile /= 2;
      if(curr_tile == 0) {
        if(collapsed_inst.rects_buffer[num_completed].volume() <= 1) {
          log_part.fatal()
              << "GPUApproxImageMicroOp cannot make progress after scratch exhaustion";
          std::abort();
        }
        size_t old_rects = collapsed_inst.num_rects;
        GPUMicroOp<N2,T2>::shatter_rects(collapsed_inst, num_completed, stream);
        if(collapsed_inst.num_rects <= old_rects) {
          log_part.fatal()
              << "GPUApproxImageMicroOp shatter did not increase available tiling";
          std::abort();
        }
        curr_tile = 1;
      }
    }
  }

  if constexpr (N == 1) {
    send_approx_output(final_rects.rects);
  } else {
    std::vector<Rect<N,T> > out;
    if(nd_bbox_valid) out.push_back(nd_bbox);
    send_approx_output(out);
  }
}

template <int N, typename T, int N2, typename T2>
void GPUApproxImageMicroOp<N,T,N2,T2>::gpu_populate_rngs()
{
  NVTX_DEPPART(gpu_approx_image_range);

  RegionInstance buffer = scratch_buffer;
  size_t tile_size = validated_scratch_bytes();
  Arena buffer_arena(buffer);
  CUstream stream = this->stream->get_stream();

  FieldDataDescriptor<IndexSpace<N2,T2>, Rect<N,T> > field_data;
  field_data.index_space = inst_space;
  field_data.inst = inst;
  field_data.field_offset = field_offset;
  field_data.scratch_buffer = scratch_buffer;
  std::vector<FieldDataDescriptor<IndexSpace<N2,T2>, Rect<N,T> > > fields(1, field_data);

  collapsed_space<N2,T2> collapsed_inst;
  collapsed_inst.offsets = buffer_arena.alloc<size_t>(2);
  collapsed_inst.num_children = 1;
  Arena sys_arena;
  GPUMicroOp<N2,T2>::collapse_multi_space(fields, collapsed_inst, sys_arena, stream);

  collapsed_space<N,T> collapsed_parent;
  GPUMicroOp<N,T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);

  AffineAccessor<Rect<N,T>,N2,T2> accessor(inst, field_offset);
  uint32_t *d_counter = buffer_arena.alloc<uint32_t>(1);

  buffer_arena.commit(false);

  DenseRectangleList<N,T> final_rects(DeppartConfig::cfg_max_rects_in_approximation);
  Rect<N,T> nd_bbox = Rect<N,T>::make_empty();
  bool nd_bbox_valid = false;
  (void)nd_bbox;
  (void)nd_bbox_valid;
  size_t num_completed = 0;
  size_t curr_tile = std::max<size_t>(1, tile_size / 2);

  while(num_completed < collapsed_inst.num_rects) {
    try {
      buffer_arena.start();
      if(num_completed + curr_tile > collapsed_inst.num_rects)
        curr_tile = collapsed_inst.num_rects - num_completed;

      Rect<N2,T2> *d_tile_rects = buffer_arena.alloc<Rect<N2,T2> >(curr_tile);
      CUDA_CHECK(cudaMemcpyAsync(d_tile_rects, collapsed_inst.rects_buffer + num_completed,
                                 curr_tile * sizeof(Rect<N2,T2>),
                                 cudaMemcpyHostToDevice, stream), stream);

      size_t *d_prefix_rects = nullptr;
      size_t total_pts = 0;
      GPUMicroOp<N2,T2>::volume_prefix_sum(d_tile_rects, curr_tile, d_prefix_rects,
                                           total_pts, buffer_arena, stream);
      if(total_pts == 0) {
        num_completed += curr_tile;
        curr_tile = std::max<size_t>(1, tile_size / 2);
        continue;
      }

      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t), stream), stream);
      image_gpuApproxRngsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(
          accessor, d_tile_rects, collapsed_parent.rects_buffer, d_prefix_rects,
          total_pts, curr_tile, collapsed_parent.num_rects, d_counter, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_count = 0;
      CUDA_CHECK(cudaMemcpyAsync(&h_count, d_counter, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      if(h_count == 0) {
        num_completed += curr_tile;
        curr_tile = std::max<size_t>(1, tile_size / 2);
        continue;
      }

      RectDesc<N,T> *d_rects = buffer_arena.alloc<RectDesc<N,T> >(h_count);
      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t), stream), stream);
      image_gpuApproxRngsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(
          accessor, d_tile_rects, collapsed_parent.rects_buffer, d_prefix_rects,
          total_pts, curr_tile, collapsed_parent.num_rects, d_counter, d_rects);
      KERNEL_CHECK(stream);

      std::vector<Rect<N,T> > tile_rects;
      approximate_rects(d_rects, h_count, tile_rects, buffer_arena);
      if constexpr (N == 1) {
        for(size_t i = 0; i < tile_rects.size(); i++)
          final_rects.add_rect(tile_rects[i]);
      } else {
        for(size_t i = 0; i < tile_rects.size(); i++) {
          if(!nd_bbox_valid) {
            nd_bbox = tile_rects[i];
            nd_bbox_valid = true;
          } else
            nd_bbox = nd_bbox.union_bbox(tile_rects[i]);
        }
      }

      num_completed += curr_tile;
      curr_tile = std::max<size_t>(1, tile_size / 2);
    } catch(arena_oom&) {
      curr_tile /= 2;
      if(curr_tile == 0) {
        if(collapsed_inst.rects_buffer[num_completed].volume() <= 1) {
          log_part.fatal()
              << "GPUApproxImageMicroOp cannot make progress after scratch exhaustion";
          std::abort();
        }
        size_t old_rects = collapsed_inst.num_rects;
        GPUMicroOp<N2,T2>::shatter_rects(collapsed_inst, num_completed, stream);
        if(collapsed_inst.num_rects <= old_rects) {
          log_part.fatal()
              << "GPUApproxImageMicroOp shatter did not increase available tiling";
          std::abort();
        }
        curr_tile = 1;
      }
    }
  }

  if constexpr (N == 1) {
    send_approx_output(final_rects.rects);
  } else {
    std::vector<Rect<N,T> > out;
    if(nd_bbox_valid) out.push_back(nd_bbox);
    send_approx_output(out);
  }
}

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
      assert(sparsity_outputs.empty());
      return;
    }

    NVTX_DEPPART(gpu_image_range);

    RegionInstance buffer = domain_transform.range_data[0].scratch_buffer;
    size_t tile_size = buffer.get_layout()->bytes_used;
    //std::cout << "Using tile size of " << tile_size << " bytes." << std::endl;
    Arena buffer_arena(buffer);

    CUstream stream = this->stream->get_stream();

    collapsed_space<N2, T2> src_space;
    src_space.offsets = buffer_arena.alloc<size_t>(sources.size()+1);
    src_space.num_children = sources.size();
    GPUMicroOp<N2, T2>::collapse_multi_space(sources, src_space, buffer_arena, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    inst_space.offsets = buffer_arena.alloc<size_t>(domain_transform.range_data.size() + 1);
    inst_space.num_children = domain_transform.range_data.size();
  
    Arena sys_arena;
    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.range_data, inst_space, sys_arena, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2 * domain_transform.range_data.size()+1);
  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.range_data.size();

    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);

    std::vector<AffineAccessor<Rect<N,T>,N2,T2>> h_accessors(domain_transform.range_data.size());
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Rect<N,T>,N2,T2>(domain_transform.range_data[i].inst, domain_transform.range_data[i].field_offset);
    }
    AffineAccessor<Rect<N,T>,N2,T2>* d_accessors =
        buffer_arena.alloc<AffineAccessor<Rect<N,T>,N2,T2>>(domain_transform.range_data.size());
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(),
                               domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N,T>,N2,T2>),
                               cudaMemcpyHostToDevice, stream), stream);

    uint32_t* d_src_counters = buffer_arena.alloc<uint32_t>(2 * sources.size() + 1);
    uint32_t* d_src_prefix = d_src_counters + sources.size();

    buffer_arena.commit(false);

    size_t num_output = 0;
    RectDesc<N, T>* output_start = nullptr;
    size_t num_completed = 0;
    size_t curr_tile = tile_size / 2;
    int count = 0;
    if (count) {}
    bool host_fallback = false;
    std::vector<Rect<N, T>*> host_rect_buffers(sources.size(), nullptr);
    std::vector<size_t> rect_counts(sources.size(), 0);
    while (num_completed < inst_space.num_rects) {
      try {
        //std::cout << "Image Range iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_rects << " rects." << std::endl;
        buffer_arena.start();
        buffer_arena.flip_parity();
        if (num_completed + curr_tile > inst_space.num_rects) {
          curr_tile = inst_space.num_rects - num_completed;
        }
        collapsed_space<N2, T2> inst_space_tile = inst_space;
        inst_space_tile.num_rects = curr_tile;
        inst_space_tile.rects_buffer = buffer_arena.alloc<Rect<N2,T2>>(curr_tile);
        CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.rects_buffer, inst_space.rects_buffer + num_completed, curr_tile * sizeof(Rect<N2,T2>), cudaMemcpyHostToDevice, stream), stream);

        size_t num_valid_rects;
        RectDesc<N2, T2>* d_valid_rects;
        // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
        GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space_tile, src_space, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);

        if (num_valid_rects == 0) {
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        // Prefix sum the valid rectangles by volume.
        size_t* d_prefix_rects;
        size_t total_pts;

        GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

        buffer_arena.flip_parity();
        RectDesc<N,T>* d_rngs = buffer_arena.alloc<RectDesc<N,T>>(total_pts);

        image_gpuPopulateBitmasksRngsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, total_pts, num_valid_rects, domain_transform.range_data.size(), d_rngs);
        KERNEL_CHECK(stream);


        CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);


        //Finally, we do another two pass count + emit to intersect with the parent rectangles
        image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_rects * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.rects_buffer, d_rngs, nullptr, collapsed_parent.num_rects, total_pts, d_src_counters, nullptr);
        KERNEL_CHECK(stream);

        std::vector<uint32_t> h_src_counters(sources.size()+1);
        h_src_counters[0] = 0; // prefix sum starts at 0
        CUDA_CHECK(cudaMemcpyAsync(h_src_counters.data()+1, d_src_counters, sources.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        for (size_t i = 0; i < sources.size(); ++i) {
          h_src_counters[i+1] += h_src_counters[i];
        }

        size_t num_valid_output = h_src_counters[sources.size()];

        if (num_valid_output == 0) {
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        buffer_arena.flip_parity();
        RectDesc<N,T>* d_valid_intersect = buffer_arena.alloc<RectDesc<N,T>>(num_valid_output);

        CUDA_CHECK(cudaMemcpyAsync(d_src_prefix, h_src_counters.data(), (sources.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);
        CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);

        image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_rects * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.rects_buffer, d_rngs, d_src_prefix, collapsed_parent.num_rects, total_pts, d_src_counters, d_valid_intersect);
        KERNEL_CHECK(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream), stream);

        size_t num_new_rects = (num_output == 0) ? 1 : 2;
        assert(!buffer_arena.get_parity());
        RectDesc<N, T>* d_new_rects;

        //Send it off for processing
        this->complete_rect_pipeline(d_valid_intersect, num_valid_output, d_new_rects, num_new_rects, buffer_arena,
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
            this->split_output(d_new_rects, num_new_rects, host_rect_buffers, rect_counts, buffer_arena);
          }

          //Set our first set of output rectangles
          if (num_output==0 || host_fallback) {

            //We need to place the new output at the rightmost end of the buffer
            num_output = num_new_rects;
            num_completed += curr_tile;
            output_start = d_new_rects;
            subtract_const<<<COMPUTE_GRID(domain_transform.range_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.range_data.size()+1, curr_tile);
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
      }
      catch (arena_oom&) {
        //std::cout << "Caught arena_oom, reducing tile size from " << curr_tile << " to " << curr_tile / 2 << std::endl;
        curr_tile /= 2;
        if (curr_tile == 0) {
          if (host_fallback) {
            GPUMicroOp<N2, T2>::shatter_rects(inst_space, num_completed, stream);
            curr_tile = 1;
          } else {
            host_fallback = true;
            if (num_output > 0) {
              this->split_output(output_start, num_output, host_rect_buffers, rect_counts, buffer_arena);
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
      this->split_output(output_start, num_output, host_rect_buffers, rect_counts, buffer_arena);
      host_fallback = true;
    }
  }

  if (host_fallback) {
    for (size_t idx = 0; idx < sparsity_outputs.size(); ++idx) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(sparsity_outputs[idx]);
      if (this->exclusive) {
        impl->set_contributor_count(1);
      }
      if (rect_counts[idx] > 0) {
        span<Rect<N, T>> h_rects_span(host_rect_buffers[idx], rect_counts[idx]);
        impl->contribute_dense_rect_list(h_rects_span, false);
        deppart_host_free(host_rect_buffers[idx]);
      } else {
        impl->contribute_nothing();
      }
    }
  }

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
      assert(sparsity_outputs.empty());
      return;
    }

    RegionInstance buffer = domain_transform.ptr_data[0].scratch_buffer;

    NVTX_DEPPART(gpu_image);

    CUstream stream = this->stream->get_stream();

    size_t tile_size = buffer.get_layout()->bytes_used;
    //std::cout << "Using tile size of " << tile_size << " bytes." << std::endl;
    Arena buffer_arena(buffer);

    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM, buffer_arena.location));

    collapsed_space<N2, T2> src_space;
    src_space.offsets = buffer_arena.alloc<size_t>(sources.size()+1);
    src_space.num_children = sources.size();

    GPUMicroOp<N2, T2>::collapse_multi_space(sources, src_space, buffer_arena, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    inst_space.offsets = buffer_arena.alloc<size_t>(domain_transform.ptr_data.size()+1);
    inst_space.num_children = domain_transform.ptr_data.size();

    Arena sys_arena;
    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.ptr_data, inst_space, sys_arena, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    uint32_t* d_inst_counters = buffer_arena.alloc<uint32_t>(2*domain_transform.ptr_data.size()+1);

  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.ptr_data.size();

    //Uniform for all tiles
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, collapsed_parent, buffer_arena, stream);

    std::vector<AffineAccessor<Point<N,T>,N2,T2>> h_accessors(domain_transform.ptr_data.size());
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Point<N,T>,N2,T2>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }
    AffineAccessor<Point<N,T>,N2,T2>* d_accessors =
        buffer_arena.alloc<AffineAccessor<Point<N,T>,N2,T2>>(domain_transform.ptr_data.size());
    CUDA_CHECK(cudaMemcpyAsync(d_accessors, h_accessors.data(),
                               domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N,T>,N2,T2>),
                               cudaMemcpyHostToDevice, stream), stream);

    uint32_t* d_prefix_points = buffer_arena.alloc<uint32_t>(domain_transform.ptr_data.size()+1);

    buffer_arena.commit(false);

    size_t left = buffer_arena.used();

    //Here we iterate over the tiles
    size_t num_output = 0;
    RectDesc<N, T>* output_start = nullptr;
    size_t num_completed = 0;
    size_t curr_tile = tile_size / 2;
    int count = 0;
    if (count) {}
    bool host_fallback = false;
    std::vector<Rect<N, T>*> host_rect_buffers(sources.size(), nullptr);
    std::vector<size_t> rect_counts(sources.size(), 0);
    while (num_completed < inst_space.num_rects) {
      try {
        //std::cout << "Image iteration " << count++ << ", completed " << num_completed << " / " << inst_space.num_rects << " rects." << std::endl;
        buffer_arena.start();
        if (num_completed + curr_tile > inst_space.num_rects) {
          curr_tile = inst_space.num_rects - num_completed;
        }
        collapsed_space<N2, T2> inst_space_tile = inst_space;
        inst_space_tile.num_rects = curr_tile;
        inst_space_tile.rects_buffer = buffer_arena.alloc<Rect<N2,T2>>(curr_tile);
        CUDA_CHECK(cudaMemcpyAsync(inst_space_tile.rects_buffer, inst_space.rects_buffer + num_completed, curr_tile * sizeof(Rect<N2,T2>), cudaMemcpyHostToDevice, stream), stream);

        size_t num_valid_rects;
        RectDesc<N2, T2>* d_valid_rects;
        GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space_tile, src_space, d_valid_rects, num_valid_rects, d_inst_counters, d_inst_prefix, buffer_arena, stream);

        if (num_valid_rects == 0) {
          num_completed += curr_tile;
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        // Prefix sum the valid rectangles by volume.
        size_t* d_prefix_rects;
        size_t total_pts;

        GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, d_prefix_rects, total_pts, buffer_arena, stream);

        CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

        //We do a two pass count + emit to chase all the pointers in parallel and check for membership in the parent index space
        image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.rects_buffer, d_prefix_rects, d_inst_prefix, nullptr, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_rects, d_inst_counters, nullptr);
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
          subtract_const<<<COMPUTE_GRID(domain_transform.ptr_data.size()), THREADS_PER_BLOCK, 0, stream>>>(inst_space.offsets, domain_transform.ptr_data.size()+1, curr_tile);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          curr_tile = tile_size / 2;
          continue;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_prefix_points, h_inst_counters.data(), (domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

        buffer_arena.flip_parity();
        PointDesc<N,T>* d_valid_points = buffer_arena.alloc<PointDesc<N,T>>(num_valid_points);
        buffer_arena.start();
        d_valid_points = buffer_arena.alloc<PointDesc<N,T>>(num_valid_points);

        CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

        image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.rects_buffer, d_prefix_rects, d_inst_prefix, d_prefix_points, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_rects, d_inst_counters, d_valid_points);
        KERNEL_CHECK(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);


        size_t num_new_rects = num_output == 0 ? 1 : 2;
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

          if (host_fallback) {
            this->split_output(d_new_rects, num_new_rects, host_rect_buffers, rect_counts, buffer_arena);
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
      }
      catch (arena_oom&) {
        //std::cout << "Caught arena_oom, reducing tile size from " << curr_tile << " to " << curr_tile / 2 << std::endl;
        curr_tile /= 2;
        if (curr_tile == 0) {
          if (host_fallback) {
            GPUMicroOp<N2, T2>::shatter_rects(inst_space, num_completed, stream);
            curr_tile = 1;
          } else {
            host_fallback = true;
            if (num_output > 0) {
              this->split_output(output_start, num_output, host_rect_buffers, rect_counts, buffer_arena);
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
      this->split_output(output_start, num_output, host_rect_buffers, rect_counts, buffer_arena);
      host_fallback = true;
    }
  }

  if (host_fallback) {
    for (size_t idx = 0; idx < sparsity_outputs.size(); ++idx) {
      SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(sparsity_outputs[idx]);
      if (this->exclusive) {
        impl->set_contributor_count(1);
      }
      if (rect_counts[idx] > 0) {
        span<Rect<N, T>> h_rects_span(host_rect_buffers[idx], rect_counts[idx]);
        impl->contribute_dense_rect_list(h_rects_span, false);
        deppart_host_free(host_rect_buffers[idx]);
      } else {
        impl->contribute_nothing();
      }
    }
  }
}
}
