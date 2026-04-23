/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cmath>
#include <climits>

#include <time.h>

#include "osdep.h"

#include "philox.h"

using namespace Realm;

#define USE_IMAGE_DIFF

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  INIT_BYFIELD_DATA_TASK,
  INIT_IMAGE_DATA_TASK,
  INIT_IMAGE_RANGE_DATA_TASK,
  INIT_PREIMAGE_DATA_TASK,
  INIT_PREIMAGE_RANGE_DATA_TASK
};

namespace std {
  template <typename T>
  std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
  {
    os << v.size() << "{";
    if(v.empty()) {
      os << "}";
    } else {
      os << " ";
      typename std::vector<T>::const_iterator it = v.begin();
      os << *it;
      ++it;
      while(it != v.end()) {
        os << ", " << *it;
        ++it;
      }
      os << " }";
    }
    return os;
  }
}; // namespace std

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely deadlock!\n");
  exit(1);
}

class TestInterface {
public:
  virtual ~TestInterface(void) {}

  virtual void print_info(void) = 0;

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs) = 0;

  virtual Event perform_partitioning(void) = 0;

  virtual int perform_dynamic_checks(void) = 0;

  virtual int check_partitioning(void) = 0;
};

// generic configuration settings
namespace {
  int random_seed = 12345;
  bool random_colors = false;
  bool wait_on_events = false;
  bool show_graph = false;
  bool skip_check = false;
  int dimension1 = 1;
  int dimension2 = 1;
  std::string op;
  TestInterface *testcfg = 0;
}; // namespace

template<typename IS, typename FT>
Event copy_piece(FieldDataDescriptor<IS, FT> src_data, FieldDataDescriptor<IS, FT> &dst_data, const std::vector<size_t> &fields, size_t field_idx, Memory dst_memory)
{
  size_t offset = 0;
  for (size_t i = 0; i < field_idx; i++) {
    offset += fields[i];
  }
  size_t size = fields[field_idx];
  dst_data.index_space = src_data.index_space;
  RegionInstance::create_instance(dst_data.inst,
                                        dst_memory,
                                        src_data.index_space,
                                        fields,
                                        0 /*SOA*/,
                                        Realm::ProfilingRequestSet()).wait();
  CopySrcDstField src_field, dst_field;
  src_field.inst = src_data.inst;
  src_field.size = size;
  src_field.field_id = offset;
  dst_field.inst = dst_data.inst;
  dst_field.size = size;
  dst_field.field_id = offset;
  dst_data.field_offset = src_data.field_offset;
  std::vector<CopySrcDstField> src_fields = {src_field};
  std::vector<CopySrcDstField> dst_fields = {dst_field};
  return src_data.index_space.copy(src_fields, dst_fields, Realm::ProfilingRequestSet());
}

Event alloc_piece(RegionInstance &result, size_t size, Memory location) {
  assert(location != Memory::NO_MEMORY);
  assert(size > 0);
  std::vector<size_t> byte_fields = {sizeof(char)};
  IndexSpace<1, long long> instance_index_space(Rect<1, long long>(0, size-1));
  return RegionInstance::create_instance(result, location, instance_index_space, byte_fields, 0, Realm::ProfilingRequestSet());
}

template <int N, typename T>
IndexSpace<N, T> create_sparse_index_space(const Rect<N, T> &bounds, size_t sparse_factor,
                                           bool randomize, size_t idx)
{
  std::vector<Point<N, T>> points;
  for(PointInRectIterator<N, T> it(bounds); it.valid; it.step()) {
    size_t flattened = idx * bounds.volume();
    size_t stride = 1;
    for (int d = 0; d < N; d++) {
      flattened += (it.p[d] - bounds.lo[d]) * stride;
      stride *= (bounds.hi[d] - bounds.lo[d] + 1);
    }
    if(randomize) {
      if(Philox_2x32<>::rand_int(random_seed, flattened, 0, 100) < sparse_factor) {
        points.push_back(it.p);
      }
    } else {
      if( (99 * flattened) % 100 < sparse_factor) {
        points.push_back(it.p);
      }
    }
  }
  return IndexSpace<N, T>(points, true);
}

/*
 * Byfield test - create a graph, partition it by
 * node subgraph id and then check that the partitioning
 * is correct
 */
template<int N>
class ByfieldTest : public TestInterface {
public:
  // graph config parameters
  int num_nodes = 1000;
  int num_pieces = 4;
  int num_colors = 4;
  size_t buffer_size = 100;
  std::string filename;

  ByfieldTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {

      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-c")) {
        num_colors = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-b")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }


    if (num_nodes <= 0 || num_pieces <= 0 || num_colors <= 0 || buffer_size <= 0 || buffer_size > 100) {
      log_app.error() << "Invalid config: nodes=" << num_nodes << " colors=" << num_colors << " pieces=" << num_pieces << " buffer size=" << buffer_size << "\n";
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_colors;
  };

  enum PRNGStreams
  {
    NODE_SUBGRAPH_STREAM,
  };

  // assign subgraph ids to nodes
  void color_point(int idx, int& color)
  {
    if(random_colors)
        color = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, num_colors);
      else
        color = (idx * num_colors / num_nodes) % num_colors;
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    ByfieldTest *me = (ByfieldTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  //Each piece has a task to initialize its data
  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    log_app.info() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_colors
                   << ")";

    i_args.ri_colors.fetch_metadata(p).wait();

    IndexSpace<N> colors_space = i_args.ri_colors.template get_indexspace<N>();

    log_app.debug() << "N: " << is_colors;

    //For each node in the graph, mark it with a random (or deterministic) subgraph id
    {
      AffineAccessor<int, N> a_piece_id(i_args.ri_colors, 0 /* offset */);

      for (IndexSpaceIterator<N> it(is_colors); it.valid; it.step()) {
        for (PointInRectIterator<N> point(it.rect); point.valid; point.step()) {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < N; d++) {
            idx += (point.p[d] - is_colors.bounds.lo[d]) * stride;
            stride *= (is_colors.bounds.hi[d] - is_colors.bounds.lo[d] + 1);
          }
          int subgraph;
          color_point(idx, subgraph);
          a_piece_id.write(point.p, subgraph);
        }
      }
    }
  }

  IndexSpace<N> is_colors;
  std::vector<RegionInstance> ri_colors;
  std::vector<FieldDataDescriptor<IndexSpace<N>, int> > piece_id_field_data;

  virtual void print_info(void)
  {
    //printf("Realm %dD Byfield dependent partitioning test: %d nodes, %d colors, %d pieces, %lu tile size\n", (int) N,
	   //(int)num_nodes, (int) num_colors, (int)num_pieces, buffer_size);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    // now create index space for nodes
    Point<N> lo, hi;
    for (int d = 0; d < N; d++) {
      lo[d] = 0;
      hi[d] = num_nodes - 1;
    }
    is_colors = Rect<N>(lo, hi);

    // equal partition is used to do initial population of edges and nodes
    std::vector<IndexSpace<N> > ss_nodes_eq;

    log_app.info() << "Creating equal subspaces\n";

    is_colors.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();

    // create instances for each of these subspaces
    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(int));
    
    ri_colors.resize(num_pieces);
    piece_id_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[i % memories.size()], ss_nodes_eq[i],
                                      node_fields, 0 /*SOA*/,
                                      Realm::ProfilingRequestSet())
          .wait();
      ri_colors[i] = ri;

      piece_id_field_data[i].index_space = ss_nodes_eq[i];
      piece_id_field_data[i].inst = ri_colors[i];
      piece_id_field_data[i].field_offset = 0;
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor p = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_colors = ri_colors[i];
      Event e = p.spawn(INIT_BYFIELD_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  p_nodes - nodes partitioned by subgraph id (from GPU)
  //  p_nodes_cpu - nodes partitioned by subgraph id (from CPU)


    std::vector<IndexSpace<N> > p_nodes, p_garbage_nodes, p_nodes_cpu;

  virtual Event perform_partitioning(void)
  {
    // Partition nodes by subgraph id - do this twice, once on CPU and once on GPU
    // Ensure that the results are identical

    std::vector<int> colors(num_colors);
    for(int i = 0; i < num_colors; i++)
      colors[i] = i;

    // We need a GPU memory for GPU partitioning
    Memory gpu_memory;
    bool found_gpu_memory = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(Memory memory : all_memories) {
      if(memory.kind() == Memory::GPU_FB_MEM) {
        gpu_memory = memory;
        found_gpu_memory = true;
        break;
      }
    }
    if (!found_gpu_memory) {
      log_app.error() << "No GPU memory found for partitioning test\n";
      return Event::NO_EVENT;
    }


    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(int));

    std::vector<FieldDataDescriptor<IndexSpace<N>, int> > piece_field_data_gpu;
    piece_field_data_gpu.resize(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
    	copy_piece(piece_id_field_data[i], piece_field_data_gpu[i], node_fields, 0, gpu_memory).wait();
    }

    std::vector<DeppartEstimateInput<N, int>> byfield_inputs(num_pieces);
    std::vector<DeppartBufferRequirements> byfield_requirements(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
      byfield_inputs[i].location = piece_field_data_gpu[i].inst.get_location();
      byfield_inputs[i].space = piece_field_data_gpu[i].index_space;
    }

    is_colors.by_field_buffer_requirements(byfield_inputs, byfield_requirements);


    for (int i = 0; i < num_pieces; i++) {
      size_t alloc_size = byfield_requirements[i].lower_bound + (byfield_requirements[i].upper_bound - byfield_requirements[i].lower_bound) * buffer_size / 100;
      alloc_piece(piece_field_data_gpu[i].scratch_buffer, alloc_size, gpu_memory).wait();
    }

    log_app.info() << "warming up" << Clock::current_time_in_microseconds() << "\n";
    Event warmup = is_colors.create_subspaces_by_field(piece_field_data_gpu,
                                                  colors,
                                                  p_garbage_nodes,
                                                  Realm::ProfilingRequestSet());
    warmup.wait();

    long long start_gpu = Clock::current_time_in_microseconds();
    Event gpu_call = is_colors.create_subspaces_by_field(piece_field_data_gpu,
                                                  colors,
                                                  p_nodes,
                                                  Realm::ProfilingRequestSet());

    gpu_call.wait();
    long long gpu_time = Clock::current_time_in_microseconds() - start_gpu;
    long long start_cpu = Clock::current_time_in_microseconds();

    Event cpu_call = is_colors.create_subspaces_by_field(piece_id_field_data,
                                                  colors,
                                                  p_nodes_cpu,
                                                  Realm::ProfilingRequestSet());

    cpu_call.wait();
    long long cpu_time = Clock::current_time_in_microseconds() - start_cpu;

    printf("RESULT,op=byfield,d1=%d,num_nodes=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
             N, num_nodes, buffer_size, gpu_time, cpu_time);

    return Event::merge_events({gpu_call, cpu_call});

  }

  virtual int perform_dynamic_checks(void)
  {
    // Nothing to do here
    return 0;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    if (!p_nodes.size()) {
      return p_nodes.size() == p_nodes_cpu.size();
    }

    log_app.info() << "Checking correctness of partitioning " << "\n";

    for(int i = 0; i < num_pieces; i++) {
      if (!p_nodes[i].dense() && (N > 1)) {
        p_nodes[i].sparsity.impl()->request_bvh();
        if (!p_nodes_cpu[i].dense()) {
          p_nodes_cpu[i].sparsity.impl()->request_bvh();
        }
      }
      for(IndexSpaceIterator<N> it(p_nodes[i]); it.valid; it.step()) {
        for(PointInRectIterator<N> point(it.rect); point.valid; point.step()) {
          if (!p_nodes_cpu[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU has extra byfield point " << point.p
                            << " on piece " << i << "\n";
            errors++;
          }
        }
      }
      for(IndexSpaceIterator<N> it(p_nodes_cpu[i]); it.valid; it.step()) {
        for(PointInRectIterator<N> point(it.rect); point.valid; point.step()) {
          if (!p_nodes[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU is missing byfield point " << point.p
                          << " on piece " << i << "\n";
            errors++;
          }
        }
      }

    }
    return errors;
  }
};

template<int N1, int N2>
class ImageTest : public TestInterface {
public:
  // graph config parameters
  int num_nodes = 1000;
  int num_edges = 1000;
  int sparse_factor = 50;
  int num_spaces = 4;
  int num_pieces = 4;
  size_t buffer_size = 100;
  std::string filename;

  ImageTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {

      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-e")) {
        num_edges = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-s")) {
        num_spaces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-f")) {
        sparse_factor = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-b")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }


    if (num_nodes <= 0 || num_pieces <= 0 || num_edges <= 0 || num_spaces <= 0) {
      log_app.error() << "Invalid config: nodes=" << num_nodes << " colors=" << num_edges << " pieces=" << num_pieces << " sources=" << num_spaces << " buffer size=" << buffer_size <<  "\n";
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes;
  };

  enum PRNGStreams
  {
    NODE_SUBGRAPH_STREAM,
  };

  // assign subgraph ids to nodes
  void chase_point(int idx, Point<N1>& color)
  {
    for (int d = 0; d < N1; d++) {
      if(random_colors)
        color[d] = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, num_edges);
      else
        color[d] = (idx * num_edges / num_nodes) % num_edges;
    }
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    ImageTest *me = (ImageTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  //Each piece has a task to initialize its data
  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    log_app.info() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes
                   << ")";

    i_args.ri_nodes.fetch_metadata(p).wait();

    IndexSpace<N2> nodes_space = i_args.ri_nodes.template get_indexspace<N2>();

    log_app.debug() << "N: " << is_nodes;

    //For each node in the graph, mark it with a random (or deterministic) subgraph id
    {
      AffineAccessor<Point<N1>, N2> a_point(i_args.ri_nodes, 0 /* offset */);

      for (IndexSpaceIterator<N2> it(is_nodes); it.valid; it.step()) {
        for (PointInRectIterator<N2> point(it.rect); point.valid; point.step()) {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < N2; d++) {
            idx += (point.p[d] - is_nodes.bounds.lo[d]) * stride;
            stride *= (is_nodes.bounds.hi[d] - is_nodes.bounds.lo[d] + 1);
          }
          Point<N1> destination;
          chase_point(idx, destination);
          a_point.write(point.p, destination);
        }
      }
    }
  }

  IndexSpace<N2> is_nodes;
  IndexSpace<N1> is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<IndexSpace<N2>, Point<N1>> > point_field_data;

  virtual void print_info(void)
  {
    //printf("Realm %dD -> %dD Image dependent partitioning test: %d nodes, %d edges, %d pieces ,%d sources, %d sparse factor, %lu tile size\n", (int) N2, (int) N1,
	   //(int)num_nodes, (int) num_edges, (int)num_pieces, (int) num_spaces, (int) sparse_factor, buffer_size);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    // now create index space for nodes
    Point<N2> node_lo, node_hi;
    for (int d = 0; d < N2; d++) {
      node_lo[d] = 0;
      node_hi[d] = num_nodes - 1;
    }
    is_nodes = Rect<N2>(node_lo, node_hi);

    Point<N1> edge_lo, edge_hi;
    for (int d = 0; d < N1; d++) {
      edge_lo[d] = 0;
      edge_hi[d] = num_edges - 1;
    }
    is_edges = Rect<N1>(edge_lo, edge_hi);

    // equal partition is used to do initial population of edges and nodes
    std::vector<IndexSpace<N2> > ss_nodes_eq;

    log_app.info() << "Creating equal subspaces\n";

    is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();

    // create instances for each of these subspaces
    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Point<N1>));

    ri_nodes.resize(num_pieces);
    point_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[i % memories.size()], ss_nodes_eq[i],
                                      node_fields, 0 /*SOA*/,
                                      Realm::ProfilingRequestSet()).wait();
      ri_nodes[i] = ri;

      point_field_data[i].index_space = ss_nodes_eq[i];
      point_field_data[i].inst = ri_nodes[i];
      point_field_data[i].field_offset = 0;
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor p = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      Event e = p.spawn(INIT_IMAGE_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  p_nodes - nodes partitioned by subgraph id (from GPU)
  //  p_nodes_cpu - nodes partitioned by subgraph id (from CPU)


    std::vector<IndexSpace<N1> > p_edges, p_garbage_edges, p_edges_cpu;

  virtual Event perform_partitioning(void)
  {
    // Partition nodes by subgraph id - do this twice, once on CPU and once on GPU
    // Ensure that the results are identical

    std::vector<IndexSpace<N2>> sources(num_spaces);
    for(int i = 0; i < num_spaces; i++) {
      if (sparse_factor <= 1) {
        sources[i] = point_field_data[i % num_pieces].index_space;
      } else {
        sources[i] = create_sparse_index_space(is_nodes.bounds, sparse_factor, random_colors, i);
      }
    }

    // We need a GPU memory for GPU partitioning
    Memory gpu_memory;
    bool found_gpu_memory = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(Memory memory : all_memories) {
      if(memory.kind() == Memory::GPU_FB_MEM) {
        gpu_memory = memory;
        found_gpu_memory = true;
        break;
      }
    }
    if (!found_gpu_memory) {
      log_app.error() << "No GPU memory found for partitioning test\n";
      return Event::NO_EVENT;
    }


    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Point<N1>));

    std::vector<FieldDataDescriptor<IndexSpace<N2>, Point<N1>>> point_field_data_gpu;
    point_field_data_gpu.resize(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
    	copy_piece(point_field_data[i], point_field_data_gpu[i], node_fields, 0, gpu_memory).wait();
    }

    std::vector<DeppartEstimateInput<N2, int>> image_inputs(num_pieces);
    std::vector<DeppartSubspace<N2, int>> image_subspaces(num_spaces);
    std::vector<DeppartBufferRequirements> image_requirements(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
      image_inputs[i].location = point_field_data_gpu[i].inst.get_location();
      image_inputs[i].space = point_field_data_gpu[i].index_space;
    }

    for (int i = 0; i < num_spaces; i++) {
      image_subspaces[i].space = sources[i];
      image_subspaces[i].entries = sources[i].dense() ? 1 : sources[i].sparsity.impl()->get_entries().size();
    }

    is_edges.by_image_buffer_requirements(image_subspaces, image_inputs, image_requirements);

    for (int i = 0; i < num_pieces; i++) {
      size_t alloc_size = image_requirements[i].lower_bound + (image_requirements[i].upper_bound - image_requirements[i].lower_bound) * buffer_size / 100;
      alloc_piece(point_field_data_gpu[i].scratch_buffer, alloc_size, gpu_memory).wait();
    }

    log_app.info() << "warming up" << Clock::current_time_in_microseconds() << "\n";
    Event warmup = is_edges.create_subspaces_by_image(point_field_data_gpu,
                                                  sources,
                                                  p_garbage_edges,
                                                  Realm::ProfilingRequestSet());
    warmup.wait();

    long long start_gpu = Clock::current_time_in_microseconds();
    Event gpu_call = is_edges.create_subspaces_by_image(point_field_data_gpu,
                                                  sources,
                                                  p_edges,
                                                  Realm::ProfilingRequestSet());

    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - start_gpu;
    long long start_cpu = Clock::current_time_in_microseconds();
    Event cpu_call = is_edges.create_subspaces_by_image(point_field_data,
                                                  sources,
                                                  p_edges_cpu,
                                                  Realm::ProfilingRequestSet());

    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - start_cpu;
    printf("RESULT,op=image,d1=%d,d2=%d,num_nodes=%d,num_edges=%d,num_spaces=%d,sparse_factor=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
                 N1, N2, num_nodes, num_edges, num_spaces, sparse_factor, buffer_size, gpu_us, cpu_us);

    return Event::merge_events({gpu_call, cpu_call});

  }

  virtual int perform_dynamic_checks(void)
  {
    // Nothing to do here
    return 0;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    if (!p_edges.size()) {
      return p_edges.size() == p_edges_cpu.size();
    }

    log_app.info() << "Checking correctness of partitioning " << "\n";

    for(int i = 0; i < num_pieces; i++) {
      if (N1 > 1) {
        if (!p_edges[i].dense()) {
          p_edges[i].sparsity.impl()->request_bvh();
        }
        if (!p_edges_cpu[i].dense()) {
          p_edges_cpu[i].sparsity.impl()->request_bvh();
        }
      }
      for(IndexSpaceIterator<N1> it(p_edges[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_edges_cpu[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU has extra image point " << point.p
                            << " on piece " << i << "\n";
            errors++;
          }
        }
      }
      for(IndexSpaceIterator<N1> it(p_edges_cpu[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_edges[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU is missing image point " << point.p
                          << " on piece " << i << "\n";
            errors++;
          }
        }
      }

    }
    return errors;
  }
};

template<int N1, int N2>
class ImageRangeTest : public TestInterface {
public:
  // graph config parameters
  int num_nodes = 1000;
  int num_edges = 1000;
  int rect_size = 10;
  int num_spaces = 4;
  int num_pieces = 4;
  int sparse_factor = 50;
  size_t buffer_size = 100;
  std::string filename;

  ImageRangeTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {

      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-e")) {
        num_edges = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-r")) {
        rect_size = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-s")) {
        num_spaces = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-f")) {
        sparse_factor = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-b")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }


    if (num_nodes <= 0 || num_pieces <= 0 || num_edges <= 0 || num_spaces <= 0 || rect_size <= 0 || sparse_factor < 0 || sparse_factor > 100 || buffer_size < 0 || buffer_size > 100) {
      log_app.error() << "Invalid config: nodes=" << num_nodes << " colors=" << num_edges << " pieces=" << num_pieces << " sources=" << num_spaces << " rect size=" << rect_size << " sparse factor=" << sparse_factor << " buffer_size=" << buffer_size <<  "\n";
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes;
  };

  enum PRNGStreams
  {
    NODE_SUBGRAPH_STREAM,
  };

  // assign subgraph ids to nodes
  void chase_rect(int idx, Rect<N1>& color)
  {
    for (int d = 0; d < N1; d++) {
      if(random_colors) {
        color.lo[d] = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, num_edges);
        color.hi[d] = color.lo[d] + Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, 2 * rect_size);
      } else {
        color.lo[d] = (idx * num_edges / num_nodes) % num_edges;
        color.hi[d] = color.lo[d] + rect_size;
      }
    }
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    ImageRangeTest *me = (ImageRangeTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  //Each piece has a task to initialize its data
  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    log_app.info() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes
                   << ")";

    i_args.ri_nodes.fetch_metadata(p).wait();

    IndexSpace<N2> nodes_space = i_args.ri_nodes.template get_indexspace<N2>();

    log_app.debug() << "N: " << is_nodes;

    //For each node in the graph, mark it with a random (or deterministic) subgraph id
    {
      AffineAccessor<Rect<N1>, N2> a_rect(i_args.ri_nodes, 0 /* offset */);

      for (IndexSpaceIterator<N2> it(is_nodes); it.valid; it.step()) {
        for (PointInRectIterator<N2> point(it.rect); point.valid; point.step()) {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < N2; d++) {
            idx += (point.p[d] - is_nodes.bounds.lo[d]) * stride;
            stride *= (is_nodes.bounds.hi[d] - is_nodes.bounds.lo[d] + 1);
          }
          Rect<N1> destination;
          chase_rect(idx, destination);
          a_rect.write(point.p, destination);
        }
      }
    }
  }

  IndexSpace<N2> is_nodes;
  IndexSpace<N1> is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<IndexSpace<N2>, Rect<N1>> > rect_field_data;

  virtual void print_info(void)
  {
    //printf("Realm %dD -> %dD Image Range dependent partitioning test: %d nodes, %d edges, %d pieces ,%d sources, %d rect size, %d sparse factor, %lu tile size\n", (int) N2, (int) N1,
	   // (int)num_nodes, (int) num_edges, (int)num_pieces, (int) num_spaces, (int) rect_size, (int) sparse_factor, buffer_size);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    // now create index space for nodes
    Point<N2> node_lo, node_hi;
    for (int d = 0; d < N2; d++) {
      node_lo[d] = 0;
      node_hi[d] = num_nodes - 1;
    }
    is_nodes = Rect<N2>(node_lo, node_hi);

    Point<N1> edge_lo, edge_hi;
    for (int d = 0; d < N1; d++) {
      edge_lo[d] = 0;
      edge_hi[d] = num_edges - 1;
    }
    is_edges = Rect<N1>(edge_lo, edge_hi);

    // equal partition is used to do initial population of edges and nodes
    std::vector<IndexSpace<N2> > ss_nodes_eq;

    log_app.info() << "Creating equal subspaces\n";

    is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();

    // create instances for each of these subspaces
    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Rect<N1>));

    ri_nodes.resize(num_pieces);
    rect_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[i % memories.size()], ss_nodes_eq[i],
                                      node_fields, 0 /*SOA*/,
                                      Realm::ProfilingRequestSet()).wait();
      ri_nodes[i] = ri;

      rect_field_data[i].index_space = ss_nodes_eq[i];
      rect_field_data[i].inst = ri_nodes[i];
      rect_field_data[i].field_offset = 0;
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor p = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      Event e = p.spawn(INIT_IMAGE_RANGE_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  p_nodes - nodes partitioned by subgraph id (from GPU)
  //  p_nodes_cpu - nodes partitioned by subgraph id (from CPU)


    std::vector<IndexSpace<N1> > p_edges, p_garbage_edges, p_edges_cpu;

  virtual Event perform_partitioning(void)
  {
    // Partition nodes by subgraph id - do this twice, once on CPU and once on GPU
    // Ensure that the results are identical

    std::vector<IndexSpace<N2>> sources(num_spaces);
    for(int i = 0; i < num_spaces; i++) {
      if (sparse_factor <= 1) {
        sources[i] = rect_field_data[i % num_pieces].index_space;
      } else {
        sources[i] = create_sparse_index_space(is_nodes.bounds, sparse_factor, random_colors, i);
      }
    }

    // We need a GPU memory for GPU partitioning
    Memory gpu_memory;
    bool found_gpu_memory = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(Memory memory : all_memories) {
      if(memory.kind() == Memory::GPU_FB_MEM) {
        gpu_memory = memory;
        found_gpu_memory = true;
        break;
      }
    }
    if (!found_gpu_memory) {
      log_app.error() << "No GPU memory found for partitioning test\n";
      return Event::NO_EVENT;
    }


    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Rect<N1>));

    std::vector<FieldDataDescriptor<IndexSpace<N2>, Rect<N1>>> rect_field_data_gpu;
    rect_field_data_gpu.resize(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
    	copy_piece(rect_field_data[i], rect_field_data_gpu[i], node_fields, 0, gpu_memory).wait();
    }

    std::vector<DeppartEstimateInput<N2, int>> image_inputs(num_pieces);
    std::vector<DeppartSubspace<N2, int>> image_subspaces(num_spaces);
    std::vector<DeppartBufferRequirements> image_requirements(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
      image_inputs[i].location = rect_field_data_gpu[i].inst.get_location();
      image_inputs[i].space = rect_field_data_gpu[i].index_space;
    }

    for (int i = 0; i < num_spaces; i++) {
      image_subspaces[i].space = sources[i];
      image_subspaces[i].entries = sources[i].dense() ? 1 : sources[i].sparsity.impl()->get_entries().size();
    }

    is_edges.by_image_buffer_requirements(image_subspaces, image_inputs, image_requirements);

    for (int i = 0; i < num_pieces; i++) {
      size_t alloc_size = image_requirements[i].lower_bound + (image_requirements[i].upper_bound - image_requirements[i].lower_bound) * buffer_size / 100;
      alloc_piece(rect_field_data_gpu[i].scratch_buffer, alloc_size, gpu_memory).wait();
    }

    log_app.info() << "warming up" << Clock::current_time_in_microseconds() << "\n";
    Event warmup = is_edges.create_subspaces_by_image(rect_field_data_gpu,
                                                  sources,
                                                  p_garbage_edges,
                                                  Realm::ProfilingRequestSet());
    warmup.wait();

    long long start_gpu = Clock::current_time_in_microseconds();
    Event gpu_call = is_edges.create_subspaces_by_image(rect_field_data_gpu,
                                                  sources,
                                                  p_edges,
                                                  Realm::ProfilingRequestSet());


    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - start_gpu;
    long long start_cpu = Clock::current_time_in_microseconds();
    Event cpu_call = is_edges.create_subspaces_by_image(rect_field_data,
                                                  sources,
                                                  p_edges_cpu,
                                                  Realm::ProfilingRequestSet());

    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - start_cpu;

    printf("RESULT,op=image,d1=%d,d2=%d,num_nodes=%d,num_edges=%d,num_spaces=%d,sparse_factor=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
                 N1, N2, num_nodes, num_edges, num_spaces, sparse_factor, buffer_size, gpu_us, cpu_us);

    return Event::merge_events({gpu_call, cpu_call});

  }

  virtual int perform_dynamic_checks(void)
  {
    // Nothing to do here
    return 0;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    if (!p_edges.size()) {
      return p_edges.size() == p_edges_cpu.size();
    }

    log_app.info() << "Checking correctness of partitioning " << "\n";

    for(int i = 0; i < num_spaces; i++) {

      if (N1 > 1) {
        if (!p_edges[i].dense()) {
          p_edges[i].sparsity.impl()->request_bvh();
        }
        if (!p_edges_cpu[i].dense()) {
          p_edges_cpu[i].sparsity.impl()->request_bvh();
        }
      }

      for(IndexSpaceIterator<N1> it(p_edges[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_edges_cpu[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU has extra image point " << point.p
                            << " on piece " << i << "\n";
            errors++;
          }
        }
      }
      for(IndexSpaceIterator<N1> it(p_edges_cpu[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_edges[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU is missing image point " << point.p
                          << " on piece " << i << "\n";
            errors++;
          }
        }
      }

    }
    return errors;
  }
};

template<int N1, int N2>
class PreimageTest : public TestInterface {
public:
  // graph config parameters
  int num_nodes = 1000;
  int num_edges = 1000;
  int num_spaces = 4;
  int num_pieces = 4;
  int sparse_factor = 50;
  size_t buffer_size = 100;
  std::string filename;

  PreimageTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {

      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-e")) {
        num_edges = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-s")) {
        num_spaces = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-f")) {
        sparse_factor = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-b")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }


    if (num_nodes <= 0 || num_pieces <= 0 || num_edges <= 0 || num_spaces <= 0 || sparse_factor < 0 || sparse_factor > 100 || buffer_size < 0 || buffer_size > 100) {
      log_app.error() << "Invalid config: nodes=" << num_nodes << " colors=" << num_edges << " pieces=" << num_pieces << " targets=" << num_spaces << " sparse factor=" << sparse_factor << " buffer size=" << buffer_size <<  "\n";
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes;
  };

  enum PRNGStreams
  {
    NODE_SUBGRAPH_STREAM,
  };

  // assign subgraph ids to nodes
  void chase_point(int idx, Point<N2>& color)
  {
    for (int d = 0; d < N2; d++) {
      if(random_colors)
        color[d] = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, num_edges);
      else
        color[d] = (idx * num_edges / num_nodes) % num_edges;
    }
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    PreimageTest *me = (PreimageTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  //Each piece has a task to initialize its data
  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    log_app.info() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes
                   << ")";

    i_args.ri_nodes.fetch_metadata(p).wait();

    IndexSpace<N1> nodes_space = i_args.ri_nodes.template get_indexspace<N1>();

    log_app.debug() << "N: " << is_nodes;

    //For each node in the graph, mark it with a random (or deterministic) subgraph id
    {
      AffineAccessor<Point<N2>, N1> a_point(i_args.ri_nodes, 0 /* offset */);

      for (IndexSpaceIterator<N1> it(is_nodes); it.valid; it.step()) {
        for (PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < N1; d++) {
            idx += (point.p[d] - is_nodes.bounds.lo[d]) * stride;
            stride *= (is_nodes.bounds.hi[d] - is_nodes.bounds.lo[d] + 1);
          }
          Point<N2> destination;
          chase_point(idx, destination);
          a_point.write(point.p, destination);
        }
      }
    }
  }

  IndexSpace<N1> is_nodes;
  IndexSpace<N2> is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<IndexSpace<N1>, Point<N2>> > point_field_data;

  virtual void print_info(void)
  {
    //printf("Realm %dD -> %dD Preimage dependent partitioning test: %d nodes, %d edges, %d pieces ,%d targets, %d sparse factor, %lu tile size\n", (int) N1, (int) N2,
	   //(int)num_nodes, (int) num_edges, (int)num_pieces, (int) num_spaces, (int) sparse_factor, buffer_size);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    // now create index space for nodes
    Point<N1> node_lo, node_hi;
    for (int d = 0; d < N1; d++) {
      node_lo[d] = 0;
      node_hi[d] = num_nodes - 1;
    }
    is_nodes = Rect<N1>(node_lo, node_hi);

    Point<N2> edge_lo, edge_hi;
    for (int d = 0; d < N2; d++) {
      edge_lo[d] = 0;
      edge_hi[d] = num_edges - 1;
    }
    is_edges = Rect<N2>(edge_lo, edge_hi);

    // equal partition is used to do initial population of edges and nodes
    std::vector<IndexSpace<N1> > ss_nodes_eq;

    log_app.info() << "Creating equal subspaces\n";

    is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();

    // create instances for each of these subspaces
    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Point<N2>));

    ri_nodes.resize(num_pieces);
    point_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[i % memories.size()], ss_nodes_eq[i],
                                      node_fields, 0 /*SOA*/,
                                      Realm::ProfilingRequestSet()).wait();
      ri_nodes[i] = ri;

      point_field_data[i].index_space = ss_nodes_eq[i];
      point_field_data[i].inst = ri_nodes[i];
      point_field_data[i].field_offset = 0;
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor p = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      Event e = p.spawn(INIT_PREIMAGE_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  p_nodes - nodes partitioned by subgraph id (from GPU)
  //  p_nodes_cpu - nodes partitioned by subgraph id (from CPU)


    std::vector<IndexSpace<N1> > p_nodes, p_garbage_nodes, p_nodes_cpu;

  virtual Event perform_partitioning(void)
  {
    // Partition nodes by subgraph id - do this twice, once on CPU and once on GPU
    // Ensure that the results are identical

    std::vector<IndexSpace<N2>> targets;
    if (sparse_factor <= 1) {
      is_edges.create_equal_subspaces(num_spaces, 1, targets, Realm::ProfilingRequestSet()).wait();
    } else {
      targets.resize(num_spaces);
      for (int i = 0; i < num_spaces; i++) {
        targets[i] = create_sparse_index_space(is_edges.bounds, sparse_factor, random_colors, i);
      }
    }

    // We need a GPU memory for GPU partitioning
    Memory gpu_memory;
    bool found_gpu_memory = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(Memory memory : all_memories) {
      if(memory.kind() == Memory::GPU_FB_MEM) {
        gpu_memory = memory;
        found_gpu_memory = true;
        break;
      }
    }
    if (!found_gpu_memory) {
      log_app.error() << "No GPU memory found for partitioning test\n";
      return Event::NO_EVENT;
    }


    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Point<N2>));

    std::vector<FieldDataDescriptor<IndexSpace<N1>, Point<N2>>> point_field_data_gpu;
    point_field_data_gpu.resize(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
    	copy_piece(point_field_data[i], point_field_data_gpu[i], node_fields, 0, gpu_memory).wait();
    }

    std::vector<DeppartEstimateInput<N1, int>> preimage_inputs(num_pieces);
    std::vector<DeppartSubspace<N2, int>> preimage_subspaces(num_spaces);
    std::vector<DeppartBufferRequirements> preimage_requirements(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
      preimage_inputs[i].location = point_field_data_gpu[i].inst.get_location();
      preimage_inputs[i].space = point_field_data_gpu[i].index_space;
    }

    for (int i = 0; i < num_spaces; i++) {
      preimage_subspaces[i].space = targets[i];
      preimage_subspaces[i].entries = targets[i].dense() ? 1 : targets[i].sparsity.impl()->get_entries().size();
    }

    is_nodes.by_preimage_buffer_requirements(preimage_subspaces, preimage_inputs, preimage_requirements);

    for (int i = 0; i < num_pieces; i++) {
      size_t alloc_size = preimage_requirements[i].lower_bound + (preimage_requirements[i].upper_bound - preimage_requirements[i].lower_bound) * buffer_size / 100;
      alloc_piece(point_field_data_gpu[i].scratch_buffer, alloc_size, gpu_memory).wait();
    }

    log_app.info() << "warming up" << Clock::current_time_in_microseconds() << "\n";
    Event warmup = is_nodes.create_subspaces_by_preimage(point_field_data_gpu,
                                                  targets,
                                                  p_garbage_nodes,
                                                  Realm::ProfilingRequestSet());
    warmup.wait();

    long long gpu_start = Clock::current_time_in_microseconds();
    Event gpu_call = is_nodes.create_subspaces_by_preimage(point_field_data_gpu,
                                                  targets,
                                                  p_nodes,
                                                  Realm::ProfilingRequestSet());

    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - gpu_start;
    long long cpu_start = Clock::current_time_in_microseconds();
    Event cpu_call = is_nodes.create_subspaces_by_preimage(point_field_data,
                                                  targets,
                                                  p_nodes_cpu,
                                                  Realm::ProfilingRequestSet());

    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - cpu_start;
    printf("RESULT,op=preimage,d1=%d,d2=%d,num_nodes=%d,num_edges=%d,sparse_factor=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
       N1, N2, num_nodes, num_edges, sparse_factor, buffer_size, gpu_us, cpu_us);
    return Event::merge_events({gpu_call, cpu_call});

  }

  virtual int perform_dynamic_checks(void)
  {
    // Nothing to do here
    return 0;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    if (!p_nodes.size()) {
      return p_nodes.size() != p_nodes_cpu.size();
    }

    log_app.info() << "Checking correctness of partitioning " << "\n";

    for(int i = 0; i < num_spaces; i++) {
      if (!p_nodes[i].dense() && (N1 > 1)) {
        p_nodes[i].sparsity.impl()->request_bvh();
        if (!p_nodes_cpu[i].dense()) {
          p_nodes_cpu[i].sparsity.impl()->request_bvh();
        }
      }
      for(IndexSpaceIterator<N1> it(p_nodes[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_nodes_cpu[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU has extra image point " << point.p
                            << " on piece " << i << "\n";
            errors++;
          }
        }
      }
      for(IndexSpaceIterator<N1> it(p_nodes_cpu[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_nodes[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU is missing image point " << point.p
                          << " on piece " << i << "\n";
            errors++;
          }
        }
      }

    }
    return errors;
  }
};

template<int N1, int N2>
class PreimageRangeTest : public TestInterface {
public:
  // graph config parameters
  int num_nodes = 1000;
  int num_edges = 1000;
  int rect_size = 10;
  int num_spaces = 4;
  int num_pieces = 4;
  int sparse_factor = 50;
  size_t buffer_size = 100;
  std::string filename;

  PreimageRangeTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {

      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-e")) {
        num_edges = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-r")) {
        rect_size = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-s")) {
        num_spaces = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-f")) {
        sparse_factor = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-b")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }


    if (num_nodes <= 0 || num_pieces <= 0 || num_edges <= 0 || num_spaces <= 0 || rect_size <= 0 || sparse_factor < 0 || sparse_factor > 100 || buffer_size < 0 || buffer_size > 100) {
      log_app.error() << "Invalid config: nodes=" << num_nodes << " colors=" << num_edges << " pieces=" << num_pieces << " targets=" << num_spaces << " rect size=" << rect_size << " sparse factor=" << sparse_factor << " buffer size=" << buffer_size <<  "\n";
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes;
  };

  enum PRNGStreams
  {
    NODE_SUBGRAPH_STREAM,
  };

  // assign subgraph ids to nodes
  void chase_rect(int idx, Rect<N2>& color)
  {
    for (int d = 0; d < N2; d++) {
      if(random_colors) {
        color.lo[d] = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, num_edges);
        color.hi[d] = color.lo[d] + Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBGRAPH_STREAM, 2 * rect_size);
      } else {
        color.lo[d] = (idx * num_edges / num_nodes) % num_edges;
        color.hi[d] = color.lo[d] + rect_size;
      }
    }
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    PreimageRangeTest *me = (PreimageRangeTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  //Each piece has a task to initialize its data
  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    log_app.info() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes
                   << ")";

    i_args.ri_nodes.fetch_metadata(p).wait();

    IndexSpace<N1> nodes_space = i_args.ri_nodes.template get_indexspace<N1>();

    log_app.debug() << "N: " << is_nodes;

    //For each node in the graph, mark it with a random (or deterministic) subgraph id
    {
      AffineAccessor<Rect<N2>, N1> a_rect(i_args.ri_nodes, 0 /* offset */);

      for (IndexSpaceIterator<N1> it(is_nodes); it.valid; it.step()) {
        for (PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < N1; d++) {
            idx += (point.p[d] - is_nodes.bounds.lo[d]) * stride;
            stride *= (is_nodes.bounds.hi[d] - is_nodes.bounds.lo[d] + 1);
          }
          Rect<N2> destination;
          chase_rect(idx, destination);
          a_rect.write(point.p, destination);
        }
      }
    }
  }

  IndexSpace<N1> is_nodes;
  IndexSpace<N2> is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<IndexSpace<N1>, Rect<N2>> > rect_field_data;

  virtual void print_info(void)
  {
    printf("Realm %dD -> %dD Preimage Range dependent partitioning test: %d nodes, %d edges, %d pieces ,%d targets, %d rect size, %d sparse factor, %lu tile size\n", (int) N1, (int) N2,
	   (int)num_nodes, (int) num_edges, (int)num_pieces, (int) num_spaces, (int) rect_size, (int) sparse_factor, buffer_size);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    // now create index space for nodes
    Point<N1> node_lo, node_hi;
    for (int d = 0; d < N1; d++) {
      node_lo[d] = 0;
      node_hi[d] = num_nodes - 1;
    }
    is_nodes = Rect<N1>(node_lo, node_hi);

    Point<N2> edge_lo, edge_hi;
    for (int d = 0; d < N2; d++) {
      edge_lo[d] = 0;
      edge_hi[d] = num_edges - 1;
    }
    is_edges = Rect<N2>(edge_lo, edge_hi);

    // equal partition is used to do initial population of edges and nodes
    std::vector<IndexSpace<N1> > ss_nodes_eq;

    log_app.info() << "Creating equal subspaces\n";

    is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();

    // create instances for each of these subspaces
    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Rect<N2>));

    ri_nodes.resize(num_pieces);
    rect_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[i % memories.size()], ss_nodes_eq[i],
                                      node_fields, 0 /*SOA*/,
                                      Realm::ProfilingRequestSet()).wait();
      ri_nodes[i] = ri;

      rect_field_data[i].index_space = ss_nodes_eq[i];
      rect_field_data[i].inst = ri_nodes[i];
      rect_field_data[i].field_offset = 0;
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor p = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      Event e = p.spawn(INIT_PREIMAGE_RANGE_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  p_nodes - nodes partitioned by subgraph id (from GPU)
  //  p_nodes_cpu - nodes partitioned by subgraph id (from CPU)

  std::vector<IndexSpace<N1> > p_nodes, p_garbage_nodes, p_nodes_cpu;

  virtual Event perform_partitioning(void)
  {
    // Partition nodes by subgraph id - do this twice, once on CPU and once on GPU
    // Ensure that the results are identical

    std::vector<IndexSpace<N2>> targets;
    if (sparse_factor <= 1) {
      is_edges.create_equal_subspaces(num_spaces, 1, targets, Realm::ProfilingRequestSet()).wait();
    } else {
      targets.resize(num_spaces);
      for (int i = 0; i < num_spaces; i++) {
        targets[i] = create_sparse_index_space(is_edges.bounds, sparse_factor, random_colors, i);
      }
    }

    // We need a GPU memory for GPU partitioning
    Memory gpu_memory;
    bool found_gpu_memory = false;
    Machine machine = Machine::get_machine();
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(Memory memory : all_memories) {
      if(memory.kind() == Memory::GPU_FB_MEM) {
        gpu_memory = memory;
        found_gpu_memory = true;
        break;
      }
    }
    if (!found_gpu_memory) {
      log_app.error() << "No GPU memory found for partitioning test\n";
      return Event::NO_EVENT;
    }


    std::vector<size_t> node_fields;
    node_fields.push_back(sizeof(Rect<N2>));

    std::vector<FieldDataDescriptor<IndexSpace<N1>, Rect<N2>>> rect_field_data_gpu;
    rect_field_data_gpu.resize(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
    	copy_piece(rect_field_data[i], rect_field_data_gpu[i], node_fields, 0, gpu_memory).wait();
    }

    std::vector<DeppartEstimateInput<N1, int>> preimage_inputs(num_pieces);
    std::vector<DeppartSubspace<N2, int>> preimage_subspaces(num_spaces);
    std::vector<DeppartBufferRequirements> preimage_requirements(num_pieces);

    for (int i = 0; i < num_pieces; i++) {
      preimage_inputs[i].location = rect_field_data_gpu[i].inst.get_location();
      preimage_inputs[i].space = rect_field_data_gpu[i].index_space;
    }

    for (int i = 0; i < num_spaces; i++) {
      preimage_subspaces[i].space = targets[i];
      preimage_subspaces[i].entries = targets[i].dense() ? 1 : targets[i].sparsity.impl()->get_entries().size();
    }

    is_nodes.by_preimage_buffer_requirements(preimage_subspaces, preimage_inputs, preimage_requirements);

    for (int i = 0; i < num_pieces; i++) {
      size_t alloc_size = preimage_requirements[i].lower_bound + (preimage_requirements[i].upper_bound - preimage_requirements[i].lower_bound) * buffer_size / 100;
      alloc_piece(rect_field_data_gpu[i].scratch_buffer, alloc_size, gpu_memory).wait();
    }

    log_app.info() << "warming up" << Clock::current_time_in_microseconds() << "\n";
    Event warmup = is_nodes.create_subspaces_by_preimage(rect_field_data_gpu,
                                                  targets,
                                                  p_garbage_nodes,
                                                  Realm::ProfilingRequestSet());
    warmup.wait();

    Event gpu_call = is_nodes.create_subspaces_by_preimage(rect_field_data_gpu,
                                                  targets,
                                                  p_nodes,
                                                  Realm::ProfilingRequestSet());

    if ( wait_on_events ) {
      gpu_call.wait();
    }
    Event cpu_call = is_nodes.create_subspaces_by_preimage(rect_field_data,
                                                  targets,
                                                  p_nodes_cpu,
                                                  Realm::ProfilingRequestSet());

    if ( wait_on_events ) {
      cpu_call.wait();
    }

    return Event::merge_events({gpu_call, cpu_call});
  }

  virtual int perform_dynamic_checks(void)
  {
    // Nothing to do here
    return 0;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    if (!p_nodes.size()) {
      return p_nodes.size() != p_nodes_cpu.size();
    }

    log_app.info() << "Checking correctness of partitioning " << "\n";

    for(int i = 0; i < num_spaces; i++) {
      if (!p_nodes[i].dense() && (N1 > 1)) {
        p_nodes[i].sparsity.impl()->request_bvh();
        if (!p_nodes_cpu[i].dense()) {
          p_nodes_cpu[i].sparsity.impl()->request_bvh();
        }
      }
      for(IndexSpaceIterator<N1> it(p_nodes[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_nodes_cpu[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU has extra image point " << point.p
                            << " on piece " << i << "\n";
            errors++;
          }
        }
      }
      for(IndexSpaceIterator<N1> it(p_nodes_cpu[i]); it.valid; it.step()) {
        for(PointInRectIterator<N1> point(it.rect); point.valid; point.step()) {
          if (!p_nodes[i].contains(point.p)) {
            log_app.error() << "Mismatch! GPU is missing image point " << point.p
                          << " on piece " << i << "\n";
            errors++;
          }
        }
      }

    }
    return errors;
  }
};

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  int errors = 0;

  testcfg->print_info();

  // find all the system memories - we'll stride our data across them
  // for each memory, we'll need one CPU that can do the initialization of the data
  std::vector<Memory> sysmems;
  std::vector<Processor> procs;

  Machine machine = Machine::get_machine();
  {
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(std::set<Memory>::const_iterator it = all_memories.begin();
        it != all_memories.end(); it++) {
      Memory m = *it;

      // skip memories with no capacity for creating instances
      if(m.capacity() == 0)
        continue;

      if(m.kind() == Memory::SYSTEM_MEM) {
        sysmems.push_back(m);
        std::set<Processor> pset;
        machine.get_shared_processors(m, pset);
        Processor p = Processor::NO_PROC;
        for(std::set<Processor>::const_iterator it2 = pset.begin(); it2 != pset.end();
            it2++) {
          if(it2->kind() == Processor::LOC_PROC) {
            p = *it2;
            break;
          }
        }
        assert(p.exists());
        procs.push_back(p);
        log_app.debug() << "System mem #" << (sysmems.size() - 1) << " = "
                        << *sysmems.rbegin() << " (" << *procs.rbegin() << ")";
      }
    }
  }
  assert(sysmems.size() > 0);

  {
    Realm::TimeStamp ts("initialization", true, &log_app);

    Event e = testcfg->initialize_data(sysmems, procs);
    // wait for all initialization to be done
    e.wait();
  }

  // now actual partitioning work
  {
    Realm::TimeStamp ts("dependent partitioning work", true, &log_app);

    Event e = testcfg->perform_partitioning();

    e.wait();
  }

  // dynamic checks (which would be eliminated by compiler)
  {
    Realm::TimeStamp ts("dynamic checks", true, &log_app);
    errors += testcfg->perform_dynamic_checks();
  }

  if(!skip_check) {
    log_app.print() << "checking correctness of partitioning";
    Realm::TimeStamp ts("verification", true, &log_app);
    errors += testcfg->check_partitioning();
  }

  if(errors > 0) {
    printf("Exiting with errors\n");
    exit(1);
  }

}

// Constructor function-pointer type
using CtorFn = TestInterface* (*)(int, const char** argv);

// ---- Byfield constructors ----
template<int D>
static TestInterface* make_byfield(int argc, const char** argv) {
  return new ByfieldTest<D>(argc, argv);
}

static constexpr CtorFn BYFIELD_CTORS[3] = {
  &make_byfield<1>,
  &make_byfield<2>,
  &make_byfield<3>,
};

// ---- Image constructors ----
template<int D1, int D2>
static TestInterface* make_image(int argc, const char** argv) {
  return new ImageTest<D1, D2>(argc, argv);
}

static constexpr CtorFn IMAGE_CTORS[3][3] = {
  { &make_image<1,1>, &make_image<1,2>, &make_image<1,3> },
  { &make_image<2,1>, &make_image<2,2>, &make_image<2,3> },
  { &make_image<3,1>, &make_image<3,2>, &make_image<3,3> },
};

// ---- Image Range constructors ----
template<int D1, int D2>
static TestInterface* make_image_range(int argc, const char** argv) {
  return new ImageRangeTest<D1, D2>(argc, argv);
}

static constexpr CtorFn IMAGE_RANGE_CTORS[3][3] = {
  { &make_image_range<1,1>, &make_image_range<1,2>, &make_image_range<1,3> },
  { &make_image_range<2,1>, &make_image_range<2,2>, &make_image_range<2,3> },
  { &make_image_range<3,1>, &make_image_range<3,2>, &make_image_range<3,3> },
};

// ---- Image constructors ----
template<int D1, int D2>
static TestInterface* make_preimage(int argc, const char** argv) {
  return new PreimageTest<D1, D2>(argc, argv);
}

static constexpr CtorFn PREIMAGE_CTORS[3][3] = {
  { &make_preimage<1,1>, &make_preimage<1,2>, &make_preimage<1,3> },
  { &make_preimage<2,1>, &make_preimage<2,2>, &make_preimage<2,3> },
  { &make_preimage<3,1>, &make_preimage<3,2>, &make_preimage<3,3> },
};

// ---- Image constructors ----
template<int D1, int D2>
static TestInterface* make_preimage_range(int argc, const char** argv) {
  return new PreimageRangeTest<D1, D2>(argc, argv);
}

static constexpr CtorFn PREIMAGE_RANGE_CTORS[3][3] = {
  { &make_preimage_range<1,1>, &make_preimage_range<1,2>, &make_preimage_range<1,3> },
  { &make_preimage_range<2,1>, &make_preimage_range<2,2>, &make_preimage_range<2,3> },
  { &make_preimage_range<3,1>, &make_preimage_range<3,2>, &make_preimage_range<3,3> },
};

using TaskWrapperFn = void (*)(const void*, size_t, const void*, size_t, Processor);

static constexpr TaskWrapperFn BYFIELD_INIT_TBL[3] = {
  &ByfieldTest<1>::init_data_task_wrapper,
  &ByfieldTest<2>::init_data_task_wrapper,
  &ByfieldTest<3>::init_data_task_wrapper,
};

static constexpr TaskWrapperFn IMAGE_INIT_TBL[3][3] = {
  { &ImageTest<1,1>::init_data_task_wrapper, &ImageTest<1,2>::init_data_task_wrapper, &ImageTest<1,3>::init_data_task_wrapper },
  { &ImageTest<2,1>::init_data_task_wrapper, &ImageTest<2,2>::init_data_task_wrapper, &ImageTest<2,3>::init_data_task_wrapper },
  { &ImageTest<3,1>::init_data_task_wrapper, &ImageTest<3,2>::init_data_task_wrapper, &ImageTest<3,3>::init_data_task_wrapper },
};

static constexpr TaskWrapperFn IMAGE_RANGE_INIT_TBL[3][3] = {
  { &ImageRangeTest<1,1>::init_data_task_wrapper, &ImageRangeTest<1,2>::init_data_task_wrapper, &ImageRangeTest<1,3>::init_data_task_wrapper },
  { &ImageRangeTest<2,1>::init_data_task_wrapper, &ImageRangeTest<2,2>::init_data_task_wrapper, &ImageRangeTest<2,3>::init_data_task_wrapper },
  { &ImageRangeTest<3,1>::init_data_task_wrapper, &ImageRangeTest<3,2>::init_data_task_wrapper, &ImageRangeTest<3,3>::init_data_task_wrapper },
};

static constexpr TaskWrapperFn PREIMAGE_INIT_TBL[3][3] = {
  { &PreimageTest<1,1>::init_data_task_wrapper, &PreimageTest<1,2>::init_data_task_wrapper, &PreimageTest<1,3>::init_data_task_wrapper },
  { &PreimageTest<2,1>::init_data_task_wrapper, &PreimageTest<2,2>::init_data_task_wrapper, &PreimageTest<2,3>::init_data_task_wrapper },
  { &PreimageTest<3,1>::init_data_task_wrapper, &PreimageTest<3,2>::init_data_task_wrapper, &PreimageTest<3,3>::init_data_task_wrapper },
};

static constexpr TaskWrapperFn PREIMAGE_RANGE_INIT_TBL[3][3] = {
  { &PreimageRangeTest<1,1>::init_data_task_wrapper, &PreimageRangeTest<1,2>::init_data_task_wrapper, &PreimageRangeTest<1,3>::init_data_task_wrapper },
  { &PreimageRangeTest<2,1>::init_data_task_wrapper, &PreimageRangeTest<2,2>::init_data_task_wrapper, &PreimageRangeTest<2,3>::init_data_task_wrapper },
  { &PreimageRangeTest<3,1>::init_data_task_wrapper, &PreimageRangeTest<3,2>::init_data_task_wrapper, &PreimageRangeTest<3,3>::init_data_task_wrapper },
};

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  // parse global options
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-seed")) {
      random_seed = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-random")) {
      random_colors = true;
      continue;
    }

    if(!strcmp(argv[i], "-wait")) {
      wait_on_events = true;
      continue;
    }

    if(!strcmp(argv[i], "-show")) {
      show_graph = true;
      continue;
    }

    if(!strcmp(argv[i], "-nocheck")) {
      skip_check = true;
      continue;
    }

    if(!strcmp(argv[i], "-d1")) {
      dimension1 = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-d2")) {
      dimension2 = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "byfield")) {
      if (dimension1 < 1 || dimension1 > 3)
        assert(false && "invalid dimension");

      op = "byfield";
      testcfg = BYFIELD_CTORS[dimension1 - 1](argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "image")) {
      if (dimension1 < 1 || dimension1 > 3 || dimension2 < 1 || dimension2 > 3)
        assert(false && "invalid dimension");
      op = "image";
      testcfg = IMAGE_CTORS[dimension1 - 1][dimension2 - 1](argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "irange")) {
      if (dimension1 < 1 || dimension1 > 3 || dimension2 < 1 || dimension2 > 3)
        assert(false && "invalid dimension");
      op = "irange";
      testcfg = IMAGE_RANGE_CTORS[dimension1 - 1][dimension2 - 1](argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "preimage")) {
      if (dimension1 < 1 || dimension1 > 3 || dimension2 < 1 || dimension2 > 3)
        assert(false && "invalid dimension");
      op = "preimage";
      testcfg = PREIMAGE_CTORS[dimension1 - 1][dimension2 - 1](argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "prange")) {
      if (dimension1 < 1 || dimension1 > 3 || dimension2 < 1 || dimension2 > 3)
        assert(false && "invalid dimension");
      op = "prange";
      testcfg = PREIMAGE_RANGE_CTORS[dimension1 - 1][dimension2 - 1](argc - i, const_cast<const char **>(argv + i));
      break;
    }

    // printf("unknown parameter: %s\n", argv[i]);
  }

  // if no test specified, use circuit (with default parameters)
  if(!testcfg) {
    assert(false);
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  if (dimension1 < 1 || dimension1 > 3 || dimension2 < 1 || dimension2 > 3)
    assert(false && "invalid dimension");

  rt.register_task(INIT_BYFIELD_DATA_TASK, BYFIELD_INIT_TBL[dimension1 - 1]);
  rt.register_task(INIT_IMAGE_DATA_TASK,   IMAGE_INIT_TBL[dimension1 - 1][dimension2 - 1]);
  rt.register_task(INIT_IMAGE_RANGE_DATA_TASK,   IMAGE_RANGE_INIT_TBL[dimension1 - 1][dimension2 - 1]);
  rt.register_task(INIT_PREIMAGE_DATA_TASK,   PREIMAGE_INIT_TBL[dimension1 - 1][dimension2 - 1]);
  rt.register_task(INIT_PREIMAGE_RANGE_DATA_TASK,   PREIMAGE_RANGE_INIT_TBL[dimension1 - 1][dimension2 - 1]);

  signal(SIGALRM, sigalrm_handler);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish
  // event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  delete testcfg;

  return 0;
}
