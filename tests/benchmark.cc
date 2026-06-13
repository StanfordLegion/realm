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
#include <algorithm>
#include <map>

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
  INIT_PREIMAGE_RANGE_DATA_TASK,
  INIT_CIRCUIT_DATA_TASK,
  INIT_PENNANT_DATA_TASK,
  INIT_MINIAERO_DATA_TASK
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
  bool show_breakdown = false;
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

static std::vector<Memory> get_gpu_memories(void)
{
  Machine machine = Machine::get_machine();
  std::set<Memory> all_memories;
  std::vector<Memory> gpu_memories;
  machine.get_all_memories(all_memories);
  for(Memory memory : all_memories) {
    if(memory.kind() == Memory::GPU_FB_MEM)
      gpu_memories.push_back(memory);
  }
  if(gpu_memories.empty())
    log_app.error() << "No GPU memory found for partitioning test\n";
  return gpu_memories;
}

static void assert_piece_gpu_count(const char *app_name, int pieces,
                                   const std::vector<Memory> &gpu_memories)
{
  if((int)gpu_memories.size() != pieces) {
    log_app.error() << app_name << " requires one piece per GPU: pieces=" << pieces
                    << " available_gpus=" << gpu_memories.size();
    exit(1);
  }
}

template <typename T>
void split_evenly(T total, T pieces, std::vector<T> &cuts)
{
  cuts.resize(pieces + 1);
  for(T i = 0; i <= pieces; i++)
    cuts[i] = ((long long)total * i) / pieces;
}

template <typename T>
int find_split(const std::vector<T> &cuts, T v)
{
  assert(v >= cuts[0]);
  for(size_t i = 1; i < cuts.size(); i++)
    if(v < cuts[i])
      return i - 1;
  assert(false);
  return 0;
}

template <int N, typename T>
static DeppartSubspace<N, T> make_deppart_subspace(IndexSpace<N, T> space)
{
  DeppartSubspace<N, T> result;
  result.space = space;
  result.entries = space.dense() ? 1 : space.sparsity.impl()->get_entries().size();
  return result;
}

static size_t scaled_requirement(const DeppartBufferRequirements &req, size_t percent)
{
  return req.lower_bound + ((req.upper_bound - req.lower_bound) * percent) / 100;
}

template <typename FT>
static void alloc_by_subspace_scratch_one_to_one(
    IndexSpace<1> target, const std::vector<IndexSpace<1>> &subspaces,
    std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
    const std::vector<Memory> &gpu_memories, size_t buffer_size, bool preimage)
{
  assert(fields.size() == gpu_memories.size());
  assert(subspaces.size() == fields.size());
  for(size_t i = 0; i < fields.size(); i++) {
    std::vector<DeppartSubspace<1, int>> dp_subspaces(1);
    std::vector<DeppartEstimateInput<1, int>> inputs(1);
    std::vector<DeppartBufferRequirements> reqs;
    dp_subspaces[0] = make_deppart_subspace(subspaces[i]);
    inputs[0].location = fields[i].inst.get_location();
    inputs[0].space = fields[i].index_space;
    if(preimage)
      target.by_preimage_buffer_requirements(dp_subspaces, inputs, reqs);
    else
      target.by_image_buffer_requirements(dp_subspaces, inputs, reqs);
    alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[0], buffer_size),
                gpu_memories[i]).wait();
  }
}

template <typename FT>
static Event create_subspaces_by_image_one_to_one(
    IndexSpace<1> target, std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
    const std::vector<IndexSpace<1>> &sources,
    std::vector<IndexSpace<1>> &images, Event precondition)
{
  assert(images.empty());
  assert(fields.size() == sources.size());
  std::set<Event> events;
  images.resize(sources.size());
  for(size_t i = 0; i < sources.size(); i++) {
    std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> field(1, fields[i]);
    Event e = target.create_subspace_by_image(field, sources[i], images[i],
                                              Realm::ProfilingRequestSet(),
                                              precondition);
    if(wait_on_events)
      e.wait();
    events.insert(e);
  }
  return Event::merge_events(events);
}

template <int N, typename T>
static int compare_index_spaces(IndexSpace<N, T> gpu, IndexSpace<N, T> cpu,
                                const char *name, int index)
{
  int errors = 0;
  gpu.make_valid().wait();
  cpu.make_valid().wait();
  if(!gpu.dense())
    gpu.sparsity.impl()->request_bvh();
  if(!cpu.dense())
    cpu.sparsity.impl()->request_bvh();
  for(IndexSpaceIterator<N, T> it(gpu); it.valid; it.step()) {
    for(PointInRectIterator<N, T> point(it.rect); point.valid; point.step()) {
      if(!cpu.contains(point.p)) {
        log_app.error() << "Mismatch! GPU has extra " << name << "[" << index
                        << "] point " << point.p;
        errors++;
      }
    }
  }
  for(IndexSpaceIterator<N, T> it(cpu); it.valid; it.step()) {
    for(PointInRectIterator<N, T> point(it.rect); point.valid; point.step()) {
      if(!gpu.contains(point.p)) {
        log_app.error() << "Mismatch! GPU is missing " << name << "[" << index
                        << "] point " << point.p;
        errors++;
      }
    }
  }
  return errors;
}

template <int N, typename T>
static int compare_index_space_vectors(const std::vector<IndexSpace<N, T>> &gpu,
                                       const std::vector<IndexSpace<N, T>> &cpu,
                                       const char *name)
{
  int errors = 0;
  if(gpu.size() != cpu.size()) {
    log_app.error() << "Mismatch! " << name << " sizes differ: gpu=" << gpu.size()
                    << " cpu=" << cpu.size();
    errors++;
  }
  size_t count = std::min(gpu.size(), cpu.size());
  for(size_t i = 0; i < count; i++)
    errors += compare_index_spaces(gpu[i], cpu[i], name, i);
  return errors;
}

template <int N, typename T>
static void destroy_index_space_vector(std::vector<IndexSpace<N, T>> &spaces)
{
  for(size_t i = 0; i < spaces.size(); i++)
    spaces[i].destroy();
  spaces.clear();
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

class CircuitTest : public TestInterface {
public:
  int num_nodes = 100;
  int num_edges = 10;
  int num_pieces = 2;
  int pct_wire_in_piece = 50;
  size_t buffer_size = 100;

  CircuitTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {
      if(!strcmp(argv[i], "-n")) {
        num_nodes = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-e")) {
        num_edges = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-p")) {
        num_pieces = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-buffer") || !strcmp(argv[i], "-buf")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }

    if(num_nodes <= 0 || num_edges <= 0 || num_pieces <= 0 || buffer_size > 100) {
      log_app.error() << "Invalid circuit config: nodes=" << num_nodes
                      << " edges=" << num_edges << " pieces=" << num_pieces
                      << " buffer size=" << buffer_size;
      exit(1);
    }
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes, ri_edges;
  };

  enum PRNGStreams
  {
    NODE_SUBCKT_STREAM,
    EDGE_IN_NODE_STREAM,
    EDGE_OUT_NODE_STREAM1,
    EDGE_OUT_NODE_STREAM2,
  };

  void random_node_data(int idx, int &subckt)
  {
    if(random_colors)
      subckt = Philox_2x32<>::rand_int(random_seed, idx, NODE_SUBCKT_STREAM, num_pieces);
    else
      subckt = idx * num_pieces / num_nodes;
  }

  void random_edge_data(int idx, Point<1> &in_node, Point<1> &out_node)
  {
    if(random_colors) {
      in_node = Philox_2x32<>::rand_int(random_seed, idx, EDGE_IN_NODE_STREAM, num_nodes);
      out_node =
          Philox_2x32<>::rand_int(random_seed, idx, EDGE_OUT_NODE_STREAM1, num_nodes);
    } else {
      int subckt = idx * num_pieces / num_edges;
      int n_lo = subckt * num_nodes / num_pieces;
      int n_hi = (subckt + 1) * num_nodes / num_pieces;
      in_node = n_lo + Philox_2x32<>::rand_int(random_seed, idx, EDGE_IN_NODE_STREAM,
                                               n_hi - n_lo);
      int pct = Philox_2x32<>::rand_int(random_seed, idx, EDGE_OUT_NODE_STREAM2, 100);
      if(pct < pct_wire_in_piece)
        out_node = n_lo + Philox_2x32<>::rand_int(random_seed, idx, EDGE_OUT_NODE_STREAM1,
                                                  n_hi - n_lo);
      else
        out_node =
            Philox_2x32<>::rand_int(random_seed, idx, EDGE_OUT_NODE_STREAM1, num_nodes);
    }
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    CircuitTest *me = (CircuitTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    i_args.ri_nodes.fetch_metadata(p).wait();
    i_args.ri_edges.fetch_metadata(p).wait();

    IndexSpace<1> is_nodes = i_args.ri_nodes.get_indexspace<1>();
    IndexSpace<1> is_edges = i_args.ri_edges.get_indexspace<1>();

    {
      AffineAccessor<int, 1> a_subckt_id(i_args.ri_nodes, 0);
      for(int i = is_nodes.bounds.lo; i <= is_nodes.bounds.hi; i++) {
        int subckt;
        random_node_data(i, subckt);
        a_subckt_id.write(i, subckt);
      }
    }

    {
      AffineAccessor<Point<1>, 1> a_in_node(i_args.ri_edges, 0);
      AffineAccessor<Point<1>, 1> a_out_node(i_args.ri_edges, sizeof(Point<1>));
      for(int i = is_edges.bounds.lo; i <= is_edges.bounds.hi; i++) {
        Point<1> in_node, out_node;
        random_edge_data(i, in_node, out_node);
        a_in_node.write(i, in_node);
        a_out_node.write(i, out_node);
      }
    }
  }

  IndexSpace<1> is_nodes, is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<IndexSpace<1>, int>> subckt_field_data;
  std::vector<RegionInstance> ri_edges;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> in_node_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> out_node_field_data;

  struct Outputs {
    IndexSpace<1> is_shared, is_private;
    bool has_node_sets = false;
    std::vector<IndexSpace<1>> p_nodes;
    std::vector<IndexSpace<1>> p_pvt, p_shr, p_ghost, p_edges;
  };

  Outputs gpu, cpu, garbage;

  virtual void print_info(void)
  {
    printf("Realm dependent partitioning benchmark - circuit: %d nodes, %d edges, %d pieces\n",
           num_nodes, num_edges, num_pieces);
  }

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    is_nodes = Rect<1>(0, num_nodes - 1);
    is_edges = Rect<1>(0, num_edges - 1);

    std::vector<IndexSpace<1>> ss_nodes_eq, ss_edges_eq;
    is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq,
                                    Realm::ProfilingRequestSet()).wait();
    is_edges.create_equal_subspaces(num_pieces, 1, ss_edges_eq,
                                    Realm::ProfilingRequestSet()).wait();

    std::vector<size_t> node_fields(1, sizeof(int));
    std::vector<size_t> edge_fields;
    edge_fields.push_back(sizeof(Point<1>));
    edge_fields.push_back(sizeof(Point<1>));

    ri_nodes.resize(num_pieces);
    subckt_field_data.resize(num_pieces);
    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance::create_instance(ri_nodes[i], memories[i % memories.size()],
                                      ss_nodes_eq[i], node_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      subckt_field_data[i].index_space = ss_nodes_eq[i];
      subckt_field_data[i].inst = ri_nodes[i];
      subckt_field_data[i].field_offset = 0;
    }

    ri_edges.resize(num_pieces);
    in_node_field_data.resize(num_pieces);
    out_node_field_data.resize(num_pieces);
    for(size_t i = 0; i < ss_edges_eq.size(); i++) {
      RegionInstance::create_instance(ri_edges[i], memories[i % memories.size()],
                                      ss_edges_eq[i], edge_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      in_node_field_data[i].index_space = ss_edges_eq[i];
      in_node_field_data[i].inst = ri_edges[i];
      in_node_field_data[i].field_offset = 0;
      out_node_field_data[i].index_space = ss_edges_eq[i];
      out_node_field_data[i].inst = ri_edges[i];
      out_node_field_data[i].field_offset = sizeof(Point<1>);
    }

    std::set<Event> events;
    for(int i = 0; i < num_pieces; i++) {
      Processor proc = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      args.ri_edges = ri_edges[i];
      events.insert(proc.spawn(INIT_CIRCUIT_DATA_TASK, &args, sizeof(args)));
    }
    return Event::merge_events(events);
  }

  template <typename FT>
  void alloc_by_field_scratch(IndexSpace<1> target,
                              std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
                              const std::vector<Memory> &gpu_memories)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    target.by_field_buffer_requirements(inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  template <typename FT>
  void alloc_by_subspace_scratch(
      IndexSpace<1> target, const std::vector<IndexSpace<1>> &subspaces,
      std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
      const std::vector<Memory> &gpu_memories,
      bool preimage)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartSubspace<1, int>> dp_subspaces(subspaces.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < subspaces.size(); i++)
      dp_subspaces[i] = make_deppart_subspace(subspaces[i]);
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    if(preimage)
      target.by_preimage_buffer_requirements(dp_subspaces, inputs, reqs);
    else
      target.by_image_buffer_requirements(dp_subspaces, inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  Event run_partitioning(std::vector<FieldDataDescriptor<IndexSpace<1>, int>> &subckt_data,
                         std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &in_data,
                         std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &out_data,
                         Outputs &out, const char *breakdown_run = 0)
  {
    long long total_start = 0;
    long long stage_start = 0;
    long long nodes_byfield_us = 0;
    long long edges_preimage_us = 0;
    long long extra_image_us = 0;
    long long ghost_diff_us = 0;
    long long shared_union_us = 0;
    long long private_diff_us = 0;
    long long shared_isect_us = 0;
    long long private_isect_us = 0;
    if(breakdown_run)
      total_start = Clock::current_time_in_microseconds();

    std::vector<int> colors(num_pieces);
    for(int i = 0; i < num_pieces; i++)
      colors[i] = i;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e1 = is_nodes.create_subspaces_by_field(subckt_data, colors, out.p_nodes,
                                                  Realm::ProfilingRequestSet());
    if(wait_on_events || breakdown_run)
      e1.wait();
    if(breakdown_run)
      nodes_byfield_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e2 = is_edges.create_subspaces_by_preimage(in_data, out.p_nodes, out.p_edges,
                                                     Realm::ProfilingRequestSet(), e1);
    if(wait_on_events || breakdown_run)
      e2.wait();
    if(breakdown_run)
      edges_preimage_us = Clock::current_time_in_microseconds() - stage_start;

    std::vector<IndexSpace<1>> p_extra_nodes;
    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e3 = create_subspaces_by_image_one_to_one(is_nodes, out_data, out.p_edges,
                                                    p_extra_nodes, e2);
    if(wait_on_events || breakdown_run)
      e3.wait();
    if(breakdown_run)
      extra_image_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e4 = IndexSpace<1>::compute_differences(p_extra_nodes, out.p_nodes, out.p_ghost,
                                                  Realm::ProfilingRequestSet(), e3);
    if(wait_on_events || breakdown_run)
      e4.wait();
    if(breakdown_run)
      ghost_diff_us = Clock::current_time_in_microseconds() - stage_start;
    for(unsigned idx = 0; idx < p_extra_nodes.size(); idx++)
      p_extra_nodes[idx].destroy(e4);

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e5 = IndexSpace<1>::compute_union(out.p_ghost, out.is_shared,
                                            Realm::ProfilingRequestSet(), e4);
    if(wait_on_events || breakdown_run)
      e5.wait();
    if(breakdown_run)
      shared_union_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e6 = IndexSpace<1>::compute_difference(is_nodes, out.is_shared, out.is_private,
                                                 Realm::ProfilingRequestSet(), e5);
    if(wait_on_events || breakdown_run)
      e6.wait();
    if(breakdown_run)
      private_diff_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e7 = IndexSpace<1>::compute_intersections(out.p_nodes, out.is_shared, out.p_shr,
                                                    Realm::ProfilingRequestSet(), e5);
    if(wait_on_events || breakdown_run)
      e7.wait();
    if(breakdown_run)
      shared_isect_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e8 = IndexSpace<1>::compute_intersections(out.p_nodes, out.is_private, out.p_pvt,
                                                    Realm::ProfilingRequestSet(), e6);
    if(wait_on_events || breakdown_run)
      e8.wait();
    if(breakdown_run)
      private_isect_us = Clock::current_time_in_microseconds() - stage_start;
    out.has_node_sets = true;

    if(breakdown_run) {
      long long total_us = Clock::current_time_in_microseconds() - total_start;
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=nodes_byfield,us=%lld\n",
             breakdown_run, num_pieces, nodes_byfield_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=edges_preimage,us=%lld\n",
             breakdown_run, num_pieces, edges_preimage_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=extra_image,us=%lld\n",
             breakdown_run, num_pieces, extra_image_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=ghost_diff,us=%lld\n",
             breakdown_run, num_pieces, ghost_diff_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=shared_union,us=%lld\n",
             breakdown_run, num_pieces, shared_union_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=private_diff,us=%lld\n",
             breakdown_run, num_pieces, private_diff_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=shared_isect,us=%lld\n",
             breakdown_run, num_pieces, shared_isect_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=private_isect,us=%lld\n",
             breakdown_run, num_pieces, private_isect_us);
      printf("BREAKDOWN,op=circuit,run=%s,pieces=%d,stage=total,us=%lld\n",
             breakdown_run, num_pieces, total_us);
    }

    return Event::merge_events(e7, e8);
  }

  virtual Event perform_partitioning(void)
  {
    std::vector<Memory> gpu_memories = get_gpu_memories();
    if(gpu_memories.empty())
      return Event::NO_EVENT;
    assert_piece_gpu_count("circuit", num_pieces, gpu_memories);

    std::vector<size_t> node_fields(1, sizeof(int));
    std::vector<size_t> edge_fields;
    edge_fields.push_back(sizeof(Point<1>));
    edge_fields.push_back(sizeof(Point<1>));

    std::vector<FieldDataDescriptor<IndexSpace<1>, int>> subckt_gpu(num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> in_gpu(num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> out_gpu(num_pieces);
    for(int i = 0; i < num_pieces; i++) {
      copy_piece(subckt_field_data[i], subckt_gpu[i], node_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(in_node_field_data[i], in_gpu[i], edge_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(out_node_field_data[i], out_gpu[i], edge_fields, 1,
                 gpu_memories[i]).wait();
    }

    std::vector<int> colors(num_pieces);
    for(int i = 0; i < num_pieces; i++)
      colors[i] = i;

    alloc_by_field_scratch(is_nodes, subckt_gpu, gpu_memories);
    Event warm_field = is_nodes.create_subspaces_by_field(
        subckt_gpu, colors, garbage.p_nodes, Realm::ProfilingRequestSet());
    warm_field.wait();
    alloc_by_subspace_scratch(is_edges, garbage.p_nodes, in_gpu, gpu_memories, true);
    Event warm_preimage = is_edges.create_subspaces_by_preimage(
        in_gpu, garbage.p_nodes, garbage.p_edges, Realm::ProfilingRequestSet(),
        warm_field);
    warm_preimage.wait();
    alloc_by_subspace_scratch_one_to_one(is_nodes, garbage.p_edges, out_gpu,
                                         gpu_memories, buffer_size, false);
    destroy_outputs(garbage);

    Event warmup = run_partitioning(subckt_gpu, in_gpu, out_gpu, garbage);
    warmup.wait();

    long long gpu_start = Clock::current_time_in_microseconds();
    Event gpu_call = run_partitioning(subckt_gpu, in_gpu, out_gpu, gpu,
                                      show_breakdown ? "gpu" : 0);
    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - gpu_start;

    long long cpu_start = Clock::current_time_in_microseconds();
    Event cpu_call =
        run_partitioning(subckt_field_data, in_node_field_data, out_node_field_data, cpu,
                         show_breakdown ? "cpu" : 0);
    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - cpu_start;

    printf("RESULT,op=circuit,num_nodes=%d,num_edges=%d,num_pieces=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
           num_nodes, num_edges, num_pieces, buffer_size, gpu_us, cpu_us);

    return Event::merge_events(gpu_call, cpu_call);
  }

  virtual int perform_dynamic_checks(void) { return 0; }

  void destroy_outputs(Outputs &out)
  {
    if(out.has_node_sets) {
      out.is_shared.destroy();
      out.is_private.destroy();
      out.has_node_sets = false;
    }
    destroy_index_space_vector(out.p_nodes);
    destroy_index_space_vector(out.p_pvt);
    destroy_index_space_vector(out.p_shr);
    destroy_index_space_vector(out.p_ghost);
    destroy_index_space_vector(out.p_edges);
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;
    errors += compare_index_spaces(gpu.is_shared, cpu.is_shared, "is_shared", 0);
    errors += compare_index_spaces(gpu.is_private, cpu.is_private, "is_private", 0);
    errors += compare_index_space_vectors(gpu.p_edges, cpu.p_edges, "p_edges");
    errors += compare_index_space_vectors(gpu.p_pvt, cpu.p_pvt, "p_pvt");
    errors += compare_index_space_vectors(gpu.p_shr, cpu.p_shr, "p_shr");
    errors += compare_index_space_vectors(gpu.p_ghost, cpu.p_ghost, "p_ghost");
    destroy_outputs(gpu);
    destroy_outputs(cpu);
    destroy_outputs(garbage);
    return errors;
  }
};

class PennantTest : public TestInterface {
public:
  int nzx = 10;
  int nzy = 10;
  int numpcx = 2;
  int numpcy = 2;
  size_t buffer_size = 100;

  int npx, npy;
  int nz, ns, np, numpc;
  std::vector<int> zxbound, zybound;
  std::vector<int> lz, ls, lp;

  typedef int INDEXTYPE;
  static const INDEXTYPE FIRST_INDEX = -2000000000;

  PennantTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {
      if(!strcmp(argv[i], "-nzx")) {
        nzx = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-nzy")) {
        nzy = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-numpcx")) {
        numpcx = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-numpcy")) {
        numpcy = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-nz")) {
        int v = atoi(argv[++i]);
        nzx = nzy = v;
        continue;
      }
      if(!strcmp(argv[i], "-numpc")) {
        int v = atoi(argv[++i]);
        numpcx = numpcy = v;
        continue;
      }
      if(!strcmp(argv[i], "-p")) {
        int v = atoi(argv[++i]);
        numpcx = v;
        numpcy = 1;
        continue;
      }
      if(!strcmp(argv[i], "-buffer") || !strcmp(argv[i], "-buf")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }

    if(nzx <= 0 || nzy <= 0 || numpcx <= 0 || numpcy <= 0 || buffer_size > 100) {
      log_app.error() << "Invalid pennant config";
      exit(1);
    }

    npx = nzx + 1;
    npy = nzy + 1;
    numpc = numpcx * numpcy;

    zxbound.resize(numpcx + 1);
    for(int i = 0; i <= numpcx; i++)
      zxbound[i] = (i * nzx) / numpcx;

    zybound.resize(numpcy + 1);
    for(int i = 0; i <= numpcy; i++)
      zybound[i] = (i * nzy) / numpcy;

    nz = ns = np = 0;
    for(int pcy = 0; pcy < numpcy; pcy++) {
      for(int pcx = 0; pcx < numpcx; pcx++) {
        int lx = zxbound[pcx + 1] - zxbound[pcx];
        int ly = zybound[pcy + 1] - zybound[pcy];

        int zones = lx * ly;
        int sides = zones * 4;
        int points = ((pcx == 0) ? (lx + 1) : lx) * ((pcy == 0) ? (ly + 1) : ly);

        lz.push_back(zones);
        ls.push_back(sides);
        lp.push_back(points);
        nz += zones;
        ns += sides;
        np += points;
      }
    }

    assert(nz == (nzx * nzy));
    assert(ns == (4 * nzx * nzy));
    assert(np == (npx * npy));
  }

  virtual void print_info(void)
  {
    printf("Realm dependent partitioning benchmark - pennant: %d x %d zones, %d x %d pieces\n",
           nzx, nzy, numpcx, numpcy);
  }

  IndexSpace<1> is_zones, is_sides, is_points;
  std::vector<RegionInstance> ri_zones;
  std::vector<FieldDataDescriptor<IndexSpace<1>, int>> zone_color_field_data;
  std::vector<RegionInstance> ri_sides;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> side_mapsz_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> side_mapss3_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> side_mapsp1_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, bool>> side_ok_field_data;

  struct InitDataArgs {
    int index;
    RegionInstance ri_zones, ri_sides;
  };

  struct Outputs {
    std::vector<IndexSpace<1>> p_zones, p_sides, p_points;
  };

  Outputs gpu, cpu, garbage;

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    is_zones = Rect<1>(FIRST_INDEX, FIRST_INDEX + nz - 1);
    is_sides = Rect<1>(FIRST_INDEX, FIRST_INDEX + ns - 1);
    is_points = Rect<1>(FIRST_INDEX, FIRST_INDEX + np - 1);

    std::vector<IndexSpace<1>> ss_zones_w, ss_sides_w, ss_points_w;
    is_zones.create_weighted_subspaces(numpc, 1, lz, ss_zones_w,
                                       Realm::ProfilingRequestSet()).wait();
    is_sides.create_weighted_subspaces(numpc, 1, ls, ss_sides_w,
                                       Realm::ProfilingRequestSet()).wait();
    is_points.create_weighted_subspaces(numpc, 1, lp, ss_points_w,
                                        Realm::ProfilingRequestSet()).wait();

    std::vector<size_t> zone_fields(1, sizeof(int));
    std::vector<size_t> side_fields;
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(bool));

    ri_zones.resize(numpc);
    zone_color_field_data.resize(numpc);
    for(size_t i = 0; i < ss_zones_w.size(); i++) {
      RegionInstance::create_instance(ri_zones[i], memories[i % memories.size()],
                                      ss_zones_w[i], zone_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      zone_color_field_data[i].index_space = ss_zones_w[i];
      zone_color_field_data[i].inst = ri_zones[i];
      zone_color_field_data[i].field_offset = 0;
    }

    ri_sides.resize(numpc);
    side_mapsz_field_data.resize(numpc);
    side_mapss3_field_data.resize(numpc);
    side_mapsp1_field_data.resize(numpc);
    side_ok_field_data.resize(numpc);
    for(size_t i = 0; i < ss_sides_w.size(); i++) {
      RegionInstance::create_instance(ri_sides[i], memories[i % memories.size()],
                                      ss_sides_w[i], side_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      side_mapsz_field_data[i].index_space = ss_sides_w[i];
      side_mapsz_field_data[i].inst = ri_sides[i];
      side_mapsz_field_data[i].field_offset = 0;
      side_mapss3_field_data[i].index_space = ss_sides_w[i];
      side_mapss3_field_data[i].inst = ri_sides[i];
      side_mapss3_field_data[i].field_offset = sizeof(Point<1>);
      side_mapsp1_field_data[i].index_space = ss_sides_w[i];
      side_mapsp1_field_data[i].inst = ri_sides[i];
      side_mapsp1_field_data[i].field_offset = 2 * sizeof(Point<1>);
      side_ok_field_data[i].index_space = ss_sides_w[i];
      side_ok_field_data[i].inst = ri_sides[i];
      side_ok_field_data[i].field_offset = 3 * sizeof(Point<1>);
    }

    std::set<Event> events;
    for(int i = 0; i < numpc; i++) {
      Processor proc = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_zones = ri_zones[i];
      args.ri_sides = ri_sides[i];
      events.insert(proc.spawn(INIT_PENNANT_DATA_TASK, &args, sizeof(args)));
    }
    return Event::merge_events(events);
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    PennantTest *me = (PennantTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  Point<1> global_point_pointer(int py, int px) const
  {
    int pp = FIRST_INDEX;
    int dy;
    if(py > zybound[1]) {
      int pcy = 1;
      while(py > zybound[pcy + 1])
        pcy++;
      int slabs = zybound[pcy] + 1;
      pp += npx * slabs;
      py -= slabs;
      dy = zybound[pcy + 1] - zybound[pcy];
    } else {
      dy = zybound[1] + 1;
    }

    int dx;
    if(px > zxbound[1]) {
      int pcx = 1;
      while(px > zxbound[pcx + 1])
        pcx++;
      int strips = zxbound[pcx] + 1;
      pp += dy * strips;
      px -= strips;
      dx = zxbound[pcx + 1] - zxbound[pcx];
    } else {
      dx = zxbound[1] + 1;
    }

    pp += py * dx + px;
    return pp;
  }

  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    i_args.ri_zones.fetch_metadata(p).wait();
    i_args.ri_sides.fetch_metadata(p).wait();

    IndexSpace<1> is_zones = i_args.ri_zones.get_indexspace<1>();
    IndexSpace<1> is_sides = i_args.ri_sides.get_indexspace<1>();

    int pcx = i_args.index % numpcx;
    int pcy = i_args.index / numpcx;

    int zxlo = zxbound[pcx];
    int zxhi = zxbound[pcx + 1];
    int zylo = zybound[pcy];
    int zyhi = zybound[pcy + 1];

    AffineAccessor<int, 1> a_zone_color(i_args.ri_zones, 0);
    AffineAccessor<Point<1>, 1> a_side_mapsz(i_args.ri_sides, 0);
    AffineAccessor<Point<1>, 1> a_side_mapss3(i_args.ri_sides, sizeof(Point<1>));
    AffineAccessor<Point<1>, 1> a_side_mapsp1(i_args.ri_sides, 2 * sizeof(Point<1>));
    AffineAccessor<bool, 1> a_side_ok(i_args.ri_sides, 3 * sizeof(Point<1>));

    Point<1> pz = is_zones.bounds.lo;
    Point<1> ps = is_sides.bounds.lo;

    for(int zy = zylo; zy < zyhi; zy++) {
      for(int zx = zxlo; zx < zxhi; zx++) {
        Point<1> ps0 = ps;
        ps[0]++;
        Point<1> ps1 = ps;
        ps[0]++;
        Point<1> ps2 = ps;
        ps[0]++;
        Point<1> ps3 = ps;
        ps[0]++;

        Point<1> pp0 = global_point_pointer(zy, zx);
        Point<1> pp1 = global_point_pointer(zy + 1, zx);
        Point<1> pp2 = global_point_pointer(zy + 1, zx + 1);
        Point<1> pp3 = global_point_pointer(zy, zx + 1);

        a_zone_color.write(pz, i_args.index);

        a_side_mapsz.write(ps0, pz);
        a_side_mapsz.write(ps1, pz);
        a_side_mapsz.write(ps2, pz);
        a_side_mapsz.write(ps3, pz);

        a_side_mapss3.write(ps0, ps1);
        a_side_mapss3.write(ps1, ps2);
        a_side_mapss3.write(ps2, ps3);
        a_side_mapss3.write(ps3, ps0);

        a_side_mapsp1.write(ps0, pp0);
        a_side_mapsp1.write(ps1, pp1);
        a_side_mapsp1.write(ps2, pp2);
        a_side_mapsp1.write(ps3, pp3);

        a_side_ok.write(ps0, true);
        a_side_ok.write(ps1, true);
        a_side_ok.write(ps2, true);
        a_side_ok.write(ps3, true);

        pz[0]++;
      }
    }
    assert(pz[0] == is_zones.bounds.hi[0] + 1);
    assert(ps[0] == is_sides.bounds.hi[0] + 1);
  }

  template <typename FT>
  void alloc_by_field_scratch(IndexSpace<1> target,
                              std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
                              const std::vector<Memory> &gpu_memories)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    target.by_field_buffer_requirements(inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  template <typename FT>
  void alloc_by_subspace_scratch(
      IndexSpace<1> target, const std::vector<IndexSpace<1>> &subspaces,
      std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
      const std::vector<Memory> &gpu_memories,
      bool preimage)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartSubspace<1, int>> dp_subspaces(subspaces.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < subspaces.size(); i++)
      dp_subspaces[i] = make_deppart_subspace(subspaces[i]);
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    if(preimage)
      target.by_preimage_buffer_requirements(dp_subspaces, inputs, reqs);
    else
      target.by_image_buffer_requirements(dp_subspaces, inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  Event run_partitioning(std::vector<FieldDataDescriptor<IndexSpace<1>, int>> &zone_color,
                         std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &mapsz,
                         std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &mapsp1,
                         std::vector<FieldDataDescriptor<IndexSpace<1>, bool>> &side_ok,
                         Outputs &out, const char *breakdown_run = 0)
  {
    long long total_start = 0;
    long long stage_start = 0;
    long long side_ok_byfield_us = 0;
    long long bad_zone_image_us = 0;
    long long good_zone_diff_us = 0;
    long long zone_byfield_us = 0;
    long long side_preimage_us = 0;
    long long point_image_us = 0;
    if(breakdown_run)
      total_start = Clock::current_time_in_microseconds();

    IndexSpace<1> bad_sides;
    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e1 = is_sides.create_subspace_by_field(side_ok, false, bad_sides,
                                                 Realm::ProfilingRequestSet());
    e1.wait();
    if(breakdown_run)
      side_ok_byfield_us = Clock::current_time_in_microseconds() - stage_start;

    IndexSpace<1> good_zones;
    Event e3 = e1;
    bool destroy_good_zones = false;
    if(bad_sides.volume() == 0) {
      good_zones = is_zones;
      bad_sides.destroy(e1);
    } else {
      IndexSpace<1> bad_zones;
      if(breakdown_run)
        stage_start = Clock::current_time_in_microseconds();
      Event e2 = is_zones.create_subspace_by_image(mapsz, bad_sides, bad_zones,
                                                   Realm::ProfilingRequestSet(), e1);
      if(wait_on_events || breakdown_run)
        e2.wait();
      if(breakdown_run)
        bad_zone_image_us = Clock::current_time_in_microseconds() - stage_start;
      bad_sides.destroy(e2);

      if(breakdown_run)
        stage_start = Clock::current_time_in_microseconds();
      e3 = IndexSpace<1>::compute_difference(is_zones, bad_zones, good_zones,
                                             Realm::ProfilingRequestSet(), e2);
      if(wait_on_events || breakdown_run)
        e3.wait();
      if(breakdown_run)
        good_zone_diff_us = Clock::current_time_in_microseconds() - stage_start;
      bad_zones.destroy(e3);
      destroy_good_zones = true;
    }

    std::vector<int> colors(numpc);
    for(int i = 0; i < numpc; i++)
      colors[i] = i;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e4 = good_zones.create_subspaces_by_field(zone_color, colors, out.p_zones,
                                                    Realm::ProfilingRequestSet(), e3);
    if(wait_on_events || breakdown_run)
      e4.wait();
    if(breakdown_run)
      zone_byfield_us = Clock::current_time_in_microseconds() - stage_start;
    if(destroy_good_zones)
      good_zones.destroy(e4);

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e5 = is_sides.create_subspaces_by_preimage(mapsz, out.p_zones, out.p_sides,
                                                     Realm::ProfilingRequestSet(), e4);
    if(wait_on_events || breakdown_run)
      e5.wait();
    if(breakdown_run)
      side_preimage_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e6 = create_subspaces_by_image_one_to_one(is_points, mapsp1, out.p_sides,
                                                    out.p_points, e5);
    if(wait_on_events || breakdown_run)
      e6.wait();
    if(breakdown_run)
      point_image_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run) {
      long long total_us = Clock::current_time_in_microseconds() - total_start;
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=side_ok_byfield,us=%lld\n",
             breakdown_run, numpc, side_ok_byfield_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=bad_zone_image,us=%lld\n",
             breakdown_run, numpc, bad_zone_image_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=good_zone_diff,us=%lld\n",
             breakdown_run, numpc, good_zone_diff_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=zone_byfield,us=%lld\n",
             breakdown_run, numpc, zone_byfield_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=side_preimage,us=%lld\n",
             breakdown_run, numpc, side_preimage_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=point_image,us=%lld\n",
             breakdown_run, numpc, point_image_us);
      printf("BREAKDOWN,op=pennant,run=%s,pieces=%d,stage=total,us=%lld\n",
             breakdown_run, numpc, total_us);
    }

    return e6;
  }

  virtual Event perform_partitioning(void)
  {
    std::vector<Memory> gpu_memories = get_gpu_memories();
    if(gpu_memories.empty())
      return Event::NO_EVENT;
    assert_piece_gpu_count("pennant", numpc, gpu_memories);

    std::vector<size_t> zone_fields(1, sizeof(int));
    std::vector<size_t> side_fields;
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(Point<1>));
    side_fields.push_back(sizeof(bool));

    std::vector<FieldDataDescriptor<IndexSpace<1>, int>> zone_color_gpu(numpc);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> mapsz_gpu(numpc);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> mapsp1_gpu(numpc);
    std::vector<FieldDataDescriptor<IndexSpace<1>, bool>> side_ok_gpu(numpc);
    for(int i = 0; i < numpc; i++) {
      copy_piece(zone_color_field_data[i], zone_color_gpu[i], zone_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(side_mapsz_field_data[i], mapsz_gpu[i], side_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(side_mapsp1_field_data[i], mapsp1_gpu[i], side_fields, 2,
                 gpu_memories[i]).wait();
      copy_piece(side_ok_field_data[i], side_ok_gpu[i], side_fields, 3,
                 gpu_memories[i]).wait();
    }

    alloc_by_field_scratch(is_sides, side_ok_gpu, gpu_memories);
    IndexSpace<1> bad_sides;
    Event e1 = is_sides.create_subspace_by_field(side_ok_gpu, false, bad_sides,
                                                 Realm::ProfilingRequestSet());
    e1.wait();
    IndexSpace<1> good_zones;
    Event e3 = e1;
    bool destroy_good_zones = false;
    IndexSpace<1> bad_zones;
    if(bad_sides.volume() == 0) {
      good_zones = is_zones;
      bad_sides.destroy(e1);
    } else {
      std::vector<IndexSpace<1>> tmp_bad_sides(1, bad_sides);
      alloc_by_subspace_scratch(is_zones, tmp_bad_sides, mapsz_gpu, gpu_memories,
                                false);
      Event e2 = is_zones.create_subspace_by_image(mapsz_gpu, bad_sides, bad_zones,
                                                   Realm::ProfilingRequestSet(), e1);
      e2.wait();
      e3 = IndexSpace<1>::compute_difference(is_zones, bad_zones, good_zones,
                                             Realm::ProfilingRequestSet(), e2);
      e3.wait();
      bad_sides.destroy(e2);
      bad_zones.destroy(e3);
      destroy_good_zones = true;
    }
    alloc_by_field_scratch(good_zones, zone_color_gpu, gpu_memories);
    std::vector<int> colors(numpc);
    for(int i = 0; i < numpc; i++)
      colors[i] = i;
    Event e4 = good_zones.create_subspaces_by_field(zone_color_gpu, colors,
                                                    garbage.p_zones,
                                                    Realm::ProfilingRequestSet(), e3);
    e4.wait();
    alloc_by_subspace_scratch(is_sides, garbage.p_zones, mapsz_gpu, gpu_memories,
                              true);
    Event e5 = is_sides.create_subspaces_by_preimage(mapsz_gpu, garbage.p_zones,
                                                     garbage.p_sides,
                                                     Realm::ProfilingRequestSet(), e4);
    e5.wait();
    alloc_by_subspace_scratch_one_to_one(is_points, garbage.p_sides, mapsp1_gpu,
                                         gpu_memories, buffer_size, false);
    if(destroy_good_zones)
      good_zones.destroy();
    destroy_outputs(garbage);

    Event warmup =
        run_partitioning(zone_color_gpu, mapsz_gpu, mapsp1_gpu, side_ok_gpu, garbage);
    warmup.wait();

    long long gpu_start = Clock::current_time_in_microseconds();
    Event gpu_call = run_partitioning(zone_color_gpu, mapsz_gpu, mapsp1_gpu, side_ok_gpu,
                                      gpu, show_breakdown ? "gpu" : 0);
    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - gpu_start;

    long long cpu_start = Clock::current_time_in_microseconds();
    Event cpu_call = run_partitioning(zone_color_field_data, side_mapsz_field_data,
                                      side_mapsp1_field_data, side_ok_field_data, cpu,
                                      show_breakdown ? "cpu" : 0);
    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - cpu_start;

    printf("RESULT,op=pennant,nzx=%d,nzy=%d,numpcx=%d,numpcy=%d,num_zones=%d,num_sides=%d,num_points=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
           nzx, nzy, numpcx, numpcy, nz, ns, np, buffer_size, gpu_us, cpu_us);

    return Event::merge_events(gpu_call, cpu_call);
  }

  virtual int perform_dynamic_checks(void) { return 0; }

  void destroy_outputs(Outputs &out)
  {
    destroy_index_space_vector(out.p_zones);
    destroy_index_space_vector(out.p_sides);
    destroy_index_space_vector(out.p_points);
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;
    errors += compare_index_space_vectors(gpu.p_zones, cpu.p_zones, "p_zones");
    errors += compare_index_space_vectors(gpu.p_sides, cpu.p_sides, "p_sides");
    errors += compare_index_space_vectors(gpu.p_points, cpu.p_points, "p_points");
    destroy_outputs(gpu);
    destroy_outputs(cpu);
    destroy_outputs(garbage);
    return errors;
  }
};

class MiniAeroTest : public TestInterface {
public:
  enum ProblemType
  {
    PTYPE_0,
    PTYPE_1,
    PTYPE_2,
  };
  enum FaceType
  {
    BC_INTERIOR = 0,
    BC_TANGENT = 1,
    BC_EXTRAPOLATE = 2,
    BC_INFLOW = 3,
    BC_NOSLIP = 4,
    BC_BLOCK_BORDER = 5,
    BC_TOTAL = 6,
  };

  ProblemType problem_type = PTYPE_0;
  int global_x = 4;
  int global_y = 4;
  int global_z = 4;
  int blocks_x = 2;
  int blocks_y = 2;
  int blocks_z = 2;
  size_t buffer_size = 100;

  int n_cells;
  int n_blocks;
  int n_faces;
  std::vector<int> xsplit, ysplit, zsplit;
  std::vector<int> cells_per_block, faces_per_block;

  typedef int INDEXTYPE;
  static const INDEXTYPE FIRST_INDEX = -2000000000;

  MiniAeroTest(int argc, const char *argv[])
  {
    for(int i = 1; i < argc; i++) {
      if(!strcmp(argv[i], "-type")) {
        problem_type = (ProblemType)atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-gx")) {
        global_x = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-gy")) {
        global_y = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-gz")) {
        global_z = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-bx")) {
        blocks_x = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-by")) {
        blocks_y = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-bz")) {
        blocks_z = atoi(argv[++i]);
        continue;
      }
      if(!strcmp(argv[i], "-g")) {
        int v = atoi(argv[++i]);
        global_x = global_y = global_z = v;
        continue;
      }
      if(!strcmp(argv[i], "-b")) {
        int v = atoi(argv[++i]);
        blocks_x = blocks_y = blocks_z = v;
        continue;
      }
      if(!strcmp(argv[i], "-p")) {
        int v = atoi(argv[++i]);
        blocks_x = v;
        blocks_y = 1;
        blocks_z = 1;
        continue;
      }
      if(!strcmp(argv[i], "-buffer") || !strcmp(argv[i], "-buf")) {
        buffer_size = atoi(argv[++i]);
        continue;
      }
    }

    assert(global_x >= blocks_x);
    assert(global_y >= blocks_y);
    assert(global_z >= blocks_z);
    if(global_x <= 0 || global_y <= 0 || global_z <= 0 || blocks_x <= 0 ||
       blocks_y <= 0 || blocks_z <= 0 || buffer_size > 100) {
      log_app.error() << "Invalid miniaero config";
      exit(1);
    }

    split_evenly<int>(global_x, blocks_x, xsplit);
    split_evenly<int>(global_y, blocks_y, ysplit);
    split_evenly<int>(global_z, blocks_z, zsplit);

    n_blocks = blocks_x * blocks_y * blocks_z;
    n_cells = 0;
    n_faces = 0;
    for(int bz = 0; bz < blocks_z; bz++)
      for(int by = 0; by < blocks_y; by++)
        for(int bx = 0; bx < blocks_x; bx++) {
          int nx = xsplit[bx + 1] - xsplit[bx];
          int ny = ysplit[by + 1] - ysplit[by];
          int nz = zsplit[bz + 1] - zsplit[bz];

          int c = nx * ny * nz;
          int f = (((nx + 1) * ny * nz) + (nx * (ny + 1) * nz) +
                   (nx * ny * (nz + 1)));
          cells_per_block.push_back(c);
          faces_per_block.push_back(f);

          n_cells += c;
          n_faces += f;
        }
    assert(n_cells == global_x * global_y * global_z);
    assert(n_faces == (((global_x + blocks_x) * global_y * global_z) +
                       (global_x * (global_y + blocks_y) * global_z) +
                       (global_x * global_y * (global_z + blocks_z))));
  }

  virtual void print_info(void)
  {
    printf("Realm dependent partitioning benchmark - miniaero: %d x %d x %d cells, %d x %d x %d blocks\n",
           global_x, global_y, global_z, blocks_x, blocks_y, blocks_z);
  }

  IndexSpace<1> is_cells, is_faces;
  std::vector<RegionInstance> ri_cells;
  std::vector<FieldDataDescriptor<IndexSpace<1>, int>> cell_blockid_field_data;
  std::vector<RegionInstance> ri_faces;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> face_left_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> face_right_field_data;
  std::vector<FieldDataDescriptor<IndexSpace<1>, int>> face_type_field_data;

  struct InitDataArgs {
    int index;
    RegionInstance ri_cells, ri_faces;
  };

  struct Outputs {
    std::vector<IndexSpace<1>> p_cells;
    std::vector<IndexSpace<1>> p_faces;
    std::vector<std::vector<IndexSpace<1>>> p_facetypes;
    std::vector<IndexSpace<1>> p_ghost;
  };

  Outputs gpu, cpu, garbage;

  virtual Event initialize_data(const std::vector<Memory> &memories,
                                const std::vector<Processor> &procs)
  {
    is_cells = Rect<1>(FIRST_INDEX, FIRST_INDEX + n_cells - 1);
    is_faces = Rect<1>(FIRST_INDEX, FIRST_INDEX + n_faces - 1);

    std::vector<IndexSpace<1>> ss_cells_w, ss_faces_w;
    is_cells.create_weighted_subspaces(n_blocks, 1, cells_per_block, ss_cells_w,
                                       Realm::ProfilingRequestSet()).wait();
    is_faces.create_weighted_subspaces(n_blocks, 1, faces_per_block, ss_faces_w,
                                       Realm::ProfilingRequestSet()).wait();

    std::vector<size_t> cell_fields(1, sizeof(int));
    std::vector<size_t> face_fields;
    face_fields.push_back(sizeof(Point<1>));
    face_fields.push_back(sizeof(Point<1>));
    face_fields.push_back(sizeof(int));

    ri_cells.resize(n_blocks);
    cell_blockid_field_data.resize(n_blocks);
    for(size_t i = 0; i < ss_cells_w.size(); i++) {
      RegionInstance::create_instance(ri_cells[i], memories[i % memories.size()],
                                      ss_cells_w[i], cell_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      cell_blockid_field_data[i].index_space = ss_cells_w[i];
      cell_blockid_field_data[i].inst = ri_cells[i];
      cell_blockid_field_data[i].field_offset = 0;
    }

    ri_faces.resize(n_blocks);
    face_left_field_data.resize(n_blocks);
    face_right_field_data.resize(n_blocks);
    face_type_field_data.resize(n_blocks);
    for(size_t i = 0; i < ss_faces_w.size(); i++) {
      RegionInstance::create_instance(ri_faces[i], memories[i % memories.size()],
                                      ss_faces_w[i], face_fields, 0,
                                      Realm::ProfilingRequestSet()).wait();
      face_left_field_data[i].index_space = ss_faces_w[i];
      face_left_field_data[i].inst = ri_faces[i];
      face_left_field_data[i].field_offset = 0;
      face_right_field_data[i].index_space = ss_faces_w[i];
      face_right_field_data[i].inst = ri_faces[i];
      face_right_field_data[i].field_offset = sizeof(Point<1>);
      face_type_field_data[i].index_space = ss_faces_w[i];
      face_type_field_data[i].inst = ri_faces[i];
      face_type_field_data[i].field_offset = 2 * sizeof(Point<1>);
    }

    std::set<Event> events;
    for(int i = 0; i < n_blocks; i++) {
      Processor proc = procs[i % procs.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_cells = ri_cells[i];
      args.ri_faces = ri_faces[i];
      events.insert(proc.spawn(INIT_MINIAERO_DATA_TASK, &args, sizeof(args)));
    }
    return Event::merge_events(events);
  }

  static void init_data_task_wrapper(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen, Processor p)
  {
    MiniAeroTest *me = (MiniAeroTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  Point<1> global_cell_pointer(int cx, int cy, int cz)
  {
    INDEXTYPE p = FIRST_INDEX;
    if((cx < 0) || (cx >= global_x) || (cy < 0) || (cy >= global_y) || (cz < 0) ||
       (cz >= global_z))
      return -1;

    int zi = find_split(zsplit, cz);
    p += global_x * global_y * zsplit[zi];
    cz -= zsplit[zi];
    int local_z = zsplit[zi + 1] - zsplit[zi];

    int yi = find_split(ysplit, cy);
    p += global_x * ysplit[yi] * local_z;
    cy -= ysplit[yi];
    int local_y = ysplit[yi + 1] - ysplit[yi];

    int xi = find_split(xsplit, cx);
    p += xsplit[xi] * local_y * local_z;
    cx -= xsplit[xi];
    int local_x = xsplit[xi + 1] - xsplit[xi];

    p += (cx + (cy * local_x) + (cz * local_x * local_y));
    return p;
  }

  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs &i_args = *(const InitDataArgs *)args;

    i_args.ri_cells.fetch_metadata(p).wait();
    i_args.ri_faces.fetch_metadata(p).wait();

    IndexSpace<1> is_cells = i_args.ri_cells.get_indexspace<1>();
    IndexSpace<1> is_faces = i_args.ri_faces.get_indexspace<1>();

    int bx = i_args.index % blocks_x;
    int by = (i_args.index / blocks_x) % blocks_y;
    int bz = i_args.index / blocks_x / blocks_y;

    size_t nx = xsplit[bx + 1] - xsplit[bx];
    size_t ny = ysplit[by + 1] - ysplit[by];
    size_t nz = zsplit[bz + 1] - zsplit[bz];
    assert(is_cells.bounds.volume() == nx * ny * nz);
    assert(is_faces.bounds.volume() ==
           (((nx + 1) * ny * nz) + (nx * (ny + 1) * nz) + (nx * ny * (nz + 1))));

    {
      AffineAccessor<int, 1> a_cell_blockid(i_args.ri_cells, 0);
      for(int cz = zsplit[bz]; cz < zsplit[bz + 1]; cz++)
        for(int cy = ysplit[by]; cy < ysplit[by + 1]; cy++)
          for(int cx = xsplit[bx]; cx < xsplit[bx + 1]; cx++) {
            Point<1> pz = global_cell_pointer(cx, cy, cz);
            assert(is_cells.bounds.contains(pz));
            a_cell_blockid.write(pz, i_args.index);
          }
    }

    AffineAccessor<Point<1>, 1> a_face_left(i_args.ri_faces, 0);
    AffineAccessor<Point<1>, 1> a_face_right(i_args.ri_faces, sizeof(Point<1>));
    AffineAccessor<int, 1> a_face_type(i_args.ri_faces, 2 * sizeof(Point<1>));

    Point<1> pf = is_faces.bounds.lo;

    for(int fx = xsplit[bx]; fx <= xsplit[bx + 1]; fx++) {
      int ftype = BC_INTERIOR;
      bool reversed = false;
      if(fx == xsplit[bx]) {
        reversed = true;
        if(fx == 0)
          ftype = (problem_type == PTYPE_0) ? BC_EXTRAPOLATE : BC_INFLOW;
        else
          ftype = BC_BLOCK_BORDER;
      } else if(fx == xsplit[bx + 1]) {
        if(fx == global_x)
          ftype = BC_EXTRAPOLATE;
        else
          ftype = BC_BLOCK_BORDER;
      }

      for(int cz = zsplit[bz]; cz < zsplit[bz + 1]; cz++)
        for(int cy = ysplit[by]; cy < ysplit[by + 1]; cy++) {
          a_face_left.write(pf, global_cell_pointer(fx - (reversed ? 0 : 1), cy, cz));
          a_face_right.write(pf, global_cell_pointer(fx - (reversed ? 1 : 0), cy, cz));
          a_face_type.write(pf, ftype);
          pf[0]++;
        }
    }

    for(int fy = ysplit[by]; fy <= ysplit[by + 1]; fy++) {
      int ftype = BC_INTERIOR;
      bool reversed = false;
      if(fy == ysplit[by]) {
        reversed = true;
        if(fy == 0)
          ftype = (problem_type == PTYPE_1) ? BC_NOSLIP : BC_TANGENT;
        else
          ftype = BC_BLOCK_BORDER;
      } else if(fy == ysplit[by + 1]) {
        if(fy == global_y)
          ftype = (problem_type == PTYPE_1) ? BC_EXTRAPOLATE : BC_TANGENT;
        else
          ftype = BC_BLOCK_BORDER;
      }

      for(int cz = zsplit[bz]; cz < zsplit[bz + 1]; cz++)
        for(int cx = xsplit[bx]; cx < xsplit[bx + 1]; cx++) {
          a_face_left.write(pf, global_cell_pointer(cx, fy - (reversed ? 0 : 1), cz));
          a_face_right.write(pf, global_cell_pointer(cx, fy - (reversed ? 1 : 0), cz));
          a_face_type.write(pf, ftype);
          pf[0]++;
        }
    }

    for(int fz = zsplit[bz]; fz <= zsplit[bz + 1]; fz++) {
      int ftype = BC_INTERIOR;
      bool reversed = false;
      if(fz == zsplit[bz]) {
        reversed = true;
        if(fz == 0)
          ftype = BC_TANGENT;
        else
          ftype = BC_BLOCK_BORDER;
      } else if(fz == zsplit[bz + 1]) {
        if(fz == global_z)
          ftype = BC_TANGENT;
        else
          ftype = BC_BLOCK_BORDER;
      }

      for(int cy = ysplit[by]; cy < ysplit[by + 1]; cy++)
        for(int cx = xsplit[bx]; cx < xsplit[bx + 1]; cx++) {
          a_face_left.write(pf, global_cell_pointer(cx, cy, fz - (reversed ? 0 : 1)));
          a_face_right.write(pf, global_cell_pointer(cx, cy, fz - (reversed ? 1 : 0)));
          a_face_type.write(pf, ftype);
          pf[0]++;
        }
    }

    assert(pf[0] == is_faces.bounds.hi[0] + 1);
  }

  template <typename FT>
  void alloc_by_field_scratch(IndexSpace<1> target,
                              std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
                              const std::vector<Memory> &gpu_memories)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    target.by_field_buffer_requirements(inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  template <typename FT>
  void alloc_by_subspace_scratch(
      IndexSpace<1> target, const std::vector<IndexSpace<1>> &subspaces,
      std::vector<FieldDataDescriptor<IndexSpace<1>, FT>> &fields,
      const std::vector<Memory> &gpu_memories,
      bool preimage)
  {
    assert(fields.size() == gpu_memories.size());
    std::vector<DeppartSubspace<1, int>> dp_subspaces(subspaces.size());
    std::vector<DeppartEstimateInput<1, int>> inputs(fields.size());
    std::vector<DeppartBufferRequirements> reqs(fields.size());
    for(size_t i = 0; i < subspaces.size(); i++)
      dp_subspaces[i] = make_deppart_subspace(subspaces[i]);
    for(size_t i = 0; i < fields.size(); i++) {
      inputs[i].location = fields[i].inst.get_location();
      inputs[i].space = fields[i].index_space;
    }
    if(preimage)
      target.by_preimage_buffer_requirements(dp_subspaces, inputs, reqs);
    else
      target.by_image_buffer_requirements(dp_subspaces, inputs, reqs);
    for(size_t i = 0; i < fields.size(); i++)
      alloc_piece(fields[i].scratch_buffer, scaled_requirement(reqs[i], buffer_size),
                  gpu_memories[i]).wait();
  }

  template <typename FT>
  void alloc_by_field_scratch(IndexSpace<1> target,
                              FieldDataDescriptor<IndexSpace<1>, FT> &field,
                              Memory gpu_memory)
  {
    std::vector<DeppartEstimateInput<1, int>> inputs(1);
    std::vector<DeppartBufferRequirements> reqs;
    inputs[0].location = field.inst.get_location();
    inputs[0].space = field.index_space;
    target.by_field_buffer_requirements(inputs, reqs);
    alloc_piece(field.scratch_buffer, scaled_requirement(reqs[0], buffer_size),
                gpu_memory).wait();
  }

  Event run_partitioning(
      std::vector<FieldDataDescriptor<IndexSpace<1>, int>> &cell_blockid,
      std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &face_left,
      std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> &face_right,
      std::vector<FieldDataDescriptor<IndexSpace<1>, int>> &face_type,
      Outputs &out, const char *breakdown_run = 0)
  {
    long long total_start = 0;
    long long stage_start = 0;
    long long cell_byfield_us = 0;
    long long face_preimage_us = 0;
    long long facetypes_byfield_us = 0;
    long long ghost_image_us = 0;
    if(breakdown_run)
      total_start = Clock::current_time_in_microseconds();

    std::vector<int> colors(n_blocks);
    for(int i = 0; i < n_blocks; i++)
      colors[i] = i;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e1 = is_cells.create_subspaces_by_field(cell_blockid, colors, out.p_cells,
                                                  Realm::ProfilingRequestSet());
    if(wait_on_events || breakdown_run)
      e1.wait();
    if(breakdown_run)
      cell_byfield_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e2 = is_faces.create_subspaces_by_preimage(face_left, out.p_cells, out.p_faces,
                                                     Realm::ProfilingRequestSet(), e1);
    if(wait_on_events || breakdown_run)
      e2.wait();
    if(breakdown_run)
      face_preimage_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    std::set<Event> evs;
    std::vector<int> ftcolors(BC_TOTAL);
    for(int i = 0; i < BC_TOTAL; i++)
      ftcolors[i] = i;
    out.p_facetypes.resize(n_blocks);
    std::vector<IndexSpace<1>> p_border_faces(n_blocks);

    for(int idx = 0; idx < n_blocks; idx++) {
      std::vector<FieldDataDescriptor<IndexSpace<1>, int>> ft_data(1, face_type[idx]);
      Event e = out.p_faces[idx].create_subspaces_by_field(ft_data, ftcolors,
                                                           out.p_facetypes[idx],
                                                           Realm::ProfilingRequestSet(),
                                                           e2);
      if(wait_on_events)
        e.wait();
      evs.insert(e);
      p_border_faces[idx] = out.p_facetypes[idx][BC_BLOCK_BORDER];
    }
    Event e3 = Event::merge_events(evs);
    if(breakdown_run)
      e3.wait();
    if(breakdown_run)
      facetypes_byfield_us = Clock::current_time_in_microseconds() - stage_start;

    if(n_blocks == 1) {
      out.p_ghost.resize(1);
      out.p_ghost[0] = IndexSpace<1>::make_empty();
      if(breakdown_run) {
        long long total_us = Clock::current_time_in_microseconds() - total_start;
        printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=cell_byfield,us=%lld\n",
               breakdown_run, n_blocks, cell_byfield_us);
        printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=face_preimage,us=%lld\n",
               breakdown_run, n_blocks, face_preimage_us);
        printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=facetypes_byfield,us=%lld\n",
               breakdown_run, n_blocks, facetypes_byfield_us);
        printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=ghost_image,us=0\n",
               breakdown_run, n_blocks);
        printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=total,us=%lld\n",
               breakdown_run, n_blocks, total_us);
      }
      return e3;
    }

    if(breakdown_run)
      stage_start = Clock::current_time_in_microseconds();
    Event e4 = create_subspaces_by_image_one_to_one(is_cells, face_right,
                                                    p_border_faces, out.p_ghost, e3);
    if(wait_on_events || breakdown_run)
      e4.wait();
    if(breakdown_run)
      ghost_image_us = Clock::current_time_in_microseconds() - stage_start;

    if(breakdown_run) {
      long long total_us = Clock::current_time_in_microseconds() - total_start;
      printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=cell_byfield,us=%lld\n",
             breakdown_run, n_blocks, cell_byfield_us);
      printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=face_preimage,us=%lld\n",
             breakdown_run, n_blocks, face_preimage_us);
      printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=facetypes_byfield,us=%lld\n",
             breakdown_run, n_blocks, facetypes_byfield_us);
      printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=ghost_image,us=%lld\n",
             breakdown_run, n_blocks, ghost_image_us);
      printf("BREAKDOWN,op=miniaero,run=%s,n_blocks=%d,stage=total,us=%lld\n",
             breakdown_run, n_blocks, total_us);
    }

    return e4;
  }

  virtual Event perform_partitioning(void)
  {
    std::vector<Memory> gpu_memories = get_gpu_memories();
    if(gpu_memories.empty())
      return Event::NO_EVENT;
    assert_piece_gpu_count("miniaero", n_blocks, gpu_memories);

    std::vector<size_t> cell_fields(1, sizeof(int));
    std::vector<size_t> face_fields;
    face_fields.push_back(sizeof(Point<1>));
    face_fields.push_back(sizeof(Point<1>));
    face_fields.push_back(sizeof(int));

    std::vector<FieldDataDescriptor<IndexSpace<1>, int>> cell_gpu(n_blocks);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> left_gpu(n_blocks);
    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> right_gpu(n_blocks);
    std::vector<FieldDataDescriptor<IndexSpace<1>, int>> type_gpu(n_blocks);
    for(int i = 0; i < n_blocks; i++) {
      copy_piece(cell_blockid_field_data[i], cell_gpu[i], cell_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(face_left_field_data[i], left_gpu[i], face_fields, 0,
                 gpu_memories[i]).wait();
      copy_piece(face_right_field_data[i], right_gpu[i], face_fields, 1,
                 gpu_memories[i]).wait();
      copy_piece(face_type_field_data[i], type_gpu[i], face_fields, 2,
                 gpu_memories[i]).wait();
    }

    alloc_by_field_scratch(is_cells, cell_gpu, gpu_memories);
    std::vector<int> colors(n_blocks);
    for(int i = 0; i < n_blocks; i++)
      colors[i] = i;
    Event e1 = is_cells.create_subspaces_by_field(cell_gpu, colors, garbage.p_cells,
                                                  Realm::ProfilingRequestSet());
    e1.wait();
    alloc_by_subspace_scratch(is_faces, garbage.p_cells, left_gpu, gpu_memories, true);
    Event e2 = is_faces.create_subspaces_by_preimage(left_gpu, garbage.p_cells,
                                                     garbage.p_faces,
                                                     Realm::ProfilingRequestSet(), e1);
    e2.wait();

    std::vector<int> ftcolors(BC_TOTAL);
    for(int i = 0; i < BC_TOTAL; i++)
      ftcolors[i] = i;
    garbage.p_facetypes.resize(n_blocks);
    std::vector<IndexSpace<1>> p_border_faces(n_blocks);
    for(int idx = 0; idx < n_blocks; idx++) {
      alloc_by_field_scratch(garbage.p_faces[idx], type_gpu[idx],
                             gpu_memories[idx]);
      std::vector<FieldDataDescriptor<IndexSpace<1>, int>> ft_data(1,
                                                                   type_gpu[idx]);
      Event e = garbage.p_faces[idx].create_subspaces_by_field(
          ft_data, ftcolors, garbage.p_facetypes[idx],
          Realm::ProfilingRequestSet(), e2);
      e.wait();
      p_border_faces[idx] = garbage.p_facetypes[idx][BC_BLOCK_BORDER];
    }
    if(n_blocks > 1)
      alloc_by_subspace_scratch_one_to_one(is_cells, p_border_faces, right_gpu,
                                           gpu_memories, buffer_size, false);
    destroy_outputs(garbage);

    Event warmup = run_partitioning(cell_gpu, left_gpu, right_gpu, type_gpu, garbage);
    warmup.wait();

    long long gpu_start = Clock::current_time_in_microseconds();
    Event gpu_call =
        run_partitioning(cell_gpu, left_gpu, right_gpu, type_gpu, gpu,
                         show_breakdown ? "gpu" : 0);
    gpu_call.wait();
    long long gpu_us = Clock::current_time_in_microseconds() - gpu_start;

    long long cpu_start = Clock::current_time_in_microseconds();
    Event cpu_call = run_partitioning(cell_blockid_field_data, face_left_field_data,
                                      face_right_field_data, face_type_field_data, cpu,
                                      show_breakdown ? "cpu" : 0);
    cpu_call.wait();
    long long cpu_us = Clock::current_time_in_microseconds() - cpu_start;

    printf("RESULT,op=miniaero,gx=%d,gy=%d,gz=%d,bx=%d,by=%d,bz=%d,type=%d,num_cells=%d,num_faces=%d,buffer_size=%zu,gpu_us=%lld,cpu_us=%lld\n",
           global_x, global_y, global_z, blocks_x, blocks_y, blocks_z,
           (int)problem_type, n_cells, n_faces, buffer_size, gpu_us, cpu_us);

    return Event::merge_events(gpu_call, cpu_call);
  }

  virtual int perform_dynamic_checks(void) { return 0; }

  void destroy_outputs(Outputs &out)
  {
    destroy_index_space_vector(out.p_cells);
    destroy_index_space_vector(out.p_faces);
    for(size_t i = 0; i < out.p_facetypes.size(); i++)
      destroy_index_space_vector(out.p_facetypes[i]);
    out.p_facetypes.clear();
    destroy_index_space_vector(out.p_ghost);
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;
    errors += compare_index_space_vectors(gpu.p_cells, cpu.p_cells, "p_cells");
    errors += compare_index_space_vectors(gpu.p_faces, cpu.p_faces, "p_faces");
    if(gpu.p_facetypes.size() != cpu.p_facetypes.size()) {
      log_app.error() << "Mismatch! p_facetypes outer sizes differ";
      errors++;
    }
    size_t count = std::min(gpu.p_facetypes.size(), cpu.p_facetypes.size());
    for(size_t i = 0; i < count; i++)
      errors += compare_index_space_vectors(gpu.p_facetypes[i], cpu.p_facetypes[i],
                                            "p_facetypes");
    errors += compare_index_space_vectors(gpu.p_ghost, cpu.p_ghost, "p_ghost");
    destroy_outputs(gpu);
    destroy_outputs(cpu);
    destroy_outputs(garbage);
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

    if(!strcmp(argv[i], "-breakdown")) {
      show_breakdown = true;
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

    if(!strcmp(argv[i], "circuit")) {
      op = "circuit";
      testcfg = new CircuitTest(argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "pennant")) {
      op = "pennant";
      testcfg = new PennantTest(argc - i, const_cast<const char **>(argv + i));
      break;
    }

    if(!strcmp(argv[i], "miniaero")) {
      op = "miniaero";
      testcfg = new MiniAeroTest(argc - i, const_cast<const char **>(argv + i));
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
  rt.register_task(INIT_CIRCUIT_DATA_TASK, CircuitTest::init_data_task_wrapper);
  rt.register_task(INIT_PENNANT_DATA_TASK, PennantTest::init_data_task_wrapper);
  rt.register_task(INIT_MINIAERO_DATA_TASK, MiniAeroTest::init_data_task_wrapper);

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
