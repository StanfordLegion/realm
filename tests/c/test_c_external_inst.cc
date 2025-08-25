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

#include "common.h"

#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_access.h"
#include <cuda_runtime.h>
#endif

// MSVC doesn't support the C++17 aligned_alloc
#if defined(_WIN32)
#define ALIGNED_ALLOC(s, a) _aligned_malloc(s, a)
#define ALIGNED_FREE(p) _aligned_free(p)
#else
#define ALIGNED_ALLOC(s, a) aligned_alloc(s, a)
#define ALIGNED_FREE(p) free(p)
#endif

Realm::Logger log_app("app");

enum
{
  MAIN_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};

enum
{
  FID_BASE = 44,
};

namespace TestConfig {
  int size = 10; // extent in each dimension
  int dims = 0;  // 0: 1-3D1: 1D, 2: 2D, 3: 3D
};               // namespace TestConfig

struct append_memory_args_t {
  std::vector<realm_memory_t> mems;
};

static realm_status_t REALM_FNPTR append_memory(realm_memory_t m, void *user_data)
{
  append_memory_args_t *args = reinterpret_cast<append_memory_args_t *>(user_data);
  args->mems.push_back(m);
  return REALM_SUCCESS;
}

template <int N, typename T>
void fill(Realm::RegionInstance inst, Realm::IndexSpace<N, T> idx_space, int fill_value)
{
  std::vector<Realm::CopySrcDstField> dsts(1);
  dsts[0].set_field(inst, FID_BASE, sizeof(int));
  idx_space.fill(dsts, Realm::ProfilingRequestSet(), &fill_value, sizeof(fill_value))
      .wait();
}

template <int N, typename T>
bool create_instance_and_copy_and_verify(
    realm_runtime_t runtime, const realm_external_resource_t *external_mem_res,
    int *external_mem, realm_region_instance_t dst_inst, realm_field_id_t *field_ids,
    size_t *field_sizes, int num_fields, Realm::IndexSpace<N, T> idx_space,
    realm_coord_type_t coord_type, realm_processor_t proc, T lower_bound[N],
    T upper_bound[N])
{
  // create external instance from external memory resource
  realm_region_instance_t src_inst;
  realm_memory_t src_mem;
  CHECK_REALM(
      realm_external_resource_suggested_memory(runtime, external_mem_res, &src_mem));
  realm_region_instance_create_params_t src_instance_params = {
      .memory = src_mem,
      .lower_bound = lower_bound,
      .upper_bound = upper_bound,
      .num_dims = N,
      .coord_type = coord_type,
      .sparsity_map = nullptr,
      .field_ids = field_ids,
      .field_sizes = field_sizes,
      .num_fields = 1,
      .block_size = 0,
      .external_resource = external_mem_res,
  };
  realm_event_t event = REALM_NO_EVENT;
  CHECK_REALM(realm_region_instance_create(runtime, &src_instance_params, nullptr,
                                           REALM_NO_EVENT, &src_inst, &event));
  CHECK_REALM(realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr));

  Realm::RegionInstance src_inst_cxx = Realm::RegionInstance(src_inst);
  const Realm::InstanceLayoutGeneric *src_layout = src_inst_cxx.get_layout();
  log_app.info("region instance created, src_inst: " IDFMT " (mem: " IDFMT
               ", size: %lu, base: %p, %p)",
               src_inst_cxx.id, src_inst_cxx.get_location().id, src_layout->bytes_used,
               external_mem, src_inst_cxx.pointer_untyped(0, sizeof(int)));

  fill<N, T>(Realm::RegionInstance(src_inst), idx_space, 7);
  fill<N, T>(Realm::RegionInstance(dst_inst), idx_space, 8);

  // copy src_inst to dst_inst
  realm_copy_src_dst_field_t srcs[1] = {{src_inst, FID_BASE, sizeof(int)}};
  realm_copy_src_dst_field_t dsts[1] = {{dst_inst, FID_BASE, sizeof(int)}};
  realm_region_instance_copy_params_t copy_params = {
      .srcs = srcs,
      .dsts = dsts,
      .num_fields = 1,
      .lower_bound = lower_bound,
      .upper_bound = upper_bound,
      .num_dims = N,
      .coord_type = coord_type,
      .sparsity_map = nullptr,
  };

  CHECK_REALM(realm_region_instance_copy(runtime, &copy_params, nullptr, REALM_NO_EVENT,
                                         0, &event));
  CHECK_REALM(realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr));

  Realm::RegionInstance dst_inst_cxx = Realm::RegionInstance(dst_inst);
  dst_inst_cxx.fetch_metadata(Realm::Processor(proc)).wait();
  bool success = true;
  Realm::GenericAccessor<int, N, T> acc(dst_inst_cxx, FID_BASE);
  for(Realm::IndexSpaceIterator<N, T> it(idx_space); it.valid; it.step()) {
    for(Realm::PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      if(acc[it2.p] != 7) {
        log_app.error() << "point=" << it2.p[0] << ", expected=7, actual=" << acc[it2.p];
        success = false;
      }
    }
  }

  // test generate external resource info
  realm_index_space_t ispace = {
      .lower_bound = lower_bound,
      .upper_bound = upper_bound,
      .num_dims = N,
      .coord_type = coord_type,
  };
  realm_external_resource_t external_resource;
  CHECK_REALM(realm_region_instance_generate_external_resource_info(
      runtime, src_inst, &ispace, field_ids, 1, 0, &external_resource));
  assert(external_resource.type == external_mem_res->type);

  CHECK_REALM(realm_region_instance_destroy(runtime, src_inst, REALM_NO_EVENT));
  return success;
}

template <int N, typename T>
static void test_copy(realm_runtime_t runtime, realm_memory_t dst_mem,
                      Realm::Rect<N, T> rect, realm_coord_type_t coord_type,
                      realm_processor_t proc)
{
  realm_event_t event = REALM_NO_EVENT;
  T lower_bound[N];
  T upper_bound[N];
  for(int i = 0; i < N; i++) {
    lower_bound[i] = 0;
    upper_bound[i] = rect.hi[i];
  }
  realm_field_id_t field_ids[1] = {FID_BASE};
  size_t field_sizes[1] = {sizeof(int)};

  // create dst instance from cpu memory
  realm_region_instance_create_params_t dst_instance_params = {
      .memory = dst_mem,
      .lower_bound = lower_bound,
      .upper_bound = upper_bound,
      .num_dims = N,
      .coord_type = coord_type,
      .sparsity_map = nullptr,
      .field_ids = field_ids,
      .field_sizes = field_sizes,
      .num_fields = 1,
      .block_size = 0,
      .external_resource = nullptr,
  };
  realm_region_instance_t dst_inst;
  CHECK_REALM(realm_region_instance_create(runtime, &dst_instance_params, nullptr,
                                           REALM_NO_EVENT, &dst_inst, &event));
  CHECK_REALM(realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr));

  Realm::RegionInstance dst_inst_cxx = Realm::RegionInstance(dst_inst);
  const Realm::InstanceLayoutGeneric *dst_layout = dst_inst_cxx.get_layout();
  log_app.info("region instance created, dst_inst: " IDFMT " (mem: " IDFMT
               ", size: %lu, base: %p)",
               dst_inst_cxx.id, dst_inst_cxx.get_location().id, dst_layout->bytes_used,
               dst_inst_cxx.pointer_untyped(0, sizeof(int)));

  int *external_mem = nullptr;
  size_t external_mem_size = rect.volume() * sizeof(int);

  // CUDA memory
#ifdef REALM_USE_CUDA
  {
    cudaSetDevice(0);
    cudaMalloc(&external_mem, external_mem_size);
    if(external_mem == nullptr) {
      log_app.error("cudaMalloc failed");
      return;
    } else {
      realm_external_resource_t external_resource;
      external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_CUDA_MEMORY;
      external_resource.resource.cuda_memory.cuda_device_id = 0;
      external_resource.resource.cuda_memory.base = external_mem;
      external_resource.resource.cuda_memory.size = external_mem_size;
      external_resource.resource.cuda_memory.read_only = 0;

      Realm::IndexSpace<N, T> idx_space(rect);
      bool success = create_instance_and_copy_and_verify<N, T>(
          runtime, &external_resource, external_mem, dst_inst, field_ids, field_sizes, 1,
          idx_space, coord_type, proc, lower_bound, upper_bound);
      assert(success);
      cudaFree(external_mem);
      external_mem = nullptr;
    }
  }
#endif

  // system memory
  {
    const size_t MIN_ALIGN = 32;
    external_mem = static_cast<int *>(ALIGNED_ALLOC(MIN_ALIGN, external_mem_size));
    if(external_mem == nullptr) {
      log_app.error("aligned_alloc failed");
      return;
    }

    realm_external_resource_t external_resource;
    external_resource.type = REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY;
    external_resource.resource.system_memory.base = external_mem;
    external_resource.resource.system_memory.size = external_mem_size;
    external_resource.resource.system_memory.read_only = 0;

    Realm::IndexSpace<N, T> idx_space(rect);
    bool success = create_instance_and_copy_and_verify<N, T>(
        runtime, &external_resource, external_mem, dst_inst, field_ids, field_sizes, 1,
        idx_space, coord_type, proc, lower_bound, upper_bound);
    assert(success);
    ALIGNED_FREE(external_mem);
    external_mem = nullptr;
  }

  CHECK_REALM(realm_region_instance_destroy(runtime, dst_inst, REALM_NO_EVENT));
}

void REALM_FNPTR main_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, realm_processor_t proc)
{
  log_app.info("main_task on proc " IDFMT, proc);
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));

  // Iterate over all CPU memories, and print their attributes
  realm_memory_query_t cpu_mem_query;
  realm_memory_query_create(runtime, &cpu_mem_query);
  // restrict to SYSTEM_MEM
  CHECK_REALM(realm_memory_query_restrict_to_kind(cpu_mem_query, SYSTEM_MEM));
  CHECK_REALM(realm_memory_query_restrict_by_capacity(cpu_mem_query, 1024));
  append_memory_args_t cpu_mem_query_args;
  // query all system memories
  CHECK_REALM(realm_memory_query_iter(cpu_mem_query, append_memory, &cpu_mem_query_args,
                                      SIZE_MAX));
  // destroy cpu_mem_query
  CHECK_REALM(realm_memory_query_destroy(cpu_mem_query));

  long long upper_size = 80;
  int dst_mem_idx = 0;

  // 1D
  if(TestConfig::dims == 1 || TestConfig::dims == 0) {
    log_app.info("external test_copy 1D long long");
    Realm::Rect<1, long long> rect1d_ll(0, upper_size - 1);
    test_copy<1, long long>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect1d_ll,
                            REALM_COORD_TYPE_LONG_LONG, proc);

    log_app.info("external test_copy 1D int");
    Realm::Rect<1, int> rect1d_int(0, upper_size - 1);
    test_copy<1, int>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect1d_int,
                      REALM_COORD_TYPE_INT, proc);
  }

  // 2D
  if(TestConfig::dims == 2 || TestConfig::dims == 0) {
    log_app.info("external test_copy 2D long long");
    Realm::Rect<2, long long> rect2d_ll(
        Realm::Point<2, long long>(0, 0),
        Realm::Point<2, long long>(upper_size - 1, upper_size - 1));
    test_copy<2, long long>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect2d_ll,
                            REALM_COORD_TYPE_LONG_LONG, proc);

    log_app.info("external test_copy 2D int");
    Realm::Rect<2, int> rect2d_int(Realm::Point<2, int>(0, 0),
                                   Realm::Point<2, int>(upper_size - 1, upper_size - 1));
    test_copy<2, int>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect2d_int,
                      REALM_COORD_TYPE_INT, proc);
  }

  // 3D
  if(TestConfig::dims == 3 || TestConfig::dims == 0) {
    log_app.info("external test_copy 3D long long");
    Realm::Rect<3, long long> rect3d_ll(
        Realm::Point<3, long long>(0, 0, 0),
        Realm::Point<3, long long>(upper_size - 1, upper_size - 1, upper_size - 1));
    test_copy<3, long long>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect3d_ll,
                            REALM_COORD_TYPE_LONG_LONG, proc);

    log_app.info("external test_copy 3D int");
    Realm::Rect<3, int> rect3d_int(
        Realm::Point<3, int>(0, 0, 0),
        Realm::Point<3, int>(upper_size - 1, upper_size - 1, upper_size - 1));
    test_copy<3, int>(runtime, cpu_mem_query_args.mems[dst_mem_idx], rect3d_int,
                      REALM_COORD_TYPE_INT, proc);
  }
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_create(&runtime));
  CHECK_REALM(realm_runtime_init(runtime, &argc, &argv));

  Realm::CommandLineParser cp;
  cp.add_option_int("-size", TestConfig::size).add_option_int("-dims", TestConfig::dims);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  realm_event_t register_task_event;
#ifdef REALM_USE_CUDA
  enum realm_processor_kind_t target_proc_kind = TOC_PROC;
#else
  enum realm_processor_kind_t target_proc_kind = LOC_PROC;
#endif

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, target_proc_kind, REALM_REGISTER_TASK_DEFAULT, MAIN_TASK, main_task, 0, 0,
      &register_task_event));
  CHECK_REALM(
      realm_event_wait(runtime, register_task_event, REALM_WAIT_INFINITE, nullptr));

  realm_processor_query_t proc_query;
  CHECK_REALM(realm_processor_query_create(runtime, &proc_query));
  CHECK_REALM(realm_processor_query_restrict_to_kind(proc_query, target_proc_kind));
  realm_processor_t proc;
  realm_processor_query_first(proc_query, &proc);
  CHECK_REALM(realm_processor_query_destroy(proc_query));
  assert(proc != REALM_NO_PROC);

  realm_event_t e;
  CHECK_REALM(realm_runtime_collective_spawn(runtime, proc, MAIN_TASK, nullptr /*args*/,
                                             0 /*arglen*/, REALM_NO_EVENT /* wait_on */,
                                             0 /*priority*/, &e));

  CHECK_REALM(realm_runtime_signal_shutdown(runtime, e, 0));
  CHECK_REALM(realm_runtime_wait_for_shutdown(runtime));
  CHECK_REALM(realm_runtime_destroy(runtime));

  return 0;
}
