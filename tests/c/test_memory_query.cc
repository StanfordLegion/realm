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
#include "realm/realm_c.h"
#include "realm/logging.h"
#include <stdio.h>
#include <vector>

Realm::Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
};

struct append_memory_args_t {
  std::vector<realm_memory_t> mems;
};

static realm_status_t REALM_FNPTR append_memory(realm_memory_t m, void *user_data)
{
  append_memory_args_t *args = reinterpret_cast<append_memory_args_t *>(user_data);
  args->mems.push_back(m);
  return REALM_SUCCESS;
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  printf("top_level_task on proc " IDFMT, proc);
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));

  // Iterate over all CPU memories, and print their attributes
  realm_memory_query_t cpu_mem_query;
  CHECK_REALM(realm_memory_query_create(runtime, &cpu_mem_query));
  // restrict to SYSTEM_MEM
  CHECK_REALM(realm_memory_query_restrict_to_kind(cpu_mem_query, SYSTEM_MEM));
  append_memory_args_t cpu_mem_query_args;
  // query all system memories
  CHECK_REALM(realm_memory_query_iter(cpu_mem_query, append_memory, &cpu_mem_query_args,
                                      SIZE_MAX));
  // print the attributes of the memories
  for(realm_memory_t mem : cpu_mem_query_args.mems) {
    realm_memory_attr_t attrs[3] = {REALM_MEMORY_ATTR_KIND,
                                    REALM_MEMORY_ATTR_ADDRESS_SPACE,
                                    REALM_MEMORY_ATTR_CAPACITY};
    uint64_t values[3];
    CHECK_REALM(realm_memory_get_attributes(runtime, mem, attrs, values, 3));
    log_app.info() << "Memory " << mem << " kind: " << values[0]
                   << ", address_space: " << values[1] << ", size: " << values[2];
  }
  CHECK_REALM(realm_memory_query_destroy(cpu_mem_query));

  // Iterate over all GPU FB memories, and print their attributes
  realm_memory_query_t gpu_mem_query;
  CHECK_REALM(realm_memory_query_create(runtime, &gpu_mem_query));
  CHECK_REALM(realm_memory_query_restrict_to_kind(gpu_mem_query, GPU_FB_MEM));
  // query all GPU FB memories
  append_memory_args_t gpu_mem_query_args;
  CHECK_REALM(realm_memory_query_iter(gpu_mem_query, append_memory, &gpu_mem_query_args,
                                      SIZE_MAX));
  // print the attributes of the memories
  for(realm_memory_t mem : gpu_mem_query_args.mems) {
    realm_memory_attr_t attrs[3] = {REALM_MEMORY_ATTR_KIND,
                                    REALM_MEMORY_ATTR_ADDRESS_SPACE,
                                    REALM_MEMORY_ATTR_CAPACITY};
    uint64_t values[3];
    CHECK_REALM(realm_memory_get_attributes(runtime, mem, attrs, values, 3));
    log_app.info() << "Memory " << mem << " kind: " << values[0]
                   << ", address_space: " << values[1] << ", size: " << values[2];
  }
  CHECK_REALM(realm_memory_query_destroy(gpu_mem_query));
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_create(&runtime));
  CHECK_REALM(realm_runtime_init(runtime, &argc, &argv));

  realm_event_t register_task_event;

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, TOP_LEVEL_TASK, top_level_task, 0,
      0, &register_task_event));
  CHECK_REALM(
      realm_event_wait(runtime, register_task_event, REALM_WAIT_INFINITE, nullptr));

  realm_processor_query_t proc_query;
  CHECK_REALM(realm_processor_query_create(runtime, &proc_query));
  CHECK_REALM(realm_processor_query_restrict_to_kind(proc_query, LOC_PROC));
  realm_processor_t proc;
  realm_processor_query_first(proc_query, &proc);
  CHECK_REALM(realm_processor_query_destroy(proc_query));
  assert(proc != REALM_NO_PROC);

  realm_event_t e;
  CHECK_REALM(
      realm_runtime_collective_spawn(runtime, proc, TOP_LEVEL_TASK, 0, 0, 0, 0, &e));

  CHECK_REALM(realm_runtime_signal_shutdown(runtime, e, 0));
  CHECK_REALM(realm_runtime_wait_for_shutdown(runtime));
  CHECK_REALM(realm_runtime_destroy(runtime));

  return 0;
}
