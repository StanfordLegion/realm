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
  HELLO_TASK,
};

struct append_process_args_t {
  std::vector<realm_processor_t> procs;
};

static realm_status_t REALM_FNPTR append_process(realm_processor_t p, void *user_data)
{
  append_process_args_t *args = reinterpret_cast<append_process_args_t *>(user_data);
  args->procs.push_back(p);
  return REALM_SUCCESS;
}

void REALM_FNPTR hello_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  log_app.info("hello_task on proc " IDFMT, proc);
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  log_app.info("top_level_task on proc " IDFMT, proc);
  realm_event_t event;
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));

  // get all cpu procs
  realm_processor_query_t proc_query;
  CHECK_REALM(realm_processor_query_create(runtime, &proc_query));
  // restrict to LOC_PROC
  CHECK_REALM(realm_processor_query_restrict_to_kind(proc_query, LOC_PROC));
  // query all LOC_PROC processors
  append_process_args_t cpu_proc_query_args;
  CHECK_REALM(realm_processor_query_iter(proc_query, append_process, &cpu_proc_query_args,
                                         SIZE_MAX));
  CHECK_REALM(realm_processor_query_destroy(proc_query));
  // spawn tasks on all cpu procs
  size_t num_cpus = cpu_proc_query_args.procs.size();
  std::vector<realm_event_t> events(num_cpus, REALM_NO_EVENT);
  for(size_t i = 0; i < num_cpus; i++) {
    CHECK_REALM(realm_processor_spawn(runtime, cpu_proc_query_args.procs[i], HELLO_TASK,
                                      0, 0, NULL, 0, 0, &events[i]));
  }
  CHECK_REALM(realm_event_merge(runtime, events.data(), events.size(), &event, 0));
  CHECK_REALM(realm_event_wait(runtime, event, 0, nullptr, nullptr));

#ifdef REALM_USE_CUDA
  CHECK_REALM(realm_processor_query_create(runtime, &proc_query));
  CHECK_REALM(realm_processor_query_restrict_to_kind(proc_query, TOC_PROC));

  realm_processor_t gpu_proc;
  realm_processor_query_first(proc_query, &gpu_proc);
  CHECK_REALM(realm_processor_query_destroy(proc_query));
  assert(gpu_proc != REALM_NO_PROC);

  CHECK_REALM(
      realm_processor_spawn(runtime, gpu_proc, HELLO_TASK, 0, 0, NULL, 0, 0, &event));
  CHECK_REALM(realm_event_wait(runtime, event, 0, nullptr, nullptr));
#endif
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_create(&runtime));
  CHECK_REALM(realm_runtime_init(runtime, &argc, &argv));

  realm_runtime_attr_t attrs[1] = {REALM_RUNTIME_ATTR_ADDRESS_SPACE};
  uint64_t values[1];
  CHECK_REALM(realm_runtime_get_attributes(runtime, attrs, values, 1));
  log_app.info("address space: %lu", values[0]);

  realm_event_t register_task_event;

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, TOP_LEVEL_TASK, top_level_task, 0,
      0, &register_task_event));
  CHECK_REALM(realm_event_wait(runtime, register_task_event, 0, nullptr, nullptr));

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, HELLO_TASK, hello_task, 0, 0,
      &register_task_event));
  CHECK_REALM(realm_event_wait(runtime, register_task_event, 0, nullptr, nullptr));

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, TOC_PROC, REALM_REGISTER_TASK_DEFAULT, HELLO_TASK, hello_task, 0, 0,
      &register_task_event));
  CHECK_REALM(realm_event_wait(runtime, register_task_event, 0, nullptr, nullptr));

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