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

Realm::Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  EVENT_TASK,
};

struct event_task_args_t {
  realm_user_event_t wait_on;
  realm_user_event_t user_event;
};

void REALM_FNPTR event_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  log_app.info("event_task on proc %llx\n", proc);
  event_task_args_t *task_args = (event_task_args_t *)args;
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));
  CHECK_REALM(
      realm_user_event_trigger(runtime, task_args->user_event, task_args->wait_on, 0));
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  log_app.info("top_level_task on proc %llx\n", proc);
  realm_user_event_t user_events[10];
  realm_event_t task_events[10];
  realm_runtime_t runtime;
  int has_triggered = 0;

  CHECK_REALM(realm_runtime_get_runtime(&runtime));
  realm_user_event_t wait_on;
  CHECK_REALM(realm_user_event_create(runtime, &wait_on));
  for(int i = 0; i < 10; i++) {
    CHECK_REALM(realm_user_event_create(runtime, &user_events[i]));
    event_task_args_t args;
    args.user_event = user_events[i];
    args.wait_on = wait_on;
    CHECK_REALM(realm_processor_spawn(runtime, proc, EVENT_TASK, &args, sizeof(args),
                                      nullptr, 0, 0, &task_events[i]));
  }

  realm_event_t merged_event;
  CHECK_REALM(realm_event_merge(runtime, task_events, 10, &merged_event, 0));
  CHECK_REALM(realm_event_wait(runtime, merged_event, REALM_WAIT_INFINITE, nullptr));

  // set a timer on the wait_on event before triggering it
  CHECK_REALM(realm_event_wait(runtime, merged_event, 10000, nullptr));

  CHECK_REALM(realm_event_has_triggered(runtime, wait_on, &has_triggered, nullptr));
  assert(has_triggered == 0);

  // trigger the wait_on event
  CHECK_REALM(realm_user_event_trigger(runtime, wait_on, REALM_NO_EVENT, 0));

  CHECK_REALM(realm_event_merge(runtime, user_events, 10, &merged_event, 0));
  CHECK_REALM(realm_event_wait(runtime, merged_event, REALM_WAIT_INFINITE, nullptr));

  // test has_triggered
  for(int i = 0; i < 10; i++) {
    has_triggered = 0;
    CHECK_REALM(
        realm_event_has_triggered(runtime, user_events[i], &has_triggered, nullptr));
    assert(has_triggered == 1);
  }
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

  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, EVENT_TASK, event_task, 0, 0,
      &register_task_event));
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
