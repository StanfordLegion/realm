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
#include "realm/event_impl.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include <stdio.h>

Realm::Logger log_app("app");

enum
{
  MAIN_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
};

struct event_task_args_t {
  realm_user_event_t wait_on;
  realm_user_event_t user_event;
};

void test_merge(realm_runtime_t runtime, int ignore_faults, int *poisoned)
{
  realm_user_event_t event_poisoned[10];
  realm_status_t status;
  for(int i = 0; i < 10; i++) {
    status = realm_user_event_create(runtime, &event_poisoned[i]);
    assert(status == REALM_SUCCESS);
    Realm::UserEvent(event_poisoned[i]).cancel();
  }
  realm_event_t merged_event;
  status = realm_event_merge(runtime, event_poisoned, 10, &merged_event, ignore_faults);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, merged_event, poisoned);
  assert(status == REALM_SUCCESS);
}

void test_trigger(realm_runtime_t runtime, int ignore_faults, bool use_wait,
                  int *poisoned)
{
  realm_user_event_t wait_on_event;
  realm_status_t status = realm_user_event_create(runtime, &wait_on_event);
  assert(status == REALM_SUCCESS);
  Realm::UserEvent(wait_on_event).cancel();
  realm_user_event_t user_event;
  status = realm_user_event_create(runtime, &user_event);
  assert(status == REALM_SUCCESS);
  status = realm_user_event_trigger(runtime, user_event, wait_on_event, ignore_faults);
  assert(status == REALM_SUCCESS);
  if(use_wait) {
    status = realm_event_wait(runtime, user_event, poisoned);
    assert(status == REALM_SUCCESS);
  } else {
    int has_triggered = 0;
    status = realm_event_has_triggered(runtime, user_event, &has_triggered, poisoned);
    assert(status == REALM_SUCCESS);
    assert(has_triggered == 1);
  }
}

void REALM_FNPTR main_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, realm_processor_t proc)
{
  log_app.info("main_task on proc %llx", proc);
  realm_user_event_t user_events[10];
  realm_event_t task_events[10];
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);

  // test merge poisoned events
  int poisoned = 0;
  test_merge(runtime, 0, &poisoned);
  assert(poisoned == 1);
  test_merge(runtime, 1, &poisoned);
  assert(poisoned == 0);

  // test trigger poisoned event
  test_trigger(runtime, 0, true, &poisoned);
  assert(poisoned == 1);
  test_trigger(runtime, 1, true, &poisoned);
  assert(poisoned == 0);

  // test trigger poisoned event with has_triggered
  test_trigger(runtime, 0, false, &poisoned);
  assert(poisoned == 1);
  test_trigger(runtime, 1, false, &poisoned);
  assert(poisoned == 0);
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_create(&runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_init(runtime, &argc, &argv);
  assert(status == REALM_SUCCESS);

  realm_event_t register_task_event;

  status = realm_processor_register_task_by_kind(runtime, LOC_PROC,
                                                 REALM_REGISTER_TASK_DEFAULT, MAIN_TASK,
                                                 main_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event, nullptr);
  assert(status == REALM_SUCCESS);

  realm_processor_query_t proc_query;
  status = realm_processor_query_create(runtime, &proc_query);
  assert(status == REALM_SUCCESS);
  status = realm_processor_query_restrict_to_kind(proc_query, LOC_PROC);
  assert(status == REALM_SUCCESS);
  realm_processor_t proc;
  realm_processor_query_first(proc_query, &proc);
  status = realm_processor_query_destroy(proc_query);
  assert(status == REALM_SUCCESS);
  assert(proc != REALM_NO_PROC);

  realm_event_t e;
  status = realm_runtime_collective_spawn(runtime, proc, MAIN_TASK, 0, 0, 0, 0, &e);
  assert(status == REALM_SUCCESS);

  status = realm_runtime_signal_shutdown(runtime, e, 0);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_wait_for_shutdown(runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_destroy(runtime);
  assert(status == REALM_SUCCESS);

  return 0;
}