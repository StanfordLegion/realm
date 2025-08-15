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
#include <assert.h>
#include <chrono>
#include <thread>

Realm::Logger log_app("app");

static inline void sleepForNanoseconds(long long delayNs)
{
  if(delayNs <= 0) {
    return;
  }
  std::this_thread::sleep_for(std::chrono::nanoseconds(delayNs));
}

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  DELAYED_TASK,
  CANCEL_TASK,
};

struct delayed_task_args_t {
  realm_user_event_t user_event;
  long long delay_ns;
};

struct cancel_task_args_t {
  realm_user_event_t user_event;
  realm_event_t task_event;
};

void REALM_FNPTR delayed_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, realm_processor_t proc)
{
  log_app.info("delayed_task on proc " IDFMT "\n", proc);
  delayed_task_args_t *task_args = (delayed_task_args_t *)args;

  // Simulate some work that takes time
  if(task_args->delay_ns > 0) {
    // Sleep for the specified delay
    sleepForNanoseconds(task_args->delay_ns);
  }

  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));
  // Avoid double-trigger if the event has already been cancelled/triggered
  int has_triggered = 0;
  int poisoned = 0;
  CHECK_REALM(realm_event_has_triggered(runtime, task_args->user_event, &has_triggered,
                                        &poisoned));
  if(!has_triggered) {
    CHECK_REALM(
        realm_user_event_trigger(runtime, task_args->user_event, REALM_NO_EVENT, 0));
  }
}

void REALM_FNPTR cancel_task(const void *args, size_t arglen, const void *userdata,
                             size_t userlen, realm_processor_t proc)
{
  log_app.info("cancel_task on proc " IDFMT "\n", proc);
  cancel_task_args_t *task_args = (cancel_task_args_t *)args;

  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));

  // Wait a bit then cancel the operation
  sleepForNanoseconds(1000000); // 1ms

  // Cancel the user event representing the operation; do not trigger it again here
  const char *reason = "Task cancelled by cancel_task";
  CHECK_REALM(realm_event_cancel_operation(runtime, task_args->user_event, reason,
                                           strlen(reason)));
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  log_app.info("top_level_task on proc " IDFMT "\n", proc);
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_get_runtime(&runtime));

  // Test 1: Basic event wait (no timeout)
  log_app.info("Test 1: Basic event wait (no timeout)\n");
  realm_user_event_t basic_event;
  CHECK_REALM(realm_user_event_create(runtime, &basic_event));

  // Spawn a task that will trigger the event
  delayed_task_args_t delay_args;
  delay_args.user_event = basic_event;
  delay_args.delay_ns = 1000000; // 1ms delay

  realm_event_t delay_task_event;
  CHECK_REALM(realm_processor_spawn(runtime, proc, DELAYED_TASK, &delay_args,
                                    sizeof(delay_args), nullptr, 0, 0,
                                    &delay_task_event));

  // Wait for the event to be triggered
  int has_triggered = 0;
  int poisoned = 0;
  CHECK_REALM(realm_event_wait(runtime, basic_event, 0, nullptr, nullptr));

  // Verify the event was triggered
  CHECK_REALM(realm_event_has_triggered(runtime, basic_event, &has_triggered, nullptr));
  assert(has_triggered == 1);
  log_app.info("Basic event wait completed successfully\n");

  // Test 2: Timed event wait (should timeout)
  log_app.info("Test 2: Timed event wait (should timeout)\n");
  realm_user_event_t timeout_event;
  CHECK_REALM(realm_user_event_create(runtime, &timeout_event));

  // Try to wait with a very short timeout
  has_triggered = 0;
  poisoned = 0;
  CHECK_REALM(realm_event_wait(runtime, timeout_event, 1000, &has_triggered,
                               &poisoned)); // 1 microsecond

  // Should have timed out
  assert(has_triggered == 0);
  assert(poisoned == 0);
  log_app.info("Timed event wait timed out as expected\n");

  // Test 3: Event wait with NO_EVENT (should always succeed)
  log_app.info("Test 3: Event wait with NO_EVENT\n");
  has_triggered = 0;
  poisoned = 0;
  CHECK_REALM(realm_event_wait(runtime, REALM_NO_EVENT, 0, &has_triggered, &poisoned));

  // NO_EVENT should always be triggered
  assert(has_triggered == 1);
  assert(poisoned == 0);
  log_app.info("NO_EVENT wait completed successfully\n");

  // Test 4: Event cancellation
  log_app.info("Test 4: Event cancellation\n");
  realm_user_event_t cancel_event;
  CHECK_REALM(realm_user_event_create(runtime, &cancel_event));

  // Spawn a task that will be cancelled
  delayed_task_args_t cancel_delay_args;
  cancel_delay_args.user_event = cancel_event;
  cancel_delay_args.delay_ns = 100000000; // 100ms delay (long enough to cancel)

  realm_event_t cancel_task_event;
  CHECK_REALM(realm_processor_spawn(runtime, proc, DELAYED_TASK, &cancel_delay_args,
                                    sizeof(cancel_delay_args), nullptr, 0, 0,
                                    &cancel_task_event));

  // Verify that before cancellation the user event has not triggered (timed wait should
  // timeout)
  has_triggered = 0;
  poisoned = 0;
  CHECK_REALM(
      realm_event_wait(runtime, cancel_event, 1000 /*1us*/, &has_triggered, &poisoned));
  assert(has_triggered == 0);
  assert(poisoned == 0);

  // Spawn another task that will cancel the first one
  cancel_task_args_t cancel_args;
  cancel_args.user_event = cancel_event;
  cancel_args.task_event = cancel_task_event; // unused now

  realm_event_t cancel_trigger_event;
  CHECK_REALM(realm_processor_spawn(runtime, proc, CANCEL_TASK, &cancel_args,
                                    sizeof(cancel_args), nullptr, 0, 0,
                                    &cancel_trigger_event));
  // Wait for the cancellation to complete (the cancel task finishes independently)
  CHECK_REALM(realm_event_wait(runtime, cancel_trigger_event, 0, nullptr, nullptr));

  log_app.info("Event cancellation completed successfully\n");

  // Test 5: Has-triggered on a cancelled event
  log_app.info("Test 5: Has-triggered on cancelled event\n");
  has_triggered = 0;
  poisoned = 0;
  CHECK_REALM(
      realm_event_has_triggered(runtime, cancel_event, &has_triggered, &poisoned));
  assert(has_triggered == 1);
  log_app.info("Has-triggered on cancelled event completed\n");

  // Test 6: Multiple event cancellation
  log_app.info("Test 6: Multiple event cancellation\n");
  realm_user_event_t multi_events[3];
  realm_event_t multi_task_events[3];

  for(int i = 0; i < 3; i++) {
    CHECK_REALM(realm_user_event_create(runtime, &multi_events[i]));

    delayed_task_args_t multi_args;
    multi_args.user_event = multi_events[i];
    multi_args.delay_ns = 50000000; // 50ms delay

    CHECK_REALM(realm_processor_spawn(runtime, proc, DELAYED_TASK, &multi_args,
                                      sizeof(multi_args), nullptr, 0, 0,
                                      &multi_task_events[i]));
  }

  // Cancel all three user events
  for(int i = 0; i < 3; i++) {
    const char *reason = "Multiple event cancellation test";
    CHECK_REALM(
        realm_event_cancel_operation(runtime, multi_events[i], reason, strlen(reason)));
  }

  // Wait for all cancelled user events
  for(int i = 0; i < 3; i++) {
    has_triggered = 0;
    poisoned = 0;
    CHECK_REALM(realm_event_wait(runtime, multi_events[i], 0, &has_triggered, &poisoned));
    assert(has_triggered == 1);
    assert(poisoned == 1);
  }
  log_app.info("Multiple event cancellation completed successfully\n");

  log_app.info("All event wait and cancellation tests passed!\n");
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  CHECK_REALM(realm_runtime_create(&runtime));
  CHECK_REALM(realm_runtime_init(runtime, &argc, &argv));

  // Register the delayed task
  realm_event_t register_task_event;
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, DELAYED_TASK, delayed_task, 0, 0,
      &register_task_event));
  CHECK_REALM(realm_event_wait(runtime, register_task_event, 0, nullptr, nullptr));

  // Register the cancel task
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, CANCEL_TASK, cancel_task, 0, 0,
      &register_task_event));
  CHECK_REALM(realm_event_wait(runtime, register_task_event, 0, nullptr, nullptr));

  // Register the top level task
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, TOP_LEVEL_TASK, top_level_task, 0,
      0, &register_task_event));
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
