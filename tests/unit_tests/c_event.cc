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

#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <thread>
#include <chrono>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

// test event without parameters
// Note: The mock runtime environment has limitations for testing event waiting
// on untriggered events. Tests focus on parameter validation and basic functionality.

class CEventTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    // Many Event code paths still use get_runtime() which relies on the global
    // runtime_singleton. Point it at our mock runtime for the duration of the test.
    Realm::runtime_singleton = runtime_impl.get();
    runtime_impl->init();
  }

  void TearDown() override
  {
    runtime_impl->finalize();
    Realm::runtime_singleton = nullptr;
  }

  std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_impl{nullptr};
};

TEST_F(CEventTest, CreateUserEventNullRuntime)
{
  realm_user_event_t event;
  realm_status_t status = realm_user_event_create(nullptr, &event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, CreateUserEventNullEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_user_event_create(runtime, nullptr);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

// TODO(wei): Fix this once get_runtime() is removed from GenEventImpl::GenEventImpl
TEST_F(CEventTest, CreateUserEventSuccess)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_user_event_create(runtime, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_TRUE(ID(event).is_event());
}

TEST_F(CEventTest, MergeEventsNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(nullptr, nullptr, 0, &event, 0);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, MergeEventsNullWaitFor)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_event_merge(runtime, nullptr, 1, &event, 0);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

TEST_F(CEventTest, MergeEventsNullEvent)
{
  realm_user_event_t wait_for_events[2] = {REALM_NO_EVENT, REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 1, nullptr, 0);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

TEST_F(CEventTest, MergeEventsZeroWaitFor)
{
  realm_user_event_t wait_for_events[1] = {REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 0, &event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, REALM_NO_EVENT);
}

TEST_F(CEventTest, MergeEventsNoEvents)
{
  realm_user_event_t wait_for_events[2] = {REALM_NO_EVENT, REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 2, &event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, REALM_NO_EVENT);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test, merge
// calls it.
TEST_F(CEventTest, DISABLED_MergeEventsSuccess)
{
  const int num_events = 2;
  realm_user_event_t wait_for_events[num_events];
  realm_runtime_t runtime = *runtime_impl;
  for(int i = 0; i < num_events; i++) {
    ASSERT_REALM(realm_user_event_create(runtime, &wait_for_events[i]));
  }
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status =
      realm_event_merge(runtime, wait_for_events, num_events, &event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_TRUE(ID(event).is_event());
}

TEST_F(CEventTest, DISABLED_MergeEventsWithPoisonedIgnoreFaults)
{
  const int num_events = 2;
  realm_user_event_t wait_for_events[num_events];
  realm_runtime_t runtime = *runtime_impl;
  for(int i = 0; i < num_events; i++) {
    ASSERT_REALM(realm_user_event_create(runtime, &wait_for_events[i]));
  }
  ASSERT_REALM(realm_event_cancel_operation(runtime, wait_for_events[0], nullptr, 0));

  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status =
      realm_event_merge(runtime, wait_for_events, num_events, &event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_TRUE(ID(event).is_event());
}

TEST_F(CEventTest, DISABLED_MergeEventsWithPoisonedNoIgnoreFaults)
{
  const int num_events = 2;
  realm_user_event_t wait_for_events[num_events];
  realm_runtime_t runtime = *runtime_impl;
  for(int i = 0; i < num_events; i++) {
    ASSERT_REALM(realm_user_event_create(runtime, &wait_for_events[i]));
  }
  ASSERT_REALM(realm_event_cancel_operation(runtime, wait_for_events[0], nullptr, 0));

  realm_user_event_t event = REALM_NO_EVENT;
  // we will get back wait_for_events[0] because it is poisoned
  realm_status_t status =
      realm_event_merge(runtime, wait_for_events, num_events, &event, 1);
  EXPECT_EQ(status, REALM_SUCCESS);
  bool poisoned = false;
  Event(event).has_triggered_faultaware(poisoned);
  EXPECT_TRUE(poisoned);
}

TEST_F(CEventTest, EventWaitNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_wait(nullptr, event, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

// TODO: finish this test once we remove the get_runtime and threading
TEST_F(CEventTest, DISABLED_EventWaitTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));

  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, DISABLED_EventWaitNotTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

// // an event id with maybe a higher generation than is triggered should return an error
TEST_F(CEventTest, DISABLED_EventWaitInvalidEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  // we bump the generation
  GenEventImpl *e = runtime_impl->get_genevent_impl(Event(event));
  e->generation.store(e->generation.load() + 1);

  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, DISABLED_EventWaitPoisoned)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_event_cancel_operation(runtime, event, nullptr, 0));

  int poisoned = 0;
  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(poisoned, 1);
}

// TODO: remove the get_runtime in the trigger function
TEST_F(CEventTest, DISABLED_UserEventTriggerWithoutWaitSuccess)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_user_event_create(runtime, &event);
  ASSERT_REALM(status);

  status = realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  Event event_cxx = Event(event);
  EXPECT_TRUE(event_cxx.has_triggered());
}

TEST_F(CEventTest, DISABLED_UserEventTriggerWithWaitSuccess)
{
  realm_user_event_t wait_on_event = REALM_NO_EVENT;
  realm_user_event_t user_event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &wait_on_event));
  ASSERT_REALM(realm_user_event_create(runtime, &user_event));

  realm_status_t status = realm_user_event_trigger(runtime, user_event, wait_on_event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  ASSERT_REALM(realm_user_event_trigger(runtime, wait_on_event, REALM_NO_EVENT, 0));
  EXPECT_TRUE(Event(user_event).has_triggered());
}

TEST_F(CEventTest, DISABLED_UserEventTriggerWithWaitPoisoned)
{
  realm_user_event_t wait_on_event = REALM_NO_EVENT;
  realm_user_event_t user_event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &wait_on_event));
  ASSERT_REALM(realm_user_event_create(runtime, &user_event));

  realm_status_t status = realm_user_event_trigger(runtime, user_event, wait_on_event, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
  ASSERT_REALM(realm_event_cancel_operation(runtime, wait_on_event, nullptr, 0));
  bool poisoned = false;
  Event(user_event).has_triggered_faultaware(poisoned);
  EXPECT_TRUE(poisoned);
}

TEST_F(CEventTest, EventHasTriggeredNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
  int has_triggered = 0;
  int poisoned = 0;
  realm_status_t status =
      realm_event_has_triggered(nullptr, event, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, EventHasTriggeredNoEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_event_t event = REALM_NO_EVENT;
  int has_triggered = 0;
  int poisoned = 0;
  realm_status_t status =
      realm_event_has_triggered(runtime, event, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1); // NO_EVENT has always triggered
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, DISABLED_EventHasTriggeredTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT,
                                        0)); // FIXME: this need runtime singleton

  int has_triggered = 0;
  int poisoned = 0;
  realm_status_t status =
      realm_event_has_triggered(runtime, event, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1);
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, EventHasTriggeredNotTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  int has_triggered = 0;
  int poisoned = 0;
  realm_status_t status =
      realm_event_has_triggered(runtime, event, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 0);
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, DISABLED_EventHasTriggeredPoisoned)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_event_cancel_operation(runtime, event, nullptr, 0));

  int has_triggered = 0;
  int poisoned = 0;
  realm_status_t status =
      realm_event_has_triggered(runtime, event, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 0);
  EXPECT_EQ(poisoned, 1);
}

// ============================================================================
// Tests for realm_event_wait function
// ============================================================================

TEST_F(CEventTest, EventWaitNullEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_event_wait(runtime, REALM_NO_EVENT, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventWaitNoEventAlwaysTriggered)
{
  realm_runtime_t runtime = *runtime_impl;
  int has_triggered = 0;
  int poisoned = 0;

  realm_status_t status =
      realm_event_wait(runtime, REALM_NO_EVENT, 0, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1);
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, EventWaitNoEventWithTimedWait)
{
  realm_runtime_t runtime = *runtime_impl;
  int has_triggered = 0;
  int poisoned = 0;

  realm_status_t status =
      realm_event_wait(runtime, REALM_NO_EVENT, 1000, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1);
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, EventWaitTimedWaitWithoutHasTriggered)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Timed wait requires has_triggered parameter
  realm_status_t status = realm_event_wait(runtime, event, 1000, nullptr, nullptr);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}

TEST_F(CEventTest, EventWaitTimedWaitWithHasTriggered)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Timed wait should return without blocking and require has_triggered
  int has_triggered = 0;
  int poisoned = 0;

  // Use a small positive timeout to exercise timed wait path
  realm_status_t status =
      realm_event_wait(runtime, event, 1000, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  // Event not triggered yet, so should time out and report not triggered
  EXPECT_EQ(has_triggered, 0);
  EXPECT_EQ(poisoned, 0);
}

TEST_F(CEventTest, EventWaitTimedWaitWithPoisonedOnly)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  int poisoned = 0;
  realm_status_t status = realm_event_wait(runtime, event, 1000, nullptr, &poisoned);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}

TEST_F(CEventTest, EventWaitZeroTimeout)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Zero timeout means infinite wait; trigger from another thread to avoid hang
  std::thread t([&]() {
    // give the waiter a moment to block
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));
  });

  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
  t.join();
}

TEST_F(CEventTest, EventWaitNegativeTimeout)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Negative timeout means infinite wait; trigger from another thread
  std::thread t([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));
  });

  realm_status_t status = realm_event_wait(runtime, event, -1000, nullptr, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
  t.join();
}

TEST_F(CEventTest, EventWaitWithOutputParameters)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  int has_triggered = 0;
  int poisoned = 0;
  // Infinite wait; trigger from another thread and expect triggered on return
  std::thread t1([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));
  });
  realm_status_t status = realm_event_wait(runtime, event, 0, &has_triggered, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1);
  EXPECT_EQ(poisoned, 0);
  t1.join();
}

TEST_F(CEventTest, EventWaitWithOnlyHasTriggered)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  int has_triggered = 0;
  // Infinite wait; trigger from another thread and expect has_triggered=1
  std::thread t2([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));
  });
  realm_status_t status = realm_event_wait(runtime, event, 0, &has_triggered, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 1);
  t2.join();
}

TEST_F(CEventTest, EventWaitWithOnlyPoisoned)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  int poisoned = 0;
  // Infinite wait; trigger from another thread and expect poisoned=0
  std::thread t3([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));
  });
  realm_status_t status = realm_event_wait(runtime, event, 0, nullptr, &poisoned);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(poisoned, 0);
  t3.join();
}

// ============================================================================
// Tests for realm_event_cancel_operation function
// ============================================================================

TEST_F(CEventTest, EventCancelOperationNullRuntime)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_cancel_operation(nullptr, event, nullptr, 0);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, EventCancelOperationNoEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_event_cancel_operation(runtime, REALM_NO_EVENT, nullptr, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithNullReason)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithEmptyReason)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  realm_status_t status = realm_event_cancel_operation(runtime, event, "", 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  const char *reason = "Test cancellation reason";
  realm_status_t status =
      realm_event_cancel_operation(runtime, event, reason, strlen(reason));
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithLargeReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  std::string large_reason(1000, 'x'); // 1000 character reason
  realm_status_t status = realm_event_cancel_operation(
      runtime, event, large_reason.c_str(), large_reason.size());
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithBinaryReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  unsigned char binary_data[] = {0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34, 0x56, 0x78};
  realm_status_t status =
      realm_event_cancel_operation(runtime, event, binary_data, sizeof(binary_data));
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationMultipleTimes)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Cancel the same event multiple times
  realm_status_t status1 =
      realm_event_cancel_operation(runtime, event, "First cancellation", 18);
  EXPECT_EQ(status1, REALM_SUCCESS);

  realm_status_t status2 =
      realm_event_cancel_operation(runtime, event, "Second cancellation", 19);
  EXPECT_EQ(status2, REALM_SUCCESS);

  realm_status_t status3 =
      realm_event_cancel_operation(runtime, event, "Third cancellation", 18);
  EXPECT_EQ(status3, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationMultipleEvents)
{
  realm_user_event_t events[3];
  realm_runtime_t runtime = *runtime_impl;

  for(int i = 0; i < 3; i++) {
    ASSERT_REALM(realm_user_event_create(runtime, &events[i]));
  }

  // Cancel all events with different reasons
  const char *reasons[] = {"First event cancelled", "Second event cancelled",
                           "Third event cancelled"};
  for(int i = 0; i < 3; i++) {
    realm_status_t status =
        realm_event_cancel_operation(runtime, events[i], reasons[i], strlen(reasons[i]));
    EXPECT_EQ(status, REALM_SUCCESS);
  }
}

TEST_F(CEventTest, EventCancelOperationWithZeroReasonSize)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  const char *reason = "This reason should be ignored";
  realm_status_t status = realm_event_cancel_operation(runtime, event, reason, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationWithMaxReasonSize)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Test with a very large reason size
  std::string max_reason(10000, 'z'); // 10KB reason
  realm_status_t status =
      realm_event_cancel_operation(runtime, event, max_reason.c_str(), max_reason.size());
  EXPECT_EQ(status, REALM_SUCCESS);
}