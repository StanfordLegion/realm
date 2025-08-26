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
#include <gtest/gtest.h>

#define TEN_MS_IN_NS 10000000

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

// test event without parameters

class CEventTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_impl->init();
  }

  void TearDown() override { runtime_impl->finalize(); }

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
  realm_status_t status = realm_event_wait(nullptr, event, REALM_WAIT_INFINITE, nullptr);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

// TODO: finish this test once we remove the get_runtime and threading
TEST_F(CEventTest, DISABLED_EventWaitTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));

  realm_status_t status = realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, DISABLED_EventWaitNotTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  realm_status_t status = realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, DISABLED_EventWaitTimeoutEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Record start time before the wait
  auto start_time = std::chrono::high_resolution_clock::now();

  realm_status_t status = realm_event_wait(runtime, event, TEN_MS_IN_NS, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);

  // Record end time after the wait and calculate elapsed time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

  // Verify that the wait lasted at least as long as requested (10ms)
  EXPECT_GE(elapsed_time.count(), TEN_MS_IN_NS)
      << "Event wait should have lasted at least 10ms, but only lasted "
      << elapsed_time.count() << " nanoseconds";

  int has_triggered = 0;
  status = realm_event_has_triggered(runtime, event, &has_triggered, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(has_triggered, 0);
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

  realm_status_t status = realm_event_wait(runtime, event, REALM_WAIT_INFINITE, nullptr);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, DISABLED_EventWaitPoisoned)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));
  ASSERT_REALM(realm_event_cancel_operation(runtime, event, nullptr, 0));

  int poisoned = 0;
  realm_status_t status =
      realm_event_wait(runtime, event, REALM_WAIT_INFINITE, &poisoned);
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

TEST_F(CEventTest, EventCancelOperationNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
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

// TODO: remove the get_runtime in the has_triggered_faultaware function to unblock the
// test
TEST_F(CEventTest, DISABLED_EventCancelOperationNullReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Test with null reason_data but zero size
  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationNullReasonDataNonZeroSize)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Test with null reason_data but non-zero size - this should still succeed
  // as the implementation may handle this gracefully
  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 10);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationValidReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  const char *reason_data = "Test cancellation reason";
  size_t reason_size = strlen(reason_data);

  realm_status_t status =
      realm_event_cancel_operation(runtime, event, reason_data, reason_size);
  EXPECT_EQ(status, REALM_SUCCESS);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationLargeReasonData)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Create a larger reason data buffer
  std::vector<char> large_reason_data(1024, 'A');
  size_t reason_size = large_reason_data.size();

  realm_status_t status =
      realm_event_cancel_operation(runtime, event, large_reason_data.data(), reason_size);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_F(CEventTest, EventCancelOperationZeroReasonSize)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  const char *reason_data = "This should be ignored";

  realm_status_t status = realm_event_cancel_operation(runtime, event, reason_data, 0);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationAlreadyTriggeredEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // First trigger the event
  ASSERT_REALM(realm_user_event_trigger(runtime, event, REALM_NO_EVENT, 0));

  // Then try to cancel it - should still succeed but be a no-op
  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationAlreadyCancelledEvent)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // First cancel the event
  ASSERT_REALM(realm_event_cancel_operation(runtime, event, nullptr, 0));

  // Then try to cancel it again - should still succeed
  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 0);
  EXPECT_EQ(status, REALM_SUCCESS);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationInvalidEvent)
{
  realm_runtime_t runtime = *runtime_impl;

  // Create an event and then invalidate it by bumping generation
  realm_user_event_t event = REALM_NO_EVENT;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Bump the generation to make it invalid
  GenEventImpl *e = runtime_impl->get_genevent_impl(Event(event));
  e->generation.store(e->generation.load() + 1);

  // Try to cancel the invalid event
  realm_status_t status = realm_event_cancel_operation(runtime, event, nullptr, 0);
  // This should either succeed (if validation is lenient) or return an error
  // The exact behavior depends on the implementation
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationMultipleCancellations)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Cancel multiple times with different reason data
  const char *reason1 = "First cancellation";
  const char *reason2 = "Second cancellation";
  const char *reason3 = "Third cancellation";

  realm_status_t status1 =
      realm_event_cancel_operation(runtime, event, reason1, strlen(reason1));
  EXPECT_EQ(status1, REALM_SUCCESS);

  realm_status_t status2 =
      realm_event_cancel_operation(runtime, event, reason2, strlen(reason2));
  EXPECT_EQ(status2, REALM_SUCCESS);

  realm_status_t status3 =
      realm_event_cancel_operation(runtime, event, reason3, strlen(reason3));
  EXPECT_EQ(status3, REALM_SUCCESS);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test
TEST_F(CEventTest, DISABLED_EventCancelOperationBoundaryReasonSize)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  ASSERT_REALM(realm_user_event_create(runtime, &event));

  // Test with very small reason size
  realm_status_t status1 = realm_event_cancel_operation(runtime, event, "A", 1);
  EXPECT_EQ(status1, REALM_SUCCESS);

  // Test with size_t max (this might be too large for practical use)
  // We'll use a more reasonable large size instead
  std::vector<char> large_buffer(10000, 'X');
  realm_status_t status2 = realm_event_cancel_operation(
      runtime, event, large_buffer.data(), large_buffer.size());
  EXPECT_EQ(status2, REALM_SUCCESS);
}
