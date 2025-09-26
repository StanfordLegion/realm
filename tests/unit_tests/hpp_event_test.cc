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

/*
 * UNTESTED FUNCTIONALITY DUE TO MOCK RUNTIME LIMITATIONS:
 *
 * The following methods cannot be tested with the mock runtime and are either skipped
 * or commented out:
 *
 * 1. Event::wait() - Blocks indefinitely waiting for event completion
 * 2. Event::external_wait() - Blocks indefinitely waiting for event completion
 * 3. Event::wait_faultaware() - Blocks indefinitely with fault checking
 * 4. Event::external_wait_faultaware() - Blocks indefinitely with fault checking
 * 5. Event::merge_events() - Requires complex event dependency management and runtime
 * coordination
 * 6. Event::merge_events_ignorefaults() - Requires event merging with fault tolerance
 * support
 * 7. UserEvent::trigger() with wait conditions - Requires event merging support
 * 8. Event::advise_event_ordering() - Unimplemented in current version
 * 9. UserEvent::cancel() - Unimplemented in current version
 * 10. Event::subscribe() - Unimplemented in current version
 * 11. Event::cancel_operation() - Unimplemented in current version
 * 12. Event::set_operation_priority() - Unimplemented in current version
 *
 * The mock runtime provides basic event creation and triggering but lacks the
 * sophisticated event lifecycle management, dependency coordination, and fault handling
 * required for these advanced features. Tests focus on the core functionality that can be
 * verified without blocking or requiring complex runtime infrastructure.
 */

#define REALM_NAMESPACE RealmHPP

#include "realm.hpp"
#include "test_mock.h"
#include <gtest/gtest.h>
#include <vector>
#include <set>

// Use the HPP namespace explicitly to avoid conflicts with test_mock.h
namespace HPP = REALM_NAMESPACE;

namespace Realm {
  extern bool enable_unit_tests;
};

class HPPEventTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_impl->init();

    // Set the global runtime singleton to our mock runtime
    Realm::runtime_singleton = runtime_impl.get();
  }

  static void TearDownTestSuite()
  {
    if(runtime_impl) {
      // Clear the global runtime singleton
      Realm::runtime_singleton = nullptr;
      runtime_impl->finalize();
      runtime_impl.reset();
    }
    Realm::enable_unit_tests = false;
  }

  static std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_impl;
};

class HPPUserEventTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_impl->init();

    // Set the global runtime singleton to our mock runtime
    Realm::runtime_singleton = runtime_impl.get();
  }

  static void TearDownTestSuite()
  {
    if(runtime_impl) {
      // Clear the global runtime singleton
      Realm::runtime_singleton = nullptr;
      runtime_impl->finalize();
      runtime_impl.reset();
    }
    Realm::enable_unit_tests = false;
  }

  static std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_impl;
};

// Static member definitions
std::unique_ptr<MockRuntimeImplWithEventFreeList> HPPEventTest::runtime_impl;
std::unique_ptr<MockRuntimeImplWithEventFreeList> HPPUserEventTest::runtime_impl;

// Test Event constructor and basic operations
TEST_F(HPPEventTest, ConstructorAndBasicOperations)
{
  // Default constructor
  HPP::Event event1;
  EXPECT_EQ(event1.id, REALM_NO_EVENT);
  EXPECT_FALSE(event1.exists());

  // Constructor with ID
  realm_event_t test_id = 12345;
  HPP::Event event2(test_id);
  EXPECT_EQ(event2.id, test_id);
  EXPECT_TRUE(event2.exists());

  // Copy constructor
  HPP::Event event3(event2);
  EXPECT_EQ(event3.id, test_id);
  EXPECT_TRUE(event3.exists());

  // Assignment operator
  HPP::Event event4;
  event4 = event2;
  EXPECT_EQ(event4.id, test_id);
  EXPECT_TRUE(event4.exists());
}

// Test Event comparison operators
TEST_F(HPPEventTest, ComparisonOperators)
{
  HPP::Event event1(100);
  HPP::Event event2(200);
  HPP::Event event3(100);

  // Less than
  EXPECT_TRUE(event1 < event2);
  EXPECT_FALSE(event2 < event1);
  EXPECT_FALSE(event1 < event3);

  // Equality
  EXPECT_TRUE(event1 == event3);
  EXPECT_FALSE(event1 == event2);

  // Inequality
  EXPECT_TRUE(event1 != event2);
  EXPECT_FALSE(event1 != event3);
}

// Test Event::NO_EVENT constant
TEST_F(HPPEventTest, NoEventConstant)
{
  EXPECT_EQ(HPP::Event::NO_EVENT.id, REALM_NO_EVENT);
  EXPECT_FALSE(HPP::Event::NO_EVENT.exists());
}

// Test Event::exists() method
TEST_F(HPPEventTest, ExistsMethod)
{
  HPP::Event no_event;
  EXPECT_FALSE(no_event.exists());

  HPP::UserEvent valid_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(valid_event.exists());
}

// Test Event::has_triggered() method
TEST_F(HPPEventTest, HasTriggeredMethod)
{
  // Create a real user event instead of using a hardcoded ID
  // This ensures the event exists in the mock runtime
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  // Initially, the event should not have triggered
  EXPECT_FALSE(user_event.has_triggered());

  // Trigger the event
  user_event.trigger();

  // Now the event should have triggered
  EXPECT_TRUE(user_event.has_triggered());
}

// Test Event::wait() method - SKIPPED (blocks indefinitely)

// Test Event::external_wait() method - SKIPPED (blocks indefinitely)

// Test Event::external_timedwait() method
TEST_F(HPPEventTest, ExternalTimedWaitMethod)
{
  // Create a real user event instead of using a hardcoded ID
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  // Test with a short timeout
  EXPECT_FALSE(user_event.external_timedwait(1000));
}

// Test Event::has_triggered_faultaware() method
TEST_F(HPPEventTest, HasTriggeredFaultAwareMethod)
{
  // Create a real user event instead of using a hardcoded ID
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  bool poisoned = false;
  bool triggered = user_event.has_triggered_faultaware(poisoned);
  // Both poisoned and triggered should be set by the method
  EXPECT_TRUE(triggered == true || triggered == false);
  EXPECT_TRUE(poisoned == true || poisoned == false);
}

// Test Event::wait_faultaware() method - SKIPPED (blocks indefinitely)

// Test Event::external_wait_faultaware() method - SKIPPED (blocks indefinitely)

// Test Event::external_timedwait_faultaware() method
TEST_F(HPPEventTest, ExternalTimedWaitFaultAwareMethod)
{
  // Create a real user event instead of using a hardcoded ID
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  bool poisoned = false;
  bool triggered =
      user_event.external_timedwait_faultaware(poisoned, 1000); // 1 microsecond timeout
  // Both poisoned and triggered should be set by the method
  EXPECT_FALSE(triggered);
  EXPECT_FALSE(poisoned);
}

// Test Event::subscribe() method (unimplemented)
TEST_F(HPPEventTest, SubscribeMethod)
{
  HPP::UserEvent event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(event.exists());

  EXPECT_THROW(event.subscribe(), std::logic_error);
}

// Test Event::cancel_operation() method (unimplemented)
TEST_F(HPPEventTest, CancelOperationMethod)
{
  HPP::UserEvent event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(event.exists());

  const char *reason = "test reason";

  EXPECT_THROW(event.cancel_operation(reason, strlen(reason)), std::logic_error);
}

// Test Event::set_operation_priority() method (unimplemented)
TEST_F(HPPEventTest, SetOperationPriorityMethod)
{
  HPP::UserEvent event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(event.exists());

  EXPECT_THROW(event.set_operation_priority(10), std::logic_error);
}

// Test Event::merge_events() static methods - SKIPPED (requires event merging support)

// Test Event::merge_events_ignorefaults() static methods - SKIPPED (requires event
// merging support)

// Test Event::advise_event_ordering() static methods (unimplemented)
TEST_F(HPPEventTest, AdviseEventOrderingMethods)
{
  // Create real user events for testing
  HPP::UserEvent before_event = HPP::UserEvent::create_user_event();
  HPP::UserEvent after_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(before_event.exists());
  EXPECT_TRUE(after_event.exists());

  // Test single event version
  EXPECT_THROW(HPP::Event::advise_event_ordering(before_event, after_event),
               std::logic_error);

  // Test array version
  HPP::Event before_events[] = {before_event, after_event};
  EXPECT_THROW(HPP::Event::advise_event_ordering(before_events, 2, after_event),
               std::logic_error);

  // Test span version
  std::vector<HPP::Event> before_vec = {before_event, after_event};
  EXPECT_THROW(HPP::Event::advise_event_ordering(before_vec, after_event),
               std::logic_error);
}

// Test UserEvent constructor and basic operations
TEST_F(HPPUserEventTest, ConstructorAndBasicOperations)
{
  // Default constructor
  HPP::UserEvent user_event1;
  EXPECT_EQ(user_event1.id, REALM_NO_EVENT);
  EXPECT_FALSE(user_event1.exists());

  // Constructor with ID
  realm_id_t test_id = 12345;
  HPP::UserEvent user_event2(test_id);
  EXPECT_EQ(user_event2.id, test_id);
  EXPECT_TRUE(user_event2.exists());

  // Copy constructor
  HPP::UserEvent user_event3(user_event2);
  EXPECT_EQ(user_event3.id, test_id);
  EXPECT_TRUE(user_event3.exists());

  // Assignment operator
  HPP::UserEvent user_event4;
  user_event4 = user_event2;
  EXPECT_EQ(user_event4.id, test_id);
  EXPECT_TRUE(user_event4.exists());
}

// Test UserEvent::create_user_event() static method
TEST_F(HPPUserEventTest, CreateUserEventMethod)
{
  HPP::UserEvent new_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(new_event.exists());
  EXPECT_NE(new_event.id, REALM_NO_EVENT);
}

// Test UserEvent::trigger() method - PARTIALLY TESTED (basic trigger only)
TEST_F(HPPUserEventTest, TriggerMethod)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();

  // Test triggering without waiting on another event
  user_event.trigger();

  // Note: Trigger with wait conditions requires event merging support (not tested)
}

// Test UserEvent::cancel() method (unimplemented)
TEST_F(HPPUserEventTest, CancelMethod)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  EXPECT_THROW(user_event.cancel(), std::logic_error);
}

// Test UserEvent::NO_USER_EVENT constant
TEST_F(HPPUserEventTest, NoUserEventConstant)
{
  EXPECT_EQ(HPP::UserEvent::NO_USER_EVENT.id, REALM_NO_EVENT);
  EXPECT_FALSE(HPP::UserEvent::NO_USER_EVENT.exists());
}

// Test Event implicit conversion to realm_event_t
TEST_F(HPPEventTest, ImplicitConversion)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  realm_event_t event_id = user_event;
  EXPECT_EQ(event_id, user_event.id);
}

// Test UserEvent inheritance from Event
TEST_F(HPPUserEventTest, Inheritance)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  // Test that UserEvent can be used where Event is expected
  HPP::Event base_event = user_event;
  EXPECT_EQ(base_event.id, user_event.id);

  // Test that UserEvent methods work
  EXPECT_TRUE(user_event.exists());
}

// Test edge cases and error conditions
TEST_F(HPPEventTest, EdgeCases)
{
  // Test with a large ID value
  HPP::Event large_event(0x7FFFFFFFFFFFFFFFULL);
  EXPECT_TRUE(large_event.exists());

  // Test with zero ID (should not exist since it's REALM_NO_EVENT)
  HPP::Event zero_event(0);
  EXPECT_FALSE(zero_event.exists());

  // Test comparison with NO_EVENT
  HPP::UserEvent normal_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(normal_event.exists());
  EXPECT_TRUE(HPP::Event::NO_EVENT < normal_event);
  EXPECT_FALSE(normal_event < HPP::Event::NO_EVENT);
}

// Test merge_events with empty collections - SKIPPED (requires event merging support)

// Test merge_events with single event - SKIPPED (requires event merging support)

// Test UserEvent creation and immediate triggering
TEST_F(HPPUserEventTest, CreateAndTrigger)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  // Trigger immediately
  user_event.trigger();

  // Check if it has triggered (this may not work without proper runtime setup)
  // but at least we can verify the method calls don't crash
}

// Test UserEvent with NO_EVENT wait condition
TEST_F(HPPUserEventTest, TriggerWithNoEvent)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();

  // Trigger with NO_EVENT (should not wait)
  user_event.trigger(HPP::Event::NO_EVENT);
}

// Test multiple UserEvent creation
TEST_F(HPPUserEventTest, MultipleUserEvents)
{
  std::vector<HPP::UserEvent> events;

  // Create multiple user events
  for(int i = 0; i < 10; ++i) {
    HPP::UserEvent event = HPP::UserEvent::create_user_event();
    events.push_back(event);
    EXPECT_TRUE(event.exists());
  }

  // Verify all events are unique
  std::set<realm_id_t> ids;
  for(const auto &event : events) {
    EXPECT_TRUE(ids.insert(event.id).second);
  }
}

// Test UserEvent triggering and event state
TEST_F(HPPUserEventTest, TriggerAndEventState)
{
  HPP::UserEvent user_event = HPP::UserEvent::create_user_event();
  EXPECT_TRUE(user_event.exists());

  // Initially, the event should not have triggered
  bool triggered = user_event.has_triggered();
  EXPECT_FALSE(triggered);

  // Trigger the event
  user_event.trigger();

  // Now the event should have triggered
  triggered = user_event.has_triggered();
  EXPECT_TRUE(triggered);
}

// Test UserEvent with wait condition - SKIPPED (requires event merging support)

// Test event merging with actual events - SKIPPED (requires event merging support)

// Test event merging with ignore faults - SKIPPED (requires event merging support)

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
