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

// C++ header for realm

#ifndef REALM_HPP
#define REALM_HPP

#include <algorithm>
#include <cassert>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <functional>
#include <string>
#include <vector>

#if __cplusplus >= 202002L
#include <span>
#endif

#include "realm/realm_c.h"

// Macro to check that C API calls return REALM_SUCCESS
#define REALM_CHECK(call)                                                                \
  do {                                                                                   \
    realm_status_t _status = (call);                                                     \
    if(_status != REALM_SUCCESS) {                                                       \
      throw std::runtime_error("Realm C API call failed with status " +                  \
                               std::to_string(_status) + ": " #call);                    \
    }                                                                                    \
  } while(0)

#ifndef REALM_NAMESPACE
#define REALM_NAMESPACE Realm
#endif
namespace REALM_NAMESPACE {
  // Forward declarations
  class ProfilingRequestSet;
  class CodeDescriptor;
  class ReductionOpUntyped;
  class CustomSerdezUntyped;
  class ModuleConfig;
  class Module;

  // Template forward declarations
  template <typename T>
  class ReductionOp;
  template <typename T>
  class CustomSerdezWrapper;
  template <int N, typename T>
  class Rect;
  template <int N, typename T>
  class IndexSpace;

  namespace Detail {
    const size_t dynamic_extent = size_t(-1);
    template <typename T, size_t Extent = dynamic_extent>
    class span;
    /**
     * \class span
     * \brief A lightweight view over a contiguous sequence of objects.
     *
     * Provides a C++11/14/17 compatible implementation of std::span for projects
     * that cannot use C++20. This class provides a non-owning view over a
     * contiguous sequence of objects with compile-time or runtime bounds.
     *
     * \tparam T The element type
     */
    template <typename T>
    class span<T, dynamic_extent> {
    public:
      typedef typename std::remove_const<T>::type value_type;
      static const size_t extent = dynamic_extent;

      /**
       * \brief Default constructor creating an empty span.
       */
      span()
        : base(0)
        , length(0)
      {}

      /**
       * \brief Construct a span from a pointer and length.
       * \param _base Pointer to the first element
       * \param _length Number of elements in the sequence
       */
      span(T *_base, size_t _length)
        : base(_base)
        , length(_length)
      {}

      /**
       * \brief Copy constructor from another span with different extent.
       * \tparam Extent2 The extent of the source span
       * \param copy_from The span to copy from
       */
      template <size_t Extent2>
      span(span<T, Extent2> copy_from)
        : base(copy_from.data())
        , length(copy_from.size())
      {}

      /**
       * \brief Construct a span from a vector.
       * \param v The vector to create a span over
       */
      // For const spans
      span(const std::vector<typename std::remove_const<T>::type> &v)
        : base(v.data())
        , length(v.size())
      {}
      // For non-const spans
      span(std::vector<T> &v)
        : base(v.data())
        , length(v.size())
      {}

      /**
       * \brief Construct a span from a single scalar reference.
       * \param v Reference to the scalar element
       */
      span(T &v)
        : base(&v)
        , length(1)
      {}

      /**
       * \brief Construct a span from a C-style array.
       * \param arr Reference to the C-style array
       */
      template <size_t N>
      span(T (&arr)[N])
        : base(arr)
        , length(N)
      {}

      /**
       * \brief Access element at specified index.
       * \param idx Index of the element to access
       * \return Reference to the element at the specified index
       */
      T &operator[](size_t idx) const { return base[idx]; }

      /**
       * \brief Get pointer to the underlying data.
       * \return Pointer to the first element
       */
      T *data() const { return base; }

      /**
       * \brief Get the number of elements in the span.
       * \return Number of elements
       */
      size_t size() const { return length; }

      /**
       * \brief Check if the span is empty.
       * \return true if the span contains no elements, false otherwise
       */
      bool empty() const { return (length == 0); }

      // Iterator support
    public:
      typedef T *iterator;
      typedef const T *const_iterator;

      iterator begin() { return base; }
      iterator end() { return base + length; }
      const_iterator begin() const { return base; }
      const_iterator end() const { return base + length; }
      const_iterator cbegin() const { return base; }
      const_iterator cend() const { return base + length; }

    private:
      T *base;
      size_t length;
    };
  } // namespace Detail
#if __cplusplus >= 202002L
  using std::dynamic_extent;
  using std::span;
#else
  using Detail::dynamic_extent;
  using Detail::span;
#endif
  /**
   * \class empty_span
   * \brief A helper class that can be implicitly converted to any span type.
   *
   * This class provides a convenient way to represent empty spans that can
   * be converted to any span<T> type.
   */
  class empty_span {
  public:
    /**
     * \brief Implicit conversion operator to any span type.
     * \tparam T The element type of the target span
     * \return An empty span of the specified type
     */
    template <typename T>
    operator span<T, dynamic_extent>() const
    {
      return span<T, dynamic_extent>();
    }
  };

  /**
   * \brief Create a span from a pointer and length.
   * \tparam T The element type
   * \param base Pointer to the first element
   * \param length Number of elements in the sequence
   * \return A span over the specified range
   */
  template <typename T>
  span<T, dynamic_extent> make_span(T *base, size_t length)
  {
    return span<T, dynamic_extent>(base, length);
  }

  /**
   * \class Event
   * Event is created by the runtime and is used to synchronize
   * operations.  An event is triggered when the operation it
   * represents is complete and can be used as pre and post conditions
   * for other operations. This class represents a handle to the event
   * itself and can be passed-by-value as well as
   * serialized/deserialized anywhere in the program. Note that events
   * do not need to be explicitly garbage collected.
   */
  class Event {
  public:
    realm_event_t id{REALM_NO_EVENT};

    Event() = default;
    constexpr explicit Event(realm_event_t id)
      : id(id)
    {}

    constexpr operator realm_event_t() const { return id; }

    bool operator<(const Event &rhs) const { return id < rhs.id; }
    bool operator==(const Event &rhs) const { return id == rhs.id; }
    bool operator!=(const Event &rhs) const { return id != rhs.id; }

    /**
     * \brief The value should be usued to initialize an event
     * handle. NO_EVENT is always in has triggered state .
     */
    static const Event NO_EVENT;

    /**
     * Check whether an event has a valid ID.
     * \return true if the event has a valid ID, false otherwise
     */
    bool exists(void) const { return id != REALM_NO_EVENT; }

    /**
     * Test whether an event has triggered without waiting.
     * \return true if the event has triggered, false otherwise
     */
    bool has_triggered(void) const
    {
      bool poisoned = false;
      return has_triggered_faultaware(poisoned);
    }

    /**
     * Wait for an event to trigger.
     */
    void wait() const
    {
      bool poisoned = false;
      wait_common(poisoned, REALM_WAIT_INFINITE);
    }

    /**
     * External wait - blocks until the event triggers.
     */
    void external_wait(void) const { wait(); }

    /**
     * External wait with a timeout - returns true if event triggers, false
     * if the maximum delay occurs first
     * \param max_ns the maximum number of nanoseconds to wait
     * \return true if the event has triggered, false if the timeout occurred
     */
    bool external_timedwait(int64_t max_ns) const
    {
      bool poisoned = false;
      return wait_common(poisoned, max_ns);
    }

    /**
     * Fault-aware versions of the above (the above versions will cause the
     * caller to fault as well if a poisoned event is queried).
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    bool has_triggered_faultaware(bool &poisoned) const
    {
      realm_runtime_t runtime;
      int tmp_has_triggered;
      int tmp_poisoned;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      REALM_CHECK(
          realm_event_has_triggered(runtime, id, &tmp_has_triggered, &tmp_poisoned));
      poisoned = (tmp_poisoned != 0);
      return tmp_has_triggered != 0;
    }

    /**
     * Fault-aware versions of the wait function.
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    void wait_faultaware(bool &poisoned) const
    {
      wait_common(poisoned, REALM_WAIT_INFINITE);
    }

    /**
     * Fault-aware versions of the external wait function.
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    void external_wait_faultaware(bool &poisoned) const
    {
      wait_common(poisoned, REALM_WAIT_INFINITE);
    }

    /**
     * Fault-aware versions of the external timed wait function.
     * \param poisoned set to true if the event is poisoned
     * \param max_ns the maximum number of nanoseconds to wait
     * \return true if the event has triggered, false if the timeout occurred
     */
    bool external_timedwait_faultaware(bool &poisoned, int64_t max_ns) const
    {
      return wait_common(poisoned, max_ns);
    }

  private:
    bool wait_common(bool &poisoned, int64_t max_ns) const
    {
      realm_runtime_t runtime;
      int has_triggered;
      int tmp_poisoned;
      realm_runtime_get_runtime(&runtime);
      realm_event_wait(runtime, id, max_ns, nullptr);
      realm_event_has_triggered(runtime, id, &has_triggered, &tmp_poisoned);
      poisoned = tmp_poisoned != 0;
      return has_triggered != 0;
    }

  public:
    /**
     * Subscribe to an event, ensuring that the triggeredness of the
     * event will be available as soon as possible (and without having to call
     * wait).
     */
    void subscribe(void) const { throw std::logic_error("Not implemented"); }

    /**
     * Attempt to cancel the operation associated with this event.
     * \param reason_data will be provided to any profilers of the operation
     * \param reason_size the size of the reason data
     */
    void cancel_operation(const void *reason_data, size_t reason_size) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * Attempt to change the priority of the operation associated with this
     * event.
     * \param new_priority the new priority.
     */
    void set_operation_priority(int new_priority) const
    {
      throw std::logic_error("Not implemented");
    }

    ///@{
    /**
     * Create an event that won't trigger until all input events
     * have.
     * \param wait_for the events to wait for
     * \return the event that will trigger when all input events
     * have.
     */
    static Event merge_events(const Event *wait_for, size_t num_events)
    {
      return merge_events_common(wait_for, num_events, false);
    }

    template <typename... Args>
    static Event merge_events(Event ev1, Event ev2, Args... args)
    {
      Event events[] = {ev1, ev2, Event(args)...};
      return merge_events_common(events, sizeof(events) / sizeof(events[0]), false);
    }

    static Event merge_events(const std::set<Event> &wait_for)
    {
      std::vector<Event> events(wait_for.begin(), wait_for.end());
      return merge_events_common(events.data(), events.size(), false);
    }

    static Event merge_events(const span<const Event> &wait_for)
    {
      return merge_events_common(wait_for.data(), wait_for.size(), false);
    }
    ///@}

    ///@{
    /**
     * Create an event that won't trigger until all input events
     * have, ignoring any poison on the input events.
     * \param wait_for the events to wait for
     * \return the event that will trigger when all input events
     * have.
     */
    static Event merge_events_ignorefaults(const Event *wait_for, size_t num_events)
    {
      return merge_events_common(wait_for, num_events, true);
    }

    template <typename... Args>
    static Event merge_events_ignorefaults(Event ev1, Event ev2, Args... args)
    {
      Event events[] = {ev1, ev2, Event(args)...};
      return merge_events_common(events, sizeof(events) / sizeof(events[0]), true);
    }

    static Event merge_events_ignorefaults(const std::set<Event> &wait_for)
    {
      std::vector<Event> events(wait_for.begin(), wait_for.end());
      return merge_events_common(events.data(), events.size(), true);
    }

    static Event merge_events_ignorefaults(const span<const Event> &wait_for)
    {
      return merge_events_common(wait_for.data(), wait_for.size(), true);
    }
    ///@}

  private:
    static Event merge_events_common(const Event *wait_for, size_t num_events,
                                     bool ignore_faults)
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));

      std::vector<realm_event_t> event_ids(num_events);
      std::transform(wait_for, wait_for + num_events, event_ids.begin(),
                     [](const Event &e) { return e.id; });

      realm_event_t merged_event_id;
      REALM_CHECK(realm_event_merge(runtime, event_ids.data(), num_events,
                                    &merged_event_id, ignore_faults ? 1 : 0));
      return Event(merged_event_id);
    }

  public:
    /**
     * The following call is used to give Realm a bound on when the UserEvent
     * will be triggered.  In addition to being useful for diagnostic purposes
     * (e.g. detecting event cycles), having a "happens_before" allows Realm
     * to judge that the UserEvent trigger is "in flight".
     * \param happens_before the event that must occur before the UserEvent
     * \param happens_after the event that must occur after the UserEvent
     */
    static void advise_event_ordering(Event happens_before, Event happens_after)
    {
      throw std::logic_error("Not implemented");
    }
    static void advise_event_ordering(const Event *happens_before, size_t num_events,
                                      Event happens_after, bool all_must_trigger = true)
    {
      throw std::logic_error("Not implemented");
    }
    static void advise_event_ordering(const span<Event> &happens_before,
                                      Event happens_after, bool all_must_trigger = true)
    {
      advise_event_ordering(happens_before.data(), happens_before.size(), happens_after,
                            all_must_trigger);
    }
  };

  /**
   * \class UserEvent
   * UserEvents are events that can be scheduled to trigger at a
   * future point in an application and can be waited upon
   * dynamically. This is in contrast to a Realm::Event, which must
   * be a direct consequence of the completion of a specific
   * operation.
   */
  class UserEvent : public Event {
  public:
    UserEvent() = default;
    constexpr UserEvent(realm_event_t id)
      : Event(id)
    {}

    /**
     * Create a new user event.
     * \return the new user event
     */
    static UserEvent create_user_event(void)
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_user_event_t newId;
      REALM_CHECK(realm_user_event_create(runtime, &newId));
      return UserEvent(newId);
    }

    /**
     * Trigger the user event.
     * \param wait_on an event that must trigger before this user event
     * can.
     * \param ignore_faults if true, the user event will be triggered even if
     * the event it is waiting on is poisoned.
     */
    void trigger(Event wait_on = Event::NO_EVENT, bool ignore_faults = false) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      REALM_CHECK(
          realm_user_event_trigger(runtime, *this, wait_on, ignore_faults ? 1 : 0));
    }

    /*
     * Attempt to cancell all the operations waiting on this user
     * event.
     */
    void cancel(void) const { throw std::logic_error("Not implemented"); }

    static const UserEvent NO_USER_EVENT;
  };

  /**
   * \class Processor
   * A processor is a handle to a specific processor in the system.
   * It can be used to spawn tasks, query the processor's attributes,
   * and manage task execution on that processor.
   */
  class Processor {
  public:
    realm_processor_t id{REALM_NO_PROC};

    Processor() = default;
    constexpr explicit Processor(realm_processor_t id)
      : id(id)
    {}

    /**
     * \brief Implicit conversion operator to realm_processor_t.
     * \return The underlying realm_processor_t instance
     */
    constexpr operator realm_processor_t() const { return id; }

    bool operator<(const Processor &rhs) const { return id < rhs.id; }
    bool operator==(const Processor &rhs) const { return id == rhs.id; }
    bool operator!=(const Processor &rhs) const { return id != rhs.id; }

    static const Processor NO_PROC;

    /**
     * \brief Check whether this processor has a valid ID.
     * \return true if the processor has a valid ID, false otherwise
     */
    bool exists(void) const { return id != REALM_NO_PROC; }

    typedef ::realm_task_func_id_t TaskFuncID;
    typedef void (*TaskFuncPtr)(const void *args, size_t arglen, const void *user_data,
                                size_t user_data_len, Processor proc);

    // Different Processor types (defined in realm_c.h)
    // can't just typedef the kind because of C/C++ enum scope rules
    enum Kind
    {
#define C_ENUMS(name, desc) name,
      REALM_PROCESSOR_KINDS(C_ENUMS)
#undef C_ENUMS
    };

    /**
     * \brief Get the kind/type of this processor.
     * \return The processor kind (CPU, GPU, etc.)
     */
    Kind kind(void) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      uint64_t value;
      realm_processor_attr_t attr = REALM_PROCESSOR_ATTR_KIND;
      REALM_CHECK(realm_processor_get_attributes(runtime, id, &attr, &value, 1));
      return static_cast<Kind>(value);
    }

    /**
     * \brief Get the address space this processor belongs to.
     * \return The address space identifier for this processor
     */
    realm_address_space_t address_space(void) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      uint64_t value;
      realm_processor_attr_t attr = REALM_PROCESSOR_ATTR_ADDRESS_SPACE;
      REALM_CHECK(realm_processor_get_attributes(runtime, id, &attr, &value, 1));
      return static_cast<realm_address_space_t>(value);
    }

    /**
     * \brief Get the member processors of a processor group.
     * \param member_list Array to store the processor members
     * \param num_members Input: size of member_list array, Output: actual number of
     * members
     */
    void get_group_members(Processor *member_list, size_t &num_members) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get the member processors of a processor group.
     * \param member_list Vector to store the processor members
     */
    void get_group_members(std::vector<Processor> &member_list) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get the number of cores available on this processor.
     * \return Number of cores, or -1 if not applicable/available
     */
    int get_num_cores(void) const { throw std::logic_error("Not implemented"); }

    // special task IDs
    enum
    {
      // Save ID 0 for the force shutdown function
      TASK_ID_PROCESSOR_NOP = REALM_TASK_ID_PROCESSOR_NOP,
      TASK_ID_PROCESSOR_INIT = REALM_TASK_ID_PROCESSOR_INIT,
      TASK_ID_PROCESSOR_SHUTDOWN = REALM_TASK_ID_PROCESSOR_SHUTDOWN,
      TASK_ID_FIRST_AVAILABLE = REALM_TASK_ID_FIRST_AVAILABLE,
    };

    /**
     * \brief Spawn a task on this processor.
     * \param func_id The task function ID to execute
     * \param args Pointer to task arguments
     * \param arglen Length of the arguments in bytes
     * \param wait_on Event to wait for before starting the task
     * \param priority Task priority (higher values = higher priority)
     * \return Event that will be triggered when the task completes
     */
    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const Event &wait_on = Event::NO_EVENT, int priority = 0) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_event_t event_id_out;
      // Note: Using nullptr for ProfilingRequestSet since it's not available in C API yet
      REALM_CHECK(realm_processor_spawn(runtime, id, func_id, args, arglen, nullptr,
                                        wait_on, priority, &event_id_out));
      return Event(event_id_out);
    }

    /**
     * \brief Spawn a task on this processor with profiling requests.
     * \param func_id The task function ID to execute
     * \param args Pointer to task arguments
     * \param arglen Length of the arguments in bytes
     * \param requests Profiling requests for this task
     * \param wait_on Event to wait for before starting the task
     * \param priority Task priority (higher values = higher priority)
     * \return Event that will be triggered when the task completes
     */
    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const ProfilingRequestSet &requests,
                const Event &wait_on = Event::NO_EVENT, int priority = 0) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_event_t event_id_out;
      // Note: ProfilingRequestSet needs to be converted to realm_profiling_request_set_t
      // For now, using nullptr until the conversion is implemented
      REALM_CHECK(realm_processor_spawn(runtime, id, func_id, args, arglen, nullptr,
                                        wait_on, priority, &event_id_out));
      return Event(event_id_out);
    }

    /**
     * \brief Get the processor that is currently executing this code.
     * \return The processor executing the current task, or NO_PROC if not in a task
     */
    static Processor get_executing_processor(void)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Change the priority of the currently running task.
     * \param new_priority The new priority value (higher values = higher priority)
     */
    static void set_current_task_priority(int new_priority)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get the finish event for the currently running task.
     * \return Event that will be triggered when the current task completes
     */
    static Event get_current_finish_event(void)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Enable scheduler lock to prevent thread from releasing execution resources.
     *
     * A scheduler lock prevents the current thread from releasing its execution
     * resources even when waiting on an Event. Multiple nested calls to
     * enable_scheduler_lock are permitted, but a matching number of calls to
     * disable_scheduler_lock are required.
     */
    static void enable_scheduler_lock(void) { throw std::logic_error("Not implemented"); }

    /**
     * \brief Disable scheduler lock to allow thread to release execution resources.
     *
     * This must be called the same number of times as enable_scheduler_lock
     * to fully disable the scheduler lock.
     */
    static void disable_scheduler_lock(void)
    {
      throw std::logic_error("Not implemented");
    }

    // dynamic task registration - this may be done for:
    //  1) a specific processor/group (anywhere in the system)
    //  2) for all processors of a given type, either in the local address space/process,
    //       or globally
    //
    // in both cases, an Event is returned, and any tasks launched that expect to use the
    //  newly-registered task IDs must include that event as a precondition

    /**
     * \brief Register a task function on this specific processor.
     * \param func_id The task function ID to register
     * \param codedesc Code descriptor containing the task implementation
     * \param prs Profiling requests for the task
     * \param user_data Optional user data to associate with the task (default nullptr)
     * \param user_data_len Length of the user data in bytes (default 0)
     * \return Event that will be triggered when registration is complete
     */
    Event register_task(TaskFuncID func_id, const CodeDescriptor &codedesc,
                        const ProfilingRequestSet &prs, const void *user_data = 0,
                        size_t user_data_len = 0) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Register a task function on all processors of the specified kind.
     * \param target_kind The processor kind to register the task on
     * \param global Whether to register globally across all nodes (true) or locally
     * (false)
     * \param func_id The task function ID to register
     * \param codedesc Code descriptor containing the task implementation
     * \param prs Profiling requests for the task
     * \param user_data Optional user data to associate with the task (default nullptr)
     * \param user_data_len Length of the user data in bytes (default 0)
     * \return Event that will be triggered when registration is complete
     */
    static Event register_task_by_kind(Kind target_kind, bool global, TaskFuncID func_id,
                                       const CodeDescriptor &codedesc,
                                       const ProfilingRequestSet &prs,
                                       const void *user_data = 0,
                                       size_t user_data_len = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Report an execution fault in the currently running task.
     * \param reason The reason code for the execution fault
     * \param reason_data Pointer to additional data describing the fault
     * \param reason_size Size of the reason_data in bytes
     */
    static void report_execution_fault(int reason, const void *reason_data,
                                       size_t reason_size)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Report a fault with this processor (primarily for fault injection).
     * \param reason The reason code for the processor fault
     * \param reason_data Pointer to additional data describing the fault
     * \param reason_size Size of the reason_data in bytes
     */
    void report_processor_fault(int reason, const void *reason_data,
                                size_t reason_size) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get the name string for a processor kind.
     * \param kind The processor kind to get the name for
     * \return String name of the processor kind
     */
    static const char *get_kind_name(Kind kind)
    {
      throw std::logic_error("Not implemented");
    }

#ifdef REALM_USE_KOKKOS
    // Kokkos execution policies will accept an "execution instance" to
    //  capture task parallelism - provide those here
    class KokkosExecInstance;

    KokkosExecInstance kokkos_work_space(void) const;
#endif
  };

#if defined(REALM_USE_KOKKOS)
  // Kokkos defines this but we can't use it :(
  template <typename T>
  class is_kokkos_execution_space {
    typedef char yes;
    typedef long no;

    template <typename C>
    static yes check(typename C::execution_space *);
    template <typename C>
    static no check(...);

  public:
    static constexpr bool value = sizeof(check<T>(0)) == sizeof(yes);
  };

  class Processor::KokkosExecInstance {
  public:
    KokkosExecInstance(Processor _p);

    // template-fu will type-check a coercion to any Kokkos execution
    //  space type - runtime will verify a valid type was requested
    template <typename exec_space,
              typename std::enable_if<is_kokkos_execution_space<exec_space>::value,
                                      int>::type = 0>
    operator exec_space() const;

  protected:
    Processor p;
  };
#endif

  /**
   * \class Runtime
   * \brief Main interface to the Realm runtime system.
   *
   * The Runtime class provides the primary interface for initializing, configuring,
   * and managing the Realm runtime. It handles network initialization, task registration,
   * and overall runtime lifecycle management.
   */
  class Runtime {
  private:
    realm_runtime_t impl;

  public:
    /**
     * \brief Default constructor that creates a new runtime instance.
     */
    Runtime(void) { REALM_CHECK(realm_runtime_create(&impl)); }

    /**
     * \brief Copy constructor.
     * \param r The runtime instance to copy from
     */
    Runtime(const Runtime &r)
      : impl(r.impl)
    {}

    /**
     * \brief Assignment operator.
     * \param r The runtime instance to assign from
     * \return Reference to this runtime instance
     */
    Runtime &operator=(const Runtime &r)
    {
      impl = r.impl;
      return *this;
    }

    /**
     * \brief Implicit conversion operator to realm_runtime_t.
     * \return The underlying realm_runtime_t instance
     */
    constexpr operator realm_runtime_t() const { return impl; }

    /**
     * \brief Destructor that cleans up the runtime instance.
     */
    ~Runtime(void) { realm_runtime_destroy((impl)); }

    /**
     * \brief Get the current runtime instance.
     * \return The current runtime instance
     */
    static Runtime get_runtime(void)
    {
      Runtime r;
      REALM_CHECK(realm_runtime_get_runtime(&r.impl));
      return r;
    }

    /**
     * \brief Get the version string of the Realm library.
     *
     * Returns a valid (but possibly empty) string pointer describing the
     * version of the Realm library. This can be compared against
     * REALM_VERSION in application code to detect a header/library mismatch.
     *
     * \return Version string of the Realm library
     */
    static const char *get_library_version()
    {
      const char *version_string;
      REALM_CHECK(realm_get_library_version(&version_string));
      return version_string;
    }

    /**
     * \brief Perform network initialization and process command line arguments.
     *
     * Performs any network initialization and, critically, makes sure
     * *argc and *argv contain the application's real command line
     * (instead of e.g. mpi spawner information).
     *
     * \param argc Pointer to argument count (may be modified)
     * \param argv Pointer to argument array (may be modified)
     * \return True if initialization was successful
     */
    bool network_init(int *argc, char ***argv)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Parse command line arguments.
     * \param argc Number of arguments
     * \param argv Array of argument strings
     */
    void parse_command_line(int argc, char **argv)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Parse command line arguments from a vector.
     * \param cmdline Vector of command line arguments
     * \param remove_realm_args Whether to remove Realm-specific arguments (default false)
     */
    void parse_command_line(std::vector<std::string> &cmdline,
                            bool remove_realm_args = false)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Finish runtime configuration.
     */
    void finish_configure(void) { throw std::logic_error("Not implemented"); }

    /**
     * \brief Configure the runtime from command line arguments.
     *
     * After this call it is possible to create user events/reservations/etc,
     * perform registrations and query the machine model, but not spawn
     * tasks or create instances.
     *
     * \param argc Number of command line arguments
     * \param argv Array of command line arguments
     * \return True if configuration was successful
     */
    bool configure_from_command_line(int argc, char **argv)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Configure the runtime from command line arguments vector.
     *
     * After this call it is possible to create user events/reservations/etc,
     * perform registrations and query the machine model, but not spawn
     * tasks or create instances.
     *
     * \param cmdline Vector of command line arguments
     * \param remove_realm_args Whether to remove Realm-specific arguments (default false)
     * \return True if configuration was successful
     */
    bool configure_from_command_line(std::vector<std::string> &cmdline,
                                     bool remove_realm_args = false)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Start the runtime, allowing task and instance creation.
     */
    void start(void) { throw std::logic_error("Not implemented"); }

    /**
     * \brief Single-call initialization (combines network_init, configure, and start).
     * \param argc Pointer to argument count (may be modified)
     * \param argv Pointer to argument array (may be modified)
     * \return True if initialization was successful
     */
    bool init(int *argc, char ***argv)
    {
      return realm_runtime_init(impl, argc, argv) == REALM_SUCCESS;
    }

    /**
     * \brief Register a task function (deprecated - use Processor::register_task
     * instead).
     * \param taskid The task function ID
     * \param taskptr Pointer to the task function
     * \return True if registration was successful
     */
    bool register_task(realm_task_func_id_t taskid, Processor::TaskFuncPtr taskptr)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Register a reduction operation with event dependency.
     * \param event Event to wait for before registration
     * \param redop_id The reduction operation ID
     * \param redop Pointer to the reduction operation
     * \return True if registration was successful
     */
    bool register_reduction(const Event &event, realm_reduction_op_id_t redop_id,
                            const ReductionOpUntyped *redop)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Register a reduction operation immediately.
     * \param redop_id The reduction operation ID
     * \param redop Pointer to the reduction operation
     * \return True if registration was successful
     */
    bool register_reduction(realm_reduction_op_id_t redop_id,
                            const ReductionOpUntyped *redop)
    {
      return register_reduction(Event::NO_EVENT, redop_id, redop);
    }

    /**
     * \brief Register a reduction operation template immediately.
     * \tparam REDOP The reduction operation template type
     * \param redop_id The reduction operation ID
     * \return True if registration was successful
     */
    template <typename REDOP>
    bool register_reduction(realm_reduction_op_id_t redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(Event::NO_EVENT, redop_id, &redop);
    }

    /**
     * \brief Register a reduction operation template with event dependency.
     * \tparam REDOP The reduction operation template type
     * \param event Event to wait for before registration
     * \param redop_id The reduction operation ID
     * \return True if registration was successful
     */
    template <typename REDOP>
    bool register_reduction(Event &event, realm_reduction_op_id_t redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(event, redop_id, &redop);
    }

    /**
     * \brief Register a custom serialization/deserialization function.
     * \param serdez_id The custom serdez ID
     * \param serdez Pointer to the custom serdez implementation
     * \return True if registration was successful
     */
    bool register_custom_serdez(realm_custom_serdez_id_t serdez_id,
                                const CustomSerdezUntyped *serdez)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Register a custom serialization/deserialization template.
     * \tparam SERDEZ The custom serdez template type
     * \param serdez_id The custom serdez ID
     * \return True if registration was successful
     */
    template <typename SERDEZ>
    bool register_custom_serdez(realm_custom_serdez_id_t serdez_id)
    {
      const CustomSerdezWrapper<SERDEZ> serdez;
      return register_custom_serdez(serdez_id, &serdez);
    }

    /**
     * \brief Spawn a task collectively on the specified processor.
     * \param target_proc The processor to spawn the task on
     * \param task_id The task function ID to execute
     * \param args Pointer to task arguments
     * \param arglen Length of task arguments in bytes
     * \param wait_on Event to wait for before spawning (default NO_EVENT)
     * \param priority Task priority (default 0)
     * \return Event that will be triggered when the task completes
     */
    Event collective_spawn(Processor target_proc, realm_task_func_id_t task_id,
                           const void *args, size_t arglen,
                           const Event &wait_on = Event::NO_EVENT, int priority = 0)
    {
      realm_event_t eventIdOut;
      REALM_CHECK(realm_runtime_collective_spawn(impl, target_proc.id, task_id, args,
                                                 arglen, wait_on, priority, &eventIdOut));
      return Event(eventIdOut);
    }

    /**
     * \brief Spawn a task collectively on processors of the specified kind.
     * \param target_kind The kind of processor to spawn tasks on
     * \param task_id The task function ID to execute
     * \param args Pointer to task arguments
     * \param arglen Length of task arguments in bytes
     * \param one_per_node Whether to spawn only one task per node (default false)
     * \param wait_on Event to wait for before spawning (default NO_EVENT)
     * \param priority Task priority (default 0)
     * \return Event that will be triggered when all tasks complete
     */
    Event collective_spawn_by_kind(Processor::Kind target_kind,
                                   realm_task_func_id_t task_id, const void *args,
                                   size_t arglen, bool one_per_node = false,
                                   const Event &wait_on = Event::NO_EVENT,
                                   int priority = 0)
    {
      throw std::logic_error("Not implemented");
    }

    // there are three potentially interesting ways to start the initial
    // tasks:
    enum RunStyle
    {
      ONE_TASK_ONLY,     // a single task on a single node of the machine
      ONE_TASK_PER_NODE, // one task running on one proc of each node
      ONE_TASK_PER_PROC, // a task for every processor in the machine
    };

    /**
     * \brief Request a shutdown of the runtime.
     * \param wait_on Event to wait for before shutting down (default NO_EVENT)
     * \param result_code Exit code to return from wait_for_shutdown (default 0)
     */
    void shutdown(const Event &wait_on = Event::NO_EVENT, int result_code = 0)
    {
      REALM_CHECK(realm_runtime_signal_shutdown(impl, wait_on, result_code));
    }

    /**
     * \brief Wait for runtime shutdown and return the result code.
     * \return The result_code passed to shutdown()
     */
    int wait_for_shutdown(void)
    {
      REALM_CHECK(realm_runtime_wait_for_shutdown(impl));
      return 0;
    }

    /**
     * \brief Create module configurations (called before runtime::init).
     * \param argc Number of command line arguments
     * \param argv Array of command line arguments
     * \return True if configuration creation was successful
     */
    bool create_configs(int argc, char **argv)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get the configuration of a specific module.
     * \param name The name of the module
     * \return Pointer to the module configuration, or nullptr if not found
     */
    ModuleConfig *get_module_config(const std::string &name) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get access to a specific module by name and type.
     *
     * Modules in Realm may offer extra capabilities specific to certain kinds
     * of hardware or software. This function will return a null pointer if the
     * module isn't present or if the expected and actual types mismatch.
     *
     * \tparam T The expected C++ type of the module
     * \param name The name of the module
     * \return Pointer to the module casted to type T, or nullptr if not available
     */
    template <typename T>
    T *get_module(const char *name)
    {
      Module *mod = get_module_untyped(name);
      if(mod) {
        return dynamic_cast<T *>(mod);
      }
      return nullptr;
    }

  protected:
    Module *get_module_untyped(const char *name)
    {
      throw std::logic_error("Not implemented");
    }
  };

  /**
   * \class Memory
   * \brief Represents a memory region in the Realm runtime system.
   *
   * A memory is a handle to a specific memory region that can store data.
   * Different memory types have different characteristics in terms of capacity,
   * latency, and accessibility from different processors.
   */
  class Memory {
  public:
    realm_memory_t id{REALM_NO_MEM};

    Memory() = default;
    constexpr explicit Memory(realm_memory_t id)
      : id(id)
    {}

    constexpr operator realm_memory_t() const { return id; }

    bool operator<(const Memory &rhs) const { return id < rhs.id; }
    bool operator==(const Memory &rhs) const { return id == rhs.id; }
    bool operator!=(const Memory &rhs) const { return id != rhs.id; }

    static const Memory NO_MEMORY;

    /**
     * \brief Check whether this memory has a valid ID.
     * \return true if the memory has a valid ID, false otherwise
     */
    bool exists(void) const { return id != REALM_NO_MEM; }

    /**
     * \brief Get the address space this memory belongs to.
     * \return The address space identifier for this memory
     */
    realm_address_space_t address_space(void) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_ADDRESS_SPACE;
      REALM_CHECK(realm_memory_get_attributes(runtime, id, &attr, &value, 1));
      return static_cast<realm_address_space_t>(value);
    }

    // Different Memory types (defined in realm_c.h)
    // can't just typedef the kind because of C/C++ enum scope rules
    enum Kind
    {
#define C_ENUMS(name, desc) name,
      REALM_MEMORY_KINDS(C_ENUMS)
#undef C_ENUMS
    };

    /**
     * \brief Get the kind/type of this memory.
     * \return The memory kind (system RAM, GPU memory, etc.)
     */
    Kind kind(void) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_KIND;
      REALM_CHECK(realm_memory_get_attributes(runtime, id, &attr, &value, 1));
      return static_cast<Kind>(value);
    }

    /**
     * \brief Get the maximum capacity of this memory.
     * \return The maximum capacity in bytes
     */
    size_t capacity(void) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_CAPACITY;
      REALM_CHECK(realm_memory_get_attributes(runtime, id, &attr, &value, 1));
      return static_cast<size_t>(value);
    }

    /**
     * \brief Report a fault with this memory for fault injection testing.
     * \param reason The reason code for the fault
     * \param reason_data Additional data describing the fault
     * \param reason_size Size of the reason_data in bytes
     */
    void report_memory_fault(int reason, const void *reason_data,
                             size_t reason_size) const
    {
      throw std::logic_error("Not implemented");
    }
  };

  inline std::ostream &operator<<(std::ostream &os, Memory m)
  {
    return os << std::hex << m.id << std::dec;
  }

  inline std::ostream &operator<<(std::ostream &os, Memory::Kind kind)
  {
#define STRING_KIND_CASE(kind, desc)                                                     \
  case Memory::Kind::kind:                                                               \
    return os << #kind;
    switch(kind) {
      REALM_MEMORY_KINDS(STRING_KIND_CASE)
    }
#undef STRING_KIND_CASE
    return os << "UNKNOWN_KIND";
  }

  /**
   * \class Machine
   * \brief Provides access to machine topology and resource information.
   *
   * The Machine class allows querying the hardware topology, including
   * processors, memories, and their interconnection characteristics.
   * It provides both high-level query interfaces and detailed affinity
   * information for performance optimization.
   */
  class Machine {
  private:
    // Templated callback for collecting handles (Processor/Memory) into a std::set
    template <typename CHandleType>
    static realm_status_t collect_handle_into_set_cb(CHandleType handle, void *user_data)
    {
      std::set<CHandleType> &set = *static_cast<std::set<CHandleType> *>(user_data);
      set.insert(handle);
      return REALM_SUCCESS;
    }

    template <typename CHandleType>
    static realm_status_t collect_handle_into_vector_cb(CHandleType handle,
                                                        void *user_data)
    {
      std::vector<CHandleType> &vec = *static_cast<std::vector<CHandleType> *>(user_data);
      vec.push_back(handle);
      return REALM_SUCCESS;
    }

    realm_runtime_t impl;

  protected:
    friend class Runtime;
    explicit Machine(realm_runtime_t _impl)
      : impl(_impl)
    {}

  public:
    Machine(const Machine &m)
      : impl(m.impl)
    {}
    Machine &operator=(const Machine &m)
    {
      impl = m.impl;
      return *this;
    }
    ~Machine(void) {}

    static Machine get_machine(void)
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      return Machine(runtime);
    }

    class ProcessorQuery;
    class MemoryQuery;

    struct AffinityDetails {
      unsigned bandwidth;
      unsigned latency;
    };

    /**
     * \brief Check if a processor has affinity to a memory.
     * \param p The processor to check
     * \param m The memory to check affinity with
     * \param details Optional pointer to store detailed affinity information
     * \return True if affinity exists, false otherwise
     */
    bool has_affinity(Processor p, Memory m, AffinityDetails *details = 0) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Check if two memories have affinity to each other.
     * \param m1 The first memory
     * \param m2 The second memory
     * \param details Optional pointer to store detailed affinity information
     * \return True if affinity exists, false otherwise
     */
    bool has_affinity(Memory m1, Memory m2, AffinityDetails *details = 0) const
    {
      throw std::logic_error("Not implemented");
    }

    // older queries, to be deprecated

    /**
     * \brief Get all memories in the system.
     * \param mset Set to populate with all available memories
     */
    void get_all_memories(std::set<Memory> &mset) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_memory_query_t query;
      REALM_CHECK(realm_memory_query_create(runtime, &query));

      REALM_CHECK(realm_memory_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_memory_t>, &mset, SIZE_MAX));
      REALM_CHECK(realm_memory_query_destroy(query));
    }

    /**
     * \brief Get all processors in the system.
     * \param pset Set to populate with all available processors
     */
    void get_all_processors(std::set<Processor> &pset) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_processor_query_t query;
      REALM_CHECK(realm_processor_query_create(runtime, &query));

      REALM_CHECK(realm_processor_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_processor_t>, &pset,
          SIZE_MAX));
      REALM_CHECK(realm_processor_query_destroy(query));
    }

    /**
     * \brief Get all processors in the local address space.
     * \param pset Set to populate with local processors
     */
    void get_local_processors(std::set<Processor> &pset) const
    {
      realm_runtime_t runtime;
      realm_processor_query_t query;
      realm_address_space_t runtime_space;
      realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
      uint64_t values;

      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
      REALM_CHECK(realm_processor_query_create(runtime, &query));

      runtime_space = static_cast<realm_address_space_t>(values);

      // Restrict to local address space
      REALM_CHECK(realm_processor_query_restrict_to_address_space(query, runtime_space));

      REALM_CHECK(realm_processor_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_processor_t>, &pset,
          SIZE_MAX));
      REALM_CHECK(realm_processor_query_destroy(query));
    }

    /**
     * \brief Get all processors of a specific kind in the local address space.
     * \param pset Set to populate with matching processors
     * \param kind The processor kind to filter by (CPU, GPU, etc.)
     */
    void get_local_processors_by_kind(std::set<Processor> &pset,
                                      Processor::Kind kind) const
    {
      realm_runtime_t runtime;
      realm_processor_query_t query;
      realm_address_space_t runtime_space;
      realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
      uint64_t values;

      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
      REALM_CHECK(realm_processor_query_create(runtime, &query));

      runtime_space = static_cast<realm_address_space_t>(values);

      // Restrict to local address space and specific kind
      REALM_CHECK(realm_processor_query_restrict_to_kind(
          query, static_cast<realm_processor_kind_t>(kind)));
      REALM_CHECK(realm_processor_query_restrict_to_address_space(query, runtime_space));

      REALM_CHECK(realm_processor_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_processor_t>, &pset,
          SIZE_MAX));
      REALM_CHECK(realm_processor_query_destroy(query));
    }

    /**
     * \brief Return the set of memories visible from a processor.
     * \param p The processor to query memory visibility from
     * \param mset Set to populate with visible memories
     * \param local_only Whether to restrict to local address space (default true)
     */
    void get_visible_memories(Processor p, std::set<Memory> &mset,
                              bool local_only = true) const
    {
      throw std::logic_error("Not implemented");
#if 0
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_memory_query_t query;
      REALM_CHECK(realm_memory_query_create(runtime, &query));

      if(local_only) {
        realm_address_space_t runtime_space;
        realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
        uint64_t values;
        REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
        runtime_space = static_cast<realm_address_space_t>(values);
        REALM_CHECK(realm_memory_query_restrict_to_address_space(query, runtime_space));
      }

      // TODO: Need C API to restrict memories by processor affinity
      // For now, just get all memories

      REALM_CHECK(realm_memory_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_memory_t>, &mset, SIZE_MAX));
      REALM_CHECK(realm_memory_query_destroy(query));
#endif
    }

    /**
     * \brief Get the set of memories visible from a memory.
     * \param m The memory to query visibility from
     * \param mset Set to populate with visible memories
     * \param local_only Whether to restrict to local address space (default true)
     */
    void get_visible_memories(Memory m, std::set<Memory> &mset,
                              bool local_only = true) const
    {
      throw std::logic_error("Not implemented");
#if 0
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_memory_query_t query;
      REALM_CHECK(realm_memory_query_create(runtime, &query));

      if(local_only) {
        realm_address_space_t runtime_space;
        realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
        uint64_t values;
        REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
        runtime_space = static_cast<realm_address_space_t>(values);
        REALM_CHECK(realm_memory_query_restrict_to_address_space(query, runtime_space));
      }

      // TODO: Need C API to restrict memories by memory affinity
      // For now, just get all memories

      REALM_CHECK(realm_memory_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_memory_t>, &mset, SIZE_MAX));
      REALM_CHECK(realm_memory_query_destroy(query));
#endif
    }

    /**
     * \brief Get the set of processors which can all see a given memory.
     * \param m The memory to query processor access for
     * \param pset Set to populate with processors that can access the memory
     * \param local_only Whether to restrict to local address space (default true)
     */
    void get_shared_processors(Memory m, std::set<Processor> &pset,
                               bool local_only = true) const
    {
      throw std::logic_error("Not implemented");
#if 0
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_processor_query_t query;
      REALM_CHECK(realm_processor_query_create(runtime, &query));

      if(local_only) {
        realm_address_space_t runtime_space;
        realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
        uint64_t values;
        REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
        runtime_space = static_cast<realm_address_space_t>(values);
        REALM_CHECK(
            realm_processor_query_restrict_to_address_space(query, runtime_space));
      }

      // TODO: Need C API to restrict processors by memory affinity
      // For now, just get all processors

      REALM_CHECK(realm_processor_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_processor_t>, &pset,
          SIZE_MAX));
      REALM_CHECK(realm_processor_query_destroy(query));
#endif
    }

    /**
     * \brief Get memories with at least the specified capacity.
     * \param min_capacity The minimum capacity in bytes
     * \param mset The set to populate with memories that meet the capacity requirement
     */
    void get_memories_by_capacity(size_t min_capacity, std::set<Memory> &mset) const
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_memory_query_t query;
      REALM_CHECK(realm_memory_query_create(runtime, &query));

      // Restrict by capacity
      REALM_CHECK(realm_memory_query_restrict_by_capacity(query, min_capacity));

      REALM_CHECK(realm_memory_query_iter(
          query, &Machine::collect_handle_into_set_cb<realm_memory_t>, &mset, SIZE_MAX));
      REALM_CHECK(realm_memory_query_destroy(query));
    }

    /**
     * \brief Get the number of address spaces in the machine.
     * \return The total number of address spaces
     */
    size_t get_address_space_count(void) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Information about the OS process for a processor.
     *
     * Note that the uniqueness of any/all of the provided information depends
     * on the underlying OS and any container runtimes.
     */
    struct ProcessInfo {
      static const size_t MAX_HOSTNAME_LENGTH = 256;
      char hostname[MAX_HOSTNAME_LENGTH]; // always null-terminated
      uint64_t hostid; // gethostid on posix, hash of hostname on windows
      uint32_t processid;
    };

    /**
     * \brief Get information about the OS process containing a processor.
     * \param p The processor to query
     * \param info Pointer to ProcessInfo struct to populate
     * \return True if successful, false if processor is unknown or info unavailable
     */
    bool get_process_info(Processor p, ProcessInfo *info) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Processor-Memory affinity information.
     */
    struct ProcessorMemoryAffinity {
      Processor p;        // accessing processor
      Memory m;           // target memory
      unsigned bandwidth; // in MB/s
      unsigned latency;   // in nanoseconds
    };

    /**
     * \brief Memory-Memory affinity information.
     */
    struct MemoryMemoryAffinity {
      Memory m1;          // source memory
      Memory m2;          // destination memory
      unsigned bandwidth; // in MB/s
      unsigned latency;   // in nanoseconds
    };

    /**
     * \brief Get processor-memory affinity information.
     * \param result Vector to populate with affinity information
     * \param restrict_proc Restrict to specific processor (default NO_PROC for all)
     * \param restrict_memory Restrict to specific memory (default NO_MEMORY for all)
     * \param local_only Whether to restrict to local address space (default true)
     * \return Number of affinity records found
     */
    int get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity> &result,
                              Processor restrict_proc = Processor::NO_PROC,
                              Memory restrict_memory = Memory::NO_MEMORY,
                              bool local_only = true) const
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Get memory-memory affinity information.
     * \param result Vector to populate with affinity information
     * \param restrict_mem1 Restrict to specific source memory (default NO_MEMORY for all)
     * \param restrict_mem2 Restrict to specific destination memory (default NO_MEMORY for
     * all)
     * \param local_only Whether to restrict to local address space (default true)
     * \return Number of affinity records found
     */
    int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity> &result,
                             Memory restrict_mem1 = Memory::NO_MEMORY,
                             Memory restrict_mem2 = Memory::NO_MEMORY,
                             bool local_only = true) const
    {
      throw std::logic_error("Not implemented");
    }
    // subscription interface for dynamic machine updates
    class MachineUpdateSubscriber {
    public:
      virtual ~MachineUpdateSubscriber(void) {}

      enum UpdateType
      {
        THING_ADDED,
        THING_REMOVED,
        THING_UPDATED
      };

      // callbacks occur on a thread that belongs to the runtime - please defer any
      //  complicated processing if possible
      virtual void processor_updated(Processor p, UpdateType update_type,
                                     const void *payload, size_t payload_size) = 0;

      virtual void memory_updated(Memory m, UpdateType update_type, const void *payload,
                                  size_t payload_size) = 0;
    };

    // subscriptions are encouraged to use a query which filters which processors or
    //  memories cause notifications
    void add_subscription(MachineUpdateSubscriber *subscriber);
    void add_subscription(MachineUpdateSubscriber *subscriber,
                          const ProcessorQuery &query);
    void add_subscription(MachineUpdateSubscriber *subscriber, const MemoryQuery &query);

    void remove_subscription(MachineUpdateSubscriber *subscriber);
  };

  template <typename QT, typename RT>
  class MachineQueryIterator {
  public:
    // explicitly set iterator traits
    typedef std::input_iterator_tag iterator_category;
    typedef RT value_type;
    typedef std::ptrdiff_t difference_type;
    typedef RT *pointer;
    typedef RT &reference;

    // would like this constructor to be protected and have QT be a friend.
    //  The CUDA compiler also seems to be a little dense here as well
#if(!defined(__CUDACC__) && !defined(__HIPCC__))
  protected:
    friend QT;
#else
  public:
#endif
    MachineQueryIterator(const QT &query_ptr, RT _result)
      : query_ptr(query_ptr)
      , result(_result)
    {}

  private:
    QT query_ptr;
    RT result;

  public:
    MachineQueryIterator(const MachineQueryIterator<QT, RT> &copy_from)
      : query_ptr(copy_from.query_ptr)
      , result(copy_from.result)
    {}

    ~MachineQueryIterator(void) {}

    MachineQueryIterator<QT, RT> &operator=(const MachineQueryIterator<QT, RT> &copy_from)
    {
      query_ptr = copy_from.query_ptr;
      result = copy_from.result;
      return *this;
    }

    bool operator==(const MachineQueryIterator<QT, RT> &compare_to) const
    {
      return (query_ptr == compare_to.query_ptr) && (result == compare_to.result);
    }

    bool operator!=(const MachineQueryIterator<QT, RT> &compare_to) const
    {
      return !(*this == compare_to);
    }

    RT operator*(void) { return result; }

    const RT *operator->(void) { return &result; }

    MachineQueryIterator<QT, RT> &operator++(/*prefix*/)
    {
      result = query_ptr.next(result);
      return *this;
    }

    MachineQueryIterator<QT, RT> operator++(int /*postfix*/)
    {
      MachineQueryIterator<QT, RT> tmp(*this);
      ++(*this);
      return tmp;
    }

    operator bool(void) const { return result != RT(); }
  };

  /**
   * \class ProcessorQuery
   * \brief A fluent interface for querying processors with filtering criteria.
   *
   * This class provides a chainable interface for building complex processor queries.
   * Multiple filter predicates can be chained together, and the intersection of
   * all matching criteria is returned.
   */
  class Machine::ProcessorQuery {
  private:
    std::shared_ptr<realm_processor_query_st> impl = {nullptr};

  public:
    /**
     * \brief Construct a processor query for the given machine.
     * \param m The machine to query processors from
     */
    explicit ProcessorQuery(const Machine &m)
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_processor_query_t query_handle;
      REALM_CHECK(realm_processor_query_create(runtime, &query_handle));
      impl = std::shared_ptr<realm_processor_query_st>(query_handle,
                                                       realm_processor_query_destroy);
    }

    /**
     * \brief Copy constructor.
     * \param q The query to copy from
     */
    ProcessorQuery(const ProcessorQuery &q)
      : impl(q.impl)
    {}

    /**
     * \brief Copy constructor from a raw query handle.
     * \param q The raw query handle to copy from
     */
    ProcessorQuery(realm_processor_query_t q)
      : impl(std::shared_ptr<realm_processor_query_st>(q, realm_processor_query_destroy))
    {}

    /**
     * \brief Destructor that cleans up the query resources.
     */
    ~ProcessorQuery(void) {
    } // shared_ptr automatically handles cleanup when reference count reaches 0

    ProcessorQuery &operator=(const ProcessorQuery &q)
    {
      impl = q.impl;
      pset.clear(); // Invalidate cache after assignment
      return *this;
    }

    bool operator==(const ProcessorQuery &compare_to) const
    {
      return impl == compare_to.impl;
    }

    bool operator!=(const ProcessorQuery &compare_to) const
    {
      return !(*this == compare_to);
    }

    // implicit conversion to raw query handle
    operator realm_processor_query_t(void) const { return impl.get(); }

    // filter predicates (returns self-reference for chaining)
    // if multiple predicates are used, they must all match (i.e. the intersection is
    // returned)

    /**
     * \brief Restrict query to processors of the specified kind.
     * \param kind The processor kind to filter by (CPU, GPU, etc.)
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &only_kind(Processor::Kind kind)
    {
      if(impl != nullptr) {
        REALM_CHECK(realm_processor_query_restrict_to_kind(
            impl.get(), static_cast<realm_processor_kind_t>(kind)));
      }
      return *this;
    }

    /**
     * \brief Restrict query to processors in the local address space.
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &local_address_space(void)
    {
      realm_runtime_t runtime;
      realm_processor_query_t query;
      realm_address_space_t runtime_space;
      realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
      uint64_t values;

      if(impl != nullptr) {
        REALM_CHECK(realm_runtime_get_runtime(&runtime));
        REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
        REALM_CHECK(realm_processor_query_create(runtime, &query));

        runtime_space = static_cast<realm_address_space_t>(values);
        REALM_CHECK(
            realm_processor_query_restrict_to_address_space(impl.get(), runtime_space));
      }
      return *this;
    }

    /**
     * \brief Restrict query to processors in the same address space as the specified
     * processor.
     * \param p The processor whose address space to match
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &same_address_space_as(Processor p)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to processors in the same address space as the specified
     * memory.
     * \param m The memory whose address space to match
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &same_address_space_as(Memory m)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to processors that have affinity to the specified memory.
     * \param m The memory to check affinity with
     * \param min_bandwidth Minimum required bandwidth (MB/s)
     * \param max_latency Maximum allowed latency (nanoseconds)
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &has_affinity_to(Memory m, unsigned min_bandwidth = 0,
                                    unsigned max_latency = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to processors with best affinity to the specified memory.
     * \param m The memory to optimize affinity for
     * \param bandwidth_weight Weight given to bandwidth in affinity calculation
     * \param latency_weight Weight given to latency in affinity calculation
     * \return Reference to this query for method chaining
     */
    ProcessorQuery &best_affinity_to(Memory m, int bandwidth_weight = 1,
                                     int latency_weight = 0)
    {
      throw std::logic_error("Not implemented");
    }

    // results - a query may be executed multiple times - when the machine model is
    //  dynamic, there is no guarantee that the results of any two executions will be
    //  consistent

    /**
     * \brief Get the number of processors matching the query criteria.
     * \return Number of matching processors
     */
    size_t count(void) const
    {
      ensure_result_set();
      return pset.size();
    }

    /**
     * \brief Get the first processor matching the query criteria.
     * \return First matching processor, or NO_PROC if none match
     */
    Processor first(void) const
    {
      ensure_result_set();
      return pset.empty() ? Processor(REALM_NO_PROC) : Processor(pset[0]);
    }

    /**
     * \brief Get the next processor after the specified one that matches the criteria.
     * \param after The processor to search after
     * \return Next matching processor, or NO_PROC if none found
     */
    Processor next(Processor after) const
    {
      ensure_result_set();
      auto it =
          std::find(pset.begin(), pset.end(), static_cast<realm_processor_t>(after));
      if(it != pset.end()) {
        ++it;
        if(it != pset.end()) {
          return Processor(*it);
        }
      }
      return Processor(REALM_NO_PROC);
    }

    /**
     * \brief Get a random processor matching the query criteria.
     * \return Random matching processor, or NO_PROC if none match
     */
    Processor random(void) const
    {
      ensure_result_set();
      if(pset.empty()) {
        return Processor(REALM_NO_PROC);
      }
      return Processor(pset[std::rand() % pset.size()]);
    }

    typedef MachineQueryIterator<ProcessorQuery, Processor> iterator;

    /**
     * \brief Get an iterator to enumerate all matching processors.
     * \return Iterator pointing to the first matching processor
     */
    iterator begin() const
    {
      ensure_result_set();
      return iterator(*this, first());
    }

    /**
     * \brief Get an iterator representing the end of the query results.
     * \return End iterator
     */
    iterator end() const
    {
      ensure_result_set();
      return iterator(*this, Processor(REALM_NO_PROC));
    }

  private:
    void ensure_result_set() const
    {
      if(pset.empty() && impl != nullptr) {
        REALM_CHECK(realm_processor_query_iter(
            impl.get(), &Machine::collect_handle_into_vector_cb<realm_processor_t>, &pset,
            SIZE_MAX));
      }
    }

    // 'pset' is marked mutable because it is modified by ensure_result_set(), which is
    // called from const member functions to allow lazy evaluation/caching.
    mutable std::vector<realm_processor_t> pset;
  };

  /**
   * \brief Query class for finding memories matching specific criteria.
   *
   * MemoryQuery allows filtering memories based on characteristics like kind,
   * capacity, and affinity relationships with processors.
   */
  class Machine::MemoryQuery {
  private:
    std::shared_ptr<realm_memory_query_st> impl = {nullptr};

  public:
    /**
     * \brief Construct a query for all memories in the machine.
     * \param m The machine to query
     */
    explicit MemoryQuery(const Machine &m)
    {
      realm_runtime_t runtime;
      REALM_CHECK(realm_runtime_get_runtime(&runtime));
      realm_memory_query_t query_handle;
      REALM_CHECK(realm_memory_query_create(runtime, &query_handle));
      impl = std::shared_ptr<realm_memory_query_st>(query_handle,
                                                    realm_memory_query_destroy);
    }

    /**
     * \brief Copy constructor.
     * \param q The query to copy
     */
    MemoryQuery(const MemoryQuery &q)
      : impl(q.impl)
    {}

    /**
     * \brief Copy constructor from a raw query handle.
     * \param q The raw query handle to copy from
     */
    MemoryQuery(realm_memory_query_t q)
      : impl(std::shared_ptr<realm_memory_query_st>(q, realm_memory_query_destroy))
    {}

    /**
     * \brief Destructor.
     */
    ~MemoryQuery(void) {
    } // shared_ptr automatically handles cleanup when reference count reaches 0

    /**
     * \brief Assignment operator.
     * \param q The query to copy
     * \return Reference to this query
     */
    MemoryQuery &operator=(const MemoryQuery &q)
    {
      impl = q.impl;
      mset.clear(); // Invalidate cache after assignment
      return *this;
    }

    /**
     * \brief Equality comparison operator.
     * \param compare_to The query to compare against
     * \return True if queries are equal
     */
    bool operator==(const MemoryQuery &compare_to) const
    {
      return impl == compare_to.impl;
    }

    /**
     * \brief Inequality comparison operator.
     * \param compare_to The query to compare against
     * \return True if queries are not equal
     */
    bool operator!=(const MemoryQuery &compare_to) const
    {
      return !(*this == compare_to);
    }

    // implicit conversion to raw query handle
    operator realm_memory_query_t(void) const { return impl.get(); }

    // filter predicates (returns self-reference for chaining)
    // if multiple predicates are used, they must all match (i.e. the intersection is
    // returned)

    /**
     * \brief Restrict query to memories of the specified kind.
     * \param kind The memory kind to filter by (SYSMEM, FBMEM, etc.)
     * \return Reference to this query for method chaining
     */
    MemoryQuery &only_kind(Memory::Kind kind)
    {
      if(impl != nullptr) {
        REALM_CHECK(realm_memory_query_restrict_to_kind(
            impl.get(), static_cast<realm_memory_kind_t>(kind)));
      }
      return *this;
    }

    /**
     * \brief Restrict query to memories in the local address space.
     * \return Reference to this query for method chaining
     */
    MemoryQuery &local_address_space(void)
    {
      realm_runtime_t runtime;
      realm_memory_query_t query;
      realm_address_space_t runtime_space;
      realm_runtime_attr_t runtime_attr = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
      uint64_t values;

      if(impl != nullptr) {
        REALM_CHECK(realm_runtime_get_runtime(&runtime));
        REALM_CHECK(realm_runtime_get_attributes(runtime, &runtime_attr, &values, 1));
        REALM_CHECK(realm_memory_query_create(runtime, &query));

        runtime_space = static_cast<realm_address_space_t>(values);
        REALM_CHECK(
            realm_memory_query_restrict_to_address_space(impl.get(), runtime_space));
      }
      return *this;
    }

    /**
     * \brief Restrict query to memories in the same address space as the specified
     * processor.
     * \param p The processor whose address space to match
     * \return Reference to this query for method chaining
     */
    MemoryQuery &same_address_space_as(Processor p)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories in the same address space as the specified
     * memory.
     * \param m The memory whose address space to match
     * \return Reference to this query for method chaining
     */
    MemoryQuery &same_address_space_as(Memory m)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories that have affinity to the specified processor.
     * \param p The processor to check affinity with
     * \param min_bandwidth Minimum bandwidth requirement (default 0)
     * \param max_latency Maximum latency requirement (default 0)
     * \return Reference to this query for method chaining
     */
    MemoryQuery &has_affinity_to(Processor p, unsigned min_bandwidth = 0,
                                 unsigned max_latency = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories that have affinity to the specified memory.
     * \param m The memory to check affinity with
     * \param min_bandwidth Minimum bandwidth requirement (default 0)
     * \param max_latency Maximum latency requirement (default 0)
     * \return Reference to this query for method chaining
     */
    MemoryQuery &has_affinity_to(Memory m, unsigned min_bandwidth = 0,
                                 unsigned max_latency = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories with the best affinity to the specified
     * processor.
     * \param p The processor to find best affinity for
     * \param bandwidth_weight Weight for bandwidth in affinity calculation (default 1)
     * \param latency_weight Weight for latency in affinity calculation (default 0)
     * \return Reference to this query for method chaining
     */
    MemoryQuery &best_affinity_to(Processor p, int bandwidth_weight = 1,
                                  int latency_weight = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories with the best affinity to the specified memory.
     * \param m The memory to find best affinity for
     * \param bandwidth_weight Weight for bandwidth in affinity calculation (default 1)
     * \param latency_weight Weight for latency in affinity calculation (default 0)
     * \return Reference to this query for method chaining
     */
    MemoryQuery &best_affinity_to(Memory m, int bandwidth_weight = 1,
                                  int latency_weight = 0)
    {
      throw std::logic_error("Not implemented");
    }

    /**
     * \brief Restrict query to memories with at least the specified total capacity.
     * \param min_bytes Minimum capacity in bytes
     * \return Reference to this query for method chaining
     */
    MemoryQuery &has_capacity(size_t min_bytes)
    {
      if(impl != nullptr) {
        REALM_CHECK(realm_memory_query_restrict_by_capacity(impl.get(), min_bytes));
      }
      return *this;
    }

    // results - a query may be executed multiple times - when the machine model is
    //  dynamic, there is no guarantee that the results of any two executions will be
    //  consistent

    /**
     * \brief Get the number of memories matching the query criteria.
     * \return Number of matching memories
     */
    size_t count(void) const
    {
      ensure_result_set();
      return mset.size();
    }

    /**
     * \brief Get the first memory matching the query criteria.
     * \return First matching memory, or NO_MEMORY if none match
     */
    Memory first(void) const
    {
      ensure_result_set();
      return mset.empty() ? Memory(REALM_NO_MEM) : Memory(mset[0]);
    }

    /**
     * \brief Get the next memory after the specified one that matches the criteria.
     * \param after The memory to search after
     * \return Next matching memory, or NO_MEMORY if none found
     */
    Memory next(Memory after) const
    {
      ensure_result_set();
      auto it = std::find(mset.begin(), mset.end(), static_cast<realm_memory_t>(after));
      if(it != mset.end()) {
        ++it;
        if(it != mset.end()) {
          return Memory(*it);
        }
      }
      return Memory(REALM_NO_MEM);
    }

    /**
     * \brief Get a random memory matching the query criteria.
     * \return Random matching memory, or NO_MEMORY if none match
     */
    Memory random(void) const
    {
      ensure_result_set();
      if(mset.empty()) {
        return Memory(REALM_NO_MEM);
      }
      return Memory(mset[std::rand() % mset.size()]);
    }

    typedef MachineQueryIterator<MemoryQuery, Memory> iterator;

    /**
     * \brief Get an iterator to enumerate all matching memories.
     * \return Iterator pointing to the first matching memory
     */
    iterator begin() const
    {
      ensure_result_set();
      return iterator(*this, first());
    }

    /**
     * \brief Get an iterator representing the end of the query results.
     * \return End iterator
     */
    iterator end() const
    {
      ensure_result_set();
      return iterator(*this, Memory(REALM_NO_MEM));
    }

  private:
    void ensure_result_set() const
    {
      if(mset.empty() && impl != nullptr) {
        REALM_CHECK(realm_memory_query_iter(
            impl.get(), &Machine::collect_handle_into_vector_cb<realm_memory_t>, &mset,
            SIZE_MAX));
      }
    }

    // 'mset' is marked mutable because it is modified by ensure_result_set(), which is
    // called from const member functions to allow lazy evaluation/caching.
    mutable std::vector<realm_memory_t> mset;
  };

  inline const Processor Processor::NO_PROC{REALM_NO_PROC};
  inline const Event Event::NO_EVENT{REALM_NO_EVENT};
  inline const UserEvent UserEvent::NO_USER_EVENT{REALM_NO_USER_EVENT};
  inline const Memory Memory::NO_MEMORY{REALM_NO_MEM};
} // namespace REALM_NAMESPACE

#define REALM_HASH_DEFINE(C)                                                             \
  template <>                                                                            \
  struct std::hash<C> {                                                                  \
    std::size_t operator()(const C &c) const noexcept                                    \
    {                                                                                    \
      return std::hash<realm_id_t>()(c);                                                 \
    }                                                                                    \
  }

REALM_HASH_DEFINE(REALM_NAMESPACE::Memory);
REALM_HASH_DEFINE(REALM_NAMESPACE::Processor);
REALM_HASH_DEFINE(REALM_NAMESPACE::Event);
REALM_HASH_DEFINE(REALM_NAMESPACE::UserEvent);

#undef REALM_HASH_DEFINE

#endif // REALM_HPP
