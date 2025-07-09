/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <cstddef>

#include "realm/realm_c.h"

namespace Realm {
  // Type definitions
  typedef ::realm_reduction_op_id_t ReductionOpID;
  typedef ::realm_custom_serdez_id_t CustomSerdezID;
  typedef ::realm_address_space_t AddressSpace;

  // Forward declarations
  class ProfilingRequestSet;
  class CodeDescriptor;
  class ReductionOpUntyped;
  class CustomSerdezUntyped;
  class ModuleConfig;
  class Module;
  class InstanceLayoutGeneric;
  class ExternalInstanceResource;

  // Template forward declarations
  template <typename T>
  class ReductionOp;
  template <typename T>
  class CustomSerdezWrapper;
  template <int N, typename T>
  class Rect;
  template <int N, typename T>
  class IndexSpace;

  // Type aliases
  typedef ::realm_task_func_id_t TaskFuncID;
  typedef ::realm_field_id_t FieldID;

  // Forward declarations for C types
  class IndexSpaceGeneric;

  // TODO: actually use C++20 version if available
  const size_t dynamic_extent_foo = size_t(-1);

  template <typename T, size_t Extent = dynamic_extent_foo>
  class notStdSpan {
  public:
    notStdSpan()
      : data_(nullptr)
      , size_(0)
    {}
    notStdSpan(T *data, size_t size)
      : data_(data)
      , size_(size)
    {}
    notStdSpan(const std::vector<T> &v)
      : data_(v.data())
      , size_(v.size())
    {}

    T *data() const { return data_; }
    size_t size() const { return size_; }

  private:
    T *data_;
    size_t size_;
  };

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
  class REALM_PUBLIC_API Event {
  public:
    typedef ::realm_id_t id_t;

    id_t id{REALM_NO_EVENT};

    Event() = default;
    constexpr explicit Event(id_t id)
      : id(id)
    {}

    constexpr operator id_t() const { return id; }

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
      // TODO: Implement with C API call when available
      // For now, return true as stub implementation
      return true;
    }

    /**
     * Wait for an event to trigger.
     */
    void wait(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_event_wait(runtime, id);
    }

    /**
     * Used by non-legion threads to wait on an event - always blocking
     */
    void external_wait(void) const
    {
      // Same as wait for now
      wait();
    }

    /**
     * External wait with a timeout - returns true if event triggers, false
     * if the maximum delay occurs first
     * \param max_ns the maximum number of nanoseconds to wait
     * \return true if the event has triggered, false if the timeout occurred
     */
    bool external_timedwait(long long max_ns) const
    {
      // TODO: Implement with C API call when available
      // For now, return true as stub implementation
      (void)max_ns;
      return true;
    }

    /**
     * Fault-aware versions of the above (the above versions will cause the
     * caller to fault as well if a poisoned event is queried).
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    bool has_triggered_faultaware(bool &poisoned) const
    {
      // Stub implementation
      poisoned = false;
      return has_triggered();
    }

    /**
     * Fault-aware versions of the wait function.
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    void wait_faultaware(bool &poisoned) const
    {
      // Stub implementation
      poisoned = false;
      wait();
    }

    /**
     * Fault-aware versions of the external wait function.
     * \param poisoned set to true if the event is poisoned
     * \return true if the event has triggered, false otherwise
     */
    void external_wait_faultaware(bool &poisoned) const
    {
      // Stub implementation
      poisoned = false;
      external_wait();
    }

    /**
     * Fault-aware versions of the external timed wait function.
     * \param poisoned set to true if the event is poisoned
     * \param max_ns the maximum number of nanoseconds to wait
     * \return true if the event has triggered, false if the timeout occurred
     */
    bool external_timedwait_faultaware(bool &poisoned, long long max_ns) const
    {
      // Stub implementation
      poisoned = false;
      return external_timedwait(max_ns);
    }

    /**
     * Subscribe to an event, ensuring that the triggeredness of the
     * event will be available as soon as possible (and without having to call
     * wait).
     */
    void subscribe(void) const
    {
      // Stub implementation
    }

    /**
     * Attempt to cancel the operation associated with this event.
     * \param reason_data will be provided to any profilers of the operation
     * \param reason_size the size of the reason data
     */
    void cancel_operation(const void *reason_data, size_t reason_size) const
    {
      // Stub implementation
    }

    /**
     * Attempt to change the priority of the operation associated with this
     * event.
     * \param new_priority the new priority.
     */
    void set_operation_priority(int new_priority) const
    {
      // Stub implementation
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
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      Event merged_event;
      realm_event_merge(runtime, reinterpret_cast<const realm_event_t *>(wait_for),
                        num_events, reinterpret_cast<realm_event_t *>(&merged_event));
      return merged_event;
    }
    static Event merge_events(Event ev1, Event ev2, Event ev3 = NO_EVENT,
                              Event ev4 = NO_EVENT, Event ev5 = NO_EVENT,
                              Event ev6 = NO_EVENT)
    {
      static constexpr size_t MAX_EVENTS = 6;
      Event events[] = {ev1, ev2, ev3, ev4, ev5, ev6};
      size_t count = 0;
      for(size_t i = 0; i < MAX_EVENTS; i++) {
        if(events[i] != NO_EVENT) {
          count = i + 1;
        }
      }
      return merge_events(events, count);
    }
    static Event merge_events(const std::set<Event> &wait_for)
    {
      std::vector<Event> events(wait_for.begin(), wait_for.end());
      return merge_events(events.data(), events.size());
    }
    static Event merge_events(const notStdSpan<const Event> &wait_for)
    {
      return merge_events(wait_for.data(), wait_for.size());
    }
    ///@}

    /**
     * Create an event that won't trigger until all input events
     * have, ignoring any poison on the input events.
     * \param wait_for the events to wait for
     * \return the event that will trigger when all input events
     * have.
     */
    static Event merge_events_ignorefaults(const Event *wait_for, size_t num_events)
    {
      // Stub implementation - same as merge_events for now
      return merge_events(wait_for, num_events);
    }
    static Event merge_events_ignorefaults(const notStdSpan<const Event> &wait_for)
    {
      return merge_events_ignorefaults(wait_for.data(), wait_for.size());
    }
    static Event merge_events_ignorefaults(const std::set<Event> &wait_for)
    {
      std::vector<Event> events(wait_for.begin(), wait_for.end());
      return merge_events_ignorefaults(events.data(), events.size());
    }
    static Event ignorefaults(Event wait_for)
    {
      // Stub implementation
      return wait_for;
    }

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
      // Stub implementation
    }
    static void advise_event_ordering(const Event *happens_before, size_t num_events,
                                      Event happens_after, bool all_must_trigger = true)
    {
      // Stub implementation
    }
    static void advise_event_ordering(const notStdSpan<Event> &happens_before,
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
  class REALM_PUBLIC_API UserEvent : public Event {
  public:
    UserEvent() = default;
    constexpr UserEvent(id_t id)
      : Event(id)
    {}

    /**
     * Create a new user event.
     * \return the new user event
     */
    static UserEvent create_user_event(void)
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      UserEvent event;
      realm_user_event_create(runtime, reinterpret_cast<realm_user_event_t *>(&event));
      return event;
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
      realm_runtime_get_runtime(&runtime);
      realm_user_event_trigger(runtime, id);
    }

    /*
     * Attempt to cancell all the operations waiting on this user
     * event.
     */
    void cancel(void) const
    {
      // TODO: Implement with C API call when available
      // Note: No C API function available for canceling user events yet
    }

    static const UserEvent NO_USER_EVENT;
  };

  /**
   * \class Barrier
   * A barrier is similar to a user event, except that it has a count
   * of how many threads (or whatever) need to "trigger" before the
   * actual trigger occurs.
   */
  class REALM_PUBLIC_API Barrier : public Event {
  public:
    typedef ::realm_barrier_timestamp_t
        timestamp_t; // used to avoid race conditions with arrival adjustments

    timestamp_t timestamp;

    Barrier() = default;
    constexpr explicit Barrier(id_t id)
      : Event(id)
      , timestamp(0)
    {}

    static const Barrier NO_BARRIER;

    static Barrier create_barrier(unsigned expected_arrivals, ReductionOpID redop_id = 0,
                                  const void *initial_value = 0,
                                  size_t initial_value_size = 0)
    {
      // TODO: Implement with C API call when available
      (void)expected_arrivals;
      (void)redop_id;
      (void)initial_value;
      (void)initial_value_size;
      return Barrier(REALM_NO_EVENT);
    }

    struct ParticipantInfo {
      AddressSpace address_space;
      unsigned count;
    };

    /**
     * Creates a barrier
     * \param expected_arrivals information about the arrival pattern
     * \param num_participants the size of expected arrivals
     * \param redop_id ID of a reduction operator
     * \param initial_value initial reduction value
     * \param initial_value_size size of the initial reduction value.
     * \return barrier handle
     */
    static Barrier create_barrier(const Barrier::ParticipantInfo *expected_arrivals,
                                  size_t num_participants, ReductionOpID redop_id = 0,
                                  const void *initial_value = 0,
                                  size_t initial_value_size = 0)
    {
      // TODO: Implement with C API call when available
      (void)expected_arrivals;
      (void)num_participants;
      (void)redop_id;
      (void)initial_value;
      (void)initial_value_size;
      return Barrier(REALM_NO_EVENT);
    }

    /**
     * Sets the arrival pattern
     * \param expected_arrivals information about the arrival pattern
     * \param num_participants the size of expected arrivals
     * \return barrier handle
     */
    Barrier set_arrival_pattern(const Barrier::ParticipantInfo *expected_arrivals,
                                size_t num_participants)
    {
      // TODO: Implement with C API call when available
      (void)expected_arrivals;
      (void)num_participants;
      return *this;
    }

    void destroy_barrier(void)
    {
      // TODO: Implement with C API call when available
    }

    static const ::realm_event_gen_t MAX_PHASES;

    /*
     * Advance a barrier to the next phase, returning a new barrier
     * handle. Attemps to advance beyond the last phase return NO_BARRIER
     * instead.
     * \return the new barrier handle.
     */
    Barrier advance_barrier(void) const
    {
      // TODO: Implement with C API call when available
      return Barrier(REALM_NO_EVENT);
    }

    /*
     * Alter the arrival count of a barrier.
     * \param delta the amount to adjust the arrival count by
     * \return the new barrier handle.
     */
    Barrier alter_arrival_count(int delta) const
    {
      // TODO: Implement with C API call when available
      (void)delta;
      return *this;
    }

    /*
     * Get the previous phase of a barrier.
     * \return the previous phase of the barrier
     */
    Barrier get_previous_phase(void) const
    {
      // TODO: Implement with C API call when available
      return Barrier(REALM_NO_EVENT);
    }

    /*
     * Adjust the arrival count of a barrier.
     * \param count the amount to adjust the arrival count by
     * \param wait_on an event that must trigger before the arrival count
     * can be adjusted.
     * \param ignore_faults if true, the arrival count will be adjusted even
     * if the event it is waiting on is poisoned.
     * \param reduce_value if non-null, the value will be used to update the
     * reduction value associated with the barrier.
     * \param reduce_value_size the size of the reduction value
     */
    void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT,
                const void *reduce_value = 0, size_t reduce_value_size = 0) const
    {
      // TODO: Implement with C API call when available
      (void)count;
      (void)wait_on;
      (void)reduce_value;
      (void)reduce_value_size;
    }

    /*
     * Get the resulting barrier value.
     * \param value the resulting value
     * \param value_size the size of the resulting value
     * \return true if the value was successfully retrieved,
     * generation hasn't triggered yet.
     */
    bool get_result(void *value, size_t value_size) const
    {
      // TODO: Implement with C API call when available
      (void)value;
      (void)value_size;
      return false;
    }
  };

  /**
   * \class CompletionQueue
   * A completion queue funnels the completion of unordered events into a
   * single queue that can be queried (and waited on) by a single servicer
   * task.
   */
  class REALM_PUBLIC_API CompletionQueue {
  public:
    typedef ::realm_id_t id_t;

    id_t id;
    bool operator<(const CompletionQueue &rhs) const;
    bool operator==(const CompletionQueue &rhs) const;
    bool operator!=(const CompletionQueue &rhs) const;

    static const CompletionQueue NO_QUEUE;

    bool exists(void) const
    {
      return id != REALM_NO_EVENT;
    }

    /**
     * Create a completion queue that can hold at least 'max_size'
     * triggered events (at the moment, overflow is a fatal error).
     * A 'max_size' of 0 allows for arbitrary queue growth, at the cost
     * of additional overhead.
     * \param max_size the maximum size of the queue
     * \return the completion queue
     */
    static CompletionQueue create_completion_queue(size_t max_size)
    {
      // TODO: Implement with C API call when available
      (void)max_size;
      return CompletionQueue{REALM_NO_EVENT};
    }

    /**
     * Destroy a completion queue.
     * \param wait_on an event to wait on before destroying the
     * queue.
     */
    void destroy(const Event &wait_on = Event::NO_EVENT)
    {
      // TODO: Implement with C API call when available
      (void)wait_on;
    }

    ///@{
    /**
     * Add an event to the completion queue (once it triggers).
     * non-faultaware version raises a fatal error if the specified 'event'
     * is poisoned
     * \param event the event to add
     */
    void add_event(Event event)
    {
      // TODO: Implement with C API call when available
      (void)event;
    }
    void add_event_faultaware(Event event)
    {
      // TODO: Implement with C API call when available
      (void)event;
    }
    ///@}

    /**
     * Requests up to 'max_events' triggered events to be popped from the
     * queue and stored in the provided 'events' array (if null, the
     * identities of the triggered events are discarded).
     * This call returns the actual number of events popped, which may be
     * zero (this call is nonblocking).
     * When 'add_event_faultaware' is used, any poisoning of the returned
     * events is not signalled explicitly - the caller is expected to
     * check via 'has_triggered_faultaware' itself.
     * \param events the array to store the events in
     * \param max_events the maximum number of events to pop
     * \return the number of events popped
     */
    size_t pop_events(Event *events, size_t max_events)
    {
      // TODO: Implement with C API call when available
      (void)events;
      (void)max_events;
      return 0;
    }

    /**
     * Get an event that, once triggered, guarantees that (at least) one
     * call to pop_events made since the non-empty event was requested
     * will return a non-zero number of triggered events.
     * Once a call to pop_events has been made (by the caller of
     * get_nonempty_event or anybody else), the guarantee is lost and
     * a new non-empty event must be requested.
     * Note that 'get_nonempty_event().has_triggered()' is unlikely to
     * ever return 'true' if called from a node other than the one that
     * created the completion queue (i.e. the query at least has the
     * round-trip network communication latency to deal with) - if polling
     * on the completion queue is unavoidable, the loop should poll on
     * pop_events directly.
     * \return the non-empty event
     */
    Event get_nonempty_event(void)
    {
      // TODO: Implement with C API call when available
      return Event::NO_EVENT;
    }
  };

  class REALM_PUBLIC_API Processor {
  public:
    typedef ::realm_id_t id_t;

    id_t id{REALM_NO_PROC};

    Processor() = default;
    constexpr explicit Processor(id_t id)
      : id(id)
    {}

    constexpr operator id_t() const { return id; }

    bool operator<(const Processor &rhs) const { return id < rhs.id; }
    bool operator==(const Processor &rhs) const { return id == rhs.id; }
    bool operator!=(const Processor &rhs) const { return id != rhs.id; }

    static const Processor NO_PROC;

    bool exists(void) const { return id != 0; }

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

    // Return what kind of processor this is
    Kind kind(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      uint64_t value;
      realm_processor_attr_t attr = REALM_PROCESSOR_ATTR_KIND;
      realm_processor_get_attributes(runtime, id, &attr, &value, 1);
      return static_cast<Kind>(value);
    }
    // Return the address space for this processor
    AddressSpace address_space(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      uint64_t value;
      realm_processor_attr_t attr = REALM_PROCESSOR_ATTR_ADDRESS_SPACE;
      realm_processor_get_attributes(runtime, id, &attr, &value, 1);
      return static_cast<AddressSpace>(value);
    }

    REALM_ATTR_DEPRECATED(
        "use ProcessorGroup::create_group instead",
        static Processor create_group(const notStdSpan<const Processor> &members));

    void get_group_members(Processor *member_list, size_t &num_members) const
    {
      // Stub implementation
      num_members = 0;
    }
    void get_group_members(std::vector<Processor> &member_list) const
    {
      // Stub implementation
      member_list.clear();
    }

    int get_num_cores(void) const
    {
      // Stub implementation
      return 1;
    }

    // special task IDs
    enum
    {
      // Save ID 0 for the force shutdown function
      TASK_ID_PROCESSOR_NOP = REALM_TASK_ID_PROCESSOR_NOP,
      TASK_ID_PROCESSOR_INIT = REALM_TASK_ID_PROCESSOR_INIT,
      TASK_ID_PROCESSOR_SHUTDOWN = REALM_TASK_ID_PROCESSOR_SHUTDOWN,
      TASK_ID_FIRST_AVAILABLE = REALM_TASK_ID_FIRST_AVAILABLE,
    };

    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const Event &wait_on = Event::NO_EVENT, int priority = 0) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      Event event;
      // Note: Using nullptr for ProfilingRequestSet since it's not available in C API yet
      realm_processor_spawn(runtime, id, func_id, args, arglen, nullptr, wait_on.id, priority, reinterpret_cast<realm_event_t *>(&event));
      return event;
    }

    // Same as the above but with requests for profiling
    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const ProfilingRequestSet &requests,
                const Event &wait_on = Event::NO_EVENT, int priority = 0) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      Event event;
      // Note: ProfilingRequestSet needs to be converted to realm_profiling_request_set_t
      // For now, using nullptr until the conversion is implemented
      realm_processor_spawn(runtime, id, func_id, args, arglen, nullptr, wait_on.id, priority, reinterpret_cast<realm_event_t *>(&event));
      return event;
    }

    static Processor get_executing_processor(void)
    {
      // TODO: Implement with C API call when available
      // Need C API function to get current executing processor
      return Processor(REALM_NO_PROC);
    }

    // changes the priority of the currently running task
    static void set_current_task_priority(int new_priority)
    {
      // TODO: Implement with C API call when available
      // Need C API function to set current task priority
      (void)new_priority;
    }

    // returns the finish event for the currently running task
    static Event get_current_finish_event(void)
    {
      // TODO: Implement with C API call when available
      // Need C API function to get current task finish event
      return Event::NO_EVENT;
    }

    // a scheduler lock prevents the current thread from releasing its
    //  execution resources even when waiting on an Event - multiple
    //  nested calls to 'enable_scheduler_lock' are permitted, but a
    //  matching number of calls to 'disable_scheduler_lock' are required
    static void enable_scheduler_lock(void)
    {
      // TODO: Implement with C API call when available
      // Need C API function to enable scheduler lock
    }
    static void disable_scheduler_lock(void)
    {
      // TODO: Implement with C API call when available
      // Need C API function to disable scheduler lock
    }

    // dynamic task registration - this may be done for:
    //  1) a specific processor/group (anywhere in the system)
    //  2) for all processors of a given type, either in the local address space/process,
    //       or globally
    //
    // in both cases, an Event is returned, and any tasks launched that expect to use the
    //  newly-registered task IDs must include that event as a precondition

    Event register_task(TaskFuncID func_id, const CodeDescriptor &codedesc,
                        const ProfilingRequestSet &prs, const void *user_data = 0,
                        size_t user_data_len = 0) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to register task on specific processor
      (void)func_id;
      (void)codedesc;
      (void)prs;
      (void)user_data;
      (void)user_data_len;
      return Event::NO_EVENT;
    }

    static Event register_task_by_kind(Kind target_kind, bool global, TaskFuncID func_id,
                                       const CodeDescriptor &codedesc,
                                       const ProfilingRequestSet &prs,
                                       const void *user_data = 0,
                                       size_t user_data_len = 0)
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      Event event;
      realm_register_task_flags_t flags = global ? REALM_REGISTER_TASK_GLOBAL : REALM_REGISTER_TASK_DEFAULT;
      // Note: CodeDescriptor needs to be converted to realm_task_pointer_t
      // For now, using nullptr until the conversion is implemented
      realm_processor_register_task_by_kind(runtime, static_cast<realm_processor_kind_t>(target_kind), flags, func_id, nullptr, const_cast<void*>(user_data), user_data_len, reinterpret_cast<realm_event_t *>(&event));
      return event;
    }

    // reports an execution fault in the currently running task
    static void report_execution_fault(int reason, const void *reason_data,
                                       size_t reason_size)
    {
      // TODO: Implement with C API call when available
      // Need C API function to report execution fault
      (void)reason;
      (void)reason_data;
      (void)reason_size;
    }

    // reports a problem with a processor in general (this is primarily for fault
    // injection)
    void report_processor_fault(int reason, const void *reason_data,
                                size_t reason_size) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to report processor fault
      (void)reason;
      (void)reason_data;
      (void)reason_size;
    }

    static const char *get_kind_name(Kind kind)
    {
      // TODO: Implement with C API call when available
      // Need C API function to get processor kind name
      (void)kind;
      return "UNKNOWN";
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

  // a processor group is a set of processors that share a ready task queue
  //  (in addition to their own processor-specific task queues)
  // NOTE: processor groups are currently limited to include processors from
  //  only a single node/rank in a distributed setting
  class REALM_PUBLIC_API ProcessorGroup : public Processor {
  public:
    ProcessorGroup() = default;
    constexpr explicit ProcessorGroup(id_t id)
      : Processor(id)
    {}

    static ProcessorGroup create_group(const Processor *members, size_t num_members)
    {
      // TODO: Implement with C API call when available
      (void)members;
      (void)num_members;
      return ProcessorGroup(REALM_NO_PROC);
    }
    /// Creates a group with \p members as the members of the groups.
    /// \p members will also (efficiently) accept a std::vector.
    static ProcessorGroup create_group(const notStdSpan<const Processor> &members)
    {
      return create_group(members.data(), members.size());
    }
    void destroy(const Event &wait_on = Event::NO_EVENT) const
    {
      // TODO: Implement with C API call when available
      (void)wait_on;
    }

    static const ProcessorGroup NO_PROC_GROUP;
  };

  class REALM_PUBLIC_API Runtime {
  protected:
    void *impl; // hidden internal implementation - this is NOT a transferrable handle

  public:
    Runtime(void) { realm_runtime_create(reinterpret_cast<realm_runtime_t *>(&impl)); }
    Runtime(const Runtime &r)
      : impl(r.impl)
    {}
    Runtime &operator=(const Runtime &r)
    {
      impl = r.impl;
      return *this;
    }

    ~Runtime(void) { realm_runtime_destroy(reinterpret_cast<realm_runtime_t>(impl)); }

    static Runtime get_runtime(void)
    {
      Runtime r;
      realm_runtime_get_runtime(reinterpret_cast<realm_runtime_t *>(&r.impl));
      return r;
    }

    // returns a valid (but possibly empty) string pointer describing the
    //  version of the Realm library - this can be compared against
    //  REALM_VERSION in application code to detect a header/library mismatch
    static const char *get_library_version() { return REALM_VERSION; }

    // performs any network initialization and, critically, makes sure
    //  *argc and *argv contain the application's real command line
    //  (instead of e.g. mpi spawner information)
    bool network_init(int *argc, char ***argv)
    {
      return realm_runtime_init(reinterpret_cast<realm_runtime_t>(impl), argc, argv) ==
             REALM_SUCCESS;
    }

    void parse_command_line(int argc, char **argv) {}
    void parse_command_line(std::vector<std::string> &cmdline,
                            bool remove_realm_args = false)
    {}

    void finish_configure(void) {}

    // configures the runtime from the provided command line - after this
    //  call it is possible to create user events/reservations/etc,
    //  perform registrations and query the machine model, but not spawn
    //  tasks or create instances
    bool configure_from_command_line(int argc, char **argv) { return true; }
    bool configure_from_command_line(std::vector<std::string> &cmdline,
                                     bool remove_realm_args = false)
    {
      return true;
    }

    // starts up the runtime, allowing task/instance creation
    void start(void) {}

    // single-call version of the above three calls
    bool init(int *argc, char ***argv)
    {
      return realm_runtime_init(reinterpret_cast<realm_runtime_t>(impl), argc, argv) ==
             REALM_SUCCESS;
    }

    // this is now just a wrapper around Processor::register_task - consider switching to
    //  that
    bool register_task(TaskFuncID taskid, Processor::TaskFuncPtr taskptr) { return true; }

    bool register_reduction(const Event &event, ReductionOpID redop_id,
                            const ReductionOpUntyped *redop)
    {
      return true;
    }
    bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      return register_reduction(Event::NO_EVENT, redop_id, redop);
    }
    template <typename REDOP>
    bool register_reduction(ReductionOpID redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(Event::NO_EVENT, redop_id, &redop);
    }
    template <typename REDOP>
    bool register_reduction(Event &event, ReductionOpID redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(event, redop_id, &redop);
    }

    bool register_custom_serdez(CustomSerdezID serdez_id,
                                const CustomSerdezUntyped *serdez)
    {
      return true;
    }
    template <typename SERDEZ>
    bool register_custom_serdez(CustomSerdezID serdez_id)
    {
      const CustomSerdezWrapper<SERDEZ> serdez;
      return register_custom_serdez(serdez_id, &serdez);
    }

    Event collective_spawn(Processor target_proc, TaskFuncID task_id, const void *args,
                           size_t arglen, const Event &wait_on = Event::NO_EVENT,
                           int priority = 0)
    {
      Event event;
      realm_runtime_collective_spawn(reinterpret_cast<realm_runtime_t>(impl), target_proc,
                                     task_id, args, arglen, wait_on, priority,
                                     reinterpret_cast<realm_event_t *>(&event));
      return event;
    }

    Event collective_spawn_by_kind(Processor::Kind target_kind, TaskFuncID task_id,
                                   const void *args, size_t arglen,
                                   bool one_per_node = false,
                                   const Event &wait_on = Event::NO_EVENT,
                                   int priority = 0)
    {
      Event event;
      // TODO: Need to get target_proc from target_kind - this requires additional C API
      // For now, using a default processor ID
      realm_processor_t target_proc = REALM_NO_PROC;
      realm_runtime_collective_spawn(reinterpret_cast<realm_runtime_t>(impl), target_proc,
                                     task_id, args, arglen, wait_on.id, priority,
                                     reinterpret_cast<realm_event_t *>(&event));
      return event;
    }

    // there are three potentially interesting ways to start the initial
    // tasks:
    enum RunStyle
    {
      ONE_TASK_ONLY,     // a single task on a single node of the machine
      ONE_TASK_PER_NODE, // one task running on one proc of each node
      ONE_TASK_PER_PROC, // a task for every processor in the machine
    };

    REALM_ATTR_DEPRECATED("use collective_spawn calls instead",
                          void run(TaskFuncID task_id = 0, RunStyle style = ONE_TASK_ONLY,
                                   const void *args = 0, size_t arglen = 0,
                                   bool background = false))
    {}

    // requests a shutdown of the runtime
    void shutdown(const Event &wait_on = Event::NO_EVENT, int result_code = 0)
    {
      realm_runtime_signal_shutdown(reinterpret_cast<realm_runtime_t>(impl),
                                    static_cast<realm_event_t>(wait_on.id), result_code);
    }

    // returns the result_code passed to shutdown()
    int wait_for_shutdown(void)
    {
      realm_runtime_wait_for_shutdown(reinterpret_cast<realm_runtime_t>(impl));
      return 0;
    }

    // called before runtime::init to create module configs for users to configure realm
    bool create_configs(int argc, char **argv) { return true; }

    // return the configuration of a specific module
    ModuleConfig *get_module_config(const std::string &name) const { return nullptr; }
    // modules in Realm may offer extra capabilities specific to certain kinds
    //  of hardware or software - to get access, you'll want to know the name
    //  of the module and it's C++ type (both should be found in the module's
    //  header file - this function will return a null pointer if the module
    //  isn't present or if the expected and actual types mismatch
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
    Module *get_module_untyped(const char *name) { return nullptr; }
  };

  class REALM_PUBLIC_API Memory {
  public:
    typedef ::realm_id_t id_t;
    id_t id{REALM_NO_MEM};

    Memory() = default;
    constexpr explicit Memory(id_t id)
      : id(id)
    {}

    constexpr operator id_t() const { return id; }

    bool operator<(const Memory &rhs) const { return id < rhs.id; }
    bool operator==(const Memory &rhs) const { return id == rhs.id; }
    bool operator!=(const Memory &rhs) const { return id != rhs.id; }

    static const Memory NO_MEMORY;

    bool exists(void) const { return id != 0; }

    // Return the address space for this memory
    AddressSpace address_space(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_ADDRESS_SPACE;
      realm_memory_get_attributes(runtime, id, &attr, &value, 1);
      return static_cast<AddressSpace>(value);
    }

    // Different Memory types (defined in realm_c.h)
    // can't just typedef the kind because of C/C++ enum scope rules
    enum Kind
    {
#define C_ENUMS(name, desc) name,
      REALM_MEMORY_KINDS(C_ENUMS)
#undef C_ENUMS
    };

    // Return what kind of memory this is
    Kind kind(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_KIND;
      realm_memory_get_attributes(runtime, id, &attr, &value, 1);
      return static_cast<Kind>(value);
    }
    // Return the maximum capacity of this memory
    size_t capacity(void) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      uint64_t value;
      realm_memory_attr_t attr = REALM_MEMORY_ATTR_CAPACITY;
      realm_memory_get_attributes(runtime, id, &attr, &value, 1);
      return static_cast<size_t>(value);
    }

    // reports a problem with a memory in general (this is primarily for fault injection)
    void report_memory_fault(int reason, const void *reason_data,
                             size_t reason_size) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to report memory fault
      (void)reason;
      (void)reason_data;
      (void)reason_size;
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

  class REALM_PUBLIC_API Machine {
  protected:
    friend class Runtime;
    explicit Machine(void *_impl)
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
      // TODO: Implement with C API call when available
      return Machine(nullptr);
    }

    class ProcessorQuery;
    class MemoryQuery;

    struct AffinityDetails {
      unsigned bandwidth;
      unsigned latency;
    };

    bool has_affinity(Processor p, Memory m, AffinityDetails *details = 0) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to check processor-memory affinity
      (void)p;
      (void)m;
      (void)details;
      return false;
    }
    bool has_affinity(Memory m1, Memory m2, AffinityDetails *details = 0) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to check memory-memory affinity
      (void)m1;
      (void)m2;
      (void)details;
      return false;
    }

    // older queries, to be deprecated

    void get_all_memories(std::set<Memory> &mset) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_memory_query_t query;
      realm_memory_query_create(runtime, &query);

      // Callback to collect memories
      auto collect_memory = [](realm_memory_t mem, void *user_data) -> realm_status_t {
        std::set<Memory> *mset = static_cast<std::set<Memory> *>(user_data);
        mset->insert(Memory(mem));
        return REALM_SUCCESS;
      };

      realm_memory_query_iter(query, collect_memory, &mset, SIZE_MAX);
      realm_memory_query_destroy(query);
    }
    void get_all_processors(std::set<Processor> &pset) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_processor_query_t query;
      realm_processor_query_create(runtime, &query);

      // Callback to collect processors
      auto collect_processor = [](realm_processor_t proc,
                                  void *user_data) -> realm_status_t {
        std::set<Processor> *pset = static_cast<std::set<Processor> *>(user_data);
        pset->insert(Processor(proc));
        return REALM_SUCCESS;
      };

      realm_processor_query_iter(query, collect_processor, &pset, SIZE_MAX);
      realm_processor_query_destroy(query);
    }

    void get_local_processors(std::set<Processor> &pset) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_processor_query_t query;
      realm_processor_query_create(runtime, &query);

      // Restrict to local address space
      realm_processor_query_restrict_to_address_space(query, 0);

      // Callback to collect processors
      auto collect_processor = [](realm_processor_t proc, void *user_data) -> realm_status_t {
        std::set<Processor> *pset = static_cast<std::set<Processor> *>(user_data);
        pset->insert(Processor(proc));
        return REALM_SUCCESS;
      };

      realm_processor_query_iter(query, collect_processor, &pset, SIZE_MAX);
      realm_processor_query_destroy(query);
    }
    void get_local_processors_by_kind(std::set<Processor> &pset,
                                      Processor::Kind kind) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_processor_query_t query;
      realm_processor_query_create(runtime, &query);

      // Restrict to local address space and specific kind
      realm_processor_query_restrict_to_kind(query,
                                             static_cast<realm_processor_kind_t>(kind));
      realm_processor_query_restrict_to_address_space(query, 0); // local address space

      // Callback to collect processors
      auto collect_processor = [](realm_processor_t proc,
                                  void *user_data) -> realm_status_t {
        std::set<Processor> *pset = static_cast<std::set<Processor> *>(user_data);
        pset->insert(Processor(proc));
        return REALM_SUCCESS;
      };

      realm_processor_query_iter(query, collect_processor, &pset, SIZE_MAX);
      realm_processor_query_destroy(query);
    }

    // Return the set of memories visible from a processor
    void get_visible_memories(Processor p, std::set<Memory> &mset,
                              bool local_only = true) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_memory_query_t query;
      realm_memory_query_create(runtime, &query);

      if(local_only) {
        realm_memory_query_restrict_to_address_space(query, 0);
      }

      // TODO: Need C API to restrict memories by processor affinity
      // For now, just get all memories

      // Callback to collect memories
      auto collect_memory = [](realm_memory_t mem, void *user_data) -> realm_status_t {
        std::set<Memory> *mset = static_cast<std::set<Memory> *>(user_data);
        mset->insert(Memory(mem));
        return REALM_SUCCESS;
      };

      realm_memory_query_iter(query, collect_memory, &mset, SIZE_MAX);
      realm_memory_query_destroy(query);
    }

    // Return the set of memories visible from a memory
    void get_visible_memories(Memory m, std::set<Memory> &mset,
                              bool local_only = true) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_memory_query_t query;
      realm_memory_query_create(runtime, &query);

      if(local_only) {
        realm_memory_query_restrict_to_address_space(query, 0); // local address space
      }

      // Callback to collect memories
      auto collect_memory = [](realm_memory_t mem, void *user_data) -> realm_status_t {
        std::set<Memory> *mset = static_cast<std::set<Memory> *>(user_data);
        mset->insert(Memory(mem));
        return REALM_SUCCESS;
      };

      realm_memory_query_iter(query, collect_memory, &mset, SIZE_MAX);
      realm_memory_query_destroy(query);
    }

    // Return the set of processors which can all see a given memory
    void get_shared_processors(Memory m, std::set<Processor> &pset,
                               bool local_only = true) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_processor_query_t query;
      realm_processor_query_create(runtime, &query);

      if(local_only) {
        realm_processor_query_restrict_to_address_space(query, 0);
      }

      // TODO: Need C API to restrict processors by memory affinity
      // For now, just get all processors

      // Callback to collect processors
      auto collect_processor = [](realm_processor_t proc, void *user_data) -> realm_status_t {
        std::set<Processor> *pset = static_cast<std::set<Processor> *>(user_data);
        pset->insert(Processor(proc));
        return REALM_SUCCESS;
      };

      realm_processor_query_iter(query, collect_processor, &pset, SIZE_MAX);
      realm_processor_query_destroy(query);
    }

    /**
     * Get memories with at least the specified capacity using the C API.
     * \param min_capacity The minimum capacity in bytes.
     * \param mset The set to populate with memories.
     */
    void get_memories_by_capacity(size_t min_capacity, std::set<Memory> &mset) const
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_memory_query_t query;
      realm_memory_query_create(runtime, &query);

      // Restrict by capacity
      realm_memory_query_restrict_by_capacity(query, min_capacity);

      // Callback to collect memories
      auto collect_memory = [](realm_memory_t mem, void *user_data) -> realm_status_t {
        std::set<Memory> *mset = static_cast<std::set<Memory> *>(user_data);
        mset->insert(Memory(mem));
        return REALM_SUCCESS;
      };

      realm_memory_query_iter(query, collect_memory, &mset, SIZE_MAX);
      realm_memory_query_destroy(query);
    }

    size_t get_address_space_count(void) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to get address space count
      return 1;
    }

    // get information about the OS process in which tasks for a given
    //  processor run - note that the uniqueness of any/all of the provided
    //  information depends on the underlying OS and any container runtimes
    struct ProcessInfo {
      static const size_t MAX_HOSTNAME_LENGTH = 256;
      char hostname[MAX_HOSTNAME_LENGTH]; // always null-terminated
      uint64_t hostid; // gethostid on posix, hash of hostname on windows
      uint32_t processid;
    };

    // populates the `info` struct with information about the processor `p`'s
    //  containing process, returning true if successful, false if the
    //  processor is unknown or the information is unavailable
    bool get_process_info(Processor p, ProcessInfo *info) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to get process info
      (void)p;
      (void)info;
      return false;
    }

  public:
    struct ProcessorMemoryAffinity {
      Processor p;        // accessing processor
      Memory m;           // target memory
      unsigned bandwidth; // in MB/s
      unsigned latency;   // in nanoseconds
    };

    struct MemoryMemoryAffinity {
      Memory m1;          // source memory
      Memory m2;          // destination memory
      unsigned bandwidth; // in MB/s
      unsigned latency;   // in nanoseconds
    };

    int get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity> &result,
                              Processor restrict_proc = Processor::NO_PROC,
                              Memory restrict_memory = Memory::NO_MEMORY,
                              bool local_only = true) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to get processor-memory affinity information
      (void)restrict_proc;
      (void)restrict_memory;
      (void)local_only;
      result.clear();
      return 0;
    }

    int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity> &result,
                             Memory restrict_mem1 = Memory::NO_MEMORY,
                             Memory restrict_mem2 = Memory::NO_MEMORY,
                             bool local_only = true) const
    {
      // TODO: Implement with C API call when available
      // Need C API function to get memory-memory affinity information
      (void)restrict_mem1;
      (void)restrict_mem2;
      (void)local_only;
      result.clear();
      return 0;
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
    void add_subscription(MachineUpdateSubscriber *subscriber)
    {
      // TODO: Implement with C API call when available
      // Need C API function to add machine update subscription
      (void)subscriber;
    }
    void add_subscription(MachineUpdateSubscriber *subscriber,
                          const ProcessorQuery &query)
    {
      // TODO: Implement with C API call when available
      // Need C API function to add processor query subscription
      (void)subscriber;
      (void)query;
    }
    void add_subscription(MachineUpdateSubscriber *subscriber, const MemoryQuery &query)
    {
      // TODO: Implement with C API call when available
      // Need C API function to add memory query subscription
      (void)subscriber;
      (void)query;
    }

    void remove_subscription(MachineUpdateSubscriber *subscriber)
    {
      // TODO: Implement with C API call when available
      // Need C API function to remove machine update subscription
      (void)subscriber;
    }

    void *impl; // hidden internal implementation - this is NOT a transferrable handle
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
#if (!defined(__CUDACC__) && !defined(__HIPCC__))
  protected:
    friend QT;
#else
  public:
#endif
    MachineQueryIterator(const QT &_query, RT _result)
      : query(_query)
      , result(_result)
    {}

  protected:
    QT query;
    RT result;

  public:
    MachineQueryIterator(const MachineQueryIterator<QT, RT> &copy_from)
      : query(copy_from.query)
      , result(copy_from.result)
    {}

    ~MachineQueryIterator(void) {}

    MachineQueryIterator<QT, RT> &operator=(const MachineQueryIterator<QT, RT> &copy_from)
    {
      query = copy_from.query;
      result = copy_from.result;
      return *this;
    }

    bool operator==(const MachineQueryIterator<QT, RT> &compare_to) const
    {
      return (query == compare_to.query) && (result == compare_to.result);
    }

    bool operator!=(const MachineQueryIterator<QT, RT> &compare_to) const
    {
      return !(*this == compare_to);
    }

    RT operator*(void) { return result; }

    const RT *operator->(void) { return &result; }

    MachineQueryIterator<QT, RT> &operator++(/*prefix*/)
    {
      // TODO: Implement with C API call
      // Need C API to advance iterator to next result
      return *this;
    }

    MachineQueryIterator<QT, RT> operator++(int /*postfix*/)
    {
      // TODO: Implement with C API call
      // Need C API to advance iterator to next result
      return *this;
    }

    // in addition to testing an iterator against .end(), you can also cast to bool,
    // allowing for(iterator it = q.begin(); q; ++q) ...
    operator bool(void) const
    {
      // TODO: Implement with C API call
      // Need C API to check if iterator is valid
      return false;
    }
  };

  class REALM_PUBLIC_API Machine::ProcessorQuery {
  public:
    explicit ProcessorQuery(const Machine &m)
      : impl(nullptr)
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_processor_query_create(runtime,
                                   reinterpret_cast<realm_processor_query_t *>(&impl));
    }

    ProcessorQuery(const ProcessorQuery &q)
      : impl(nullptr)
    {
      // TODO: Implement with C API call to copy query
      // For now, just copy the pointer (shallow copy)
      (void)q;
    }

    ~ProcessorQuery(void)
    {
      if(impl) {
        realm_processor_query_destroy(reinterpret_cast<realm_processor_query_t>(impl));
      }
    }

    ProcessorQuery &operator=(const ProcessorQuery &q)
    {
      // TODO: Implement with C API call to copy query
      // For now, just copy the pointer (shallow copy)
      (void)q;
      return *this;
    }

    bool operator==(const ProcessorQuery &compare_to) const
    {
      // Stub implementation
      return impl == compare_to.impl;
    }

    bool operator!=(const ProcessorQuery &compare_to) const
    {
      return !(*this == compare_to);
    }

    // filter predicates (returns self-reference for chaining)
    // if multiple predicates are used, they must all match (i.e. the intersection is
    // returned)

    // restrict to just those of the specified 'kind'
    ProcessorQuery &only_kind(Processor::Kind kind)
    {
      if(impl) {
        realm_processor_query_restrict_to_kind(
            reinterpret_cast<realm_processor_query_t>(impl),
            static_cast<realm_processor_kind_t>(kind));
      }
      return *this;
    }

    // restrict to those managed by this address space
    ProcessorQuery &local_address_space(void)
    {
      if(impl) {
        realm_processor_query_restrict_to_address_space(
            reinterpret_cast<realm_processor_query_t>(impl), 0);
      }
      return *this;
    }

    // restrict to those in same address space as specified Processor or Memory
    ProcessorQuery &same_address_space_as(Processor p)
    {
      // TODO: Implement with C API call
      // Need to get the address space of the processor and restrict to it
      (void)p;
      return *this;
    }

    ProcessorQuery &same_address_space_as(Memory m)
    {
      // TODO: Implement with C API call
      // Need to get the address space of the memory and restrict to it
      (void)m;
      return *this;
    }

    // restrict to those that have affinity to a given memory
    ProcessorQuery &has_affinity_to(Memory m, unsigned min_bandwidth = 0,
                                    unsigned max_latency = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict processors by memory affinity with bandwidth/latency constraints
      (void)m;
      (void)min_bandwidth;
      (void)max_latency;
      return *this;
    }

    // restrict to those whose best affinity is to the given memory
    ProcessorQuery &best_affinity_to(Memory m, int bandwidth_weight = 1,
                                     int latency_weight = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict processors by best memory affinity
      (void)m;
      (void)bandwidth_weight;
      (void)latency_weight;
      return *this;
    }

    // results - a query may be executed multiple times - when the machine model is
    //  dynamic, there is no guarantee that the results of any two executions will be
    //  consistent

    // return the number of matched processors
    size_t count(void) const
    {
      // TODO: Implement with C API call
      // Need C API to count processors in query
      return 0;
    }

    // return the first matched processor, or NO_PROC
    Processor first(void) const
    {
      // TODO: Implement with C API call
      // Need C API to get first processor from query
      return Processor(REALM_NO_PROC);
    }

    // return the next matched processor after the one given, or NO_PROC
    Processor next(Processor after) const
    {
      // TODO: Implement with C API call
      // Need C API to get next processor from query
      (void)after;
      return Processor(REALM_NO_PROC);
    }

    // return a random matched processor, or NO_PROC if none exist
    Processor random(void) const
    {
      // TODO: Implement with C API call
      // Need C API to get random processor from query
      return Processor(REALM_NO_PROC);
    }

    typedef MachineQueryIterator<ProcessorQuery, Processor> iterator;

    // return an iterator that allows enumerating all matched processors
    iterator begin(void) const { return iterator(*this, first()); }

    iterator end(void) const { return iterator(*this, Processor(REALM_NO_PROC)); }

  protected:
    void *impl;
  };

  class REALM_PUBLIC_API Machine::MemoryQuery {
  public:
    explicit MemoryQuery(const Machine &m)
      : impl(nullptr)
    {
      realm_runtime_t runtime;
      realm_runtime_get_runtime(&runtime);
      realm_memory_query_create(runtime, reinterpret_cast<realm_memory_query_t *>(&impl));
    }

    MemoryQuery(const MemoryQuery &q)
      : impl(nullptr)
    {
      // TODO: Implement with C API call to copy query
      // For now, just copy the pointer (shallow copy)
      (void)q;
    }

    ~MemoryQuery(void)
    {
      if(impl) {
        realm_memory_query_destroy(reinterpret_cast<realm_memory_query_t>(impl));
      }
    }

    MemoryQuery &operator=(const MemoryQuery &q)
    {
      // TODO: Implement with C API call to copy query
      // For now, just copy the pointer (shallow copy)
      (void)q;
      return *this;
    }

    bool operator==(const MemoryQuery &compare_to) const
    {
      // Stub implementation
      return impl == compare_to.impl;
    }

    bool operator!=(const MemoryQuery &compare_to) const
    {
      return !(*this == compare_to);
    }

    // filter predicates (returns self-reference for chaining)
    // if multiple predicates are used, they must all match (i.e. the intersection is
    // returned)

    // restrict to just those of the specified 'kind'
    MemoryQuery &only_kind(Memory::Kind kind)
    {
      if(impl) {
        realm_memory_query_restrict_to_kind(reinterpret_cast<realm_memory_query_t>(impl),
                                            static_cast<realm_memory_kind_t>(kind));
      }
      return *this;
    }

    // restrict to those managed by this address space
    MemoryQuery &local_address_space(void)
    {
      if(impl) {
        realm_memory_query_restrict_to_address_space(
            reinterpret_cast<realm_memory_query_t>(impl), 0);
      }
      return *this;
    }

    // restrict to those in same address space as specified Processor or Memory
    MemoryQuery &same_address_space_as(Processor p)
    {
      // TODO: Implement with C API call
      // Need to get the address space of the processor and restrict to it
      (void)p;
      return *this;
    }

    MemoryQuery &same_address_space_as(Memory m)
    {
      // TODO: Implement with C API call
      // Need to get the address space of the memory and restrict to it
      (void)m;
      return *this;
    }

    // restrict to those that have affinity to a given processor or memory
    MemoryQuery &has_affinity_to(Processor p, unsigned min_bandwidth = 0,
                                 unsigned max_latency = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict memories by processor affinity with bandwidth/latency constraints
      (void)p;
      (void)min_bandwidth;
      (void)max_latency;
      return *this;
    }

    MemoryQuery &has_affinity_to(Memory m, unsigned min_bandwidth = 0,
                                 unsigned max_latency = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict memories by memory affinity with bandwidth/latency constraints
      (void)m;
      (void)min_bandwidth;
      (void)max_latency;
      return *this;
    }

    // restrict to those whose best affinity is to the given processor or memory
    MemoryQuery &best_affinity_to(Processor p, int bandwidth_weight = 1,
                                  int latency_weight = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict memories by best processor affinity
      (void)p;
      (void)bandwidth_weight;
      (void)latency_weight;
      return *this;
    }

    MemoryQuery &best_affinity_to(Memory m, int bandwidth_weight = 1,
                                  int latency_weight = 0)
    {
      // TODO: Implement with C API call
      // Need C API to restrict memories by best memory affinity
      (void)m;
      (void)bandwidth_weight;
      (void)latency_weight;
      return *this;
    }

    // restrict to those whose total capacity is at least 'min_size' bytes
    MemoryQuery &has_capacity(size_t min_bytes)
    {
      if(impl) {
        realm_memory_query_restrict_by_capacity(
            reinterpret_cast<realm_memory_query_t>(impl), min_bytes);
      }
      return *this;
    }

    // results - a query may be executed multiple times - when the machine model is
    //  dynamic, there is no guarantee that the results of any two executions will be
    //  consistent

    // return the number of matched processors
    size_t count(void) const
    {
      // TODO: Implement with C API call
      // Need C API to count memories in query
      return 0;
    }

    // return the first matched processor, or NO_PROC
    Memory first(void) const
    {
      // TODO: Implement with C API call
      // Need C API to get first memory from query
      return Memory(REALM_NO_MEM);
    }

    // return the next matched processor after the one given, or NO_PROC
    Memory next(Memory after) const
    {
      // TODO: Implement with C API call
      // Need C API to get next memory from query
      (void)after;
      return Memory(REALM_NO_MEM);
    }

    // return a random matched processor, or NO_PROC if none exist
    Memory random(void) const
    {
      // TODO: Implement with C API call
      // Need C API to get random memory from query
      return Memory(REALM_NO_MEM);
    }

    typedef MachineQueryIterator<MemoryQuery, Memory> iterator;

    // return an iterator that allows enumerating all matched processors
    iterator begin(void) const { return iterator(*this, first()); }

    iterator end(void) const { return iterator(*this, Memory(REALM_NO_MEM)); }

  protected:
    void *impl;
  };
#undef REALM_TYPE_KINDS
} // namespace Realm

#endif // REALM_HPP
