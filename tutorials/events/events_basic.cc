/* Copyright 2024 Stanford University, NVIDIA Corporation
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

/**
 * Realm Event Tutorial
 *
 * Realm is a fully asynchronous, event-based runtime, and events form the backbone
 * of Realm’s programming model, describing the dependencies between operations.
 * Realm operations are deferred by the runtime, which returns an event that triggers
 * upon completion of the operation. These events are created by the runtime and can
 * be used as pre- or post-conditions for other operations. Events provide a mechanism
 * that allows the runtime to efficiently manage asynchronous program execution,
 * offering opportunities to hide latencies when communications are required.
 *
 * All operations accept completion events as preconditions.
 * All operations must return completion events.
 *
 * Realm Event: Optimized distributed data structure
 * - Fixed-size, no reference counting
 * - No dependency on application runtime
 * - Small memory footprint
 * - Communication/storage needed only for interested nodes
 *
 * Realm applications create a huge number of events:
 *   > 1 million events per processor per node per second
 *   > 1 billion events per second cluster-wide
 *
 * Topics Covered:
 * - Events Basics
 * - Creating Events
 * - Triggering Events
 * - Creating Control Dependencies
 *
 * Events Basics
 * -------------
 * Usually, Realm creates Events as part of handling application requests for asynchronous
 * operations. An Event is a lightweight handle that can be easily transported around the
 * system. The node that creates an Event owns it, and the space for these handles is
 * statically divided across all nodes by including the node ID in the upper bits of the
 * handle. This design ensures that any node can create new handles without the risk of
 * collision or requiring inter-node communication.
 *
 * When a new Event is created, the owning node allocates a data structure to track its
 * state, which is initially untriggered but will eventually become triggered or poisoned.
 * The data structure also includes a list of local waiters and remote waiters. Local
 * waiters are dependent operations on the owner node, and remote waiters are other nodes
 * that are interested in the Event (event dependencies).
 */

#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  READER_TASK_0,
  READER_TASK_1,
};

Logger log_app("app");

namespace ProgramConfig {
  size_t num_tasks = 1;
};

struct TaskArgs {
  int x;
};

void reader_task_0(const void *args, size_t arglen, const void *userdata, size_t userlen,
                   Processor p)
{
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 0: proc=" << p << " x=" << task_args->x;
}

void reader_task_1(const void *args, size_t arglen, const void *userdata, size_t userlen,
                   Processor p)
{
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 1: proc=" << p << " x=" << task_args->x;
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs task_args{.x = 7};

  /**
   *
   * Creating Events
   * ---------------
   * In this program, we launch several tasks (reader_task_0 and reader_task_1)
   * responsible for printing an integer value x. Each task launch is a non-blocking
   * asynchronous call that returns an internal event handle. Once created, the Event
   * handle can be passed around through task arguments or shared data structures and
   * eventually used as a pre- or post- condition for operations to be executed on other
   * nodes.
   *
   * When a remote node makes the first reference to task_event, it allocates the same
   * data structure, sets its state to untriggered, and adds the dependent operation to
   * its own local waiter list. Then, an event subscription active message is sent to the
   * owner node to indicate that the remote node is interested and should be added to the
   * list of remote waiters, so it can be informed when task_event triggers. Any
   * additional dependent operations on a remote node are added to the list of local
   * waiters without requiring communication with the owner node. When task_event
   * eventually triggers, the owner node notifies all local waiters and sends an event
   * trigger message to each subscribed node on the list of remote waiters. If the owner
   * node receives additional subscription messages after it has been triggered, it
   * immediately responds to the new subscribers with a trigger message as well.
   *
   * ---------------------------
   * A UserEvent is created explicitly by the application. It starts in an
   * untriggered state and can later be triggered manually by the user. This allows
   * dynamic construction of control dependencies, enabling other events or tasks to
   * be conditioned on it. The event handle is small, fixed-size, and efficiently
   * propagated.
   */
  UserEvent user_event = UserEvent::create_user_event();

  std::vector<Event> events;

  /* Creating Control Dependencies
   * -----------------------------
   * We demonstrate how to establish a control dependency using events by making
   * reader_task_1 dependent on the completion of reader_task_0. We achieve this by
   * passing reader_event0 to the task invocation procedure:
   *
   *   Event reader_event0 = p.spawn(READER_TASK_0, &task_args, sizeof(TaskArgs),
   * user_event); Event reader_event1 = p.spawn(READER_TASK_1, &task_args,
   * sizeof(TaskArgs), reader_event0);
   *
   * Often, it is necessary to spawn multiple tasks simultaneously and express a
   * collective wait using a single event handle. To illustrate this, the program runs
   * num_tasks, stores the events produced by reader_task_1 into an events vector, and
   * combines them by calling:
   *
   *   Event::merge_events(events).wait();
   */
  for(size_t i = 0; i < ProgramConfig::num_tasks; i++) {
    /**
     * READER_TASK_0 runs once user_event is triggered.
     */
    Event reader_event0 =
        p.spawn(READER_TASK_0, &task_args, sizeof(TaskArgs), user_event);

    /**
     * READER_TASK_1 depends on READER_TASK_0.
     * This forms a chain of control dependencies.
     */
    Event reader_event1 =
        p.spawn(READER_TASK_1, &task_args, sizeof(TaskArgs), reader_event0);

    events.push_back(reader_event1);
  }

  /**
   * Triggering Events
   * -----------------
   * An event can be triggered from any node, not necessarily the owner node. One common
   * scenario in which this happens is with UserEvent. These are created and triggered
   * from the application code, where we create user_event to start an operation. User
   * events offer greater flexibility in building the event graph by allowing users to
   * connect different parts of the graph independently. However, it is important to note
   * that using user events carries the risk of creating cycles, which can cause the
   * program to hang. Therefore, it is the user’s responsibility to avoid creating cycles
   * while leveraging user events.
   *
   * When a user_event is triggered on a node that does not own it, a trigger message is
   * sent from the trigger node to the owner node, which then forwards the message to all
   * other subscribed nodes. If the triggering node has any local waiters, it immediately
   * notifies them without sending a message back to the owner node. Although triggering a
   * remote event incurs a latency of at least two active message flight times, it limits
   * the number of active messages required per event trigger to 2*N - 2, where N is the
   * number of nodes interested in the event.
   *
   */
  user_event.trigger();

  /**
   * Wait for all READER_TASK_1 events to complete.
   */
  Event::merge_events(events).wait();

  log_app.info() << "Completed successfully";
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  if(!p.exists()) {
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  }
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK_0,
                                   CodeDescriptor(reader_task_0), ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK_1,
                                   CodeDescriptor(reader_task_1), ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, MAIN_TASK, 0, 0);

  int ret = rt.wait_for_shutdown();
  return ret;
}