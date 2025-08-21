<!-- omit from toc -->
# Hello world

The tutorial begins with a simple “hello world” example that showcases the basics. You can access the source code, the Makefile and CMakeList.txt for building and running the application, in the `tutorial/realm` directory of the repository. By going through these tutorial programs in detail, we will demonstrate how to effectively use the Realm C++ runtime API.

Here is a list of covered topics:

- [Realm namespaces](#realm-namespaces)
- [Program setup](#program-setup)
  - [Realm task tdentifiers](#realm-task-tdentifiers)
- [Task implementations in Realm](#task-implementations-in-realm)
  - [`hello_task`](#hello_task)
- [Launching tasks](#launching-tasks)
  - [`launch_task`](#launch_task)
  - [`main_task`](#main_task)
- [Realm runtime initialization and task execution](#realm-runtime-initialization-and-task-execution)
  - [Initializing the Realm runtime](#initializing-the-realm-runtime)
  - [Task registration](#task-registration)
  - [Launching the `MAIN_TASK`](#launching-the-main_task)
  - [Shutting down the Realm runtime](#shutting-down-the-realm-runtime)

## Realm namespaces

Each Realm class has its own C++ header file. All classes are aggregated in realm.h and can be included in an application for convenience. Each class definition is placed in a Realm namespace to avoid naming conflicts.

## Program setup

We start our C++ Realm example with some typical imports:

- `realm.h`
- `realm/cmdline.h`
- `realm/network.h`

and also set up the Realm logger instance:

```c++
#include "realm.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#ifdef REALM_USE_OPENMP
#include <omp.h>
#endif

using namespace Realm;

// Logger instance to capture logs from the application
Logger log_app("app");
```

### Realm task tdentifiers

The first thing we need to do is to set up IDs for the Realm tasks we are going to define.
Realm tasks are assigned unique IDs within a program, and the first available user-defined task ID must be at or above `Processor::TASK_ID_FIRST_AVAILABLE` to avoid collisions with internal Realm tasks.

We use an `enum` for this:

```c++
enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};
```

## Task implementations in Realm

Tasks are Realm’s fundamental unit of execution. They are functions that execute
asynchronously on different types of processors. In Realm, tasks are bound to specific processor types (CPU, GPU, OpenMP, etc.), and their execution is managed by the
runtime.

### `hello_task`

Here, we actually define the functions for the "hello" tasks we want to run.
We unconditionally define a CPU variant, but also define GPU and OpenMP variants, in case those features are available on our system:

```c++
void hello_cpu_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  log_app.print() << "Hello world from CPU!";
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
void hello_gpu_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  log_app.print() << "Hello world from GPU!";
}
#endif

#ifdef REALM_USE_OPENMP
void hello_omp_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
#pragma omp parallel
  {
    log_app.print() << "Hello world from OMP thread = " << omp_get_thread_num();
  }
}
#endif
```

## Launching tasks

Since Realm uses tasks for everything, we also need a task to launch our "hello" tasks using appropriate available resources.

### `launch_task`

The `launch_task` function defined below dispatches `HELLO_TASK` to available processors.
Realm supports querying machine topology at runtime, enabling dynamic task assignment.
This function finds available CPU, GPU, and OpenMP processors and spawns `HELLO_TASK` on them.
Realm's event system is then used to synchronize completion of these different variants.

```c++
inline Event launch_task(Processor cpu)
{
  Event cpu_e = cpu.spawn(HELLO_TASK, NULL, 0);

  Processor gpu = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::TOC_PROC)
                      .first();
  Event gpu_e = Event::NO_EVENT;
  if(gpu.exists()) {
    gpu_e = gpu.spawn(HELLO_TASK, NULL, 0);
  }

  Processor omp = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::OMP_PROC)
                      .first();
  Event omp_e = Event::NO_EVENT;
  if(omp.exists()) {
    omp_e = omp.spawn(HELLO_TASK, NULL, 0);
  }

  return Event::merge_events(cpu_e, gpu_e, omp_e);
}
```

### `main_task`

For convenience, we also define a `main_rask` as Realm's task-based execution entry point.
Realm does not enforce a hierarchical task model, so the main task explicitly waits
for all `HELLO_TASK` instances to finish, by using events.

```c++
void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  launch_task(p).wait();
}
```

## Realm runtime initialization and task execution

The standard C++ `main` function initializes the Realm runtime, registers tasks, and schedules execution.
Realm enables both single-process and distributed execution models, managed through runtime API calls.

```c++
int main(int argc, char **argv)
{
```

### Initializing the Realm runtime

Realm follows a decentralized execution model, where each process in a parallel execution (e.g., launched via `mpirun` or another job launcher) independently initializes its runtime. This is different from centralized tasking systems where a master process coordinates execution.

- The function `rt.init(&argc, &argv);` performs the following key operations:
  - Discovers available processors (CPUs, GPUs, OpenMP, etc.).
  - Establishes communication channels between processes.
  - Registers the process within the Realm machine model.
- Command-line arguments are passed so that Realm can handle its own runtime-specific options, such as enabling debugging or configuring network settings.
- Because every process initializes independently, Realm can be used in both single-process and distributed environments. However, task launching and synchronization must be handled accordingly to ensure correct execution.

```c++
  Runtime rt;
  rt.init(&argc, &argv);

  bool use_collective_spawn = false;
  CommandLineParser cp;
  cp.add_option_bool("-coll_spawn", use_collective_spawn);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
```

### Task registration

Before tasks can be executed, they must be registered with Realm.
Task registration associates a task ID (that we defined first, up above) with an implementation and a processor kind.
Unlike traditional threading models, Realm allows multiple task implementations for different processors, enabling heterogeneous execution.

```c++
  // Registering Tasks: Binding function implementations to processor types
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet(), 0, 0)
      .wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, HELLO_TASK,
                                   CodeDescriptor(hello_cpu_task), ProfilingRequestSet(),
                                   0, 0)
      .wait();
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, HELLO_TASK,
                                   CodeDescriptor(hello_gpu_task), ProfilingRequestSet(),
                                   0, 0)
      .wait();
#endif
#ifdef REALM_USE_OPENMP
  Processor::register_task_by_kind(Processor::OMP_PROC, false /*!global*/, HELLO_TASK,
                                   CodeDescriptor(hello_omp_task), ProfilingRequestSet(),
                                   0, 0)
      .wait();
#endif

  // Selecting the first available CPU processor for launching the main task
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
```

### Launching the `MAIN_TASK`

The main task can be launched using collective or manual spawning.
If `collective_spawn` is enabled, then `MAIN_TASK` is launched on all processes simultaneously.
Otherwise, process 0 is responsible for launching tasks and initiating shutdown.

```c++
  if(use_collective_spawn) {
    Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
    rt.shutdown(e);
  } else {
    Processor local_proc = Machine::ProcessorQuery(Machine::get_machine())
                               .only_kind(Processor::LOC_PROC)
                               .local_address_space()
                               .first();
    if(local_proc.address_space() == 0) {
      Event e = launch_task(p);
      rt.shutdown(e);
    }
  }
```

### Shutting down the Realm runtime

Realm applications must explicitly shut down the runtime when all tasks have completed. This ensures that all resources are properly released and prevents processes from hanging.

- The `rt.shutdown(e)` function is used to trigger the shutdown process. It takes an event `e` as a precondition, meaning the runtime will only shut down once this event has completed. In our case, this event represents the completion of the `MAIN_TASK` or all `HELLO_TASK` tasks.
- The `rt.wait_for_shutdown()` function is necessary to ensure that all processes remain synchronized and do not exit prematurely before the runtime fully shuts down.
- If collective spawning (`rt.collective_spawn()`) is used, then shutdown must be called on all processes with the same event to ensure consistency.
- When manually managing task launching without collective_spawn, we ensure only rank 0 (the process with `address_space() == 0`) triggers shutdown.

```c++
  // Ensuring all processes wait for the runtime to shut down
  int ret = rt.wait_for_shutdown();
  return ret;
}
```