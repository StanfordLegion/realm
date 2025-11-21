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

// Realm runtime object

#ifndef REALM_RUNTIME_H
#define REALM_RUNTIME_H

#include "realm/processor.h"
#include "realm/redop.h"
#include "realm/custom_serdez.h"
#include "realm/module_config.h"

namespace Realm {

  class Module;

  class REALM_PUBLIC_API Runtime {
  protected:
    void *impl; // hidden internal implementation - this is NOT a transferrable handle

  public:
    Runtime(void);
    Runtime(const Runtime &r)
      : impl(r.impl)
    {}
    Runtime &operator=(const Runtime &r)
    {
      impl = r.impl;
      return *this;
    }

    ~Runtime(void) {}

    static Runtime get_runtime(void);

    // returns a valid (but possibly empty) string pointer describing the
    //  version of the Realm library - this can be compared against
    //  REALM_VERSION in application code to detect a header/library mismatch
    static const char *get_library_version();

    // performs any network initialization and, critically, makes sure
    //  *argc and *argv contain the application's real command line
    //  (instead of e.g. mpi spawner information)
    bool network_init(int *argc, char ***argv);
    // Some networks prefer to bootstrap via callbacks using a vtable
    // A client provides an implementation of the functions in the vtable
    // and Realm will invoke them as part of bootstrapping the network.
    // Some functions are required in order to successfully bootstrap
    // the network while others are optional and provide information
    // whenever the state of the network changes. All callbacks will
    // either be performed in an external thread (one not made by Realm
    // but has called into Realm) or by a designated Realm thread
    // independent of Realm's background worker threads so that clients
    // can use non-Realm synchronization primitives in the implementation
    // of these functions and not need to worry about blocking or
    // impacting forward progress.
    struct NetworkVtable {
      ////////////////////////
      // REQUIRED FUNCTIONS //
      ////////////////////////
      // The "put" function must store a global key value pair in a way that it
      // can be retrieved from any other process using a corresponding get call.
      // The function is passed a buffer containing a key and buffer containing
      // a value. The implementation must copy these values before returning from
      // the callback if it needs to persist them as they are not guaranteed to
      // live longer than the function call. The function should return true if
      // the put succeeds and false if it doesn't. If the call fails then it is
      // likely that the network initialization might not succeed.
      bool (*put)(const void *key, size_t key_size, const void *value,
                  size_t value_size) = nullptr;
      // The "get" function must retrieve the value associated with the given key
      // if it can be found. Realm will call this function with the value buffer
      // already allocated with the given value_size. If the key is found and the
      // value is the correct size for the value buffer, then the callee should
      // copy the value into the value buffer and return true indicating that
      // the call succeeded. If the key cannot be found or if the value found by
      // the callee is different from the value size, then function should
      // return false indicating that the callbcak failed. If the callback fails
      // then the network initialization might not succeed.
      bool (*get)(const void *key, size_t key_size, void *value,
                  size_t value_size) = nullptr;
      ////////////////////////
      // OPTIONAL FUNCTIONS //
      ////////////////////////
      // You can pass in an optional buffer of data to uniquely identify this
      // process to other processes as part of the bootstrap. Realm will make
      // a copy of this data if provided during the network_init call so it
      // you do not need to keep it alive longer than that.
      const void *unique_data = nullptr;
      size_t unique_data_size = 0;
      // This callback is an optional callback that will be invoked any time a
      // process joins the Realm. It will be told of the assigned Realm
      // AddressSpace given to the process along with the unique data for the
      // process if it was provided (might be null). Note that you will also
      // get a callback for yourself upon joining as well. This call back is 
      // only done *AFTER* the joining process has successfully joined meaning
      // that this callback marks when it is safe to start referring to the
      // new address space and all of the handles associated with it.
      void (*join)(AddressSpace space, const void *data, size_t size) = nullptr;
      // This callback is performed any time a process leaves the Realm.
      // It will be told of the AddressSpace for the process as well as the
      // unique data associated with the process that is leaving. This
      // callback is done *BEFORE* the process leaves the Realm. After this
      // callback returns it is no longer safe to refer to handles from the
      // address space that is leaving in the process where the callback occurs.
      void (*leave)(AddressSpace space, const void *data, size_t size) = nullptr;
    };
    bool network_init(const NetworkVtable &vtable);

    void parse_command_line(int argc, char **argv);
    void parse_command_line(std::vector<std::string> &cmdline,
                            bool remove_realm_args = false);

    void finish_configure(void);

    // configures the runtime from the provided command line - after this
    //  call it is possible to create user events/reservations/etc,
    //  perform registrations and query the machine model, but not spawn
    //  tasks or create instances
    bool configure_from_command_line(int argc, char **argv);
    bool configure_from_command_line(std::vector<std::string> &cmdline,
                                     bool remove_realm_args = false);

    // starts up the runtime, allowing task/instance creation
    void start(void);

    // single-call version of the above three calls
    bool init(int *argc, char ***argv);

    // this is now just a wrapper around Processor::register_task - consider switching to
    //  that
    bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);

    bool register_reduction(Event &event, ReductionOpID redop_id,
                            const ReductionOpUntyped *redop);
    bool register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      Event event = Event::NO_EVENT;
      if(register_reduction(event, redop_id, redop)) {
        event.wait();
        return true;
      }
      return false;
    }
    template <typename REDOP>
    bool register_reduction(ReductionOpID redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(redop_id, &redop);
    }
    template <typename REDOP>
    bool register_reduction(Event &event, ReductionOpID redop_id)
    {
      const ReductionOp<REDOP> redop;
      return register_reduction(redop_id, &redop);
    }

    bool register_custom_serdez(CustomSerdezID serdez_id,
                                const CustomSerdezUntyped *serdez);
    template <typename SERDEZ>
    bool register_custom_serdez(CustomSerdezID serdez_id)
    {
      const CustomSerdezWrapper<SERDEZ> serdez;
      return register_custom_serdez(serdez_id, &serdez);
    }

    Event collective_spawn(Processor target_proc, Processor::TaskFuncID task_id,
                           const void *args, size_t arglen,
                           Event wait_on = Event::NO_EVENT, int priority = 0);

    Event collective_spawn_by_kind(Processor::Kind target_kind,
                                   Processor::TaskFuncID task_id, const void *args,
                                   size_t arglen, bool one_per_node = false,
                                   Event wait_on = Event::NO_EVENT, int priority = 0);

    // there are three potentially interesting ways to start the initial
    // tasks:
    enum RunStyle
    {
      ONE_TASK_ONLY,     // a single task on a single node of the machine
      ONE_TASK_PER_NODE, // one task running on one proc of each node
      ONE_TASK_PER_PROC, // a task for every processor in the machine
    };

    REALM_ATTR_DEPRECATED("use collective_spawn calls instead",
                          void run(Processor::TaskFuncID task_id = 0,
                                   RunStyle style = ONE_TASK_ONLY, const void *args = 0,
                                   size_t arglen = 0, bool background = false));

    // requests a shutdown of the runtime
    void shutdown(Event wait_on = Event::NO_EVENT, int result_code = 0);

    // returns the result_code passed to shutdown()
    int wait_for_shutdown(void);

    // called before runtime::init to create module configs for users to configure realm
    bool create_configs(int argc, char **argv);

    // return the configuration of a specific module
    ModuleConfig *get_module_config(const std::string &name) const;
    // modules in Realm may offer extra capabilities specific to certain kinds
    //  of hardware or software - to get access, you'll want to know the name
    //  of the module and it's C++ type (both should be found in the module's
    //  header file - this function will return a null pointer if the module
    //  isn't present or if the expected and actual types mismatch
    template <typename T>
    T *get_module(const char *name)
    {
      Module *mod = get_module_untyped(name);
      if(mod)
        return dynamic_cast<T *>(mod);
      else
        return 0;
    }

  protected:
    Module *get_module_untyped(const char *name);
  };

}; // namespace Realm

  // include "runtime.inl"

#endif // ifndef REALM_RUNTIME_H
