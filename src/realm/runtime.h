/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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
    /**
     * \struct KeyValueStoreVtable
     * Some networks prefer to bootstrap via callbacks using a vtable
     * A client provides an implementation of the functions in the vtable
     * and Realm will invoke them as part of bootstrapping the network and
     * providing support for elasticity (when networks support it).
     * Some functions are required while others are either/or options.
     * All callbacks will either be performed in an external thread
     * (one not made by Realm but has called into Realm) or by a designated
     * Realm thread independent of Realm's background worker threads so that
     * clients can use non-Realm synchronization primitives in the
     * implementation of these functions and not need to worry about
     * blocking or impacting forward progress.
     */ 
    struct KeyValueStoreVtable {
      /**
       * Optional blob of data passed to all the network vtable functions
       * when they are invoked by Realm. Realm will not attempt to
       * interpret this data at all but will simply pass it through
       * to each call. You do not have to pass any data through to
       * implement the callbacks, it is purely for your convenience.
       */
      const void *vtable_data;
      size_t vtable_data_size;
      ////////////////////////
      // REQUIRED FUNCTIONS //
      ////////////////////////
      /**
       * The "put" function must store a global key value pair in a way that it
       * can be retrieved from any other process using a corresponding get call.
       * The function is passed a buffer containing a key and buffer containing
       * a value. The implementation must copy these values before returning from
       * the callback if it needs to persist them as they are not guaranteed to
       * live longer than the function call. The function should return true if
       * the put succeeds and false if it doesn't. If the call fails then it is
       * likely that the network initialization might not succeed.
       */
      bool (*put)(const void *key, size_t key_size, const void *value, size_t value_sizem,
                  const void *vtable_data, size_t vtable_data_size) = nullptr;
      /**
       * The "get" function must retrieve the value associated with the given key
       * if it can be found. Realm will call this function with the value buffer
       * already allocated with the value_size populated with the maximum size
       * of the value that can be returned. If the key is found and the value
       * size is less than or equal to the value_size passed in by Realm, then
       * the value buffer should be populated and the value_size updated with
       * the actual size of the value found. If the key is found but the
       * resulting value is larger than the buffer size, then the function
       * should update value_size with the actual size of the buffer but
       * does not need to populate the value buffer (sinze it obviously will
       * not fit). If the key cannot be found then the value size should be
       * set to zero. The function should always return true as long as the
       * get call works correctly (even if a key is not found or the buffer
       * is not larger enough). Returning false should only occur if the
       * function call fails in some way that makes it impossible to know
       * if the key exists or not. If the callback fails then the network
       * initialization might not succeed.
       */
      bool (*get)(const void *key, size_t key_size, void *value, size_t *value_size,
                  const void *vtable_data, size_t vtable_data_size) = nullptr;
      //////////////////////////////
      // SYNCRONIZATION FUNCTIONS //
      // (PROVIDE ONE OR BOTH)    //
      //////////////////////////////
      // For the synchronization callbacks you can provide one or both
      // functions. All three combinations correspond to different use cases.
      // * Providing only "bar": this is an inelastic job with a fixed
      //   universe of processes that will never change.
      // * Providing only "cas": this is an elastic job with processes
      //   that will come and go one at a time.
      // * Providing both: this is an elastic job with processes that
      //   will come and go as groups. Groups of processes must both
      //   join and leave together.
      /**
       * The "bar" function should be provided in cases where processes
       * are joining and leaving the Realm as a group. It must perform
       * a barrier across all the processes in the (implicit) group that
       * this process is a part of along with flushing any puts done
       * before it. It should return true if the barrier succeeds and
       * false if it fails. If the barrier fails then it can be expected
       * the Realm bootstrap will also fail. If you provide a bar method,
       * then you must also provide support in the "get" method for three
       * special keys:
       * - "realm_group": the value associated with this key must be
       *   interpretable as an integer that is the same for all processes
       *   that cooperate in this barrier. It must be distinct from
       *   any other group identifier for other processes that are
       *   joining the same Realm.
       * - "realm_ranks": the value associated with this key must be
       *   interpretable as an integer that indicates how many processes
       *   are participating in this barrier operation together.
       * - "realm_rank": the value associated with this key must be
       *   interpretable as an integer that uniquely identifies this
       *   process in its local group. It must be in the range
       *   [0, realm_ranks).
       * Note that each group should have its rank numbering start at zero
       * and grow incrementally. Rank numbers are therefore the same
       * across groups. Realm will generate a unique address space for
       * each process as part of the bootstrap.
       */
      bool (*bar)(const void *vtable_data, size_t vtable_data_size) = nullptr;
      /**
       * The "cas" function should be provided in cases of elastic
       * bootstrap when an arbitrary number of processes can join or
       * leave the Realm during its execution. The cas function should
       * perform an atomic compare-and-swap operation on a key by
       * checking that the key matches a particular value and if it
       * does then updating it with the desired value in a single atomic
       * operation. If the key does not yet exist then the transaction
       * should create the key with the desirecd value. If the value
       * associated with the key does not match the expected result,
       * the call should fail, but return the updated expected
       * value and size as long as it is less than or equal to the
       * original expected size. If the new value size is larger than
       * the expected size, then only the expected_size should be updated.
       * It is possible for this call to fail and for the bootstrap to
       * continue, although a large number of repetitive failures will
       * likely lead to a timeout.
       */
      bool (*cas)(const void *key, size_t key_size, void *expected, size_t *expected_size,
                  const void *desired, size_t desired_size, const void *vtable_data,
                  size_t vtable_data_size) = nullptr;
    };
    bool network_init(const KeyValueStoreVtable &vtable);

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
