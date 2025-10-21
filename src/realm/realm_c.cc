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

#include "realm/runtime_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/event_impl.h"
#include "realm/inst_impl.h"
#include "realm/indexspace.h"
#include "realm/mutex.h"
#include "realm/utils.h"
#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_access.h"
#endif
#include <cassert>

namespace Realm {

  extern RuntimeImpl *runtime_singleton; // defined in runtime_impl.cc

  extern const char *realm_library_version; // defined in runtime_impl.cc

  Mutex runtime_singleton_mutex;

  bool enable_unit_tests = false;

  class ProcessorQueryImplWrapper {
  public:
    explicit ProcessorQueryImplWrapper(const MachineImpl *machine_impl)
    {
      impl = new ProcessorQueryImpl(machine_impl);
    }

    ~ProcessorQueryImplWrapper(void) { impl->remove_reference(); }

    operator ProcessorQueryImpl *&() { return impl; }

    inline void restrict_to_kind(Realm::Processor::Kind kind)
    {
      impl->restrict_to_kind(kind);
    }

    inline void restrict_to_address_space(Realm::AddressSpace address_space)
    {
      impl->restrict_to_node(address_space);
    }

  protected:
    ProcessorQueryImpl *impl;
  };

  class MemoryQueryImplWrapper {
  public:
    explicit MemoryQueryImplWrapper(const MachineImpl *machine_impl)
    {
      impl = new MemoryQueryImpl(machine_impl);
    }

    ~MemoryQueryImplWrapper(void) { impl->remove_reference(); }

    operator MemoryQueryImpl *&() { return impl; }

    inline void restrict_to_kind(Realm::Memory::Kind kind)
    {
      impl->restrict_to_kind(kind);
    }

    inline void restrict_to_address_space(Realm::AddressSpace address_space)
    {
      impl->restrict_to_node(address_space);
    }

    inline void restrict_by_capacity(size_t min_bytes)
    {
      impl->restrict_by_capacity(min_bytes);
    }

  protected:
    MemoryQueryImpl *impl;
  };

  static const ProfilingRequestSet empty_prs_cxx;

}; // namespace Realm

Realm::Logger log_realm_c("realmc");

[[nodiscard]] static inline realm_status_t
check_runtime_validity_and_assign(realm_runtime_t runtime, Realm::RuntimeImpl *&impl)
{
  if(runtime == nullptr) {
    return REALM_RUNTIME_ERROR_NOT_INITIALIZED;
  }
  if(!Realm::enable_unit_tests &&
     runtime != reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton)) {
    return REALM_RUNTIME_ERROR_INVALID_RUNTIME;
  }
  impl = reinterpret_cast<Realm::RuntimeImpl *>(runtime);
  return REALM_SUCCESS;
}

[[nodiscard]] static inline realm_status_t
check_processor_validity(realm_processor_t proc)
{
  if(proc == REALM_NO_PROC) {
    return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
  }
  return Realm::ID(proc).is_processor() ? REALM_SUCCESS
                                        : REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
}

[[nodiscard]] static inline realm_status_t
check_processor_kind_validity(realm_processor_kind_t kind)
{
  if(kind >= TOC_PROC && kind <= PY_PROC) {
    return REALM_SUCCESS;
  }
  return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR_KIND;
}

[[nodiscard]] static inline realm_status_t check_event_validity(realm_event_t event)
{
  if(event == REALM_NO_EVENT) {
    return REALM_SUCCESS;
  }
  return Realm::ID(event).is_event() ? REALM_SUCCESS : REALM_EVENT_ERROR_INVALID_EVENT;
}

[[nodiscard]] static inline realm_status_t check_memory_validity(realm_memory_t mem)
{
  if(mem == REALM_NO_MEM) {
    return REALM_MEMORY_ERROR_INVALID_MEMORY;
  }
  return Realm::ID(mem).is_memory() ? REALM_SUCCESS : REALM_MEMORY_ERROR_INVALID_MEMORY;
}

[[nodiscard]] static inline realm_status_t
check_memory_kind_validity(realm_memory_kind_t kind)
{
  if(kind >= GLOBAL_MEM && kind <= GPU_DYNAMIC_MEM) {
    return REALM_SUCCESS;
  }
  return REALM_MEMORY_ERROR_INVALID_MEMORY_KIND;
}

[[nodiscard]] static inline realm_status_t
check_index_space_validity(const realm_index_space_t *index_space)
{
  if(index_space == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  if((index_space->lower_bound == nullptr) || (index_space->upper_bound == nullptr)) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  if((index_space->num_dims == 0) || (index_space->num_dims > REALM_MAX_DIM)) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  return REALM_SUCCESS;
}

[[nodiscard]] static inline realm_status_t
check_region_instance_validity(realm_region_instance_t instance)
{
  if(instance == REALM_NO_INST) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE;
  }
  return Realm::ID(instance).is_instance() ? REALM_SUCCESS
                                           : REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE;
}

[[nodiscard]] static inline realm_status_t check_region_instance_create_params_validity(
    const realm_region_instance_create_params_t *instance_creation_params)
{
  if(instance_creation_params == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS;
  }
  realm_status_t status = check_memory_validity(instance_creation_params->memory);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(instance_creation_params->lower_bound == nullptr ||
     instance_creation_params->upper_bound == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  if(instance_creation_params->num_dims == 0 ||
     instance_creation_params->num_dims > REALM_MAX_DIM) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  if(instance_creation_params->field_ids == nullptr ||
     instance_creation_params->field_sizes == nullptr ||
     instance_creation_params->num_fields == 0) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS;
  }
  return REALM_SUCCESS;
}

[[nodiscard]] static inline realm_status_t check_region_instance_copy_params_validity(
    const realm_region_instance_copy_params_t *instance_copy_params)
{
  if(instance_copy_params == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS;
  }
  if(instance_copy_params->srcs == nullptr || instance_copy_params->dsts == nullptr ||
     instance_copy_params->num_fields == 0) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS;
  }
  realm_status_t status =
      check_region_instance_validity(instance_copy_params->srcs->inst);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_validity(instance_copy_params->dsts->inst);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(instance_copy_params->srcs->size == 0 || instance_copy_params->dsts->size == 0) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS;
  }
  if(instance_copy_params->lower_bound == nullptr ||
     instance_copy_params->upper_bound == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  if(instance_copy_params->num_dims == 0 ||
     instance_copy_params->num_dims > REALM_MAX_DIM) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  return REALM_SUCCESS;
}

// Public C API starts here

realm_status_t realm_get_library_version(const char **version)
{
  if(version == nullptr) {
    return REALM_ERROR_INVALID_PARAMETER;
  }
  *version = Realm::realm_library_version;
  return REALM_SUCCESS;
}

/* Runtime API */

realm_status_t realm_runtime_create(realm_runtime_t *runtime)
{
  {
    Realm::AutoLock<> lock(Realm::runtime_singleton_mutex);
    if(Realm::runtime_singleton == nullptr) {
      Realm::runtime_singleton = new Realm::RuntimeImpl;
    }
  }
  *runtime = reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton);
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_destroy(realm_runtime_t runtime)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::AutoLock<> lock(Realm::runtime_singleton_mutex);
  delete runtime_impl;
  Realm::runtime_singleton = nullptr;
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_get_runtime(realm_runtime_t *runtime)
{
  *runtime = reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton);
  return (*runtime == nullptr) ? REALM_RUNTIME_ERROR_NOT_INITIALIZED : REALM_SUCCESS;
}

realm_status_t realm_runtime_init(realm_runtime_t runtime, int *argc, char ***argv)
{
  REALM_ENTRY_EXIT(log_realm_c);
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }

  // if we get null pointers for argc and argv, use a local version so
  //  any changes from network_init are seen in configure_from_command_line
  int my_argc = 0;
  char **my_argv = nullptr;
  if(argc == nullptr) {
    argc = &my_argc;
  }
  if(argv == nullptr) {
    argv = &my_argv;
  }

  // TODO: we need to let each of these functions to return a specific error code
  if(!runtime_impl->network_init(argc, argv)) {
    return REALM_ERROR;
  }
  if(!runtime_impl->create_configs(*argc, *argv)) {
    return REALM_ERROR;
  }

  // TODO: do not create a vector here, just pass the array
  std::vector<std::string> cmdline;
  cmdline.reserve(*argc);
  for(int i = 1; i < *argc; i++) {
    cmdline.push_back((*argv)[i]);
  }
  if(!runtime_impl->configure_from_command_line(cmdline)) {
    return REALM_ERROR;
  }
  runtime_impl->start();
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_signal_shutdown(realm_runtime_t runtime,
                                             realm_event_t wait_on, int result_code)
{

  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  runtime_impl->shutdown(Realm::Event(wait_on), result_code);
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_wait_for_shutdown(realm_runtime_t runtime)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  runtime_impl->wait_for_shutdown();
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_collective_spawn(realm_runtime_t runtime,
                                              realm_processor_t target_proc,
                                              realm_task_func_id_t task_id,
                                              const void *args, size_t arglen,
                                              realm_event_t wait_on, int priority,
                                              realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(target_proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  // TODO: check the validation of the task id if target_proc is local, if it is not
  // local, we will poison the event.
  *event = runtime_impl->collective_spawn(Realm::Processor(target_proc), task_id, args,
                                          arglen, Realm::Event(wait_on), priority);
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_get_attributes(realm_runtime_t runtime,
                                            realm_runtime_attr_t *attrs, uint64_t *values,
                                            size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(num == 0) {
    return REALM_SUCCESS;
  }
  if(attrs == nullptr || values == nullptr) {
    return REALM_RUNTIME_ERROR_INVALID_ATTRIBUTE;
  }
  for(size_t attr_idx = 0; attr_idx < num; attr_idx++) {
    switch(attrs[attr_idx]) {
    case REALM_RUNTIME_ATTR_ADDRESS_SPACE:
      // We can not use RuntimeImpl::num_nodes, because it is initialized at the very late
      // stage.
      values[attr_idx] = Realm::Network::max_node_id + 1;
      break;
    case REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE:
      values[attr_idx] = Realm::Network::my_node_id;
      break;
    default:
      return REALM_RUNTIME_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_get_memory_memory_affinity(realm_runtime_t runtime,
                                                        realm_memory_t mem1,
                                                        realm_memory_t mem2,
                                                        realm_affinity_details_t *details)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_memory_validity(mem1);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_memory_validity(mem2);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(details == nullptr) {
    return REALM_ERROR_INVALID_PARAMETER;
  }
  bool has_affinity = runtime_impl->machine->has_affinity(Realm::Memory(mem1),
                                                          Realm::Memory(mem2), details);
  return has_affinity ? REALM_SUCCESS : REALM_RUNTIME_ERROR_INVALID_AFFINITY;
}

realm_status_t
realm_runtime_get_processor_memory_affinity(realm_runtime_t runtime,
                                            realm_processor_t proc, realm_memory_t mem,
                                            realm_affinity_details_t *details)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_memory_validity(mem);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(details == nullptr) {
    return REALM_ERROR_INVALID_PARAMETER;
  }
  bool has_affinity = runtime_impl->machine->has_affinity(Realm::Processor(proc),
                                                          Realm::Memory(mem), details);
  return has_affinity ? REALM_SUCCESS : REALM_RUNTIME_ERROR_INVALID_AFFINITY;
}

/* Processor API */

realm_status_t realm_processor_register_task_by_kind(
    realm_runtime_t runtime, realm_processor_kind_t target_kind,
    realm_register_task_flags_t flags, realm_task_func_id_t task_id,
    realm_task_pointer_t func, void *user_data, size_t user_data_len,
    realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_kind_validity(target_kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(func == nullptr) {
    return REALM_PROCESSOR_ERROR_INVALID_TASK_FUNCTION;
  }
  bool global = (flags & REALM_REGISTER_TASK_GLOBAL) != 0;
  Realm::CodeDescriptor code_desc(
      Realm::Type::from_cpp_type<Realm::Processor::TaskFuncPtr>());
  code_desc.add_implementation(
      new Realm::FunctionPointerImplementation(reinterpret_cast<void (*)()>(func)));
  *event = Realm::Processor::register_task_by_kind(
      static_cast<Realm::Processor::Kind>(target_kind), global, task_id, code_desc,
      Realm::ProfilingRequestSet(), user_data, user_data_len);
  return REALM_SUCCESS;
}

realm_status_t realm_processor_spawn(realm_runtime_t runtime,
                                     realm_processor_t target_proc,
                                     realm_task_func_id_t task_id, const void *args,
                                     size_t arglen, realm_profiling_request_set_t prs,
                                     realm_event_t wait_on, int priority,
                                     realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(target_proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  // TODO: check the validation of the task id for local processor
  Realm::ProcessorImpl *proc_impl =
      runtime_impl->get_processor_impl(Realm::Processor(target_proc));

  Realm::GenEventImpl *finish_event = Realm::GenEventImpl::create_genevent(runtime_impl);
  Realm::Event cxx_event = finish_event->current_event();
  const Realm::ProfilingRequestSet *prs_cxx = &Realm::empty_prs_cxx;
  if(prs != nullptr) {
    prs_cxx = reinterpret_cast<const Realm::ProfilingRequestSet *>(prs);
  }
  proc_impl->spawn_task(task_id, args, arglen, *prs_cxx, Realm::Event(wait_on),
                        finish_event, Realm::ID(cxx_event).event_generation(), priority);
  *event = cxx_event;
  return REALM_SUCCESS;
}

realm_status_t realm_processor_get_attributes(realm_runtime_t runtime,
                                              realm_processor_t proc,
                                              realm_processor_attr_t *attrs,
                                              uint64_t *values, size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(num == 0) {
    return REALM_SUCCESS;
  }
  if((attrs == nullptr) || (values == nullptr)) {
    return REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE;
  }

  Realm::ProcessorImpl *proc_impl =
      runtime_impl->get_processor_impl(Realm::Processor(proc));
  if(proc_impl == nullptr) {
    return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
  }
  for(size_t attr_idx = 0; attr_idx < num; attr_idx++) {
    switch(attrs[attr_idx]) {
    case REALM_PROCESSOR_ATTR_KIND:
      values[attr_idx] = static_cast<uint64_t>(proc_impl->kind);
      break;
    case REALM_PROCESSOR_ATTR_ADDRESS_SPACE:
      values[attr_idx] = Realm::ID(proc).proc_owner_node();
      break;
    default:
      return REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

/* ProcessorQuery API */

realm_status_t realm_processor_query_create(realm_runtime_t runtime,
                                            realm_processor_query_t *query)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  *query = reinterpret_cast<realm_processor_query_t>(
      new Realm::ProcessorQueryImplWrapper(runtime_impl->machine));
  return REALM_SUCCESS;
}

realm_status_t realm_processor_query_destroy(realm_processor_query_t query)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::ProcessorQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query);
  delete query_impl_wrapper;
  return REALM_SUCCESS;
}

realm_status_t realm_processor_query_restrict_to_kind(realm_processor_query_t query,
                                                      realm_processor_kind_t kind)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  realm_status_t status = check_processor_kind_validity(kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::ProcessorQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query);
  query_impl_wrapper->restrict_to_kind(static_cast<Realm::Processor::Kind>(kind));
  return REALM_SUCCESS;
}

realm_status_t
realm_processor_query_restrict_to_address_space(realm_processor_query_t query,
                                                realm_address_space_t address_space)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::ProcessorQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query);
  query_impl_wrapper->restrict_to_address_space(address_space);
  return REALM_SUCCESS;
}

[[nodiscard]] static Realm::Processor
realm_processor_query_next(Realm::ProcessorQueryImpl *query_impl, Realm::Processor after)
{
  Realm::Processor proc;
  if(Realm::Config::use_machine_query_cache) {
    proc = query_impl->cache_next(after);
  } else {
    proc = query_impl->next_match(after);
  }
  return proc;
}

realm_status_t realm_processor_query_iter(realm_processor_query_t query,
                                          realm_processor_query_cb_t cb, void *user_data,
                                          size_t max_queries)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  if(cb == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_CALLBACK;
  }
  Realm::ProcessorQueryImpl *query_impl =
      *(reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query));
  size_t num_queries = 0;
  Realm::Processor proc = query_impl->first_match();
  while(num_queries < max_queries && proc != Realm::Processor::NO_PROC) {
    realm_status_t status = cb(proc, user_data);
    if(status != REALM_SUCCESS) {
      return status;
    }
    proc = realm_processor_query_next(query_impl, proc);
    num_queries++;
  }
  return REALM_SUCCESS;
}

/* Memory API */

realm_status_t realm_memory_get_attributes(realm_runtime_t runtime, realm_memory_t mem,
                                           realm_memory_attr_t *attrs, uint64_t *values,
                                           size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_memory_validity(mem);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(num == 0) {
    return REALM_SUCCESS;
  }
  if(attrs == nullptr || values == nullptr) {
    return REALM_MEMORY_ERROR_INVALID_ATTRIBUTE;
  }

  Realm::MemoryImpl *mem_impl = runtime_impl->get_memory_impl(Realm::Memory(mem));
  if(mem_impl == nullptr) {
    return REALM_MEMORY_ERROR_INVALID_MEMORY;
  }
  for(size_t attr_idx = 0; attr_idx < num; attr_idx++) {
    switch(attrs[attr_idx]) {
    case REALM_MEMORY_ATTR_KIND:
      values[attr_idx] = static_cast<uint64_t>(mem_impl->get_kind());
      break;
    case REALM_MEMORY_ATTR_ADDRESS_SPACE:
      values[attr_idx] = Realm::ID(mem).memory_owner_node();
      break;
    case REALM_MEMORY_ATTR_CAPACITY:
      values[attr_idx] = mem_impl->size;
      break;
    default:
      return REALM_MEMORY_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

/* MemoryQuery API */

realm_status_t realm_memory_query_create(realm_runtime_t runtime,
                                         realm_memory_query_t *query)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  *query = reinterpret_cast<realm_memory_query_t>(
      new Realm::MemoryQueryImplWrapper(runtime_impl->machine));
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_destroy(realm_memory_query_t query)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query);
  delete query_impl_wrapper;
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_restrict_to_kind(realm_memory_query_t query,
                                                   realm_memory_kind_t kind)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  realm_status_t status = check_memory_kind_validity(kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::MemoryQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query);
  query_impl_wrapper->restrict_to_kind(static_cast<Realm::Memory::Kind>(kind));
  return REALM_SUCCESS;
}

realm_status_t
realm_memory_query_restrict_to_address_space(realm_memory_query_t query,
                                             realm_address_space_t address_space)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query);
  query_impl_wrapper->restrict_to_address_space(address_space);
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_restrict_by_capacity(realm_memory_query_t query,
                                                       size_t min_bytes)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query);
  query_impl_wrapper->restrict_by_capacity(min_bytes);
  return REALM_SUCCESS;
}

[[nodiscard]] static Realm::Memory
realm_memory_query_next(Realm::MemoryQueryImpl *query_impl, Realm::Memory after)
{
  Realm::Memory m;
  if(Realm::Config::use_machine_query_cache) {
    m = query_impl->cache_next(after);
  } else {
    m = query_impl->next_match(after);
  }
  return m;
}

realm_status_t realm_memory_query_iter(realm_memory_query_t query,
                                       realm_memory_query_cb_t cb, void *user_data,
                                       size_t max_queries)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  if(cb == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_CALLBACK;
  }
  Realm::MemoryQueryImpl *query_impl =
      *(reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query));
  size_t num_queries = 0;
  Realm::Memory mem = query_impl->first_match();
  while(num_queries < max_queries && mem != Realm::Memory::NO_MEMORY) {
    realm_status_t status = cb(mem, user_data);
    if(status != REALM_SUCCESS) {
      return status;
    }
    mem = realm_memory_query_next(query_impl, mem);
    num_queries++;
  }
  return REALM_SUCCESS;
}

/* Event API */

realm_status_t realm_event_wait(realm_runtime_t runtime, realm_event_t event,
                                int64_t max_ns, int *poisoned)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }

  status = check_event_validity(event);
  if(status != REALM_SUCCESS) {
    return status;
  }

  // special case: NO_EVENT is always triggered
  if(event == REALM_NO_EVENT) {
    if(poisoned != nullptr) {
      *poisoned = 0;
    }
    return REALM_SUCCESS;
  }

  Realm::Event cxx_event = Realm::Event(event);
  bool poisoned_cxx = false;

  if(max_ns == REALM_WAIT_INFINITE) {
    cxx_event.wait_faultaware(poisoned_cxx);
  } else {
    cxx_event.external_timedwait_faultaware(poisoned_cxx, max_ns);
  }

  if(poisoned != nullptr) {
    *poisoned = poisoned_cxx ? 1 : 0;
  }

  return REALM_SUCCESS;
}

realm_status_t realm_event_merge(realm_runtime_t runtime, const realm_event_t *wait_for,
                                 size_t num_events, realm_event_t *event,
                                 int ignore_faults)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(wait_for == nullptr || event == nullptr) {
    return REALM_EVENT_ERROR_INVALID_EVENT;
  }
  Realm::Event *event_array =
      const_cast<Realm::Event *>(reinterpret_cast<const Realm::Event *>(wait_for));
  *event = Realm::GenEventImpl::merge_events(
      Realm::span<const Realm::Event>(event_array, num_events), ignore_faults != 0);
  return REALM_SUCCESS;
}

realm_status_t realm_event_has_triggered(realm_runtime_t runtime, realm_event_t event,
                                         int *has_triggered, int *poisoned)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(event);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(has_triggered == nullptr) {
    return REALM_ERROR_INVALID_PARAMETER;
  }
  // special case: NO_EVENT is always triggered
  if(event == REALM_NO_EVENT) {
    *has_triggered = 1;
    if(poisoned != nullptr) {
      *poisoned = 0;
    }
    return REALM_SUCCESS;
  }

  Realm::EventImpl *event_impl = runtime_impl->get_event_impl(Realm::Event(event));
  bool poisoned_cxx = false;
  bool has_triggered_cxx =
      event_impl->has_triggered(Realm::ID(event).event_generation(), poisoned_cxx);
  *has_triggered = has_triggered_cxx ? 1 : 0;
  if(poisoned != nullptr) {
    *poisoned = poisoned_cxx ? 1 : 0;
  }
  return REALM_SUCCESS;
}

/* UserEvent API */

realm_status_t realm_user_event_create(realm_runtime_t runtime, realm_user_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(event == nullptr) {
    return REALM_EVENT_ERROR_INVALID_EVENT;
  }

  Realm::Event cxx_event =
      Realm::GenEventImpl::create_genevent(runtime_impl)->current_event();
  assert(cxx_event.id != 0);
  *event = cxx_event;

  return REALM_SUCCESS;
}

realm_status_t realm_user_event_trigger(realm_runtime_t runtime, realm_user_event_t event,
                                        realm_event_t wait_on, int ignore_faults)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(event);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::UserEvent(event).trigger(Realm::Event(wait_on), ignore_faults != 0);
  return REALM_SUCCESS;
}

// Region Instance API

template <typename T, typename Functor, typename... Fnargs>
constexpr decltype(auto) realm_dim_dispatch(size_t dim, Functor f, Fnargs &&...args)
{
  switch(dim) {
#if REALM_MAX_DIM >= 1
  case 1:
  {
    return f.template operator()<1, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 2
  case 2:
  {
    return f.template operator()<2, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 3
  case 3:
  {
    return f.template operator()<3, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 4
  case 4:
  {
    return f.template operator()<4, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 5
  case 5:
  {
    return f.template operator()<5, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 6
  case 6:
  {
    return f.template operator()<6, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 7
  case 7:
  {
    return f.template operator()<7, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 8
  case 8:
  {
    return f.template operator()<8, T>(std::forward<Fnargs>(args)...);
  }
#endif
#if REALM_MAX_DIM >= 9
  case 9:
  {
    return f.template operator()<9, T>(std::forward<Fnargs>(args)...);
  }
#endif
  default:
  {
    log_realm_c.error("Invalid number of dimension: %zu", dim);
    return REALM_REGION_INSTANCE_ERROR_INVALID_DIMS;
  }
  }
  return f.template operator()<1, T>(std::forward<Fnargs>(args)...);
}

static realm_status_t convert_external_instance_resource_c_to_cxx(
    const realm_external_resource_t *external_resource,
    std::unique_ptr<Realm::ExternalInstanceResource> &external_resource_cxx)
{
  switch(external_resource->type) {
  case REALM_EXTERNAL_RESOURCE_TYPE_CUDA_MEMORY:
  {
#ifdef REALM_USE_CUDA
    const realm_external_cuda_memory_resource_t &cuda_memory_external_resource =
        external_resource->resource.cuda_memory;
    if(cuda_memory_external_resource.base == nullptr) {
      return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_BASE;
    }
    if(cuda_memory_external_resource.size == 0) {
      return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_SIZE;
    }
    if(cuda_memory_external_resource.cuda_device_id < 0) {
      return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_CUDA_DEVICE_ID;
    }
    external_resource_cxx = std::make_unique<Realm::ExternalCudaMemoryResource>(
        cuda_memory_external_resource.cuda_device_id,
        reinterpret_cast<uintptr_t>(cuda_memory_external_resource.base),
        cuda_memory_external_resource.size,
        static_cast<bool>(cuda_memory_external_resource.read_only));
    break;
#else
    return REALM_CUDA_ERROR_NOT_ENABLED;
#endif
  }
  case REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY:
  {
    const realm_external_system_memory_resource_t &system_memory_external_resource =
        external_resource->resource.system_memory;
    if(system_memory_external_resource.base == nullptr) {
      return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_BASE;
    }
    if(system_memory_external_resource.size == 0) {
      return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_SIZE;
    }
    external_resource_cxx = std::make_unique<Realm::ExternalMemoryResource>(
        reinterpret_cast<uintptr_t>(system_memory_external_resource.base),
        system_memory_external_resource.size,
        static_cast<bool>(system_memory_external_resource.read_only));
    break;
  }
  default:
  {
    return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_TYPE;
  }
  }
  return REALM_SUCCESS;
}

static realm_status_t convert_external_instance_resource_cxx_to_c(
    const std::unique_ptr<Realm::ExternalInstanceResource> &external_resource_cxx,
    realm_external_resource_t *external_resource)
{
  switch(external_resource_cxx->get_type_id()) {
  case REALM_HASH_TOKEN(Realm::ExternalCudaMemoryResource):
  {
#ifdef REALM_USE_CUDA
    external_resource->type = REALM_EXTERNAL_RESOURCE_TYPE_CUDA_MEMORY;
    const Realm::ExternalCudaMemoryResource *cuda_memory_external_resource_cxx =
        reinterpret_cast<const Realm::ExternalCudaMemoryResource *>(
            external_resource_cxx.get());
    realm_external_cuda_memory_resource_t &cuda_memory_external_resource =
        external_resource->resource.cuda_memory;
    cuda_memory_external_resource.cuda_device_id =
        cuda_memory_external_resource_cxx->cuda_device_id;
    cuda_memory_external_resource.base =
        reinterpret_cast<const void *>(cuda_memory_external_resource_cxx->base);
    cuda_memory_external_resource.size = cuda_memory_external_resource_cxx->size_in_bytes;
    cuda_memory_external_resource.read_only =
        cuda_memory_external_resource_cxx->read_only;
    break;
#else
    return REALM_CUDA_ERROR_NOT_ENABLED;
#endif
  }
  case REALM_HASH_TOKEN(Realm::ExternalMemoryResource):
  {
    external_resource->type = REALM_EXTERNAL_RESOURCE_TYPE_SYSTEM_MEMORY;
    const Realm::ExternalMemoryResource *system_memory_external_resource_cxx =
        reinterpret_cast<const Realm::ExternalMemoryResource *>(
            external_resource_cxx.get());
    realm_external_system_memory_resource_t &system_memory_external_resource =
        external_resource->resource.system_memory;
    system_memory_external_resource.base =
        reinterpret_cast<const void *>(system_memory_external_resource_cxx->base);
    system_memory_external_resource.size =
        system_memory_external_resource_cxx->size_in_bytes;
    system_memory_external_resource.read_only =
        system_memory_external_resource_cxx->read_only;
    break;
  }
  default:
  {
    return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_TYPE;
  }
  }
  return REALM_SUCCESS;
}

class RealmRegionInstanceCreate {
public:
  template <int N, typename T>
  [[nodiscard]] realm_status_t
  operator()(Realm::RuntimeImpl *runtime_impl, Realm::Memory memory, const T *lower_bound,
             const T *upper_bound, const Realm::FieldID *field_ids,
             const size_t *field_sizes, size_t num_fields, size_t block_size,
             const realm_external_resource_t *external_resource,
             const Realm::ProfilingRequestSet &prs, Realm::Event wait_on,
             Realm::RegionInstance &inst, Realm::Event &out_event)
  {
    std::unique_ptr<Realm::ExternalInstanceResource> external_resource_cxx{nullptr};
    if(external_resource != nullptr) {
      // external instance

      // convert the C API external instance resource to the C++ API external instance
      // resource Do not delete the resource_cxx, it will be maintained by the
      // RegionInstanceImpl
      realm_status_t status = convert_external_instance_resource_c_to_cxx(
          external_resource, external_resource_cxx);
      if(status != REALM_SUCCESS) {
        return status;
      }
    }
    // create instance layout
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1) {
      block_size = 0;
    }
    Realm::InstanceLayoutConstraints ilc(field_ids, field_sizes, num_fields, block_size);
    // We use fortran order here
    Realm::Rect<N, T> rect;
    int dim_order[N];
    for(int dim = 0; dim < N; dim++) {
      dim_order[dim] = dim;
      rect.lo[dim] = lower_bound[dim];
      rect.hi[dim] = upper_bound[dim];
    }
    Realm::InstanceLayoutGeneric *layout =
        Realm::InstanceLayoutGeneric::choose_instance_layout<N, T>(
            Realm::IndexSpace<N, T>(rect), ilc, dim_order);

    out_event = Realm::RegionInstanceImpl::create_instance(
        inst, runtime_impl->get_memory_impl(memory.id), layout,
        external_resource_cxx.get(), prs, wait_on);
    return REALM_SUCCESS;
  }
};

realm_status_t realm_region_instance_create(
    realm_runtime_t runtime,
    const realm_region_instance_create_params_t *instance_creation_params,
    realm_profiling_request_set_t prs, realm_event_t wait_on,
    realm_region_instance_t *instance, realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_create_params_validity(instance_creation_params);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(instance == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_INSTANCE;
  }
  if(event == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_EVENT;
  }

  const Realm::ProfilingRequestSet *prs_cxx = &Realm::empty_prs_cxx;
  if(prs != nullptr) {
    prs_cxx = reinterpret_cast<const Realm::ProfilingRequestSet *>(prs);
  }

  Realm::RegionInstance inst = Realm::RegionInstance::NO_INST;
  Realm::Event out_event = Realm::Event::NO_EVENT;

  switch(instance_creation_params->coord_type) {
  case REALM_COORD_TYPE_LONG_LONG:
  {
    const long long *lower_bound_long_long =
        reinterpret_cast<const long long *>(instance_creation_params->lower_bound);
    const long long *upper_bound_long_long =
        reinterpret_cast<const long long *>(instance_creation_params->upper_bound);
    status = realm_dim_dispatch<long long>(
        instance_creation_params->num_dims, RealmRegionInstanceCreate(), runtime_impl,
        Realm::Memory(instance_creation_params->memory), lower_bound_long_long,
        upper_bound_long_long, instance_creation_params->field_ids,
        instance_creation_params->field_sizes, instance_creation_params->num_fields,
        instance_creation_params->block_size, instance_creation_params->external_resource,
        *prs_cxx, Realm::Event(wait_on), inst, out_event);
    break;
  }
  case REALM_COORD_TYPE_INT:
  {
    const int *lower_bound_int =
        reinterpret_cast<const int *>(instance_creation_params->lower_bound);
    const int *upper_bound_int =
        reinterpret_cast<const int *>(instance_creation_params->upper_bound);
    status = realm_dim_dispatch<int>(
        instance_creation_params->num_dims, RealmRegionInstanceCreate(), runtime_impl,
        Realm::Memory(instance_creation_params->memory), lower_bound_int, upper_bound_int,
        instance_creation_params->field_ids, instance_creation_params->field_sizes,
        instance_creation_params->num_fields, instance_creation_params->block_size,
        instance_creation_params->external_resource, *prs_cxx, Realm::Event(wait_on),
        inst, out_event);
    break;
  }
  default:
  {
    return REALM_REGION_INSTANCE_ERROR_INVALID_COORD_TYPE;
  }
  }
  *instance = inst;
  *event = out_event;
  return status;
}

class RealmRegionInstanceCopy {
public:
  template <int N, typename T>
  [[nodiscard]] realm_status_t
  operator()(Realm::RuntimeImpl *runtime_impl, std::vector<Realm::CopySrcDstField> &&srcs,
             std::vector<Realm::CopySrcDstField> &&dsts, size_t num_fields,
             const T *lower_bound, const T *upper_bound, size_t num_dims,
             realm_sparsity_handle_t sparsity_map, const Realm::ProfilingRequestSet &prs,
             Realm::Event wait_on, int priority, Realm::Event &out_event)
  {
    Realm::Rect<N, T> rect;
    for(int dim = 0; dim < N; ++dim) {
      rect.lo[dim] = lower_bound[dim];
      rect.hi[dim] = upper_bound[dim];
    }
    // using IndirectionBase = typename CopyIndirection<N, T>::Base;
    // std::vector<const IndirectionBase *> empty_indirects;
    Realm::IndexSpace<N, T> ispace(rect);
    out_event = ispace.copy(std::move(srcs), std::move(dsts), prs, wait_on, priority);
    return REALM_SUCCESS;
  }
};

realm_status_t realm_region_instance_copy(
    realm_runtime_t runtime,
    const realm_region_instance_copy_params_t *instance_copy_params,
    realm_profiling_request_set_t prs, realm_event_t wait_on, int priority,
    realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_copy_params_validity(instance_copy_params);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(event == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_EVENT;
  }

  // TODO: do not copy the srcs and dsts.
  // construct srcs and dsts
  std::vector<Realm::CopySrcDstField> srcs_vec(instance_copy_params->num_fields);
  std::vector<Realm::CopySrcDstField> dsts_vec(instance_copy_params->num_fields);
  for(size_t i = 0; i < instance_copy_params->num_fields; i++) {
    srcs_vec[i].set_field(Realm::RegionInstance(instance_copy_params->srcs[i].inst),
                          instance_copy_params->srcs[i].field_id,
                          instance_copy_params->srcs[i].size);
    dsts_vec[i].set_field(Realm::RegionInstance(instance_copy_params->dsts[i].inst),
                          instance_copy_params->dsts[i].field_id,
                          instance_copy_params->dsts[i].size);
  }

  const Realm::ProfilingRequestSet *prs_cxx = &Realm::empty_prs_cxx;
  if(prs != nullptr) {
    prs_cxx = reinterpret_cast<const Realm::ProfilingRequestSet *>(prs);
  }

  Realm::Event out_event = Realm::Event::NO_EVENT;

  switch(instance_copy_params->coord_type) {
  case REALM_COORD_TYPE_LONG_LONG:
  {
    const long long *lower_bound_long_long =
        reinterpret_cast<const long long *>(instance_copy_params->lower_bound);
    const long long *upper_bound_long_long =
        reinterpret_cast<const long long *>(instance_copy_params->upper_bound);
    status = realm_dim_dispatch<long long>(
        instance_copy_params->num_dims, RealmRegionInstanceCopy(), runtime_impl,
        std::move(srcs_vec), std::move(dsts_vec), instance_copy_params->num_fields,
        lower_bound_long_long, upper_bound_long_long, instance_copy_params->num_dims,
        instance_copy_params->sparsity_map, *prs_cxx, Realm::Event(wait_on), priority,
        out_event);
    break;
  }
  case REALM_COORD_TYPE_INT:
  {
    const int *lower_bound_int =
        reinterpret_cast<const int *>(instance_copy_params->lower_bound);
    const int *upper_bound_int =
        reinterpret_cast<const int *>(instance_copy_params->upper_bound);
    status = realm_dim_dispatch<int>(
        instance_copy_params->num_dims, RealmRegionInstanceCopy(), runtime_impl,
        std::move(srcs_vec), std::move(dsts_vec), instance_copy_params->num_fields,
        lower_bound_int, upper_bound_int, instance_copy_params->num_dims,
        instance_copy_params->sparsity_map, *prs_cxx, Realm::Event(wait_on), priority,
        out_event);
    break;
  }
  default:
  {
    return REALM_REGION_INSTANCE_ERROR_INVALID_COORD_TYPE;
  }
  }
  *event = out_event;
  return status;
}

realm_status_t realm_region_instance_destroy(realm_runtime_t runtime,
                                             realm_region_instance_t instance,
                                             realm_event_t wait_on)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_validity(instance);
  if(status != REALM_SUCCESS) {
    return status;
  }

  Realm::ID id{instance};
  Realm::MemoryImpl *mem_impl = runtime_impl->get_memory_impl(id);
  assert(mem_impl != nullptr && "invalid memory handle");
  Realm::RegionInstanceImpl *inst_impl = mem_impl->get_instance(id);
  mem_impl->release_storage_deferrable(inst_impl, Realm::Event(wait_on));

  return REALM_SUCCESS;
}

realm_status_t realm_region_instance_fetch_metadata(realm_runtime_t runtime,
                                                    realm_region_instance_t instance,
                                                    realm_processor_t target,
                                                    realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_validity(instance);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(target);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(event == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_EVENT;
  }
  Realm::ID id{instance};
  Realm::MemoryImpl *mem_impl = runtime_impl->get_memory_impl(id);
  assert(mem_impl != nullptr && "invalid memory handle");
  Realm::RegionInstanceImpl *inst_impl = mem_impl->get_instance(id);
  *event = inst_impl->fetch_metadata(Realm::Processor(target));
  return REALM_SUCCESS;
}

realm_status_t realm_region_instance_get_attributes(
    realm_runtime_t runtime, realm_region_instance_t instance,
    realm_region_instance_attr_t *attrs, realm_region_instance_attr_value_t *values,
    size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_validity(instance);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(attrs == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_ATTRIBUTE;
  }
  if(values == nullptr) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_ATTRIBUTE;
  }

  for(size_t i = 0; i < num; i++) {
    switch(attrs[i]) {
    case REALM_REGION_INSTANCE_ATTR_MEMORY:
      values[i].type = REALM_REGION_INSTANCE_ATTR_MEMORY;
      values[i].value.memory =
          Realm::ID::make_memory(Realm::ID(instance).instance_owner_node(),
                                 Realm::ID(instance).instance_mem_idx())
              .convert<Realm::Memory>();
      break;
    default:
      return REALM_REGION_INSTANCE_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

class RealmRegionInstanceGenerateExternalResourceInfo {
public:
  template <int N, typename T>
  [[nodiscard]] realm_status_t
  operator()(Realm::RuntimeImpl *runtime_impl, realm_region_instance_t instance,
             const T *lower_bound, const T *upper_bound,
             Realm::span<const Realm::FieldID> &fields, bool read_only,
             std::unique_ptr<Realm::ExternalInstanceResource> &resource)
  {
    Realm::ID inst_id = Realm::ID(instance);
    Realm::MemoryImpl *mem_impl = runtime_impl->get_memory_impl(inst_id);
    // The instance is not in any memory, something is wrong.
    assert(mem_impl != nullptr && "invalid memory handle");
    Realm::RegionInstanceImpl *inst_impl = mem_impl->get_instance(inst_id);

    Realm::Rect<N, T> rect;
    for(int dim = 0; dim < N; dim++) {
      rect.lo[dim] = lower_bound[dim];
      rect.hi[dim] = upper_bound[dim];
    }
    Realm::IndexSpaceGeneric ispace(rect);
    resource.reset(
        mem_impl->generate_resource_info(inst_impl, &ispace, fields, read_only));
    return REALM_SUCCESS;
  }
};

realm_status_t realm_region_instance_generate_external_resource_info(
    realm_runtime_t runtime, realm_region_instance_t instance,
    const realm_index_space_t *index_space, const realm_field_id_t *field_ids,
    size_t num_fields, int read_only, realm_external_resource_t *external_resource)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_region_instance_validity(instance);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_index_space_validity(index_space);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(field_ids == nullptr || num_fields == 0) {
    return REALM_REGION_INSTANCE_ERROR_INVALID_FIELDS;
  }
  if(external_resource == nullptr) {
    return REALM_EXTERNAL_RESOURCE_ERROR_INVALID_RESOURCE;
  }

  // retrieve the external instance resource info
  Realm::span<const Realm::FieldID> fields(field_ids, num_fields);
  std::unique_ptr<Realm::ExternalInstanceResource> external_resource_cxx{nullptr};
  bool read_only_cxx = (read_only == 1);
  switch(index_space->coord_type) {
  case REALM_COORD_TYPE_LONG_LONG:
  {
    const long long *lower_bound_long_long =
        reinterpret_cast<const long long *>(index_space->lower_bound);
    const long long *upper_bound_long_long =
        reinterpret_cast<const long long *>(index_space->upper_bound);
    status = realm_dim_dispatch<long long>(
        index_space->num_dims, RealmRegionInstanceGenerateExternalResourceInfo(),
        runtime_impl, instance, lower_bound_long_long, upper_bound_long_long, fields,
        read_only_cxx, external_resource_cxx);
    break;
  }
  case REALM_COORD_TYPE_INT:
  {
    const int *lower_bound_int = reinterpret_cast<const int *>(index_space->lower_bound);
    const int *upper_bound_int = reinterpret_cast<const int *>(index_space->upper_bound);
    status = realm_dim_dispatch<int>(
        index_space->num_dims, RealmRegionInstanceGenerateExternalResourceInfo(),
        runtime_impl, instance, lower_bound_int, upper_bound_int, fields, read_only_cxx,
        external_resource_cxx);
    break;
  }
  default:
  {
    return REALM_REGION_INSTANCE_ERROR_INVALID_COORD_TYPE;
  }
  }

  // convert the C++ API external instance resource to the C API external instance
  // resource
  status = convert_external_instance_resource_cxx_to_c(external_resource_cxx,
                                                       external_resource);

  return status;
}

realm_status_t realm_external_resource_suggested_memory(
    realm_runtime_t runtime, const realm_external_resource_t *external_resource,
    realm_memory_t *memory)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  std::unique_ptr<Realm::ExternalInstanceResource> external_resource_cxx{nullptr};
  status = convert_external_instance_resource_c_to_cxx(external_resource,
                                                       external_resource_cxx);
  if(status != REALM_SUCCESS) {
    return status;
  }
  *memory = Realm::Memory(external_resource_cxx->suggested_memory());
  return REALM_SUCCESS;
}
