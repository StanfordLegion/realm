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

#include "realm/module_config.h"

#include <cstring>

namespace Realm {

  Logger log_moduleconfig("module_config");

  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleConfig
  //

  namespace {
    // the process-wide registry of module bindings - a module registers its
    //  binding (via ModuleConfig::bind<>) whenever it creates a configuration
    //  object, i.e. during Runtime::create_configs, which the runtime
    //  serializes, and the entries are only read after that
    const size_t MAX_BINDINGS = 64;
    ModuleConfig::Binding bindings[MAX_BINDINGS];
    size_t num_bindings = 0;
  }; // namespace

  ModuleConfig::ModuleConfig(const char *name)
    : module_name(name)
  {}

  /*static*/ void ModuleConfig::register_binding(const Binding &binding)
  {
    // re-registration (e.g. by a runtime that is initialized a second time in
    //  the same process) replaces the previous entry
    for(size_t i = 0; i < num_bindings; i++) {
      if(strcmp(bindings[i].name, binding.name) == 0) {
        bindings[i] = binding;
        return;
      }
    }
    assert(num_bindings < MAX_BINDINGS);
    bindings[num_bindings++] = binding;
  }

  const ModuleConfig::Binding *ModuleConfig::find_binding(void) const
  {
    for(size_t i = 0; i < num_bindings; i++)
      if(strcmp(bindings[i].name, module_name) == 0)
        return &bindings[i];
    log_moduleconfig.warning("Module %s is not present in this runtime, so its "
                             "configuration can not be accessed by name",
                             module_name);
    return nullptr;
  }

  void ModuleConfig::finish_configure(void)
  {
    assert(finish_configured == false);
    finish_configured = true;
  }

  std::string ModuleConfig::get_name(void) const { return module_name; }

  void ModuleConfig::configure_from_cmdline(std::vector<std::string> &cmdline)
  {
    const Binding *binding = find_binding();
    if(binding == nullptr) {
      log_moduleconfig.error("Module %s did not register a command-line parser",
                             module_name);
      abort();
    }
    binding->cmdline_parser(*this, cmdline);
  }

  /*static*/ void ModuleConfig::destroy(ModuleConfig *config)
  {
    const Binding *binding = config->find_binding();
    assert(binding != nullptr);
    binding->deleter(config);
  }

  /*static*/ const char *ModuleConfig::property_type_name(PropertyType type)
  {
    switch(type) {
    case PROPERTY_TYPE_BOOL:
      return "bool";
    case PROPERTY_TYPE_INT:
      return "int";
    case PROPERTY_TYPE_UNSIGNED:
      return "unsigned";
    case PROPERTY_TYPE_LONG:
      return "long";
    case PROPERTY_TYPE_UNSIGNED_LONG:
      return "unsigned long";
    case PROPERTY_TYPE_LONG_LONG:
      return "long long";
    case PROPERTY_TYPE_UNSIGNED_LONG_LONG:
      return "unsigned long long";
    case PROPERTY_TYPE_FLOAT:
      return "float";
    case PROPERTY_TYPE_DOUBLE:
      return "double";
    case PROPERTY_TYPE_STRING:
      return "std::string";
    case PROPERTY_TYPE_SIZE_T_VECTOR:
      return "std::vector<size_t>";
    case PROPERTY_TYPE_STRING_VECTOR:
      return "std::vector<std::string>";
    default:
      return "unknown";
    }
  }

  /*static*/ const ModuleConfig::Property *
  ModuleConfig::find(const Property *table, size_t count, const std::string &name)
  {
    for(size_t i = 0; i < count; i++)
      if(strcmp(table[i].name, name.c_str()) == 0)
        return &table[i];
    return nullptr;
  }

  bool ModuleConfig::check_type(const Property &prop, PropertyType requested) const
  {
    if(prop.type == requested)
      return true;
    log_moduleconfig.warning("Module %s: '%s' has type %s, not %s", module_name,
                             prop.name, property_type_name(prop.type),
                             property_type_name(requested));
    return false;
  }

}; // namespace Realm
