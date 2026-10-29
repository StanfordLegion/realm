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

// model configure for Realm

#ifndef REALM_MODULECONFIG_H
#define REALM_MODULECONFIG_H

#include "realm/realm_config.h"
#include "realm/realm_c.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Realm {

  // Base class for the per-module configuration objects that are created by
  //  Runtime::create_configs() and handed out (by pointer) by
  //  Runtime::get_module_config().  Each module defines a subclass (e.g.
  //  Realm::Cuda::CudaModuleConfig) whose settings are plain public data
  //  members, and which declares a `static constexpr const char *MODULE_NAME`
  //  naming the module.  The headers declaring these subclasses
  //  contain only plain data types and are installed with every Realm build,
  //  whether or not the corresponding module is compiled in, so an application
  //  may always refer to them and use the typed
  //  Runtime::get_module_config<T>() to find out (via a null return) whether a
  //  module is actually present.
  //
  // Every subclass is default constructible and copyable.  A default
  //  constructed object holds Realm's default settings, and copies hold the
  //  values of the settings - assigning one configuration to another changes
  //  the destination's settings but not the resources that were discovered
  //  for it (see the *ModuleResources structs), so an application may prepare
  //  a configuration by value and then assign it to the object owned by the
  //  runtime:
  //
  //    CoreModuleConfig cfg;
  //    cfg.num_cpu_procs = 4;
  //    *runtime.get_module_config<CoreModuleConfig>() = cfg;
  //
  // This class holds nothing but the module's name, so a subclass whose
  //  settings are all of trivially copyable types is itself trivially
  //  copyable.
  //
  // The string-based set_property()/get_property()/get_resource() interface
  //  below is retained for applications that prefer to avoid naming the
  //  configuration types at compile time.  It works on any object of a
  //  configuration class whose module is present in the current runtime.
  //
  // NOTE for module authors: this class, and therefore every subclass, has
  //  no virtual functions - a class with no non-inline virtual functions has
  //  no "key function", so the compiler emits whatever vtable and typeinfo it
  //  needs with vague linkage in every translation unit that uses the class.
  //  That is what lets an application name the configuration class of a
  //  module that is not compiled into this build of Realm - including under
  //  -fsanitize=vptr, which references the typeinfo - without an unresolved
  //  symbol at link time.  Subclasses must not add virtual functions either.
  //  Module-specific behavior is instead reached through a process-wide
  //  registry of per-module function pointers and tables, keyed by module
  //  name, that the module's factory fills in with bind<>() below.
  class REALM_PUBLIC_API ModuleConfig {
  public:
    // the types a setting or resource may have - the string-based interface
    //  checks the type an application asks for against the actual type of the
    //  member (there is no RTTI involved, so the list is explicit)
    enum PropertyType
    {
      PROPERTY_TYPE_UNKNOWN = 0,
      PROPERTY_TYPE_BOOL,
      PROPERTY_TYPE_INT,
      PROPERTY_TYPE_UNSIGNED,
      PROPERTY_TYPE_LONG,
      PROPERTY_TYPE_UNSIGNED_LONG,
      PROPERTY_TYPE_LONG_LONG,
      PROPERTY_TYPE_UNSIGNED_LONG_LONG,
      PROPERTY_TYPE_FLOAT,
      PROPERTY_TYPE_DOUBLE,
      PROPERTY_TYPE_STRING,
      PROPERTY_TYPE_SIZE_T_VECTOR,
      PROPERTY_TYPE_STRING_VECTOR,
    };
    static const char *property_type_name(PropertyType type);

    // a named setting (or discovered resource) of a configuration class - the
    //  accessor returns the address of the corresponding data member of the
    //  given instance, so a table of these is a property of the class, not of
    //  any object, and can be shared by every instance
    struct Property {
      const char *name;
      void *(*access)(ModuleConfig &config);
      PropertyType type;
    };

    // builds the table entry for a data member of a configuration class, e.g.
    //    property<&CudaModuleConfig::cfg_num_gpus>("gpu")
    template <auto MemberPtr>
    static constexpr Property property(const char *name);

    // the same for a member that was inherited from a base class of the
    //  configuration class (where the class can not be deduced from the member
    //  pointer), e.g. property<CudaModuleConfig, &CudaModuleConfig::res_num_gpus>("gpu")
    template <typename Config, auto MemberPtr>
    static constexpr Property property(const char *name);

    // the accessor used by the entries built above
    template <typename Config, auto MemberPtr>
    static void *member_accessor(ModuleConfig &config);

    ModuleConfig(const ModuleConfig &other) = default;
    ModuleConfig &operator=(const ModuleConfig &other) = default;

    template <typename T>
    RealmStatus set_property(const std::string &name, T value);
    template <typename T>
    RealmStatus get_property(const std::string &name, T &value) const;
    void finish_configure(void);
    std::string get_name(void) const;
    template <typename T>
    RealmStatus get_resource(const std::string &name, T &value) const;

    // parses the module's command-line options by invoking the parser the
    //  module registered - called by the runtime during Runtime::init()
    void configure_from_cmdline(std::vector<std::string> &cmdline);

    // destroys a configuration object created by a module's factory (the
    //  runtime owns those objects and uses this to clean them up)
    static void destroy(ModuleConfig *config);

    // binds a configuration class to its module: this registers the module's
    //  non-virtual command-line parser
    //    void T::parse_cmdline(std::vector<std::string> &cmdline);
    //  and the tables that name its settings and (if any) discovered resources
    //  for the string-based interface - a module's factory calls this when it
    //  creates a configuration object, so the symbols it references are only
    //  ever referenced from the module's own translation unit (the tables
    //  must have static storage duration)
    template <typename T, size_t NumProperties>
    static void bind(const Property (&property_table)[NumProperties]);
    template <typename T, size_t NumProperties, size_t NumResources>
    static void bind(const Property (&property_table)[NumProperties],
                     const Property (&resource_table)[NumResources]);

    typedef void (*CmdlineParserFn)(ModuleConfig &config,
                                    std::vector<std::string> &cmdline);
    typedef void (*DeleterFn)(ModuleConfig *config);
    typedef bool (*ResourcesDiscoveredFn)(const ModuleConfig &config);

    // everything the runtime and the string-based interface need to know
    //  about a configuration class - one of these exists per module, not per
    //  object
    struct Binding {
      const char *name = nullptr;
      CmdlineParserFn cmdline_parser = nullptr;
      DeleterFn deleter = nullptr;
      const Property *properties = nullptr;
      size_t num_properties = 0;
      const Property *resources = nullptr;
      size_t num_resources = 0;
      ResourcesDiscoveredFn resources_discovered = nullptr;
    };

  protected:
    // constructs a default-valued configuration for the named module (the name
    //  must have static storage duration - subclasses pass their MODULE_NAME)
    ModuleConfig(const char *name);

    template <typename T>
    static Binding make_binding(void);
    static void register_binding(const Binding &binding);
    // the binding of this object's module, or null (with a complaint logged)
    //  if the module is not present
    const Binding *find_binding(void) const;

    // finds the named entry of a table
    static const Property *find(const Property *table, size_t count,
                                const std::string &name);
    // checks the type of a table entry against the type an application
    //  requested, logging a complaint if they differ
    bool check_type(const Property &prop, PropertyType requested) const;

    // the module this configuration belongs to (its MODULE_NAME)
    const char *module_name = nullptr;

    bool finish_configured = false;
  };

}; // namespace Realm

#endif // ifndef REALM_MODULECONFIG_H

#include "realm/module_config.inl"
