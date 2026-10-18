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

#include <string>
#include <unordered_map>
#include <vector>

namespace Realm {

  // Base class for the per-module configuration objects that are created by
  //  Runtime::create_configs() and handed out (by pointer) by
  //  Runtime::get_module_config().  Each module defines a subclass (e.g.
  //  Realm::Cuda::CudaModuleConfig) whose settings are plain public data
  //  members, and which declares a `static constexpr const char *const
  //  MODULE_NAME` naming the module.  The headers declaring these subclasses
  //  contain only plain data types and are installed with every Realm build,
  //  whether or not the corresponding module is compiled in, so an application
  //  may always refer to them and use the typed
  //  Runtime::get_module_config<T>() to find out (via a null return) whether a
  //  module is actually present.
  //
  // The string-based set_property()/get_property()/get_resource() interface
  //  below is retained for applications that prefer to avoid naming the
  //  configuration types at compile time.
  class REALM_PUBLIC_API ModuleConfig {
  protected:
    // NOTE for module authors: subclasses must NOT declare any virtual
    //  functions (in particular they must not override
    //  configure_from_cmdline).  A class with no non-inline virtual functions
    //  has no "key function", so the compiler emits its vtable and typeinfo
    //  with vague linkage in every translation unit that needs them.  That is
    //  what lets an application name the configuration class of a module that
    //  is not compiled into this build of Realm - including under
    //  -fsanitize=vptr or via dynamic_cast, both of which reference the
    //  typeinfo - without an unresolved symbol at link time.  A subclass
    //  instead provides a non-virtual
    //    void parse_cmdline(std::vector<std::string> &cmdline);
    //  and registers it by passing cmdline_parser_for<Subclass>() to this
    //  constructor.  The parser is only ever referenced from the subclass's
    //  own constructor, i.e. from the module's own translation unit.
    typedef void (*CmdlineParserFn)(ModuleConfig &config,
                                    std::vector<std::string> &cmdline);

    ModuleConfig(const std::string &name, CmdlineParserFn parser = nullptr);

    template <typename T>
    static CmdlineParserFn cmdline_parser_for(void)
    {
      return [](ModuleConfig &config, std::vector<std::string> &cmdline) {
        static_cast<T &>(config).parse_cmdline(cmdline);
      };
    }

  public:
    virtual ~ModuleConfig(void);

    // configuration objects are owned by the runtime and only ever handed out
    //  by pointer - copying one would alias the member addresses stored in
    //  config_map/resource_map (so the copy would silently modify the
    //  original), so copies are not permitted
    ModuleConfig(const ModuleConfig &) = delete;
    ModuleConfig &operator=(const ModuleConfig &) = delete;

    template <typename T>
    RealmStatus set_property(const std::string &name, T value);
    template <typename T>
    RealmStatus get_property(const std::string &name, T &value) const;
    void finish_configure(void);
    const std::string &get_name(void) const;
    template <typename T>
    RealmStatus get_resource(const std::string &name, T &value) const;
    // parses the module's command-line options by invoking the parser the
    //  subclass registered (see the note above) - called by the runtime during
    //  Runtime::init()
    virtual void configure_from_cmdline(std::vector<std::string> &cmdline);

  protected:
    // these maps store a map between configuration name and address of configurations,
    //  so that we can access these configurations using the name
    std::unordered_map<std::string, void *const> config_map;

    // these maps store a map between resource name and address of resource,
    //  so that we can access these resources using the name
    std::unordered_map<std::string, void *const> resource_map;

    std::string module_name;

    // registered by the subclass constructor via cmdline_parser_for<>()
    CmdlineParserFn cmdline_parser = nullptr;

    bool resource_discover_finished = false;

    bool finish_configured = false;
  };

}; // namespace Realm

#endif // ifndef REALM_MODULECONFIG_H

#include "realm/module_config.inl"
