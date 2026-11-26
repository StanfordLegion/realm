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

#ifndef REALM_MODULECONFIG_INL
#define REALM_MODULECONFIG_INL

// nop, but helpful for IDEs
#include "realm/module_config.h"

#include "realm/logging.h"

#include <type_traits>

namespace Realm {

  REALM_INTERNAL_API_EXTERNAL_LINKAGE
  extern Logger log_moduleconfig;

  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleConfig
  //

  namespace ModuleConfigDetail {
    // extracts the class a pointer-to-data-member belongs to, and the type of
    //  the member
    template <typename MemberPtrType>
    struct MemberPointerTraits;
    template <typename Class, typename Member>
    struct MemberPointerTraits<Member Class::*> {
      typedef Class class_type;
      typedef Member member_type;
    };

    // maps a C++ type onto a PropertyType (unknown unless listed here)
    template <typename T>
    struct PropertyTypeOf {
      static constexpr ModuleConfig::PropertyType value =
          ModuleConfig::PROPERTY_TYPE_UNKNOWN;
    };
#define REALM_MODULECONFIG_PROPERTY_TYPE(T, tag)                                         \
  template <>                                                                            \
  struct PropertyTypeOf<T> {                                                             \
    static constexpr ModuleConfig::PropertyType value = ModuleConfig::tag;               \
  }
    REALM_MODULECONFIG_PROPERTY_TYPE(bool, PROPERTY_TYPE_BOOL);
    REALM_MODULECONFIG_PROPERTY_TYPE(int, PROPERTY_TYPE_INT);
    REALM_MODULECONFIG_PROPERTY_TYPE(unsigned, PROPERTY_TYPE_UNSIGNED);
    REALM_MODULECONFIG_PROPERTY_TYPE(long, PROPERTY_TYPE_LONG);
    REALM_MODULECONFIG_PROPERTY_TYPE(unsigned long, PROPERTY_TYPE_UNSIGNED_LONG);
    REALM_MODULECONFIG_PROPERTY_TYPE(long long, PROPERTY_TYPE_LONG_LONG);
    REALM_MODULECONFIG_PROPERTY_TYPE(unsigned long long,
                                     PROPERTY_TYPE_UNSIGNED_LONG_LONG);
    REALM_MODULECONFIG_PROPERTY_TYPE(float, PROPERTY_TYPE_FLOAT);
    REALM_MODULECONFIG_PROPERTY_TYPE(double, PROPERTY_TYPE_DOUBLE);
    REALM_MODULECONFIG_PROPERTY_TYPE(std::string, PROPERTY_TYPE_STRING);
    REALM_MODULECONFIG_PROPERTY_TYPE(std::vector<size_t>, PROPERTY_TYPE_SIZE_T_VECTOR);
    REALM_MODULECONFIG_PROPERTY_TYPE(std::vector<std::string>,
                                     PROPERTY_TYPE_STRING_VECTOR);
#undef REALM_MODULECONFIG_PROPERTY_TYPE
  }; // namespace ModuleConfigDetail

  template <auto MemberPtr>
  /*static*/ constexpr ModuleConfig::Property ModuleConfig::property(const char *name)
  {
    typedef
        typename ModuleConfigDetail::MemberPointerTraits<decltype(MemberPtr)>::class_type
            Config;
    return property<Config, MemberPtr>(name);
  }

  template <typename Config, auto MemberPtr>
  /*static*/ constexpr ModuleConfig::Property ModuleConfig::property(const char *name)
  {
    typedef
        typename ModuleConfigDetail::MemberPointerTraits<decltype(MemberPtr)>::member_type
            Member;
    static_assert(ModuleConfigDetail::PropertyTypeOf<Member>::value !=
                      PROPERTY_TYPE_UNKNOWN,
                  "the type of this configuration member is not known to the "
                  "string-based interface - add it to PropertyType/PropertyTypeOf");
    return Property{name, &member_accessor<Config, MemberPtr>,
                    ModuleConfigDetail::PropertyTypeOf<Member>::value};
  }

  template <typename Config, auto MemberPtr>
  /*static*/ void *ModuleConfig::member_accessor(ModuleConfig &config)
  {
    static_assert(std::is_base_of<ModuleConfig, Config>::value,
                  "member_accessor must name a member of a ModuleConfig subclass");
    return &(static_cast<Config &>(config).*MemberPtr);
  }

  template <typename T, size_t NumProperties>
  /*static*/ void ModuleConfig::bind(const Property (&properties)[NumProperties])
  {
    Binding binding = make_binding<T>();
    binding.properties = properties;
    binding.num_properties = NumProperties;
    register_binding(binding);
  }

  template <typename T, size_t NumProperties, size_t NumResources>
  /*static*/ void ModuleConfig::bind(const Property (&properties)[NumProperties],
                                     const Property (&resources)[NumResources])
  {
    Binding binding = make_binding<T>();
    binding.properties = properties;
    binding.num_properties = NumProperties;
    binding.resources = resources;
    binding.num_resources = NumResources;
    // T is expected to derive from a *ModuleResources struct that records
    //  whether discovery succeeded
    binding.resources_discovered = [](const ModuleConfig &config) {
      return static_cast<const T &>(config).resources_discovered();
    };
    register_binding(binding);
  }

  template <typename T>
  /*static*/ ModuleConfig::Binding ModuleConfig::make_binding(void)
  {
    static_assert(std::is_base_of<ModuleConfig, T>::value,
                  "bind<T> requires a ModuleConfig subclass");
    static_assert(!std::is_polymorphic<T>::value,
                  "ModuleConfig subclasses must not have virtual functions");
    Binding binding;
    binding.name = T::MODULE_NAME;
    binding.cmdline_parser = [](ModuleConfig &config, std::vector<std::string> &cmdline) {
      static_cast<T &>(config).parse_cmdline(cmdline);
    };
    binding.deleter = [](ModuleConfig *config) { delete static_cast<T *>(config); };
    return binding;
  }

  template <typename T>
  RealmStatus ModuleConfig::set_property(const std::string &name, T value)
  {
    const Binding *binding = find_binding();
    if(binding == nullptr)
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    const Property *prop = find(binding->properties, binding->num_properties, name);
    if(prop == nullptr) {
      log_moduleconfig.warning("Module %s does not have the configuration: %s",
                               module_name, name.c_str());
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    }
    if(!check_type(*prop, ModuleConfigDetail::PropertyTypeOf<T>::value))
      return REALM_MODULE_CONFIG_ERROR_INVALID_TYPE;
    *static_cast<T *>(prop->access(*this)) = value;
    return REALM_SUCCESS;
  }

  template <typename T>
  RealmStatus ModuleConfig::get_property(const std::string &name, T &value) const
  {
    value = T();
    const Binding *binding = find_binding();
    if(binding == nullptr)
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    const Property *prop = find(binding->properties, binding->num_properties, name);
    if(prop == nullptr) {
      log_moduleconfig.warning("Module %s does not have the configuration: %s",
                               module_name, name.c_str());
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    }
    if(!check_type(*prop, ModuleConfigDetail::PropertyTypeOf<T>::value))
      return REALM_MODULE_CONFIG_ERROR_INVALID_TYPE;
    // the accessor only computes a member address - we only read through it
    value = *static_cast<const T *>(prop->access(const_cast<ModuleConfig &>(*this)));
    return REALM_SUCCESS;
  }

  template <typename T>
  RealmStatus ModuleConfig::get_resource(const std::string &name, T &value) const
  {
    const Binding *binding = find_binding();
    if(binding == nullptr)
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    if((binding->resources_discovered == nullptr) ||
       !binding->resources_discovered(*this)) {
      log_moduleconfig.info("Module %s can not detect resources.", module_name);
      return REALM_MODULE_CONFIG_ERROR_NO_RESOURCE;
    }
    const Property *prop = find(binding->resources, binding->num_resources, name);
    if(prop == nullptr) {
      log_moduleconfig.warning("Module %s does not have the resource: %s", module_name,
                               name.c_str());
      return REALM_MODULE_CONFIG_ERROR_INVALID_NAME;
    }
    if(!check_type(*prop, ModuleConfigDetail::PropertyTypeOf<T>::value))
      return REALM_MODULE_CONFIG_ERROR_INVALID_TYPE;
    value = *static_cast<const T *>(prop->access(const_cast<ModuleConfig &>(*this)));
    return REALM_SUCCESS;
  }

}; // namespace Realm

#endif // ifndef REALM_MODULECONFIG_INL
