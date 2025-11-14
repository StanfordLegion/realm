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

#include <realm/loader.h>
#include <realm/realm_config.h>

#if defined(REALM_ON_WINDOWS)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Realm {

  lib_handle_t load_library(const char *name, int flags)
  {
#if defined(REALM_ON_WINDOWS)
    return static_cast<lib_handle_t>(LoadLibrary(name));
#elif defined(REALM_USE_DLFCN)
    int dlopen_flags = 0;
    if(flags & LOADLIB_NOW) {
      dlopen_flags |= RTLD_NOW;
    } else {
      dlopen_flags |= RTLD_LAZY;
    }
    lib_handle_t handle = dlopen(name, dlopen_flags);
    if(handle == nullptr) {
      dlerror(); // Clear error
    }
    return handle;
#else
    return nullptr;
#endif
  }

  void close_library(lib_handle_t hdl)
  {
#if defined(REALM_ON_WINDOWS)
    FreeLibrary(static_cast<HMODULE>(hdl));
#elif defined(REALM_USE_DLFCN)
    dlclose(hdl);
#endif
  }

  void *get_symbol(lib_handle_t hdl, const char *name)
  {
#if defined(REALM_ON_WINDOWS)
    if(hdl == THIS_LIB) {
      hdl = reinterpret_cast<lib_handle_t>(GetModuleHandle(nullptr));
    }
    return GetProcAddress(static_cast<HMODULE>(hdl), name);
#elif defined(REALM_USE_DLFCN)
    void *sym = dlsym(hdl, name);
    if(sym == nullptr) {
      dlerror();
    }
    return sym;
#else
    return nullptr;
#endif
  }

} // namespace Realm