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

#include "realm/nvtx.h"

#include <functional>
#include <iostream>
#include <memory>
#include "realm/atomics.h"
#ifdef REALM_ON_WINDOWS
#include <processthreadsapi.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

namespace Realm {

  // Definitions of the predefined category handles declared in the header.
  NvtxCategory *nvtx_amsg = nullptr;
  NvtxCategory *nvtx_bgwork = nullptr;
#ifdef REALM_USE_CUDA
  NvtxCategory *nvtx_cuda = nullptr;
#endif
#ifdef REALM_USE_HIP
  NvtxCategory *nvtx_hip = nullptr;
#endif
#ifdef REALM_USE_GASNET1
  NvtxCategory *nvtx_gasnet1 = nullptr;
#endif
#ifdef REALM_USE_GASNETEX
  NvtxCategory *nvtx_gasnetex = nullptr;
#endif
#ifdef REALM_USE_MPI
  NvtxCategory *nvtx_mpi = nullptr;
#endif
#ifdef REALM_USE_OPENMP
  NvtxCategory *nvtx_openmp = nullptr;
#endif
#ifdef REALM_USE_PYTHON
  NvtxCategory *nvtx_python = nullptr;
#endif

  // Static description of a predefined category: its id, default color, and the
  // global handle to point at the created object. `slot` is a reference to one
  // of the `nvtx_*` pointers defined above.
  struct nvtx_category_def {
    uint32_t id;
    nvtx3::color color;
    std::reference_wrapper<NvtxCategory *> slot;
  };

  static const std::map<std::string, nvtx_category_def> nvtx_categories_predefined = {
      {"amsg", {1, nvtx_color::red, std::ref(nvtx_amsg)}},
      {"bgwork", {2, nvtx_color::blue, std::ref(nvtx_bgwork)}},
#ifdef REALM_USE_CUDA
      {"cuda", {100, nvtx_color::green, std::ref(nvtx_cuda)}},
#endif
#ifdef REALM_USE_HIP
      {"hip", {101, nvtx_color::purple, std::ref(nvtx_hip)}},
#endif
#ifdef REALM_USE_GASNET1
      {"gasnet1", {102, nvtx_color::lawn_green, std::ref(nvtx_gasnet1)}},
#endif
#ifdef REALM_USE_GASNETEX
      {"gasnetex", {103, nvtx_color::cyan, std::ref(nvtx_gasnetex)}},
#endif
#ifdef REALM_USE_MPI
      {"mpi", {104, nvtx_color::maroon, std::ref(nvtx_mpi)}},
#endif
#ifdef REALM_USE_OPENMP
      {"openmp", {105, nvtx_color::navy, std::ref(nvtx_openmp)}},
#endif
#ifdef REALM_USE_PYTHON
      {"python", {106, nvtx_color::magenta, std::ref(nvtx_python)}},
#endif
  };

  // Owns the category objects created at init; cleared at finalize. The global
  // `nvtx_*` handles point into these.
  static std::vector<std::unique_ptr<NvtxCategory>> nvtx_owned_categories;

  static std::vector<std::string> enabled_nvtx_modules;

  static atomic<uint32_t> nvtx_proc_starting_category_id{1000};

  // Create the category object, take ownership, and point its global handle at it.
  static void create_category(const std::string &name, const nvtx_category_def &def)
  {
    nvtx_owned_categories.push_back(
        std::make_unique<NvtxCategory>(name, def.id, def.color));
    def.slot.get() = nvtx_owned_categories.back().get();
  }

  void init_nvtx_thread(const char *thread_name)
  {
#ifdef REALM_ON_WINDOWS
    nvtxNameOsThread(GetCurrentThreadId(), thread_name);
#else
    nvtxNameOsThread(pthread_self(), thread_name);
#endif
  }

  void init_nvtx(std::vector<std::string> &nvtx_modules)
  {
    enabled_nvtx_modules = nvtx_modules;

    if(enabled_nvtx_modules.size() == 1 && enabled_nvtx_modules[0] == "all") {
      // handle -ll:nvtx_modules all
      for(const auto &entry : nvtx_categories_predefined) {
        create_category(entry.first, entry.second);
      }
    } else {
      for(const std::string &name : enabled_nvtx_modules) {
        if(name == "all") {
          std::cerr << "If all specified, then no other modules are needed." << std::endl;
          abort();
        }
        std::map<std::string, nvtx_category_def>::const_iterator it =
            nvtx_categories_predefined.find(name);
        if(it == nvtx_categories_predefined.end()) {
          std::cerr << "Unable to find specified nvtx module: " << name << std::endl;
          abort();
        }
        create_category(it->first, it->second);
      }
    }

    init_nvtx_thread("MainThread");
  }

  void finalize_nvtx(void)
  {
    // The nvtx3 domain is intentionally never destroyed (see nvtx3 docs). Drop
    // the owned categories and reset the global handles so none dangle.
    nvtx_owned_categories.clear();
    for(const auto &entry : nvtx_categories_predefined) {
      entry.second.slot.get() = nullptr;
    }
  }

  uint32_t nvtx_get_next_category_id(void)
  {
    return nvtx_proc_starting_category_id.fetch_add(1);
  }

}; // namespace Realm
