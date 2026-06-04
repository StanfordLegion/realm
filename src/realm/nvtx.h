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

#ifndef NVTX_H
#define NVTX_H

#include "realm/realm_config.h"

// Realm's NVTX support is built on the nvtx3 C++ API. The C-only headers
// (<nvtx3/nvToolsExt.h>) are intentionally not used as a fallback.
#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#else
#error "Realm NVTX support requires the nvtx3 C++ headers (<nvtx3/nvtx3.hpp>)"
#endif

#include <map>
#include <string>
#include <vector>

namespace Realm {

  struct NvtxARGB {
    constexpr NvtxARGB(uint8_t red_, uint8_t green_, uint8_t blue_,
                       uint8_t alpha_ = 0xFF) noexcept
      : red{red_}
      , green{green_}
      , blue{blue_}
      , alpha{alpha_}
    {}
    constexpr uint32_t to_uint(void) const
    {
      return uint32_t{alpha} << 24 | uint32_t{red} << 16 | uint32_t{green} << 8 |
             uint32_t{blue};
    }
    uint8_t const red{};
    uint8_t const green{};
    uint8_t const blue{};
    uint8_t const alpha{};
  };

  enum nvtx_color : uint32_t
  {
    white = NvtxARGB(255, 255, 255).to_uint(),
    red = NvtxARGB(255, 0, 0).to_uint(),
    green = NvtxARGB(0, 255, 0).to_uint(),
    blue = NvtxARGB(0, 0, 255).to_uint(),
    purple = NvtxARGB(128, 0, 128).to_uint(),
    lawn_green = NvtxARGB(124, 252, 0).to_uint(),
    cyan = NvtxARGB(0, 255, 255).to_uint(),
    maroon = NvtxARGB(128, 0, 0).to_uint(),
    navy = NvtxARGB(0, 0, 128).to_uint(),
    magenta = NvtxARGB(255, 0, 255).to_uint(),
    yellow = NvtxARGB(255, 255, 0).to_uint(),
    gray = NvtxARGB(128, 128, 128).to_uint(),
    teal = NvtxARGB(0, 128, 128).to_uint(),
    olive = NvtxARGB(128, 128, 0).to_uint(),
  };

  // The single NVTX domain that groups all of Realm's annotations. This is an
  // nvtx3 domain tag type: it just needs a static `name` member.
  struct realm_nvtx_domain {
    static constexpr char const *name{"Realm"};
  };

  // Metadata for one NVTX category. Immutable after construction. Categories are
  // created only during init_nvtx (on the main thread, before worker threads are
  // started), so no synchronization is needed.
  //
  // `category` carries the category id (via get_id()) and, on construction,
  // registers the id<->name association with NVTX so tools display the name.
  struct NvtxCategory {
    NvtxCategory(const std::string &category_name, uint32_t category_id, uint32_t color)
      : name(category_name)
      , category(category_id, category_name.c_str())
      , default_color(color)
    {}
    std::string name;
    nvtx3::named_category_in<realm_nvtx_domain> category;
    uint32_t default_color;
  };

  // Handles to the predefined categories. Each is non-null only if its module
  // was enabled via -ll:nvtx_modules; otherwise it stays null and annotation
  // calls on it are cheap no-ops. Call sites pass these directly, with no name
  // lookup. Defined in nvtx.cc.
  extern NvtxCategory *nvtx_amsg;
  extern NvtxCategory *nvtx_bgwork;
#ifdef REALM_USE_CUDA
  extern NvtxCategory *nvtx_cuda;
#endif
#ifdef REALM_USE_HIP
  extern NvtxCategory *nvtx_hip;
#endif
#ifdef REALM_USE_GASNET1
  extern NvtxCategory *nvtx_gasnet1;
#endif
#ifdef REALM_USE_GASNETEX
  extern NvtxCategory *nvtx_gasnetex;
#endif
#ifdef REALM_USE_MPI
  extern NvtxCategory *nvtx_mpi;
#endif
#ifdef REALM_USE_OPENMP
  extern NvtxCategory *nvtx_openmp;
#endif
#ifdef REALM_USE_PYTHON
  extern NvtxCategory *nvtx_python;
#endif

  // RAII nested range. A null `category` (disabled module) pushes nothing and
  // the destructor pops nothing, keeping the NVTX range stack balanced.
  struct nvtxScopedRange {
    nvtxScopedRange(NvtxCategory *category, char const *message, int32_t payload = 0);
    ~nvtxScopedRange();

  private:
    bool active;
  };

  // Base category id for application-defined categories (predefined ones use
  // small ids). An application that wants its own category can construct an
  // NvtxCategory with an id >= this value and pass its address to the calls below.
  static constexpr uint32_t nvtx_proc_starting_category_id = 1000;

  // Called by each kernel thread to name the OS thread in NVTX tools.
  void init_nvtx_thread(const char *thread_name);

  // Called by RuntimeImpl::configure_from_command_line on the main thread to
  // create the enabled categories. Must run before worker threads start.
  void init_nvtx(std::vector<std::string> &nvtx_modules);

  // Called by RuntimeImpl::wait_for_shutdown on the main thread.
  void finalize_nvtx(void);

  void nvtx_range_push(NvtxCategory *category, const char *message,
                       uint32_t color = nvtx_color::white, int32_t payload = 0);

  void nvtx_range_pop(void);

  nvtx3::range_handle nvtx_range_start(NvtxCategory *category, const char *message,
                                       uint32_t color = nvtx_color::white,
                                       int32_t payload = 0);

  void nvtx_range_end(nvtx3::range_handle id);

  void nvtx_mark(NvtxCategory *category, const char *message,
                 uint32_t color = nvtx_color::white, int32_t payload = 0);

}; // namespace Realm

#endif // NVTX_H
