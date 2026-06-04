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
#include <optional>
#include <string>
#include <vector>

namespace Realm {

  // Named colors for NVTX events, as nvtx3::color values (fully opaque ARGB,
  // since rgb{} fills the alpha channel). Pass one of these to an annotation
  // call, or pass std::nullopt to fall back to the category's default color.
  namespace nvtx_color {
    inline constexpr nvtx3::color white{nvtx3::rgb{255, 255, 255}};
    inline constexpr nvtx3::color red{nvtx3::rgb{255, 0, 0}};
    inline constexpr nvtx3::color green{nvtx3::rgb{0, 255, 0}};
    inline constexpr nvtx3::color blue{nvtx3::rgb{0, 0, 255}};
    inline constexpr nvtx3::color purple{nvtx3::rgb{128, 0, 128}};
    inline constexpr nvtx3::color lawn_green{nvtx3::rgb{124, 252, 0}};
    inline constexpr nvtx3::color cyan{nvtx3::rgb{0, 255, 255}};
    inline constexpr nvtx3::color maroon{nvtx3::rgb{128, 0, 0}};
    inline constexpr nvtx3::color navy{nvtx3::rgb{0, 0, 128}};
    inline constexpr nvtx3::color magenta{nvtx3::rgb{255, 0, 255}};
    inline constexpr nvtx3::color yellow{nvtx3::rgb{255, 255, 0}};
    inline constexpr nvtx3::color gray{nvtx3::rgb{128, 128, 128}};
    inline constexpr nvtx3::color teal{nvtx3::rgb{0, 128, 128}};
    inline constexpr nvtx3::color olive{nvtx3::rgb{128, 128, 0}};
  } // namespace nvtx_color

  // The single NVTX domain that groups all of Realm's annotations. This is an
  // nvtx3 domain tag type: it just needs a static `name` member.
  struct realm_nvtx_domain {
    static constexpr char const *name{"Realm"};
  };

  // Sentinel payload meaning "no payload". NVTX ignores the payload value when
  // its type is NVTX_PAYLOAD_UNKNOWN, so events built with this carry no
  // payload at all (rather than an int32 value of 0). Used as the default for
  // the annotation calls below so callers can simply omit the payload.
  inline constexpr nvtx3::payload nvtx_no_payload{NVTX_PAYLOAD_UNKNOWN,
                                                  nvtx3::payload::value_type{}};

  // Metadata for one NVTX category. Immutable after construction. Categories are
  // created only during init_nvtx (on the main thread, before worker threads are
  // started), so no synchronization is needed.
  //
  // `category` carries the category id (via get_id()) and, on construction,
  // registers the id<->name association with NVTX so tools display the name.
  struct NvtxCategory {
    NvtxCategory(const std::string &category_name, uint32_t category_id,
                 nvtx3::color color)
      : name(category_name)
      , category(category_id, category_name.c_str())
      , default_color(color)
    {}
    std::string name;
    nvtx3::named_category_in<realm_nvtx_domain> category;
    nvtx3::color default_color;
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

  // Called by each kernel thread to name the OS thread in NVTX tools.
  void init_nvtx_thread(const char *thread_name);

  // Called by RuntimeImpl::configure_from_command_line on the main thread to
  // create the enabled categories. Must run before worker threads start.
  void init_nvtx(std::vector<std::string> &nvtx_modules);

  // Called by RuntimeImpl::wait_for_shutdown on the main thread.
  void finalize_nvtx(void);

  // Called by an application to get the next available category id.
  uint32_t nvtx_get_next_category_id(void);

  // Internal helpers for the inline annotation calls below. Kept in the header
  // (inline) so the thin wrappers fully inline at call sites; in particular the
  // common "disabled module" case (null category) collapses to nothing.

  // Build a fresh event on the stack for each annotation. NVTX consumes the
  // attributes synchronously, so there is no need to keep one around. An empty
  // `color` means "use the category's default color".
  inline nvtx3::event_attributes nvtx_make_event(const NvtxCategory &category,
                                                 const char *message,
                                                 std::optional<nvtx3::color> color,
                                                 nvtx3::payload payload)
  {
    return nvtx3::event_attributes{category.category,
                                   color.value_or(category.default_color), payload,
                                   nvtx3::message{message}};
  }

  inline nvtxDomainHandle_t nvtx_realm_domain()
  {
    return nvtx3::domain::get<realm_nvtx_domain>();
  }

  // RAII nested range. A null `category` (disabled module) pushes nothing and
  // the destructor pops nothing, keeping the NVTX range stack balanced.
  struct [[nodiscard]] nvtxScopedRange {
    nvtxScopedRange(NvtxCategory *category, char const *message,
                    nvtx3::payload payload = nvtx_no_payload) noexcept
      : active(false)
    {
      if(category) {
        nvtx3::event_attributes attr =
            nvtx_make_event(*category, message, std::nullopt, payload);
        nvtxDomainRangePushEx(nvtx_realm_domain(), attr.get());
        active = true;
      }
    }
    ~nvtxScopedRange()
    {
      if(active) {
        nvtxDomainRangePop(nvtx_realm_domain());
      }
    }

  private:
    bool active;
  };

  inline void nvtx_range_push(NvtxCategory *category, const char *message,
                              std::optional<nvtx3::color> color = std::nullopt,
                              nvtx3::payload payload = nvtx_no_payload)
  {
    if(!category) {
      return;
    }
    nvtx3::event_attributes attr = nvtx_make_event(*category, message, color, payload);
    nvtxDomainRangePushEx(nvtx_realm_domain(), attr.get());
  }

  inline void nvtx_range_pop(void) { nvtxDomainRangePop(nvtx_realm_domain()); }

  inline nvtx3::range_handle nvtx_range_start(NvtxCategory *category,
                                              const char *message,
                                              std::optional<nvtx3::color> color = std::nullopt,
                                              nvtx3::payload payload = nvtx_no_payload)
  {
    if(!category) {
      return nullptr;
    }
    nvtx3::event_attributes attr = nvtx_make_event(*category, message, color, payload);
    return nvtx3::start_range_in<realm_nvtx_domain>(attr);
  }

  inline void nvtx_range_end(nvtx3::range_handle id)
  {
    if(id) {
      nvtx3::end_range_in<realm_nvtx_domain>(id);
    }
  }

  inline void nvtx_mark(NvtxCategory *category, const char *message,
                        std::optional<nvtx3::color> color = std::nullopt,
                        nvtx3::payload payload = nvtx_no_payload)
  {
    if(!category) {
      return;
    }
    nvtx3::event_attributes attr = nvtx_make_event(*category, message, color, payload);
    nvtx3::mark_in<realm_nvtx_domain>(attr);
  }

}; // namespace Realm

#endif // NVTX_H
