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

#ifndef __LOADER_H__
#define __LOADER_H__

#include <type_traits>
#include <initializer_list>
#include <utility>
namespace Realm {

  typedef void *lib_handle_t;

  static constexpr lib_handle_t THIS_LIB = nullptr;

  enum LoadLibraryFlags
  {
    LOADLIB_NOW = 1,
    LOADLIB_DEFAULT = LOADLIB_NOW,
  };

  /// @brief Helper to load a library from the given name
  /// @param name Name of the library to load
  /// @param flags Extra flags for how the library should be loaded
  /// @return handle for the given library
  lib_handle_t load_library(const char *name, int flags = LOADLIB_DEFAULT);

  /// @brief Helper to close / free a library handle from \sa load_library
  /// @param hdl Handle to release
  void close_library(lib_handle_t hdl);

  /// @brief Helper to retrieve a symbol from a loaded library by it's symbol name
  /// @param hdl Handle to retrieve from
  /// @param name Name of the symbol to retrieve
  /// @return Address of the symbol, or nullptr if not found.
  void *get_symbol(lib_handle_t hdl, const char *name);

  /// @brief Class to hold a reference to a dynamically loaded library
  /// @tparam SymbolTable The class whose load_symbols() function will be called to
  /// initialize all it's symbols
  template <typename D>
  class Loader {
    lib_handle_t handle = nullptr;

  public:
    Loader() = default;
    Loader(lib_handle_t &hdl)
      : handle(std::move(hdl))
    {
      hdl = nullptr;
    }
    Loader(const Loader &) = delete;
    Loader(Loader &&) = delete;
    Loader &operator=(Loader &&) = delete;
    Loader &operator=(Loader &) = delete;

    ~Loader()
    {
      if(handle != nullptr) {
        Realm::close_library(handle);
      }
    }
    operator bool() const { return handle != nullptr; }

    /// @brief Initializes the loader from the given name
    bool load(const char *name, int flags = LOADLIB_DEFAULT)
    {
      handle = Realm::load_library(name, flags);
      if(handle == nullptr) {
        return false;
      }
      if(!static_cast<D *>(this)->load_symbols()) {
        Realm::close_library(handle);
        handle = nullptr;
      }
      return handle != nullptr;
    }

    /// @brief Initializes the loader from the given names
    bool load(std::initializer_list<const char *> names, int flags = LOADLIB_DEFAULT)
    {
      for(const char *name : names) {
        if(load(name, flags)) {
          return true;
        }
      }
      return false;
    }

    /// @brief Gets the address of the symbol for the given symbol name
    /// @tparam T type of the object this symbol references (must be a pointer type)
    /// @param name Name of the symbol this references
    /// @return True if successful, false otherwise.
    template <typename T, typename = std::enable_if_t<std::is_pointer<T>::value>>
    bool get_symbol(const char *name, T &ptr)
    {
      ptr = reinterpret_cast<T>(Realm::get_symbol(handle, name));
      return ptr != nullptr;
    }
  };

} // namespace Realm
#endif // __LOADER_H__