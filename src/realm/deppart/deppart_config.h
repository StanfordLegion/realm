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

// implementation sparsity maps

#ifndef REALM_DEPPART_CONFIG_H
#define REALM_DEPPART_CONFIG_H

#include <cstddef>

namespace Realm {

  namespace DeppartConfig {

    extern int cfg_num_partitioning_workers;
    extern bool cfg_disable_intersection_optimization;
    extern int cfg_max_rects_in_approximation;
    extern bool cfg_worker_threads_sleep;
    extern size_t cfg_host_pool_size;

  };

  void *deppart_host_alloc_bytes(size_t bytes, size_t alignment);
  void deppart_host_free(void *ptr);
  void deppart_host_pool_shutdown(void);

  template <typename T>
  inline T *deppart_host_alloc(size_t count)
  {
    if(count == 0)
      return nullptr;
    return static_cast<T *>(deppart_host_alloc_bytes(count * sizeof(T), alignof(T)));
  }

}; // namespace Realm

#endif // defined REALM_DEPPART_CONFIG_H
