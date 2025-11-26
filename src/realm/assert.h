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

// atomics for Realm - this is a simple wrapper around C++11's std::atomic
//  if available and uses gcc's __sync_* primitives otherwise

#ifndef REALM_ASSERT_H
#define REALM_ASSERT_H

#if 1
#if(defined(__CUDACC__) && defined(__CUDA_ARCH__)) ||                                    \
    (defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__))
#define REALM_ASSERT(cond)                                                               \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      __trap();                                                                          \
    }                                                                                    \
  } while(0)
#else
#include "realm/logging.h"

namespace Realm {
  extern Logger log_runtime;
} // namespace Realm

#define REALM_ASSERT(cond)                                                               \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      Realm::log_runtime.fatal("Assertion failed: (%s), at %s:%d", #cond, __FILE__,      \
                               __LINE__);                                                \
      abort();                                                                           \
    }                                                                                    \
  } while(0)
#endif
#else
#include <assert.h>
#define REALM_ASSERT(cond) assert(cond)
#endif

#endif // REALM_ASSERT_H