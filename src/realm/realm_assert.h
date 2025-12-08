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

#ifndef REALM_ASSERT_H
#define REALM_ASSERT_H

#include "realm_config.h"

#ifdef NDEBUG
// =============================================
// Device-side (CUDA or HIP)
// =============================================
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

// Clang CUDA or HIP: __assert_fail is available
#if defined(__clang__)
#include <assert.h>
#define REALM_ASSERT(cond)                                                               \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      __assert_fail(#cond, __FILE__, __LINE__, __func__);                                \
    }                                                                                    \
  } while(0)

// NVCC CUDA: use trap
#elif defined(__CUDACC__)
#define REALM_ASSERT(cond)                                                               \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      __trap();                                                                          \
    }                                                                                    \
  } while(0)

#else
#error "Unknown device compilation environment"
#endif

// =============================================
// Host-side
// =============================================
#else
namespace Realm {
  REALM_INTERNAL_API_EXTERNAL_LINKAGE void realm_assert_fail(const char* cond_text, const char* file, int line);
}

#define REALM_ASSERT(cond)                                                               \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      Realm::realm_assert_fail(#cond, __FILE__, __LINE__);                               \
      abort();                                                                           \
    }                                                                                    \
  } while(0)

#endif
#else
#include <assert.h>
#define REALM_ASSERT(cond) assert(cond)
#endif

#endif // REALM_ASSERT_H