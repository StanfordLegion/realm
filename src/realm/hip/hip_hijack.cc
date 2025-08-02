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

// Our implementation of the HIP runtime API for Legion
// so we can intercept all of these calls

// All these extern C methods are for internal implementations
// of functions of the cuda runtime API that nvcc assumes
// exists and can be used for code generation. They are all
// pretty simple to map to the driver API.

// NOTE: This file must only include CUDA runtime API calls and any helpers used
//  ONLY by those calls - there may be NO references to symbols defined in this file
//  by any other parts of Realm, or it will mess up the ability to let an app link
//  against the real libcudart.so

#include "realm/hip/hip_hijack.h"

#include "realm/hip/hip_module.h"
#include "realm/hip/hip_internal.h"
#include "realm/logging.h"

namespace Realm {
  namespace Hip {

    extern Logger log_cudart;
    extern Logger log_stream;

    static GPUProcessor *get_gpu_or_die(const char *funcname)
    {
      // mark that the hijack code is active - this covers the calls below
      cudart_hijack_active = true;

      GPUProcessor *p = GPUProcessor::get_current_gpu_proc();
      if(!p) {
        log_cudart.fatal() << funcname << "() called outside HIP task";
        assert(false);
      }
      return p;
    }

  }; // namespace Hip
};   // namespace Realm

// these are all "C" functions
extern "C" {

using namespace Realm;
using namespace Realm::Hip;

REALM_PUBLIC_API
hipError_t hipMemcpy_H(void *dst, const void *src, size_t size, hipMemcpyKind kind)
{
  printf("in hipMemcpy_H\n");
  GPUProcessor *p = get_gpu_or_die("cudaMemcpy");
  p->gpu_memcpy(dst, src, size, kind);
  return hipSuccess;
}

REALM_PUBLIC_API
hipError_t hipMemcpyAsync_H(void *dst, const void *src, size_t size, hipMemcpyKind kind,
                            hipStream_t stream)
{
  printf("in hipMemcpyAsync_H\n");
  GPUProcessor *p = get_gpu_or_die("cudaMemcpy");
  p->gpu_memcpy_async(dst, src, size, kind, stream);
  return hipSuccess;
}

REALM_PUBLIC_API
hipStream_t hipGetTaskStream()
{
  hipStream_t stream = HIPModule::get_task_hip_stream();
  HIPModule::set_task_ctxsync_required(false);
  return stream;
}
}; // extern "C"
