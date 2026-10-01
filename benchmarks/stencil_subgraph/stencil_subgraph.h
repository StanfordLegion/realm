/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef STENCIL_SUBGRAPH_H
#define STENCIL_SUBGRAPH_H

#include "realm.h"

enum
{
  FID_INPUT = 100,
  FID_OUTPUT = 101,
};

struct StencilArgs {
  Realm::RegionInstance buffer = Realm::RegionInstance::NO_INST;
  Realm::IndexSpace<2> local_space = Realm::IndexSpace<2>();
  int64_t hx = -1, hy = -1;
};

struct IncrementArgs {
  Realm::RegionInstance buffer = Realm::RegionInstance::NO_INST;
  Realm::IndexSpace<2> local_space = Realm::IndexSpace<2>();
};

#ifdef REALM_USE_CUDA
void stencil_task_gpu(const void *_args, size_t arglen, const void *userdata,
                      size_t userlen, Realm::Processor p);
void increment_task_gpu(const void *_args, size_t arglen, const void *userdata,
                        size_t userlen, Realm::Processor p);
#endif

#endif
