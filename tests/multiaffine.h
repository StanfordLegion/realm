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

#ifndef MULTIAFFINE_H
#define MULTIAFFINE_H

#include "realm.h"

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0,
  PTR_WRITE_TASK_BASE,
};

enum
{
  FID_BASE = 44,
  FID_ADDR,
};

template <int N, typename T>
struct PtrWriteTaskArgs {
  Realm::IndexSpace<N, T> space;
  Realm::RegionInstance inst;
};

#endif
