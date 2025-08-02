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

#ifndef __REALM_SAXPY__
#define __REALM_SAXPY__

#ifdef ENABLE_PROFILING
#include "realm/prealm/prealm.h"
namespace realm = PRealm;
#else
#include "realm.h"
namespace realm = Realm;
#endif

using namespace realm;

enum
{
  FID_X = 101,
  FID_Y = 102,
  FID_Z = 103,
};

struct SaxpyArgs {
public:
  realm::RegionInstance x_inst, y_inst, z_inst;
  float alpha;
  realm::Rect<1> bounds;
};

#endif
