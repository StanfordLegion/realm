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

#ifndef __REALM_STENCIL__
#define __REALM_STENCIL__

#ifndef DTYPE
#define DTYPE double
#endif

#ifndef RESTRICT
#define RESTRICT
#endif

#ifndef RADIUS
#define RADIUS (2)
#endif

#ifdef ENABLE_PROFILING
#include "realm/prealm/prealm.h"
namespace realm = PRealm;
#else
#include "realm.h"
namespace realm = Realm;
#endif

using namespace realm;

#include "cpu_kernels.h" // for coord_t

typedef realm::Point<1, coord_t> Point1;
typedef realm::Point<2, coord_t> Point2;
typedef realm::Rect<1, coord_t> Rect1;
typedef realm::Rect<2, coord_t> Rect2;

struct CreateRegionArgs {
public:
  Rect2 bounds;
  realm::Memory memory;
  realm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  realm::RegionInstance *dest_inst;
};

struct CreateRegionDoneArgs {
public:
  realm::RegionInstance inst;
  realm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  realm::RegionInstance *dest_inst;
};

struct ShardArgs {
public:
  realm::RegionInstance xp_inst_in, xm_inst_in, yp_inst_in, ym_inst_in;
  realm::RegionInstance xp_inst_out, xm_inst_out, yp_inst_out, ym_inst_out;
  realm::Barrier xp_empty_in, xm_empty_in, yp_empty_in, ym_empty_in;
  realm::Barrier xp_empty_out, xm_empty_out, yp_empty_out, ym_empty_out;
  realm::Barrier xp_full_in, xm_full_in, yp_full_in, ym_full_in;
  realm::Barrier xp_full_out, xm_full_out, yp_full_out, ym_full_out;
  realm::Barrier sync, first_start, last_start, first_stop, last_stop;
  coord_t tsteps, tprune, init;
  Point2 point;
  Rect2 interior_bounds, exterior_bounds, outer_bounds;
  realm::Memory sysmem, regmem;
};

struct StencilArgs {
public:
  realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 interior_bounds;
  DTYPE *weights;
};

struct IncrementArgs {
public:
  realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 outer_bounds;
};

struct CheckArgs {
public:
  realm::RegionInstance private_inst;
  coord_t tsteps, init;
  Rect2 interior_bounds;
};

#endif
