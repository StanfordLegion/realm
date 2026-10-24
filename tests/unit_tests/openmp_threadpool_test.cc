/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

#include <gtest/gtest.h>

#include "realm/hardware_topology.h"
#include "realm/openmp/openmp_threadpool.h"

using namespace Realm;

namespace {

  class OpenMPThreadPoolTest : public ::testing::Test {
  protected:
    // Synthetic cores avoid affinity restrictions and work on small CI runners.
    static HardwareTopology make_topology()
    {
      std::vector<HardwareTopology::Proc> cores;
      for(int i = 0; i < 4; i++) {
        HardwareTopology::Proc core{};
        core.id = i;
        core.domain = 0;
        cores.push_back(core);
      }
      const size_t memory_size = 1 << 20;
      return HardwareTopology(cores, {{memory_size, 0}}, memory_size);
    }

    HardwareTopology topology = make_topology();
    CoreReservationSet reservations{&topology};
  };

  TEST_F(OpenMPThreadPoolTest, ShutdownAfterDeferredStartup)
  {
    ThreadPool pool(Processor::NO_PROC, 4, "openmp test", -1, 0, reservations);

    // Match runtime startup: creation is deferred until reservations are satisfied.
    pool.start_worker_threads();
    ASSERT_TRUE(reservations.satisfy_reservations());
    pool.stop_worker_threads();
  }

  TEST_F(OpenMPThreadPoolTest, ImmediateShutdownAndRestart)
  {
    ThreadPool pool(Processor::NO_PROC, 4, "openmp test", -1, 0, reservations);
    ASSERT_TRUE(reservations.satisfy_reservations());

    // No work submission or barrier may hide a missing startup/shutdown handoff.
    // Reusing the pool also checks that every handle is joined and cleared.
    for(int i = 0; i < 100; i++) {
      pool.start_worker_threads();
      pool.stop_worker_threads();
    }
  }

  TEST_F(OpenMPThreadPoolTest, EmptyPool)
  {
    ThreadPool pool(Processor::NO_PROC, 0, "openmp test", -1, 0, reservations);
    pool.start_worker_threads();
    pool.stop_worker_threads();
    pool.stop_worker_threads();
  }

} // namespace
