/*
 * Copyright 2026 Stanford University
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

#include "realm/hardware_topology.h"
#include "realm/openmp/openmp_threadpool.h"
#include "gtest/gtest.h"

using namespace Realm;

TEST(OpenMPThreadPoolTest, StopImmediatelyAfterStart)
{
  HardwareTopology topology = HardwareTopology::create_topology();
  CoreReservationSet reservations(&topology);
  // Zero requests PTHREAD_STACK_MIN, which is too small for Realm on FreeBSD.
  const size_t stack_size = 1 << 20;
  ThreadPool pool(Processor::NO_PROC, 2, "startup/shutdown test", -1, stack_size,
                  reservations);
  ASSERT_TRUE(reservations.satisfy_reservations(true));

  // Shutdown must also handle workers that have not reached their entry point.
  for(int i = 0; i < 100; i++) {
    pool.start_worker_threads();
    pool.stop_worker_threads();
  }
}
