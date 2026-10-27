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

// batched delivery of profiling responses (see
//  ProfilingRequest::set_batched_delivery) - internal to Realm

#ifndef REALM_PROFILING_BATCH_H
#define REALM_PROFILING_BATCH_H

#include "realm/processor.h"

namespace Realm {

  namespace ProfilingBatch {

    // the body of a delivery task (see Task::spawn_profiling_delivery): runs
    //  each response queued on 'queue' through
    //  exec(ctx, response_task_id, response, response_size)
    typedef void (*ExecFn)(void *ctx, Processor::TaskFuncID func_id, const void *args,
                           size_t arglen);
    void deliver(void *queue, ExecFn exec, void *ctx);

    // frees the queues (and drops undelivered responses) at shutdown
    void cleanup(void);

  }; // namespace ProfilingBatch

}; // namespace Realm

#endif
