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

#include "realm_bootstrap.h"
#include <cstring>
#include <cstdlib>

namespace App {

static int app_get(bootstrap_handle_t *handle, int target_rank, const char *key,
                   int timeout_ms, bootstrap_blob_t *out_val)
{
  if(!key || !out_val) {
    return -1;
  }
  
  const char *value = getenv(key);
  if(!value) {
    return -1;
  }
  
  out_val->bytes = (void*)value;
  out_val->len = strlen(value);
  return 0;
}

static int app_init(bootstrap_handle_t *handle, const char *config)
{
  // example keys app could set for Realm to read via handle->get()
  setenv("REALM_COMM_MODE", "env", 1);
  setenv("REALM_BOOTSTRAP_TYPE", "mpi", 1);
  
  // for now, still need this for Realm to actually work
  setenv("REALM_UCP_BOOTSTRAP_MODE", "mpi", 1);
  return 0;
}

int bootstrap_init(bootstrap_handle_t *handle)
{
  if(!handle) {
    return -1;
  }

  memset(handle, 0, sizeof(*handle));

  handle->get = app_get;
  handle->init = app_init;
  handle->put = nullptr;
  handle->join = nullptr;
  handle->stop = nullptr;
  handle->remove = nullptr;
  handle->watch = nullptr;
  handle->get_rank = nullptr;
  handle->get_size = nullptr;

  return 0;
}

int bootstrap_finalize(bootstrap_handle_t *handle)
{
  return 0;
}

} // namespace App

