/*
 * Copyright 2024 NVIDIA Corporation
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

#include <stdlib.h>
#include <limits.h>
#include <mpi.h>

#include "bootstrap.h"
#include "bootstrap_util.h"

#if !defined(REALM_MPI_HAS_COMM_SPLIT_TYPE)
#if(OMPI_MAJOR_VERSION * 100 + OMPI_MINOR_VERSION) >= 107
#define REALM_MPI_HAS_COMM_SPLIT_TYPE 1
#endif
#endif

static int bootstrap_mpi_finalize(bootstrap_handle_t *handle)
{
  int status = MPI_SUCCESS;

    MPI_Finalize();

  return status;
}

int realm_ucp_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle)
{
  int status = MPI_SUCCESS;

  handle->finalize = bootstrap_mpi_finalize;

  int provided_level;
  //   // Legate collInit requires MPI_THREAD_MULTIPLE
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided_level);

  return status;
}
