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

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <pmix.h>

#include "bootstrap.h"
#include "bootstrap_util.h"

#define BOOTSTRAP_PMIX_KEYSIZE 64

static pmix_proc_t myproc;

static pmix_status_t bootstrap_pmix_exchange(void)
{
  pmix_status_t status;
  pmix_info_t info;
  bool flag = true;

  status = PMIx_Commit();
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Commit failed\n");

  PMIX_INFO_CONSTRUCT(&info);
  PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

  status = PMIx_Fence(NULL, 0, &info, 1);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, destruct_info,
                         "PMIx_Fence failed\n");

destruct_info:
  PMIX_INFO_DESTRUCT(&info);
out:
  return status;
}

static pmix_status_t bootstrap_pmix_put(const char *key, const void *value,
                                        size_t valuelen)
{
  pmix_value_t val;
  pmix_status_t status;

  PMIX_VALUE_CONSTRUCT(&val);
  val.type = PMIX_BYTE_OBJECT;
  val.data.bo.bytes = (char *)value;
  val.data.bo.size = valuelen;

  status = PMIx_Put(PMIX_GLOBAL, key, &val);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, cleanup_val,
                         "PMIx_Put failed\n");

cleanup_val:
  val.data.bo.bytes = NULL; // protect the data
  val.data.bo.size = 0;
  PMIX_VALUE_DESTRUCT(&val);

  return status;
}

static pmix_status_t bootstrap_pmix_get(int pe, const char *key, void *value,
                                        size_t valuelen)
{
  pmix_proc_t proc;
  pmix_value_t *val;
  pmix_status_t status;

  /* ensure the region is zero'd out */
  memset(value, 0, valuelen);

  /* setup the ID of the proc whose info we are getting */
  PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);

  proc.rank = (uint32_t)pe;

  status = PMIx_Get(&proc, key, NULL, 0, &val);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Get failed\n");

  if(val == NULL) {
    goto out;
  }

  /* see if the data fits into the given region */
  if(valuelen < val->data.bo.size) {
    status = PMIX_ERROR;
    goto rel_val;
  }

  /* copy the results across */
  memcpy(value, val->data.bo.bytes, val->data.bo.size);

rel_val:
  PMIX_VALUE_RELEASE(val);
out:
  return status;
}

static int bootstrap_pmix_allgather(const void *sendbuf, void *recvbuf, int length,
                                    bootstrap_handle_t *handle)
{
  static int key_index = 1;
  pmix_status_t status;
  char key[BOOTSTRAP_PMIX_KEYSIZE];

  if(handle->pg_size == 1) {
    memcpy(recvbuf, sendbuf, length);
    return 0;
  }

  snprintf(key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-ALLGATHER-%04x", key_index);

  status = bootstrap_pmix_put(key, sendbuf, length);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "bootstrap_pmix_put failed\n");

  status = bootstrap_pmix_exchange();
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "bootstrap_pmix_exchange failed\n");

  for(int i = 0; i < handle->pg_size; i++) {
    // assumes that same length is passed by all the processes
    status = bootstrap_pmix_get(i, key, (char *)recvbuf + length * i, length);
    BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                           "bootstrap_pmix_get failed\n");
  }

out:
  key_index++;
  return status;
}

static int populate_shared_ranks(bootstrap_handle_t *handle)
{
  pmix_status_t status;
  pmix_value_t *val;
  pmix_proc_t proc, *local_procs;

  handle->shared_ranks = NULL;
  handle->num_shared_ranks = 0;

  /* set PMIX_RANK_WILDCARD because older PMIx versions need it
   * https://github.com/openpmix/openpmix/pull/3323
   */
  PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);
  proc.rank = PMIX_RANK_WILDCARD;

  status = PMIx_Get(&proc, PMIX_LOCAL_PROCS, NULL, 0, &val);
  if(status == PMIX_ERR_NOT_FOUND) {
    /* Older or trimmed PMIx builds may not provide PMIX_LOCAL_PROCS.
     * Fall back to PMIX_LOCAL_PEERS (comma-separated ranks).
     */
    status = PMIx_Get(&proc, PMIX_LOCAL_PEERS, NULL, 0, &val);
    if(status == PMIX_ERR_NOT_FOUND) {
      status = PMIX_SUCCESS;
      goto out; /* no local peers information available */
    }
    BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                           "PMIx_Get(PMIX_LOCAL_PEERS) failed\n");

    if(val && val->type == PMIX_STRING && val->data.string && val->data.string[0] != '\0') {
      char *dup = strdup(val->data.string);
      BOOTSTRAP_NULL_ERROR_JMP(dup, status, BOOTSTRAP_ERROR_INTERNAL, rel_val,
                               "Failed to duplicate local peers string\n");

      char *saveptr = NULL;
      char *tok = strtok_r(dup, ",", &saveptr);
      while(tok) {
        uint32_t *tmp = realloc(handle->shared_ranks,
                                (handle->num_shared_ranks + 1) * sizeof(handle->shared_ranks[0]));
        BOOTSTRAP_NULL_ERROR_JMP(tmp, status, BOOTSTRAP_ERROR_INTERNAL, free_dup,
                                 "Failed to allocate space for shared ranks\n");
        handle->shared_ranks = tmp;
        handle->shared_ranks[handle->num_shared_ranks++] = (uint32_t)strtoul(tok, NULL, 10);
        tok = strtok_r(NULL, ",", &saveptr);
      }

free_dup:
      free(dup);
      status = PMIX_SUCCESS;
    }
    goto rel_val;
  }
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Get failed\n");

  // val must be an array of pmix_proc_t
  assert(val->type == PMIX_DATA_ARRAY);
  assert(val->data.darray->type == PMIX_PROC);

  local_procs = (pmix_proc_t *)val->data.darray->array;
  handle->num_shared_ranks = val->data.darray->size;

  if(handle->num_shared_ranks > 0) {
    handle->shared_ranks =
        malloc(handle->num_shared_ranks * sizeof(handle->shared_ranks[0]));
    BOOTSTRAP_NULL_ERROR_JMP(handle->shared_ranks, status, BOOTSTRAP_ERROR_INTERNAL,
                             rel_val, "Failed to allocate space for shared ranks\n");

    for(int i = 0; i < handle->num_shared_ranks; i++) {
      handle->shared_ranks[i] = local_procs[i].rank;
    }
  }

rel_val:
  PMIX_VALUE_RELEASE(val);
out:
  return status;
}

static int bootstrap_pmix_finalize(bootstrap_handle_t *handle)
{
  pmix_status_t status = PMIx_Finalize(NULL, 0);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Finalize failed\n");

out:
  return status;
}

int realm_ucp_bootstrap_plugin_init(void *attr, bootstrap_handle_t *handle)
{
  pmix_status_t status;
  pmix_value_t *val;
  pmix_proc_t proc;

  assert(PMIX_MAX_KEYLEN >= BOOTSTRAP_PMIX_KEYSIZE);

  PMIX_PROC_CONSTRUCT(&myproc);

  status = PMIx_Init(&myproc, NULL, 0);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Init failed\n");

  PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);
  proc.rank = PMIX_RANK_WILDCARD;

  status = PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
  BOOTSTRAP_NE_ERROR_JMP(status, PMIX_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "PMIx_Get(PMIX_JOB_SIZE) failed\n");

  status = populate_shared_ranks(handle);
  //BOOTSTRAP_NZ_ERROR_JMP(status, BOOTSTRAP_ERROR_INTERNAL, rel_val,
                         ///"populate_shared_ranks failed\n");

  handle->pg_rank = myproc.rank;
  handle->pg_size = val->data.uint32;
  handle->finalize = bootstrap_pmix_finalize;
  handle->allgather = bootstrap_pmix_allgather;

rel_val:
  PMIX_VALUE_RELEASE(val);

out:
  return status;
}
