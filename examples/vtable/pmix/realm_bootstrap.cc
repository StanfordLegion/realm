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

#include "realm_bootstrap.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <pmix.h>

namespace App {

  // Realm KeyValueStoreVtable required keys
  static constexpr const char *REALM_KEY_RANK = "realm_rank";
  static constexpr const char *REALM_KEY_RANKS = "realm_ranks";
  static constexpr const char *REALM_KEY_GROUP = "realm_group";

  struct VtableContext {
    // pending_sync is an optimization to avoid redundant PMIx_Commit() calls
    bool pending_sync = false; 
    pmix_proc_t myproc;
    uint32_t pmix_rank = 0;
    uint32_t pmix_size = 0;
  };

  static bool app_put(const void *key, size_t key_size, const void *value,
                      size_t value_size, const void *vtable_data, size_t vtable_data_size)
  {
    if(!key || !value || key_size == 0 || !vtable_data)
      return false;
    VtableContext *ctx = *(VtableContext **)vtable_data;

    std::string k(static_cast<const char *>(key), key_size);

    pmix_value_t val;
    val.type = PMIX_BYTE_OBJECT;
    val.data.bo.bytes = (char *)value;
    val.data.bo.size = value_size;

    pmix_status_t rc = PMIx_Put(PMIX_GLOBAL, k.c_str(), &val);
    if(rc != PMIX_SUCCESS)
      return false;

    ctx->pending_sync = true;
    return true;
  }

  static bool app_get(const void *key, size_t key_size, void *value, size_t *value_size,
                      const void *vtable_data, size_t vtable_data_size)
  {
    if(!key || !value || !value_size || key_size == 0 || !vtable_data)
      return false;
    VtableContext *ctx = *(VtableContext **)vtable_data;

    std::string k(static_cast<const char *>(key), key_size);

    if(k == REALM_KEY_RANK) {
      if(*value_size < sizeof(uint32_t)) {
        *value_size = 0;
        return false;
      }
      uint32_t v = ctx->pmix_rank;
      memcpy(value, &v, sizeof(v));
      *value_size = sizeof(v);
      return true;
    }
    if(k == REALM_KEY_RANKS) {
      if(*value_size < sizeof(uint32_t)) {
        *value_size = 0;
        return false;
      }
      uint32_t v = ctx->pmix_size;
      memcpy(value, &v, sizeof(v));
      *value_size = sizeof(v);
      return true;
    }
    if(k == REALM_KEY_GROUP) {
      if(*value_size < sizeof(uint32_t)) {
        *value_size = 0;
        return false;
      }
      uint32_t v = 0;
      memcpy(value, &v, sizeof(v));
      *value_size = sizeof(v);
      return true;
    }

    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, ctx->myproc.nspace, PMIX_RANK_WILDCARD);
    pmix_value_t *val = NULL;

    pmix_status_t rc = PMIx_Get(&wildcard, k.c_str(), NULL, 0, &val);
    if(rc == PMIX_SUCCESS && val && val->type == PMIX_BYTE_OBJECT) {
      if(val->data.bo.size > *value_size) {
        *value_size = val->data.bo.size;
        PMIX_VALUE_RELEASE(val);
        return false;
      }
      memcpy(value, val->data.bo.bytes, val->data.bo.size);
      *value_size = val->data.bo.size;
      PMIX_VALUE_RELEASE(val);
      return true;
    }

    if(val)
      PMIX_VALUE_RELEASE(val);
    *value_size = 0;
    return true;
  }

  static bool app_bar(const void *vtable_data, size_t vtable_data_size)
  {
    if(!vtable_data)
      return false;
    VtableContext *ctx = *(VtableContext **)vtable_data;

    if(ctx->pending_sync) {
      pmix_status_t rc = PMIx_Commit();
      if(rc != PMIX_SUCCESS)
        return false;
      ctx->pending_sync = false;
    }

    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, ctx->myproc.nspace, PMIX_RANK_WILDCARD);

    pmix_info_t fence_info[1];
    bool collect_data = true;
    PMIX_INFO_LOAD(&fence_info[0], PMIX_COLLECT_DATA, &collect_data, PMIX_BOOL);

    pmix_status_t rc = PMIx_Fence(&wildcard, 1, fence_info, 1);
    if(rc != PMIX_SUCCESS)
      return false;

    return true;
  }

  Realm::Runtime::KeyValueStoreVtable create_key_value_store_vtable()
  {
    VtableContext *ctx = new VtableContext();

    if(PMIx_Init(&ctx->myproc, NULL, 0) != PMIX_SUCCESS) {
      delete ctx;
      return Realm::Runtime::KeyValueStoreVtable();
    }

    pmix_value_t *val = NULL;
    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, ctx->myproc.nspace, PMIX_RANK_WILDCARD);

    if(PMIx_Get(&wildcard, PMIX_JOB_SIZE, NULL, 0, &val) != PMIX_SUCCESS || !val) {
      PMIx_Finalize(NULL, 0);
      delete ctx;
      return Realm::Runtime::KeyValueStoreVtable();
    }

    ctx->pmix_size = val->data.uint32;
    PMIX_VALUE_RELEASE(val);

    ctx->pmix_rank = static_cast<uint32_t>(ctx->myproc.rank);

    Realm::Runtime::KeyValueStoreVtable vtable;
    vtable.vtable_data = new VtableContext *(ctx);
    vtable.vtable_data_size = sizeof(VtableContext *);
    vtable.put = app_put;
    vtable.get = app_get;
    vtable.bar = app_bar;
    vtable.cas = nullptr;
    return vtable;
  }

  void finalize_key_value_store_vtable(const Realm::Runtime::KeyValueStoreVtable &vtable)
  {
    if(vtable.vtable_data) {
      VtableContext *ctx = *(VtableContext **)vtable.vtable_data;
      delete ctx;
      delete(VtableContext **)vtable.vtable_data;
    }
    PMIx_Finalize(NULL, 0);
  }

} // namespace App
