/*
 * Copyright 2025 NVIDIA Corporation
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

// UCP network module internals

#include "realm/logging.h"
#include "realm/runtime_impl.h"
#include "realm/ucx/bootstrap/bootstrap_internal.h"
#include "realm/ucx/bootstrap/bootstrap_loader.h"

namespace Realm {

  // defined in ucp_module.cc
  extern Logger log_ucp;

  namespace UCP {

    static int bootstrap_vtable_allgather(const void *sendbuf, void *recvbuf, int bytes,
                                          struct bootstrap_handle *handle)
    {
      RuntimeImpl *runtime = get_runtime();
      assert(runtime->has_network_vtable());
      if(runtime->network_vtable_group()) {
        // Need our group ID too to avoid interfering with other groups
        // that might be trying to join at the same time
        const std::optional<uint64_t> group = runtime->network_vtable_local_group();
        if(!group)
          return 1;
        // Synthesize our local key
        constexpr size_t max_key_size = 1024;
        char key[max_key_size];
        size_t key_size = snprintf(key, max_key_size, "realm_bootstrap_key_%ld_%d",
                                   *group, handle->pg_rank);
        // < and not <= because we don't care about null terminator
        if(max_key_size < key_size) {
          log_ucp.error() << "Internal bootstrap error, key too large";
          // Explode since this is our fault
          std::abort();
        }
        // Put our local value in the key-value store
        if(!runtime->network_vtable_put(key, key_size, sendbuf, bytes)) {
          log_ucp.error() << "Failed bootstrap 'put' operation, "
                          << "the UCX bootstrap will not succeed.";
          return 1;
        }
        // Synchronize to make sure everyone is done
        if(!runtime->network_vtable_bar())
          return 1;
        // Get all the values from everyone else
        uint8_t *ptr = (uint8_t *)recvbuf;
        for(int rank = 0; rank < handle->pg_size; rank++) {
          key_size =
              snprintf(key, max_key_size, "realm_boostrap_key_%ld_%d", *group, rank);
          if(max_key_size < key_size) {
            log_ucp.error() << "Internal bootstrap error, key too large";
            // Explode since this is our fault
            std::abort();
          }
          size_t actual_size = bytes;
          if(!runtime->network_vtable_get(key, key_size, ptr, &actual_size) ||
             (actual_size != ((size_t)bytes))) {
            log_ucp.error() << "Failed bootstrap 'get' operation, "
                            << "the UCX boostrap will not succeed";
            return 1;
          }
          ptr += bytes;
        }
      } else {
        // Not a group, we're just a single process so just copy things over
        std::memcpy(recvbuf, sendbuf, bytes);
      }
      return 0;
    }

    int bootstrap_init(const BootstrapConfig *config, bootstrap_handle_t *handle)
    {
      int status = 0;

      switch(config->mode) {
      case BOOTSTRAP_MPI:
        if(config->plugin_name != NULL) {
          status = bootstrap_loader_init(config->plugin_name, NULL, handle);
        } else {
          status = bootstrap_loader_init(BOOTSTRAP_MPI_PLUGIN, NULL, handle);
        }
        if(status != 0) {
          log_ucp.error() << "bootstrap_loader_init failed";
        }
        break;
      case BOOTSTRAP_P2P:
        if(config->plugin_name != NULL) {
          status = bootstrap_loader_init(config->plugin_name, NULL, handle);
        } else {
          status = bootstrap_loader_init(BOOTSTRAP_P2P_PLUGIN, NULL, handle);
        }
        if(status != 0) {
          log_ucp.error() << "bootstrap_loader_init failed";
        }
        break;
      case BOOTSTRAP_PLUGIN:
        status = bootstrap_loader_init(config->plugin_name, NULL, handle);
        if(status != 0) {
          log_ucp.error() << "bootstrap_loader_init failed";
        }
        break;
      case BOOTSTRAP_VTABLE:
      {
        RuntimeImpl *runtime = get_runtime();
        assert(runtime->has_network_vtable());
        // We need to get our local process group information here and fill
        // in our all-gather implementation
        std::optional<uint64_t> rank = runtime->network_vtable_local_rank();
        if(!rank) {
          return 1;
        }
        std::optional<uint64_t> ranks = runtime->network_vtable_local_ranks();
        if(!ranks) {
          return 1;
        }
        handle->pg_rank = *rank;
        handle->pg_size = *ranks;
        handle->shared_ranks = nullptr;
        handle->num_shared_ranks = 0;
        handle->barrier = nullptr;
        handle->bcast = nullptr;
        handle->gather = nullptr;
        handle->allgather = bootstrap_vtable_allgather;
        handle->alltoall = nullptr;
        handle->allreduce_ull = nullptr;
        handle->allgatherv = nullptr;
        handle->finalize = nullptr;
        break;
      }
      default:
        status = BOOTSTRAP_ERROR_INTERNAL;
        log_ucp.error() << ("invalid bootstrap mode");
      }

      return status;
    }

    int bootstrap_finalize(bootstrap_handle_t *handle)
    {
      // Need this for the case of vtable where there is no finalize
      if(handle->finalize == nullptr)
        return 0;
      int status = bootstrap_loader_finalize(handle);
      if(status != 0) {
        log_ucp.error() << "bootstrap_finalize failed";
      }
      return status;
    }

  }; // namespace UCP

}; // namespace Realm
