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
#include <string>
#include <map>
#include <vector>
#include <mpi.h>

namespace App {

// Realm NetworkVtable required keys
static constexpr const char *REALM_KEY_RANK = "realm_rank";
static constexpr const char *REALM_KEY_RANKS = "realm_ranks";
static constexpr const char *REALM_KEY_GROUP = "realm_group";

struct VtableContext {
  std::map<std::string, std::vector<uint8_t>> local_kv_store;
  std::map<std::string, std::vector<uint8_t>> global_kv_store;
  bool pending_sync = false;
  int mpi_rank = 0;
  int mpi_size = 0;
};

static bool app_put(const void *key, size_t key_size, const void *value, size_t value_size,
                    const void *vtable_data, size_t vtable_data_size)
{
  if(!key || !value || key_size == 0 || !vtable_data) return false;
  VtableContext *state = *(VtableContext**)vtable_data;
  
  std::string k(static_cast<const char*>(key), key_size);
  std::vector<uint8_t> v(static_cast<const uint8_t*>(value), 
                         static_cast<const uint8_t*>(value) + value_size);
  state->local_kv_store[k] = v;
  state->pending_sync = true;
  // std::cout << "app_put: key='" << k << "' value_size=" << value_size << std::endl;
  return true;
}

static bool app_get(const void *key, size_t key_size, void *value, size_t *value_size,
                    const void *vtable_data, size_t vtable_data_size)
{
  if(!key || !value || !value_size || key_size == 0 || !vtable_data) return false;
  VtableContext *state = *(VtableContext**)vtable_data;
  
  std::string k(static_cast<const char*>(key), key_size);
  // std::cout << "app_get: key='" << k << "'" << std::endl;
  
  if(k == REALM_KEY_RANK) {
    if(*value_size < sizeof(uint32_t)) { *value_size = 0; return false; }
    uint32_t v = static_cast<uint32_t>(state->mpi_rank);
    memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }
  if(k == REALM_KEY_RANKS) {
    if(*value_size < sizeof(uint32_t)) { *value_size = 0; return false; }
    uint32_t v = static_cast<uint32_t>(state->mpi_size);
    memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }
  if(k == REALM_KEY_GROUP) {
    if(*value_size < sizeof(uint32_t)) { *value_size = 0; return false; }
    uint32_t v = 0;
    memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }
  
  auto it = state->global_kv_store.find(k);
  if(it != state->global_kv_store.end()) {
    if(it->second.size() > *value_size) {
      *value_size = it->second.size();
      return false;
    }
    memcpy(value, it->second.data(), it->second.size());
    *value_size = it->second.size();
    return true;
  }
  
  it = state->local_kv_store.find(k);
  if(it != state->local_kv_store.end()) {
    if(it->second.size() > *value_size) {
      *value_size = it->second.size();
      return false;
    }
    memcpy(value, it->second.data(), it->second.size());
    *value_size = it->second.size();
    return true;
  }
  
  *value_size = 0;
  return true;
}

static bool app_bar(const void *vtable_data, size_t vtable_data_size)
{
  if(!vtable_data) return false;
  VtableContext *state = *(VtableContext**)vtable_data;
  
  // std::cout << "app_bar called" << std::endl;
  
  // Exchange local KV data. Two stores needed because bootstrap reuses keys across
  // rounds with different sizes - only sync new data to avoid clobbering.
  if(state->pending_sync && !state->local_kv_store.empty()) {
    std::vector<uint8_t> sendbuf;
    for(const auto &kv : state->local_kv_store) {
      uint32_t klen = kv.first.size();
      uint32_t vlen = kv.second.size();
      sendbuf.insert(sendbuf.end(), reinterpret_cast<uint8_t*>(&klen),
                     reinterpret_cast<uint8_t*>(&klen) + sizeof(klen));
      sendbuf.insert(sendbuf.end(), kv.first.begin(), kv.first.end());
      sendbuf.insert(sendbuf.end(), reinterpret_cast<uint8_t*>(&vlen),
                     reinterpret_cast<uint8_t*>(&vlen) + sizeof(vlen));
      sendbuf.insert(sendbuf.end(), kv.second.begin(), kv.second.end());
    }
    
    int sendsize = sendbuf.size();
    std::vector<int> recvsizes(state->mpi_size);
    MPI_Allgather(&sendsize, 1, MPI_INT, recvsizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> displs(state->mpi_size);
    int total = 0;
    for(int i = 0; i < state->mpi_size; i++) {
      displs[i] = total;
      total += recvsizes[i];
    }
    
    std::vector<uint8_t> recvbuf(total);
    MPI_Allgatherv(sendbuf.data(), sendsize, MPI_BYTE,
                   recvbuf.data(), recvsizes.data(), displs.data(), MPI_BYTE,
                   MPI_COMM_WORLD);
    
    size_t off = 0;
    while(off < recvbuf.size()) {
      uint32_t klen, vlen;
      memcpy(&klen, &recvbuf[off], sizeof(klen)); off += sizeof(klen);
      std::string k(reinterpret_cast<char*>(&recvbuf[off]), klen); off += klen;
      memcpy(&vlen, &recvbuf[off], sizeof(vlen)); off += sizeof(vlen);
      std::vector<uint8_t> v(&recvbuf[off], &recvbuf[off] + vlen); off += vlen;
      state->global_kv_store[k] = v;
    }
    
    state->pending_sync = false;
    state->local_kv_store.clear();
  }
  
  return true;
}

Realm::Runtime::NetworkVtable create_network_vtable()
{
  MPI_Init(NULL, NULL);
  
  VtableContext *state = new VtableContext();
  MPI_Comm_rank(MPI_COMM_WORLD, &state->mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &state->mpi_size);
  
  Realm::Runtime::NetworkVtable vtable;
  vtable.vtable_data = new VtableContext*(state);
  vtable.vtable_data_size = sizeof(VtableContext*);
  vtable.put = app_put;
  vtable.get = app_get;
  vtable.bar = app_bar;
  vtable.cas = nullptr;
  return vtable;
}

void finalize_network_vtable(const Realm::Runtime::NetworkVtable &vtable)
{
  if(vtable.vtable_data) {
    VtableContext *state = *(VtableContext**)vtable.vtable_data;
    delete state;
    delete (VtableContext**)vtable.vtable_data;
  }
  MPI_Finalize();
}

}

