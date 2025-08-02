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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#include "oob_group_comm.h"
#include "bootstrap/bootstrap.h"
#include "realm_defines.h"

namespace Realm {
  namespace ucc {
    bootstrap_handle OOBGroupComm::boot_handle_;

    OOBGroupComm::OOBGroupComm(int r, int ws, bootstrap_handle_t *bh)
      : rank_(r)
      , world_sz_(ws)
    {
      boot_handle_ = *bh;
    }

    ucc_status_t OOBGroupComm::oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                             void *coll_info, void **req)
    {

      ucc_status_t status{UCC_OK};
      int ret = boot_handle_.allgather(sbuf, rbuf, msglen, &boot_handle_);
      if(0 != ret) {
        std::cerr << "OOB-Allgather() error in allgather" << std::endl;
        status = UCC_ERR_LAST;
      }
      return status;
    }

    ucc_status_t OOBGroupComm::oob_allgather_test(void *req) { return UCC_OK; }

    ucc_status_t OOBGroupComm::oob_allgather_free(void *req)
    {
      return UCC_OK;
      ;
    }

    int OOBGroupComm::get_rank() { return rank_; }

    int OOBGroupComm::get_world_size() { return world_sz_; }

    void *OOBGroupComm::get_coll_info() { return this; }
  } // namespace ucc
} // namespace Realm
