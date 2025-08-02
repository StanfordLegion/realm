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

#ifndef __UCCLAYER_H__
#define __UCCLAYER_H__

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>

#include "client.h"
#include "logger.h"
#include "types.h"
#include "server.h"
#include "worker.h"

namespace p2p {
  // @class P2PComm
  // @brief Group communicator based on p2p communication.
  //
  // It currently implements  only one collective operation  - allgather(). If
  // necessary, it has  enough information to implement any  of the collective
  // operations.  All the communcation is using p2p connections.
  class P2PComm {
    int rank_;
    int world_sz_;
    std::shared_ptr<mesh::Server> receiver_;
    std::shared_ptr<mesh::Client> sender_;

    std::string self_;
    std::vector<std::string> peers_;

    std::unique_ptr<mesh::Worker> worker_;

    std::shared_ptr<Logger::p2p_log> p2p_log_{nullptr};

  public:
    P2PComm(const std::string &self, const std::vector<std::string> &peers,
            const std::string &log_file = mesh::DEF_LOG);

    inline int get_rank() { return rank_; };
    inline int get_world_size() { return world_sz_; };

    // @brief Initialize p2p connectiions betweenn all the pairs of workers.
    // @return true on success, false otherwise.
    bool Init();

    // @brief Allgather collective using p2p communication.
    int Allgather(void *sbuf, int sendcount, uint8_t sendtype, void *rbuf, int recvcount,
                  uint8_t recvtype);

    // @brief shutdown the p2p communicator
    int Shutdown();
  };
} // namespace p2p
#endif
