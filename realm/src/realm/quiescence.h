/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef QUIESCENCE_H
#define QUIESCENCE_H

#include "realm/realm_config.h"

namespace Realm {

  class NetworkModule;
  class NodeDirectory;

  struct QuiescenceCounters {
    uint64_t msg_sent{0};
    uint64_t msg_recv{0};
    uint64_t rcomp_sent{0};
    uint64_t rcomp_recv{0};
    uint64_t outstanding{0};
  };

  // TODO: Run for data & control plane
  //       control plane should exempt quiece messages

  void quiescence_init(const std::vector<NetworkModule *> &net, NodeDirectory *ndir);
  bool quiescence_exec(NodeID node);
} // namespace Realm

#endif
