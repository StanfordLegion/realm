/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
  using namespace Realm;

  Runtime runtime;
  if(!runtime.network_init(&argc, &argv))
    return 1;

  std::vector<std::string> args(argv + 1, argv + argc);
  NodeID gpu_rank = 0;
  NodeID ipc_disabled_rank = -1;
  std::string backend = "cuda";
  CommandLineParser parser;
  parser.add_option_int("-test:gpu-rank", gpu_rank)
      .add_option_int("-test:ipc-disabled-rank", ipc_disabled_rank)
      .add_option_string("-test:backend", backend);
  if(!parser.parse_command_line(args) || (Network::max_node_id == 0) || (gpu_rank < -1) ||
     (gpu_rank > Network::max_node_id) || (ipc_disabled_rank < -1) ||
     (ipc_disabled_rank > Network::max_node_id) ||
     (backend != "cuda" && backend != "hip")) {
    std::fprintf(stderr,
                 "GPU IPC test requires multiple Realm ranks and valid options\n");
    return 1;
  }

  const bool use_gpu = (gpu_rank == -1) || (Network::my_node_id == gpu_rank);
  const bool use_ipc = (Network::my_node_id != ipc_disabled_rank);
  args.insert(args.end(), {"-ll:cpu", "1", "-ll:gpu", use_gpu ? "1" : "0",
                           "-" + backend + ":ipc", use_ipc ? "1" : "0"});
  if(!runtime.create_configs(argc, argv) || !runtime.configure_from_command_line(args))
    return 1;
  runtime.start();

  if(Network::my_node_id == 0) {
    auto gpus =
        Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::TOC_PROC);
    const size_t expected = (gpu_rank == -1) ? Network::max_node_id + 1 : 1;
    bool passed = (gpus.count() == expected);
    if(passed && (gpu_rank >= 0))
      passed = (gpus.first().address_space() == static_cast<AddressSpace>(gpu_rank));
    if(!passed)
      std::fprintf(stderr, "GPU processor placement did not match test configuration\n");
    runtime.shutdown(Event::NO_EVENT, passed ? 0 : 1);
  }
  return runtime.wait_for_shutdown();
}
