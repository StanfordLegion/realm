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

#include "realm.h"
#include "realm/network.h"
#include "app.h"
#include <iostream>

using namespace Realm;

Logger log_app("app");

int main(int argc, char **argv)
{
  std::cout << "Bootstrap test starting..." << std::endl;
  
  bootstrap_handle_t app_bootstrap;
  int status = App::bootstrap_init(&app_bootstrap);
  if(status != 0) {
    std::cerr << "Application bootstrap initialization failed!" << std::endl;
    return 1;
  }
  
  std::cout << "Application bootstrap handle initialized" << std::endl;
  
  // Future: Register handle with Realm so it can call init() internally
  // Realm::register_bootstrap(&app_bootstrap);
  
  // OR : Keep the bootstrap_handle_t at global REALM namespace
  // REALM::external_bootstrap_handle = &app_bootstrap;
  
  // Temporary workaround: manually call init() to set env vars
  if(app_bootstrap.init) {
    app_bootstrap.init(&app_bootstrap, nullptr);
  }
  
  Runtime rt;
  bool ok = rt.init(&argc, &argv);
  if(!ok) {
    std::cerr << "Runtime initialization failed!" << std::endl;
    App::bootstrap_finalize(&app_bootstrap);
    return 1;
  }
  
  std::cout << "Runtime initialized successfully" << std::endl;
  std::cout << "Realm Node ID: " << Network::my_node_id << std::endl;
  std::cout << "Realm Total nodes: " << (Network::max_node_id + 1) << std::endl;
  
  rt.shutdown();
  
  std::cout << "Waiting for shutdown..." << std::endl;
  
  int ret = rt.wait_for_shutdown();
  
  std::cout << "Bootstrap test completed with return code: " << ret << std::endl;
  
  App::bootstrap_finalize(&app_bootstrap);
  
  return ret;
}

