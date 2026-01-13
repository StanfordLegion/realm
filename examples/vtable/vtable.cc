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

#include "realm.h"
#include "realm/network.h"
#include "realm_bootstrap.h"
#include <iostream>

using namespace Realm;

int main(int argc, char **argv)
{
  Runtime::KeyValueStoreVtable vtable = App::create_key_value_store_vtable();
  Runtime rt;
  
  if(!rt.network_init(vtable)) return 1;
  if(!rt.create_configs(argc, argv)) return 1;
  if(!rt.configure_from_command_line(argc, argv)) return 1;
  rt.start();
  
  std::cout << "node " << Network::my_node_id << " of " 
            << (Network::max_node_id + 1) << std::endl;
  
  rt.shutdown();
  int rc = rt.wait_for_shutdown();
  App::finalize_key_value_store_vtable(vtable);
  return rc;
}

