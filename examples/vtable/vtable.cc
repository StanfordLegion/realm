#include "realm.h"
#include "realm/network.h"
#include "realm_bootstrap.h"
#include <iostream>

using namespace Realm;

int main(int argc, char **argv)
{
  Runtime::NetworkVtable vtable = App::create_network_vtable();
  Runtime rt;
  
  if(!rt.network_init(vtable)) return 1;
  if(!rt.create_configs(argc, argv)) return 1;
  if(!rt.configure_from_command_line(argc, argv)) return 1;
  rt.start();
  
  std::cout << "node " << Network::my_node_id << " of " 
            << (Network::max_node_id + 1) << std::endl;
  
  rt.shutdown();
  int rc = rt.wait_for_shutdown();
  App::finalize_network_vtable();
  return rc;
}

