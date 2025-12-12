#ifndef REALM_BOOTSTRAP_H
#define REALM_BOOTSTRAP_H

#include "realm/runtime.h"

namespace App {

Realm::Runtime::NetworkVtable create_network_vtable();
void finalize_network_vtable(const Realm::Runtime::NetworkVtable &vtable);

}

#endif

