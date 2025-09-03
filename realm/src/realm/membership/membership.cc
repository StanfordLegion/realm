#include "membership.h"
#include "realm/realm_c.h"
#include "realm_defines.h"
#include <stdlib.h>
#include <cassert>
#include <map>

#ifdef REALM_USE_UDP
extern std::unique_ptr<IMembership> create_am_mesh(const IMembership::Hooks &hooks);
#endif

std::unique_ptr<IMembership> create_membership(const std::string &backend,
                                               const IMembership::Hooks &hooks)
{
#ifdef REALM_USE_UDP
  return create_am_mesh(hooks);
#else
  return nullptr;
#endif
}
