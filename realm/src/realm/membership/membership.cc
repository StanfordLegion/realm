#include "membership.h"
#include "realm_defines.h"
#include <stdlib.h>
#include <cassert>

struct realmMembership_ctx {
  const realmMembershipOps_t *ops;
  void *state;
};

#define CHECK(h) ((h) && (h)->ops)

realmStatus_t realmMembershipCreate(const realmMembershipOps_t *ops, void *state,
                                    realmMembership_t *out)
{
  if(!ops || !out) {
    return realmStatus_t::REALM_ERROR;
  }

  realmMembership_t h = (realmMembership_t)malloc(sizeof(*h));

  if(!h) {
    return realmStatus_t::REALM_ERROR;
  }

  h->ops = ops;
  h->state = state;
  *out = h;
  return realmStatus_t::REALM_SUCCESS;
}

realmStatus_t realmMembershipDestroy(realmMembership_t)
{
  /*if(!CHECK(h)) {
    return REALM_ERR_BAD_ARG;
  }

  if(h->ops->destroy) {
    h->ops->destroy(h->state);
  }

  free(h);*/
  return realmStatus_t::REALM_SUCCESS;
}

/* ----- wrappers (CALL macro) --------------------------------*/
#define CALL(h, fn, ...)                                                                 \
  do {                                                                                   \
    if(!CHECK(h) || !(h)->ops->fn)                                                       \
      return realmStatus_t::REALM_ERROR;                                                 \
    return (h)->ops->fn((h)->state, __VA_ARGS__);                                        \
  } while(0)

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *s)
{
  CALL(h, join_request, s);
}

realmStatus_t realmLeave(realmMembership_t h, const realmNodeMeta_t *s)
{
  CALL(h, leave_request, s);
}

#undef CALL

#ifdef REALM_USE_UDP
extern realmStatus_t realmMembershipMeshInit(realmMembership_t *out,
                                             realmMembershipHooks_t hooks);
#endif

realmStatus_t realmMembershipInit(realmMembership_t *out, realmMembershipHooks_t hooks)
{
#ifdef REALM_USE_UDP
  return realmMembershipMeshInit(out, hooks);
#else
  assert(0);
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}
