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

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *s,
                        realmMembershipHooks_t hooks)
{
  CALL(h, join_request, s, hooks);
}

realmStatus_t realmLeave(realmMembership_t h, const realmNodeMeta_t *s,
                         realmMembershipHooks_t hooks)
{
  CALL(h, leave_request, s, hooks);
}

#undef CALL

#ifdef REALM_USE_UDP
extern realmStatus_t realmMembershipP2PInit(realmMembership_t *out);
#endif

realmStatus_t realmMembershipInit(realmMembership_t *out)
{
#ifdef REALM_USE_UDP
  return realmMembershipP2PInit(out);
#else
  assert(0);
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}
