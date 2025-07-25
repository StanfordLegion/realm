#include "membership.h"
#include <stdlib.h>
#include "realm/event.h"

struct realmMembership_ctx {
  const realmMembershipOps_t *ops;
  void *state;
};

#define CHECK(h) ((h) && (h)->ops)

realmStatus_t realmMembershipCreate(const realmMembershipOps_t *ops, void *state,
                                    realmMembership_t *out)
{
  if(!ops || !out) {
    return REALM_ERR_BAD_ARG;
  }

  realmMembership_t h = (realmMembership_t)malloc(sizeof(*h));

  if(!h) {
    return REALM_ERR_NOMEM;
  }

  h->ops = ops;
  h->state = state;
  *out = h;
  return REALM_OK;
}

realmStatus_t realmMembershipDestroy(realmMembership_t h)
{
  /*if(!CHECK(h)) {
    return REALM_ERR_BAD_ARG;
  }

  if(h->ops->destroy) {
    h->ops->destroy(h->state);
  }

  free(h);*/
  return REALM_OK;
}

/* ----- wrappers (CALL macro) --------------------------------*/
#define CALL(h, fn, ...)                                                                 \
  do {                                                                                   \
    if(!CHECK(h) || !(h)->ops->fn)                                                       \
      return REALM_ERR_BAD_ARG;                                                          \
    return (h)->ops->fn((h)->state, __VA_ARGS__);                                        \
  } while(0)

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *s, realmEvent_t done,
                        uint64_t *epoch_out)
{
  CALL(h, join_request, s, done, epoch_out);
}

/*realmStatus_t realmProgress(realmMembership_t h) { CALL(h, progress); }
realmStatus_t realmGetEpoch(realmMembership_t h, uint64_t *e) { CALL(h, epoch, e); }
realmStatus_t realmGetMembers(realmMembership_t h, realmNodeMeta_t *b, size_t *c)
{
  CALL(h, members, b, c);
}*/
#undef CALL

/* ------------------------------------------------------------------ */
/* Default backend selection                                           */
/* ------------------------------------------------------------------ */

#ifdef REALM_USE_UDP
extern realmStatus_t realmCreateP2PMembershipBackend(realmMembership_t *out);
#endif

realmStatus_t realmMembershipCreateDefaultBackend(realmMembership_t *out)
{
#if defined(REALM_USE_UDP)
  return realmCreateP2PMembershipBackend(out);
#else
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}
