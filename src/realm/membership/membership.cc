#include "membership.h"
#include <stdlib.h>

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

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *s,
                        uint64_t *epoch_out, bool lazy_mode, realmMembershipHooks_t hooks)
{
  CALL(h, join_request, s, epoch_out, lazy_mode, hooks);
}

/*realmStatus_t realmSubscribe(realmMembership_t h, realmEvent_t done, bool lazy_mode)
{
  CALL(h, subscribe_request, done, lazy_mode);
}

realmStatus_t realmProgress(realmMembership_t h) { CALL(h, progress); }
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
extern realmStatus_t realmMembershipP2PInit(realmMembership_t *out);
#endif

realmStatus_t realmMembershipInit(realmMembership_t *out)
{
#if defined(REALM_USE_UDP)
  return realmMembershipP2PInit(out);
#else
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}
