#include "membership.h"
#include "realm_defines.h"
#include <stdlib.h>
#include <cassert>

struct membership_ctx {
  const membership_ops_t *ops;
  void *state;
};

#define CHECK(h) ((h) && (h)->ops)

realm_status_t membership_create(const membership_ops_t *ops, void *state,
                                 membership_handle_t *out)
{
  if(!ops || !out) {
    return realm_status_t::REALM_ERROR;
  }

  membership_handle_t h = (membership_handle_t)malloc(sizeof(*h));

  if(!h) {
    return realm_status_t::REALM_ERROR;
  }

  h->ops = ops;
  h->state = state;
  *out = h;
  return realm_status_t::REALM_SUCCESS;
}

realm_status_t membership_delete(membership_handle_t h)
{
  if(!h) {
    return realm_status_t::REALM_ERROR;
  }

  free(h);
  return realm_status_t::REALM_SUCCESS;
}

/* ----- wrappers (CALL macro) --------------------------------*/
#define CALL(h, fn, ...)                                                                 \
  do {                                                                                   \
    if(!CHECK(h) || !(h)->ops->fn)                                                       \
      return realm_status_t::REALM_ERROR;                                                \
    return (h)->ops->fn((h)->state, __VA_ARGS__);                                        \
  } while(0)

realm_status_t membership_join(membership_handle_t h, const node_meta_t *s)
{
  CALL(h, join_request, s);
}

realm_status_t membership_leave(membership_handle_t h, const node_meta_t *s)
{
  CALL(h, leave_request, s);
}

#undef CALL

#ifdef REALM_USE_UDP
extern realm_status_t membership_mesh_init(membership_handle_t *out,
                                           membership_hooks_t hooks);
extern realm_status_t membership_mesh_destroy(membership_handle_t h);
#endif

realm_status_t membership_init(membership_handle_t *out, membership_hooks_t hooks)
{
#ifdef REALM_USE_UDP
  return membership_mesh_init(out, hooks);
#else
  assert(0);
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}

realm_status_t membership_destroy(membership_handle_t h)
{
#ifdef REALM_USE_UDP
  return membership_mesh_destroy(h);
#else
  assert(0);
  (void)out;
  return REALM_ERR_INTERNAL;
#endif
}
