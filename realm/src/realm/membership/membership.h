#ifndef MEMBERSHIP_H
#define MEMBERSHIP_H

#include <stdint.h>
#include <stddef.h>
#include "realm/realm_c.h"

/* -------- light-weight node header -------------------------- */
typedef struct {
  int32_t node_id;
  int32_t seed_id;
  bool announce_mm;
} node_meta_t;

/* -------- opaque handles ------------------------------------ */
typedef struct membership_ctx *membership_handle_t;

/* -------- membership change callback -------------------------- */
typedef void (*membership_change_cb_fn)(const node_meta_t *n, const void *machine_blob,
                                        size_t machine_bytes, bool joined, void *arg);

typedef bool (*membership_filter_fn)(const node_meta_t *node, void *arg);

typedef struct membership_hooks_t {
  membership_change_cb_fn pre_join;
  membership_change_cb_fn post_join;
  membership_change_cb_fn pre_leave;
  membership_change_cb_fn post_leave;
  membership_filter_fn filter;
  void *user_arg;
} membership_hooks_t;

/* -------- back-end v-table ---------------------------------- */
typedef struct {

  realm_status_t (*join_request)(void *state, const node_meta_t *self);
  realm_status_t (*leave_request)(void *st, const node_meta_t *self);

  // realm_status_t (*destroy)(void *state);

} membership_ops_t;

realm_status_t membership_create(const membership_ops_t *ops, void *state,
                                 membership_handle_t *out);
realm_status_t membership_destroy(membership_handle_t h);

#ifdef __cplusplus
extern "C" {
#endif

realm_status_t membership_join(membership_handle_t h, const node_meta_t *self);
realm_status_t membership_leave(membership_handle_t h, const node_meta_t *self);
realm_status_t membership_init(membership_handle_t *out, membership_hooks_t hooks);

#ifdef __cplusplus
}
#endif

#endif /* MEMBERSHIP_H */
