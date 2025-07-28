#ifndef MEMBERSHIP_H
#define MEMBERSHIP_H

#include <stdint.h>
#include <stddef.h>

/* -------- status codes -------------------------------------- */
typedef enum
{
  REALM_OK = 0,
  REALM_ERR_BAD_ARG = -1,
  REALM_ERR_NOMEM = -2,
  REALM_ERR_INTERNAL = -3
} realmStatus_t;

/* -------- light-weight node header -------------------------- */
typedef struct {
  int32_t node_id;
  int32_t seed_id;
  bool announce_mm;
} realmNodeMeta_t;

/* -------- opaque handles ------------------------------------ */
typedef struct realmMembership_ctx *realmMembership_t;

/* -------- membership change callback -------------------------- */
typedef void (*realmMembershipChangeCB_fn)(const realmNodeMeta_t *n,
                                           const void *machine_blob, size_t machine_bytes,
                                           bool joined, void *arg);

typedef struct realmMembershipHooks_t {
  realmMembershipChangeCB_fn pre_join;
  realmMembershipChangeCB_fn post_join;
  void *user_arg;
} realmMembershipHooks_t;

/* -------- back-end v-table ---------------------------------- */
typedef struct {

  realmStatus_t (*join_request)(void *state, const realmNodeMeta_t *self,
                                realmMembershipHooks_t hooks);

  // realmStatus_t (*subscribe_request)(void *state, realmEvent_t done, bool announce_mm);
  // realmStatus_t (*destroy)(void *state);
  // realmStatus_t (*progress)(void *state);

} realmMembershipOps_t;

realmStatus_t realmMembershipCreate(const realmMembershipOps_t *ops, void *state,
                                    realmMembership_t *out);
realmStatus_t realmMembershipDestroy(realmMembership_t h);

// realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *self,
//                       realmEvent_t done, uint64_t *epoch_out);

// realmStatus_t realmProgress(realmMembership_t h);
// *cnt_io);

#ifdef __cplusplus
extern "C" {
#endif

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *self,
                        realmMembershipHooks_t hooks);
realmStatus_t realmMembershipInit(realmMembership_t *out);
// realmStatus_t realmSubscribe(realmMembership_t h, realmEvent_t done, bool announce_mm);

#ifdef __cplusplus
}
#endif

#endif /* MEMBERSHIP_H */
