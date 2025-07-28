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
  uint32_t ip;        /* NBO                           */
  uint16_t udp_port;  /* host order                    */
  uint8_t flags;      /* bit 0 == heavy blob present   */
  const void *worker; /* UCX/whatever blob             */
  size_t worker_len;

  const void *mm;
  size_t mm_len;
} realmNodeMeta_t;

/* -------- opaque handles ------------------------------------ */
typedef struct realmMembership_ctx *realmMembership_t;
#ifdef __cplusplus
#include "realm/event.h"
typedef Realm::Event realmEvent_t;
#else
typedef struct realmEvent_st realmEvent_t;
#endif

/* -------- membership change callback -------------------------- */
typedef void (*realmMembershipChangeCB_fn)(const realmNodeMeta_t *n,
                                           const void *machine_blob,
                                           size_t machine_bytes,
                                           bool joined,
                                           void *arg);

/* -------- back-end v-table ---------------------------------- */
typedef struct {

  realmStatus_t (*join_request)(void *state, const realmNodeMeta_t *self,
                                realmEvent_t done, uint64_t *cluster_epoch_out,
                                bool lazy_mode,
                                realmMembershipChangeCB_fn cb_fn,
                                void *cb_arg);

  // realmStatus_t (*subscribe_request)(void *state, realmEvent_t done, bool lazy_mode);
  // realmStatus_t (*destroy)(void *state);
  // realmStatus_t (*progress)(void *state);
  // realmStatus_t (*epoch)(void *state, uint64_t *epoch_out);
  // realmStatus_t (*members)(void *state, realmNodeMeta_t *buf, size_t *count_io);
} realmMembershipOps_t;

realmStatus_t realmMembershipCreate(const realmMembershipOps_t *ops, void *state,
                                    realmMembership_t *out);
realmStatus_t realmMembershipDestroy(realmMembership_t h);

// realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *self,
//                       realmEvent_t done, uint64_t *epoch_out);

// realmStatus_t realmProgress(realmMembership_t h);
// realmStatus_t realmGetEpoch(realmMembership_t h, uint64_t *e);
// realmStatus_t realmGetMembers(realmMembership_t h, realmNodeMeta_t *buf, size_t
// *cnt_io);

#ifdef __cplusplus
extern "C" {
#endif

realmStatus_t realmJoin(realmMembership_t h, const realmNodeMeta_t *self,
                        realmEvent_t done, uint64_t *epoch_out, bool lazy_mode,
                        realmMembershipChangeCB_fn cb_fn,
                        void *cb_arg);

realmStatus_t realmSubscribe(realmMembership_t h, realmEvent_t done, bool lazy_mode);

realmStatus_t realmMembershipCreateDefaultBackend(realmMembership_t *out);

#ifdef __cplusplus
}
#endif

#endif /* MEMBERSHIP_H */
