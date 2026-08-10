/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Barrier implementations for Realm

#ifndef REALM_BARRIER_IMPL_H
#define REALM_BARRIER_IMPL_H

#include "realm/event.h"
#include "realm/event_impl.h"
#include "realm/id.h"
#include "realm/multicast.h"
#include "realm/nodeset.h"
#include "realm/redop.h"

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <set>

namespace Realm {

  // NOTE: there is deliberately no 'BarrierCommunicator' indirection here
  //  (decision Q8).  Every send in this file goes through the active message
  //  interface directly, and the one place that used to route through a
  //  communicator - the non-owner forward in adjust_arrival - now calls
  //  BarrierAdjustMessage::send_request like everything else.

  // one entry per remote node that has to be told about a trigger.
  // 'previous_gen' is the last generation that node was told about; it bounds
  //  the slice of 'final_values' that node is sent, which is why it is
  //  per-recipient rather than a single watermark.
  //
  // LEGACY / REDUCTION PATH ONLY.  The scalable path replaced this with the
  //  owner's single subscriber set plus a delta notification
  //  (BarrierNotifyMessage), which is what removes the O(N) per-barrier map.
  struct RemoteNotification {
    NodeID node;
    EventImpl::gen_t trigger_gen, previous_gen;
  };

  // LEGACY / REDUCTION PATH ONLY.  The owner sends this only when
  //  'redop_id != 0', because its whole reason to exist is the per-recipient
  //  slice of 'final_values' it carries.  Receiving one therefore PROVES the
  //  barrier is a reduction barrier, which is how a non-owner that has never
  //  been told 'redop_id' learns it (tla/STATE_AND_LOCKING.md D2).
  struct BarrierTriggerMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t trigger_gen;
    // the generation the recipient was last told about
    EventImpl::gen_t previous_gen;
    EventImpl::gen_t first_generation;
    ReductionOpID redop_id;

    static void handle_message(NodeID sender, const BarrierTriggerMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id,
                             EventImpl::gen_t trigger_gen, EventImpl::gen_t previous_gen,
                             EventImpl::gen_t first_generation, ReductionOpID redop_id,
                             const void *data, size_t datalen);
  };

  // NOTIFICATION_PROTOCOL rules 1, 2 and 4 - THE SCALABLE TRIGGER NOTIFICATION.
  //
  // It is a DELTA (rule 4): generations '(prev, wm]' have just triggered, and
  //  the payload names whichever of THOSE are poisoned.  Its size therefore
  //  never grows with the barrier's poison history - which was the last
  //  unbounded quantity in the design.  The cost of that is gap sensitivity: a
  //  recipient whose watermark is below 'prev' cannot splice this range on to
  //  what it knows, because it would advance over generations whose poison
  //  status it does not know.  It DISCARDS the delta and PULLS
  //  (BarrierSubscribeMessage); it does NOT buffer it.
  //
  // It is MULTICAST to the owner's subscriber set, and the multicast layer
  //  plans a fresh forwarding tree from the encoded set on every send, so there
  //  is no learned tree to go stale and no per-node map at the owner.
  //
  // MEMBERSHIP RIDES ALONG (rule 1: published, never self-decided).  A
  //  per-recipient membership flag cannot survive a multicast - every recipient
  //  gets identical bytes - so the message carries the DEPARTING SET 'R' and
  //  each recipient computes its own membership as '(me not in R)'.  That is
  //  EXACT, because a shrink is published to the PRE-SHRINK set: every
  //  recipient is in that set, so "in the set afterwards" is precisely "not
  //  being removed".  'set_ver' version-gates it (rule 2) - apply the
  //  membership only if it is strictly newer than the highest already applied,
  //  or a stale in-flight notification resurrects membership the owner has
  //  dropped and the node hangs on its next wait.  When nothing is being
  //  removed, 'set_ver' is unchanged and no membership bytes are sent at all.
  //
  // Wire format of the payload:
  //
  //    num_poisoned x EventImpl::gen_t  (ascending; all in (prev, wm])
  //    departing_bytes of EncodedMulticastTargets  (R; empty unless set_ver
  //                                                 changed)
  struct BarrierNotifyMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t wm;   // the owner's watermark AFTER this trigger
    EventImpl::gen_t prev; // the owner's watermark BEFORE it - the GAP DETECTOR
    uint64_t set_ver;
    uint32_t num_poisoned;
    uint32_t departing_bytes;
    // NOTIFICATION_PROTOCOL rule 8, the optional ONE-BYTE HINT: does shrinking
    //  the subscriber set currently pay?  It is the verdict the owner's cost
    //  test returned the last time it actually evaluated one, so a node that
    //  would only have its request declined can suppress it and save the
    //  round trip.  Purely advisory - it is not a permission, and a node that
    //  ignores it is merely wasteful.  Optimistic by default (1), so the first
    //  departure of a barrier always gets a hearing.
    uint8_t shrink_hint;

    static void handle_message(NodeID sender, const BarrierNotifyMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(const MulticastTargetSet &targets, ID::IDType barrier_id,
                             EventImpl::gen_t wm, EventImpl::gen_t prev, uint64_t set_ver,
                             bool shrink_hint,
                             const std::vector<EventImpl::gen_t> &poison,
                             const std::vector<unsigned char> &departing);
  };

  // NOTIFICATION_PROTOCOL rule 5 - THE ANSWER TO A PULL.
  //
  // Point to point, so unlike a notification it can be keyed to exactly what
  //  the subscriber said it already had: the poison list is the poisoned
  //  generations in '(lk, wm]', where 'lk' is the subscriber's own last-known
  //  generation from its BarrierSubscribeMessage.  The receiver MERGES that
  //  list into what it has - substituting would drop the poison it already knew
  //  about below 'lk'.
  //
  // THE REPLY MUST CARRY THE WATERMARK.  That is what covers the race where the
  //  generation triggers while the subscribe is still on the wire, and omitting
  //  it is caught in the model as a deadlock.  The owner therefore replies to
  //  EVERY subscribe, even one it can answer with "nothing has triggered yet" -
  //  the reply is also what closes the recipient's rule-7 pull window.
  //
  // Payload: num_poisoned x EventImpl::gen_t, ascending.
  struct BarrierSubscribeReplyMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t wm;
    uint64_t set_ver;
    uint32_t num_poisoned;

    static void handle_message(NodeID sender, const BarrierSubscribeReplyMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t wm,
                             uint64_t set_ver,
                             const std::vector<EventImpl::gen_t> &poison);
  };

  // NOTIFICATION_PROTOCOL rule 8 - "I have gone idle; drop me if it pays."
  //
  // An INTENT, never a command.  Adds are mandatory and removals are
  //  DISCRETIONARY (rule 3), so the owner is free to decline this after its
  //  cost test - and a declined removal is simply the next publication in
  //  which this node still appears.  The node does NOT change its own
  //  membership on sending one: membership is published, never self-decided
  //  (rule 1), so the only thing that moves 'member' to MEMBER_NO is a
  //  version-gated publication naming this node in the departing set.
  //
  // UNICAST to the owner, and deliberately NOT aggregated up the multicast ack
  //  tree.  That tree is real but carries NO PAYLOAD today, so aggregating
  //  departures would mean adding a payload-reduction facility to the multicast
  //  layer, designed around one speculative user (NOTIFICATION_PROTOCOL rule
  //  8).  The subscribe path is already O(N) unicast, so an O(N) unsubscribe
  //  burst is not a new complexity class; the sender's
  //  'K + (node_id % J)' stagger is what keeps a phase change that retires many
  //  nodes at once from arriving as one spike.
  //
  // Nothing about this message is load bearing.  Rule 8 is unmodelled on
  //  purpose (NOTIFICATION_PROTOCOL section 8.2): 'Depart' may fire on ANY
  //  eligible node and 'Trigger' may apply ANY subset of the requests, which is
  //  strictly more general than any tuning, so its failure modes are bandwidth
  //  and never correctness.
  struct BarrierDepartMessage {
    ID::IDType barrier_id;
    // carried explicitly, exactly like BarrierSubscribeMessage's 'subscriber',
    //  rather than inferred from the active message's sender
    NodeID departing;

    static void handle_message(NodeID sender, const BarrierDepartMessage &msg,
                               const void *data, size_t datalen);

    static void send_request(NodeID target, ID::IDType barrier_id, NodeID departing);
  };

  // a cumulative subtree total travelling toward the barrier owner.
  // ARRIVAL_PROTOCOL rule 7: 'val' is the SENDER'S RUNNING TOTAL for this
  //  generation, never an increment.  The receiver REPLACES the value it has
  //  stored for the sender, and a report that does not strictly increase that
  //  value is stale and is discarded.  That is what makes duplicate and
  //  out-of-order delivery harmless.
  //
  // The PAYLOAD, when there is one, is the eager-flush report's PER-NODE
  //  ARRIVAL COUNT MAP (ARRIVAL_PROTOCOL section 11.1) - which nodes had
  //  arrivals in this generation and how many each, for the whole subtree the
  //  sender speaks for.  It is what the owner builds the next plan out of.
  //  Wire format:
  //
  //    flushmap := varint set_bytes
  //                set_bytes of EncodedMulticastTargets  (the node SET)
  //                one varint count per node, in ASCENDING node order
  //
  //  The node set uses the multicast codec (realm/multicast.h) rather than a
  //  naive list, so a contiguous run of nodes costs a couple of bytes instead
  //  of one entry each; the counts ride alongside in the set's own (sorted)
  //  iteration order, so no node id is ever repeated on the wire.
  //
  //  A map is carried ONLY when the sender is in eager-flush mode - a
  //  steady-state rule-1 report is O(1) and carries none, which is the whole
  //  point of the tree.  datalen == 0 therefore means "this report carries no
  //  map", and the receiver LEAVES the map it already has for this sender
  //  alone; it does not clear it.  The maps merge cumulatively, exactly like
  //  the counts (section 11.1), so they only ever grow.
  struct BarrierReportMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t gen;
    int64_t val;
    // the node the total is FOR.  Always the sender today, but the receiver
    //  keys its accumulator on this and never on its own child list - see
    //  ARRIVAL_PROTOCOL section 8.2.
    NodeID from;
    // 'direct' (rule 3): the sender holds no plan record, so it has no parent
    //  and no completion condition to wait for, and reports straight to the
    //  owner.  The owner treats a 'direct' as evidence its plan is wrong.
    bool is_direct;
    // decision Q4: an arrival whose PRECONDITION was poisoned poisons the
    //  generation it arrives on.  The bit is STICKY per generation and is OR-ed
    //  along the tree, so it is monotone and reordering cannot lose it.  It is
    //  applied by the receiver BEFORE the rule-7 staleness test, and it always
    //  rides the same report as the arrival COUNT that carried it - a count the
    //  generation cannot complete without - so it can never arrive too late.
    bool poisoned;

    static void handle_message(NodeID sender, const BarrierReportMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t gen,
                             int64_t val, NodeID from, bool is_direct, bool poisoned,
                             const void *data, size_t datalen);
  };

  // "enter eager-flush mode for this generation" (ARRIVAL_PROTOCOL rule 4).
  //  It carries no count - it is an announcement, not data - and it is
  //  IDEMPOTENT: a node already flushing this generation drops it, which is
  //  what terminates the fan-out down the tree.  Flush state is PER
  //  GENERATION; the next generation starts back in planned mode.
  struct BarrierFlushMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t gen;

    static void handle_message(NodeID sender, const BarrierFlushMessage &msg,
                               const void *data, size_t datalen);

    static void send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t gen);
  };

  // "the plan you hold is being retired" (ARRIVAL_PROTOCOL rules 5 and 6).
  //  Broadcast by the owner down the tree BEING RETIRED at the same moment the
  //  new plan goes down the NEW tree; the two race, and the deferral rule of
  //  BarrierArrive RecvNewPlan is the verified resolution.  'epoch' is the plan
  //  index being retired: an invalidation that is not newer than what this node
  //  has already retired is stale, and is consumed and ignored.
  struct BarrierInvalidateMessage {
    ID::IDType barrier_id;
    uint32_t epoch;

    static void handle_message(NodeID sender, const BarrierInvalidateMessage &msg,
                               const void *data, size_t datalen);

    static void send_request(NodeID target, ID::IDType barrier_id, uint32_t epoch);
  };

  // "install plan 'epoch'" (ARRIVAL_PROTOCOL rule 5).
  //
  // The payload is the SUBTREE rooted at the recipient (ARRIVAL_PROTOCOL
  //  section 11.4: a newplan must carry the recipient's own plan record, not
  //  just an epoch).  It is a pre-order encoding in which every child's record
  //  is a CONTIGUOUS, length-prefixed slice, so a relay forwards each child's
  //  sub-subtree verbatim without re-encoding anything:
  //
  //    subtree := u32 quota
  //               u32 num_kids
  //               num_kids x ( i32 kid_node, u32 kid_bytes, kid_bytes of subtree )
  //
  //  The sender of the message is the recipient's parent - that is how
  //  ArrivalPlan::parent is filled in, since section 8.1 stores the parent
  //  rather than deriving it.
  struct BarrierNewPlanMessage {
    ID::IDType barrier_id;
    uint32_t epoch;

    static void handle_message(NodeID sender, const BarrierNewPlanMessage &msg,
                               const void *data, size_t datalen);

    static void send_request(NodeID target, ID::IDType barrier_id, uint32_t epoch,
                             const void *data, size_t datalen);
  };

  // ARRIVAL_PROTOCOL rule 8 - an alter_arrival_count on its way to the owner.
  //
  // The change is PERSISTENT (event.h:268-273, decision D5): 'delta' applies to
  //  'gen' AND to every generation after it.  The implementation this replaces
  //  folded the delta into a single generation and never touched the base
  //  count, which is not what event.h documents and not what Legion expects.
  //
  // 'ts' is this alteration's causal timestamp - globally unique (the issuing
  //  node id sits in the high bits) and monotone per issuing node.  'prev_ts'
  //  is the timestamp of the PREVIOUS alteration the same node issued on this
  //  barrier, or 0 for its first.  The owner HOLDS an alteration whose
  //  predecessor it has not applied yet, so one node's alterations are always
  //  applied in issue order even though active messages may reorder.  That is
  //  what lets a single timestamp on an arrival stand for the whole chain
  //  behind it, and it is the answer to the "TODO: really need two timestamps
  //  to properly order increments" the old ordering code carried.
  struct BarrierAlterMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t gen;
    int delta;
    Barrier::timestamp_t ts;
    Barrier::timestamp_t prev_ts;

    static void handle_message(NodeID sender, const BarrierAlterMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t gen,
                             int delta, Barrier::timestamp_t ts,
                             Barrier::timestamp_t prev_ts);
  };

  // ARRIVAL_PROTOCOL rule 8.1 - AN ARRIVAL CARRYING A TIMESTAMP BYPASSES THE
  //  TREE.  A relay collapses its whole subtree into a single integer, which
  //  ERASES the timestamps, so an arrival that has to be gated on an alteration
  //  is reported straight to the owner instead.  Alterations are rare relative
  //  to arrivals, so the direct traffic is negligible.
  //
  // 'val' is CUMULATIVE (rule 7, decision D4), exactly like a report: it is how
  //  many arrivals this sender has issued for 'gen' under this pair of
  //  timestamps, so the owner REPLACES rather than adds and reordering is
  //  harmless.  No message ever carries a SET of timestamps - that is where the
  //  previous design's causal-DAG machinery came from.
  //
  // The gate is the pair below, and the owner may not count the arrival until
  //  BOTH have been applied (EXACT SET MEMBERSHIP, never "ts <= the highest
  //  applied timestamp"):
  //
  //   'ts'       - the timestamp of the handle the application arrived on,
  //                naming the alteration this arrival is the promised use of
  //                (event.h:275-281).  0 if the handle carries none.
  //   'local_ts' - the SENDER'S OWN alteration floor for 'gen' (the model's
  //                myTs[n][g]).  Once a node has altered, EVERY arrival it
  //                issues for an affected generation is gated, including one
  //                made on a pre-alteration handle: that reserved arrival is
  //                what holds the generation open until the alteration lands,
  //                and counting it early is exactly the "barrier that triggered
  //                too early" of event.h:305.  0 if this node has never altered
  //                this barrier.
  //
  // Two timestamps rather than one because the two dependencies are
  //  independent: the handle names somebody else's alteration, the floor is
  //  this node's own.  Each is the tip of a per-node chain (see 'prev_ts'
  //  above), so a pair is the whole witnessed set and not a truncation of it.
  struct BarrierTsArrivalMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t gen;
    int64_t val;
    Barrier::timestamp_t ts;
    Barrier::timestamp_t local_ts;
    // decision Q4 - see BarrierReportMessage::poisoned.  A bypassed arrival
    //  carries the bit for exactly the same reason a report does.
    bool poisoned;

    static void handle_message(NodeID sender, const BarrierTsArrivalMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t gen,
                             int64_t val, Barrier::timestamp_t ts,
                             Barrier::timestamp_t local_ts, bool poisoned);
  };

  class BarrierImpl : public EventImpl {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

    static constexpr int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;
    static atomic<Barrier::timestamp_t> barrier_adjustment_timestamp;

    BarrierImpl(void);
    ~BarrierImpl(void);

    void init(ID _me, unsigned _init_owner);

    static ID make_id(const BarrierImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_barrier(owner, index, 0);
    }

    // get the Barrier (id+generation) for the current (i.e. untriggered) generation
    Barrier current_barrier(Barrier::timestamp_t timestamp = 0) const;

    // helper to create the Barrier for an arbitrary generation
    Barrier make_barrier(gen_t gen, Barrier::timestamp_t timestamp = 0) const;

    static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
                                       const void *initial_value = 0,
                                       size_t initial_value_size = 0);

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned);

    virtual void subscribe(gen_t subscribe_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned);
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned, long long max_ns);

    virtual bool add_waiter(gen_t needed_gen,
                            EventWaiter *waiter /*, bool pre_subscribed = false*/);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    // 'poisoned' is decision Q4: the arrival's PRECONDITION was poisoned, which
    //  poisons the generation it arrives on.  The arrival itself still counts -
    //  refusing to count it would hang the barrier instead of poisoning it.
    void adjust_arrival(gen_t barrier_gen, int delta, Barrier::timestamp_t timestamp,
                        Event wait_on, NodeID sender, const void *reduce_value,
                        size_t reduce_value_size, TimeLimit work_until,
                        bool poisoned = false);

    // ARRIVAL_PROTOCOL rules 8 and 9 - alter_arrival_count, the whole of action
    //  AL in ONE critical section.  The causal timestamp is MINTED INSIDE that
    //  section, together with the floor it installs and the eager flush it
    //  enters, and it is returned so the caller can put it on the new handle.
    //  (It used to be minted by Barrier::alter_arrival_count and consumed much
    //  later, in a different section - two halves of one spec action.)
    Barrier::timestamp_t alter_arrival_count(gen_t barrier_gen, int delta);

    // NOTIFICATION_PROTOCOL action S (rules 3 and 5), owner only for the
    //  scalable path.  'last_known' is the subscriber's own watermark - what
    //  lets the reply be an exact delta and what replaces the per-node map the
    //  owner used to keep.  'subscribe_gen' is what the subscriber NEEDS, and
    //  is used only by the legacy path's one-shot 'remote_subscribe_gens'.
    void handle_remote_subscription(NodeID subscriber, EventImpl::gen_t subscribe_gen,
                                    EventImpl::gen_t last_known, const void *data,
                                    size_t datalen);

    // NOTIFICATION_PROTOCOL action N (rules 1, 2, 4, 6, 7, 8).  'inset' is this
    //  node's own membership, computed by the handler from the departing set
    //  the message carried - see BarrierNotifyMessage.  'shrink_hint' is rule
    //  8's advisory byte.
    void handle_remote_notify(gen_t wm, gen_t prev, uint64_t sv, const gen_t *poison,
                              size_t num_poisoned, bool inset, bool hint,
                              TimeLimit work_until);

    // NOTIFICATION_PROTOCOL action D, OWNER SIDE (rule 8).  Collect one
    //  departure intent; it is applied - or declined - at the next trigger,
    //  which is also where the resulting shrink is published to the PRE-SHRINK
    //  set.
    void handle_remote_depart(NodeID from);

    // NOTIFICATION_PROTOCOL action RP (rules 2, 5, 7)
    void handle_remote_subscribe_reply(gen_t wm, uint64_t sv, const gen_t *poison,
                                       size_t num_poisoned, TimeLimit work_until);

    // a cumulative subtree total from 'from' (ARRIVAL_PROTOCOL rules 1 and 7).
    //  'data'/'datalen' is the eager-flush (node,count) map, or empty - see
    //  BarrierReportMessage.
    void handle_remote_report(NodeID from, gen_t report_gen, int64_t val, bool is_direct,
                              bool poisoned, const void *data, size_t datalen,
                              TimeLimit work_until);

    // "enter eager-flush mode for 'flush_gen'" (ARRIVAL_PROTOCOL rule 4)
    void handle_remote_flush(gen_t flush_gen);

    // "the plan you hold is being retired" (ARRIVAL_PROTOCOL rule 6)
    void handle_remote_invalidate(uint32_t epoch);

    // "install plan 'epoch'" (ARRIVAL_PROTOCOL rule 5).  'parent' is the node
    //  that sent it - see ArrivalPlan::parent.
    void handle_remote_new_plan(NodeID parent, uint32_t epoch, const void *data,
                                size_t datalen);

    // OWNER ONLY - ARRIVAL_PROTOCOL rules 8 and 9 (action RA).  Apply one
    //  alteration persistently, open the gate on every arrival that was waiting
    //  for it, and let go of any alteration that was waiting for THIS one.
    void handle_remote_alter(NodeID from, gen_t alter_gen, int delta,
                             Barrier::timestamp_t ts, Barrier::timestamp_t prev_ts,
                             TimeLimit work_until);

    // OWNER ONLY - ARRIVAL_PROTOCOL rule 8.1 (action TS).  A cumulative count
    //  of arrivals that BYPASSED the tree because they carry a timestamp.
    void handle_remote_ts_arrival(NodeID from, gen_t arrival_gen, int64_t val,
                                  Barrier::timestamp_t ts, Barrier::timestamp_t local_ts,
                                  bool poisoned, TimeLimit work_until);

    bool get_result(gen_t result_gen, void *value, size_t value_size);

  public:
    // ---- TIER 0: readable with NO lock ------------------------------------
    // Written ONLY inside 'mutex'.  Publication order is fixed
    //  (tla/STATE_AND_LOCKING.md section 3.5) and add_poison_locked() /
    //  publish_watermark_locked() are the only writers:
    //    poison slots -> num_poisoned_generations (release)
    //                 -> generation (release)
    // has_triggered() is the reader and must stay a lock-free, side-effect-free
    //  load - the consultation signal of NOTIFICATION_PROTOCOL rule 8 is
    //  explicitly NOT on that path (section 4).
    atomic<gen_t> generation = atomic<gen_t>(0);

    // decision Q3/D7 - the SAME representation GenEventImpl uses
    //  (event_impl.h:326-332): an append-only array that is never reallocated,
    //  plus an atomic count published with a release store.  Entries below the
    //  published count are immutable, so a reader that acquire-loads the count
    //  can scan them without a lock.  On the OWNER this doubles as the
    //  retention required by NOTIFICATION_PROTOCOL section 8.5 - a new
    //  subscriber has to be told every poisoned generation above its 'lk'.
    // 'constexpr' rather than 'const': this is streamed into a log message in
    //  add_poison_locked(), which binds it to a reference and therefore ODR-uses
    //  it.  A 'static const int' with no out-of-line definition does not link.
    static constexpr int POISONED_GENERATION_LIMIT = 16;
    atomic<int> num_poisoned_generations = atomic<int>(0);
    gen_t *poisoned_generations = nullptr;

    // lock-free (Q3): acquire-load the count, then scan the published slots
    bool is_generation_poisoned(gen_t gen) const;

    // The highest generation this node has ever needed.  It is NOT the
    //  scalable path's subscription state any more - membership is 'member'
    //  plus 'my_set_ver' - it only fills the 'subscribe_gen' field the LEGACY
    //  path's one-shot 'remote_subscribe_gens' is keyed on.
    atomic<gen_t> gen_subscribed = atomic<gen_t>(0);
    gen_t first_generation = 0;
    BarrierImpl *next_free = nullptr;

    Mutex mutex; // controls which local thread has access to internal data (not
                 // runtime-visible event)

    // class to track per-generation status
    class Generation {
    public:
      EventWaiter::EventWaiterList local_waiters;

      // ---- SCALABLE ARRIVAL (redop_id == 0) --------------------------------
      // Every accumulator here is CUMULATIVE, replace-if-higher, and counts UP
      //  (tla/STATE_AND_LOCKING.md D4).  Never mix in an incrementing counter
      //  and never carry the legacy path's count-down convention in here.
      int64_t local_total = 0; // localTotal[n][g] - arrivals issued at this node
      int64_t reported_up = 0; // reportedUp[n][g] - what we last told our parent
      int64_t child_sum = 0;   // running sum of child_acc[*].total
      // THE PINNED REPORT TARGET for this generation, -1 until the first report
      //  goes out.  A report is CUMULATIVE, and the receiver keys its
      //  accumulator on the SENDER and REPLACES that sender's previous
      //  contribution (rule 7) - so one node's contribution to one generation
      //  must live in exactly ONE accumulator chain.  Re-aiming a report at a
      //  different target part-way through a generation puts the SAME arrivals
      //  under two different sender keys at their common ancestor, and nothing
      //  can ever retract the first one: the owner's count steps over
      //  'expected' (a hang) or reaches it early (a barrier that triggers
      //  before every arrival is in).
      //
      // 'cur_plan.parent' changes whenever a new plan is installed - including
      //  the very first one, which moves a node from "direct to the owner" to
      //  "under a relay" - so the plan record cannot be the report target for a
      //  generation this node has already spoken about.  It is pinned here
      //  instead, and the stale edge is flagged 'is_direct' so the path it
      //  travels goes eager rather than aggregating (see record_report_locked).
      //
      // A PINNED EDGE POINTS INTO A PLAN THAT NO LONGER EXISTS, so what stops
      //  two of them forming a CYCLE - A still reporting to its old parent B
      //  while B reports to its new parent A, each report growing the other's
      //  subtree total without bound - is an invariant of plan CONSTRUCTION:
      //  build_new_plan_locked lays 'tv' out with the owner at index 0 and
      //  every other member in ASCENDING NodeID order, and the heap parent of
      //  index i is (i-1)/radix, which is strictly less than i.  So in EVERY
      //  plan a node's parent is either the owner (which never reports) or a
      //  node with a SMALLER NodeID; any report edge, from any plan, strictly
      //  decreases the NodeID; and no cycle can exist.  This is the same
      //  property that already bounds a retired node's reports to its old
      //  parent.  A construction that assigns parents in some other order has
      //  to re-establish it.
      NodeID report_to = -1;
      // flushing[n][g] - eager-flush mode, PER GENERATION (rules 2, 3 and 4).
      //  Once set, every arrival for this generation reports immediately
      //  instead of aggregating.  It is NOT carried into the next generation:
      //  a deviation at generation g says nothing about g+1, so g+1 starts
      //  back in planned mode.
      bool flushing = false;
      // ARRIVAL_PROTOCOL rule 3, propagated.  A 'direct' says "your plan did
      //  not predict me", and only the OWNER acts on it - but a node whose
      //  plan has been retired still reports to its STORED parent (changing
      //  the target mid-generation would let the same arrivals reach the owner
      //  down two paths and be counted twice), so without this the signal is
      //  swallowed by the first relay.  It is sticky for the generation and
      //  OR-ed into every report this node forwards.
      bool saw_direct = false;

      // decision Q4 - an arrival on this generation had a POISONED
      //  PRECONDITION, so the generation itself is poisoned.  Sticky, OR-ed
      //  with every child's bit as reports come up the tree, and read at the
      //  owner when the generation triggers.  It travels with the arrival COUNT
      //  that set it (see BarrierReportMessage::poisoned), so the owner cannot
      //  complete the generation without having seen it.
      bool poisoned = false;

      struct ChildReport {
        // the staleness key of rule 7: the highest cumulative total accepted
        //  from this child for this generation
        int64_t total = 0;
        // ARRIVAL_PROTOCOL section 11.1 - the per-node arrival counts this
        //  child's last EAGER-FLUSH report carried, for its whole subtree.
        //  NON-EMPTY ONLY in flush mode: a steady-state report is O(1) and
        //  carries no map at all.  Replaced wholesale when a report carries
        //  one, left alone when it does not.
        //
        // THIS IS THE PLAN-GATHERING STRUCTURE (section 11.3), and it lives
        //  HERE, inside the generation record, precisely so that it is DELETED
        //  when that generation triggers.  It is never copied into a member of
        //  BarrierImpl - see build_new_plan_locked().
        std::map<NodeID, int64_t> counts;
      };
      // childAcc[n][c][g], keyed by SENDER and never by the current child list:
      //  ARRIVAL_PROTOCOL section 8.2 requires accepting reports from nodes
      //  that are not in cur_plan.kids.
      std::map<NodeID, ChildReport> child_acc;

      // ---- ARRIVAL_PROTOCOL rule 8 - arrivals that BYPASS the tree ----------
      // The gate one bypassed arrival is counted behind.  See
      //  BarrierTsArrivalMessage for what the two timestamps mean and why there
      //  are two of them.  Both may not be zero: an arrival with no timestamp
      //  at all does not bypass anything.
      struct TsKey {
        Barrier::timestamp_t ts = 0;       // the handle's causal timestamp
        Barrier::timestamp_t local_ts = 0; // the issuing node's own floor

        bool operator<(const TsKey &rhs) const
        {
          if(ts != rhs.ts) {
            return (ts < rhs.ts);
          }
          return (local_ts < rhs.local_ts);
        }
      };

      // ISSUING NODE: how many arrivals this node has issued for this
      //  generation under each gate.  It is what makes the message CUMULATIVE
      //  (D4) rather than an increment.  In every realistic case there is
      //  exactly one entry - a second one only appears if a node arrives on
      //  two different alteration branches within one generation.
      std::map<TsKey, int64_t> ts_issued;

      // OWNER: one bypass stream per (sender, gate).  Each stream is
      //  cumulative and replace-if-higher, exactly like child_acc, and is
      //  folded into the count only once EVERY timestamp in its gate has been
      //  applied.  Keying by the gate as well as the sender is what keeps
      //  "cumulative" meaningful when one node's arrivals for a generation sit
      //  behind different alterations.
      struct TsStream {
        int64_t seen = 0;     // highest cumulative value this sender reported
        bool counted = false; // whether 'seen' has been credited yet
      };
      struct TsStreamKey {
        NodeID from = -1;
        TsKey gate;

        bool operator<(const TsStreamKey &rhs) const
        {
          if(from != rhs.from) {
            return (from < rhs.from);
          }
          return (gate < rhs.gate);
        }
      };
      std::map<TsStreamKey, TsStream> ts_streams;
      // the model's TsTotal(g): the sum of 'seen' over every COUNTED stream.
      //  Never an independent counter - it is maintained as those streams are
      //  credited, so it moves in exactly one direction (D4).
      int64_t ts_acc = 0;

      // derived, never stored (ARRIVAL_PROTOCOL section 3)
      int64_t subtree_known(void) const { return local_total + child_sum; }
      int64_t unreported(void) const { return subtree_known() - reported_up; }
      bool holding(void) const { return unreported() > 0; }

      // ---- LEGACY / REDUCTION PATH ONLY ------------------------------------
      // the count-down accumulator: expected(gen) + unguarded_delta == 0
      int unguarded_delta = 0;
    };

    std::map<gen_t, Generation *> generations;

    // external waiters on this node are notifies via a condition variable
    bool has_external_waiters = false;
    // use kernel mutex for timedwait functionality
    KernelMutex external_waiter_mutex;
    KernelMutex::CondVar external_waiter_condvar;

    // a list of remote waiters and the latest generation they're interested in.
    //  LEGACY / REDUCTION PATH ONLY - the scalable path's replacement is
    //  'sub_set' plus 'set_ver', which is a bitmask rather than an O(N) map.
    std::map<unsigned, gen_t> remote_subscribe_gens;

    // ---- NOTIFICATION, NODE SIDE (needed on BOTH paths) --------------------
    // NB: this is THIS NODE's belief about ITS OWN membership of the owner's
    //  subscriber set.  It is one enum per barrier, NOT a map over nodes: the
    //  owner keeps no per-node map (NOTIFICATION_PROTOCOL section 3).
    //
    // MEMBERSHIP IS PUBLISHED, NEVER SELF-DECIDED (rule 1).  The only writers
    //  that may set YES/NO are the two publication paths - a notification and a
    //  subscribe reply - and both are VERSION GATED on 'my_set_ver' (rule 2).
    //  PENDING means "I have asked and have not been answered", and it is what
    //  keeps a second consultation from issuing a second subscribe.
    enum MemberState : uint8_t
    {
      MEMBER_NO = 0,
      MEMBER_PENDING = 1,
      MEMBER_YES = 2
    };
    MemberState member = MEMBER_NO;
    // rule 2: the highest 'set_ver' applied here.  A membership update is
    //  applied only if it is STRICTLY newer, on BOTH the notify and the reply
    //  path.  Without this a stale in-flight notification resurrects membership
    //  the owner has already dropped, and the node hangs on its next wait.
    uint64_t my_set_ver = 0;
    // rule 7: at most one outstanding pull.  Set when a subscribe goes out,
    //  cleared by the reply (or, on the legacy path, by the trigger message
    //  that serves as the reply).
    bool pull_outstanding = false;
    // The model suppresses a pull only while a SUBSCRIBE is in flight
    //  (BarrierNotify:157); 'pull_outstanding' stays set until the REPLY lands,
    //  which is a strictly longer window.  A pull the gap rule wanted during
    //  that extra window is remembered here and issued when the reply closes
    //  it - otherwise a notification discarded for a gap while a subscribe was
    //  being answered would never be recovered, and a node whose barrier has no
    //  further generations would wait forever.
    bool pull_deferred = false;

    // ---- NOTIFICATION_PROTOCOL rule 8: DEPARTURE HYSTERESIS, node side -----
    //
    // PERFORMANCE ONLY, and deliberately unverified.  BarrierNotify abstracts
    //  this whole mechanism as nondeterminism - 'Depart' may fire on ANY
    //  eligible node and 'Trigger' may apply ANY subset of the requests - which
    //  is strictly more general than any tuning, so NO CHOICE OF THESE
    //  CONSTANTS CAN BE WRONG (section 8.2).  The failure modes are bandwidth;
    //  correctness is carried by rule 6 (a node removed while holding a waiter
    //  re-subscribes at once) and by the pull path, which has to exist anyway.
    //
    // K, in GENERATIONS.  A node asks to leave after this many consecutive
    //  triggered generations without consulting the barrier.
    static constexpr unsigned DEPART_K_INITIAL = 8;
    // the ceiling the churn adaptation doubles toward.  A node that keeps being
    //  dropped and re-subscribing eventually stops asking altogether, which is
    //  the correct steady state for a node that is simply slow rather than idle.
    static constexpr unsigned DEPART_K_MAX = 1024;
    // J.  The eligibility threshold is 'K + (my_node_id % J)', so a phase change
    //  that retires many nodes at once spreads their requests over J
    //  generations instead of delivering them as one spike.  EXPRESSED IN
    //  GENERATIONS, NOT IN TIME - so it needs no timer and the protocol stays
    //  feed-forward.  Sized for spike smoothing, not for asymptotics: the
    //  subscribe path is already O(N) unicast.
    static constexpr unsigned DEPART_STAGGER_J = 32;
    // "shortly after having left", in generations: a re-subscribe this soon
    //  after a departure request is CHURN, and doubles K.
    static constexpr gen_t DEPART_CHURN_WINDOW = 64;

    // THE IDLE COUNTER, measured as the WATERMARK DELTA since this node last
    //  consulted the barrier - not as a count of notifications received, which
    //  would be wrong the moment the owner coalesced several triggers into one
    //  message (and coalescing is explicitly supported: see action T).
    //
    // CONSULTING means add_waiter, subscribe, or external_wait /
    //  external_timedwait - and explicitly NOT has_triggered(), which must stay
    //  a lock-free single atomic load (tla/STATE_AND_LOCKING.md section 4).
    //  This field is written ONLY from action C, which already holds 'mutex'
    //  for other reasons.
    gen_t last_consult_wm = 0;
    // the watermark at which this node last ASKED to leave (0 = never).  It is
    //  the churn window's origin, and it is not evidence the node actually left
    //  - the owner may have declined.
    gen_t last_depart_wm = 0;
    // K, doubled on observed churn and capped at DEPART_K_MAX
    unsigned depart_K = DEPART_K_INITIAL;
    // one request per idle episode.  Cleared when this node next consults
    //  (action C) or when it is actually removed (action N step 1), which is
    //  what keeps a declined request from turning into a retry storm - a
    //  decline costs one message and then silence.
    bool depart_outstanding = false;
    // the owner's advisory byte from the last notification.  Optimistic until
    //  told otherwise, so the first departure always gets a hearing.
    bool shrink_hint = true;

    // ---- NOTIFICATION, OWNER SIDE (NOTIFICATION_PROTOCOL section 3) --------
    // D8: the wire form IS the storage form.  There is deliberately no learned
    //  notification tree - the multicast layer plans a fresh forwarding tree
    //  from the encoded set on every send, so a set change can never leave a
    //  stale tree behind.  This is the ONLY O(N) term on the scalable path, and
    //  it is a bitmask (or a couple of runs), not a map.
    MulticastTargetSet sub_set;
    // rule 2 - bumped on EVERY change to 'sub_set' and stamped onto every
    //  notification and every reply
    uint64_t set_ver = 0;
    // rule 8 - departure intents collected but not yet applied.  Normally
    //  empty.  Applied (or declined - removals are DISCRETIONARY, rule 3) at
    //  the next trigger, which is also where the shrink is published to the
    //  PRE-SHRINK set.  It is CLEARED at every trigger whether or not the
    //  shrink was applied: a declined intent is forgotten rather than retried,
    //  and the node's own one-request-per-idle-episode latch is what stops it
    //  being re-sent.
    MulticastTargetSet want_out;
    // rule 8's advisory byte, as the owner currently believes it: the verdict
    //  the cost test returned the last time it actually evaluated a shrink.
    //  STICKY - a trigger with nothing to weigh leaves it alone.  The only
    //  thing that clears a declined verdict is an ADD, because that is what
    //  changes the shape of the set the verdict was about.  Optimistic to
    //  start, so the first departure of a barrier always gets a hearing.
    bool shrink_pays = true;

    // ---- SCALABLE ARRIVAL: the plan record, on every node -------------------
    // curPlan[n].  MEMBERSHIP RULE (ARRIVAL_PROTOCOL section 3): only nodes
    //  with a NON-ZERO expected contribution are in a plan at all.  A node with
    //  no plan is not a leaf, not a relay, not anything - it has no parent and
    //  no completion condition it could wait for, so every arrival goes
    //  straight to the owner (rule 3) and it is permanently in eager-flush
    //  mode.  That is what stops a relay ever waiting on a child that will
    //  never speak.
    struct ArrivalPlan {
      uint32_t quota = 0;       // arrivals this plan predicts at this node
      bool inplan = false;      // quota 0 => not in the tree at all
      NodeID parent = -1;       // -1 => report direct to the owner.  STORED, not
                                //  derived - the model's global ParentOf() search
                                //  is a modelling convenience (section 8.1)
      std::vector<NodeID> kids; // O(radix)
    };

    ArrivalPlan cur_plan;     // curPlan[n]
    uint32_t my_epoch = 0;    // myEpoch[n]   - which plan index cur_plan is
    uint32_t inval_epoch = 0; // invalEpoch[n] - highest plan index retired here
    uint32_t defer_epoch = 0; // deferEpoch[n] - a parked new plan, 0 = none

    // ---- OWNER ONLY: the plan lifecycle (ARRIVAL_PROTOCOL section 11) ------
    // The next plan index to hand out.  Epochs are monotone and never reused,
    //  which is what makes the invalidate/newplan race resolvable locally
    //  (rule 5).  0 means "no plan has ever existed", so counting starts at 1.
    uint32_t next_epoch = 1;
    // Set by the only thing that can tell the owner its plan is wrong: the
    //  owner entering eager-flush mode (a 'direct' from a node the plan did not
    //  predict, or an over-arrival at the owner itself).  Consumed by the next
    //  trigger, which is where the aggregated maps are still in hand.
    //
    // NOTE what is NOT here: the merged node->count structure of section 11.2.
    //  It is a LOCAL of the trigger path, merged out of the generation record
    //  that is about to be freed and destroyed with the stack frame.  O(N)
    //  barriers each retaining an O(N) participant map is the O(N^2) blow-up
    //  this whole design exists to avoid (section 11.3, decision D9), and
    //  nothing fails loudly if it leaks - so it has no home to leak into.
    bool plan_rebuild_pending = false;
    // At most ONE parked plan (BarrierArrive BoundedRetention).  Holds the encoded
    //  SUBTREE payload, not just an epoch, because the node has to forward it
    //  to its own children when the invalidation lands.  COPIED out of the
    //  active-message buffer, which is dead once the handler returns.
    std::vector<unsigned char> deferred_plan_payload;
    // who sent the parked plan - it becomes cur_plan.parent when the plan is
    //  finally applied, so it has to be remembered with the payload
    NodeID defer_parent = -1;

    // ---- ARRIVAL_PROTOCOL rule 8: alteration state on the ISSUING node -----
    // myTs[n][g] as a STEP FUNCTION: the greatest entry with key <= g gives the
    //  timestamp this node's arrivals for g must carry, and 0 (no entry) means
    //  they carry none.  It is a step function because alterations are
    //  PERSISTENT - one covers every later generation too - and installing an
    //  entry at 'g' therefore drops every entry above 'g', which is superseded
    //  by it.
    std::map<gen_t, Barrier::timestamp_t> ts_floor;
    // the last alteration this node issued on this barrier: the 'prev_ts' of
    //  the next one.  That chain is what keeps one node's alterations applied
    //  at the owner in the order they were issued, and therefore what lets a
    //  single timestamp on an arrival stand for everything behind it.
    Barrier::timestamp_t last_alter_ts = 0;

    // ---- OWNER ONLY: the expected arrival count, as a step function --------
    // expected(g) == base_arrival_count + alter_floor
    //                + sum of alter_steps entries with key <= g
    //
    // ALTERATIONS ARE PERSISTENT (D5, event.h:268-271): a delta applies to the
    //  generation it names and to every generation after it, which is why this
    //  is a step function and not a per-generation delta.  'alter_floor'
    //  absorbs the steps at or below the watermark as generations trigger, so
    //  the map holds only the breakpoints still in front of us.
    //  (tla/STATE_AND_LOCKING.md 2.4.4 writes the same quantity as a single
    //  'expected_floor' seeded from base_arrival_count; keeping the base
    //  separate means nothing has to be re-seeded after create_barrier sets
    //  it.)
    int64_t alter_floor = 0;
    std::map<gen_t, int64_t> alter_steps;

    // appliedTs.  A bypassed arrival is counted only when EVERY timestamp in
    //  its gate is IN THIS SET - exact membership, never "ts <= the highest
    //  applied".  Two alterations can be applied out of order (the transport
    //  reorders), and a "highest applied" comparison admits an arrival whose
    //  own alteration has not landed: a barrier that triggers too early.  The
    //  value is the generation the alteration applies from, kept for pruning.
    std::map<Barrier::timestamp_t, gen_t> applied_ts;

    // OWNER ONLY - the nodes whose arrivals BYPASS THE TREE (rule 8.1), and so
    //  the nodes that must never appear in a plan.
    //
    // An alteration is PERSISTENT: once a node has altered, myTs[n][g] is
    //  non-zero for that generation and EVERY later one, so from then on every
    //  arrival it issues is a 'tsdirect' and its 'local_total' stays at zero
    //  forever.  Giving such a node a quota hands it a completion condition it
    //  can never reach: as a relay it goes silent, and as a LEAF its parent
    //  waits on rule 1's child-wait for a report that will never come
    //  (ARRIVAL_PROTOCOL section 11.2 - "only nodes with a non-zero count
    //  appear ... it is what stops a relay ever waiting on a child that will
    //  never speak").  The plan is built from 'agg', which is one PAST
    //  generation's local totals, so a node that altered for a LATER generation
    //  is still in that evidence and would be re-admitted on every rebuild.
    //
    // Bounded by the number of distinct nodes that ever alter this barrier, and
    //  strictly smaller than 'applied_ts', which already retains one entry per
    //  alteration.
    std::set<NodeID> ts_bypass_nodes;

    // alterations whose PREDECESSOR (BarrierAlterMessage::prev_ts) has not been
    //  applied yet, keyed by the timestamp they are waiting for.  A multimap
    //  because nothing stops two alterations naming the same predecessor.
    //  Bounded by the alterations actually in flight.
    struct HeldAlter {
      gen_t gen = 0;
      int delta = 0;
      Barrier::timestamp_t ts = 0;
    };
    std::multimap<Barrier::timestamp_t, HeldAlter> held_alters;

    unsigned base_arrival_count = 0;
    ReductionOpID redop_id = 0;
    const ReductionOpUntyped *redop = nullptr;
    std::unique_ptr<char[]> initial_value{};
    unsigned value_capacity = 0;
    std::vector<char> final_values;

    // ---- LEGACY / REDUCTION PATH ONLY ------------------------------------
    // Reduction barriers stay on the existing non-scalable path (see
    //  tla/STATE_AND_LOCKING.md D1).  Everything in here is unreachable when
    //  redop_id == 0, so the scalable path pays neither the memory nor the
    //  per-node bookkeeping.
    struct LegacyReductionState {
      // the latest generation that each node (that has ever subscribed) has
      //  been told about.  This is what slices 'final_values' per recipient, so
      //  it cannot be collapsed into a single owner-side watermark the way the
      //  scalable path's notification can.
      std::map<unsigned, gen_t> remote_trigger_gens;
      // trigger notifications that arrived out of order, buffered until the
      //  generations below them show up.  The scalable path discards instead
      //  (NOTIFICATION_PROTOCOL rule 4); the legacy path keeps buffering.
      std::map<gen_t, gen_t> held_triggers;
    };
    // non-null iff this is a reduction barrier (redop_id != 0)
    std::unique_ptr<LegacyReductionState> legacy;

    // ---- messages RECORDED under 'mutex' and emitted AFTER it (S2) ---------
    // The emit phase reads ONLY this struct and immutable members - never
    //  'cur_plan' (S3).  That is what makes rule 6's "forward before
    //  forgetting" work without any extra effort: every target was resolved
    //  inside the critical section, before anything could replace the child
    //  list.
    struct PendingReport {
      gen_t gen = 0;
      NodeID to = -1;         // -1 => this slot holds no report
      int64_t val = 0;        // CUMULATIVE subtree total (rule 7)
      bool is_direct = false; // rule 3: straight to the owner
      bool poisoned = false;  // Q4, sticky per generation
      // the encoded (node,count) map of section 11.1, EMPTY unless this node
      //  is in eager-flush mode.  A default-constructed vector allocates
      //  nothing, so a steady-state report still costs no allocation.
      std::vector<unsigned char> flush_map;
    };

    // NOTIFICATION_PROTOCOL action T, the notification half.  EVERY field is
    //  materialised inside the critical section (S3), including the target set,
    //  which is a snapshot of 'sub_set' taken BEFORE the shrink is applied -
    //  rule 1's "any shrink must be published to the PRE-SHRINK set".  That is
    //  the one thing in this protocol that reordering cannot fix, so it is
    //  handled by snapshotting rather than by ordering.
    struct PendingNotify {
      bool valid = false;
      MulticastTargetSet targets; // the PRE-SHRINK subscriber set
      gen_t wm = 0;
      gen_t prev = 0;
      uint64_t set_ver = 0;
      std::vector<gen_t> poison;            // poisoned generations in (prev, wm]
      std::vector<unsigned char> departing; // encoded R; empty if set_ver is
                                            //  unchanged, in which case no
                                            //  recipient will apply membership
      bool shrink_hint = true;              // rule 8's advisory byte
    };

    // one child of the NEW tree, with its own sub-subtree payload
    struct PendingPlan {
      NodeID to = -1;
      std::vector<unsigned char> payload;
    };

    // one generation's flush fan-out.  Rule 8 enters eager flush for EVERY open
    //  generation at or after the altered one, so a single (generation, targets)
    //  pair is no longer enough.
    struct PendingFlush {
      gen_t gen = 0;
      std::vector<NodeID> to;
    };

    // rule 8 - the alteration travelling to the owner
    struct PendingAlter {
      gen_t gen = 0;
      int delta = 0;
      Barrier::timestamp_t ts = 0;
      Barrier::timestamp_t prev_ts = 0;
    };

    // rule 8.1 - the bypassed arrival travelling to the owner
    struct PendingTsArrival {
      gen_t gen = 0;
      int64_t val = 0; // CUMULATIVE for this gate (rule 7)
      Barrier::timestamp_t ts = 0;
      Barrier::timestamp_t local_ts = 0;
      bool poisoned = false; // Q4, sticky per generation
    };

    struct PendingSends {
      // The first - and, on every hot path, only - report.  Kept scalar so a
      //  steady-state arrival allocates nothing at all.
      PendingReport report;
      // Rule 6 flushes EVERY open generation, so one report is not enough: a
      //  node may be holding arrivals for several generations at once.
      //  Generations after the first spill into here, which stays empty on the
      //  arrival and report paths.
      std::vector<PendingReport> more_reports;

      // one entry per generation entering eager flush; empty in steady state,
      //  so a default-constructed vector allocates nothing
      std::vector<PendingFlush> flushes;

      // rule 8: at most ONE alteration and at most ONE bypassed arrival per
      //  action, so these stay scalar
      bool has_alter = false;
      PendingAlter alter;
      bool has_ts_arrival = false;
      PendingTsArrival ts_arrival;

      // rule 6 step (a): the invalidation being forwarded down the tree BEING
      //  RETIRED.  0 => nothing to forward.
      uint32_t fwd_inval_epoch = 0;
      std::vector<NodeID> inval_to;

      // rule 5: the new plan being passed on to this node's children in the NEW
      //  tree.  0 => nothing to forward.
      uint32_t fwd_plan_epoch = 0;
      std::vector<PendingPlan> plan_to;

      // ---- NOTIFICATION_PROTOCOL --------------------------------------------
      // action T's notification half, owner only.  At most one per section,
      //  because a trigger drains one contiguous run of generations and
      //  coalescing them into a single delta is explicitly supported.
      PendingNotify notify;

      // the pull (rule 5's 'subscribe'), always unicast to the owner.  At most
      //  one per section - rule 7.
      bool has_subscribe = false;
      gen_t subscribe_lk = 0;   // what I already know: bounds the reply's delta
      gen_t subscribe_need = 0; // what I am waiting for: LEGACY path only

      // rule 8's departure intent, unicast to the owner.  At most one per
      //  section - the eligibility test latches 'depart_outstanding'.
      bool has_depart = false;
    };

    // ---- IMPLEMENTATION_PLAN section 5 - instrument from day one -----------
    //
    // The specs abstract K, J, the shrink policy and the plan-rebuild trigger
    //  as NONDETERMINISM precisely so that no choice of constant can be wrong,
    //  which is the same reason none of them can be validated by a model.
    //  Only measurement can tune them, so these are the four quantities
    //  IMPLEMENTATION_PLAN section 5 asks to have counted from the start.
    //
    // Every field is written ONLY under 'mutex', beside the state it describes,
    //  so it costs one non-atomic increment on a path that is already locked.
    //  Nothing branches on any of them - they are diagnostics.
    struct BarrierCounters {
      // --- NODE SIDE, rule 8.  'leave_rejoin_cycles' near zero is the
      //     expectation that validates K = 8; a rising count is exactly what
      //     doubles K, so the counter and the adaptation share one signal.
      uint64_t departs_sent = 0;
      uint64_t departs_suppressed = 0; // rule 8's hint said it would be declined
      uint64_t leave_rejoin_cycles = 0;
      uint64_t churn_backoffs = 0; // rejoins inside DEPART_CHURN_WINDOW

      // --- OWNER SIDE.  'subscribe_fan_in' is the last unaggregated O(N) path
      //     in the design, so it is the one to watch for the escape hatch.
      uint64_t subscribe_fan_in = 0;
      uint64_t departs_received = 0;
      uint64_t shrinks_applied = 0;
      uint64_t shrinks_declined = 0; // the cost test said the removal loses
      uint64_t nodes_removed = 0;

      // --- ARRIVAL, the deviation path (ARRIVAL_PROTOCOL section 11.1).  An
      //     eager-flush report is the only O(subtree) payload in the protocol.
      uint64_t flush_episodes = 0;
      uint64_t flush_report_bytes = 0;
      size_t flush_report_bytes_max = 0;

      // --- OWNER, plan lifecycle (ARRIVAL_PROTOCOL sections 11.2 and 11.3).
      //     'agg_peak_entries' is the memory bound D9 exists to protect: the
      //     aggregation structure is a LOCAL of the trigger path, and this is
      //     the number that would show it silently growing if it ever stopped
      //     being one.
      uint64_t plan_rebuilds = 0;
      size_t agg_peak_entries = 0;

      // --- RULE 10, the pinned-edge machinery (SCALE_TEST_PLAN section 1).
      //     Each is the exercise-proof for a verified rule: a scale run where
      //     one of these never moves has NOT tested that rule, no matter how
      //     green it is.  The park/dead/retro/stale counters are the race
      //     windows - they cannot be forced deterministically from the
      //     application, only made probable (churn phases, many seeds).
      uint64_t plans_parked = 0;          // rule 5 deferral engaged
      uint64_t parked_plans_applied = 0;  // ...and resolved by an invalidation
      uint64_t dead_plans_discarded = 0;  // rule 10.2/10.3 guard fired
      uint64_t retro_flushes_sent = 0;    // rule 10.4 late case 3
      uint64_t stale_edge_forwards = 0;   // rule 10.5 receiver-side forward
      uint64_t report_edges_pinned = 0;   // rule 10.1, per (generation) pin
      uint64_t pin_conflicts_avoided = 0; // pinned target differed from current parent
      uint64_t gap_pulls = 0;             // NOTIFICATION rule 4 gap -> pull

      // has anything at all happened worth reporting?
      bool any(void) const;
    };
    BarrierCounters counters;

    // IMPLEMENTATION_PLAN section 5 - THE WAY OUT.  The counters above are
    //  written under 'mutex' on paths that already hold it and nothing branches
    //  on them, so without a reader they are write-only.  Two exist:
    //
    //   * dump_counters(), called when the barrier slot is recycled and
    //     when it is destroyed, which logs one line per barrier that actually
    //     used the deviation or hysteresis paths (log level 'info', category
    //     'barrier'); and
    //   * get_counters(), for a debugger or a test that wants the numbers
    //     directly.  Caller must hold 'mutex' if the barrier is live.
    const BarrierCounters &get_counters(void) const { return counters; }

    // Silent unless something was counted.  Called where the barrier is
    //  quiescent (slot re-initialisation and destruction), so it takes no lock
    //  of its own; any other caller must hold 'mutex'.
    void dump_counters(const char *why) const;

#ifdef DEBUG_REALM
    // DEBUG ONLY - fail loudly on a '..._locked' helper called without the
    //  lock.  If 'mutex' can be taken here then NOBODY held it, and in
    //  particular this thread did not, which is the bug.  Failing to take it is
    //  INCONCLUSIVE (another thread may hold it), so this never fires falsely.
    void assert_locked(const char *who) const;
#define REALM_BARRIER_ASSERT_LOCKED() this->assert_locked(__func__)
#else
#define REALM_BARRIER_ASSERT_LOCKED()                                                    \
  do {                                                                                   \
  } while(0)
#endif

    // ---- scalable arrival helpers.  ALL of these require 'mutex' -----------

    // find or create the record for 'gen'
    Generation *get_generation_locked(gen_t gen);

    // ARRIVAL_PROTOCOL rule 1, BOTH halves: this node's local arrival count
    //  equals the quota its plan predicts AND every child the plan predicts has
    //  reported at least once.
    bool plan_satisfied_locked(const Generation &g) const;

    // where a report from this node goes (section 8.1: the parent is stored)
    NodeID report_target_locked(void) const;

    // has the plan this node holds been retired without a replacement having
    //  arrived yet?  See should_report_locked() for why this matters.
    bool plan_retired_locked(void) const;

    // is this node reporting eagerly for 'g' rather than aggregating under a
    //  valid plan?  The union of rules 2/4 (the per-generation flag), rule 3
    //  (no plan record at all), rule 6's retired-but-not-yet-replaced state and
    //  rule 8.3 (this node's arrivals bypass the tree, so its quota is
    //  unreachable).  It is also exactly the condition under which a report
    //  carries the (node,count) map of section 11.1.
    bool flush_mode_locked(gen_t gen, const Generation &g) const;

    // the forwarding decision of action A step 5 / action R step 6
    bool should_report_locked(gen_t gen, const Generation &g) const;

    // record the cumulative report this node owes its parent, and remember we
    //  have now passed that much on (reported_up)
    void record_report_locked(Generation &g, gen_t gen, PendingSends &sends);

    // BarrierArrive's 'held' set: report EVERY open generation this node is
    //  sitting on, whatever mode it is in.  Used by rules 5 and 6.
    void report_held_locked(PendingSends &sends);

    // ARRIVAL_PROTOCOL rule 4 - enter eager-flush mode for one generation.
    //  The shared core of rules 2, 3, 4 and 8; IDEMPOTENT.
    void enter_flush_locked(Generation &g, gen_t gen, PendingSends &sends);

    // ARRIVAL_PROTOCOL rule 8.3 - enter eager flush for EVERY open generation
    //  at or after 'from_gen'.  Generations this node has no record for yet are
    //  covered by ts_floor rather than by a flag, exactly as rule 6's
    //  quantifier is covered by plan_retired_locked() - see flush_mode_locked.
    void enter_flush_from_locked(gen_t from_gen, PendingSends &sends);

    // ARRIVAL_PROTOCOL rule 8 - the timestamp this node's arrivals for 'gen'
    //  must carry (the model's myTs[n][g]): the greatest ts_floor entry with
    //  key <= gen, or 0 if this node has never altered this barrier.
    Barrier::timestamp_t ts_floor_locked(gen_t gen) const;

    // OWNER ONLY - expected(gen).  See alter_floor / alter_steps.
    int64_t expected_locked(gen_t gen) const;

    // OWNER ONLY - is every timestamp in this gate applied?  EXACT SET
    //  MEMBERSHIP: never a "<= highest applied" comparison.
    bool ts_gate_open_locked(const Generation::TsKey &gate) const;

    // OWNER ONLY - credit 'count' arrivals to a generation, on whichever
    //  accumulator that generation's path uses
    void credit_arrivals_locked(Generation &g, int64_t count) const;

    // OWNER ONLY - action TS.  Accept one cumulative bypassed-arrival report
    //  and credit it if its gate is open (park it in place if not).
    void apply_ts_arrival_locked(NodeID from, Generation &g,
                                 const Generation::TsKey &gate, int64_t val);

    // OWNER ONLY - a timestamp has just been applied, so every parked stream
    //  gated on it may now be counted
    void open_ts_gates_locked(Barrier::timestamp_t ts);

    // OWNER ONLY - ARRIVAL_PROTOCOL rule 9.  Retire the current plan: send the
    //  invalidation down the tree being retired and retire this node's own
    //  epoch, which leaves every node in it eagerly reporting until a new plan
    //  is built.  IDEMPOTENT - a plan already retired is left alone.
    void retire_plan_locked(PendingSends &sends);

    // OWNER ONLY - action RA.  Apply one alteration persistently, open the
    //  gates it holds shut, and release any alteration held behind it.
    void apply_alter_locked(gen_t alter_gen, int delta, Barrier::timestamp_t ts,
                            Barrier::timestamp_t prev_ts, PendingSends &sends);

    // ARRIVAL_PROTOCOL section 8.8 - retention.  Fold the alteration steps at
    //  or below the watermark into the floor and drop the ts_floor breakpoints
    //  no arrival can name any more.
    void prune_alter_state_locked(gen_t watermark);

    // ARRIVAL_PROTOCOL rule 5 - install a plan record decoded from a 'newplan'
    //  payload and record the fan-out to this node's children in the new tree.
    //  Reports held work FIRST, while the OLD plan is still in place, because
    //  that is who owes the old parent (BarrierArrive evaluates Send() in the
    //  unprimed state).
    void apply_plan_locked(uint32_t epoch, const void *data, size_t datalen,
                           NodeID parent, PendingSends &sends);

    // ARRIVAL_PROTOCOL section 11.1 - the (node,count) map this node speaks
    //  for: its own local arrivals merged with every child's last flush map.
    //  Merging is per-node REPLACE-IF-HIGHER, never addition: each entry is
    //  that node's own cumulative local total, so a node that appears under
    //  two children (its old parent and its new one, mid-switch) must be
    //  counted ONCE, at its highest observed value.  Summing would over-count
    //  and an over-count is what makes a plan deadlock - see
    //  build_new_plan_locked().
    void merge_counts_locked(const Generation &g, std::map<NodeID, int64_t> &out) const;

    // ARRIVAL_PROTOCOL section 11.2 and 11.4 - OWNER ONLY.  Turn the merged
    //  node->count map into a tree plus quotas, install the owner's own
    //  record, and record the invalidate/newplan fan-out.  'agg' is a local of
    //  the caller and is not retained (section 11.3).
    //
    // 'tree_arrivals' is the number of arrivals the gathered generation took
    //  THROUGH THE TREE (its expected count less the arrivals that bypassed
    //  it).  Section 11.2 reads the merged maps as "the participant set", which
    //  they only are when every participant was eager and therefore carried a
    //  map; comparing their sum against this says whether that was the case.
    void build_new_plan_locked(const std::map<NodeID, int64_t> &agg,
                               int64_t tree_arrivals, PendingSends &sends);

    // owner only: has 'g' accounted for every arrival it is waiting for?
    bool generation_complete_locked(gen_t gen, const Generation &g) const;

    // ---- NOTIFICATION_PROTOCOL helpers.  ALL of these require 'mutex' ------

    // decision Q3 - record one poisoned generation and PUBLISH it.  Idempotent.
    //  Past POISONED_GENERATION_LIMIT this logs fatal and aborts: there is no
    //  representation that stays lock-free past the cap.
    void add_poison_locked(gen_t gen);

    // Advance the watermark to 'wm', publishing the poison for the range FIRST
    //  (section 3.5), then drain and free every generation record at or below
    //  it, sorting the waiters by whether their generation was poisoned.
    //  Signals external waiters inside the section, as the three existing
    //  signal sites do.
    void publish_watermark_locked(gen_t wm, const gen_t *poison, size_t num_poisoned,
                                  EventWaiter::EventWaiterList &clean,
                                  EventWaiter::EventWaiterList &poisoned);

    // the model's 'waiting[n] # {}': does this node still hold an untriggered
    //  waiter?  Used by rule 6's re-subscribe test.  No new member is needed.
    bool has_waiters_locked(void) const;

    // the generation to put in a subscribe's 'subscribe_gen' field
    gen_t subscribe_need_locked(void) const;

    // NOTIFICATION_PROTOCOL action C step 4 - the CONSULTATION-driven pull.
    //  Deliberately NOT gated on 'pull_outstanding': the model's Consult has no
    //  such guard (BarrierNotify:122-126), because 'member' going to PENDING is
    //  what suppresses a duplicate.  Does nothing if this node is the owner or
    //  already believes itself covered.
    void consult_subscribe_locked(PendingSends &sends);

    // NOTIFICATION_PROTOCOL action N step 5 / action RP - the RECOVERY pull,
    //  rule 7 gated.  If a pull is already outstanding this remembers the
    //  request in 'pull_deferred' rather than dropping it.
    void record_pull_locked(PendingSends &sends);

    // ---- NOTIFICATION_PROTOCOL rule 8 helpers.  ALL require 'mutex' --------

    // action C step 3 - THE CONSULTATION SIGNAL.  Resets the idle counter and
    //  retires any departure intent this node was holding.  Called from
    //  add_waiter, subscribe and (through subscribe) external_wait /
    //  external_timedwait.  NEVER from has_triggered().
    void note_consultation_locked(void);

    // this node is going from "not a member" back to wanting in.  If it had
    //  asked to leave, that is a leave->rejoin cycle; inside
    //  DEPART_CHURN_WINDOW generations it is CHURN and doubles K.
    void note_rejoin_locked(void);

    // action N step 6 / action RP - the departure eligibility test, evaluated
    //  where the watermark moves.  Records the unicast intent in 'sends'.
    void consider_departure_locked(PendingSends &sends);

    // OWNER ONLY - rule 3's COST TEST.  Would dropping 'departing' from
    //  'sub_set' actually reduce the cost of a notification?  Removals are
    //  discretionary precisely so this can say no: dropping scattered nodes
    //  from an ALL_NODES encoding turns a 1-byte target set into a per-hop
    //  bitmap, and that bitmap rides EVERY hop of the forwarding tree.
    bool shrink_pays_locked(const MulticastTargetSet &departing) const;

    // owner only: drain every contiguous generation that is now complete.
    //  Returns the highest generation triggered (0 if none) and records the
    //  waiters and remote notifications to be emitted AFTER the unlock.  A
    //  plan switch due at the new watermark + 1 is built and recorded here
    //  too, because this is the last moment the drained generation's
    //  aggregation maps still exist (section 11.3).
    //
    // BarrierArrive's 'Trigger' and BarrierNotify's 'Trigger' are ONE critical
    //  section here (tla/STATE_AND_LOCKING.md section 3.4 action T): they are
    //  the same real-world event, they share the watermark, and the
    //  notification's 'prev' must be the pre-trigger watermark of the same
    //  contiguous chain.  Splitting them leaves a window in which the watermark
    //  has advanced but no notification describes it.
    gen_t check_triggers_locked(EventWaiter::EventWaiterList &local_notifications,
                                EventWaiter::EventWaiterList &poisoned_notifications,
                                std::vector<RemoteNotification> &remote_notifications,
                                gen_t &oldest_previous, PendingSends &sends);

    // 'mutex' must NOT be held (S2/S3)
    void emit_pending_sends(const PendingSends &sends);
    void emit_report(const PendingReport &r);

    // 'mutex' must NOT be held.  Waiters woken with the wrong poison status is
    //  one of the three unsafe imprecisions this protocol exists to prevent, so
    //  the two lists are kept apart all the way to the trigger call.
    static void deliver_waiters(EventWaiter::EventWaiterList &clean,
                                EventWaiter::EventWaiterList &poisoned,
                                TimeLimit work_until);

    // 'mutex' must NOT be held: the emit phase reads only its arguments and
    //  immutable members (S2/S3)
    void emit_trigger_notifications(gen_t trigger_gen,
                                    const std::vector<RemoteNotification> &rns,
                                    gen_t oldest_previous, const void *final_values_copy,
                                    size_t sizeof_lhs);

    // caller MUST hold 'mutex'
    LegacyReductionState &legacy_state(void)
    {
      if(!legacy) {
        legacy.reset(new LegacyReductionState);
      }
      return *legacy;
    }
  };
}; // namespace Realm

#include "realm/barrier_impl.inl"

#endif // ifndef REALM_BARRIER_IMPL_H
