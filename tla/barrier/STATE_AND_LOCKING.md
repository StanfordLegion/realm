# BarrierImpl — State Layout and Locking Discipline

> ## §0. DECISIONS — these supersede anything below
>
> Reviewed and decided by the owner after this document was drafted. Where the
> body disagrees, **these win**; the body is retained as the supporting analysis.
>
> - **D1 — no the legacy reduction members object.** All state lives directly in
>   `BarrierImpl`. Reduction barriers still fall back to the existing eager
>   path, but that is a code path, not a separate object. No unnecessary
>   abstractions.
> - **D4 — one counter direction, one semantics.** Every accumulator is
>   cumulative and replace-if-higher, counting up. `BarrierArrive.tla` was changed to
>   match: a timestamped arrival now carries the node's *cumulative* bypassed
>   count and the owner keeps it per-node, exactly like `childAcc`. The `seq`
>   discriminator is gone.
> - **D8 — store the subscriber set directly as `MulticastTargetSet`.** It
>   already has `add`/`remove`/`contains`/`size`/`num_ranges`. No `NodeSet`
>   staging and no cached conversion: the set changes rarely by design, so
>   run-list mutation cost is irrelevant and a per-send conversion is not.
> - **Q8 — drop `BarrierCommunicator`.** No injectable transport seam; use the
>   active message interface directly.
> - **Q9 — do not touch `external_wait`.** Out of scope for both protocols.
> - **Q5 — notifications carry one triggered-generation number.** Barriers
>   trigger in order, so a wider range buys nothing; poison for a new subscriber
>   rides the subscribe reply. Do not widen `prev`.
> - **Q6 — do not hand-roll payload chunking.** Use active-message
>   fragmentation.
> - **Q1/D5 — `alter_arrival_count` is persistent**, and the current code is
>   **buggy**, not merely unimplemented. Fixing it is part of C6.
> - **Q2** destroy stays a no-op. **Q3** reuse `GenEventImpl`'s poison structure;
>   past the cap, fatal log then `std::abort`. **Q4** a poisoned precondition
>   poisons the generation. **Q7** exactly one mutex per barrier. **Q10** defer
>   tests.
> - **D2, D3, D6, D7, D10** as drafted. Note D3: migration goes, but the lock is
>   still needed in most paths it touched, to race correctly with notifications.


**Phase B of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §2. This is the hard
gate: every stage of Phase C is written against this document.**

Normative inputs, in priority order:

1. `BarrierArrive.tla`, `BarrierNotify.tla` — the model-checked artifacts. Where anything
   below disagrees with them, they win.
2. [`ARRIVAL_PROTOCOL.md`](ARRIVAL_PROTOCOL.md) (9 rules, §11 plan lifecycle, §12
   atomicity), [`NOTIFICATION_PROTOCOL.md`](NOTIFICATION_PROTOCOL.md) (8 rules).
3. This document, which fixes the C++ realisation: members, ownership, and which
   critical section each spec ACTION runs in.

Nothing here changes a protocol rule. Where this document makes a choice the specs
leave open, it says so and §9 records it as an open question rather than pretending it
is settled.

---

## 1. Decisions this document fixes

These are the choices the four scouts left open. They are stated here so no
implementation stage has to re-derive them.

| # | Decision |
|---|---|
| D1 | ~~Forked into a separate the legacy reduction members object.~~ **SUPERSEDED by §0: no separate object — these members live directly in `BarrierImpl` and are simply unused on the scalable path.** `remote_trigger_gens`, `remote_subscribe_gens`, `held_triggers`, `final_values`, `value_capacity` and `initial_value` move inside it and are unreachable from the scalable path. This answers the deletion-map scout's first open question: the map is not globally deleted and it is not shared — it is fenced. |
| D2 | **The fork point needs no advance knowledge on a remote node.** `redop_id` is immutable and known at the owner from `create_barrier`. A non-owner never has to guess: it aggregates only if it holds a plan record, and the owner only ever sends a `newplan` for a barrier with `redop_id == 0`. A node with no plan record sends its arrival straight to the owner — which is simultaneously the legacy behaviour and arrival rule 3. |
| D3 | **`owner` becomes write-once.** Migration is deleted, so `EventImpl::owner` is written only by `init()` and may be read without the mutex. Several lock acquisitions that exist only to read it consistently (`subscribe`, `add_waiter`, the `adjust_arrival` prologue) lose their reason to exist. |
| D4 | **The scalable arrival path counts UP.** `local_total`, `child_acc`, `ts_acc` are non-negative running totals compared against `expected(g)`. The legacy path keeps the existing count-down convention (`base_arrival_count + unguarded_delta == 0`). `Barrier::arrive(count)` passes `-int(count)` today; on the scalable path it contributes `+count`. Mixing the two conventions in one accumulator is the single easiest way to produce a barrier that triggers early. |
| D5 | **`alter_arrival_count` is persistent**, per `event.h:271-273` and `BarrierArrive:433-434`. The current implementation is *not* — it folds the delta into one generation's `unguarded_delta` and `base_arrival_count` never changes. This is a behaviour change; see Q1. |
| D6 | **`BarrierImpl` never calls `get_runtime()`.** Collaborators are injected through a `BarrierImplAllocator`, exactly as `GenEventImpl` already does (`event_impl.h:263-277`, `DynamicTableAllocator`'s `Constructor` hook at `dynamic_table.h:61,144`). This is the structural fix for the previous attempt's failure. |
| D7 | **Poison uses the `GenEventImpl` representation**: an append-only, never-reallocated `gen_t*` array plus an `atomic<int>` count (`event_impl.h:326-332`, `event_impl.cc:1396-1410`). That is what keeps `has_triggered` lock-free while `PoisonAccurate` holds. |
| D8 | ~~Lives as a `NodeSet`, converted and cached.~~ **SUPERSEDED by §0: store it directly as `MulticastTargetSet`** (it has `add`/`remove`/`contains`/`size`/`num_ranges`). No staging type, no cached conversion — the set changes rarely by design, so run-list mutation cost is irrelevant and a per-send conversion is not. `NodeSet` is inline for ≤4 nodes or ≤2 ranges and otherwise a pooled bitmask (`nodeset.h:145-172`); `MulticastTargetSet` stores 8 bytes per run, which is worse than a bitmask for a scattered set. The codec is the wire form and the cost-test oracle, not the storage form. |
| D9 | **The plan-construction aggregation structure is never a member.** It *is* the per-generation flush maps; the plan is computed from them inside the `Trigger` critical section, before that generation's record is freed. There is nothing to forget, so arrival §11.3 cannot be violated by omission. |
| D10 | **No barrier active-message handler declares `handle_inline`.** This matches today (`activemsg.inl:711-715` SFINAE fallback) and is now a rule, not an accident. |

---

## 2. The new `BarrierImpl` member layout

Members are grouped into tiers by *what synchronises access to them*. The tier is the
contract; the grouping in the header must make the tier obvious.

### 2.0 Tier summary

| Tier | Access rule |
|---|---|
| 0 — lock-free | Read with no lock. Written only inside `mutex`, published with release stores in the order of §3.5. |
| 1 — immutable | Written by `init()` / `create_barrier` before the object is reachable; read anywhere. |
| 2 — injected | Set once by the allocator; never `get_runtime()`. |
| 3 — mutex | Every read and every write inside `BarrierImpl::mutex`. Includes everything reached through the sub-state pointers. |
| 4 — external wait | `external_waiter_mutex` (INNER). Only `has_external_waiters` bridges tiers 3 and 4. |

### 2.1 Tier 0 — lock-free

```cpp
  // ---- TIER 0: readable without the mutex ---------------------------------
  // Written ONLY inside 'mutex'.  Publication order is fixed by §3.5:
  //   poison slots -> num_poisoned_generations (release) -> generation (release).
  atomic<gen_t> generation{0};                    // THE WATERMARK
  atomic<int>   num_poisoned_generations{0};
  gen_t        *poisoned_generations = nullptr;   // append-only, never realloc'd,
                                                  //  entries below the published
                                                  //  count are immutable
```

`gen_subscribed` is **deleted**. Its two jobs are taken over by tier-3 fields:
duplicate-pull suppression becomes `pull_outstanding` (notification rule 7, which is
about *outstanding-ness*, not about a generation), and "have I told the owner I care"
becomes `member`.

### 2.2 Tier 1 — immutable after init / create

```cpp
  // ---- TIER 1: immutable once the object is reachable ---------------------
  // ID     me;      (EventImpl)
  // NodeID owner;   (EventImpl) - WRITE-ONCE in init(); see D3
  gen_t                     first_generation = 0;  // retention floor
  ReductionOpID             redop_id = 0;          // 0 => scalable, !=0 => legacy
  const ReductionOpUntyped *redop = nullptr;       // non-null iff redop_id != 0

  // INVARIANT: a barrier ALWAYS has a non-zero expected arrival count
  //  (arrival rule 9), so 0 is an unambiguous "not yet known" sentinel on a
  //  node that has not yet been told.  This is the only thing that makes the
  //  sentinel safe - NOTIFICATION_PROTOCOL §3.
  unsigned                  base_arrival_count = 0;
```

`base_arrival_count` on a non-owner is filled in by the `newplan` payload
(arrival §11.4), **not** by a trigger message. The trigger-message field it rides on
today is dead (`barrier_impl.cc:363-368` is gated on `migration_target != -1`, which
`DISABLE_BARRIER_MIGRATION` makes unreachable) and is deleted with migration.

### 2.3 Tier 2 — injected collaborators

```cpp
  // ---- TIER 2: injected; NEVER get_runtime() ------------------------------
  EventTriggerNotifier                 *event_triggerer = nullptr;
  // Q8: BarrierCommunicator is DROPPED - send active messages directly.
  const ReductionOpTable               *reduce_op_table = nullptr;  // legacy only
```

with

```cpp
  struct BarrierImplAllocator {   // mirrors GenEventImpl::GenEventImplAllocator
    EventTriggerNotifier   *triggerer = nullptr;
    const ReductionOpTable *redop_table = nullptr;
    void construct(BarrierImpl *storage, ID id, unsigned owner) const;
  };
  typedef DynamicTableAllocator<BarrierImpl, 10, 4, BarrierImplAllocator>
      BarrierTableAllocator;
```

This deletes the `get_runtime()->get_module_config("core")` call in the default
constructor (`barrier_impl.cc:512-513`) and both `get_runtime()` calls on the deferred
arrival path (`barrier_impl.cc:582`, and the `reduce_op_table` lookups at `:422`,
`:1425`). See §5.

> **§0/Q8 SUPERSEDES the following paragraph: `BarrierCommunicator` is dropped
> entirely. Use the active message interface directly.**

`BarrierCommunicator` is retained as the single injectable transport seam, but with
**all** sends routed through it (today only 6 of 12 sites use it) and with the new
method set of §6. Whether it stays a per-object `unique_ptr` or becomes a per-node hook
is a cost question, not a correctness one — see Q8.

### 2.4 Tier 3 — under `BarrierImpl::mutex`

#### 2.4.1 Core, present on every node for every barrier

```cpp
  Mutex mutex;                       // THE lock.  UnfairMutex.  Non-reentrant.

  // open generations only; an entry is created on first need and DELETED when
  //  the generation triggers (or when this node learns it triggered)
  std::map<gen_t, Generation *> generations;

  // ---- notification, node side (needed on BOTH paths) --------------------
  // NB: this is THIS NODE's belief about ITS OWN membership.  It is one enum
  //  per barrier, not a map over nodes - the owner keeps no per-node map
  //  (NOTIFICATION_PROTOCOL §3).
  enum MemberState : uint8_t { MEMBER_NO = 0, MEMBER_PENDING = 1, MEMBER_YES = 2 };
  MemberState member          = MEMBER_NO;
  uint64_t    my_set_ver      = 0;     // highest setVer applied here (rule 2)
  bool        pull_outstanding = false; // rule 7: at most one outstanding pull
  gen_t       last_consult_wm = 0;     // rule 8 idle counter, in GENERATIONS
  gen_t       last_depart_wm  = 0;     // for the churn test that doubles K
  unsigned    depart_K        = 8;     // K, doubled on observed churn
  bool        depart_outstanding = false;

  // ---- scalable arrival: the plan record, on every node -------------------
  ArrivalPlan cur_plan;                // curPlan[n]
  uint32_t    my_epoch    = 0;         // myEpoch[n]
  uint32_t    inval_epoch = 0;         // invalEpoch[n]
  uint32_t    defer_epoch = 0;         // deferEpoch[n], 0 = none

  // At most ONE parked plan (BarrierArrive BoundedRetention).  This holds the
  //  encoded SUBTREE payload, not just an epoch, because the node must forward
  //  it to its own children when the invalidation lands.  COPIED out of the
  //  active-message buffer before the handler returns (§5 R4).
  std::vector<unsigned char> deferred_plan_payload;

  // ---- alteration state on the ISSUING node -------------------------------
  // myTs[n][g] as a step function: the greatest breakpoint <= g gives the
  //  timestamp this node's arrivals for g must carry.  Alterations are
  //  persistent, so this is a handful of entries, pruned at trigger.
  std::map<gen_t, Barrier::timestamp_t> ts_floor;

  // ---- external waiters (bridges to tier 4) -------------------------------
  bool has_external_waiters = false;

  // ---- owner-only / legacy-only sub-state ---------------------------------
  std::unique_ptr<OwnerState>            owner_state;  // non-null iff owner == me
  // D1: NO separate object. These live directly in BarrierImpl and are used
  //  only when redop_id != 0 (the legacy eager path).
  std::map<unsigned, gen_t> remote_trigger_gens, remote_subscribe_gens;
  std::map<gen_t, gen_t>    held_triggers;
  std::vector<char>         final_values;
```

#### 2.4.2 `ArrivalPlan` — `curPlan[n]`, O(fanout)

```cpp
  struct ArrivalPlan {
    uint32_t            quota  = 0;      // predicted local arrivals
    bool                inplan = false;  // membership rule: quota 0 => not in the
                                         //  tree at all (ARRIVAL §3)
    NodeID              parent = -1;     // -1 means "report direct to owner";
                                         //  stored, not derived (ARRIVAL §8.1)
    std::vector<NodeID> kids;            // O(radix)
  };
```

`ParentOf` in the model is a global search (`BarrierArrive:144-146`); the implementation
stores it. `inplan == false` ⇒ `parent` is unused and every arrival goes `direct` to
the owner.

#### 2.4.3 `Generation` — per open generation

```cpp
  class Generation {
  public:
    EventWaiter::EventWaiterList local_waiters;

    // ---- scalable arrival, every node ------------------------------------
    int64_t local_total = 0;   // localTotal[n][g]  (COUNTS UP, see D4)
    int64_t reported_up = 0;   // reportedUp[n][g]
    int64_t child_sum   = 0;   // running sum of child_acc[*].total
    bool    flushing    = false;

    struct ChildReport {
      int64_t                  total = 0;  // staleness key (rule 7)
      std::map<NodeID,int64_t> counts;     // per-node map; NON-EMPTY ONLY in
                                           //  flush mode (ARRIVAL §11.1)
    };
    // childAcc[n][c][g].  Keyed by sender, NOT by the current child list -
    //  ARRIVAL §8.2 requires accepting reports from nodes not listed as kids.
    std::map<NodeID, ChildReport> child_acc;

    // ---- owner only -------------------------------------------------------
    int64_t ts_acc = 0;                              // tsAcc[g]
    std::map<Barrier::timestamp_t,int64_t> ts_pending; // arrivals waiting on the
                                                       //  alteration they witnessed

    // ---- legacy (reduction) only -----------------------------------------
    int                            unguarded_delta = 0;
    std::map<int, PerNodeUpdates*> pernode;
  };
```

Derived, never stored:

```
  SubtreeKnown = local_total + child_sum
  Unreported   = SubtreeKnown - reported_up
  Holding      = Unreported > 0
  PlanSatisfied = cur_plan.inplan
                  && local_total == cur_plan.quota
                  && every k in cur_plan.kids has child_acc[k].total > 0
```

The flush report's `(node,count)` map is also derived on demand —
`{me: local_total} ∪ ⋃_c child_acc[c].counts` — so the merged map is never a member.
That is D9.

#### 2.4.4 `OwnerState` — owner only

```cpp
  struct OwnerState {
    // ---- notification (NOTIFICATION_PROTOCOL §3) -------------------------
    MulticastTargetSet sub_set;       // D8: the wire form IS the storage form
    uint64_t set_ver  = 0;            // bumped on EVERY change to sub_set
    MulticastTargetSet want_out;      // departure intents, usually empty

    // derived form for multicast + the rule-3 cost test; valid iff
    //  cached_ver == set_ver
    uint64_t           cached_ver = ~uint64_t(0);
    MulticastTargetSet cached_targets;

    // ---- arrival: the expected count, as a step function -----------------
    // expected(g) = expected_floor + sum of alter_steps entries with key <= g.
    // Alterations are PERSISTENT (D5), so this is base + accumulated deltas,
    //  NOT a per-generation delta.
    int64_t                 expected_floor = 0;   // == base_arrival_count at init
    std::map<gen_t,int64_t> alter_steps;          // breakpoints above the watermark

    // appliedTs, with the generation each alteration applies from, so entries
    //  can be pruned (§7.3).  Gate is EXACT membership, never "<= max" -
    //  two alterations from one node can be applied out of order.
    std::map<Barrier::timestamp_t, gen_t> applied_ts;

    // ---- arrival: plan epochs --------------------------------------------
    uint32_t next_epoch          = 1;
    bool     plan_rebuild_pending = false;   // set by any deviation signal
  };
```

`ownerAcc[g]` from the model is **not** a separate member: at the owner it is exactly
`generations[g]->child_sum`, and the model's
`ownerAcc' = @ - childAcc[Owner][from][g] + val` is the incremental maintenance of that
sum. Keeping one representation removes a whole class of divergence bug.

#### 2.4.5 Legacy reduction members — reduction barriers only

> **§0/D1: these are plain `BarrierImpl` members, not a separate object.** The
> text below describes which members they are; ignore the wrapper struct.

```cpp
  // D1: plain BarrierImpl members - NO wrapper object.  Used only when
  //  redop_id != 0; inert on the scalable path.
  std::map<unsigned, gen_t> remote_subscribe_gens;
  std::map<unsigned, gen_t> remote_trigger_gens;   // KEPT: slices final_values
                                                   //  per recipient
  std::map<gen_t, gen_t>    held_triggers;         // KEPT: legacy path only
  std::unique_ptr<char[]>   initial_value;
  unsigned                  value_capacity = 0;
  std::vector<char>         final_values;
```

Rationale for keeping `held_triggers` here rather than deleting it everywhere: C1's
mandate is "route reduction barriers to the legacy path", and the lowest-risk reading
of that is a pure relocation. The deletion-map scout is right that reduction barriers
could also move to discard-and-pull, but that is a change to a path this project is not
otherwise touching, and it would need its own tests.

### 2.5 Tier 4 — external wait

```cpp
  KernelMutex           external_waiter_mutex;    // INNER lock
  KernelMutex::CondVar  external_waiter_condvar;
```

Unchanged, including the hand-over-hand handoff at `barrier_impl.cc:1148-1152`. See
§3.6.

### 2.6 What is deleted

Everything below goes, on both paths, in stage C1.

| Deleted | Where | Why |
|---|---|---|
| `BarrierMigrationMessage` (struct, handler, registration) | `cc:296-319`, `:1573` | migration removed; unreachable today (needs `forwarded`, which needs a stale `owner`) |
| Migration election block + `#define DISABLE_BARRIER_MIGRATION` | `cc:28`, `:902-924` | already compiled out; its `redop == 0` guard shows it never served reduction barriers |
| `migration_target` — wire field, parameter, both receive blocks | `h:40,86,95,157,168`, `inl:87`, `cc:335,362-368,799,1011-1015,1474-1478` | always `-1` on the wire |
| `inform_migration` (both sites) and the `subscriber == my_node_id` self-loop guard | `cc:801,812-819,992-995,1275,1287-1294,1354-1359` | migration machinery |
| `bool forwarded` — 4 signatures, 3 structs, ~29 refs | `h:61,68,147,152`, `cc:188,263,559-601` | its only two readers were the `inform_migration` branches |
| `base_arrival_count` on the trigger message | `h:87` | its only consumer was gated on `migration_target != -1`; the count moves to `newplan` |
| `needs_ordering`, `ordered_buffer` | `h:222-223` | never read |
| `broadcast_radix`, `get_broadcast_targets`, `broadcast_trigger`, `BarrierTriggerPayload`, `BarrierTriggerMessageArgs{,Internal}`, `RemoteNotification`, their `serdez` | `h:35-54,74-77,164-171,224`, `inl:31,62-92`, `cc:659-725,1029-1036,1363-1397` | the `BARRIER_ENABLE_BROADCAST` half; superseded by `realm/multicast.h`, and it references `ActiveMessageAuto`/`AutoMessageRegistrar`, which do not exist in the tree |
| `BarrierImpl::handle_remote_trigger` | `cc:1405-1541` | the second, divergent, *unlocked* copy of the trigger receive path |
| `held_triggers` drain loops + `held_triggers[prev] = trigger_gen` on the scalable path | `cc:377-389,408-414` (and the dead twin at `:1487-1499,1520-1525`) | notification rule 4: discard and pull. The `previous_gen` **field and the gap test at `cc:375` stay** — they become rule 4's `gap == m.prev > known` |
| `remote_trigger_gens` / `remote_subscribe_gens` on the scalable path | `h:213` | replaced by `sub_set` + `set_ver`; the maps survive only inside the legacy reduction members |
| `gen_subscribed` | `h:175` | replaced by `member` + `pull_outstanding` |
| `#ifdef BARRIER_HAS_TRIGGERED_DOES_SUBSCRIBE` block | `cc:1055-1086` | defined nowhere; it is the exact anti-pattern §4 forbids |
| `-ll:barrier_radix` / `barrier_broadcast_radix` config | `runtime_impl.cc:763,803`, `runtime_impl.h:187` | **only if** `activemsg.cc:1691-1700` does not still consume it — the scout reports it does. Verify before removing the key; the *barrier's* use of it goes regardless |

Also deleted, though not a member: the four `TimeLimit::responsive()` call sites
(`cc:95,125,1239` and the deferred-arrival path). See §5 R2.

### 2.7 Memory bound

Arrival §11.3 requires O(fanout), not O(N). Per barrier, per node, with `R` = plan
radix, `G` = open (untriggered, locally-known) generations, `N` = nodes:

| Component | Steady state | Deviation episode | Freed by |
|---|---|---|---|
| tier 0 + tier 1 + tier 2 scalars | ~96 B | — | destroy |
| `poisoned_generations` | 0 B until a generation is actually poisoned, then 64 B (16 × `gen_t`) | — | destroy |
| `cur_plan` | O(R) — 24 B + 4R | — | plan switch |
| `deferred_plan_payload` | 0 | **O(subtree)** bytes, at most one | the invalidation that applies it |
| `generations` | G × (~120 B + O(R) child entries) | G × O(deviating senders) | the generation's trigger |
| `ts_floor` | O(1) — one live breakpoint | O(open alterations) | trigger (prune) |
| **owner** `sub_set` | 16 B inline for ≤4 subscribers or ≤2 ranges; otherwise a pooled bitmask, `N/8` B | — | destroy |
| **owner** `cached_targets` | O(ranges) | — | rebuilt on `set_ver` change |
| **owner** `want_out` | 16 B (normally empty) | O(departing) bits | applied at the next trigger |
| **owner** `alter_steps`, `applied_ts` | O(nodes that ever altered *this* barrier) | — | trigger (prune) |
| **owner** aggregation structure | **does not exist** (D9) | is the per-generation flush maps | the generation's trigger |
| legacy the legacy reduction members | O(N) maps + O(generations × sizeof_lhs) | — | destroy. Reduction barriers are explicitly not scalable |

Two honest statements:

- **The only O(N) term on the scalable path is the owner's subscriber set, and it is a
  bitmask, not a map.** Today the owner holds `remote_subscribe_gens` **and**
  `remote_trigger_gens`, two `std::map<unsigned,gen_t>` at roughly 48 B per node per
  barrier — about 96·N bytes. The replacement is N/8 bytes worst case and *zero
  additional allocation* whenever the subscriber set is ≤4 nodes or ≤2 contiguous
  ranges, which covers "a few waiters" and "everybody". That is the O(N²)-across-N
  barriers reduction notification §3 is asking for; it is a large constant-factor and
  representation win, not an asymptotic one for the pathological scattered case.
- **The deviation-episode terms are genuinely O(subtree)**, and both of them —
  `deferred_plan_payload` and the per-generation flush maps — are bounded by the ~1 MB
  `MessageBlock` ceiling on the receive side (`activemsg.cc:353`, an assert, and
  fragmentation does **not** raise it because reassembly happens first). Whether that
  ceiling can be reached at target scale is Q6.

---

## 3. The critical sections

### 3.1 Lock inventory and order

| Lock | Kind | Scope |
|---|---|---|
| `BarrierImpl::mutex` | `UnfairMutex`, non-reentrant | all of tier 3, including everything behind `owner_state` / `legacy` / `generations` |
| `BarrierImpl::external_waiter_mutex` | `KernelMutex` | the condvar handoff only |

**Order: `mutex` (OUTER) → `external_waiter_mutex` (INNER). Never the reverse.** The
three existing signal sites (`cc:449-454`, `:968-974`, `:1527-1533`) already obey this
and the broadcast must stay *inside* the outer section — it is what pairs with the
hand-over-hand in `external_wait`.

There is exactly one lock per barrier. Arrival §12 asks for each ACTION to be atomic
with respect to that barrier's state; a single per-barrier mutex satisfies it trivially.
Splitting per-generation state under a second lock would tear `Arrive`, which reads
`flushing`, the quota, the child check and `local_total` together — a §12 defect. See
Q7 for the arrival-rate question.

### 3.2 The five structural rules

**S1 — A critical section may perform a *sequence* of whole spec actions, never a
*fraction* of one.** Merging is always sound (it is two atomic actions with no
interleaving, which the spec permits); splitting is a defect even when both halves are
locked, because the model never exposes the intermediate state and no invariant
constrains it. This is what lets `Arrive` and the `Trigger` it completes share one
section.

**S2 — Compute under the lock, emit after it.** Never send an active message and never
invoke a waiter callback while holding `mutex`. Under the lock, append to a local list
of outbound actions; release; then send.

**S3 — The emit phase reads only locals.** Every outbound message is fully materialised
inside the section: target node, generation, values, and any payload bytes. The emit
phase must not read `first_generation`, `redop_id`, `base_arrival_count`, `cur_plan`, or
anything else off `this`. This is a live defect today at `cc:1022-1028` and
`cc:1364-1367`, and it is what makes `RecvInvalidate`'s "forward before forgetting" work
without effort: the target was resolved before the child list was replaced.

**S4 — Every helper declares its locking contract.** A helper that assumes the lock is
held says so in its name (`..._locked`) and asserts it with a `MutexChecker`
(`mutex.h:80-112`, already in the tree). `broadcast_trigger` — called once with the lock
and once without — is the anti-pattern being removed.

**S5 — Emit-after-unlock is admissible for every message in both protocols.** Two
threads can leave the lock in one order and reach the wire in the other; that is
indistinguishable from network reordering, which both models already explore because
`msgs` is a set. Per kind:

| Kind | Why reordering is harmless |
|---|---|
| `report`, `direct` | cumulative; receiver replaces and drops non-increasing (rule 7) |
| `flush` | idempotent per generation (rule 4) |
| `invalidate`, `newplan` | epoch-monotone; the deferral rule is *specifically* the verified resolution of these two racing (rule 5) |
| `alter` | `expected` is updated by addition (commutes); `applied_ts` is a set insert (commutes) |
| `tsdirect` | counted by addition; the gate is exact-set membership, so it needs no ordering with respect to its `alter` — parking it is the designed behaviour |
| `notify` | gap rule discards what it cannot apply; membership is version-gated, so an older shrink notice cannot resurrect membership (rule 2) |
| `subscribe`, `reply` | reply is a delta keyed on `lk`, merged not substituted (rule 5) |

The one thing that is *not* reorder-tolerant is publishing a shrink to the post-shrink
set. That is handled by snapshotting the **pre-shrink** target set inside the section
(§3.4, action T) — not by ordering.

### 3.3 Notation

`recorded` means "appended to a local outbound list, emitted after the unlock".
`drained` means "absorbed into a local `EventWaiter::EventWaiterList`, triggered after
the unlock". All actions below hold `BarrierImpl::mutex` for exactly one section unless
stated.

### 3.4 Action → critical section

#### A — `Arrive(n,g)` — `BarrierArrive:182`
Entry points: `Barrier::arrive`, `DeferredBarrierArrival::event_triggered`, subgraph
arrivals (`tests/subgraphs.cc:214-248` reaches `adjust_arrival` by a different route and
must land on the same routing decision).

One section:
1. `g <= generation` ⇒ **application error** (arriving on a triggered generation), log
   fatal. Distinct from a stale *message*, which is silently dropped (action R step 1).
   The current `assert(barrier_gen > generation.load())` at `cc:827` conflates the two.
2. Find or create `generations[g]`.
3. `ts = greatest ts_floor entry with key <= g` (0 if none).
4. **If `ts != 0`** (rule 8.1): the arrival **bypasses the tree**. Do *not* touch
   `local_total`. Record `tsdirect(owner, g, ts, count)`. If this node is the owner,
   perform action TS inline instead of recording a message.
5. **Else** `local_total += count`; `sub = local_total + child_sum`; then exactly one of:
   - `flushing` ⇒ record `report(cur_plan.parent, g, sub, flush_map)`; `reported_up = sub`
   - `!cur_plan.inplan` ⇒ record `direct(owner, g, sub, {me: local_total})` (rule 3)
   - `local_total > quota` ⇒ `flushing = true`; record `report(parent, g, sub, flush_map)`
     **and** `flush(k, g)` for every `k` in `cur_plan.kids`; `reported_up = sub` (rule 2)
   - `local_total == quota` **and** every `k` in `cur_plan.kids` has
     `child_acc[k].total > 0` ⇒ record `report(parent, g, sub)`; `reported_up = sub`
     (rule 1 — **both** halves; the child-wait is a correctness requirement)
   - otherwise ⇒ **silence**, no state change beyond `local_total`
6. If this node is the owner, evaluate action T in the same section.
7. If the watermark advanced, signal external waiters (inner lock, still inside).

Tear set honoured (§12): `local_total`, `flushing`, the quota and child check, and the
send decision are one section. `unissued` has no implementation counterpart — it is the
application's own arrival budget, and its guard is the API contract of
`event.h:290-292`, not runtime state.

After unlock: send; then `event_triggerer->trigger_event_waiters(drained, poisoned, work_until)`.

#### R — `RecvReport(m)` / `DropStale` — `BarrierArrive:231`, `:260`
Handler for `report` and `direct`. One section:
1. `m.gen <= generation` ⇒ **drop silently and return.** This is what makes freeing a
   triggered generation's record safe: without it, a late report would look
   "increasing" from an absent `child_acc` entry and be forwarded again.
2. Find or create `generations[m.gen]`.
3. `m.val <= child_acc[m.from].total` ⇒ **`DropStale`**: return with no state change.
   Never treat a report as an increment (rule 7 — accepting stale reports lets the
   owner's count go *down*, a caught mutation).
4. `child_sum += m.val - child_acc[m.from].total`; `child_acc[m.from].total = m.val`;
   if the message carried a `(node,count)` map, replace `child_acc[m.from].counts`
   wholesale (cumulative-replace applies to the map too).
   **Accept from any `m.from`, including nodes not in `cur_plan.kids`** (§8.2 — this is
   easy to get wrong because the defensive check looks worthwhile).
5. If `m.kind == direct` and this node is the owner and `!flushing` ⇒ `flushing = true`
   and record `flush(k, m.gen)` for every `k` in `cur_plan.kids` (rule 3).
6. `sub = local_total + child_sum`; forward iff
   `not owner && sub > reported_up && (flushing || PlanSatisfied)`; if so record the
   report and set `reported_up = sub`.
7. If owner, evaluate action T.

`DropStale` is not a separate critical section — it is step 3's early return. The model
makes it an action only because a set-of-messages model needs an explicit way to remove
an element.

#### F — `RecvFlush(m)` — `BarrierArrive:272`
One section:
1. Find **or create** `generations[m.gen]` — the record must exist even with no local
   arrivals yet, or a later arrival will not know it is in flush mode.
2. `flushing` already true ⇒ return (this is what terminates the fan-out).
3. `flushing = true`; record `flush(k, m.gen)` for every `k` in `cur_plan.kids`;
   if `Unreported > 0`, record `report(parent, m.gen, SubtreeKnown, flush_map)` and set
   `reported_up = SubtreeKnown`.

Flush is per generation. The next generation starts back in planned mode.

#### T — `Trigger` — `BarrierArrive:292` **and** `BarrierNotify:96`, owner only
**These are one critical section.** They are separate actions in two separate models but
they are the same real-world event, they share `watermark`, and the notification's
`prev` must be the pre-trigger watermark of the same contiguous chain. Splitting them
leaves a window in which the watermark has advanced but no notification describes it.

Guard, evaluated inside: `g == generation + 1` and
`child_sum + local_total + ts_acc == expected(g)`.

One section, in this order:
1. **Drain the contiguous run.** While the next generation's record satisfies the guard:
   drain its `local_waiters`; note whether it is poisoned; free the record. Record
   `old_wm` before the loop and `new_wm` after. Coalescing several triggers into one
   notification is explicitly supported (notification rule 8: the idle counter is
   "measured as watermark delta since last consultation, so it is robust to the owner
   coalescing several triggers into one notification").
2. **Publish poison, then the watermark** — in that order, §3.5.
3. **Prune** `alter_steps` (fold entries `<= new_wm` into `expected_floor`),
   `applied_ts` (§7.3), and `ts_floor`.
4. **Notification half.** Choose `R ⊆ want_out` by the cost test (rule 3: encode
   `sub_set` with and without `R`, compare `EncodedMulticastTargets::encoded_size` and
   the delivery count; decline shrinks that do not pay — dropping scattered nodes from
   `ALL_NODES` turns a 0-byte encoding into a per-hop bitmap). Then:
   - **snapshot the PRE-shrink `sub_set`** into the local outbound target set — rule 1,
     "any shrink must be published to the PRE-shrink set";
   - `sub_set -= R`; `set_ver += (R ∩ sub_set_pre ≠ ∅)`; `want_out -= R`;
     invalidate `cached_ver` if the set changed;
   - compose the notify header `(wm = new_wm, prev = old_wm, sv = set_ver)` plus poison
     in `(old_wm, new_wm]`, and — **only if `sv` changed** — the encoded `R` as payload.
5. **Arrival half.** If a plan switch is due at `new_wm + 1`:
   - build the new plan from the just-drained generation's flush maps (`std::map::swap`
     them into a local first — O(1) — then build). This is D9: there is no persistent
     aggregation structure to forget;
   - record `invalidate(k, my_epoch)` for every `k` in the **current** `cur_plan.kids`;
   - install the owner's own new record; `my_epoch = k`; `inval_epoch = k - 1`;
   - record `newplan(k, subtree payload)` for every `k` in the new plan's kids.
   **At most one plan switch per section.** The owner chooses `PlanStart`, so it can
   simply never schedule two switches inside one drain; this keeps the implementation a
   faithful refinement of the model's one-switch-per-`Trigger`.
6. Signal external waiters (inner lock).

After unlock: `multicast_message` the notify to the snapshot target set; unicast the
invalidate/newplan messages; trigger drained waiters.

Tear set honoured: "count comparison and the plan switch that follows it".

Note on `inset`: the model gives each recipient its own `inset` field, but a multicast
delivers identical bytes to everyone. The resolution is that the notify carries **`R`**,
and each recipient computes `inset = (my_node_id ∉ R)`. This is exact, because the
message goes only to the pre-shrink set — every recipient is in it, so membership is
precisely "not being removed". When `R = {}` the `sv` is unchanged, `newv` is false at
every recipient, and no membership bytes are sent at all.

#### I — `RecvInvalidate(m)` — `BarrierArrive:320`
One section. **Order is load-bearing and both halves are separately verified.**
1. `inval_epoch >= m.epoch` ⇒ return.
2. **Forward first.** Record `invalidate(k, m.epoch)` for every `k` in the **current**
   `cur_plan.kids`, *before* anything replaces that list. Dropping the child list first
   strands the whole subtree — a caught mutation.
3. **Flush every open generation, not just the switch generation.** For every record in
   `generations` with `gen > generation`: set `flushing = true`; if `Unreported > 0`
   record `report(cur_plan.parent, gen, SubtreeKnown, flush_map)` and set `reported_up`.
   The target is `cur_plan.parent` as it is *now* — the old parent — which S3 gives for
   free because the target is resolved into the recorded message before step 5 replaces
   the plan. Flushing only the switch generation is a caught mutation.
4. `inval_epoch = m.epoch`.
5. If `defer_epoch != 0`: decode and install the parked plan; `my_epoch = defer_epoch`;
   record `newplan` to the parked plan's kids; `defer_epoch = 0`; free
   `deferred_plan_payload`.

#### P — `RecvNewPlan(m)` — `BarrierArrive:358`
One section:
1. `my_epoch >= m.epoch` ⇒ drop.
2. **`cur_plan.inplan && inval_epoch < my_epoch` ⇒ PARK.** `defer_epoch = m.epoch`;
   **copy** the subtree payload into `deferred_plan_payload` (the AM buffer is dead
   after the handler returns). At most one parked plan — a second one replaces it, which
   is safe because epochs are monotone. This is the deferral whose removal TLC catches
   in ~46 s as `ReachableWhileHolding`.
3. Otherwise install: `cur_plan = my record from the payload`; `my_epoch = m.epoch`;
   record `newplan` with the appropriate sub-subtree for each of the new kids; and for
   every open generation with `Unreported > 0`, record a report and set `reported_up`.
   **Do not set `flushing` here** — the model reports held work but leaves the flag
   alone (`BarrierArrive:367-377`), because the new plan is authoritative and the node can
   aggregate under it.

#### AL — `Alter(delta)` — `BarrierArrive:391`, on the issuing node
`Barrier::alter_arrival_count`. One section — the timestamp is minted **inside** it.
Today it is minted at `cc:91` and consumed at `cc:839` in a different section.
1. Mint `ts = barrier_adjustment_timestamp.fetch_add(1)`. The counter is seeded at
   `runtime_impl.cc:1962-1965` with `my_node_id << 48`, so timestamps are globally
   unique and **monotone per node**.
2. Reserved-arrival guard: the caller must still hold an unissued arrival from the
   pre-alteration count (`event.h:290-292`). The runtime cannot check this; it is the
   API contract, and removing the model's guard is caught by `TriggerCorrect`.
3. `ts_floor[g] = ts` — every subsequent arrival for `g' >= g` now carries `ts` and
   bypasses the tree.
4. **Enter eager flush for every affected open generation** (`g' >= g`, untriggered):
   `flushing = true`; record `flush` to `cur_plan.kids`; if `Unreported > 0` record a
   report. Skipping this is a caught mutation (deadlock): a node whose arrivals bypass
   the tree can never reach its own quota, so as a *relay* it would go silent and strand
   its children.
5. Record `alter(owner, g, delta, ts)`; if this node is the owner, run action RA inline.
6. If `delta < 0` (rule 9): the alteration also invalidates the plan — set
   `plan_rebuild_pending` at the owner (or let the owner set it on receipt).

Tear set honoured: "reserved-arrival guard, `unissued`, `myTs`, flush state, sends".

#### RA — `RecvAlter(m)` — `BarrierArrive:431`, owner only
One section:
1. `m.gen <= generation` ⇒ **contract violation** (`event.h:305-306` names the symptom
   as "a barrier that triggered too early"). Log fatal; still apply persistently for
   generations above the watermark so later counts stay consistent.
2. `alter_steps[m.gen] += m.delta`. If `expected(g)` would become `<= 0` for any open
   generation ⇒ fatal (rule 9: a zero base arrival count is an error, and it is what
   makes 0 a safe sentinel).
3. `applied_ts[m.ts] = m.gen`.
4. **Drain the gate**: for every open generation, move any `ts_pending[t]` with
   `t ∈ applied_ts` into `ts_acc`. Steps 3 and 4 must be in the same section — the model
   expresses this as `RecvTsDirect` becoming *enabled*, and fusing an enabling action
   with the action it enables is a legal refinement.
5. If `m.delta < 0`: `plan_rebuild_pending = true`; enter flush for the affected open
   generations and record the fan-out.
6. Evaluate action T.

#### TS — `RecvTsDirect(m)` — `BarrierArrive:447`, owner only
One section:
1. `m.gen <= generation` ⇒ drop.
2. `m.ts ∈ applied_ts` ⇒ `ts_acc += m.count`; evaluate action T.
3. Otherwise `ts_pending[m.ts] += m.count` — parked until its alteration lands.

**The wire form carries a count, not a `seq`.** `seq` (`BarrierArrive:192`) exists only to
keep two identical arrivals from collapsing into one element of a message *set*
(§8.4); putting it on the wire would be implementing a modelling artifact.

The gate is **exact set membership**, never `ts <= max_applied_ts_for_node`. Two
alterations from one node can be applied out of order (messages reorder), and a
"max" gate would admit an arrival whose alteration has not yet been applied — a safety
failure. The existing code's `pn->last_ts` comparison at `cc:647` is precisely that
shape and carries its own `TODO: really need two timestamps` comment.

#### C — `Consult(n,g)` — `BarrierNotify:116`
Entry points: `add_waiter`, `subscribe`, `external_wait`, `external_timedwait`.
**Explicitly NOT `has_triggered`** — see §4.

One section:
1. `g <= generation` ⇒ record `trigger_now` (and the poison answer); skip to 3.
2. Find or create `generations[g]`; push the waiter (for `add_waiter`).
3. `last_consult_wm = generation` — the idle counter reset. In the model this is
   `wantOut' = wantOut \ {n}`; the node-side realisation is resetting its own counter,
   and the owner-side realisation is that any `subscribe` removes the node from
   `want_out` (action S step 1).
4. If `owner != my_node_id && member == MEMBER_NO && !pull_outstanding`:
   record `subscribe(owner, lk = generation)`; `pull_outstanding = true`;
   `member = MEMBER_PENDING`.

After unlock: send; if `trigger_now`, call `waiter->event_triggered(poisoned, work_until)`
with the **caller's** budget, never a manufactured one (§5 R2).

`subscribe` carries **both** `lk` (what I have) and `subscribe_gen` (what I need).
`lk` is what replaces the pull-side `remote_trigger_gens` lookup (notification rule 5);
`subscribe_gen` is still needed by the legacy path's one-shot `remote_subscribe_gens`
erase, and it is 4 bytes.

#### N — `RecvNotify(m)` — `BarrierNotify:136`, scalable path only
One section, in this order:
1. `newv = m.sv > my_set_ver`. If `newv`: `member = (my_node_id ∈ decoded R) ? MEMBER_NO
   : MEMBER_YES`; `my_set_ver = m.sv`. **Membership is applied even on a gap** — this
   message may be the node's only notice of its own removal (`BarrierNotify:148-151`).
2. `gap = m.prev > generation`; `fresh = !gap && m.wm > generation`.
3. If `fresh`: append the poison entries in `(m.prev, m.wm]` (deduplicated) and publish;
   then `generation.store_release(m.wm)` (§3.5); drain every `generations` record with
   `gen <= m.wm` and free it; signal external waiters.
   If `gap`: **discard the delta entirely.** Do not buffer it. `held_triggers` is not
   coming back — pulling is strictly simpler because the pull path has to exist anyway.
4. `resub = (member == MEMBER_NO) && waiters remain`; if so `member = MEMBER_PENDING`.
5. `pull = (gap || resub) && !pull_outstanding` ⇒ record
   `subscribe(owner, lk = generation)`; `pull_outstanding = true`.
6. Departure hysteresis (rule 8), evaluated here because this is where the watermark
   advances: if `member == MEMBER_YES && no waiters && !depart_outstanding &&
   (generation - last_consult_wm) >= depart_K + (my_node_id % J)` ⇒ record
   `depart(owner)`; `depart_outstanding = true`; `last_depart_wm = generation`.
   **The stagger is expressed in generations, not in time**, so it needs no timer and
   the protocol stays feed-forward.

"Waiters remain" is `any record in generations has a non-empty local_waiters, or
has_external_waiters` — the model's `waiting[n] # {}`. No new member is needed.

#### S — `RecvSubscribe(m)` — `BarrierNotify:175`, owner only
One section:
1. `sv2 = (m.from ∈ sub_set) ? set_ver : set_ver + 1`; `sub_set.add(m.from)`
   — **unconditionally; adds are mandatory** (rule 3: refusing an add strands a waiter);
   `set_ver = sv2`; `want_out.remove(m.from)`; invalidate `cached_ver` if the set changed.
2. Compose the reply: `wm = generation`, `pois = poison ∩ (m.lk, generation]`,
   `sv = sv2`. **The reply must carry the watermark** — that is what covers the
   trigger-during-subscribe race, and omitting it is caught as a deadlock.
3. If `redop_id != 0`: instead compose the **legacy** trigger message (`final_values`
   slice derived from `m.lk`) and maintain `legacy->remote_subscribe_gens` /
   `legacy->remote_trigger_gens` exactly as today. One subscribe message type, two reply
   shapes, forked at the owner — which is why a remote node never needs to know in
   advance whether the barrier is a reduction barrier (D2).

The set mutation and the `sv` stamping are in one section because rule 2 requires it.

#### RP — `RecvReply(m)` — `BarrierNotify:187`
One section:
1. `pull_outstanding = false`.
2. `newv = m.sv > my_set_ver` ⇒ `member = MEMBER_YES`; `my_set_ver = m.sv`. The same
   version gate is required here — a stale reply resurrects membership exactly the way a
   stale notify does.
3. `fresh = m.wm > generation` ⇒ **union** `m.pois` into the poison array (dedup),
   publish, then `generation.store_release(m.wm)`; drain and free satisfied records;
   signal external waiters.
   **Union, never substitute** — substituting drops poison the node already knew about
   below `lk`. Caught by `PoisonAccurate`.
4. If `!fresh`, do not merge poison. Safe because `PoisonAccurate` says the node's
   knowledge is already exactly the truth up to its own (higher) watermark.

#### D — `Depart(n)` — `BarrierNotify:210`
Not a message receipt and not a timer. The eligibility test lives in action N step 6
(and in RP, at the same point) because that is where the watermark moves. The unicast is
emitted after the unlock.

Owner side: the `depart` handler takes one section and does `want_out.add(m.from)`.
It is applied — or declined — at the next `Trigger` (action T step 4).

`depart_outstanding` is cleared when the node next consults (action C) or when it is
removed (action N step 1 sets `member = MEMBER_NO`). Churn adaptation: if the node
re-subscribes with `generation - last_depart_wm` below a small window, `depart_K *= 2`
(capped). Both are node-local and cost 8 bytes.

### 3.5 Publication order for the lock-free tier

Every writer, inside `mutex`:

```
  for each newly poisoned generation g in (old_wm, new_wm]:
      poisoned_generations[n++] = g;          // plain store, slot not yet published
  num_poisoned_generations.store_release(n);  // publishes the slots
  generation.store_release(new_wm);           // publishes the watermark
```

Reader (`has_triggered`), no lock:

```
  gen_t wm = generation.load_acquire();
  if (needed_gen > wm) { poisoned = false; return false; }
  poisoned = is_generation_poisoned(needed_gen);   // acquire-load the count, scan
  return true;
```

Why this is sufficient: the reader's acquire load of `generation` synchronises with the
writer's release store, so every poison slot written before that store is visible.
Slots are **append-only and never rewritten**, so a concurrent later append can only
extend the array beyond the count the reader observes, and the reader's own acquire load
of the count bounds its scan to slots whose writes happen-before that count's release
store. This is exactly `GenEventImpl::is_generation_poisoned`
(`event_impl.cc:1396-1410`), which is already lock-free in production.

### 3.6 External wait

`external_wait` / `external_timedwait` keep the hand-over-hand handoff verbatim
(`cc:1148-1152`): `external_waiter_mutex.lock(); mutex.unlock(); condvar.wait();
external_waiter_mutex.unlock(); mutex.lock();`. That is what makes the signal
lost-wakeup-free — the signaller cannot enter `external_waiter_mutex` until the waiter
is already inside `wait()`, because until then the waiter still holds the outer mutex.

Do not "simplify" this by moving `external_wait` off `mutex` on the strength of
`generation` being atomic. It would shrink the `Trigger` section (no nested
`KernelMutex` acquisition), but the lost-wakeup argument has to be re-derived from
scratch together with all three signal sites. Q9.

### 3.7 The §12 tear table, honoured

| Action (§12) | Reads/writes that must not tear | Section |
|---|---|---|
| `Arrive` | `unissued`, `localTotal`, `flushing`, quota + child check, then the send | A, steps 2–5 (the send is *recorded* in the section) |
| `RecvReport` | staleness test, `childAcc`, recomputed subtree total, forward decision, `reportedUp` | R, steps 3–6 |
| `RecvInvalidate` | child list read **before** it is replaced, every open generation flushed, parked plan applied | I, steps 2–5, in that order; S3 makes step 2's target survive step 5 |
| `Trigger` | count comparison and the plan switch that follows it | T, one section, together with the notification `Trigger` |
| `Alter` | reserved-arrival guard, `unissued`, `myTs`, flush state, sends | AL, including the timestamp mint |

---

## 4. The `has_triggered` carve-out

`BarrierImpl::has_triggered` is already what §12 demands (`cc:1046-1054`) and must stay
that way.

```cpp
  bool BarrierImpl::has_triggered(gen_t needed_gen, bool &poisoned)
  {
    // TIER 0 ONLY.  No lock.  No allocation.  No message.  No member write.
    gen_t wm = generation.load_acquire();
    if(needed_gen > wm) {
      poisoned = false;
      return false;
    }
    poisoned = is_generation_poisoned(needed_gen);
    return true;
  }
```

Cost: one acquire load, plus — only when the generation has triggered — one more
acquire load and a scan bounded by the poison limit. **The poison machinery costs
nothing at all until a barrier is actually poisoned**, because
`num_poisoned_generations` is 0 and the early-out in `is_generation_poisoned` returns
immediately, exactly as `GenEventImpl` does today.

Three prohibitions, all of which the current file demonstrates the temptation for:

1. **No subscribe.** Delete the `#ifdef BARRIER_HAS_TRIGGERED_DOES_SUBSCRIBE` block
   (`cc:1055-1086`). It is defined nowhere in the tree and it is a ready-made regression:
   it takes `mutex`, mutates `gen_subscribed`, and sends.
2. **No consultation signal.** Notification rule 8 defines *consulting* as `add_waiter`,
   `subscribe`, `external_wait`/`external_timedwait`, and **explicitly not**
   `has_triggered`. `last_consult_wm` is written **only** in action C, which already
   holds the lock for other reasons. Nothing on the `has_triggered` path may touch it.
3. **No member write of any kind**, including a relaxed store. The reason to be
   absolute: the moment one relaxed store is permitted, the natural next step is a
   second, then a compare-exchange, then a `trylock`.

**How the prohibition is enforced, not merely stated:**

- `last_consult_wm`, `member`, `pull_outstanding` and `depart_*` are tier-3 members, and
  tier 3 is defined as "every access inside `mutex`". A `MutexChecker` (`mutex.h:80-112`)
  asserting the barrier lock is held can be placed in the `..._locked` helpers that own
  them, which makes a stray write from `has_triggered` fail in a debug build rather than
  silently.
- Plan §4 risk 4 ("`has_triggered` regressing to take a lock") gets an explicit test:
  Phase F asserts that `has_triggered` on a fresh `BarrierImpl` built with a mock
  communicator sends **zero** messages and leaves `member`, `pull_outstanding` and
  `last_consult_wm` unchanged across a large number of calls, including after a trigger.

**The consequence, stated plainly:** a node that only ever polls `has_triggered` never
subscribes, so it is never in `sub_set` and is never notified. That is already true
today. What is *new* is that a node which subscribed once and then only polls will be
departed after `K + (id % J)` idle generations and will then poll a frozen watermark
forever. `benchmarks/task_throughput/task_throughput.cc:92-101` contains exactly that
loop (behind `TestConfig::use_posttriger_barrier`, default false). See Q4 — this needs a
decision, and it is the one place where the carve-out and rule 8 genuinely pull against
each other.

---

## 5. The deferred-handler hazard

The previous attempt did not fail on protocol logic. It failed because barrier code
called

```cpp
  get_runtime()->event_triggerer.trigger_event_waiters(list, POISON_FIXME, work_until);
```

with a manufactured 10 µs `TimeLimit::responsive()`. `trigger_event_waiters`
(`event_impl.cc:456-511`) drains through `static thread_local` lists and never touches
`this` — **until** the budget expires with waiters still queued, at which point
`event_impl.cc:497-507` locks `this->mutex` and `make_active()` reads `this->manager`
(`bgwork.cc:386`). With a null `runtime_singleton` both are null-derived. The bug is
invisible with one waiter, invisible when the callbacks are fast, and fatal under load.

Six rules. R1 is the structural one; the rest are defence in depth.

**R1 — `BarrierImpl` must never call `get_runtime()`.** Every collaborator is injected
(§2.3) via `BarrierImplAllocator`, the same mechanism `GenEventImpl` already uses. After
this change the failing expression is `event_triggerer->trigger_event_waiters(...)` on an
injected, non-null pointer, and the null-runtime path *does not exist* — including in
`realm_unit_tests`, where `runtime_singleton` is null for the whole binary. Grep target:
zero occurrences of `get_runtime` in `barrier_impl.cc`.

**R2 — Propagate `TimeLimit`; never manufacture one.** `TimeLimit::responsive()` must
not appear anywhere in barrier code. Handlers propagate the `work_until` they were given.
Public API entry points that have no inbound budget (`Barrier::arrive`,
`Barrier::alter_arrival_count`, the immediate-trigger path in `add_waiter`) pass a
default-constructed `TimeLimit()`, which is infinite (`timers.inl:289-292`). An infinite
budget means `trigger_event_waiters` never takes the deferral branch and never touches
`this` at all — the hazard is removed at the source rather than survived. An application
thread that completes a generation has no bgwork quantum to respect; shortening its
budget buys nothing and costs the deferral.

**R3 — No `handle_inline` on any barrier handler.** (D10.) The inline contract
(`activemsg.h:326-335`) forbids blocking on a mutex and forbids allocation; every barrier
action takes `mutex` and touches maps. Multicast-delivered notifications arrive through
`MulticastForwarder::deliver_local` → `IncomingMessageManager::add_incoming_message`
(`activemsg.cc:1358-1360`), so a barrier handler that declared `handle_inline` would run
*inside* the envelope handler with a 5 µs budget. If an inline fast path is ever wanted,
it must be `mutex.trylock()` + fixed-size state only, returning false on any miss — and
it must be justified with measurements, not added opportunistically.

**R4 — A deferred continuation captures what it needs at creation.**
`DeferredBarrierArrival` stores the `BarrierImpl*` (and, if needed, the injected
notifier), not a `Barrier` handle to be re-resolved through `get_runtime()`
(`cc:582`). It must also copy any payload out of the active-message buffer before the
handler returns — the pointer handed to a handler is valid only for that call
(`activemsg.cc:353`, `:404-438`). Same obligation for `deferred_plan_payload` (§3.4
action P step 2) and for any parked report.

**R5 — Correct with an already-expired budget, and with no budget.** A queued handler
runs either on a bgwork worker with the *shared* 100 µs quantum
(`activemsg.cc:765-767`, `bgwork.cc:570`, `bgwork.h:48`) — so it routinely starts with
microseconds left, and `TimeLimit::is_expired()` also returns true whenever the worker's
`interrupt_flag` flips asynchronously — or on a dedicated handler thread with an
infinite `TimeLimit()` (`activemsg.cc:874`). **Protocol progress must never be
conditional on time remaining.** No loop may assume it gets one iteration. Every
critical section runs to completion; only the post-unlock waiter drain consults the
budget, and only because `trigger_event_waiters` does.

**R6 — S2/S3 apply to the deferral seam too.** The post-unlock phase reads only locals.
That is what makes it safe for the drained waiter list to outlive the section, and it is
what stops a waiter callback that re-enters the barrier from observing half an action.

---

## 6. Consequences for the wire (summary only)

Recorded here because the state layout depends on them; the message catalogue itself is
C-stage work.

| Message | Carries | Notes |
|---|---|---|
| `report` / `direct` | `gen`, cumulative `val`; **plus** a `(node,count)` map when the sender is flushing | header stays < 128 B (GASNet-EX `INLINE_SIZE`, `gasnetex_module.cc:159-186`); the map is payload. Use `EncodedMulticastTargets` for the node set plus a parallel varint count array, or an interleaved `(delta_node,count)` varint stream — Q6 |
| `flush` | `gen` | fixed size |
| `invalidate` | `epoch` | fixed size |
| `newplan` | `epoch`, `base_arrival_count`, and the recipient's **subtree** (its own quota/kids plus its descendants') | must carry the subtree, not just the node's own record, because the recipient forwards it. This is why `deferred_plan_payload` is bytes, not a struct |
| `alter` | `gen`, `delta`, `ts` | fixed size |
| `tsdirect` | `gen`, `ts`, `count` | **no `seq`** (§8.4) |
| `notify` | `wm`, `prev`, `sv`, poison in `(prev,wm]`; payload = encoded `R` **only when `sv` changed** | sent via `multicast_message` (`activemsg.inl:796-806`), fire-and-forget, origin-as-sender |
| `subscribe` | `lk`, `subscribe_gen` | one type for both paths; the owner forks on `redop_id` |
| `reply` | `wm`, `sv`, poison in `(lk,wm]` | point-to-point, so it can be keyed exactly to `lk` |
| `depart` | — | unicast to owner, staggered in generations |

Message structs must move out of the anonymous namespace in `barrier_impl.cc` into a
shared header if the implementation is split across translation units; internal linkage
would otherwise pin every send site to one `.cc`.

`prev` is the owner's pre-trigger watermark, exactly as the model has it. A tempting
robustness extension — send `prev = max(first_generation, wm - W)` for a small window
`W` with `pois` widened to match, so one reordered notification does not cost every
subscriber a pull — appears sound (the receiver's gap test only becomes *more*
permissive, and the wider `pois` keeps the union exact). It is **not** adopted here,
because it is not what was model-checked. Q5.

---

## 7. Reset, reclaim, and the retention contract

Arrival §8.8 and notification §8.5 both make this an explicit obligation, and
`BarrierImpl::init` currently discharges none of it: it clears `remote_subscribe_gens`
and `remote_trigger_gens` but not `generations` or `held_triggers`, never resets
`redop_id`, `has_external_waiters`, `first_generation` or `value_capacity`, and
`~BarrierImpl` is empty so every `Generation` in the map leaks.

### 7.1 `init(ID, owner)` — full reset

Must leave the object indistinguishable from a fresh one. Every tier-3 member is reset;
every `Generation*` in `generations` is deleted; `owner_state` and `legacy` are reset to
null; `poisoned_generations` is freed and `num_poisoned_generations` stored to 0;
`generation` and `first_generation` are stored last. `init` runs before the object is
reachable by any other thread, so it needs no lock — but that is a *precondition to be
asserted*, not an assumption to leave implicit.

### 7.2 Per-generation reclamation

A `Generation` record is freed:
- at the owner, in action T, as its generation is drained;
- at a non-owner, in action N/RP, for every record at or below the new watermark.

Once freed, any late `report`/`direct`/`tsdirect` for that generation is dropped by the
`gen <= generation` test that opens actions R and TS. Those two tests are what make
reclamation safe; they are not optional.

### 7.3 Pruning the alteration state (owner)

At each trigger:
- fold `alter_steps` entries with key `<= watermark` into `expected_floor`;
- drop `applied_ts` entry `(ts, g)` for node `n` once there is a later entry `(ts2, g2)`
  from the same node with `watermark >= g2 - 1` — at that point every generation below
  `g2` has triggered, so no future arrival from `n` can carry `ts`
  (arrivals for generation `h` carry the timestamp of `n`'s latest alteration with
  `gen <= h`);
- on the issuing node, drop `ts_floor` breakpoints below the greatest one `<= watermark + 1`.

Steady state is O(1) per altering node. The residual is one `applied_ts` entry per node
that has *ever* altered this barrier — O(N) for a barrier every node alters. Alterations
are rare by assumption; if that assumption breaks, this is where it shows up.

### 7.4 Poison retention

The owner retains the poisoned set over `[first_generation, watermark]` because rule 5
requires answering a new subscriber with everything above its `lk`. With the
fixed-capacity array of D7 that is a hard cap on how many generations of one barrier may
be poisoned. Q3.

### 7.5 `destroy_barrier`

`Barrier::destroy_barrier` is a bare log statement today (`cc:63-66`) and there is no
`free_entry` to match `alloc_entry` (`cc:139`), so barriers are never recycled and
`first_generation` never advances. Notification §8.5's bound on poison retention rests
on `first_generation` advancing when an id is reused — a mechanism that does not exist.
Q2 asks whether to implement it or to state explicitly that retention is bounded by
process lifetime. Whichever is chosen, **the retention floor is `first_generation`** and
every retention rule above is expressed against it.

---

## 8. What this buys, versus today

| | Today | After |
|---|---|---|
| Owner state per barrier | 2 × `std::map<unsigned,gen_t>` ≈ 96·N bytes | `NodeSet` (16 B inline, else N/8 B) + `set_ver` |
| Trigger notification | one unicast per subscriber, keyed by a per-node map | one multicast to the encoded set; the delta is uniform |
| Out-of-order notification | buffered in `held_triggers` indefinitely | discarded; one pull; `previous_gen` survives as the gap detector |
| `owner` | mutable, read unlocked at `cc:744` (a real race) | write-once |
| Trigger receive path | two divergent copies, one of them unlocked | one |
| Sends under the lock | `broadcast_trigger` at `cc:1031-1035` | forbidden by S2, asserted by S4 |
| `get_runtime()` in barrier code | 12 call sites, one on a deferred path | zero |
| `TimeLimit::responsive()` | 4 manufactured budgets | zero |

---

## 9. Open questions

These are the things I could not settle from the code and the two protocol documents.
Each one changes either the layout above or an externally visible behaviour, so each one
needs an answer before the stage that depends on it.

**Q1 — Is `alter_arrival_count` persistent, and does Legion already depend on it not
being?** `event.h:271-273` says the change applies to this generation *and every
subsequent one*; `BarrierArrive:433-434` models it that way; `TriggerCorrect` is checked
against it. The current implementation is **not** persistent — the delta lands in one
generation's `unguarded_delta` and `base_arrival_count` never changes. If Legion's
`PhaseBarrier` / dynamic-collective code calls `alter_arrival_count` once per generation
(which is the natural thing to do against the current behaviour), making it persistent
double-counts and the barrier hangs. This is the single highest-risk item in the whole
plan and it is not visible in this repository: there is **zero** coverage of
`alter_arrival_count` anywhere in `tests/`, `tutorials/`, `examples/` or `benchmarks/`.
Blocks C6. *Needs: an answer from the Legion side, or a Legion-integration run that
actually exercises the API.*

**Q2 — Is destroy/reuse in scope?** `destroy_barrier` is a no-op, there is no
`free_entry`, and `first_generation` never advances. Notification §8.5 bounds poison
retention on that advance; arrival §8.8 makes `destroy_barrier` the cleanup point. Either
implement destroy + id reuse (which also decides what a use-after-destroy does — today
`has_triggered` on a stale handle silently answers from the recycled barrier's watermark),
or state that retention is bounded by process lifetime and that the §8.5 bound is
currently unenforced. Also: `tests/barrier_reduce.cc:289` destroys the *advanced* handle
rather than the created one, which will start to matter.

**Q3 — Poison capacity and the overflow policy.** D7 adopts `GenEventImpl`'s fixed
array. `POISONED_GENERATION_LIMIT` is 16 (`event_impl.h:325`) and `GenEventImpl` handles
overflow by refusing to recycle the id (`event_impl.cc:1768-1769`). A barrier has up to
2²⁰ generations, so 16 is a real cap on how many of them may be poisoned in one
barrier's life. Options: (a) same limit, same "retire the id" policy — cheapest, and
barriers are never recycled today anyway; (b) a larger fixed limit; (c) the §8.5 escape
hatch — cap what a *notification* carries and let nodes pull for detail, which needs the
owner to retain more than the reader array does. Blocks C7.

**Q4 — Can a barrier generation be poisoned at all, and by what?** The only plausible
source is `arrive(wait_on = <poisoned event>)`, which today hits
`assert(poisoned == POISON_FIXME)` at `cc:579` and aborts. Three possible semantics:
propagate (the generation is poisoned and every waiter is woken poisoned), count the
arrival and ignore the poison, or keep the fatal error. Notification rules 4 and 5 and
`PoisonAccurate` are being implemented for behaviour that has never existed, and there is
no baseline to preserve — nothing in the tree would notice any of the three. Related, and
listed separately because it is a *regression* rather than a gap: §4's last paragraph —
a node that subscribes once and then only polls `has_triggered` will now be departed and
will poll a frozen watermark forever. The clean options are (i) document that polling
without an outstanding waiter or subscribe is unsupported and fix
`benchmarks/task_throughput/task_throughput.cc:92-101`, or (ii) allow `has_triggered` a
single **relaxed** store to a `poll_watermark` that `Depart` consults — lock-free and
wait-free, but it violates the letter of §12 ("a single atomic load of the watermark and
nothing else"). I have specified (i); (ii) needs an explicit ruling because it is a
deliberate deviation from a normative document.

**Q5 — May a notification use `prev < watermark` with a widened poison range?** §6
sketches it: it costs nothing, it makes a single reordered notification free instead of
costing every subscriber a pull, and the receiver logic is unchanged. I believe it is
sound (the gap test only becomes more permissive and the union stays exact), but it is
not the state `BarrierNotify` was checked in. If it is wanted, it needs a re-run of
`Notify.cfg` with `prev` weakened, not an argument.

**Q6 — What bounds an eager-flush report and a `newplan` subtree payload?** Both are
O(subtree). The ~1 MB `MessageBlock` (`activemsg.cc:353`) is an assert, not graceful
degradation, and active-message fragmentation does **not** raise it because reassembly
happens first. If either can exceed it at target scale, chunking has to be a
*protocol-level* rule (the receiver reassembling a multi-part plan or report), which is
new protocol surface and would want its own model. Also unsettled: whether the
`(node,count)` payload uses the codec's node-set encoding plus a parallel varint count
array, or a single interleaved `(delta_node,count)` varint stream. The codec wins hard
when the participating set is contiguous or nearly all nodes; the interleaved form is
simpler for small scattered sets. Settle with `encoded_size()` measurements before C5.

**Q7 — Is one mutex per barrier acceptable for the arrival rate?** §12 is satisfied
trivially by a single lock, but every local `arrive()` serialises on it. The obvious
refinement — make `local_total` an atomic increment and take the lock only when it
crosses the quota — tears `Arrive`, whose guard reads `flushing`, the quota, the child
states and `local_total` together. If measurement shows the barrier lock is hot, the
change is a protocol question (does a "fast arrive" that observes a stale `flushing`
still satisfy rule 2?), not a local optimisation. Do not attempt it before Phase F has a
concurrency baseline.

**Q8 — One injectable transport seam, or none?** `BarrierCommunicator` today is used by
6 of 12 send sites, its `trigger()` body is compiled out in the shipping configuration,
and its only consumer — `tests/unit_tests/barrier_test.cc` — **is not in any CMake
target and has never run in CI**. §2.3 keeps it and routes everything through it, because
the alternative (deleting it) gives up the ability to drive both protocols in-process the
way `MulticastTransport` lets `multicast_test.cc` drive a whole tree — and Phase F needs
exactly that. But the current per-object `unique_ptr` costs a heap allocation, a vtable
and (today) a runtime config lookup per barrier, and barriers are bulk-constructed 16 at
a time. A per-node hook injected by `BarrierImplAllocator` would be cheaper. Decide
before C1 fixes the method set, because every Phase F unit test depends on its shape.

**Q9 — ANSWERED: NO. Leave `external_wait` exactly as it is; it is unrelated to these protocols and out of scope. Original question retained below for context.** ~~Should `external_wait` come off the main mutex?~~ The wait condition is
`gen_needed > watermark` and the watermark is atomic, so the outer lock is arguably there
only to serialise the hand-over-hand against the signal sites. Removing it would take a
nested `KernelMutex` acquisition out of the `Trigger` critical section — which is
attractive precisely because `Trigger` is now the biggest section in the file. But the
lost-wakeup argument and all signal sites have to be re-derived together; a plain
`generation.load()` plus a condvar reintroduces the race. Not required for correctness;
worth doing only if `Trigger` measures badly.

**Q10 — Which existing tests are the baseline?** `tests/unit_tests/barrier_test.cc` is
absent from `REALM_UNIT_TESTS` (`tests/CMakeLists.txt:93-130`) and has never run —
`git log -S` finds it was never registered there at all. It still compiles. Before C1
deletes anything, it should be registered and run *as-is*, purely to learn whether the
current implementation passes its own tests; roughly 8 of its 24 tests assert on
machinery being deleted and must be rewritten, 7 are genuine regression anchors, and 5
cover the reduction path. Related: `tests/barrier_arrivals.cc` returns immediately below
8 ranks (CI runs 1–2), has 2 of its 3 patterns commented out, and creates a zero-count
barrier that arrival rule 9 makes a fatal error — it looks like coverage and is inert.
And the Phase F TSAN gate does not exist: every sanitizer configuration in
`.github/workflows/ci.yml` sets `gasnet:OFF` and `ucx:OFF`, so all sanitizer coverage is
single-node, and the two networked configurations have no sanitizer.
