# Realm Scalable Barriers — Subscription / Notification Protocol

**Status: design frozen, formally verified.** Companion to
[`ARRIVAL_PROTOCOL.md`](ARRIVAL_PROTOCOL.md). `BarrierNotify.tla` in this
directory is the formal specification; where this prose and the spec disagree,
**the spec wins** — it is the artifact that was model-checked.

Written for whoever implements this in C++. Same shape as the arrival document:
what the protocol is, why each rule exists, where it lives in the TLA+ source,
what was verified, and where the model deliberately isn't the implementation.

---

## 1. Scope

**In scope:** how the owner tells the rest of the machine that a barrier
generation has triggered, and which generations were poisoned, without O(N) fan
out from one node and without O(N) per-barrier state.

**Out of scope, deliberately:**

| Excluded | Reason |
|---|---|
| Message loss | The transport is reliable (user decision) |
| **Reduction barriers** | **Fall back to the existing non-scalable path entirely** (user decision) |
| Node failure | No fault model |
| Barrier migration | Being removed as part of this work (user decision) |
| The multicast tree itself | Separately implemented; delivers to exactly the encoded set |

Reduction barriers being excluded is load-bearing, not incidental. Per-generation
reduction values cannot be re-derived from a watermark, so they would break the
delta scheme in §5. Keeping them on the legacy path means the scalable path never
grows `held_triggers`, `final_values`, or generation buffering.

---

## 2. Why this half is easier

Arrivals must be exact: a lost arrival means the generation never triggers, and
silence is fatal. Notification is best-effort, because **a node that isn't told
pulls, and the owner answers.** The pull is the correctness guarantee; the
broadcast set is a cache.

What that buys, precisely:

| Imprecision | Safe? | Cost |
|---|---|---|
| Node not notified | ✅ *if it pulls* | one round trip |
| Node notified that never asked | ✅ | bandwidth |
| Duplicate or reordered notification | ✅ | none — see the gap rule |
| Stale broadcast set | ✅ | suboptimal fan-out |
| **Watermark reported too high** | ❌ | waiter proceeds early |
| **Waiter woken with wrong poison status** | ❌ | silent corruption |
| **Pull lost or misrouted** | ❌ | hang — no second fallback |

The last three are why this still needed a spec. In particular the poison one is
subtle and is what the gap rule exists for.

---

## 3. State

From `BarrierNotify.tla:56-69`. The owner is modelled as the owner-side
variables rather than as a node.

**Owner:**

| Model variable | Meaning |
|---|---|
| `watermark` | highest triggered generation |
| `subSet` | the subscriber set, in the multicast codec's encodings |
| `setVer` | version, bumped on **every** change to `subSet` |
| `wantOut` | departure requests received but not yet applied |

Poison is not owner state in the model — it is derived from the `PoisonGens`
constant. In the implementation the owner retains the poisoned-generation set
over `[first_generation, watermark]`, since a new subscriber must be told all of
it (§5, rule 3).

**Per node:**

| Model variable | Meaning |
|---|---|
| `known[n]` | this node's known watermark |
| `knownPois[n]` | poisoned generations it knows about |
| `member[n]` | `NO` / `PENDING` / `YES` — belief about its own membership |
| `myVer[n]` | highest `setVer` it has applied |
| `waiting[n]` | generations it still needs |

**There is no learned notification tree and no per-node map at the owner.** The
multicast layer plans a fresh forwarding tree from the encoded set on every send,
so a set change can never leave a stale tree behind — which is why this protocol
needs none of the invalidation machinery the arrival one does.

### What this deletes from the current implementation

- **`remote_trigger_gens`** — the per-node O(N) map. It exists only to compute
  each node's individual gap. With deltas plus the gap rule, each receiver
  computes its own. Gone, and with it the O(N²) across N barriers.
- **`held_triggers` buffering** — a notification arriving with a gap is currently
  buffered until the gap fills. Now it is discarded and the node pulls. **Note
  `previous_gen` itself stays** — it is exactly the gap-detection field.
- **Barrier migration** and `migration_target` on the wire.

`base_arrival_count` moves to the arrival-plan dissemination path; it rides the
notification today only as a lazy-init channel for nodes with no local
`BarrierImpl`. **A barrier always has a non-zero arrival count, so `0` is an
unambiguous "not yet known" sentinel** — document that invariant at the field,
because it is the only thing making the sentinel safe.

---

## 4. Messages

`BarrierNotify.tla:71-76`.

| Kind | Fields | Purpose |
|---|---|---|
| `notify` | `to, wm, prev, pois, inset, sv` | multicast: generations `(prev, wm]` triggered, which of them are poisoned, recipient's membership, set version |
| `subscribe` | `from, lk` | pull; `lk` is the sender's last-known generation |
| `reply` | `to, wm, pois, sv` | answer: watermark plus poison in `(lk, wm]` |

`msgs` is a **set**, so TLC explores every delivery reordering. That matters —
Realm's network backends are explicitly permitted to reorder active messages
between the same pair of nodes, and two of the four protocol defects found during
design were reordering hazards.

---

## 5. The protocol rules

Eight rules. Seven have a mutation that a validated check catches; the eighth is
verified *not* to matter, and is documented as an optimization so nobody later
mistakes it for a safety property.

### Rule 1 — Membership is published, never self-decided

`BarrierNotify.tla:147-151` (notify), `:194-200` (reply).

> **The owner's published set is authoritative, and any shrink must be published
> to the PRE-shrink set.**

A node never concludes its own membership. It reads it off publications, which
ride the notification going out anyway. Because a removal is always announced to
the membership that still contains the departing node, nobody is silently
dropped.

This is what removes the subscribe/unsubscribe reorder hazard *structurally*,
rather than serialising around it with an acknowledgement handshake.

### Rule 2 — Membership updates are version gated

`BarrierNotify.tla:147-151`, `:194-200`.

```tla
newv == m.sv > myVer[m.to]
mem0 == IF newv THEN (IF m.inset THEN "YES" ELSE "NO") ELSE member[m.to]
```

**This is not optional.** TLC found the failure in the first second of the first
run: a removal at generation 3 is applied, then an older in-flight notification
from generation 2 sets `member = YES` again. The node then believes it is covered
while the owner has dropped it, never subscribes, and hangs on its next wait.

`setVer` is bumped at both places the set can change — on shrink in `Trigger`
(`:100`) and on add in `RecvSubscribe` (`:177`) — and stamped onto every
notification and reply. The same gate is required on the reply path; a stale
reply resurrects membership exactly the same way.

### Rule 3 — Adds are mandatory, removals are discretionary

`BarrierNotify.tla:178` — `subSet' = subSet \cup {m.from}`, unconditionally.

Refusing an add strands a waiter. Refusing a removal only costs bandwidth. That
asymmetry is what lets the owner apply a cost test to shrinks (encode the set
with and without the departing nodes, compare bytes and deliveries) and decline
the ones that don't pay — **dropping scattered nodes from `ALL_NODES` can turn a
0-byte encoding into a per-hop bitmap.** A declined removal is simply a
publication in which the node still appears.

### Rule 4 — Notifications are deltas, and a delta with a gap is discarded

`BarrierNotify.tla:142-143`.

```tla
gap   == m.prev > known[m.to]
fresh == (~gap) /\ (m.wm > known[m.to])
```

A notification carries `(prev, wm]` and the poison **within that range only**, so
its size never grows with the barrier's poison history. That was the last
unbounded quantity in the design.

The cost is gap sensitivity, and it is a real hazard rather than a theoretical
one. Active messages reorder, so a node can receive `(3,4]` before `(2,3]`. If it
applied that, it would advance its watermark to 4 and satisfy a waiter on
generation 3 **without knowing whether generation 3 was poisoned** — a waiter
woken with wrong poison status.

So a node whose watermark is below `prev` discards the message and pulls. This
does *not* reinstate `held_triggers`: buffering the out-of-order notification is
the alternative, and pulling is strictly simpler because the pull path has to
exist anyway.

Membership is still applied on a gap (`:148-151`) — that message may be the
node's only notice of its own removal.

### Rule 5 — Replies are deltas keyed on `lk`, and must be MERGED

`BarrierNotify.tla:183` (owner side), `:193` (node side).

```tla
pois |-> { g \in PoisonGens : (g > m.lk) /\ (g <= watermark) }   \* owner
np   == IF fresh THEN knownPois[m.to] \cup m.pois ELSE knownPois[m.to]   \* node
```

A reply is point-to-point and the subscriber declares what it already has, so the
owner can answer exactly. A notification cannot do this — it goes to nodes with
different watermarks, and keying it per-recipient is precisely the
`remote_trigger_gens` map we deleted. (That is what the current code's
`broadcast_previous = min(previous_gen)` is doing, and why it needs the map.)

**Union, do not substitute.** Substituting drops poison the node already knew
about below `lk`. This is a caught mutation.

Soundness rests on the node's poison knowledge being *complete* up to the
generation it reports, which is exactly what `PoisonAccurate` asserts — so the
two compose.

### Rule 6 — A node removed while holding a waiter re-subscribes at once

`BarrierNotify.tla:152` — `resub == (mem0 = "NO") /\ (w2 # {})`.

The idle counter advances with the watermark, so **a node waiting on a far-future
generation looks idle.** Waiting on generation 100 while the watermark climbs
1…99 accumulates K idle generations with no consultation; the node departs and
never learns.

The obvious guard — don't depart while holding a waiter — **is not sufficient**,
because a node can register a waiter after its departure request is sent but
before the owner applies the shrink. The recovery is the correctness rule; the
guard (`:213`, verified benign) is an optimization that reduces churn.

### Rule 7 — At most one outstanding pull per node

`BarrierNotify.tla:156-157`.

An in-flight subscribe already carries an `lk` at or below what a new one would,
so its reply is a superset of what the second pull would ask for. Suppressing the
duplicate is safe and keeps a burst of reordered notifications from producing a
burst of pulls.

### Rule 8 — Departure hysteresis (performance only)

Not modelled — see §8.2. A node signals departure after **K = 8** consecutive
generations without consulting the barrier, where *consulting* means
`add_waiter`, `subscribe`, or `external_wait`/`external_timedwait`, and
explicitly **not** `has_triggered`, which must stay lock-free.

- Measured as watermark delta since last consultation, so it is robust to the
  owner coalescing several triggers into one notification.
- K adapts by doubling on observed churn (a leave→rejoin within a short window),
  which is also the metric worth exporting.
- Departures are **unicast to the owner**, staggered by `K + (node_id mod J)`
  with **J ≈ 16–32**. Phase changes retire many nodes at once, so the stagger is
  load-bearing rather than defensive. J is sized for spike smoothing, not
  asymptotics: the subscribe path is *already* O(N) unicast, so an O(N)
  unsubscribe burst is not a new complexity class.
- The owner may publish a one-byte hint on notifications saying whether shrinking
  currently pays, so nodes suppress unsubscribes that would just be declined.

**Why unicast rather than aggregating up the multicast ack tree:** that tree is
real (`MulticastAckMessage`, `activemsg.h:559-576`) but carries **no payload** —
`send_ack` at `activemsg.cc:1762` attaches none and the handler ignores the
argument. Aggregating departures would mean adding a payload-reduction facility
to the multicast layer, and multicast completion tracking currently has *no
callers at all*, so we would be designing that API around one speculative user.
Deferred until measurement justifies it.

---

## 6. Safety properties

`BarrierNotify.tla:242-274`. All five are checked in every run.

| Property | Line | Meaning |
|---|---|---|
| `TypeOK` | 242 | well-formedness; also `myVer[n] <= setVer` |
| `NeverOverstate` | 249 | a node is never told a generation triggered before it did |
| `PoisonAccurate` | 254 | a node's poison knowledge is **exactly** the truth up to its own watermark |
| `NoStranded` | 261 | **the liveness-critical one** |
| `MembershipPublished` | 270 | a node believing it is covered is covered, or its correction is on the wire |

### `NoStranded`

```tla
NoStranded ==
    \A n \in Nodes :
        (waiting[n] # {}) =>
            \/ n \in subSet
            \/ (\E m \in msgs : (m.kind = "notify" \/ m.kind = "reply") /\ m.to = n)
            \/ (\E m \in msgs : m.kind = "subscribe" /\ m.from = n)
```

A node holding an outstanding waiter must be covered, or have something in flight
that will cover it. This is the same shape as `ReachableWhileHolding` in the
arrival spec — a liveness property encoded as safety — and the in-flight
disjuncts matter for the same reason: a node being removed is briefly outside the
set while its own removal notice is still on the wire.

### `MembershipPublished`

```tla
(member[n] = "YES") =>
    \/ n \in subSet
    \/ (\E m \in msgs : m.kind = "notify" /\ m.to = n /\ ~m.inset /\ m.sv > myVer[n])
```

The `m.sv > myVer[n]` clause is essential: a correction the node would *ignore as
stale* is not a correction. Without it this check passes on the very state that
Rule 2 exists to prevent.

### Why `Done` exists

`BarrierNotify.tla:222` — a settled run stutters, so TLC's deadlock detection
never mistakes "finished" for "stuck." **Deadlock checking must stay ON**; it is
what catches the "reply carries no watermark" mutation.

---

## 7. What was verified

### Baseline

`MCNotify.tla` + `Notify.cfg` — 3 remote nodes, 3 generations, 1 poisoned:
**83,011,199 distinct states, no error, ~18 minutes.**

Scripted: which generations each node consults, and which are poisoned.
Nondeterministic: *when* each node consults (including generations far in the
future), when a node signals departure, which subset of requests the owner
actually applies, and every message delivery order. That covers **any** choice of
K, J, or shrink policy — which is the point. The constants must not be able to
affect correctness.

### Mutation battery

| Mutation | Caught by |
|---|---|
| apply a delta notification despite a gap | `PoisonAccurate` |
| removed node with a live waiter doesn't re-subscribe | `NoStranded` |
| stale notify may resurrect membership | `MembershipPublished` |
| shrink published to the post-shrink set | `MembershipPublished` |
| adds are discretionary | `NoStranded` |
| subscribe reply carries no watermark | deadlock |
| delta reply substituted rather than merged | `PoisonAccurate` |
| **may depart while holding a waiter** | **not caught — benign** ✓ |

Seven catches across four different invariants, each by the check that should own
it. The eighth is a negative control confirming the model distinguishes
correctness rules from optimizations.

**Caveat, stated precisely: the negative control is verified at 2 nodes / 3
generations, not 3.** At 3 nodes it diverged past 268M distinct states with a
growing queue. The 2-node scenario was separately shown to have teeth — it
catches the gap, no-recovery, and discretionary-adds mutations — so its clean
verdict is meaningful rather than vacuous, but it is weaker evidence than the
other seven rows.

### Sizing

| Scenario | Nodes | Gens | Distinct states | Time |
|---|---|---|---|---|
| `MCNotifySmall` (in `/tmp`, not retained) | 2 | 3 | 341,647 | seconds |
| **`MCNotify`** | **3** | **3** | **83,011,199** | **~18 min** |
| `MCNotifyBig` | 4 | 3 | >47M, diverging | abandoned |

**Three nodes is the ceiling.** Two lessons worth carrying forward:

1. **Unconstrained nondeterminism is what explodes this.** The first attempt left
   the consultation pattern free and blew past 75M states without finishing;
   scripting the pattern and leaving only interleavings free brought the same
   scenario to 898k. (The later jump to 83M is the delta change, not the
   patterns.)
2. **A guard-removal mutation enlarges the state space rather than shrinking it**,
   so clearing one as benign costs far more than catching a real defect — caught
   mutations stop at the first violation. Size scenarios for the negative
   controls, not for the baseline.

Also worth knowing: switching notifications from snapshots to deltas cost
898k → 83M states and 6s → 18min. Gap-insensitivity was doing enormous work for
the *model* even while it was a liability on the *wire*.

---

## 8. Fidelity notes — where the model isn't the implementation

### 8.1 The multicast tree is abstracted away

A notification is modelled as one message per member. The forwarding layer is
verified separately and delivers to exactly the encoded set, so this is sound —
but it means **nothing here verifies the tree**.

Relevant property the implementation gets for free: `plan_children` partitions
the target set and `activemsg.cc:1210` sends each slice to `slice.first_node()`,
so **every relay is itself a member of the destination set.** No node outside it
is ever touched, and the delivered message count is exactly `|set|`.

### 8.2 K, J, and the shrink policy are nondeterminism

`Depart` (`:210`) may fire on any eligible node, and `Trigger` may apply any
subset of pending requests. That is strictly more general than any tuning, so no
choice of constants can be wrong — but it also means **§5 rule 8 is entirely
unverified**. It is a performance mechanism whose failure modes are bandwidth,
not correctness.

### 8.3 The owner is variables, not a node

`Nodes` is the set of *remote* nodes. An owner that is itself a waiter is a
local, trivially-satisfied case and is not modelled.

### 8.4 What the owner does with departure requests is unmodelled

The cost test that decides whether a shrink pays is described in rule 3 but not
specified or verified. Same gap as plan construction in the arrival spec: what is
verified is that **any** shrink the owner chooses is safe, not that it chooses
well.

### 8.5 Poison retention is a constant here

The model derives poison from `PoisonGens`. The implementation must actually
retain the poisoned-generation set over `[first_generation, watermark]`, because
rule 5 requires answering a new subscriber with everything above its `lk`.

**That set is bounded by the barrier's lifetime, not by MAX_PHASES.**
`first_generation` advances when a barrier is destroyed and its id handed out
again, so retention only ever reaches back to the generation at which *this*
barrier was created. `destroy_barrier` also implicitly waits for every generation
up to that point to trigger, so there is a clean point at which the whole set is
discarded.

If poisoning ever were dense enough to matter within one barrier's life, the
escape hatch is to cap what a *notification* carries and let nodes pull for
detail — rule 5 is what makes that pull cheap. Not specified; no workload has
justified it.

### 8.6 Concurrency is not modelled

Every action here is atomic; the implementation is not. See
[`ARRIVAL_PROTOCOL.md` §12](ARRIVAL_PROTOCOL.md#12-atomicity) — the requirement
applies identically to this protocol, and the `has_triggered` carve-out
originates here.

---

## 9. Reproducing

```sh
cd tla/barrier
java -Xmx8g -XX:+UseParallelGC \
    -Djava.io.tmpdir=jtmp -cp tools/tla2tools.jar tlc2.TLC \
    -workers 8 -config Notify.cfg MCNotify.tla
```

Expect `Model checking completed. No error has been found.` and 83,011,199
distinct states.

**Two operational warnings, both learned the hard way:**

- **This run consumes tens of GB of scratch.** Delete `jtmp/` and `states/`
  afterwards. Filling the volume kills runs in ways that look like unrelated
  failures — an empty log and a bare non-zero exit.
- **Never mutate `BarrierNotify.tla` in place.** Copy the module, the `MC*`
  scenario and the `.cfg` into a scratch directory, mutate *there*, and run TLC
  in that directory. A `finally` block does not survive a kill, and a cleanup
  step that never runs leaves a silently corrupted spec on disk. Assert that each
  mutation pattern matches **exactly one** site — after the delta change,
  `RecvNotify` and `RecvReply` contain identically-worded poison-merge lines, and
  an unanchored replace silently patches the wrong one and reports a result for a
  test it never ran.

To classify a verdict, grep for `Error: Invariant <name> is violated.` — note the
`Error:` prefix.

---

## 10. Implementation checklist

- [ ] Owner keeps an encoded subscriber set + `setVer` + watermark + poisoned set;
      **no per-node map.**
- [ ] Bump `setVer` on every set change; stamp it on every notification and reply.
- [ ] Apply a membership update only if `sv` exceeds the highest already applied —
      on **both** the notify and reply paths.
- [ ] Publish every shrink to the **pre-shrink** set.
- [ ] Never refuse an add. Refuse removals freely, subject to the cost test.
- [ ] Notification carries `(prev, wm]` + poison in that range; a node below
      `prev` **discards and pulls**, but still applies membership.
- [ ] Subscribe carries `lk`; reply carries poison in `(lk, wm]`; the node
      **unions** it.
- [ ] Reply must carry the watermark — that is what covers the trigger-during-
      subscribe race.
- [ ] On removal while holding a waiter, **re-subscribe immediately.**
- [ ] At most one outstanding pull per node.
- [ ] Consultation = `add_waiter` / `subscribe` / `external_wait`; **never**
      `has_triggered`.
- [ ] Departures unicast, `K = 8` adaptive, staggered by `J ≈ 16–32`.
- [ ] Reduction barriers route to the legacy path; `base_arrival_count == 0`
      means unknown, documented at the declaration.

If you change any of these, re-run the battery. A rule with no mutation covering
it is a rule nobody has verified.
