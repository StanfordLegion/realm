# Realm Scalable Barriers — Arrival Protocol

**Status: design frozen, formally verified.** Revised after the pinned-edge
rework (see rule 10) — the spec grew five rules, and one rule this document
previously required (install-clears-flush) is now a **caught mutation**. This document is the normative
description of the barrier *arrival* protocol. `BarrierArrive.tla` in this directory
is the formal specification; where this prose and the spec disagree, **the spec
wins** — it is the artifact that was model-checked.

This document is written for an agent or engineer implementing the protocol in
C++. It covers what the protocol is, why each rule exists, exactly where each
rule lives in the TLA+ source, what was verified and how, and — importantly —
the places where the model is deliberately not the implementation.

---

## 1. Scope

**In scope:** how arrivals at many nodes are aggregated up to the barrier owner
so the owner can decide a generation has triggered, including adaptation when
the arrival pattern changes from one generation to the next, and
`alter_arrival_count`, which changes how many arrivals the owner is waiting for.

**Out of scope, deliberately not modelled:**

| Excluded | Reason |
|---|---|
| Message duplication | The transport is reliable (user decision) |
| Reduction barriers | Not in the first implementation (user decision) |
| Node failure | No fault model in this design |
| Subscription / trigger-notification tree | A separate protocol; needs its own spec |
| Resource reclamation | `destroy_barrier` is the cleanup point (user decision) |

The exclusions are recorded in the spec header at `BarrierArrive.tla:35-37`.

Negative alterations and the terminal-negative case (`event.h:293-297`) are
modelled only in so far as the delta is a constant in `AlterOps`; the scenario
uses a positive delta. See §8.6.

---

## 2. The problem this solves

A barrier generation triggers when the owner has accounted for every arrival.
The naive approach — every node reports every arrival directly to the owner — is
O(N) messages at one node. The scalable approach aggregates through a tree.

The tree cannot be static, because Realm does not know in advance which nodes
will arrive on a barrier or how many times. So the owner *learns* an arrival
plan from observed traffic and pushes it out; nodes then aggregate along it.

The hazard is **silence**. A node that is aggregating stays quiet until its
subtree is complete. If the actual arrival pattern deviates from the plan — a
node arrives that the plan did not expect, or a node arrives more times than
predicted — then some node waits for a completion condition that will never be
met, and the generation never triggers. The previous design (see
`../../src/realm/SCALABLE_BARRIERS_IMPLEMENTATION_PLAN.md`) died on exactly this;
TLC found a terminal deadlock in the model of it (`BarrierArrival.tla`).

This protocol's entire job is to guarantee that **every arrival eventually
reaches the owner, without timers, polling, or background sweeps.** Every action
is caused by an arrival or by the receipt of a message — the protocol is
feed-forward.

---

## 3. Per-node state

From `BarrierArrive.tla:67-86`. Model variables are global functions over nodes; in
the implementation each node holds its own slice.

| Model variable | Per-node meaning | Suggested C++ home |
|---|---|---|
| `localTotal[n][g]` | arrivals issued *at this node* for generation `g` | per-generation counter |
| `childAcc[n][c][g]` | highest cumulative value accepted from child `c` for `g` | small map keyed by child |
| `reportedUp[n][g]` | cumulative value last sent to the parent for `g` | per-generation counter |
| `flushing[n][g]` | eager-flush mode, **per generation** | per-generation flag |
| `curPlan[n]` | this node's own plan record: `quota`, `inplan`, `kids` | plan struct |
| `myEpoch[n]` | which plan index `curPlan` is | integer |
| `invalEpoch[n]` | highest plan index invalidated here | integer |
| `deferEpoch[n]` | a parked new plan, 0 = none | integer |

| `myTs[n][g]` | timestamp this node's arrivals for `g` carry, 0 = none | per-generation field |

Owner-only: `ownerAcc[g]` (accepted total per generation), `watermark` (highest
contiguous triggered generation), `triggered[g]`, plus the alteration state
`expected[g]` (the count currently being waited for), `appliedTs` (alterations
applied) and `tsAcc[g]` (timestamped arrivals counted).

Two derived quantities are used constantly (`BarrierArrive.tla:143-145`):

```tla
SubtreeKnown(n, g) == localTotal[n][g] + (sum over children of childAcc[n][c][g])
Unreported(n, g)   == SubtreeKnown(n, g) - reportedUp[n][g]
Holding(n, g)      == Unreported(n, g) > 0
```

`Holding` is the danger condition: this node knows about arrivals it has not
passed on. Every liveness argument in the protocol is about making sure a
holding node is always reachable by something that will make it speak.

### Membership rule

**Only nodes with a non-zero expected contribution are in a plan at all.** A node
whose predicted arrival count is zero is not in the tree — it is not a leaf, not
a relay, not anything. It is in eager-flush mode permanently, by virtue of
`curPlan[n].inplan = FALSE`.

This is a load-bearing simplification (it was a user design decision, and it
removes a whole class of "waiting on a child that will never speak" states). The
model encodes it as the `NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]`
record used in every scenario file, e.g. `MCSeven.tla:48`.

---

## 4. Messages

`BarrierArrive.tla:114-124`.

| Kind | Fields | Purpose |
|---|---|---|
| `report` | `from, to, gen, val` | cumulative subtree total, sent to parent |
| `direct` | `from, to, gen, val` | same, but sent straight to the owner by a node outside the plan |
| `flush` | `from, to, gen` | "enter eager-flush mode for this generation" |
| `invalidate` | `from, to, epoch` | "the plan you hold is being retired" |
| `newplan` | `from, to, epoch` | "install plan `epoch`" |
| `alter` | `from, to, gen, delta, ts` | `alter_arrival_count` — persistent change to the expected count |
| `tsdirect` | `from, to, gen, ts, val` | an arrival carrying a causal timestamp, sent straight to the owner; `val` is cumulative |

`msgs` is a **set**, not a queue — TLC therefore explores every delivery
reordering. Reports are **cumulative, not incremental**: `val` is the running
subtree total, so a later report replaces an earlier one rather than adding to
it. This is what makes duplicate and out-of-order delivery harmless.

---

## 5. The protocol rules

Ten rules. Each one has a mutation in the battery that a validated check
catches, so none of them is decorative. Rule 10 is a bundle of five, added
after the pinned-edge rework; its history is instructive and is told inline.

### Rule 1 — Steady state: report on match, otherwise stay silent

`BarrierArrive.tla:216-230` (arrival path), `BarrierArrive.tla:246-251` (report path),
predicate at `BarrierArrive.tla:163-166`.

A node forwards its cumulative subtree total when its local arrival count equals
the quota its plan predicts. Below quota it stays silent. This is the aggregation
that makes the protocol scalable.

> **There is NO child-wait, and an earlier version of this document was wrong to
> say there was.** It claimed a relay must also wait for every predicted child,
> called that a correctness requirement, and warned against removing it.
>
> It is an optimisation at best. Reports are cumulative and **replace**, so a
> forward that omits an absent child is superseded the moment that child
> reports; and `Trigger` demands exact equality with the expected count, so a
> temporarily-low total simply does not fire.
>
> Seven scenarios agree, including `MCChain` and `MCDeepSwitch`, both built
> specifically to contain the relay-below-relay structure the old claim named.
> **No scenario in the suite had that structure when the claim was recorded** —
> MCLarge and MCSeven are both owner → relay → leaf.
>
> Removing it also matters: the child-wait is the only reason a relay cares
> *which* child reports to it, and therefore the only reason a re-parented
> generation can strand. It is what forced flush mode to be sticky. See rule 5.

### Rule 2 — Over-arrival at a predicted node triggers eager flush

`BarrierArrive.tla:217-222`.

If a node in the plan receives more arrivals than its quota, its plan is wrong
and waiting is unsafe. It:

1. sets `flushing[n][g] = TRUE`,
2. reports its cumulative total upward immediately,
3. fans a `flush` announcement out to its children.

From then on, for that generation, every arrival reports immediately.

### Rule 3 — Arrival at a node outside the plan reports directly to the owner

`BarrierArrive.tla:212-216`, owner-side handling at `BarrierArrive.tla:254-255,262-264`.

A node not in the plan has no parent and no completion condition to wait for, so
it cannot aggregate. It sends a `direct` report to the owner. The owner, on
receiving a `direct`, enters flush mode for that generation and fans a `flush`
out through its children.

### Rule 4 — Flush is per generation and idempotent

`BarrierArrive.tla:284-295`.

A node receiving `flush` for a generation it is already flushing just drops the
message (this is what terminates the fan-out). Otherwise it sets the flag,
re-fans to its children, and — critically — **immediately reports anything it is
currently holding** for that generation.

Flush mode is per generation. The next generation starts back in planned mode.

### Rule 5 — Pattern change: invalidate the old tree, broadcast the new plan, defer on overlap

`BarrierArrive.tla:304-330` (owner initiates), `:332-368` (invalidation),
`:370-401` (new plan).

When the owner triggers generation `g` and a new plan starts at `g+1`, it sends
**both** broadcasts at once:

- `invalidate` down the tree being retired, carrying the retiring epoch;
- `newplan` down the *new* tree, carrying the new epoch.

These race. The resolution is entirely local, with no global knowledge:

> **A node that receives `newplan` while it is still an un-invalidated member of
> the retiring plan PARKS it** (`deferEpoch`). When the `invalidate` arrives, it
> does its invalidation work and *then* applies and forwards the parked plan.

The condition is `curPlan[m.to].inplan /\ (invalEpoch[m.to] < myEpoch[m.to])` at
`BarrierArrive.tla:375`. A node's own membership is the flag, and it is
self-clearing — if the invalidation already passed, the node looks like it was
never in the old plan and the new plan installs immediately.

This is the piece that motivated the whole verification effort. **It is
verified**: disabling the deferral is the first mutation in the battery and TLC
reports `Invariant ReachableWhileHolding is violated` in ~46 seconds.

### Rule 6 — On invalidation: forward first, flush *every* open generation, then switch

`BarrierArrive.tla:338-366`.

Order matters, and both halves are separately verified:

- **Forward before forgetting.** The node re-sends `invalidate` to the children
  in its *current* (old) list *before* replacing that list. Dropping the child
  list first strands the entire subtree — a mutation TLC catches.
- **Report held work for every open generation, not just the switch
  generation** — `held == { g \in Gens : ~triggered[g] /\ Unreported(...) > 0 }`
  (`BarrierArrive.tla:412`), unconditional on the flushing flag.
- **The flushing flag itself is also set** for open generations with state
  (`:452-456`) — but honesty requires recording that under the final protocol
  this half is **no longer independently caught**: narrowing it to the switch
  generation passes MCStrand2, MCStale and MCPark. The rule-10 machinery
  subsumes it — held work goes out unconditionally at the invalidation, a
  planless node is eager for everything, and any straddling-generation holder
  under a live plan is unstuck by the owner's deviation fan, which conservation
  guarantees will fire. Retained as defense-in-depth: over-flushing is always
  safe, and the flag is what carries `saw_direct` onto held reports in the
  implementation.
- **A plan install NEVER clears flush** (`BarrierArrive.tla:454-456` sets it,
  RecvNewPlan leaves it unchanged). An earlier revision of this document said
  the opposite — that installing a plan returns open generations to planned
  mode — and that rule is now a **caught mutation** (`MCStale`, deadlock): the
  owner's deviation flush and the `newplan` race, and when the install wins the
  flush is lost and the generation strands behind a quota the new plan cannot
  meet.

  A flushed generation stays eager **until it triggers**. That is naturally
  bounded — the flag lives on the generation record and dies with it, and new
  generations start planned under whatever plan the node holds — so nothing is
  sticky across the barrier's future. The unbounded-stickiness scare that
  motivated install-clears-flush was partly a modelling artifact: the model
  pre-enumerates all generations, so flagging them all at an invalidation
  looked like eager-forever when it never was.

### Rule 7 — Cumulative totals only increase; non-increasing reports are stale

`BarrierArrive.tla:242` (guard), `:269-280` (`DropStale`), `:258-261` (owner update).

A report whose `val` does not exceed the stored `childAcc` for that child and
generation is stale and is discarded. The owner's accumulator **replaces** the
child's previous contribution rather than adding to it:

```tla
ownerAcc' = [ownerAcc EXCEPT ![m.gen] = @ - childAcc[Owner][m.from][m.gen] + m.val]
```

Accepting stale reports lets the owner's count go *down*, which is a mutation
TLC catches. Implementations must not treat reports as increments.

### Rule 8 — `alter_arrival_count`: timestamped arrivals bypass the tree

`BarrierArrive.tla:403-441` (issue), `:443-457` (owner applies), `:459-471` (the gate),
arrival path at `:198-202`.

`alter_arrival_count` changes how many arrivals the owner is waiting for. Unlike
every other deviation in this protocol that is a **counting** problem, not a
routing one, and getting it wrong is a *safety* failure — `event.h:303-306` names
the symptom as "a barrier that triggered too early."

The ordering is already solved and does **not** need message ordering. A negative
delta (an arrival) names the positive adjustment it depends on, and the owner
holds it until that adjustment lands — `barrier_impl.h:144-146` plus the
per-generation `pending` map at `:188-199`. The model encodes that buffer as a
guard (`m.ts \in appliedTs`), which is why it needs no ordering assumption.

What tree aggregation breaks is different: **a relay collapses its subtree into a
single integer, which erases the timestamps.** Three rules follow.

1. **An arrival carrying a timestamp bypasses the tree** and is reported to the
   owner carrying that node's **cumulative** count of bypassed arrivals — the
   same replace-if-higher shape as every other report in the protocol
   (`:201`, `TsIssued` at `:107`). Alterations are rare relative to arrivals, so
   the direct traffic is negligible, and no report ever has to carry a set of
   timestamps, which is where the previous design's causal-DAG machinery came
   from.

   **Every accumulator in this protocol is cumulative, replace-if-higher, and
   counts in one direction.** The owner's timestamped total is a sum of per-node
   cumulative values (`TsTotal`, `:112`), exactly like `childAcc`. Do not
   introduce an incrementing counter alongside them, and do not carry the legacy
   path's count-down convention (`base_arrival_count + unguarded_delta == 0`)
   into the scalable path.
2. **The owner may not count a timestamped arrival until every alteration it
   witnessed has been applied.** This is the existing `pending` map.
3. **Issuing an alteration puts the node into eager flush** for every affected
   open generation. This one is easy to miss: a node that bypasses the tree can
   never satisfy its own plan quota via `localTotal`, so as a *relay* it would go
   silent and strand its children. Removing it is a caught mutation (deadlock).

The safety argument for why the owner cannot trigger on a stale count lives in
the **API contract, not the protocol**: `event.h:290-292` requires the caller to
still hold an unissued arrival from the pre-alteration count, and that reserved
arrival is what holds the generation open until the alteration lands. The model
encodes it as the guard at `:411`, and removing it is caught by `TriggerCorrect`.

> **The current implementation does not match this contract and must be fixed.**
> `event.h:271` says the delta is persistent across all future generations, and
> that is what Legion expects. The code applies it to a single generation: the
> delta lands in `Generation::unguarded_delta` (`barrier_impl.h:192`), the
> trigger test is `base_arrival_count + unguarded_delta == 0`
> (`barrier_impl.cc:847`), and `base_arrival_count` is never modified by an
> alteration. Each generation gets a fresh `Generation`, so nothing carries
> forward. There is no test anywhere in the repo that calls the API, which is why
> this went unnoticed.

### Rule 9 — A negative alteration is an arrival that also invalidates the plan

Not modelled (see §8.6); recorded here as a design decision.

A negative delta behaves as an **arrival** — it reduces the remaining count the
same way — and additionally **invalidates the current arrival plan**, because
lowering a node's expected contribution makes at least one quota in the plan
unreachable, and a node waiting on an unreachable quota is exactly the silence
failure this protocol exists to prevent.

Two guards:

- The barrier's base arrival count reaching **zero is an error**. A barrier with
  no expected arrivals is meaningless, which is also what makes `0` a safe
  "not yet known" sentinel for `base_arrival_count` (§3).
- The terminal-negative case of `event.h:293-297` needs no arrival on the
  returned handle, so it is the one branch whose safety argument does *not* rest
  on a reserved arrival. It has no coverage — see §8.6.

### Rule 10 — Plan changes race with everything: the pinned-edge rules

Added after `MCDeepSwitch` (three plans over a relay chain) violated
`NoOverCount` — a **safety** failure, the owner counting more arrivals than
issued — in a spec whose battery was all-green. Five sub-rules; every one is a
caught mutation, and four candidate fixes died to counterexamples on the way.
The unifying observation: **a re-routed generation loses either a count (it
arrives twice) or a signal (it arrives with nobody told), and both are fatal.**

1. **Pinned report edges** (`TargetOf`/`Pin`, `BarrierArrive.tla:187-195`; pinned
   at every touch, e.g. `:278,318,351,471`). A node's report target for
   generation `g` is fixed at its first touch of `g`; later plan changes do
   not re-aim it. Without this the owner double-counts: reports are cumulative
   and can never be retracted, so a contribution folded into an old parent's
   aggregate plus the same node's post-switch report arrive under two sender
   keys and are summed. Caught by `MCDouble` — `NoOverCount`.
   *Cycle-freedom assumption:* pinned edges from different plans must not form
   a cycle. The implementation guarantees this by building every plan tree in
   ascending node id with the owner at the root, so every report edge strictly
   decreases.

2. **The live guard** (`live == dk > m.epoch`, `:407`). A parked plan is
   applied only if strictly newer than the invalidation delivering it —
   otherwise it is being delivered by its own death notice: installing it
   re-enters planned mode with the plan's only invalidation already consumed,
   and forwarding it strands every descendant, who never see that invalidation
   at all. A dead parked plan is DISCARDED: not installed, not forwarded.
   Caught by `MCStrand2`.

3. **The install guard** (`invalEpoch >= m.epoch` in RecvNewPlan's discard
   condition, `:489`). Same principle at the second door: messages reorder, so
   a `newplan` can arrive after both of its own broadcast's invalidations have
   overtaken it, and a plan whose retirement this node has witnessed is dead
   on arrival. One principle, two doors: **a plan with epoch `e` is
   installable only if `e > myEpoch` and `e > invalEpoch`.** Caught by
   `MCStrand2`.

4. **A planless node becomes an outsider** (`:464-467`), and **case 3
   delivers late**. A node invalidated with no live replacement clears its
   plan record, so later run-ahead arrivals fire case 3 natively — and for
   arrivals it already made, it sends a **count-free flush signal to the
   owner** (`:432-436`): case 3 delivered retroactively. In `Arrive`, case 3
   **outranks the flushing flag** — the count still follows the pinned edge;
   only the signal goes to the owner, because a cumulative value sent off-pin
   double-counts. All three are caught by `MCStale`.

5. **Stale-edge forwarding** (`:302`). A report arriving from a node not in
   the receiver's child list is forwarded at once rather than held behind the
   receiver's quota — the pinned edge is the only route that contribution has.
   Caught by `MCDouble`, `MCStrand` and `MCStrand2` (deadlock). *This rule
   evaded five scenarios and looked deletable; it is load-bearing. Pinning is
   what creates legitimate non-child reports, so a rule that was decorative in
   the unpinned protocol became essential — the child-wait lesson, reversed.*

The recurring failure shape, named for whoever extends this: **the count and
the signal must never travel separately.** Four of the five bugs were a count
arriving with its signal lost — folded into an aggregate, cleared by an
install, suppressed by a branch priority. The implementation's `is_direct`
flag (the signal riding the count's own message) is immune to this class by
construction, which is why the C++ port of rule 10 was three edits, not five.

---

## 6. Safety properties

`BarrierArrive.tla:487-527`. All six are checked as invariants in every scenario
config.

| Property | Line | Meaning |
|---|---|---|
| `TypeOK` | 493 | well-formedness; also bounds-checks counters |
| `TriggerInOrder` | 499 | generations trigger in order, no gaps |
| `TriggerCorrect` | 505 | a triggered generation has accounted for *exactly* every arrival |
| `NoOverCount` | 507 | the owner never counts more than were issued |
| `ReachableWhileHolding` | 519 | **the liveness-critical one** — see below |
| `BoundedRetention` | 527 | at most one parked plan per node; retained state is bounded |
| `ExpectedSane` | 513 | scenario sanity: the owner's expected count never exceeds what will be issued |

`TriggerCorrect` is also the check that owns `alter_arrival_count`. `Total(g)`
(`:99`) is ground truth — the pattern's base arrivals plus the deltas of every
alteration actually issued — and is independent of what the owner currently
believes `expected` to be. A trigger against a stale count violates it at once.

### `ReachableWhileHolding` — read this before changing anything

```tla
ReachableWhileHolding ==
    \A n \in Nodes, g \in Gens :
        (~triggered[g] /\ Holding(n, g) /\ n # Owner) =>
            \/ flushing[n][g]
            \/ ~curPlan[n].inplan
            \/ (n \in Reachable)
```

"No node may sit on unreported work with nothing that will ever collect it."
This is a *safety* encoding of the liveness property that actually matters, and
it is the check that catches every one of the seven mutations.

Two subtleties, both of which were bugs in earlier versions of this check:

1. **`KidsOf(n) == curPlan[n].kids`** (`BarrierArrive.tla:127`) — a node's children
   come from its **own** plan record. An earlier version derived them from other
   nodes' beliefs about who their parent was (`{m : plan[m].parent = n}`), which
   let the check conclude a node was reachable when nobody would ever forward to
   it. That single defect hid a real race through three attempts. The comment at
   `BarrierArrive.tla:124-126` marks this.

2. **`Reachable` is seeded from in-flight messages, not just the owner**
   (`BarrierArrive.tla:137-139`). A parent that has forwarded an invalidation has
   discharged its duty and may drop the child, so the child is briefly in no tree
   while the message is on the wire. Crediting only messages addressed to the
   *holder* is not enough either — an invalidation travelling toward an
   **ancestor** will reach the whole subtree under it. Both narrower versions
   fire on perfectly healthy states.

### Why `Done` exists

`BarrierArrive.tla:473`:

```tla
Done == (\A g \in Gens : triggered[g]) /\ UNCHANGED vars
```

A completed run stutters, so TLC's deadlock detection does not mistake
"finished" for "stuck". Any remaining state with no successor is a genuine
deadlock. **Deadlock checking must stay ON** in every config — it is doing real
work here.

> A `EventuallyTriggers` temporal property was tried and **abandoned**. It never
> fired, even on a state space containing a genuinely stuck state. Every earlier
> claim of "liveness caught it" was false. Deadlock detection plus
> `ReachableWhileHolding` is what actually works. Do not reintroduce the temporal
> property without validating it fires on a known-broken spec first.

---

## 7. What was verified

### Scenarios

| Module | Config | Nodes | Gens | Plans | Arrivals | Distinct states | Purpose |
|---|---|---|---|---|---|---|---|
All state counts below are on the **final spec** (post rule 10) unless marked.

| Module | Config | Nodes | Gens | Plans | Distinct states | Purpose |
|---|---|---|---|---|---|---|
| `MCOver.tla` | `Over.cfg` | 5 | 2 | 1 | 10,591 | rule 2 in isolation (no outsider to mask it) |
| `MCOverSwitch.tla` | `OverSwitch.cfg` | 3 | 2 | 2 | 14,582 | **over-arrival × plan switch** — closes the gap the unaffordable trio left |
| `MCDeviate.tla` | `Deviate.cfg` | 5 | 2 | 1 | 814,685 | rule 3, outsider |
| `MCDouble.tla` | `Double.cfg` | 4 | 2 | 2 | 41,154 | **the double-count, minimal** — re-parented node with a child of its own |
| `MCStale.tla` | `Stale.cfg` | 4 | 2 | 2 | 7,826 | **run-ahead outsider + over-predicting quota** — found five bugs in candidate fixes |
| `MCStrand.tla` | `Strand.cfg` | 4 | 2 | 2 | 3,936 | single-switch stranding (control: passes, one switch is not enough) |
| `MCStrand2.tla` | `Strand2.cfg` | 4 | 3 | 3 | 20,034,804 | **two successive switches + run-ahead + relay chain** — the strand, minimal |
| `MCChain.tla` | `Chain.cfg` | 5 | 2 | 1 | 435,848 | relay below a relay |
| `MCAlter.tla` | `Alter.cfg` | 4 | 2 | 1 | 71,124 | `alter_arrival_count` — altering node is a *relay*, not a leaf |
| `MCDeepSwitch.tla` | `DeepSwitch.cfg` | 5 | 4 | 3 | **unaffordable** | found the `NoOverCount` violation that forced rule 10; outgrew the machine after the fix |
| `MCLarge.tla` | `Large.cfg` | 6 | 3 | 3 | **unaffordable** — >27M, growing | superseded by MCStrand2 + MCOverSwitch |
| `MCLate.tla`, `MCSeven.tla`, `MCBig*.tla` | — | — | — | — | **unaffordable** | historical; retained as records |

The suite philosophy changed with rule 10 and it matters: **the load-bearing
scenarios are now the minimal ones.** `MCStale` — four nodes, ~8k states —
found five distinct bugs in candidate fixes that 20M-state scenarios never
reached, because it was built from a failing *trace*, not from topology
intuition. When a big scenario fails, distil the trace into a minimal scenario
first; iterate there; keep the big one only as breadth.

**`MCDeepSwitch` and `MCChain` are the primary evidence now.** `MCSeven` was,
until removing the child-wait roughly tripled its state space and put it out of
reach (abandoned past 28M distinct states with the queue still growing). It is
retained but no longer runnable on the current spec.

That is a smaller loss than it sounds: `MCSeven` never actually contained a
relay below a relay — like `MCLarge` it is owner → relay → leaf — so the two
purpose-built scenarios cover structure it never had, at a few percent of the
cost. Its topology is still documented in `MCSeven.tla:12-24`:

```
  plan 1 (gen 1)        plan 2 (gen 2)        plan 3 (gen 3)
    0                     0                     0
    |- 1  q1              |- 2  q1              |- 1  q1
    |  |- 3  q2           |  |- 5  q1           |  |- 6  q1
    |- 2  q1
       |- 4  q1

  g1  matches plan 1                       -> steady state
  g2  DISAGREES with plan 2: node 2 over-arrives (case 2) and node 6
      arrives while in no plan (case 3)
  g3  matches plan 3, except node 3 runs ahead while still on plan 1,
      where its quota of 2 keeps it silent - so reaching it needs an
      invalidation across two pattern changes and two hops.
```

That was its baseline on the pre-correction spec: 9,793,928 distinct states,
~15 minutes. It does not finish on the current one.

**The state space is driven by total arrivals and tree depth, not by node count
alone** — each `arrive()` is its own action, and each level of depth adds another
layer of forwarding interleavings. The jump from `MCLarge` (6 nodes, 11 arrivals,
1.56M states) to `MCSeven` (7 nodes, 12 arrivals, >28M and unfinished) is an
order of magnitude for one extra node and one extra arrival. `MCBig3`, at 16
arrivals, is far out of reach. If you want to scale past 6 nodes you need a state
constraint or symmetry reduction, and both need care not to hide the very
interleavings that matter.

### Mutation battery

The full battery on the **final spec** — every mutation paired with a scenario
that can actually catch it, which took three rounds of re-pairing to get right
(see the masking note below; two "NOT CAUGHT" verdicts were mis-pairings, found
by asking what structure the mutation strands and building it).

| Mutation | Scenario | Caught by |
|---|---|---|
| no deferral: install a new plan before being invalidated | **`MCPark`** | `ReachableWhileHolding` |
| forget the child list *before* forwarding the invalidation | **`MCPark`** | `ReachableWhileHolding` |
| accept stale reports (running total may go down) | `MCDouble` | `TriggerCorrect` |
| case 3 sends its count but no flush signal | `MCStale` | deadlock |
| no eager flush on over-arrival | `MCOver` | deadlock |
| no pinning: reports follow the current plan | `MCDouble` | `NoOverCount` |
| no live guard (rule 10.2) | `MCStrand2` | `ReachableWhileHolding` |
| no install guard (rule 10.3) | `MCStrand2` | `ReachableWhileHolding` |
| no planless-outsider (rule 10.4) | `MCStale` | deadlock |
| no retroactive case 3 (rule 10.4) | `MCStale` | deadlock |
| case 3 loses priority to the flushing flag (rule 10.4) | `MCStale` | deadlock |
| a plan install clears flush (rule 5) | `MCStale` | deadlock |
| no stale-edge forwarding (rule 10.5) | `MCDouble`, `MCStrand`, `MCStrand2` | deadlock |
| flush only the switch generation at invalidate | — | **not caught — subsumed, see rule 6** |

Why `MCPark` exists: the deferral's real job is **routing the invalidation down
the old tree** — a node that installs early forwards it down its *new* kid
list, so old-only descendants never receive it. Only a descendant holding
*below quota* exposes that; every earlier scenario's descendants met their
quotas, which is why MCStrand2 could not catch it.

Alteration rules (rules 8/9), re-verified on the final spec — all caught on
`MCAlter`: ts-arrival counted without its alteration (`TriggerCorrect`),
ts-arrivals aggregating through the tree (`TriggerCorrect`), no eager flush on
alter (deadlock), no reserved-arrival guard (`TriggerCorrect`).

> **Two lessons from getting this wrong.**
>
> **Masking.** "No eager flush on over-arrival" comes back NOT CAUGHT on any
> scenario containing an outsider or a run-ahead — their case-3 signals make
> the owner flush anyway, covering for the missing one (`MCSeven`, `MCChain`,
> and later `MCOverSwitch` all mask it). It is caught on `MCOver`, which has no
> outsider for exactly this reason. **A battery reporting "all caught" proves
> nothing unless each rule has a scenario that could have caught it** — and a
> NOT CAUGHT is a mis-pairing until the structure the mutation strands has been
> deliberately built and still fails to catch it.
>
> **Structure.** The child-wait was recorded as verified when no scenario in the
> suite contained the structure it supposedly protected. Adding `MCChain` and
> `MCDeepSwitch` showed it was never a rule at all.

And on `MCAlter`, all caught:

| Mutation | Caught by |
|---|---|
| owner counts a timestamped arrival without applying its alteration | `TriggerCorrect` |
| timestamped arrivals aggregate through the tree (timestamp erased) | `TriggerCorrect` |
| altering node does not enter eager flush (stops relaying) | deadlock |
| no reserved-arrival contract guard | `TriggerCorrect` |

The alteration extension was **conservative** when it landed: with
`AlterOps = {}` all four then-existing scenarios reproduced their exact prior
counts (`MCSeven` 9,793,928, `MCLarge` 338,390, `MCDeviate` 73,168, `MCOver`
3,784). Those figures predate the rule-1 correction, which changed every count —
see the table above for current values.

**Methodology note for whoever maintains this:** a scenario earns trust only when
it kills every mutation, and a property earns trust only when validated in *both*
directions — silent on a known-healthy spec, firing on a known-broken one. Six
separate verification-apparatus bugs were found in this effort, several of which
produced false "verified" results. An uncaught mutation is a scenario defect
until proven otherwise, not a protocol result.

---

## 8. Fidelity notes — where the model is not the implementation

Read this section carefully. These are the gaps between `BarrierArrive.tla` and a
real C++ implementation, and each one is an implementation obligation that the
model does not force on you.

### 8.1 Report addressing is a global lookup in the model

`ParentOf(n)` at `BarrierArrive.tla:153-155` finds the parent by searching all nodes
for one that lists `n` as a child, falling back to `Owner`. That is a modelling
convenience — a real node stores its parent locally in its plan record.

Two consequences:

- The fallback-to-`Owner` branch corresponds to "I am in no plan, so I report
  direct to the owner" (rule 3). Implementations get this from the plan record.
- During a switch, two nodes can transiently both list `n` as a child (an old
  parent that has not switched and a new parent that has). The model's `CHOOSE`
  picks one arbitrarily. A real implementation sends to whatever its own plan
  record says.

### 8.2 A receiver must accept reports from nodes it does not list as children

The model handles this — `childAcc[m.to][m.from][m.gen]` is defined over *all*
nodes, so `RecvReport` accepts from anyone. **The implementation must do the same
explicitly.** A node whose parent has already switched plans and dropped it will
still send to that old parent; if the old parent rejects reports from unknown
children, the count is lost. This is easy to get wrong because it looks like a
defensive check worth adding.

### 8.3 `ReportWith` at `BarrierArrive.tla:147-149` is dead code

Defined but never referenced — superseded by `Send` at `:157-159`. It was left in
place so the verified artifact stays byte-identical to what was checked. Ignore
it; do not implement it.

### 8.4 A set-of-messages model deduplicates silently

Not a live issue — recorded because it cost a spurious deadlock trace and the
shape recurs. `msgs` is a **set**, so any message with no varying field collapses
with an identical one. An earlier draft had timestamped arrivals reported
individually, and two arrivals from the same node for the same generation merged
into one element; the owner undercounted and TLC reported a deadlock that looked
like a protocol defect.

Making those reports cumulative fixed it as a side effect, because cumulative
values differ. **If you add a message kind to either spec, check it carries
something that varies per instance.**

### 8.5 The model constrains the application contract, and that can hide a rule

`Alter` is guarded by `unissued[a.node][a.gen] > 0` — the reserved-arrival
requirement. An earlier version *also* forbade a node from arriving while it held
an unissued alteration. That is stronger than the API, which permits arriving on
the pre-alteration handle, and it silently **subsumed** the reserved-arrival
guard, so removing that guard came back NOT CAUGHT.

Generalised: an over-constrained model does not fail loudly, it makes a real rule
look unnecessary. If a mutation you expect to be caught is not, suspect the
scenario before concluding the rule is redundant.

### 8.6 Negative alterations are not exercised

`AlterOps` carries an arbitrary integer delta, but the scenario uses `+1`. The
terminal-negative case of `event.h:293-297` — a negative alteration driving the
remaining count to exactly zero, which needs no arrival on the returned handle —
has **no coverage**. It is the one branch of the contract with a different safety
argument, so it deserves its own scenario before that path is implemented.

### 8.7 The plan is a given, not a computed thing — see §11

The model takes `Plans` and `PlanStart` as constants. **How the owner constructs a
new plan from observed arrivals is not modelled and not verified.** The design
intent (from the earlier discussion) is: during eager-flush mode every report
carries where the arrival occurred, the owner aggregates those into a new plan,
distributes it, and then forgets the aggregation structure — it must not persist,
or O(N) barriers with the same pattern cost O(N²) memory.

**§11 specifies that construction.** What the model verifies is the other half:
whatever plan the owner produces, the switch to it is safe. Construction itself
needs no model — it is a local computation at the owner — but it does need the
wire formats and the memory bound in §11.

### 8.8 Retained state and cleanup

`BoundedRetention` only checks the parked-plan slot. The broader constraint,
agreed with the user: a node may forget retained per-generation state once every
possible impacted generation has triggered, and `destroy_barrier` is an
acceptable place to require full cleanup. Unbounded lifetime is not acceptable.

---

## 9. Reproducing the results

TLA+ tools are at `tools/tla2tools.jar` (TLA+ 2.19, gitignored). Java is not on
the default PATH on this machine:

```sh
cd tla/barrier
java -Xmx6g -XX:+UseParallelGC \
    -Djava.io.tmpdir=jtmp -cp tools/tla2tools.jar tlc2.TLC \
    -workers 8 -config DeepSwitch.cfg MCDeepSwitch.tla
```

Expect `Model checking completed. No error has been found.` and 311,772 distinct
states. Swap for any other pair from the table in §7 — but **not**
`Seven.cfg`/`MCSeven.tla`, which no longer completes on this spec.

To classify a run's verdict, grep for the verdict line rather than tailing
output — TLC prints violations as `Error: Invariant <name> is violated.` A
harness that only looks for the bare string `Invariant ... is violated` will
miss the `Error:` prefix, which is exactly the reporting bug that made the first
battery report an unhelpful fallback verdict for all seven mutations.

### Other files in this directory

- `BarrierArrival.tla` + `MCPlanned.tla` + `DeadlockNoFlush.cfg` — the model of
  the **old, abandoned** protocol, in which TLC proved a terminal deadlock.
  Kept as the record of why the redesign happened. Its liveness-property
  results were later shown to be unreliable (the property never fired); the
  deadlock proof is the citable result.
- `MCSeven.tla`, `MCLarge.tla`, `MCLate.tla` — early scenarios that outgrew
  practical checking as the spec gained rules; retained as records, superseded
  by the minimal suite (see README.md).

---

## 10. Implementation checklist

Derived from the rules above; each maps to a verified property.

- [ ] Reports carry **cumulative** subtree totals; receivers replace, never add.
- [ ] Reject reports that do not strictly increase the stored per-child value.
- [ ] Accept reports from nodes not in the current child list (§8.2).
- [ ] A relay forwards when local count == quota. **No child-wait** — do not add
      one (rule 1).
- [ ] A plan install **never clears flush**; a flushed generation stays eager
      until it triggers (rule 5 — a caught mutation, do not "restore" clearing).
- [ ] Nodes with zero predicted arrivals are not in the tree and flush always.
- [ ] Over-arrival ⇒ set flush, report immediately, fan flush to children.
- [ ] Outsider arrival ⇒ `direct` to owner; owner flushes and fans out.
- [ ] Flush is per generation, idempotent, and reports held work on entry.
- [ ] On `newplan` while an un-invalidated member of the old plan ⇒ **park it**.
- [ ] On `invalidate`: forward to old children **first**, then flush **every**
      open generation with state, then apply any parked plan — **only if the
      parked plan's epoch exceeds the invalidation's** (rule 10.2); a dead
      parked plan is discarded, never installed or forwarded.
- [ ] A `newplan` is dead on arrival if its retirement was already witnessed:
      install only when `epoch > myEpoch` **and** `epoch > invalEpoch`
      (rule 10.3).
- [ ] A node's report target for a generation is **pinned at first touch** and
      never re-aimed by a later plan (rule 10.1).
- [ ] A report from a sender not in the current child list is **forwarded at
      once**, never held behind the receiver's quota (rule 10.5).
- [ ] A node left planless becomes an **outsider**; its post-invalidation
      arrivals fire case 3, and its pre-invalidation arrivals are covered by a
      count-free flush signal to the owner (rule 10.4). The **count follows the
      pinned edge; only the signal goes to the owner** — a cumulative value
      sent off-pin double-counts.
- [ ] Plan trees are built in ascending node id, owner at the root, so pinned
      report edges can never form a cycle (rule 10.1's assumption).
- [ ] No timers, no polling, no background sweeps — every action is caused by an
      arrival or a message receipt.
- [ ] Per-generation state is reclaimable; `destroy_barrier` frees the rest.
- [ ] An arrival carrying a causal timestamp **bypasses the tree** and reports to
      the owner on its own.
- [ ] The owner does not count a timestamped arrival until every alteration it
      witnessed has been applied (the existing `pending` map).
- [ ] Issuing an alteration puts the node into **eager flush** for every affected
      open generation — otherwise it stops relaying for its children.

- [ ] A negative alteration behaves as an arrival **and** invalidates the plan;
      a base arrival count of zero is an error.
- [ ] Every spec action is applied atomically with respect to barrier state (§12).

If you change any of these, re-run the mutation battery. A rule with no mutation
covering it is a rule nobody has actually verified.

---

## 11. The plan lifecycle

The model treats plans as constants: reports carry a bare count
(`BarrierArrive.tla:140,150,205`) and `newplan` carries only an epoch, with content
read from `Plans[epoch]`. So the spec verifies **plan switching given that plans
arrive**, and specifies neither what a plan message contains nor how one is
built. This section fills that gap. None of it is model-checked; all of it is
constrained by what the model *does* verify — that any plan is safe to adopt.

### 11.1 Gathering

An **eager-flush report carries per-node arrival counts**, not just a total: the
set of nodes on which arrivals occurred and how many occurred on each. A single
deviating arrival is one `(node, count)` pair; a node that had buffered arrivals
before entering flush mode reports all of them.

Relays merge these maps on the way up exactly as they merge counts today, with
the same cumulative-replace semantics — a relay re-sends its whole subtree map,
so a later report supersedes an earlier one and reordering stays harmless.

Cost note: an eager-flush report is O(subtree) rather than O(1) and is re-sent on
each child update. That is the deviation path, not steady state, but the node set
should use the multicast codec's encodings rather than a naive list.

### 11.2 Construction

The owner accumulates the merged maps into one structure: node → arrival count.
From that it has both the participant set and the expected count per node, which
is exactly a plan — a tree shape over that node set, plus each node's quota.

Only nodes with a non-zero count appear. That is the membership rule from §3, and
it is what stops a relay ever waiting on a child that will never speak.

### 11.3 Forgetting — the memory bound

**The aggregation structure is transient.** It exists from the start of gathering
until the new plan has been broadcast, and is then deleted. It must not persist:
O(N) barriers each retaining an O(N) participant map is the O(N²) blow-up this
design exists to avoid.

What persists is only what each node holds of its own plan record — quota,
in-plan flag, child list — which is O(fanout) per node.

### 11.4 Distribution

A `newplan` message must carry the recipient's own plan record, not just an
epoch: its quota and its child list. Epoch numbering, deferral and invalidation
are already specified and verified (§5 rules 5 and 6); only the payload is new.

---

## 12. Atomicity

**Neither this model nor `BarrierNotify.tla` represents concurrency.** Every
action in both is atomic. The implementation is not: multiple application tasks
may invoke barrier APIs in parallel on the same node, and active messages from
remote nodes are delivered asynchronously on handler threads at the same time.

> **Every action in the specification must be applied atomically with respect to
> that barrier's state.**

"Action" means one whole numbered ACTION in the module — its guard evaluation and
every primed-variable update together. Splitting one action across two critical
sections is a defect even when each half is individually locked, because the
model never exposes the intermediate state and so no invariant constrains it.

Where this bites hardest — each of these reads several variables and writes
several more:

| Action | Reads and writes that must not tear |
|---|---|
| `Arrive` (`:191`) | `unissued`, `localTotal`, `flushing`, quota and child check, then the send |
| `RecvReport` (`:240`) | staleness test, `childAcc`, recomputed subtree total, forward decision, `reportedUp` |
| `RecvInvalidate` (`:332`) | child list read **before** it is replaced, every open generation flushed, parked plan applied |
| `Trigger` (`:304`) | count comparison and the plan switch that follows it |
| `Alter` (`:403`) | reserved-arrival guard, `unissued`, `myTs`, flush state, sends |

One constraint pulls the other way and must be honoured: **`has_triggered` stays
lock-free.** It may do a single atomic load of the watermark and nothing else. In
particular the consultation signal for departure hysteresis must not be on that
path — see the notification document's rule 8.

The previous implementation attempt did not fail on protocol logic; it failed on
a deferred handler path that reached a null runtime. A correct protocol applied
non-atomically is still broken, and neither model can catch it.
