<!--
Copyright 2026 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Scalable Barriers — Implementation Plan

Plan for the C++ work that enacts the two verified designs:

- [`ARRIVAL_PROTOCOL.md`](ARRIVAL_PROTOCOL.md) — `BarrierArrive.tla`, 9 rules
- [`NOTIFICATION_PROTOCOL.md`](NOTIFICATION_PROTOCOL.md) — `BarrierNotify.tla`, 8 rules

Those documents are normative. This one is about sequencing, parallelism and
where the risk sits.

---

## 0. What is and isn't settled

**Settled and verified** — every rule in both protocols has a mutation that a
both-directions-validated check catches.

**Settled by decision, not verified** — recorded in the documents, no model:

| Item | Where |
|---|---|
| Plan gathering / construction / forgetting / distribution | arrival §11 |
| Negative alterations invalidate the plan; zero base count is an error | arrival rule 9 |
| Atomicity requirement | arrival §12 |
| Poison retention bounded by barrier lifetime | notification §8.5 |
| K, J, shrink cost test | notification rule 8, §8.2, §8.4 |

**Settled by owner decision** (recorded here; `STATE_AND_LOCKING.md` §0 has the
same list and supersedes anything in its body that contradicts them):

| # | Decision |
|---|---|
| D1 | **No separate reduction-state object.** All state lives directly in `BarrierImpl`. Reduction barriers still fall back to the existing path — eager messages straight to the owner — but that is a code path, not a separate object. |
| D2 | A non-owner never guesses. With a plan it may buffer; without one it eagerly flushes down the tree, or direct to the owner if no tree exists. |
| D3 | Migration is removed. **The lock is still required** in most of the paths it touched, to race correctly against notifications. |
| D4 | **One counter direction, one semantics.** Every accumulator is cumulative and replace-if-higher, counting up. No incrementing counter alongside; no count-down convention from the legacy path. |
| D5 | `alter_arrival_count` is persistent. |
| D8 | **Store the subscriber set directly as `MulticastTargetSet`** — it already has `add`/`remove`/`contains`/`size`/`num_ranges`. No `NodeSet` staging, no cached conversion. Mutation cost is irrelevant because the set changes rarely by design (that is what K=8 hysteresis is for). |
| D9 | The plan-gathering structure may be a temporary `BarrierImpl` member, but **must not persist across generations**. |
| D10 | Handlers that take locks cannot be inline. |
| Q2 | `destroy_barrier` stays a no-op for now. |
| Q3 | Reuse `GenEventImpl`'s poison structure; past the cap, fatal log then `std::abort`. |
| Q4 | A poisoned arrival precondition poisons that generation. |
| Q5 | Notifications carry a single triggered-generation number — trigger-in-order makes anything wider unnecessary. Poison for a new subscriber rides the subscribe reply. **Do not widen the range.** |
| Q6 | Payload sizing is not a concern: use active-message fragmentation. (At 4,096 nodes an eager-flush report is ~9–16 KB against a ~1 MB block.) |
| Q7 | Exactly one mutex per `BarrierImpl` per node. |
| Q8 | **Drop `BarrierCommunicator`.** Use the active message interface directly. |
| Q9 | **Do not touch `external_wait`.** Unrelated to these protocols. |
| Q10 | Defer barrier tests — a new test will be synthesised later, and it needs a few thousand nodes to be meaningful. |

**A pre-existing bug to fix, not just implement around:** `alter_arrival_count`
is documented (`event.h:271`) and expected by Legion to be **persistent** across
all future generations. The current code applies the delta to a single
generation — it lands in `Generation::unguarded_delta` and never touches
`base_arrival_count`. No test in the repo calls the API. **C6 fixes this**, and
`BarrierArrive.tla` already models the correct (persistent) semantics.

**Known gap, deliberately carried:** the terminal-negative branch of
`alter_arrival_count` (arrival §8.6) has no coverage and a *different* safety
argument from the positive case. Either model it before implementing that branch
or gate it behind a fatal-error check.

**Scope:** `barrier_impl.{h,cc,inl}` is ~1,900 lines. Reduction barriers stay on
the existing non-scalable path. Migration is removed (34 references, all three
files).

---

## 1. Why this is mostly sequential

The tempting shape is "arrival protocol and notification protocol in parallel."
That's wrong here: both live in `BarrierImpl`, both touch the same per-generation
state, and both need the same locking discipline. Parallel agents on one class
produce conflicts that cost more than the concurrency saves.

What *does* parallelise is **reading** and **checking** — scouting the existing
code, and auditing finished code rule-by-rule against the specs. The plan below
fans out on those and keeps the writing sequential.

Worktree isolation only pays where agents genuinely write disjoint files, which
here is essentially only the test phase.

---

## 2. Phases

### Phase A — Scout (parallel, 4 agents, read-only)

Independent questions over existing code; no writes, so no conflicts.

1. **Deletion map.** Every site touched by migration removal; every use of
   `remote_trigger_gens`, `held_triggers`, `previous_gen` gap-buffering. Which
   are reachable only from the reduction path (keep) vs. the general path (cut).
2. **Locking map.** What `BarrierImpl::mutex` currently protects, which methods
   assume it held, where handlers run (inline vs deferred), and every path
   `has_triggered` touches — that must end up lock-free.
3. **Message plumbing.** How `BarrierAdjustMessage` / `BarrierTriggerMessage` /
   `BarrierSubscribeMessage` are registered, fragmented and dispatched; what a
   new variable-length payload costs.
4. **Test surface.** `tests/unit_tests/barrier_test.cc`, `barrier_arrivals.cc`,
   `barrier_reduce.cc`, `event_test.cc` — which assert on behaviour that is about
   to change, and which are reusable as regression anchors.

**Output:** one merged findings document. Nothing is written to the tree.

### Phase B — State layout and locking discipline (single, sequential)

The design decision everything else depends on. Produces the new `BarrierImpl`
member layout for both protocols, and writes down the critical sections —
explicitly mapping each spec action to one, per arrival §12.

Gate: no implementation starts until this is reviewed. Getting it wrong is what
makes the later phases conflict.

### Phase C — Implementation (sequential stages)

Ordered so each stage is testable before the next depends on it.

| Stage | Content | Spec |
|---|---|---|
| C1 | Strip migration; strip `remote_trigger_gens`, gap-buffering from the general path; route reduction barriers to the legacy path | arrival §3, notification §3 |
| C2 | Plan record + epochs; cumulative reports with staleness; steady-state quota + child-wait | arrival rules 1, 7 |
| C3 | Eager flush: over-arrival, outsider-direct, per-generation idempotent flush | arrival rules 2, 3, 4 |
| C4 | Plan switch: invalidate, newplan, deferral, forward-before-forget, flush-all-generations | arrival rules 5, 6 |
| C5 | Plan lifecycle: `(node,count)` report payload, owner aggregation, tree construction, **forgetting**, plan-record distribution | arrival §11 |
| C6 | `alter_arrival_count`: **fix persistence first**, then timestamped arrivals bypass the tree (cumulative `val`), exact-set `pending` gate, eager flush on alter, negative alterations | arrival rules 8, 9 |
| C7 | Notification: subscriber set + `setVer`, delta notify with gap-pull, subscribe/reply with `lk` and merge, membership publication | notification rules 1–7 |
| C8 | Departure: consultation signal, `K = 8` adaptive, unicast with `J` jitter, owner cost test and hint | notification rule 8 |

C5 is the highest-risk stage — it is the only one with no model behind it, and
the memory bound (arrival §11.3) is easy to violate by accident.

### Phase D — Rule audit (parallel, ~17 agents, read-only)

One agent per rule: 9 arrival + 8 notification. Each gets the rule text, its spec
line references, and the implementation, and answers one question — *does the
code do exactly this, including the parts the document calls out as easy to get
wrong?*

This is the phase that actually benefits from fan-out: the rules are independent,
the failure mode is a rule quietly not implemented, and a reviewer holding one
rule finds that better than a reviewer holding seventeen.

Feed each agent the specific traps already documented, e.g.:

- a receiver must accept reports from nodes not in its child list (§8.2)
- forward the invalidation *before* dropping the child list (rule 6)
- `seq` is a spec artifact and must **not** appear in the wire format (§8.4)
- membership updates are version-gated on **both** notify and reply paths
- poison deltas are **merged**, never substituted

### Phase E — Adversarial verify (parallel, 3 per finding)

Every Phase D finding goes to independent skeptics prompted to **refute** it,
defaulting to refuted when uncertain. Majority refutes → dropped.

Non-optional given the history: in this project roughly as many defects appeared
in the *verification apparatus* as in the protocols. An unverified finding is a
hypothesis.

### Phase F — Tests (parallel, worktree-isolated)

The one phase where agents write genuinely disjoint files.

- **Per-rule unit tests**, one file per protocol area, asserting the observable
  consequence of each rule. Note Q10: full validation needs thousands of nodes
  and will be a separate exercise; this phase is the local-testable subset.
- **`alter_arrival_count` coverage**, which does not exist anywhere today — the
  reason its persistence bug survived.
- **Concurrency tests** for arrival §12 — parallel `arrive`/`wait`/`alter` from
  several tasks against one barrier, plus AM delivery, under TSAN.
- **Deviation tests** driving the exact scenarios the specs encode: over-arrival,
  outsider, two successive pattern changes, a node running two plans behind, a
  far-future waiter, subscribe racing a trigger.
- **Scale/regression** against the existing barrier tests from Phase A4.

TSAN is not optional here. Atomicity is the one requirement no model checks, and
it is what broke the previous attempt.

---

## 3. Sizing and gates

| Phase | Shape | Gate before proceeding |
|---|---|---|
| A | 4 parallel, read-only | findings merged |
| B | 1 sequential | **human review** — everything downstream depends on it |
| C1–C8 | sequential | each stage builds and passes existing tests |
| D | ~17 parallel, read-only | — |
| E | 3 per finding, parallel | only confirmed findings reach a fix list |
| F | parallel, worktrees | clean TSAN run |

Phase B is the one hard gate. Phase C5 is the one to slow down on.

---

## 4. Risks

**The previous attempt failed here, and it is worth naming how.** It was not
protocol logic: it was a test harness using a 10µs `TimeLimit` that pushed work
onto a deferred path where `get_runtime()` was null. Three "fixes" were made and
reverted before the real cause was found. Hence Phase F's TSAN gate and Phase
B's explicit critical sections.

Other risks, in order:

1. **C5 memory bound.** The aggregation structure must be deleted after
   broadcast. O(N) barriers × O(N) map is the blow-up the whole design exists to
   avoid, and nothing will fail loudly if it leaks — it just scales badly.
2. **Eager-flush report size.** O(subtree) payloads re-sent per child update.
   Use the multicast codec's encodings; measure before assuming it is fine.
3. **Silent partial implementation of a rule.** Exactly what Phase D exists for.
4. **`has_triggered` regressing to take a lock.** Cheap to do by accident while
   adding the consultation signal; add an explicit test.

---

## 5. What to instrument from day one

The specs deliberately abstract these as nondeterminism, so only measurement can
tune them:

- leave→rejoin cycles per barrier (validates `K = 8`; near zero is the
  expectation)
- subscribe fan-in per barrier (the last unaggregated O(N) path; escape hatch is
  notification §"growth")
- eager-flush episodes and their report sizes
- plan rebuild frequency, and peak size of the aggregation structure

---

## 6. Re-verification triggers

Change the protocol, re-run the battery. Concretely:

```sh
cd tla/barrier
# arrival:      Seven.cfg/MCSeven (~15 min), Large, Deviate, Over, Alter
# notification: Notify.cfg/MCNotify (~18 min)
```

Two operational warnings, both learned the hard way and repeated in the
documents: a full run consumes tens of GB of scratch — delete `jtmp/` and
`states/` afterwards — and mutations must be applied to a **copy** in a scratch
directory, never to the canonical `.tla` in place.
