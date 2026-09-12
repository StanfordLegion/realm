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

# Formal Specifications for Realm's Scalable Barriers

TLA+ models, model-checking scenarios, and the design documents for the
scalable barrier arrival and notification protocols implemented in
`src/realm/barrier_impl.{h,cc}`.

**Start with the documents — they are the normative record:**

| Document | What |
|---|---|
| [`ARRIVAL_PROTOCOL.md`](ARRIVAL_PROTOCOL.md) | The arrival/aggregation protocol: 10 rules, each a caught mutation; safety properties; fidelity notes; implementation checklist |
| [`NOTIFICATION_PROTOCOL.md`](NOTIFICATION_PROTOCOL.md) | The subscription/notification protocol: 8 rules; same structure |
| [`STATE_AND_LOCKING.md`](STATE_AND_LOCKING.md) | `BarrierImpl` member layout, one critical section per spec action, the decisions record (§0 supersedes the body where they disagree) |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | How the implementation was staged; the settled design decisions (D1–D10, Q1–Q10); known deferred items |
| [`SCALE_TEST_PLAN.md`](SCALE_TEST_PLAN.md) | The 2–128 node testing program; drives `tests/barrier_scale_test.cc` |

Where prose and spec disagree, **the spec wins** — it is what was model-checked.

## The specifications

- **`BarrierArrive.tla`** — the arrival protocol: learned plans, eager flush,
  plan switching with deferral, `alter_arrival_count`, and the pinned-edge
  rules (rule 10). Checked with six safety invariants plus deadlock detection.
- **`BarrierNotify.tla`** — the notification protocol: subscriber set,
  version-gated membership, delta notifications with gap-pull, departure.

## Running

TLC is not vendored. Fetch the TLA+ tools (2.19 or later) into `tools/`:

    curl -Lo tools/tla2tools.jar \
      https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

Check one scenario (each `MC*.tla` pairs with the like-named `.cfg`):

    java -Xmx8g -XX:+UseParallelGC -Djava.io.tmpdir=jtmp \
      -cp tools/tla2tools.jar tlc2.TLC -workers 8 -config Stale.cfg MCStale.tla

Expect `No error has been found`. Delete `jtmp/` and `states/` afterwards —
large runs spill tens of GB.

**If you change a protocol rule, run the battery:**

    python3 battery.py all        # ~15-30 min; suites: arrival, alter, notify

Every rule in both documents is paired with a mutation and a scenario chosen
to catch exactly it. The battery fails loudly on a pattern that no longer
matches the spec — do not "fix" that by loosening the pattern; it exists
because an unanchored replacement once silently tested the wrong site.

## The scenarios

The load-bearing scenarios are the **minimal** ones, distilled from failing
traces; the big ones exist for breadth. A scenario is trusted only because it
kills mutations — see the battery table in each document.

| Scenario | Size | What it uniquely covers |
|---|---|---|
| `MCDouble` | 4 nodes | the re-parent double-count (`NoOverCount`); pinning; stale-edge forwarding |
| `MCStale` | 4 nodes | run-ahead outsider + over-predicting quota; found five bugs in candidate fixes |
| `MCStrand` | 4 nodes | single-switch control (one switch is *not* enough to strand) |
| `MCStrand2` | 4 nodes | two successive switches + run-ahead: the dead-plan guards |
| `MCPark` | 4 nodes | why the deferral exists (invalidation routing down the old tree) |
| `MCOver` | 5 nodes | over-arrival with no outsider to mask it |
| `MCOverSwitch` | 3 nodes | over-arrival × plan switch |
| `MCDeviate` | 5 nodes | the outsider (case 3) |
| `MCChain` | 5 nodes | a relay below a relay |
| `MCAlter` | 4 nodes | `alter_arrival_count`, altering node is a relay |
| `MCGateOver` | 4 nodes | quota gate: over/under pair straddling a gated relay (the value-exceeds valve) |
| `MCGateAhead` | 4 nodes | quota gate: run-ahead mixed pins routing counts around the new relay (the non-kid valve) |
| `MCGateMove` | 4 nodes | quota gate: a value accepted legally under plan k−1 made deviant by plan k (the switch audit) |
| `MCGateDemote` | 4 nodes | quota gate: demoted owner-child + flush fan dying at a planless relay (audit orphan branch; install re-fan) |
| `MCGatePark` | 4 nodes | quota gate: parked install — catches the parked re-fan AND the stale-edge signal via its two interleavings |
| `MCGateJump` | 3 nodes | quota gate: cumulative replace jumps PAST the quota (why `>=`, never `=`) |
| `MCLate` | 5 nodes | plan governs only generations after the deviation; outgrew the ungated spec, **completes again under quota gating** (~22M distinct states, checkpointed) |
| `MCDeepSwitch` | 5 nodes | three plans over a relay chain — found the `NoOverCount` violation that forced rule 10; **no longer completes** (bounded partial runs only, no violation) |
| `MCNotify` | 3 nodes | notification baseline (83M states) |
| `MCNotifySmall` | 2 nodes | small enough to *clear* a guard-removal mutation (negative controls) |

Historical, kept as records (do not extend): `MCSeven`, `MCLarge`
— early scenarios that outgrew practical checking as the spec gained rules; `BarrierArrival.tla` + `MCPlanned.tla` +
`DeadlockNoFlush.cfg` — the model of the *previous* barrier design, in which
TLC proved a terminal deadlock (all arrivals issued, network empty, no action
enabled anywhere); that proof is why the redesign happened. A note on the old
model's other configs: its `EventuallyTriggers` liveness property was later
shown never to have fired at all — the deadlock detector did the real work —
so the `Liveness*.cfg` results should not be cited.

## Hard-won methodology (short form; details in the documents)

- A green run proves nothing about which paths executed. A rule is verified
  only when a scenario **catches its removal**, and a scenario is trusted only
  when it kills mutations. A NOT-CAUGHT is a mis-pairing until the structure
  the mutation strands has been deliberately built and still fails to catch.
- Validate every property in both directions: silent on a known-healthy spec,
  firing on a known-broken one.
- When a big scenario fails, distil the trace into a minimal scenario and
  iterate there. `MCStale` (~8k states) found bugs that 20M-state scenarios
  never reached.
- Guard-removal mutations *enlarge* the state space; size negative controls
  for that (`MCNotifySmall` exists for exactly this reason).
- Mutate copies in scratch directories, never the canonical files; assert
  every mutation pattern matches exactly once. `battery.py` encodes both.
