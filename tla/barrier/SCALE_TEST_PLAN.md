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

# Scalable Barriers — Scale Testing Program (2–128 nodes)

Companion to [`ARRIVAL_PROTOCOL.md`](ARRIVAL_PROTOCOL.md) and
[`NOTIFICATION_PROTOCOL.md`](NOTIFICATION_PROTOCOL.md). The specs verify the
protocol against every message interleaving at 3–5 nodes; this program tests
what the specs cannot: real transports, real concurrency, real scale, and the
implementation's fidelity to the verified rules.

**The governing principle is inherited from the mutation battery: a green run
proves nothing about which paths executed.** Every deviation case in this
program is paired with a counter that must move, and a run reports which
counters *never* moved so coverage holes are visible instead of silent. This is
the masking lesson applied at runtime.

---

## 0. Preflight gates — before any cluster time

These run on one workstation and gate everything below. Skipping them means
debugging protocol-shaped phantoms at 128 nodes.

| Gate | What | Why | Status |
|---|---|---|---|
| G1 | **2–4 rank smoke over a real transport** (MPI and UCX builds), steady-state + one forced deviation per case | The new wire formats — plan subtree payloads, flush maps, `BarrierFlushMessage` to owner, delta notifications — had never been serialized across a transport | **Steady-state half DONE** (2026-08-09): open-mpi 5.0.9 on the dev machine, 12/12 runs of `barrier_arrivals`+`barrier_reduce` at 2 and 4 ranks, exit clean. **Deviation half needs the driver** — counters and INFO logs confirm the deviation machinery never executes under the existing tests. |
| G2 | **TSAN soak, single rank, many threads**: concurrent `arrive`/`wait`/`alter_arrival_count`/`external_wait` from 8+ tasks against shared barriers, thousands of generations | Atomicity (ARRIVAL §12) is the one requirement no model checks, and a race at scale presents as a protocol bug. The previous implementation attempt died exactly here. | **Smoke DONE**: TSAN build at `/tmp/claude/btsan`, 8 runs of both tests, zero reports. Full soak needs the driver's concurrency mode. |
| G3 | **Debug-build ladder pass at 2–8 ranks** with `REALM_BARRIER_ASSERT_LOCKED` active and counters asserted | Catches lock-contract violations and counter regressions before they get expensive. | **Driver DONE** (`tests/barrier_scale_test.cc`, registered in ctest): 9 phases, 9/9 PASS at 1, 2 and 4 ranks; TSAN run clean; the 8 rule-10 counters are wired and the per-phase dump works. |

**STRUCTURAL FINDING (2026-08-09), changes the ladder:** the plan tree radix is
8, so below ~10 ranks every plan is a flat owner→all tree — **no relays exist,
and therefore parks, dead-plan discards, stale-edge forwards and pin conflicts
are structurally impossible, not just improbable**. A 6-seed × 128-generation
churn campaign at 4 ranks produced 281 plan rebuilds and 395 retroactive
flushes with all four race-window counters at zero, which is consistent.
Consequences:
- ~~run at ≥10 ranks~~ **DONE instead: `-ll:barrier_plan_radix N` added**
  (default 8 unchanged; distinct from `-ll:barrier_radix`, which is the
  MULTICAST fan-out radix consumed by activemsg.cc). Radix 2 puts relay trees
  at 4 ranks.

**Race-window campaign results with radix 2** (2026-08-09, this machine):
- 4 ranks, all 9 phases: `plans_parked=11`, all applied correctly, 9/9 PASS —
  **the deferral race ran on hardware for the first time**.
- 6 ranks × 8 seeds × 128 gens (churn+runahead+chaos): `plans_parked=21`,
  `stale_edge_forwards=853`, `retro_flushes_sent=489`, `gap_pulls=1`,
  zero failures.
- Still never moved: `dead_plans_discarded` (needs a THIRD switch racing a
  parked plan — cross-pair message reordering loopback rarely produces) and
  `pin_conflicts_avoided`. These two are the remaining cluster-tier targets;
  everything else in the exercise contract is now demonstrated at ≤6 ranks.

First 4-rank exercise table (9 phases × 64 gens): plan_rebuilds=76,
flush_episodes=860, report_edges_pinned=1317, retro_flushes_sent=205,
gap_pulls=9, departs→shrinks→removals=3→2→2, adaptive K doubled (churn_backoffs=2).
Everything moved except the four relay-dependent counters above.

**Build facts learned running G1/G2** (bake into every test build):
- `-DREALM_LOG_LEVEL=INFO` is required to see protocol logs at all — the
  default compile-time floor is WARNING, and `-level barrier=N` silently shows
  nothing below it (a one-line warning is the only hint).
- `dump_counters` fires only on `destroy_barrier`/slot recycle. The existing
  tests never destroy barriers, so counters are invisible. The driver must
  destroy its barriers (or a shutdown-time dump must be added) — otherwise the
  exercise-proof contract of §1 cannot be checked.

Also before the PR: decide the **`alter_arrival_count` flag-day question**. Its
semantics changed from (buggy) per-generation to (documented) persistent. No
test in the repo calls it, so nothing will catch a Legion app that compensated
for the old behaviour. Recommend: grep Legion for call sites and eyeball each
before merging, and say so in the PR description.

---

## 1. Instrumentation: the exercise-proof contract

`BarrierCounters` exists and dumps at destroy. The following **must be added**
so every rule-10 case is observable (each is one increment on a path that
already holds the mutex):

| New counter | Increments when | Proves exercised |
|---|---|---|
| `plans_parked` | RecvNewPlan parks (deferral) | rule 5 deferral |
| `parked_plans_applied` | invalidation applies a live parked plan | deferral resolution |
| `dead_plans_discarded` | live guard or install guard fires | rule 10.2/10.3 — the race windows |
| `retro_flushes_sent` | retroactive case-3 flush to owner | rule 10.4 |
| `stale_edge_forwards` | receiver forwards a non-child report | rule 10.5 |
| `report_edges_pinned` / `pin_conflicts_avoided` | first-report pin; pinned target ≠ current parent at report time | rule 10.1 |
| `gap_pulls` (notification) | delta discarded, pull sent | notification rule 4 |
| `plan_rebuilds_declined` | completeness gate said no | §11.5 — high ratio vs `plan_rebuilds` under P_STEP = cannot re-learn |
| `gathering_declared` | owner declared a gathering generation | §11.5 convergence |
| `identical_plans_skipped` | rebuild reproduced the current plan | §11.5; ≥1 per barrier at startup is the normal signature |

Add a per-rank end-of-run dump (aggregate across barriers) and a tiny reducer
script that merges all ranks' dumps into one table with zero-counters
highlighted. **The run artifact is that table, not the exit code.**

---

## 2. The oracle: what "correct" means at runtime

Each test wraps barriers with independent verification (side-channel MPI
collectives, not Realm barriers):

1. **No early trigger** (the safety property, `TriggerCorrect`/`NoOverCount` at
   runtime): each rank counts arrivals it issued per generation; after a
   waiter wakes for generation g, an `MPI_Allreduce` confirms the global sum
   for g had reached the expected count *before* the wake was observed. Run
   the check on a sampled subset of generations (e.g. 1 in 64) to keep the
   oracle from serialising the test.
2. **Exactly-once, in-order triggering** per rank: generation g wakes after
   g−1, never twice.
3. **No hang** (the liveness property): every phase has a global timeout
   generous enough for the slowest tier (minutes, not seconds — this is a
   deadlock detector, not a performance gate). A timeout dumps all ranks'
   counters and current barrier state before aborting.
4. **Poison fidelity**: a rank that observes generation g poisoned must never
   have observed it triggered clean, and vice versa; poison must be observed
   by *every* waiter of that generation.
5. **Memory bound** (the O(N²) guard, D9/§11.3): RSS per rank sampled per
   phase; `agg_peak_entries` and `flush_report_bytes_max` reported. Growth
   across a soak run fails the test.

---

## 3. The driver phases — one per protocol case

One parameterised driver (`barrier_scale_test`) executes a script of phases.
Each phase runs G generations (default 64) of one pattern, then rotates. Every
rank knows the full script (seeded), so expected counts are computable locally.

| Phase | Pattern | Protocol case | Counter that must move |
|---|---|---|---|
| P1 steady | same participants + counts every generation | rule 1 steady state, plan reuse | `plan_rebuilds` stays **flat** after first |
| P2 grow | participant set grows by one rank every k generations | outsider → case 3 → plan rebuild | `flush_episodes`, `plan_rebuilds`, subscribe fan-in |
| P3 shrink | set shrinks by one rank every k generations | under-arrival absorbed by conservation; departure hysteresis (K=8) on the notify side | `departs_sent`, `shrinks_applied`, `nodes_removed` |
| P4 over | one rank doubles its arrivals for one generation, rotating | rule 2 over-arrival | `flush_episodes` on non-owner ranks |
| P5 outsider | one silent rank arrives for one generation, rotating | rule 3 | `retro_flushes_sent` or direct-flagged reports |
| P6 run-ahead | ranks arrive on advanced handles up to R generations ahead (staggered by rank id), R up to 8 | pinning, straddling generations, retroactive case 3 | `report_edges_pinned`, `retro_flushes_sent`, `stale_edge_forwards` |
| P7 churn | pattern changes **every** generation (rotate a random subset) | plan switch machinery under pressure: deferral, dead plans, invalidation routing | `plans_parked`, `parked_plans_applied`, `dead_plans_discarded` |
| P8 alter | `alter_arrival_count` ±d mid-stream, rotating ranks; verify persistence across ≥3 subsequent generations | rules 8/9, the persistence fix | ts-bypass traffic; expected counts shift permanently |
| P9 waiters | waiter set disjoint from arriver set (few-to-many / many-to-few); far-future waiters (wait on g+50) | notification rules 1–7, far-future waiter, subscribe race | `gap_pulls` (opportunistic), subscribe fan-in, re-subscribes |
| P10 poison | poisoned precondition feeds an arrival, rotating | Q4 poison propagation | oracle check 4 |
| P11 chaos | seeded random: participants, counts, waiters, alters, run-ahead all drawn per generation | everything at once — the runtime analogue of TLC's nondeterminism | full counter spread; **report zero-counters** |
| P12 step | ONE persistent pattern shift at mid-phase | §11.5: can the owner re-learn a plan whose deviation is discovered mid-generation? | `plan_rebuilds` = 2, `gathering_declared` ~ 1, flush quiet after convergence |

Phases P6/P7 deserve emphasis: the race-window cases (park, dead-plan discard,
gap-pull) **cannot be forced deterministically from application level** — they
need message timing. They are exercised probabilistically: high generation
counts, many seeds, and the counters are the proof. A campaign where
`dead_plans_discarded` never moved has not tested rule 10.2/10.3, no matter how
green it is — rerun P7 with more seeds or tighter generation pacing until it
moves.

---

## 4. The scale ladder

| Tier | Nodes | Build | What runs | What it answers |
|---|---|---|---|---|
| T1 | 2, 4, 8 | Debug + assertions | P1–P11, all seeds, oracle on every generation | functional correctness with maximum checking |
| T2 | 16, 32 | Release + counters | P1–P11, oracle sampled 1/64; plus P11 with 20+ seeds | race-window coverage (this is where parks/dead-plans actually fire) |
| T3 | 64 | Release | P1–P11 + soak: 10⁵ generations of P7+P11 interleaved; RSS tracked | memory bound, leak detection, plan-lifecycle stability |
| T4 | 128 | Release | full script + **performance comparison** | scaling claims |

Performance comparison at T4 (and spot-checked at T2): run the identical
script against a build of `main` (the pre-rewrite implementation) — **A/B by
build, not by runtime toggle**. Metrics: wall time per 1k generations for P1
(steady state must not regress — this is the O(log N) claim), owner-rank
message counts (steady-state fan-in must be O(radix), not O(N)), and P9
notification latency (trigger-to-wake on a far rank).

Scaling assertions worth automating:
- steady-state messages received at the owner per generation ≈ radix, flat in N
- `subscribe_fan_in` ≈ N once per barrier, not per generation
- `flush_report_bytes_max` bounded by the deviating subtree, not N, in P4/P5
- `agg_peak_entries` ≤ participants, and zero between rebuilds

---

## 5. Suggested schedule

1. G1–G3 preflight (workstation, ~a day of machine time).
2. T1 with every phase; fix what falls out; iterate until counters table is
   fully non-zero except the documented-probabilistic ones.
3. T2 seeds campaign until `plans_parked`, `dead_plans_discarded`, `gap_pulls`
   have all moved. **This is the exit criterion that matters most** — these are
   the paths TLC verified and hardware has never run.
4. T3 soak overnight.
5. T4 with the A/B baseline.

Each tier's artifact: merged counter table + oracle violations (must be zero) +
RSS curve + (T4) the perf comparison. Keep the seeds of any failing run —
a seed is this program's equivalent of a TLC trace.

---

## 6. Campaign record — eos, August 2026 (COMPLETE through T4 functional)

The ladder was run on eos (GASNet-EX/IB, 4 ranks/node, `gens=512`,
`phases=-1`, radix 2 and radix 4 at every rung). **Every rung passed every
phase on every seed**; the driver grew an 11th phase (`depart`, mask 1024)
mid-campaign to reach the notification-shrink machinery the split phase
structurally cannot exercise (its departs are products of the final triggers,
so they always arrive after the last trigger and die unweighed with the
barrier).

| Ranks (nodes) | Seeds×radix | Result | Firsts at this rung |
|---|---|---|---|
| 8 (2) | 4×2, +depart probe | PASS | gathering-generation fix validated; depart phase: `shrinks_applied`, `leave_rejoin`, `churn_backoffs` all first-ever |
| 16 (4) | 4×2 | PASS | `pin_conflicts_avoided` (53, radix 2) |
| 32 (8) | 10×2 + 512-gen soak | PASS | decline storm confirmed dead at scale |
| 64 (16) | 10×2 | PASS | `dead_plans_discarded` (2, radix 2, install-guard door) — **last dormant counter; exercise table 100% complete** |
| 128 (32) | 5×2 | PASS | — |
| 256 (64) | 3×2 | PASS | — |
| 512 (128) | 2×2 | PASS | — |

**Scaling shape (the campaign's quantitative findings):**

- **Decline volume is generation-linear and rank-invariant**: ~285/seed at
  gens=128 and ~1210–1280/seed at gens=512, flat across SIX rank doublings
  (8→512). Partial-evidence declines are a per-generation constant, not a
  scaling term.
- **Steady state converges in one plan build** (plus at most one
  identical-plan skip from the documented startup transient, which vanishes
  entirely at ≥256 ranks). Step converges in exactly two builds within ~4
  generations of the shift at every rung.
- **`flush_report_bytes_max` approaches linear-in-nodes**: 64→73→101→155 KB
  across 64→512 ranks (per-doubling ratio 1.15→1.38→1.53). This is the design
  floor, not a defect — flush reports carry per-node counts — but budget
  ~O(nodes) bytes for worst-case reports during full-eager windows
  (~1.2 MB at 4K nodes; fragmented-message territory).
- **`pin_conflicts_avoided` scales with tree depth** (radix 2 ≫ radix 4;
  ~24.5k/seed at 512 ranks) with zero conservation failures — rule 10.1 is
  load-bearing in production, not just in TLC.

**Behaviors observed and classified as design-working-as-intended:**

- At ≥256 ranks the depart phase's shrinks collapse to zero via the rule-3
  cost test + rule-8 hint: the phase's even-rank departure pattern is the
  encoded-target worst case (scattered removals shatter a compact range set
  into a per-delivery bitmap — see `multicast_cost()` in barrier_impl.cc,
  whose comment predicts exactly this). Contiguous idle blocks — the realistic
  phase-change pattern — keep paying at any scale. Coverage for the
  shrink/rejoin paths comes from the ≤128-rank rungs.
- The depart phase's churn arm sits deliberately on the
  `DEPART_CHURN_WINDOW` boundary, so its churn/clean classification flips
  with per-generation latency across rungs (all-churn at 128 ranks, all-clean
  at 256 ranks, radix-dependent in between). The honest-clock judgment
  (deferred to the subscribe reply — see NOTIFICATION_PROTOCOL.md) was
  validated at 8 ranks where both arms behave deterministically.

**Never exercised (and why that is acceptable):**

- The **dead-parked-plan door** ("discarding dead parked barrier plan") never
  fired; both 64-rank kills came through the install guard. It requires an
  invalidation to overtake its own newplan in flight, and GASNet-over-IB
  delivery is near-FIFO per peer pair. TLC-verified (MCStrand2); not
  reachable on this fabric without message-reordering injection.

**T3 soak and T4 A/B are both COMPLETE — see §7 and §8.  Every tier of §4
has now run.**

---

## 7. Benchmark record — T4 A/B, eos, August 2026 (COMPLETE)

Instrument: `benchmarks/barrier_bench` (public-API-only; the identical source
compiles against both builds, so A/B is by build) driven by
`benchmarks/barrier_bench/barrier_bench.sbatch` (interleaved A,B,A,B reps in
one allocation).  Fast-path proof comes from the `reports_received` /
`notifies_received` counters, not from timing.  Patterns: `uniform` (all
arrive 1) and `half` (lower half arrive 2, upper half only wait).

**Serial medians (µs, one arrive→trigger→notify→wake round) and pipelined
throughput (gens/sec, window 32), default plan radix 8:**

| Ranks | serial uniform main→new | serial half main→new | pipe uniform | pipe half |
|---|---|---|---|---|
| 8   | 41 → 47 (main +15%) | 35 → 42 (main +20%) | comparable | comparable |
| 32  | 124 → 90 (**1.4×**) | 109 → 71 (**1.5×**) | comparable | comparable |
| 128 | 485 → 211 (**2.3×**) | 425 → 130 (**3.3×**) | 6.0k → 10.9k (**1.8×**) | 5.4–8.9k → 11.1k |
| 512 | 7503 → 591 (**12.7×**) | 1650 → 344 (**4.8×**) | 585 → ~5.9k (**~10×**) | ~1.0k → ~6.8k (**6.5×**) |

- **Crossover ≈ 16 ranks.**  Legacy's round grows linearly in N (its 512-rank
  uniform trace shows EVERY generation at 7.4–9.7 ms — the O(N) owner incast
  plus O(N) unicast notification saturated into the steady state); the new
  path grows sub-linearly (16× ranks → 6.6× time).
- **Adoption cost**: the plan installs in 2–4 generations at every scale
  (spikes of ~3–17 ms, then flat), `plan_rebuilds=1`, flush confined to
  startup.  Legacy has no adoption but also no fast path to adopt.
- **Tails**: at 128 ranks the implementations stall in OPPOSITE patterns —
  main-half is catastrophic (p95 4–12 ms in 4/5 reps) while new-half is
  pristine (p99 ≤ 145 µs every rep); main-uniform is saturated-but-smooth
  while new-uniform stalls episodically (rare multi-ms maxes).  The common
  trigger is per-generation message burst size at the owner.

**The cascade (measured, radix-independent).**  The verified no-child-wait
rule means a relay forwards every child report that arrives after its own
quota lands, so steady-state owner fan-in is NOT O(radix): measured ≈ N−1
per generation in serial mode and ≈ N/6 pipelined, at EVERY plan radix
(2/4/8/16).  The tree buys latency and relay locality, not owner message
count.  This cascade is the sole source of the new build's remaining tail
artifact (episodic uniform-serial stalls) and the likely ceiling on its pipe
throughput.

**Plan-radix sweep at 512 ranks**: serial medians are radix-flat (±10%);
radix 8 wins throughput (6.1k vs 4.9k @16, 3.5k @4, 2.1k @2); radix 16 has
the cleanest tails; radix 2 is pathological (nine forwarding hops amplify
the cascade storm: half p95 11.3 ms).  **Default radix 8 is validated.**
The sweep also refuted burst-SHAPE mitigation: deeper trees make tails
worse, so only reducing the COUNT helps.

**Follow-up identified (TLC-first, not yet designed): quota-gated
forwarding (see end of §8).**  Gate a relay's re-reports on its subtree quota so the owner
sees ~radix messages per steady generation instead of ~N−1.  Child-wait was
removed as a verified stranding trap, but rule 10.1's pinned edges postdate
that decision and fix a generation's inflow at first touch, which may make
quota-gating safe now (deviations already bypass via flush mode).  Requires
new mutation-battery rows against MCStale/MCStrand2/MCDeepSwitch before any
implementation.  Expected payoff: removes the last tail artifact and lifts
pipelined throughput.

---

## 8. Soak record — T3, eos, August 2026 (COMPLETE)

One seed, 64 nodes / 256 ranks, plan radix 8 (the shipping default),
`PHASES=272` (churn + chaos — the two plan-lifecycle stress phases),
`GENS=100000` each.  Instrumentation: `rss_mb` samples from rank 0 every
8192 generations (tests/barrier_scale_test.cc), plus the counter dump.

- **Both phases PASS**: 2×10⁵ generations, ~790 s wall (churn 3.7 ms/gen at
  radix 8; an aborted first attempt at radix 2 ran exactly 2× slower,
  consistent with §7's radix findings — its per-seed `RUN_TIMEOUT` was the
  only failure).
- **RSS flat**: 1193 → 1201 MB end to end; +5 MB in the first sample
  interval (warm-up), then +3 MB across ~190k generations — a per-generation
  residue bound of ~15 bytes, allocator noise.  No leak.
- **Memory bound held by counters too**: `agg_peak_entries = 510`
  (= 2 barriers × 255 participants) and `flush_report_bytes_max = 4130` —
  both track PARTICIPANTS, not generation count, at 200× the campaign's
  generation length.  `subscribe_fan_in = 510`: once per barrier, never per
  generation (§4's scaling assertion, verified at soak length).
- **Plan lifecycle held cadence without drift**: `plan_rebuilds = 100001`
  over 2×10⁵ generations — exactly the one-rebuild-per-two-generations
  churn/chaos design rate, sustained; 899 parks, all applied.

With §6 (functional ladder), §7 (A/B benchmark) and this section, every
tier of §4 has run and every claim in §0's gates is either verified or
recorded with its measured value.

---

## 9. Quota-gated forwarding — TLC campaign record (follow-up phase 1, COMPLETE)

§7 identified the follow-up: cut steady-state owner fan-in from ≈N−1 to
≈radix messages per generation by gating a relay's report on its SUBTREE
quota.  Child-wait was a verified stranding trap in the original design, so
this went TLC-first.  Seven design iterations, four of them killed by
counterexamples — each kill producing a permanent probe scenario:

| Iter | Killed by | Lesson | Mechanism added |
|---|---|---|---|
| v1 | MCDouble | model infidelity: report targets were derived from OTHER nodes' records, a re-aiming barrier_impl cannot perform | F0: parent from the sender's own epoch's plan |
| v2 | MCGateAhead | owner silently accepts counts from plan-disowned senders | owner receipt valve: non-kid ⇒ flush |
| v4 | MCGateMove | a value accepted legally under plan k−1 is made deviant RETROACTIVELY by plan k | owner switch-time audit (value-exceeds + orphaned-sender) |
| v5 | MCGateDemote | flush fans are kid-list snapshots; later-gained children never hear | install-time re-fan (newplan + parked) |
| v6 | MCGatePark | mis-pinned counts hide value-legal inside a legitimate chain; only the stale-edge forwarder witnesses it | stale-edge ⇒ count-free flush signal to the owner |

**Final design (v7), landed in BarrierArrive.tla rules 1 and 6**: the `>=`
subtree-quota gate at relays, the composite owner valve (receipt +
switch-audit), install re-fans, and the stale-edge signal.  Full protocol
text: ARRIVAL_PROTOCOL.md §13.

**Verification record (all from the landed repo files):**
- 16 scenarios pass: the 10 affordable arrival scenarios plus the 6 new
  MCGate probes.  MCLate — previously outgrown — COMPLETES again under
  gating (~22M distinct states, checkpointed); MCDeepSwitch remains
  bounded-partial (no violation, ~50M+ generated states), as it already was
  on the ungated spec.
- Battery: arrival 14/14, gate 9/9 (8 mechanisms CATCH-certified + the gate
  itself documented-benign to remove), alter 4/4.  MCGatePark certifies TWO
  mechanisms through its two interleavings.
- Mask probe: under the valves, the case-3 flush signals (immediate and
  retroactive) are NO LONGER LOAD-BEARING FOR SAFETY — no scenario strands
  without them across the full roster.  They stay in the protocol as the
  plan-learning evidence (§11); their battery rows are documented-benign.
- Control: the F0 fidelity fix alone passes the entire ungated fast suite.

**Phase 2 (implementation) — COMPLETE.**  Landed per §13's mapping:
`should_report_locked` gates on `subtree_quota` (computed at install from
the payload slices each node already receives — nothing new on the wire,
O(radix) storage); the owner receipt valve and switch-time audit in
`handle_remote_report`/`build_new_plan_locked`; the receiver-side
stale-edge signal (the sender-side `is_direct` half already existed, and
the C++ already had the install re-fan the model campaign had to add — the
implementation was AHEAD of the model on two mechanisms).  Counters
`reports_gated` and `owner_valve_flushes` join the exercise table.

Local validation (8 ranks, MPI, gens=512): all 11 functional phases PASS at
plan radix 2 and 8, several seeds; `barrier_arrivals`/`barrier_reduce`
regression OK.  **The acceptance metric holds exactly: owner
`reports_received` = 2.01/gen serial and 2.04/gen pipelined at radix 2 —
down from the measured 4.35 and ~5 — i.e. fan-in = radix, with
`reports_gated` counting thousands of suppressed forwards.**
`owner_valve_flushes` stayed 0 locally: the broad `is_direct` machinery
covers every driver-reachable deviation shape, and the valve's unique cases
(stowaway values across switches, receiver-side stale edges) are
rare-interleaving paths for phase 3's scale runs to hunt, exactly as
`pin_conflicts_avoided` was.

**Phase 3, functional half — COMPLETE** (eos: 2 nodes ×2 radices ×2 seeds,
16 nodes ×2 radices ×3 seeds, all 11 phases green).  `reports_gated` = 340k
(radix 2) / 564k (radix 8) per 3-seed job at 64 ranks; declines stayed
generation-flat (and DROPPED at radix 2 — gating reduces eager traffic);
steady/step convergence signatures unchanged.  And a first: the
**dead-PARKED-plan door fired on hardware** ("discarding dead parked
barrier plan: parked=65 retired-by=65", churn, radix 2) — the one mechanism
§6 documented as fabric-unreachable; gating's hold patterns widened exactly
that window.  Every arrival-protocol mechanism has now been exercised on
real hardware except `owner_valve_flushes`, whose scenarios are
TLC-certified and whose hardware hunt continues at the A/B scales.

**Phase 3, A/B half — COMPLETE.**  The §7 rungs re-run with the gated
build (main arm unchanged); every prediction verified:

| Ranks | serial uniform main→gated (ungated) | pipe uniform main→gated (ungated) | owner fan-in serial (was) |
|---|---|---|---|
| 32  | 123 → **83 µs** (90)   | 22k → 24k (21k)          | **8.1/gen** (31)  |
| 128 | 486 → **116 µs** (211) | 6.1k → **~23k** (10.9k)  | **9.4/gen** (126) |
| 512 | 7570 → **~150 µs** (591) | 570 → **~17k** (5.9k)  | **13.8/gen** (501) |

- **Owner fan-in tracks the radix, not N, at every scale and radix**
  (512-rank sweep, pipe barriers: 6.2/7.1/8.6/17.0 per gen at radix
  2/4/8/16 - was ~N/6 to N−1).  The cascade is gone.
- **At 512 ranks the gated build is ~50× main on serial-uniform latency
  and ~30× on pipe throughput**, and 2.6-4× better than its own ungated
  self.  Gated serial-uniform grows 83→116→147 µs across 32→512 ranks
  (~+27% per 4× ranks) - the log-depth shape the tree was built for.
- Half-pattern tails: max ≤ 181 µs every rep at 128 ranks.  Uniform tails
  improved from most-reps-stalling to ~1-2 reps in 5 with episodic maxes,
  no longer N-scaling and now appearing in BOTH patterns at 512 - i.e. a
  background transport event, not protocol incast.  Radix 2 - pathological
  ungated BECAUSE of the cascade - improved ~9× on pipe and is no longer an
  outlier; **radix 8 stays the default**.
- `owner_valve_flushes` = 0 on hardware everywhere (4096 counter lines at
  512 ranks): the valve's strand shapes remain TLC-certified
  (MCGateOver/Move/Demote/Park) but unreproduced by this fabric, joining
  the §6 note on reordering-dependent paths.

Quota-gated forwarding is COMPLETE: verified (§9 phases 1), implemented
(phase 2), and measured at target scale (this section).
