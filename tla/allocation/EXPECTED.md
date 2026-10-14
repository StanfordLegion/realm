# Expected TLC outcomes per configuration

Drive file for the Phase 4 verification loop. Every run is judged against
this table; anything off-table is either a spec bug or an unregistered Realm
bug candidate — triage against DESIGN.md §8 before assuming either.

All configs check `MCDeferredAlloc.tla` (INIT/NEXT `MCInit`/`MCNext` unless
the row says SPECIFICATION). "dlk ON" = run TLC **without** `-deadlock`
(deadlock checking enabled); "dlk OFF" = pass `-deadlock`. Constant
substitution uses the protocol module's `Size` constant (`Size <- SizesX`).

Run mechanics (validated 2026-08-25): SANY and TLC work with
`/opt/homebrew/opt/openjdk/bin/java` + `../barrier/tools/tla2tools.jar`.
TLC must run **outside the bash sandbox** (its `states/` metadir and trace
writes are blocked by the sandbox write allowlist; no RMI issue was seen with
`-Djava.io.tmpdir=<local jtmp>`). **Exception found in Phase 4:** liveness
(SPECIFICATION/temporal) runs additionally bind a local RMI socket at startup
and die under the sandbox with `java.rmi.server.ExportException: Listen
failed on port: 0 / Operation not permitted` even with writable tmp/metadir —
temporal configs (Liveness, LivenessNoCross) strictly require an
unsandboxed JVM; safety-only configs only need writable tmp/metadir paths.

Post-Phase-3 fix batch (FIX 1-5, 2026-08-25, applied to DeferredAlloc.tla):
instOffset is now built from placement-time offsets carried through the
drain/ARR helpers (cc:1668/cc:1693 semantics; the old final-allocator re-read
aborted TLC in C2-off configs); the cc:1542 entry assert is ghost-flagged;
the DELAYEDDESTROY-with-triggered-preD branch now acks (cc:915-917
release-build behavior); and a new **`INV_NoDupAlloc`** detector (`dupAlloc`
ghost; inl:500 duplicate-tag leak, the #442 class — the partial-function
representation cannot express the leak itself) was added to the
Safety/SafetyMini/Poison4/Big batteries. Expected green in contract-ON
configs; a violation is a new bug candidate, not noise. Baseline outcome
classes below re-validated after the batch (state counts may shift slightly).

| Config | Extra TLC flags | Checks | Expected outcome | Status / if it deviates |
|---|---|---|---|---|
| Smoke | *(none — dlk ON)* | green battery (expected-FAIL invariants BUG-6 excluded) | **PASS or a BUG-1-shape deadlock** (both acceptable; BUG-1 is reachable even here). | **RAN: deadlock at depth 7** (4029 states) — BUG-1 shape via the DELAYEDDESTROY variant; trace `traces/Smoke-run1.txt`. Any **invariant** violation ⇒ spec bug. |
| EventLoop | *(none — dlk ON)* | dlk + small green set | **FAIL: deadlock**, 7 states, matching the hand-simulated trace in MCDeferredAlloc.tla's trailer. | **RAN: CONFIRMED** — deadlock in exactly 7 states (57 generated), final state `instState = <ALLOCATED, ALLOC_DEFERRED>`, `pendingAllocs` waiting on a release whose precondition needs `eCreated(2)`. Trace `traces/EventLoop.trace.txt`. **BUG-1 confirmed by TLC.** |
| Safety | `-deadlock` | full battery incl. INV_NoReadyWhenNoPendingAllocs + INV_NoReadyAtRebuild | **FAIL: INV_NoReadyWhenNoPendingAllocs** (BUG-6). | **RAN ON SAPLING (round 1, job 77808): CONFIRMED** — INV_NoReadyWhenNoPendingAllocs violated at depth 10 after 4.4B generated / 1.44B distinct, 5h04m, 40 workers. **BUG-6(a) confirmed at 4-instance scale.** Trace in `slurm-77808-Safety.out`. TLC halted there, so the deeper BUG-3 hunt was PREEMPTED → superseded by **SafetyHunt.cfg** (round 2). |
| SafetyMini | `-deadlock` | same battery, 3 inst / H=3 / sizes (2,2,1) | **FAIL: INV_NoReadyWhenNoPendingAllocs** (BUG-6, canonical witness). | **RAN: CONFIRMED in 10s** (5.5M states, violation at depth 9). Final state: `pendingAllocs = <<>>`, `pendingReleases = [destroy(2) ¬ready, destroy(3) READY+defNote]` — the stranded-ready shape, action-for-action the canonical witness (destroy(3) triggered at request in this variant). Trace `traces/SafetyMini.trace.txt`. **BUG-6 confirmed by TLC.** |
| SafetyHunt (was "Safety iter. 2") | `-deadlock` | battery minus the two BUG-6 invariants (checked-in cfg, round 2) | **GREEN = BUG-3 absent at these bounds**; a hit on INV_InOrderUnblockSucceeds / INV_FutureOffsetConsistency is the first BUG-3 witness | Not yet run (sapling round 2). Anything found is a fresh Realm bug candidate → Phase 5. Likely runs longer than Safety's 5h04m — it will not stop early; use `-recover`. |
| PoisonHunt | `-deadlock` | Poison4's battery minus the two BUG-6 invariants (checked-in cfg, round 2) | a hit = first witness of **BUG-4-standalone** (INV_NoOrphanTags / INV_CurrentMatchesGround / INV_QuiescentHeapEmpty), **unfixed BUG-5** (INV_FutureOffsetConsistency / INV_InOrderUnblockSucceeds / INV_NoOverlap), or **cc:1587** (INV_PoisonReplayOnlyFailsAfterPoint); GREEN = absent at these bounds | Not yet run (sapling round 2). |
| Liveness | `-deadlock` | SPECIFICATION LiveSpec; PROPERTY LIVE_NoStuckAllocs | **FAIL: LIVE_NoStuckAllocs** via a BUG-1 lasso. | **RAN: CONFIRMED** at reduced bounds (2 inst / H=3, SizesEventLoop — the original 3-inst/H=4 run's throughput collapsed ~5x from behavior-graph maintenance at ~500k distinct/10 min and was stopped; reduction documented in the cfg header). 8-action counterexample ending in **Stuttering**: `instState[1] = ALLOC_DEFERRED`, its enabling release destroy(2) has `preD[2].deps = {1,2}` with `eCreated(1)` UNFIRED — the BUG-1 wait cycle. 7,273 gen / 3,531 distinct, seconds. Trace `traces/Liveness-bug1.txt`. |
| LivenessNoCross (confirmatory) | `-deadlock`; own cfg (`LivenessNoCross.cfg`, `CLIENT_MODE = "NO_CROSS_DEPS"`, same bounds as Liveness) | same | **PASS** (BUG-1 shape excluded). | **RAN: PASS** — complete state space, no error (3,929 gen / 1,987 distinct, seconds). Same-bounds control: the Liveness failure is specifically the cross-instance destroy dependency cycle. Log `traces/LivenessNoCross-pass.txt`. A future FAIL would be a *second* liveness bug → Phase 5, high interest. |
| Composite4 | `-deadlock` | 5 inst / H=4 / sizes (2,2,1,1,1), `CLIENT_MODE = "SCRIPTED_COMPOSITE"`; battery minus the two BUG-6 markers | **FAIL: INV_CurrentMatchesGround / INV_NoOrphanTags** — the BUG-6→BUG-4 composite leak (bugs/BUG-6.md item 6). | **RAN: CONFIRMED in 5s** (93,795 gen / 51,904 distinct, violation at depth 12). Final state: tag 3 in `cur`, `instState[3]=DESTROYED`, `notifyCount[3]=1`, no pendingReleases entry, `readyAtRebuild=TRUE` (path went through the BUG-6 stranding) — permanent range leak + notify-while-tag-live, **no poison anywhere**. TLC reported INV_CurrentMatchesGround (first in battery order); INV_NoOrphanTags is violated in the same state. Trace `traces/Composite4.txt`. The predicted composite is now machine-confirmed. |
| Poison4 | `-deadlock` | full battery, USER_POISON | **FAIL: INV_NoReadyWhenNoPendingAllocs / INV_NoReadyAtRebuild** first (BUG-6). | **RAN ON SAPLING (round 1, job 77809): CONFIRMED** — INV_NoReadyWhenNoPendingAllocs violated after 887M generated / 272M distinct, 47min. **BUG-6 confirmed on the poison paths.** Trace in `slurm-77809-Poison4.out`. TLC halted there, so the BUG-4-standalone / unfixed-BUG-5 / cc:1587 hunts were PREEMPTED → superseded by **PoisonHunt.cfg** (round 2). |
| Big | `-deadlock` | full battery, USER_POISON, 5 inst / H=6 | Known violations first (comment out to go deeper); then open hunt. | **SAPLING ROUND 1 (job 77812): IN PROGRESS / RESUMABLE** — no violation through 6.26B generated / 2.87B distinct at depth 9 (~12h in). Resume with `-recover` (see SAPLING_JOBS.md) or check `squeue` — it may still be running. Anything new ⇒ Phase 5 with full trace. |

## Canonical BUG-6 witness (3 instances, H=3 — found by TLC in Phase 2)

1. destroy(i1) requested, deferred → R1 (¬ready).
2. create(i2) → DEFERRED, `lastSeq = seq(R1)`.
3. destroy(i2) requested, deferred → R2 (legal: request ≠ trigger; C2 only
   constrains when the precondition can *fire*).
4. destroy(i3) **triggered** → cc:871 frees rel/fut; ARR front-gate fails →
   R3 pushed **READY** (cc:884-887).
5. trigger destroy(i1) → oldest drain frees i1, unblock scan places i2
   (`i2.lastSeq < seq(R2)`), `pendingAllocs` empties; the do-while stops at
   non-ready R2 → **ready R3 stranded** behind it with no pending allocs.
6. Any later alloc request that reaches the cc:768 rebuild trips the cc:772
   `assert(!it->is_ready)` (INV_NoReadyAtRebuild); the stranded state itself
   violates INV_NoReadyWhenNoPendingAllocs.

## Iteration protocol (Phase 4)

1. Order: Smoke → EventLoop → Safety → Liveness → Poison4 → Big.
2. On an **expected** violation: save the trace (TLC stdout) under
   `traces/<Config>-<inv>.txt` (or `<Config>.trace.txt`), comment out that
   invariant/property in the cfg, re-run, repeat until the config passes or
   produces an unexpected result.
3. On an **unexpected** violation: stop iterating that config; hand the trace
   to Phase 5. Do not "fix" the spec to make a violation disappear without a
   line-level fidelity argument against mem_impl.cc.
4. A TLC *error* (parse, undefined name, type) is a Phase 2 seam issue — the
   Phase 2 reconciliation is complete, so treat any new one as a regression.

## Known model caveats (do not misread as Realm bugs)

- **Poisoned-destroy leaks are legal terminals.** A destroy whose
  precondition fires poisoned is silently cancelled (cc:818-825) or removed
  (cc:1754); the instance stays ALLOCATED with its tag in `cur` forever
  ("POSSIBLE LEAK", ii:87). `Quiescent` admits this via
  `DestroyResolvedLeak`, and `INV_QuiescentHeapEmpty` permits exactly those
  tags — a BUG-4-stranded tag (DESTROYED/notified instance) is still flagged.
  With intrinsic poison + C2 this is reachable in every config, including
  Smoke.
- `fut`/`rel` are initialized to the empty-domain allocator, which under the
  derived-gap representation reads as **all-free**, whereas the code's
  default-constructed allocators have **no managed range** (CanAlloc ≡ false;
  only `current_allocator` gets `add_range`, cc:680). Neither is read before
  first assignment under the code's validity convention; a trace that reads
  them earlier is a genuine staleness finding, but its concrete allocator
  values in that window will not match the C++.
- Smoke/EventLoop keep deadlock checking ON; Safety/Poison4/Big deliberately
  run `-deadlock` so short BUG-1 deadlock traces don't preempt deeper
  invariant hunts (TLC halts at the first violation). This deviates from the
  "dlk" column of DESIGN.md §7's table for Safety/Poison4/Big — intentional.
- With deadlock ON, Smoke halts at its first BUG-1-shape deadlock, so its
  invariant coverage is truncated; Smoke is a parse/typecheck/fast-sanity
  gate, not a coverage run.
- `SeqCtrBound` (`seqCtr <= 2·|INSTANCES|`) is a backstop CONSTRAINT in every
  config; in v1 (one release per instance) it should never bind. If a run
  reports states being constrained away, that itself is a finding.

## Fix validation (v-next) — three-toggle bundle

Three spec-side toggles model the candidate fixes before any C++ changes
(all declared in DeferredAlloc.tla):

- **FIX_CAP** — BUG-1 capped admission: request-time seqid snapshot
  (`reqCap`), capped canonical-order admission test, monotone-cap guard,
  capped-fail → ALLOC_INSTANT_FAILURE. Pure cap — **no ready-fold in v1**.
- **FIX_SWEEP** — BUG-6 stranded-ready sweep at the pendingAllocs→empty
  transitions, plus the BUG-4-standalone `rel` re-apply in
  remove_pending_release.
- **FIX_RPR** — BUG-5 close: remove_pending_release's replay processes
  **trailing** pending allocs after the walk (place-or-EVENTUAL_FAIL), so a
  poisoned release can no longer strand an admitted alloc it funded.

**Composition discovery — explicit C++ gate:** the first validation round
proved **FIX_CAP without FIX_RPR is NOT shippable**. The cap increases
poisoned-release frequency (honest capped failures poison eCreated →
dependent destroy preconditions fire poisoned → remove_pending_release runs
more often), which makes BUG-5's trailing-alloc hole load-bearing for drain
liveness: Inversion with deadlock checking ON deadlocked via a trailing
alloc that the replay never revisited — neither failed nor refunded
(witness kept: `traces/Inversion-bug5-deadlock.txt`; it doubles as the
canonical 3-instance BUG-5 liveness witness). The verified bundle is
**CAP + SWEEP + RPR**: any C++ landing must take all three together (at
minimum, the cap is strictly gated on the RPR trailing replay). At
2-instance bounds the stranding shape is unreachable (C1 request order +
the cap leave no admissible trailing alloc), which is why the 2-instance
bundle configs pass their deadlock/liveness checks regardless.

**Battery hardening (post round-1 triage):** `INV_NoDupAlloc` is now checked
in **every** bundle config (SmokeFixed, EventLoopFixed, EventLoopCapOnly,
LivenessFixed, SafetyMiniFixed, SafetyMiniSweepOnly, Composite4Fixed,
GCRipple, Inversion — plus the sapling three, where it was already present).
Round 1's wiring bug was caught first at 4-instance sapling scale precisely
because no local bundle config checked the detector; that coverage gap is
closed. Local bundle configs should re-run green with the corrected
TrailingRPR wiring before any sapling resubmission.

Toggle pinning: every non-bundle config pins `FIX_RPR = FALSE` (regression
semantics unchanged — the constant reproduces pre-fix behavior when FALSE).
The attribution pair deliberately stays partial: EventLoopCapOnly
(CAP only — sound because its 2-instance bounds cannot reach the BUG-5
shape) and SafetyMiniSweepOnly (SWEEP only — sound because without the cap
its bounds produce no poisoned funding releases; flags are `-deadlock`
regardless).

| Config | CAP | SWEEP | RPR | Expectation | RESULT — two-toggle round (2026-08-26); re-validation with FIX_RPR pending |
|---|---|---|---|---|---|
| SmokeFixed | on | on | on | **fully green incl. deadlock check**; BUG-6 invariants checked and passing | PASS, complete — 13,303 gen / 6,337 distinct, depth 13, ~1s |
| EventLoopFixed | on | on | on | green: BUG-1 client's create INSTANT-FAILS, cascade drains, deadlock check passing | PASS, complete — 71 gen / 47 distinct, depth 9 |
| EventLoopCapOnly | on | off | off | green — **BUG-1 fixed by the cap alone** (BUG-5 shape unreachable at 2 inst) | PASS, complete — identical counts to EventLoopFixed; attribution confirmed |
| LivenessFixed | on | on | on | **LIVE_NoStuckAllocs PASSES** (FAILED is in the resolved target set) | PASS — no error, 13,303 gen / 6,337 distinct, 41s; same bounds where base FAILS |
| SafetyMiniFixed | on | on | on | fully green; BUG-6 pair in battery and passing at the base's 10s-fail bounds | PASS, complete exhaustion — 64.7M gen / 23.5M distinct, depth 19, 5m40s |
| SafetyMiniSweepOnly | off | on | off | green — **BUG-6 fixed by the sweep alone** | PASS, complete exhaustion — 53.3M gen / 20.5M distinct, depth 19, 5m33s |
| Composite4Fixed | on | on | on | fully green; INV_NoOrphanTags + INV_CurrentMatchesGround passing, BUG-6 pair added and passing | PASS, complete — 245,493 gen / 119,728 distinct, depth 16, ~10s |
| GCRipple | on | on | on | green; both intent invariants hold; both end classes reachable (honest INSTANT-FAIL = the **accepted behavior change** of the pure cap) | PASS after one harness fix (create-order guard added to `CreateOrderOK`; misfire trace `traces/GCRipple-orderguard-misfire.txt`) — 72 gen / 43 distinct, depth 10 |
| Inversion | on | on | on | **GREEN over the full space with deadlock checking ON**: the cap/monotone guard instant-fails A in every interleaving, B ends **EVENTUAL_FAILURE cleanly via the RPR trailing replay**, every interleaving drains; INV_InversionCapped + SAFETY_PromisesKept hold. A deadlock here ⇒ **FIX_RPR design bug** (trailing replay missed a case). | two-toggle round: `-deadlock` PASS full space (479/257); deadlock-ON **deadlocked = the BUG-5 composition witness** (see gate above). Superseded by the bundle expectation. |
| Inversion (CAP+SWEEP, RPR=FALSE) — historical, not a checked-in config | on | on | off | *documents the gate:* reproduces the BUG-5 stranding as a deadlock — **FIX_CAP without FIX_RPR is not shippable** | witness: `traces/Inversion-bug5-deadlock.txt` |

Regression sweep (all toggles FALSE) re-validated on the two-toggle round:
Smoke deadlock @ 7 ✓, EventLoop deadlock @ 7 ✓, SafetyMini
INV_NoReadyWhenNoPendingAllocs @ 9 ✓, Composite4 INV_CurrentMatchesGround
@ 12 ✓. FIX_RPR = FALSE preserves pre-fix RPR behavior, so these
expectations carry over unchanged; the verification pass should spot-check
one of them after the FIX_RPR spec edit lands.

### Sapling fix-validation configs (created by the verification fork)

| Config | CAP | SWEEP | RPR | Expectation | Round-1 result (2026-08-26) |
|---|---|---|---|---|---|
| SafetyFixed4 | on | on | on | **FULLY GREEN** — BUG-5 detectors included (the bundle now addresses BUG-5); any violation = fix-design bug or new candidate | **VIOLATED: INV_NoDupAlloc** at depth 12 (job 77810; 12.8B generated / 4.0B distinct, 14h10m; trace in `slurm-77810-SafetyFixed4.out` from line 901). **TRIAGED (bugs/DUPALLOC-TRIAGE.md): verdict (a) — spec artifact.** The FIX_RPR *call-site wiring* fed `TrailingRPR` the full survivor list instead of the trailing remainder (a two-line DeferredAlloc.tla correction, applied); the fix DESIGN is unaffected. The same wiring bug has a second in-model flavor — a kept (already-replayed) alloc spuriously EVENTUAL_FAILED with its placement stranded in `fut` — covered by the same correction. **Corrected + locally re-validated (2026-08-26): full local matrix green on the corrected spec — fast set incl. Inversion deadlock-ON (478/255), SafetyMiniFixed exhaustion 64.68M/23.54M, SweepOnly 53.30M/20.55M, LivenessFixed pass, toggles-off regressions exact. Fresh resubmission pending** (spec changed → old checkpoints invalid). |
| Poison4Fixed | on | on | on | **FULLY GREEN** — BUG-5 detectors included (flipped from "expected-possible" now that FIX_RPR closes BUG-5); a BUG-5-detector hit here ⇒ FIX_RPR design bug on the poison paths | **DID NOT RUN** — round-1 submission typo (`PoisonFixed4` instead of `Poison4Fixed`, job 77811 exited immediately: "no such config"). Resubmit with the correct name **after the TrailingRPR wiring correction re-validates locally** (fresh start; it runs the corrected bundle). |
| BigFixed | on | on | on | FULLY GREEN on the BUG-1/4/5/6 families; anything else = new bug candidate → Phase 5 | **HELD** (was: in progress, job 77813, clean through 1.06B generated / 470M distinct at depth 9). Round-1 progress checked the dup detector against the STALE TrailingRPR wiring, and the spec correction invalidates the checkpoint — restart fresh after local re-validation; do not `-recover`. |

## Sapling round 1 summary (2026-08-26)

| Job | Config | Toggles | Outcome |
|---|---|---|---|
| 77808 | Safety | off | **BUG-6(a) confirmed at 4-inst scale** (INV_NoReadyWhenNoPendingAllocs, depth 10; 4.4B gen / 1.44B distinct, 5h04m). Expected. BUG-3 hunt preempted → SafetyHunt (round 2). |
| 77809 | Poison4 | off | **BUG-6 confirmed on poison paths** (887M gen / 272M distinct, 47min). Expected. BUG-4-standalone / unfixed-BUG-5 / cc:1587 hunts preempted → PoisonHunt (round 2). |
| 77810 | SafetyFixed4 | bundle | **UNEXPECTED: INV_NoDupAlloc violated** at depth 12 (12.8B gen / 4.0B distinct, 14h10m). **Triaged: spec artifact** — TrailingRPR call-site wiring, corrected; fresh resubmission after local re-validation. |
| 77811 | (Poison4Fixed) | bundle | **Did not run** — submission typo `PoisonFixed4`. Resubmit round 2 (after local re-validation of the wiring correction). |
| 77812 | Big | off | In progress / resumable — clean through 6.26B gen / 2.87B distinct, depth 9. Toggles-off: unaffected by the wiring correction; resume freely. |
| 77813 | BigFixed | bundle | Was in progress (clean through 1.06B gen / 470M distinct, depth 9) — **HELD**: checkpoint invalidated by the spec correction; restart fresh after local re-validation. |
