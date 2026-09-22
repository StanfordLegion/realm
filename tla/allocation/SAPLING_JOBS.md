# Sapling job list — deferred-allocation TLA+ campaign — ROUND 2

Round 1 (jobs 77808-77813, 2026-08-26) is complete; results are summarized
at the bottom of this file and recorded per-config in EXPECTED.md. Round 2
exists because:

1. **Both Group-A hunts were preempted by the expected BUG-6 violation** —
   TLC halts at the first violation, so Safety and Poison4 confirmed BUG-6
   at scale (good) but never reached the bug classes they were submitted to
   hunt. New checked-in configs **SafetyHunt.cfg** and **PoisonHunt.cfg**
   are identical minus the two expected-FAIL invariants
   (`INV_NoReadyWhenNoPendingAllocs`, `INV_NoReadyAtRebuild`), so the deep
   detectors are now reachable.
2. **Poison4Fixed never ran** — round-1 submission typo. The config is
   **`Poison4Fixed`** (job 77811 was submitted as `PoisonFixed4` and exited
   immediately with "no such config"). Copy-paste the command below.
3. **Big / BigFixed were still clean but unfinished** at snapshot time —
   Big (toggles off) is checkpointed and resumable; BigFixed is now HELD
   (see below).
4. **Bundle jobs were gated on a local re-validation — the gate is now
   CLEARED (2026-08-26):** the corrected bundle re-validated green across
   the full local matrix (fast set + both SafetyMini-scale full
   exhaustions + LivenessFixed) with exact toggles-off regressions.
   §2(iii) is submittable. History below.

## Bundle jobs (SafetyFixed4 / Poison4Fixed / BigFixed): re-validation history (gate CLEARED)

Round 1's SafetyFixed4 violated **INV_NoDupAlloc** at depth 12 (12.8B
generated / 4.0B distinct, 14h10m). **Triage verdict
(`bugs/DUPALLOC-TRIAGE.md`): spec artifact** — the FIX_RPR call-site
wiring fed `TrailingRPR` the full survivor list instead of the trailing
remainder; a two-line DeferredAlloc.tla correction has been applied (the
fix DESIGN is unaffected; a second in-model flavor of the same wiring bug
— a kept alloc spuriously EVENTUAL_FAILED with its placement stranded in
`fut` — is covered by the same correction). Consequences for sapling:

- **Local re-validation on the corrected spec: COMPLETE and green
  (2026-08-26)** — SmokeFixed/EventLoopFixed/EventLoopCapOnly/
  Composite4Fixed/GCRipple/Inversion all pass (Inversion 478/255,
  deadlock-ON), SafetyMiniFixed exhausts 64.68M gen / 23.54M distinct and
  SafetyMiniSweepOnly 53.30M / 20.55M with zero violations, LivenessFixed
  passes, and the toggles-off regressions reproduce their exact baselines
  (EventLoop deadlock@7, SafetyMini violation@9). `INV_NoDupAlloc` is now
  in every local bundle battery, closing the coverage gap that let
  sapling catch this first.
- **Bundle checkpoints from round 1 are INVALID** (the spec changed):
  SafetyFixed4 and BigFixed must restart FRESH — no `-recover`, and
  delete `states/SafetyFixed4` / `states/BigFixed` before resubmitting.
  BigFixed's round-1 progress in particular checked the dup detector
  against the STALE wiring, so its "clean through 470M distinct" tells us
  nothing about the corrected bundle.
- **Toggles-off jobs are unaffected** (`FIX_RPR = FALSE` behavior is
  untouched by the correction): SafetyHunt / PoisonHunt / Big submit or
  resume freely now.

## 1. Get the tree onto sapling

Same as round 1 — the new/changed files are `SafetyHunt.cfg`,
`PoisonHunt.cfg`, `EXPECTED.md`, this file:

```sh
# from the laptop
rsync -av --exclude states --exclude jtmp --exclude 'slurm-*' \
    ~/realm/tla/ sapling:realm-tla/
```

Note: rsyncing does NOT touch `states/` on sapling. **However:
DeferredAlloc.tla changed since round 1 (the TrailingRPR wiring
correction), and TLC checkpoint recovery requires the BYTE-IDENTICAL spec
that wrote the checkpoint** — the serialized state stream references the
string-intern table built at parse time, so ANY spec edit (even a
semantically inert one under `FIX_RPR = FALSE`) shifts the table and
recovery fails with `ValueInputStream: Can not unpickle a value of kind
<N>` (observed 2026-08-27 attempting the Big resume). Consequence: ALL
round-1 checkpoints are invalid, Big's included. Rule for future rounds:
never edit the `.tla` files while a checkpointed run you intend to resume
is outstanding.

## 2. Submit (round 2)

```sh
cd realm-tla/allocation        # submit FROM this dir (SLURM_SUBMIT_DIR)

# (i) SUBMITTABLE NOW - the unpreempted Group-A hunts (toggles off,
#     unaffected by the TrailingRPR correction)
sbatch -t 48:00:00 sapling_tlc.sbatch SafetyHunt
sbatch -t 24:00:00 sapling_tlc.sbatch PoisonHunt

# (ii) Big (toggles off): round-1 checkpoint is UNRECOVERABLE (spec changed;
#      see the note in §1 — recovery was attempted 2026-08-27 and failed
#      with the unpickle error). Options:
#        (a) RECOMMENDED: skip Big this round. Its expected outcome was an
#            eventual BUG-6-family hit, already confirmed at scale twice;
#            SafetyHunt/PoisonHunt are the informative toggles-off runs.
#        (b) If queue time is cheap, restart fresh:
# rm -rf states/Big
# sbatch -t 48:00:00 sapling_tlc.sbatch Big
#      (For reference, a VALID resume of an unchanged spec passes the
#      checkpoint dir's ABSOLUTE path: -recover is a filesystem path
#      resolved from the JVM's cwd, not an id looked up under the metadir.)

# (iii) SUBMITTABLE NOW - the corrected bundle re-validated green locally
#      on 2026-08-26 (full matrix incl. both SafetyMini-scale exhaustions,
#      LivenessFixed, and exact toggles-off regressions; see EXPECTED.md).
#      FRESH starts, no -recover - the spec changed:
rm -rf states/SafetyFixed4 states/BigFixed
sbatch sapling_tlc.sbatch SafetyFixed4
sbatch sapling_tlc.sbatch Poison4Fixed     # CORRECT NAME (round 1
                                           # typo'd it as "PoisonFixed4")
sbatch -t 48:00:00 sapling_tlc.sbatch BigFixed
```

## 2b. Round-2 outcome and round 3 (2026-08-27)

All five round-2 jobs (77823-77827) died simultaneously at ~05:18 after
~4.6-4.9 h — no violations, no completions. Cause: TLC's disk state queues
were on shared `/scratch2` (a design mistake in this script's round-1/2
version), which exhausted the shared filesystem and impacted other users.
The states were deleted to unblock the cluster, so round-2 checkpoints are
GONE — round 3 restarts from zero. Clean-so-far bounds from round 2 (still
valid as bounded-verification evidence, all no-violation): SafetyFixed4
989M distinct @ depth 10, Poison4Fixed 1.21B @ 10, SafetyHunt 1.02B @ 10,
PoisonHunt 1.16B @ 11, BigFixed 900M @ 9.

**HARD RULE (recorded 2026-08-27): never put TLC data on /scratch on
sapling — this or any other work.** The script now keeps the metadir on
node-local /tmp ($SLURM_TMPDIR when set) and checkpoint sync-back to the
submit dir is DEFAULT OFF (opt in per-run with SYNC_CHECKPOINT=1 only if a
single bounded checkpoint write to shared storage is acceptable). With sync
off, an interrupted run restarts from zero — size time limits so runs
finish in one shot.

Round 3: fresh starts. The jobs are INDEPENDENT — the `-d afterany` chain
below is optional: it guarantees the gate job (SafetyFixed4) runs first,
limits the footprint to one node at a time, and prevents two jobs from
sharing one node's /tmp if nodes are wider than 40 cores. If the queue is
quiet, submitting all four flat (no `-d`, add `--exclusive` on wide nodes)
is equally correct and finishes ~4x sooner in wall-clock. Re-sync the tree
first (the sbatch script changed):

```sh
cd /scratch2/mebauer/tla/allocation
# --parsable makes sbatch print just the job id, so chaining is automatic
J1=$(sbatch --parsable -t 48:00:00 sapling_tlc.sbatch SafetyFixed4)   # the C++ gate
J2=$(sbatch --parsable -t 48:00:00 -d afterany:$J1 sapling_tlc.sbatch Poison4Fixed)
J3=$(sbatch --parsable -t 48:00:00 -d afterany:$J2 sapling_tlc.sbatch SafetyHunt)
J4=$(sbatch --parsable -t 48:00:00 -d afterany:$J3 sapling_tlc.sbatch PoisonHunt)
echo "queued: $J1 -> $J2 -> $J3 -> $J4"
# BigFixed: dropped (largest state space, least marginal info vs SafetyFixed4)
```

Gate criterion (BFS ⇒ "clean through depth D" = complete coverage of all
behaviors of length ≤ D): the only scale-level failure ever observed was at
depth 12 (round-1 SafetyFixed4, stale wiring). **SafetyFixed4 clean through
completed depth 13 (log shows `Progress(14)`) passes the round-1 failure
point and opens the C++ gate**, with the local full exhaustions as the
semantic backbone — full exhaustion of these spaces is a bonus, not a
requirement.

## 2c. Round-3 outcome (2026-08-27, jobs 77843/77853/77854/77855)

All four died on node-local disk exhaustion — they ran the PRE-gzip script
(the tree was not re-synced before submission). All violation-free at
death: Poison4Fixed clean through depth 10 @ 1.90B distinct (4h25m, best
bundle+poison coverage yet); SafetyFixed4 depth 9 @ 1.10B (node had only
261G free at start); SafetyHunt depth 10 @ 1.67B; PoisonHunt depth 10 @
1.42B. Measured queue cost ~210-270 bytes/state uncompressed → with -gzip
(now default in the script) a clean 733G node holds roughly 15-30B queued
states, which should reach the depth-13 gate. Round-4 prep: (1) RSYNC THE
TREE (the missed step), (2) sweep leftover /tmp/mebauer-tlc-* dirs off the
compute nodes (scancel'ed jobs can't finish their cleanup trap in the kill
grace window), (3) resubmit the serial chain with --exclusive; verify the
job header shows `-gzip` in extra and "checkpoint-sync=OFF".

Site defaults (partition `cpu`, 40 cpus, 128 G, java discovery) are at the
top of `sapling_tlc.sbatch`, overridable per submission. The script
auto-appends `-deadlock` for every config named here (deadlock checking
stays on only for the small local Smoke/EventLoop-family runs); SafetyHunt
and PoisonHunt fall under the same default.

## 3. What each round-2 job hunts

| Job | Toggles | Hunts | Expected outcome |
|---|---|---|---|
| SafetyHunt (Safety minus BUG-6 invariants) | off | **BUG-3 class**: in-order unblock soundness (`INV_InOrderUnblockSucceeds`, cc:1668-1670) and future-offset determinism (`INV_FutureOffsetConsistency`, cc:1674-1691) after ARR partial-path history rewrites; plus the full remaining battery | **GREEN = BUG-3 absent at these bounds.** A detector hit = first BUG-3 witness → save trace, Phase 5. |
| PoisonHunt (Poison4 minus BUG-6 invariants) | off | **BUG-4-standalone** (`INV_NoOrphanTags`/`INV_CurrentMatchesGround`/`INV_QuiescentHeapEmpty`), **unfixed BUG-5** (`INV_FutureOffsetConsistency`/`INV_InOrderUnblockSucceeds`/`INV_NoOverlap`), **cc:1587** (`INV_PoisonReplayOnlyFailsAfterPoint`) | Any hit = first TLC witness of that variant → save trace, Phase 5. GREEN = absent at these bounds. |
| Poison4Fixed | bundle | FIX_SWEEP poison-path coverage + FIX_RPR trailing replay under USER_POISON | FULLY GREEN (incl. INV_NoDupAlloc). **Gated**: submit only after the corrected bundle re-validates locally; fresh start. |
| Big (resume) | off | Open hunt continuation (clean through 2.87B distinct, depth 9) | **Submittable now** (toggles-off, unaffected by the correction). Known markers first (BUG-6 family will eventually fire here too and end the run — when it does, treat as confirmation and stop; a BigHunt variant is only worth creating if Big's BUG-6 hit comes early). |
| SafetyFixed4 (fresh) | bundle | Fixed-model soundness at ARR-partial scale, now with the CORRECTED TrailingRPR wiring | FULLY GREEN (incl. INV_NoDupAlloc — the round-1 violation was a spec artifact, corrected). **Gated** on local re-validation; fresh start, delete `states/SafetyFixed4` first. |
| BigFixed (fresh) | bundle | Largest fixed-model sweep | **Gated + fresh start** — the round-1 checkpoint ran the stale wiring and is invalid; delete `states/BigFixed` first. |

## 4. Expected wall-times (round-1 calibrated)

Round-1 sapling throughput at 40 workers: ~14-18M generated/min (Safety),
~19M/min (Poison4), ~10M/min at depth 9+ (Big-sized states).

- **SafetyHunt**: Safety hit BUG-6 at 4.4B generated / 5h04m *while
  stopping early*; SafetyHunt explores the same space to exhaustion or a
  BUG-3 hit — plan for **>5h, possibly 24h+**; submit with `-t 48:00:00`
  and expect a `-recover` cycle.
- **PoisonHunt**: Poison4 hit BUG-6 at 887M / 47min; the unpreempted space
  is larger — **several hours**, `-t 24:00:00` should suffice.
- **Big resume**: unknown total; keep `-t 48:00:00` and `-recover` cycles.
- **BigFixed fresh start** (when un-gated): budget the full run again
  (round-1's ~2h reached 470M distinct on the stale wiring); `-t 48:00:00`.

## 5. Bring back

- `slurm-<jobid>-<config>.out` for every job (full TLC output incl. any
  counterexample trace and the summary lines).
- On any **detector hit** in SafetyHunt/PoisonHunt: that log contains a
  first-of-its-kind witness — bring it back immediately, don't wait for the
  other jobs.

## 6. Round-1 results (2026-08-26, jobs 77808-77813)

| Job | Config | Outcome |
|---|---|---|
| 77808 | Safety | **BUG-6(a) CONFIRMED at 4-instance scale** — INV_NoReadyWhenNoPendingAllocs at depth 10; 4.4B gen / 1.44B distinct, 5h04m. Expected; deep hunt preempted → SafetyHunt. |
| 77809 | Poison4 | **BUG-6 CONFIRMED on poison paths** — same invariant; 887M gen / 272M distinct, 47min. Expected; deep hunts preempted → PoisonHunt. |
| 77810 | SafetyFixed4 | **UNEXPECTED: INV_NoDupAlloc violated** at depth 12; 12.8B gen / 4.0B distinct, 14h10m. Trace: `slurm-77810-SafetyFixed4.out` line 901 on. **Triaged: spec artifact** (TrailingRPR call-site wiring; corrected in DeferredAlloc.tla — fix design unaffected). |
| 77811 | — | Submission typo (`PoisonFixed4`); Poison4Fixed never ran. |
| 77812 | Big | Clean through 6.26B gen / 2.87B distinct, depth 9 (~12h); checkpointed, resumable. |
| 77813 | BigFixed | Clean through 1.06B gen / 470M distinct, depth 9; checkpointed, resumable. |

Round-1 positives worth keeping in mind: the two expected BUG-6
confirmations extend the local witnesses to 4-instance scale and to the
poison paths, and neither Big run found anything new in ~3B combined
distinct states.
