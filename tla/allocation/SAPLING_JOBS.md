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

# Sapling job list — deferred-allocation TLA+ campaign

## CURRENT ROUND: taint open sweeps (2026-09-07) — gates the taint C++ landing

The taint fix design (bugs/BUG-8.md resolution; DIST-DESIGN.md §5c) is
locally certified: full matrix 14/14, 2-node MAX_UE=0 open sweep fully
exhausted (304M gen / 53M distinct, full battery), adversarial review 0
blocking. Two open-client sweeps exceeded local capacity and run here.
**Pre-registered discipline: a violation in either run STOPS the taint C++
prototype from landing — triage first.** Expected outcome for both: green;
full exhaustion may be unreachable (the local MAX_UE=1 attempt was still
growing at 1.17B generated / 291.6M distinct, violation-free) — a
depth-complete bounded green well past the local depths (local witnesses
top out at depth 21) is the acceptance criterion, per the fix-bundle
precedent.

**2026-09-09 update — the gate is now SIX jobs.** The F4 model round
(cross-memory funding rings; DIST-DESIGN.md §5d) extended the spec with
FOREIGN_TOP/ROOT_UNION and added the TaintXMem* configs; the C++
consolidation on `mbauer-deferred-alloc-fixes` implements that final
design, so the landing gate is the full set below against the CURRENT
tree. Any TaintOpenUE/TaintOpen3 runs submitted before this update ran
the pre-F4 spec — their green is real evidence for what they checked but
does not open the gate; resubmit them. The sbatch script also changed
(TaintXMemOpenLive now gets the temporal `-deadlock` flag, matching
run.sh), so THE RSYNC IS MANDATORY — round 3 died from skipping it.

```sh
# from the laptop first — spec, configs, AND sapling_tlc.sbatch changed:
rsync -av --exclude states --exclude jtmp --exclude 'slurm-*' \
    ~/realm/tla/ sapling:realm-tla/

# on sapling:
cd realm-tla/allocation
# flat submission (queue permitting): --exclusive keeps each job's state
# queue alone on its node's /tmp. If the queue is contended, chain them
# with --parsable + `-d afterany:` as in round 3 instead.
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintOpenUE       # 2-node open sweep, user events ON
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintOpen3        # 3-node open sweep
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintXMemOpen     # 2-memory open sweep (F4 rings)
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintXMemOpenRoot # + pre-create roots (ROOT_UNION load-bearing)
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintXMemOpen6    # widest cross-memory space
sbatch --exclusive -t 24:00:00 sapling_tlc.sbatch TaintXMemOpenLive # temporal: no stuck allocs (fairness)
```

### Taint round 1 OUTCOME (2026-09-12, jobs 78166/78167 + 78237-78246)

**Zero violations anywhere — but the gate is NOT open.** Six of the
eight runs died on node-local disk exhaustion; the other two ended
healthy (one operator scancel to free nodes, one 48h time limit). None
reached the pre-registered bar (clean completed depth well past 21). Per
the pre-registered discipline this is *insufficient coverage*, not red:
no triage owed, no landing yet.

| Job | Config | Spec | Completed depth | Distinct states | Productive time / end |
|---|---|---|---|---|---|
| 78166 | TaintOpenUE | pre-F4 | 18 | 9.66B (55.0B gen) | 27h13m — disk full; JVM hung 5.7h more until scancel |
| 78167 | TaintOpen3 | pre-F4 | 14 | 13.65B (75.8B gen) | 32h57m — HEALTHY (41M s/min) at operator scancel |
| 78237 | TaintOpenUE | gate | 17 | 8.82B (49.3B gen) | 48h00m — HEALTHY at the 48h TIME LIMIT |
| 78238 | TaintOpen3 | gate | 13 | 9.66B (51.5B gen) | 39h33m — disk full; JVM hung 8.5h more to the limit |
| 78243 | TaintXMemOpen | gate | 13 | 2.15B (7.98B gen) | 7h44m — disk full; JVM hung ~40h more to the limit |
| 78244 | TaintXMemOpenRoot | gate | 10 | 1.65B (4.2B gen) | 4h32m — disk full, clean exit |
| 78245 | TaintXMemOpen6 | gate | 9 | 211M (362M gen) | 36min — disk full (6.4G free at start), clean exit |
| 78246 | TaintXMemOpenLive | gate | 11 | 79M (235M gen) | 61min — disk full, clean exit |

Positives: ~46B cumulative distinct states, every invariant (incl.
INV_NoFundingCycle) held; the Live run completed **10 full temporal
passes** (last over ~60M states) with no liveness violation; the
`-deadlock` dispatch fix for TaintXMemOpenLive verifiably took effect
(`extra=' -deadlock -gzip'` in its header).

Log-reading notes for future rounds: (1) a terminal `StatePoolReader:
... (No such file or directory)` after healthy progress lines is a KILL
ARTIFACT (the cleanup trap removes the state dir under the still-running
JVM), not a failure — that is how 78167/78237 look; the real disk deaths
end in `No space left on device`. (2) TLC's fp-merge disk-full exception
can leave a dead-but-not-exited JVM squatting on the node (~54 node-hours
lost across 78166/78238/78243); the script now runs a watchdog that kills
the JVM 10 minutes after a disk-full report if it hasn't exited itself.

Root cause of the disk deaths, entirely operational: the six gate jobs
were submitted within two minutes WITHOUT `--exclusive`; the XMem four
landed co-scheduled on 183G-disk nodes starting with 68/53/16/6.4G free
(78245 was doomed before it started). c0001's 435G baseline is FOREIGN
data (identical before 78166 and after its cleanup) — not ours to clean.
The sbatch script now REFUSES to start below `MIN_AVAIL_G` (default
300G, per-run overridable) and prints the top disk consumers, so a bad
placement fails in seconds instead of hours in short of the bar.

Time-limit finding (new binder): 78237 was still healthy at the full 48h
with disk to spare — for the 2-node open sweeps, TIME, not only disk, now
binds. At its closing rate (~2.3M ds/min and falling) "well past depth
21" is realistically a multi-week exhaustive run — submit OpenUE/Open3
with the longest -t the partition allows and treat their result as a
bound, while the XMem four (disk-starved at 4-17M ds/min, never
time-starved) are the runs a clean big node genuinely fixes.

### Taint round 2 — resubmission

```sh
# from the laptop: re-sync (sapling_tlc.sbatch changed - the disk floor):
rsync -av --exclude states --exclude jtmp --exclude 'slurm-*' \
    ~/realm/tla/ sapling:realm-tla/

# on sapling: FIRST clean stale /tmp/mebauer-tlc-* dirs off the compute
# nodes. The dirs are node-LOCAL, so survey and delete via srun per node.
# Precondition: `squeue -u $USER -h` is empty (then every mebauer-tlc-*
# dir anywhere is stale by definition).
#
#   # survey (read-only) - also reveals whether a node's consumed disk is
#   # even ours; other users' /tmp data is invisible-to-du here and NOT
#   # ours to clean (if a big node is full of foreign data: ask the
#   # admins, or pin the chain elsewhere with -w):
#   for n in $(sinfo -N -h -p cpu -o %N | sort -u); do
#     srun -p cpu -w "$n" -N1 -n1 -t 5 --immediate=30 --quiet bash -c \
#       'printf "%s: " "$(hostname -s)"; du -sh /tmp/${USER}-tlc-* 2>/dev/null || echo clean'
#   done
#   # delete pass (only our own dirs; busy nodes are skipped by
#   # --immediate - rerun later for stragglers):
#   for n in $(sinfo -N -h -p cpu -o %N | sort -u); do
#     srun -p cpu -w "$n" -N1 -n1 -t 5 --immediate=30 --quiet bash -c \
#       'rm -rf /tmp/${USER}-tlc-*; df -h /tmp | tail -1'
#   done
#
# Then submit as an EXCLUSIVE serial chain PINNED (-w) to the biggest
# clean c-class node the survey found - the XMem four first (the
# load-bearing F4 configs, disk-starved in round 1, genuinely fixed by a
# clean node), then Open3 (died on disk at 39.5h; a clean node buys it
# the full walltime). Round-1 reference: c0004 showed 306-432G free,
# c0001 only 261G (435G foreign baseline). The g-nodes' 183G disks
# cannot hold these runs.
cd realm-tla/allocation
# flat submission - the jobs are independent; --exclusive keeps each
# alone on its node and the MIN_AVAIL_G floor makes a small-disk
# placement abort in seconds (exit 75) instead of dying hours in:
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintXMemOpen
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintXMemOpenRoot
sbatch --exclusive -t 24:00:00 sapling_tlc.sbatch TaintXMemOpen6
sbatch --exclusive -t 24:00:00 sapling_tlc.sbatch TaintXMemOpenLive
sbatch --exclusive -t 48:00:00 sapling_tlc.sbatch TaintOpen3
# CAVEAT: an exit-75 abort is NOT auto-requeued. A few minutes after
# submitting, check for early deaths and resubmit those pinned to a
# surveyed big-disk node (they will pend until it frees - that is fine):
#   sacct -u $USER -S now-1hour -o jobid,jobname%24,state,exitcode
#   sbatch --exclusive -w <big-node> -t 48:00:00 sapling_tlc.sbatch <Config>
# All five pinned to ONE node without -d is also fine: exclusivity
# serializes them; only the order becomes slurm's choice.

# TaintOpenUE: do NOT blindly rerun. A healthy full-48h current-spec
# bound already exists (78237: completed depth 17, 8.8B distinct, zero
# violations) and its queue was near c0001's capacity at the limit - an
# identical rerun reproduces the same datum. Rerun ONLY given BOTH
# (a) partition MaxTime > 48h (check: sinfo -p cpu -o "%P %l") and
# (b) a >=400G-free node, e.g.:
#   sbatch --exclusive -w <big-node> -t 96:00:00 sapling_tlc.sbatch TaintOpenUE
# Otherwise record 78237 as the OpenUE datum.
```

Calibration honesty: even a clean 733G node projects OpenUE to completed
depth ~19-20 at best — likely still short of "well past 21", and 78237
shows TIME binds it too. If round 2 again ends short of the bar, the
principled options are (a) record the best-achievable bounded green and
present it as such, (b) a symmetry/view-reduction model round to deepen
effective coverage, (c) an explicit, documented revision of the gate. No
silent weakening.

### Taint round 2 OUTCOME (2026-09-14, jobs 78535-78539)

**Zero violations. One new-best datum; four floor aborts costing seconds
each (mechanism correct; the stale-dir cleanup step had been skipped).**

- **78535 TaintXMemOpen — NEW BEST:** full 24h on c0004 (306G free),
  completed depth 15 (round 1: 13), 6.53B distinct (3x round 1), 26.5B
  generated, healthy 20.5M s/min at the time limit. Now TIME-bound
  (~200G of 306G used) — 48h rerun on cleaned c0004 = definitive datum.
- **78536/78537/78538 (Root/Open6/Live):** refused, 224G on c0001 <
  300G floor (37G stale dir + 435G foreign baseline).
- **78539 TaintOpen3:** refused 7G short — 293G on c0004 (127G stale
  from scancel'd 78167 + 13G stranded by 78535's own truncated cleanup).

**Systemic finding: cleanup traps do not survive hard kills** (epilog
SIGKILL truncates rm -rf: 78167→127G, 78237→37G, 78535→13G stranded).
Script fix: `#SBATCH --signal=B:TERM@600` + TERM trap that kills TLC
first, giving cleanup a 10-min grace window.

**Sync discipline:** pulling results with a broad reverse rsync of tla/
clobbered this file and the script's newest fixes TWICE. Pull results
with a narrow filter instead:
`rsync -av 'sapling:realm-tla/allocation/slurm-78*' ~/realm/tla/allocation/`

### Taint round 3 (2026-09-15, nodes pre-cleaned: c0004=432G, c0001=261G)

```sh
# laptop first - the script changed again (TERM grace):
rsync -av --exclude states --exclude jtmp --exclude 'slurm-*' \
    ~/realm/tla/ sapling:realm-tla/

# sapling:
cd realm-tla/allocation
# c0004 (432G): the two big runs, pinned; --exclusive serializes them
sbatch --exclusive -w c0004 -t 48:00:00 sapling_tlc.sbatch TaintOpen3
sbatch --exclusive -w c0004 -t 48:00:00 sapling_tlc.sbatch TaintXMemOpen
# c0001 (261G ceiling, foreign baseline): the three never-run configs.
# Floor lowered to 200G (261G = 2.5x their 100G envelope; the 60G slack
# means foreign-data drift between the serial runs cannot spuriously
# abort a successor). PINNED - unpinned they can land on a 183G g-node
# and abort even at floor 200:
MIN_AVAIL_G=200 sbatch --exclusive -w c0001 -t 24:00:00 sapling_tlc.sbatch TaintXMemOpenRoot
MIN_AVAIL_G=200 sbatch --exclusive -w c0001 -t 24:00:00 sapling_tlc.sbatch TaintXMemOpen6
MIN_AVAIL_G=200 sbatch --exclusive -w c0001 -t 24:00:00 sapling_tlc.sbatch TaintXMemOpenLive
```

After round 3 every gate config has its best-achievable-on-sapling datum
and the pre-registered gate decision is on the table: accept documented
bounds, deepen via model reduction, or revise the gate explicitly.

### Taint round 3 OUTCOME (2026-09-15..18, jobs 78614-78618) — FINAL COVERAGE ROUND

**Zero violations, all five ran to their resource limits, every config
improved its bound.** Round-3 alone: ~39B distinct / ~152B generated.
Campaign cumulative across rounds 1-3: **~91B distinct states, zero
violations anywhere.**

| Job | Config | Completed depth | Distinct | Ran | End |
|---|---|---|---|---|---|
| 78614 | TaintOpen3 | 13 | **12.13B** (65.4B gen) | ~48h c0004 | TERM-grace at limit, healthy 22.6M s/min |
| 78615 | TaintXMemOpen | 15 | **11.98B** (50.6B gen) | 48h c0004 | TERM-grace at limit, healthy |
| 78616 | TaintXMemOpenRoot | 12 | **7.52B** (21.9B gen) | 21.8h c0001 | disk full → WATCHDOG killed hung JVM (first live firing) |
| 78617 | TaintXMemOpen6 | 10 | **7.22B** (12.8B gen) | 20.3h c0001 | disk full, clean exit |
| 78618 | TaintXMemOpenLive | 12 | 378M (1.22B gen), **17 temporal passes**, last over a 1.43B-node behavior graph | ~24h c0001 | TERM-grace at limit |

Infrastructure fully validated in the wild: the TERM-grace cleanup left
c0004 at its clean 264G-used baseline for the back-to-back 48h runs
(78615 started seconds after 78614 with full disk), the watchdog fired
exactly once and correctly (78616), and the floor produced zero spurious
refusals.

**FINAL GATE LEDGER (best-achievable-on-sapling bounds, current spec,
all zero-violation):**

| Config | Completed depth | Distinct | Source |
|---|---|---|---|
| TaintOpenUE | 17 | 8.82B | 78237 (r1, 48h, time-bound) |
| TaintOpen3 | 13 | 12.13B | 78614 (r3, 48h, time-bound) |
| TaintXMemOpen | 15 | 11.98B | 78615 (r3, 48h, time-bound) |
| TaintXMemOpenRoot | 12 | 7.52B | 78616 (r3, disk-bound @261G) |
| TaintXMemOpen6 | 10 | 7.22B | 78617 (r3, disk-bound @261G) |
| TaintXMemOpenLive | 12 + 17 temporal passes | 378M | 78618 (r3, time-bound) |

The pre-registered bar (depth-complete green well past 21) is NOT met by
any config and is unreachable on this hardware for open sweeps (time- or
disk-bound at depths 10-17). The pre-registered decision is now due:
(a) accept these documented bounds as the verification datum,
(b) a symmetry/view-reduction model round to raise effective depth, or
(c) an explicit, recorded gate revision. No silent weakening. The
decision is moot for landing purposes unless the taint C++ goes forward.

### GATE DECISION (2026-09-20, Mike Bauer): option (a) — ACCEPTED

The bounds in the final ledger above are accepted as the verification
datum for the taint design. Basis: zero violations across ~91B cumulative
distinct states over three rounds on the current spec; every closed
(non-open) config in the local matrix fully exhausted; 17 clean temporal
passes on the cross-memory liveness config; the open sweeps green to
their hardware-bound depths (10-17), which exceed every depth at which
any toggles-off counterexample was ever observed in the corresponding
config. Known limits, stated plainly: the open sweeps are bounded
verification, not proofs — behaviors longer than the completed depths
are uncovered, and the original "well past 21" aspiration was set
before the open spaces' true growth rates were measured.

**Consequence: the verification gate on the taint C++ landing is OPEN.**
Whether the taint implementation (uncommitted on
`mbauer-deferred-alloc-fixes`) actually ships remains a separate
engineering decision (complexity vs the round-trip fallback on
`mbauer-bug8-roundtrip-fallback`, itself certified by the DistRT*
matrix). No further sapling runs are owed for either design.

Notes: the sbatch script dispatches Dist*/Taint* configs to
MCDistDeferredAlloc and keeps the deadlock check ON for them, EXCEPT the
two temporal Live configs which pass `-deadlock` (mirrors run.sh's flag
table). Standing storage rules apply automatically (node-local /tmp,
gzip, checkpoint-sync OFF — size the time limit for one shot). Verify
each job's log header shows `-gzip` and "checkpoint-sync=OFF" before
walking away. Bring back the slurm-<jobid>-<config>.out files; the
deepest completed `Progress(N)` line is the depth datum even if the run
hits its time limit. On any violation: bring that log back immediately —
it stops the C++ landing until triaged.

---

# HISTORICAL: fix-bundle rounds — ROUND 2

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
