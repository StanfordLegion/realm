# TLA+ model: Realm deferred instance allocation

Models `LocalManagedMemory`'s deferred allocation/deletion protocol
(`src/realm/mem_impl.cc`): the three heap states (`current`, `future`,
`release` allocators), the `pending_allocs`/`pending_releases` queues, release
reordering, and poisoned-precondition cancellation, together with a client
bound by Realm's contract (topologically sorted requests, destroy-after-create).
The goal is to model-check the code's own assertions and its two promises —
"never say an allocation will succeed and then fail" and "never get stuck" —
and to adjudicate the bug hypotheses catalogued in `DESIGN.md` §8.

## Files

| File | Purpose |
|---|---|
| `DESIGN.md` | Authoritative C++→TLA mapping; every action cites `mem_impl.cc` lines. Read this first. |
| `DeferredAlloc.tla` | Protocol spec: allocator operators, state, actions. |
| `MCDeferredAlloc.tla` | Client/environment model, constants, ghost variables, invariants. |
| `Smoke/Safety/EventLoop/Liveness/Poison4/Big.cfg` | Configurations (constants, checks, expectations — see `DESIGN.md` §7 and the `run.sh` header). |
| `EXPECTED.md` | Expected pass/fail per config with the bug each failure corresponds to (written during verification). |
| `run.sh` | Local runner. |
| `sapling_tlc.sbatch` | Slurm script for the expensive configs on sapling. |
| `FUTURE-VERIFICATION.md` | Options for deeper verification (scale-run tuning, distributed TLC, symmetry/simulation, v2 model debt, C++-side harnesses) if ever wanted beyond the accepted baseline. |

## Running locally

```sh
./run.sh sany        # parse-check only
./run.sh Smoke       # one config
./run.sh             # default sweep: Smoke, EventLoop, Safety, Liveness
```

Overrides: `JAVA`, `JAR`, `WORKERS`, `HEAP`, `JTMP` (see `run.sh` header).
Default jar: `../barrier/tools/tla2tools.jar`; default java: homebrew openjdk.

## Running on sapling

`Poison4` and `Big` (and `Safety` at larger bounds) are projected > 1 hour —
run them on sapling:

```sh
# on sapling, with the repo cloned anywhere:
cd <repo>/tla/allocation
sbatch sapling_tlc.sbatch Poison4
sbatch sapling_tlc.sbatch Big
```

Resource defaults (partition `cpu`, 24 h, 40 cpus, 128 GB) are guesses —
override on the command line (`sbatch -p ... -t ... -c ... --mem=...`).
TLC checkpoints hourly; see the sbatch header for `-recover` resume
instructions.

## Status

**TrailingRPR correction re-validated locally; sapling round 2 fully
unlocked.** The corrected bundle passed the entire hardened local matrix
(2026-08-26): fast set green incl. Inversion deadlock-ON (478/255), both
SafetyMini-scale full exhaustions green (SafetyMiniFixed 64.68M gen /
23.54M distinct; SweepOnly 53.30M / 20.55M), LivenessFixed green, and the
toggles-off regressions exact (EventLoop deadlock@7, SafetyMini
violation@9). Round 1 had confirmed BUG-6 at scale (4-instance Safety and
poison-path Poison4); the SafetyFixed4 INV_NoDupAlloc violation was
triaged as a **spec artifact** (FIX_RPR call-site wiring fed TrailingRPR
the full survivor list instead of the trailing remainder — two-line
correction applied, fix design unaffected; `bugs/DUPALLOC-TRIAGE.md`).
`INV_NoDupAlloc` is now checked in every bundle config. See
`SAPLING_JOBS.md`: ALL round-2 jobs submittable — SafetyHunt/PoisonHunt/
Big (resume) plus the bundle jobs (SafetyFixed4/Poison4Fixed/BigFixed,
fresh starts).

Prior milestone (still true modulo the triage): fix bundle (CAP+SWEEP+RPR)
verified locally.

The candidate fixes are modeled as three spec toggles — `FIX_CAP` (BUG-1
capped admission), `FIX_SWEEP` (BUG-6/BUG-4 stranded-ready sweep), and
`FIX_RPR` (BUG-5 trailing-alloc replay in remove_pending_release) — and
validated by the 2026-08-26 local matrix (EXPECTED.md "Fix validation"
section): the full bundle is green on every local config, including
full-exhaustion passes at the exact bounds where the base model fails and
**Inversion green with deadlock checking ON**. The bundle is indivisible:
the two-toggle round's Inversion deadlock
(`traces/Inversion-bug5-deadlock.txt`) proved pre-existing BUG-5 is
load-bearing for drain liveness under FIX_CAP — **FIX_CAP must not land in
C++ without FIX_RPR** (bugs/BUG-5.md). Regression: all toggles-FALSE
baselines reproduce exactly.

Machine-confirmed Realm bug candidates, each with a saved trace and a
written report:

- **BUG-1** (`bugs/BUG-1.md`) — deferred-create ordered at trigger time, not
  request time → silent event-loop deadlock. Confirmed three ways: EventLoop
  deadlock (7 states, `traces/EventLoop.trace.txt`), Smoke deadlock
  (`traces/Smoke-run1.txt`), and a temporal-property lasso
  (`traces/Liveness-bug1.txt`) whose same-bounds control
  (`LivenessNoCross.cfg`, `traces/LivenessNoCross-pass.txt`) passes —
  isolating the cross-instance destroy dependency as the cause.
- **BUG-6** (`bugs/BUG-6.md`) — the `assert(!it->is_ready)` at
  mem_impl.cc:772 is reachable by a legal poison-free client
  (`traces/SafetyMini.trace.txt`), and its **composite with the BUG-4 stale-release
  mechanism is confirmed** (`Composite4.cfg`, `traces/Composite4.txt`):
  permanent heap-range leak with the dealloc notify already fired (#442
  class), no poison involved.

Per-config outcomes and iteration protocol live in `EXPECTED.md`. No Realm
source changes have been made — findings feed the bug reports first
(`DESIGN.md` §8).
