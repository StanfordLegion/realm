# Realm Deferred Allocation — TLA+ Verification Campaign: Findings

Status: **Phase 4 (local verification) complete; fix bundle modeled and
locally verified.** Sapling runs pending (see `SAPLING_JOBS.md`). No Realm
source code has been modified — all proposed fixes are analysis-only,
recorded in `bugs/`.

**Fix-bundle addendum (2026-08-26):** the three candidate fixes are modeled
as spec toggles — `FIX_CAP` (BUG-1 capped admission), `FIX_SWEEP`
(BUG-6/BUG-4 stranded-ready sweep), `FIX_RPR` (BUG-5 trailing-alloc replay)
— and the full bundle is **green across the entire local matrix**, including
full-exhaustion passes at the exact bounds where the unfixed model fails and
the Inversion client green with deadlock checking ON. The validation round
itself produced a fourth adjudicated report: **BUG-5 is confirmed
load-bearing for the cap fix** (a capped rejection's poison cascade strands
any trailing dependent alloc; witness
`traces/Inversion-bug5-deadlock.txt`), so the bundle is indivisible —
FIX_CAP must not land in C++ without FIX_RPR. Four bug reports are final:
`bugs/BUG-1.md`, `bugs/BUG-5.md`, `bugs/BUG-6.md` (BUG-4 escalated inside
BUG-6), plus `bugs/FIX-REVIEW.md`. Details in EXPECTED.md's "Fix
validation" section. *Round-1 sapling addendum (2026-08-26):* the scale
runs confirmed BUG-6 at 4 instances and on the poison paths, and surfaced
one spec wiring artifact in the FIX_RPR toggle (TrailingRPR call-site fed
the full survivor list; corrected and re-validated green across the whole
local matrix — the C++ blueprint gained the continue-from-cursor rule,
`bugs/DUPALLOC-TRIAGE.md`); Big/BigFixed remain in progress on sapling.

**C++ port started (2026-08-30):** the port of the indivisible bundle began
on the accumulated evidence — local full exhaustions plus sapling job 77858
(7.52B distinct states violation-free, depth 11 complete, ~3.9B states into
depth 12; the pre-registered depth-13 gate proved physically out of reach on
single-node disk). `FUTURE-VERIFICATION.md` records the options if deeper
verification is ever wanted.

The model covers deferred instance allocation and deletion in
`LocalManagedMemory` (`src/realm/mem_impl.{h,cc,inl}`): the
current/future/release allocator triple, `pending_allocs`/`pending_releases`,
seqid ordering, release reordering, and poison handling. Instance
redistricting is deferred to v2 (roadmap in `DESIGN.md` §9). Client behavior
is constrained by the documented contract: topologically sorted requests (no
back edges), destroy preconditions incorporate the created event.

## Machine-confirmed bugs (3)

All three reproduce with **legal clients** under the stated contract, with
TLC traces on disk and line-cited C++ executions in the bug reports.

### BUG-1 — Event-loop deadlock from trigger-time ordering  (`bugs/BUG-1.md`)

A deferred create is inserted into the release/alloc total order when its
precondition **triggers** (`mem_impl.cc:784/801`), but its `e_created` was
published at **request** time (cc:712-717 records nothing). Any release
requested in the request→trigger window may legally depend on that
`e_created`; the future rebuild (cc:768-781) counts its space anyway, so
Realm plans the allocation out of a release that can only happen after the
allocation completes. Result: permanent silent hang — worse, the mapper is
first told `InstanceAllocResult{success=true}` (inst_impl.cc:1140-1142).
This is the bug Sean suspected in the design talk (~24:47); his conservatism
rule is enforced against the wrong clock. Minimal witness: 2 instances +
1 user event (`traces/EventLoop.trace.txt`, 7 states; also `traces/Smoke-run1.txt`,
`traces/Liveness-bug1.txt` with passing `LivenessNoCross` control).

**Fix (hardened by adversarial review, `bugs/FIX-REVIEW.md`):** fund a
deferred admission only from {releases with seqid ≤ request-time cap} ∪
{releases whose precondition has already triggered clean at admission},
with a monotone-cap queue rule; capped-fit failure → honest
`ALLOC_INSTANT_FAILURE` instead of a hang. The pure request-time cap is
**not** shippable: it false-fails the canonical GC-ripple pattern, and no
arrival-order-only policy can distinguish that client from the cycle client
(identical arrival sequences; the difference is in the event graph Realm
cannot see). **Open question for Legion:** does the GC-ripple `e_pre` fire
at-or-after the victims' destroy preconditions? If not, the fallback is a
ballistic-style declaration flag on destroy.

### BUG-6 — Stranded ready release; `assert(!it->is_ready)` reachable  (`bugs/BUG-6.md`)

An ARR-failure pushback (cc:884-887) leaves a READY entry behind a non-ready
one; the oldest-drain then empties `pending_allocs`, and both cleanup sites
(cc:1706 rebuild, cc:1751 tail ARR) are skipped precisely because the queue
just emptied. The next allocation needing deferral fires the cc:772 assert in
debug builds. In release builds the future rebuild survives, but the
documented `release = current + ready releases` invariant (mem_impl.h:399-405)
is broken from cc:787 on, costing reorderings and delaying dealloc acks.
Witness: 3 instances, H=3, 9 steps, no poison (`traces/SafetyMini.trace.txt`).

**Fix:** a shared `sweep_stranded_ready_releases()` helper at all three
`pending_allocs`→empty transitions (both oldest-drain tails **and**
`remove_pending_release`), sweeping ready entries into `current_allocator`
in list order, redistrict-aware, firing their deferred notifies. Fix B
(tolerate-ready + re-apply after both resets) is a safety-equivalent smaller
fallback.

### BUG-6→BUG-4 composite — Permanent range leak, notify-while-tag-live  (`bugs/BUG-6.md` §6)

After the BUG-6 stranding, the next deferred admission resets
`release_allocator := current` **including the stale-but-ready tag**
(cc:787; same shape as the cc:1556-1557 reset flagged as BUG-4); a later
triggered destroy reaching ARR full-success swaps that state into
`current_allocator` and erases the ready entry, firing its deferred dealloc
notify — the tag is never deallocated. Permanent leak + instance-slot
recycling with the tag still tracked: the #442 double-tracking class, with
**zero poison involved**. Witness: `Composite4.cfg`, 5 instances (4 provably
insufficient), 12-step trace (`traces/Composite4.txt`), violates
`INV_NoOrphanTags`/`INV_CurrentMatchesGround`. The BUG-6 sweep fix closes it.

## Registered candidates pending sapling runs

- **BUG-3** — soundness of the in-order unblock `assert(ok)` (cc:1670) and
  the future-offset cross-check (cc:1674-1691) after partial reorderings
  have rewritten history. Hunted by `Safety.cfg` (4 inst; ~110M+ distinct
  states locally at 12 min, sapling-bound).
- **BUG-4 standalone (poison variant)** — the cc:1556-1557 stale-release
  reset reached via a poisoned release. Hunted by `Poison4.cfg`.
- **BUG-5** — `remove_pending_release` never replays trailing pending allocs
  (lastSeq beyond every walked seqid) onto the rebuilt future (cc:1562-1595)
  → potential overlapping future planning. Hunted by `Poison4.cfg`.
- **BUG-2** — whether the client contract truly closes every
  `missing_ok=false` path for failed allocations (C2-off configs document
  what breaks without the contract).
- **BUG-7** (`bugs/BUG-7.md`) — `reuse_storage_immediate`'s oldest-drain
  applies the offsets-flavor release to every drained prefix entry
  (main:1372): debug abort / child mis-notification leak / OOB write on
  mixed redistrict-plain prefixes. Pre-existing on main; found during the
  C++ fidelity review of the fix branch; TLC-unverified — first
  pre-registered expected-FAIL for the v2 (redistrict) model.

Submit with `SAPLING_JOBS.md`: `sbatch sapling_tlc.sbatch Safety` (est.
1-6 h), `Poison4` (several hours), `Big` (use `-t 48:00:00`). Checkpoint
resume via TLC `-recover` is wired into the sbatch script.

## Model confidence

- Design doc (`DESIGN.md`, 828 lines) adversarially reviewed twice before
  authoring; every behavioral claim carries a `file:line` citation.
- Spec (`DeferredAlloc.tla`, 949 lines) fidelity-reviewed by six independent
  function-level passes: **zero blocking divergences**; five minor fixes
  applied and revalidated (baseline outcomes reproduced exactly).
- C++ asserts are modeled as ghost-flag invariants, never action guards, so
  assert-reachable states are reported, not pruned. Known bugs (BUG-4/5) are
  deliberately present in the spec.
- First-fit abstraction (tag→interval map, minimal-feasible-offset) proven
  equivalent to `BasicRangeAllocator`'s address-ordered first fit, twice
  independently.
- Every expected-green config is green; every expected-red config is red for
  the predicted reason (`EXPECTED.md` is the authoritative matrix).

## Running

Local: `./run.sh` (Smoke → EventLoop → SafetyMini → Composite4 → Liveness →
LivenessNoCross; seconds each at current bounds). Safety-only runs need a
writable tmp/metadir; **temporal (liveness) runs need an unsandboxed JVM**
(TLC's liveness checker binds an RMI socket). Sapling: `SAPLING_JOBS.md`.

## v2 roadmap (DESIGN.md §9)

Redistricting (`split_range`/reuse paths — required before trusting the
BUG-6 sweep fix's redistrict arm), alignment, duplicate releases from
network delays, dealloc-completion feedback shapes, instance-ID reuse
interaction (#442), and the v-next fix-validation specs pre-registered in
both bug reports (capped-ADA variant; three-site sweep variant).
