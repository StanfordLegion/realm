# BUG-6: `assert(!it->is_ready)` at mem_impl.cc:772 is reachable by a legal client (stranded ready release)

**Status:** machine-confirmed by TLC (SafetyMini.cfg, 3 instances, heap = 3 units, sizes 2/2/1;
invariant `INV_NoReadyWhenNoPendingAllocs` violated at depth 9; 5.5M states, 10 s).
Trace: `traces/SafetyMini.trace.txt`. Independently constructed on paper by two Phase-1 reviewers before TLC confirmed it.

## Summary

A `PendingRelease` entry that was pushed *already-ready* (the `attempt_release_reordering`
failure path, mem_impl.cc:884-887) can be **stranded behind a non-ready entry** while the
oldest-release drain empties `pending_allocs`. Both cleanup sites that would normally retire
ready entries are skipped in that final drain iteration:

- the `release_allocator` rebuild at cc:1706-1717 requires
  `!successful_allocs.empty() && !pending_allocs.empty()` — the drain just emptied `pending_allocs`;
- the tail `attempt_release_reordering` at cc:1751-1753 requires `!pending_allocs.empty()` — same.

The system is left with `pending_allocs` empty and `pending_releases = [R_nonready, R_ready]`.
(A poison-path variant reaches the same stranded state through `remove_pending_release`, whose
inner replay loop can erase the last pending alloc while ready entries survive the walk,
cc:1587-1592 — BUG-6 variant (b), reachable at 3 instances with one poisoned event. Any fix must
cover that transition too; see fix (A).)
The *next* allocation request that misses `current_allocator` and needs the future rebuild walks
`pending_releases` at cc:768-779 and fires `assert(!it->is_ready)` at cc:772. **Debug builds abort
on a fully legal workload.** Release-build consequences are analyzed in "Impact" below — the
rebuild itself survives, but the stranded entry breaks the documented `release_allocator`
invariant (mem_impl.h:399-405) and composes with the BUG-4 mechanism into a **TLC-confirmed
permanent-leak path** (`Composite4.cfg`) that does **not** require poison.

## Concrete C++ execution (from the TLC trace)

Memory: 3 units. Instances: i1 (size 2), i2 (size 2), i3 (size 1). Every call below is
contract-clean (see "Legality"). `R#(inst, ready?, seq)` denotes a `PendingRelease`.

| # | Client call | Code path | State after |
|---|---|---|---|
| 1 | `create(i1)`, no precondition | `attempt_deferrable_allocation` cc:754-757: fits current | cur = {i1:[0,2)} |
| 2 | `create(i3)`, no precondition | same, cc:754-757 | cur = {i1:[0,2), i3:[2,3)} — heap full |
| 3 | `destroy(i1)`, precondition pending (user event) | cc:857-860: push, no future yet | releases = [R1(i1,¬rdy,1)] |
| 4 | `create(i2)`, no precondition | cc:762-788: misses current; future := current − i1; i2 fits at [0,2) → **ALLOC_DEFERRED**, `lastSeq = 1`; `release_allocator := current` (cc:787) | allocs = [A(i2, lastSeq 1)]; fut = {i2:[0,2), i3:[2,3)}; rel = cur |
| 5 | `destroy(i2)`, precondition = eCreated(i2) (unfired) | cc:892-895: future-free i2 (missing_ok), push | releases = [R1, R2(i2,¬rdy,2)]; fut = {i3:[2,3)} |
| 6 | `destroy(i3)`, precondition already triggered | cc:871-872: free i3 from rel and fut; ARR gate cc:1211-1215: front alloc i2 (size 2) cannot fit rel's 1-unit hole → **reordering fails**; cc:884-887: push **ready**, `deferred_dealloc_notify = true`; i3's tag stays in `current` | releases = [R1, R2, **R3(i3, READY, 3)**]; rel = {i1:[0,2)}; fut = {} |
| 7 | i1's destroy precondition triggers → `release_storage_immediate(i1)` | oldest path cc:1634-1702: rel-free i1 (cc:1636); drain R1 into current (cc:1641); unblock scan: A(i2).lastSeq = 1 < R2.seq = 2 → no break (cc:1653-1658); `current.allocate(i2)` at [0,2) — `assert(ok)` holds (cc:1668-1670); future cross-check takes the lookup-miss branch, finds R2 ¬ready (cc:1674-1691) — passes; **`pending_allocs` empties**; do-while stops at R2 (¬ready, cc:1700); prefix erase leaves **[R2(¬rdy), R3(READY)]**; rebuild skipped (`pending_allocs` empty, cc:1706); tail ARR skipped (cc:1751) | cur = {i2:[0,2), i3:[2,3)}; allocs = []; **stranded state** — `INV_NoReadyWhenNoPendingAllocs` violated |
| 8 | any `create(i4)` (size ≥ 1) | current is full → cc:762 falls into the rebuild loop cc:768-779; iteration reaches R3 | **`assert(!it->is_ready)` fires at cc:772** |

Step 8 needs a fourth allocation request, which SafetyMini's 3-instance client cannot issue —
the invariant at step 7 is the precursor state and is exactly what the assert guards against.

Note also: R3's `deferred_dealloc_notify` means i3's `notify_deallocation()` (profiling ack +
instance-slot recycle) is delayed until R3 drains — which now cannot happen until R2's
precondition (eCreated-i2-derived) triggers. Bounded under normal progress, but the latency is
inherited by whatever the stranding delays.

## Legality of the client

- Every destroy is requested after its instance's create request (topологically sorted; no back edges).
- `destroy(i2)` while i2 is still ALLOC_DEFERRED is legal: the *request* arrives early, but its
  precondition includes eCreated(i2), so it cannot *trigger* before creation (contract C2). This is
  the standard "run ahead" pattern the deferred allocator exists to support.
- `destroy(i3)` with an already-triggered precondition after i3's successful creation is plain usage.
- No poison, no failures, no user-event tricks anywhere in the trace.

## Impact

**(a) Debug builds (DEBUG_REALM / any build with asserts): hard abort.** Any workload that
reaches the stranded state and then requests one more allocation that needs deferral kills the
process at cc:772. The trace is short and un-exotic; debug CI and developer runs are exposed.

**(b) Release builds — the rebuild itself is functionally correct.** The stranded entry's tag is
still in `current_allocator` (that is precisely why its ack was deferred, mem_impl.h:427-436),
so the `missing_ok=true` replay at cc:778 actually *succeeds* in freeing it from the rebuilt
future state; the future picture and any resulting admission decisions are sound.

**(c) Release builds — the documented `release_allocator` invariant is broken, with two
consequences:**

1. *Lost reordering (conservatism).* On the next deferred admission, cc:787 sets
   `release_allocator = current_allocator` **without re-applying the stranded ready entry**
   (nothing after cc:787 does either; the only sites that re-apply ready entries,
   cc:1438-1447/1707-1717, require a later oldest-drain with both queues nonempty — site
   enumeration independently re-verified by the adversarial fix review, FIX-REVIEW.md §3.4).
   `release_allocator` therefore
   under-reports free space versus its definition "current + ready releases" (mem_impl.h:399-405),
   so `attempt_release_reordering`'s gate can falsely fail and allocations wait longer than needed.

2. *Confirmed composite with BUG-4 — permanent leak without poison.* Continue past step 8 in a
   release build: `create(i4)` is admitted DEFERRED (future rebuild is correct), and cc:787 resets
   `release_allocator := current` — **with the stranded i3 tag still allocated in it**. Now let any
   later instance's destroy arrive with a triggered precondition: cc:871-872 free it from
   rel/fut and call ARR. If ARR reaches **full success** (cc:1236-1252): `current := test`
   (derived from the stale rel, i.e. *still containing i3's tag*), and the ready-entry sweep at
   cc:1241-1250 **erases R3 and fires i3's deferred dealloc notify**. Result: i3's range is
   permanently allocated in `current_allocator` with no `pending_releases` entry referencing it,
   while the instance slot has been recycled — if the recycled `RegionInstance` ID ever re-enters
   this allocator, `allocated[tag] = idx` (mem_impl.inl:500) silently double-tracks. This is the
   same corruption class as issue #442, reached **without any poisoned event** (BUG-4 proper
   needs poison; this composite does not). **TLC-confirmed:** `Composite4.cfg` (5 instances,
   H=4, sizes 2,2,1,1,1, `SCRIPTED_COMPOSITE` client mode) violates `INV_CurrentMatchesGround`
   with `INV_NoOrphanTags` violated in the same state — 12-step trace at `traces/Composite4.txt`,
   93,795 states generated / 51,904 distinct, ~5 s, **zero poison**. Final state: tag 3 still in
   `current_allocator` with `instState[3] = DESTROYED` and `notifyCount[3] = 1` (dealloc notify
   already fired), no `pendingReleases` entry referencing it, and `readyAtRebuild = TRUE` —
   proving the path went through the BUG-6 stranding. Four instances are provably insufficient
   (the stranding consumes two instances, the rel-resurrection needs a fresh deferred create, and
   the ARR invocation needs a fresh request-time-triggered destroy), hence 5. The config is named
   after BUG-4, whose mechanism it completes.

**(d)** The cc:772 assert is load-bearing documentation: the rebuild, the cc:787 reset, and the
cc:871-872 strict frees all *assume* no ready entry exists outside a pending-allocs regime. The
assumption is false; every consumer of it should be re-audited once a fix direction is chosen.

## Candidate fixes (no code changed yet)

**(A) Recommended: a shared helper — `sweep_stranded_ready_releases()` — invoked at every
`pending_allocs` → empty transition.** There are three such transitions (the fourth emptying
site, ARR full-success cc:1236-1252, already erases all ready entries by construction and needs
nothing):

1. `release_storage_immediate`, oldest-drain tail (after the prefix erase, ~cc:1702);
2. `reuse_storage_immediate`, the mirrored oldest-drain tail (~cc:1433-1448);
3. `remove_pending_release`, after its replay loop (~cc:1596) — the inner loop can erase the
   last pending alloc while ready entries survive the walk (cc:1587-1592, variant (b)); a sweep
   placed only at the drain tails misses this poison-path stranding entirely.

The helper: if `pending_allocs.empty()`, walk the remaining `pending_releases` **in list order**;
for every `is_ready` entry, apply it to `current_allocator`, collect its
`deferred_dealloc_notify` ack, and erase it. The body must be **redistrict-aware**: a stranded
ready entry can carry `redistrict_tags` (the reuse path pushes ready-with-defNote at
cc:1037-1041), and such an entry requires the `split_range` flavor with child-offset collection
and `ALLOC_EVENTUAL_SUCCESS/FAILURE` notifications for the new instances (the
`it->release(current_allocator, offsets)` form used at cc:1372/1459) — not just
deallocate-and-ack.

*Correctness:* with `pending_allocs` empty there are no admitted futures to invalidate and
`future_allocator` is invalid-by-convention (rebuilt from `current` on next use, so it inherits
the sweep automatically). Plain frees are tag-keyed and commute — the final free-set is
order-independent; order only matters against interleaved *allocations*, and there are none.
Redistrict entries are also order-safe because `split_range` carves children inside the old
instance's own range (mem_impl.inl:200-262), independent of other swept frees — and sweeping in
list order makes the outcome identical to the eventual in-order drain the entries would have
received. *Notify timing is safe by construction* (confirmed by the adversarial fix review): the
sweep fires each `notify_deallocation` at the moment the tag leaves `current_allocator`, which is
exactly what the `deferred_dealloc_notify` contract demands (mem_impl.h:427-436 — the sweep *is*
a drain of the entry); the #442 guard condition (tag out of current before slot recycle) holds.
The sweep bounds i3-style ack latency, removes the stranded-entry precondition of the composite
leak in (c)(2) at the root, and makes the cc:772 assert a true invariant again, unchanged.

**(B) Safety-equivalent alternative: tolerate ready entries.** Weaken cc:772 to
`assert(!it->is_ready || tag-still-in-current)` (the rebuild replay already handles ready entries
via `missing_ok`), and re-apply all ready entries to `release_allocator` after **both** resets:
cc:787 (`attempt_deferrable_allocation`) **and** cc:1557 (`remove_pending_release` — the BUG-4
site, same stale-rel shape; cc:1309 needs nothing since ARR's partial path erases ready entries
before assigning). The fix review confirmed this **also closes the composite leak**: ARR then
builds `test` from a rel that already excludes the stranded tag, so `current := test` drops the
tag exactly when the entry is erased and its notify fired. The A-vs-B choice is therefore
**latency and hygiene, not safety**: (A) restores the documented invariant, bounds dealloc-ack
latency, and keeps "no ready entries outside a pending-allocs regime" true for every future
reader of `pending_releases`; (B) is the smaller diff but leaves stranded entries live longer
(ack latency remains, and all readers must stay ready-aware).

**(C) Recommendation:** (A) as primary, with (B)'s strengthened assert kept as belt-and-suspenders
documentation; (B) is an acceptable fallback if the sweep is judged too invasive, provided its
re-apply lands at *both* reset sites.

## Verification plan

- Model fix (A) — all three sweep sites — in a spec branch: `INV_NoReadyWhenNoPendingAllocs`
  flips from expected-FAIL to expected-HOLD; SafetyMini goes green; Poison4's expected variant-(b)
  violation (the `remove_pending_release` stranding) also flips green, which specifically
  validates sweep site 3; Smoke's acceptable-deadlock (BUG-1, unrelated) and EventLoop unchanged.
- `Composite4.cfg` is now an expected-FAIL config (the confirmed (c)(2) leak); the three-site
  sweep fix (A) — or (B) with both re-apply sites — must flip it green. `INV_NoOrphanTags` in
  no-poison Big / sapling Safety remains the open-hunt backstop.
- The `reuse_storage_immediate` mirror (sweep site 2, redistrict-aware body) gets model coverage
  when v2 adds redistricting.
