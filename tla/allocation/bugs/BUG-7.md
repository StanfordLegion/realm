# BUG-7 — `reuse_storage_immediate` drains prefix followers with the offsets-flavor release

**Status: TLC-unverified (the v1 model excludes redistricting) — confirmed
by code review with a hand-constructed trace, adversarially re-verified
twice. Pre-existing on `main` (mem_impl.cc:1372), independent of the
CAP/SWEEP/RPR fix bundle; unchanged on `mbauer-deferred-alloc-fixes`.
Found during the C++ fidelity review of the fix branch (2026-08-30).
One manifestation is an out-of-bounds write — memory-safety severity.**

## STATUS UPDATE (2026-08-31): fixed, test-first, detection confirmed

Fixed on branch **`mbauer-bug7-reuse-drain`** (worktree
`/Users/mebauer/realm-bug7`, changes left **uncommitted** for review). The
fix is the §5 direction verbatim: the oldest-drain do-while applies the
offsets-flavor release only when `it->inst == old_inst` and the void flavor
to every follower (~10 lines in `reuse_storage_immediate`).

Two regression tests added to the `DeferredAllocBadPathTest` suite
(`tests/unit_tests/deferred_alloc_test.cc`), one per drainable follower
kind, both constructed exactly in the §2 shape:

- `ReuseOldestDrainSweepsPlainFollower` — on unmodified `main` (Debug):
  `Assertion failed: (!redistrict_tags.empty()), function release, file
  mem_impl.cc, line 1855.` (manifestation 1, exactly as predicted).
- `ReuseOldestDrainSweepsRedistrictFollower` — on unmodified `main`
  (Debug): `Assertion failed: (offsets.size() == redistrict_tags.size()),
  function release, file mem_impl.cc, line 1856.` (the debug face of
  manifestation 3, the OOB write).

With the fix applied both pass, alongside the four pre-existing
`DeferredAllocBadPathTest` units; the tests also assert the release-build
obligations (children keep their promised offsets in `current_allocator`,
parents' slots recycle exactly once, no stranded tags). The fix does not
textually or semantically conflict with `mbauer-deferred-alloc-fixes`
(disjoint hunks in `reuse_storage_immediate`; the bundle's sweep and this
drain fix are complementary). v2 model confirmation remains pre-registered
per §6.

## 1. Summary

When a deferred redistrict's precondition fires, `reuse_storage_immediate`
takes the oldest-entry drain path and catches up `current_allocator` with a
do-while over the ready prefix (main mem_impl.cc:1365-1431). Every iteration
applies the **offsets-flavor** `PendingRelease::release(allocator, offsets)`
(main:1372) — but only the *first* entry is guaranteed to be `old_inst`'s
redistrict entry. The do-while continues through any **ready followers**
(main:1431), and those can be:

- a **plain destroy** (empty `redistrict_tags`), or
- a **different redistrict** with a different child count.

The offsets flavor asserts `!redistrict_tags.empty()` (main:1855) and
`offsets.size() == redistrict_tags.size()` (main:1856), and it overwrites
the caller's `allocated`/`offsets` — which the function tail then uses to
notify `old_inst`'s children (main:1527-1535). The sibling drain in
`release_storage_immediate` uses the void flavor (main:1641) and is immune;
only the reuse drain has the defect.

## 2. Concrete reachable scenario (legal client)

Heap of 6 units. All calls contract-clean: destroys after creates, forward
edges only, two independent user events `eA`, `eP` triggered in order.

| # | Call | Effect |
|---|------|--------|
| 1 | create P (size 4), create A (size 2) | both INSTANT_SUCCESS; heap full: P@[0,4), A@[4,6) |
| 2 | redistrict P → child C1 (size 1), precondition `eP` (untriggered) | `pending_releases = [R_P(redistrict, !ready, seq 1)]` (cc:1049-1051) |
| 3 | create D (size 3), no precondition | current full; future = current + split(P→C1) has hole [1,4) → **ALLOC_DEFERRED**, `last_release_seqid = 1` (cc:781-788); `pending_allocs = [D]` |
| 4 | destroy A, precondition `eA` (untriggered) | `pending_releases = [R_P, R_A(plain, !ready, seq 2)]` (cc:894-895) |
| 5 | trigger `eA` → `release_storage_immediate(A)` | non-oldest path (R_P is front): R_A **marked ready**, applied to `release_allocator` only, `deferred_dealloc_notify = true` (main:1724-1740); tail ARR gate fails (D=3 > the 2-unit hole in release) → R_A stays ready in the list |
| 6 | trigger `eP` → `reuse_storage_immediate(P)` | oldest path. Iteration 1: offsets-flavor on R_P — correct; C1 placed, `allocated=1`, `offsets` filled; unblock scan places D@[1,4) and empties `pending_allocs`. `++it` → R_A is ready → do-while continues (main:1431). Iteration 2: `allocated = it->release(current_allocator, offsets)` on the **plain** R_A → **BUG** |

## 3. Manifestations and severity

1. **Debug builds: hard abort on legal input.** Iteration 2 fires
   `assert(!redistrict_tags.empty())` (main:1855). With a redistrict
   follower of different child count, `assert(offsets.size() ==
   redistrict_tags.size())` (main:1856) fires instead. Real exposure for
   any DEBUG_REALM CI running redistricts concurrently with plain deferred
   destroys.
2. **Release builds, plain follower: child mis-notification + leak
   (#442 class).** The asserts compile out; `split_range` with zero new
   tags deallocates A correctly but returns 0, overwriting `allocated = 0`.
   The tail (main:1527-1535) then notifies **every child of P**
   `ALLOC_EVENTUAL_FAILURE` — while their tags were already placed in
   `current_allocator` by iteration 1. The children are failure-notified
   yet their ranges stay allocated forever: a permanent leak plus
   dead-slot/live-tag divergence, the instance-ID-reuse (#442)
   double-tracking class. (In debug, the consistency assert at
   main:1528-1529 fires first.)
3. **Release builds, redistrict follower with more children than
   `old_inst`: out-of-bounds write.** The follower's `split_range` writes
   `allocs_first[i]` for every child i (mem_impl.inl:210) into the caller's
   `offsets` vector sized for `old_inst`'s children — the inl:180 size
   assert is compiled out — a heap-buffer overflow. **This is
   memory-safety severity**, not just a bookkeeping error. Additionally
   `allocated`/`offsets` then describe the follower's children, so
   `old_inst`'s children are notified with another instance's counts and
   offsets.

## 4. Why it is pre-existing, and the fix branch's effect

The defect is on `main` and does not involve the CAP/SWEEP/RPR bundle: it
needs only [redistrict entry at the front of `pending_releases`] +
[ready follower behind it], reachable since the reuse paths landed. The
branch's new `sweep_ready_releases()` **reduces** exposure — stragglers
that previously waited for a later drain are now retired with the void
flavor (which handles both entry kinds) whenever `pending_allocs` drains —
but the do-while window itself is untouched: a follower that is ready at
the moment the redistrict's precondition fires is still drained through
main:1372's offsets flavor.

## 5. Proposed fix direction (no code changes yet)

Small and contained, mirroring `release_storage_immediate`'s drain: inside
the do-while, apply the **void flavor** `it->release(current_allocator)` to
every entry, and capture `allocated`/`offsets` via the offsets flavor
**only when `it->inst == old_inst`** (the first iteration by construction).
Followers' own obligations are already handled elsewhere: a ready plain
follower's dealloc ack flows through `deferred_dealloc_notify`
(main:1374-1376), and a ready redistrict follower's children were notified
at its mark-ready time with offsets that are intrinsic to the parent's
interval (see the sweep's offset-match argument, branch mem_impl.cc:1734).

## 6. Verification plan

This is redistrict territory — the v1 model deliberately excludes it
(DESIGN.md §1), which is exactly why TLC never saw it. It becomes the **v2
model's first pre-registered expected-FAIL**, the role BUG-6 played for v1:
model the reuse paths and `split_range` (already first on the v2 roadmap,
FUTURE-VERIFICATION.md §4), give the offsets flavor its size/emptiness
preconditions as ghost flags, and add a child-notify consistency invariant
(children notified success ⇔ tag live in current — an `INV_NoOrphanTags`
sibling). A 3-instance scripted config in the shape of §2 should violate it
in seconds; the §5 fix should flip it green.

## 7. Provenance

Found while adjudicating the fix branch's sweep design (choice-1 review of
child-notification timing); constructed and re-verified twice against
`main` (`git show main:src/realm/mem_impl.cc`), lines cited throughout.
