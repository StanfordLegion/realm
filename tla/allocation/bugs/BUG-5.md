# BUG-5 — `remove_pending_release` never revisits trailing pending allocs

**Status: TLC-confirmed in composition with FIX_CAP (Inversion witness,
`traces/Inversion-bug5-deadlock.txt`); latent-variant hunt in unfixed code
pending at scale (sapling Poison4). SHIPPING CONSTRAINT: FIX_CAP MUST NOT
LAND WITHOUT THE FIX BELOW (FIX_RPR).**

## 1. Summary

When a pending release is poison-cancelled, `remove_pending_release`
(mem_impl.cc:1538-1597) rewrites future history: it resets
`future_allocator := current_allocator` (cc:1556), walks the surviving
`pending_releases` replaying each onto the rebuilt future (cc:1562-1595),
and — interleaved with that walk — retries pending allocs whose
`last_release_seqid` is ≤ the seqid being walked (cc:1579-1594), keeping
the ones that fit and EVENTUAL_FAILing the ones that don't (cc:1587-1592).

The walk ends when the release list is exhausted. Any pending alloc whose
`last_release_seqid` **exceeds every walked seqid** is never reached by the
inner loop: it stays in `pending_allocs`, but it is **absent from the
rebuilt `future_allocator`** and is **never failed**. Such *trailing*
allocs are possible because `last_release_seqid` is a counter snapshot
(cc:784/801), not a reference to a surviving entry — the entry that raised
the counter may itself have been erased by an earlier poison cancellation.

Two distinct consequences:

- **Planning soundness (unfixed code, latent):** later admissions test
  against a future state missing the trailing alloc's reservation and can
  plan overlapping space — surfacing downstream as the cc:1668
  `assert(ok)` failing, the cc:1674-1691 offset cross-check failing, or
  overlapping placements.
- **Drain liveness (composition with the BUG-1 capped-admission fix):** a
  trailing alloc whose only funding release was poison-erased is never
  failed, so nothing downstream of its `e_created` can ever run —
  permanent hang. Under FIX_CAP this shape is **common, not exotic**:
  capped rejections poison `e_created` as the designed failure path
  (ii:1121-1122), C2 makes dependent destroy preconditions fire poisoned
  as a matter of course, and every such poisoned release runs
  `remove_pending_release`.

## 2. The TLC witness (Inversion, FIX_CAP+FIX_SWEEP, no user poison)

H=3; I1 size 3, A=I2 size 1, B=I3 size 2. Client is contract-clean (C1
topological order, C2 destroy-after-create). Model trace mapped to the C++:

| # | Call / event | Code path | State after |
|---|---|---|---|
| 1 | create(I1), no pre | cc:755 INSTANT_SUCCESS | I1 @ [0,3); heap full |
| 2 | create(A), pre = user event BA | cc:712-717 deferral; **cap(A) := seqid 0** (the fix's request-time snapshot) | A CREATE_PENDING |
| 3 | destroy(I1)=R1, pre deps {eCr(1), eCr(A)} | untriggered → push (cc:857-860), waiter (cc:918-921) | pending_releases=[R1 seq 1] |
| 4 | destroy(A), pre ⊇ eCr(A) | A still pending → DELAYEDDESTROY (cc:845-849) | marker only |
| 5 | BA fires → attempt A | capped test: funding ∅ (cap 0), current full → **INSTANT_FAILURE**; eCr(A) POISONED (ii:1121-1122); A's destroy entry pushed **seq 2** (cc:1146-1147) | list=[R1(1), RA(2)] |
| 6 | create(B), pre = user event BB | deferral; **cap(B) := seqid 2** | B CREATE_PENDING |
| 7 | destroy(A)'s pre fires POISONED | `remove_pending_release` fast path (pending_allocs empty, cc:1547-1553) erases RA | list=[R1(1)] |
| 8 | BB fires → attempt B | cap 2 ≥ seq(R1)=1 → funded by R1 → **ALLOC_DEFERRED, lastSeq = 2** | pending_allocs=[B(lastSeq 2)] |
| 9 | R1's pre fires POISONED (eCr(A) poisoned) | RPR rebuild path: fut/rel := cur (cc:1556-1557); walk erases R1 (saved seqid **1**, cc:1564-1570); inner loop bound: allocs with lastSeq ≤ 1 — **B (lastSeq 2) is trailing, never processed** | B still queued; fut = full heap, no B |
| 10 | destroy(B) requested, pre ⊇ eCr(B) | untriggered (eCr(B) UNFIRED) → push seq 3 | list=[RB(3)] |
| 11 | — | B unblocks only via a completing release (cc:1631-1753 drain / cc:1207 ARR / RPR); the only release waits on eCr(B), which waits on B | **permanent hang** (TLC deadlock, depth 14, 469/254 states) |

The guard itself is not at fault: the same config run invariant-only
(`-deadlock`) exhausts the full space with INV_InversionCapped and
SAFETY_PromisesKept holding everywhere. The deadlock is purely the
trailing-alloc omission. Every trigger ordering of this client ends in the
same stranding — the config cannot drain until BUG-5 is fixed.

## 3. Reachability in current, unfixed code

The same shape exists without FIX_CAP because `last_release_seqid` is the
counter value at admission, which can already exceed every *surviving*
seqid at admission time:

1. destroy(X) queued untriggered (seq 1); create(Y) with pre, destroy(Y)
   queued (DELAYEDDESTROY); Y's create INSTANT_FAILs at trigger →
   eCr(Y) poisoned intrinsically, Y's entry (seq 2) pushed, then erased by
   its poisoned destroy (fast path) — no user poison needed.
2. create(Z) too big for current → cc:768 rebuild admits Z **with
   lastSeq = cur_release_seqid = 2** funded by R1 (cc:781-784).
3. destroy(X)'s precondition fires poisoned → RPR rebuild: walked seqids =
   {1} < 2 → **Z is trailing**: not failed (though its only funding is
   gone → hang if no further releases), absent from the rebuilt future →
   a later release + admission can overspend Z's space
   (INV_NoOverlap / INV_FutureOffsetConsistency /
   INV_InOrderUnblockSucceeds — reviewer A's original shape).

**Honesty note:** we have a TLC witness only for the FIX_CAP composition
(§2). The unfixed variant above is a paper construction; the sapling
Poison4 run (toggles off, full battery) hunts it at scale, and a targeted
scripted config is easy to add if we want a local witness. The difference
in urgency stands regardless: in unfixed code poisoned releases are the
exception; under FIX_CAP they are the designed failure path.

## 4. Fix (modeled as the FIX_RPR toggle)

After the outer walk of cc:1562-1595 completes, run the cc:1579-1594 inner
loop **once more with no seqid bound** over the remaining `pending_allocs`
in order:

- `future_allocator.allocate(...)` succeeds → keep the alloc (its
  reservation is now refunded into the rebuilt future);
- fails → EVENTUAL_FAILURE + erase + poison `e_created` — the standard
  cascade, identical to the existing cc:1587-1592 arm (the `assert(found)`
  is trivially satisfied there since the erased target was already seen).

The cascade terminates: each poisoned-destroy RPR erases at least one
release entry. **Sequencing with FIX_SWEEP:** if the trailing pass empties
`pending_allocs`, the RPR-tail stranded-ready sweep (FIX_SWEEP site 3)
must run on the **post-trailing-pass** state — otherwise surviving ready
entries strand exactly as in BUG-6.

## 5. Shipping constraint

**FIX_CAP must not land in C++ without FIX_RPR** (and FIX_SWEEP, already
established). The capped fix converts BUG-1's silent hang into honest
failures *only if* the poison cascade actually drains; BUG-5's hole makes
the cascade strand any trailing dependent alloc, reintroducing a permanent
silent hang on a legal client — the exact defect class FIX_CAP exists to
eliminate. The three changes are one bundle.

## 6. Verification status

| Run | Toggles | Status / result |
|---|---|---|
| Inversion, deadlock ON | CAP+SWEEP (no RPR) | **FAIL = the witness** — deadlock depth 14, `traces/Inversion-bug5-deadlock.txt` |
| Inversion, `-deadlock` | CAP+SWEEP (no RPR) | PASS full space — isolates the deadlock to the stranding (guard sound) |
| Inversion, deadlock ON | CAP+SWEEP+**RPR** | **PENDING** (spec fork landing FIX_RPR) — must flip GREEN: B EVENTUAL_FAILs, cascade drains |
| Local fixed matrix re-run | full bundle | PENDING — all nine configs must stay green |
| Poison4 (sapling) | none | PENDING — hunts the unfixed overspend variant (§3) |
| SafetyFixed4 / Poison4Fixed / BigFixed (sapling) | CAP+SWEEP | BUG-5 detectors expected-possible until FIX_RPR joins; re-run with the full bundle flips expectation to green |
