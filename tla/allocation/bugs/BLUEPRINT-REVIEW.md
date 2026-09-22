# Blueprint review: FIX_CAP / FIX_SWEEP as landed in DeferredAlloc.tla

Reviewer: blueprint-review fork. Scope: (1) adversarial soundness review of the
landed fix semantics (DeferredAlloc.tla ~1152 lines: `ApplyCappedFrees`,
`CanonReplay`, `ADAResCap`, `ADAResv`, `Sweep`, `reqCap`) against the agreed
design (bugs/BUG-1.md, bugs/BUG-6.md, bugs/FIX-REVIEW.md, and Mike's
accepted-behavior-change decision); (2) the definitive C++ change list the spec
corresponds to. No TLC runs were performed here (verification fork owns that);
all soundness arguments are on-paper against the spec text and mem_impl.{h,cc,inl}.

## 0. Headline caveat (read first): the spec models the PURE cap, not BUG-1.md's union rule

The landed `ADAResCap` funds an admission from **{surviving releases with
`seq <= cap`} only** (spec line ~299: "NO ready-fold in v1"). bugs/BUG-1.md's
recommendation section still describes the fuller repaired rule — cap **∪
releases whose precondition has already triggered clean** — including an
implementation item ("PendingRelease stores its precondition Event") that the
pure cap does **not** need. Both are sound; the pure cap is strictly more
conservative and was chosen as v1 after Mike accepted spurious instant-fails
("I would rather be correct... even if Realm's behavior might change").

Consequences:

- **What TLC validates is the pure cap.** The union term is UNVERIFIED. The
  C++ v1 must implement the pure cap exactly as modeled. If the union term is
  ever wanted (it saves the GC-ripple pattern under memory pressure when
  funding destroys are stuck as ready entries behind other pending work),
  extend `ApplyCappedFrees`/`CanonReplay` to also free `is_ready` survivors
  with `seq > cap` and re-run the whole matrix FIRST.
- v1 does NOT need `PendingRelease` to store its precondition `Event` — drop
  that item from the v1 change list (a real simplification vs BUG-1.md).
- BUG-1.md should gain a short "v1 as modeled/verified = pure cap" note
  (documentation action for the report owner; not edited here per scope).
- Note the pure cap still gets most of ARR's opportunism for free: ARR funds
  only from `current` + already-triggered (ready) releases, which are
  dependency sinks — see §1.9.

Verdict for §0: **NEEDS-DOCUMENTATION, not a spec issue.** No blocker.

## 1. Soundness verdicts on the eight design decisions (+ ARR interaction)

### 1.1 cur-first fast path kept under FIX_CAP — SOUND
`ADAResCap` guards the fast path with `paA = <<>> /\ CanAlloc(curA, sz)`
(spec ~286), preserving the legacy pending-allocs-empty gate (cc:754-757).
With pending allocs it falls through to the monotone guard + capped test —
never a direct-cur admission, so queue monotonicity cannot be bypassed.
Cur-funded allocations consume only space free *now*; no dependency on any
pending release, hence no cycle risk. The legacy `prA = <<>> -> IF` special
case (cc:762-765) is subsumed: an empty capped funding set degenerates the
test to cur-only, which already failed.

### 1.2 missing-ok frees in the capped/canonical replay — SOUND
Mirrors cc:778. A missing tag can only make the replay state *less* free
(the free no-ops), i.e. the capped test can only under-admit — the safe
direction (spurious IF, accepted). Over-admission via missing-ok is
impossible; v1's one-release-per-instance rule excludes double-entry masking.
The DELAYEDDESTROY self-release (`seq = cap + k > cap`) is excluded from the
admission bound and applied after the alloc in the fut rebuild — matching
cc:1146-1153 semantics and guaranteeing an instance never funds itself.

### 1.3 queued-alloc placement failure in canonical replay -> flag + IF — SOUND (TLC is the adjudicator)
The determinism claim: every queued alloc re-places successfully in a
canonical replay from the current `cur`. The inductive argument: (i) at its
own admission each alloc was verified against the canonical state; (ii) the
drain applies operations in exactly canonical order, and placement agreement
between drain and canonical fut is `INV_FutureOffsetConsistency` — checked by
TLC under FIX_CAP; (iii) after an ARR swap, `cur' = g.test` and the remaining
survivors are all non-ready, so `CanonReplay(cur', ...)` replays exactly
ARR's own successful `Replay` (same gate `seq <= lastSeq`, same base) — the
directive's erased-ready concern resolves because erased ready entries'
space is already inside `cur'` and they are gone from the list, while
ready-but-unerased entries still hold their tags in `cur`, which is exactly
the state `CanonReplay` frees them from. I could not construct a divergence.
The spec surfaces any residual hole via `capAssert -> structuralAssertFailed`
so TLC hunts it; **the C++ must carry the same assert** (§2.3).

### 1.4 fut unchanged on capped-test failure — SOUND
Every `fut` read in the module is guarded by `pendingAllocs # <<>>` (the
validity convention): the ADA nonempty test, RequestDestroy's fut frees, the
TriggerCreate dd-free, UnblockScan's cross-check, ARR's pass-through. On a
failed admission with the queue empty, fut is invalid-by-convention and the
next successful admission rewrites it from scratch (`ADAResCap` never reads
the stale value). Legacy's cc:790-791 stale write is dead state in legacy
too; not writing it is safe and cleaner. C++: do not clobber
`future_allocator` on failed admissions.

### 1.5 ready survivors with seq <= cap DO fund — SOUND
`seq <= cap` iff the entry was *pushed* before the create request (seqids are
assigned monotonically under the mutex at the push sites cc:859/885/895/1147).
A destroy pushed before the create request had its precondition fixed before
`e_created(i)` existed, so it cannot depend on it — ready or not, it is
cycle-safe funding. The one wrinkle: a destroy *requested* early but pushed
late (DELAYEDDESTROY, push at create-trigger cc:1146-1147) gets `seq > cap`
and is excluded — an error in the conservative direction only. Redistrict /
reuse pushbacks (cc:1037-1041) are v2 scope; the same push-time argument will
apply but must be re-checked when redistricts enter the model.

### 1.6 RPR sweep rel handling + poisoned ack suppression — SOUND
When the queue empties in RPR, `rel'` keeps the pre-sweep `cur` (legacy
cc:1557 value) while `cur'` is swept — `rel` is invalid when the queue is
empty and the next first admission overwrites it (`rel := cur`, cc:787
analog), so the unswept value is dead state; C++ need not touch
`release_allocator` in that branch. When the queue stays nonempty, `rel' =
cur + surviving ready frees` (strict, cc:1713-1714 pattern) — exactly
h:399-405 restored; this is the BUG-4-standalone fix. The poisoned instance's
own ack stays suppressed (its entry is erased by the walk, never swept;
matches the cc:1789 guard); swept defNote acks fire exactly when their tags
leave `cur` — the h:427-436 contract.

### 1.7 strict sweep frees — SOUND for v1, note for v2
Ready survivors' tags are in `cur` by the deferred-dealloc invariant, so
strict is correct and doubles as a corruption detector (`missingFree`).
v2 caveat: duplicate releases from network delays could legally present two
ready entries for one instance; the second strict free would false-trip.
Keep strict in the v1 C++; revisit under the v2 duplicate-release model.
Separately: the C++ sweep must be redistrict-aware (BUG-6.md fix A) even
though the v1 model has no redistricts — see §2.4.

### 1.8 append-free ≡ canonical-interleave for new releases — SOUND
A new release's seq (`seqCtr + 1`) exceeds every queued watermark (watermarks
are caps `<=` the seqCtr at their admission), so it sorts after every queued
alloc in canonical order — appending its free to a canonical `fut` is the
canonical result. Post-ARR-partial `fut := rp.tf` is canonical because ARR's
`Replay` uses the same gate (`seq <= lastSeq`) as `CanonReplay`, the same
base (`g.test = cur'`), and post-swap survivors are all non-ready; under
FIX_CAP `lastSeq == cap == watermark`, so ARR's boundaries and the canonical
watermark rule coincide *by construction*. The drain preserves canonicity by
the prefix property of deterministic replay. ARR full-success leaves `fut`
stale — harmless: the queue is empty, fut invalid-by-convention.

### 1.9 ARR × FIX_CAP interaction — SOUND
ARR funds placements only from `release_allocator` = `current` + ready
(already-triggered) releases. Triggered releases are dependency sinks — they
can never be waiting on any `e_created` — so ARR needs no cap awareness at
all: it is precisely the safe opportunistic half of the funding story, kept
verbatim. After an ARR partial swap the remaining queue is a suffix, so the
non-decreasing-watermark (monotone-cap) invariant is preserved, and §1.8
gives fut-canonicity for the next `ADAResCap` admission.

### Also verified
- Request-triggered creates pass `cap = seqCtr`: the monotone guard can never
  fire for them and the capped test equals the legacy behavior given
  fut-canonicity — the dominant path is behavior-identical (and §2.2 notes
  the O(1) C++ fast path this licenses).
- The sweep runs whenever the queue is empty (not only on the transition) —
  idempotent, establishes `INV_NoReadyWhenNoPendingAllocs` inductively; all
  four queue-emptying transitions are covered (two sweeps + two ARR
  full-success paths that erase every ready entry by construction); ready
  entries are only ever created on paths requiring a nonempty queue, closing
  the induction.
- `reqCap` reset at TriggerCreate is model-state canonicalization only (the
  C++ field simply dies with the `DeferredCreate` use).
- BUG-5 (RPR trailing-alloc replay omission) was untouched by FIX_CAP and
  FIX_SWEEP; it is now covered by the third toggle FIX_RPR (§2.7), added to
  the bundle after the Inversion matrix showed FIX_CAP composes badly with
  unfixed BUG-5 (traces/Inversion-bug5-deadlock.txt).

**No blocking issue found. FIX_CAP and FIX_SWEEP as landed are sound
transcriptions of the agreed v1 design; the only action item is §0's
documentation alignment.**

## 2. C++ implementation blueprint (v1 = the verified pure cap + sweep)

All line numbers refer to the current tree (mem_impl.cc as read in this
campaign). No code has been changed; this is the transcription target.

### 2.1 New state
- `inst_impl.h`, `RegionInstanceImpl::DeferredCreate` (inst_impl.h:61-74):
  add `unsigned seqid_cap;`, set via a new parameter on `defer()` (called at
  mem_impl.cc:715).
- `mem_impl.h`, `LocalManagedMemory` (h:396-447): `cur_release_seqid` stays a
  plain `unsigned`; take `allocator_mutex` briefly in the deferral path to
  snapshot it (**recommended over an atomic**: preconditioned creates are not
  hot, and the mutex version needs no memory-order argument. If an atomic is
  preferred later, a relaxed load is safe because a stale/smaller read only
  shrinks the funding set — conservative direction).
- `PendingAlloc` (h:413-419): unchanged; `last_release_seqid` now stores the
  **cap** at admission instead of the admission-time watermark. Every
  downstream consumer (unblock scan cc:1654-1662, ARR replay cc:1258-1277,
  RPR inner loop cc:1579) keys off `last_release_seqid` and needs no change —
  this identification (lastSeq := cap) is the heart of the fix.
- `PendingRelease`: **no new fields in v1** (the precondition-Event storage in
  BUG-1.md's sketch belongs to the unverified union term — omit).

### 2.2 allocate_storage_deferrable + attempt_deferrable_allocation
- cc:712-717 (deferral path): snapshot `seqid_cap = cur_release_seqid` under
  the mutex; pass into `deferred_create.defer(...)`.
- `attempt_deferrable_allocation` (cc:749-807) gains `unsigned seqid_cap`;
  callers: cc:734 passes `cur_release_seqid` (request == trigger), cc:1136
  passes the stored snapshot.
- Body, in order:
  1. Keep the cc:754-757 current-allocator fast path verbatim (pending_allocs
     empty only).
  2. If `!pending_allocs.empty() && seqid_cap <
     pending_allocs.back().last_release_seqid` -> `ALLOC_INSTANT_FAILURE`
     (monotone-cap guard; back() suffices because caps are non-decreasing).
  3. Capped canonical test on a scratch allocator seeded from
     `current_allocator`: walk `pending_releases` (list order == seq order)
     and `pending_allocs` interleaved by watermark; free survivors with
     `seq <= min(next alloc's last_release_seqid, seqid_cap)` using
     missing_ok=true (cc:778 form); place each queued alloc
     (**assert placement succeeds** — §1.3); after the last queued alloc,
     free remaining survivors with `seq <= seqid_cap`; finally test the new
     instance. Failure -> `ALLOC_INSTANT_FAILURE`, and **do not write
     `future_allocator`** (§1.4).
  4. On success: push `PendingAlloc(..., last_release_seqid = seqid_cap)`;
     rebuild `future_allocator` canonically (same walk, bound = max seq,
     new alloc placed at its cap watermark); `release_allocator =
     current_allocator` when this is the first pending alloc (cc:787).
- Optimization licensed by §1.8 (optional): when `seqid_cap ==
  cur_release_seqid` and the queue is nonempty, the capped test is provably
  equal to the legacy O(1) `future_allocator.allocate()` because fut is
  maintained canonically — keep the O(1) path for the dominant
  request-triggered case. The uniform slow path is what the model verifies;
  add the fast path only with the equivalence comment.
- Shared helper suggestion: one `rebuild_future_canonical()` used by step 3/4
  subsumes the cc:768-779 replay; unification with the cc:1439-1447 and
  cc:1706-1717 rebuilds is a follow-on cleanup, not required.
- The cc:772 `assert(!it->is_ready)` site disappears with the replaced
  rebuild; carry its intent as `assert(pending_allocs.empty() => no ready
  entries)` at ADA entry — valid again because of the sweep (BUG-6 fix).

### 2.3 Asserts to carry (all TLC-backed)
- Canonical-replay queued-alloc placement succeeds (§1.3; `capAssert`).
- Sweep and rel-re-apply frees find their tags (strict; `missingFree`).
- The DEBUG cross-check cc:1674-1691 is *expected to hold* under the fix
  (INV_FutureOffsetConsistency) — keep it enabled; it now guards the
  fut-canonicity invariant the whole design leans on.

### 2.4 sweep_ready_releases() — three sites + the rel re-apply
Signature sketch: walks `pending_releases` in list order while
`pending_allocs.empty()`; for each `is_ready` entry: apply to
`current_allocator` (strict), collect its deferred_dealloc_notify instance,
erase. Notifies fire after the mutex is released (existing
`deferred_dealloc_notifies` vector pattern).
1. `release_storage_immediate`, oldest path: after the cc:1702 prefix erase
   and the cc:1751-1753 tail ARR, whenever `pending_allocs.empty()`.
2. `remove_pending_release`: after the cc:1562-1595 walk, whenever the walk
   left `pending_allocs` empty. Additionally (BUG-4-standalone), when the
   queue stays NONEMPTY: after `release_allocator = current_allocator`
   (cc:1556-1557), re-apply surviving ready entries to `release_allocator`
   (cc:1713-1714 pattern, strict).
3. `reuse_storage_immediate` mirror (~cc:1433-1448): same sweep — but the
   sweep body must be **redistrict-aware** here and at site 1 (a ready entry
   with `redistrict_tags` takes the `split_range` flavor with child-offset
   collection and EVENTUAL_SUCCESS/FAILURE notifies, cc:1372/1459 form).
   **Confidence caveat:** the v1 model contains no redistricts; treat the
   redistrict arm of the sweep as unverified until the v2 model covers
   `reuse_storage_*` — implement it, but flag it in review.
ARR full-success needs no sweep (erases every ready entry, cc:1241-1250).

### 2.5 Behavior changes to document in the commit
- Allocations that previously deferred against untriggered deletions
  requested after the create can now return `ALLOC_INSTANT_FAILURE`
  (poisoning `e_created`, cascading per existing poison semantics). This is
  deliberate: those DEFERRED answers were unsound promises that could
  deadlock (BUG-1). Includes the GC-ripple fired-but-undelivered window.
- Dealloc notifications for previously-stranded ready releases now fire at
  the sweep instead of at the next unrelated trigger — earlier, and now
  bounded (BUG-6).
- The debug assert at cc:772 (previously reachable on legal input, BUG-6) is
  restored as a true invariant in its new form.

### 2.6 The eight design decisions as normative notes
Carry §1.1-1.8 verbatim into the patch discussion: (1) cur fast path only
when queue empty; (2) missing-ok in replays; (3) placement-assert in the
canonical replay; (4) never write fut on failed admission; (5) `seq <= cap`
membership is push-order, DELAYEDDESTROY self-excludes; (6) RPR: rel dead
when queue empties, rel re-apply when it doesn't; (7) strict sweep frees
(v2 duplicates caveat); (8) append-free of a fresh release preserves
fut-canonicity — preserve the seqid-assignment sites exactly.

### 2.7 FIX_RPR — trailing alloc replay in remove_pending_release (BUG-5)
After the outer walk ends at cc:1595, run the cc:1579-1594 inner loop once
more with no seqid bound: place each remaining queued alloc onto the rebuilt
`future_allocator` in order (success stays queued, `last_release_seqid`
unchanged; failure gets `ALLOC_EVENTUAL_FAILURE` exactly like the in-walk
path, downstream poison cascades included). **Normative: the trailing pass
must CONTINUE from the walk's final `it2` cursor and must never restart from
`pending_allocs.begin()` — the SafetyFixed4 trace (slurm-77810,
bugs/DUPALLOC-TRIAGE.md) is a live demonstration that a restart-from-begin
variant re-runs `allocate()` on an already-placed tag, double-allocating it
(#442 class: second range inserted, `allocated[tag]` overwritten, old range
leaked).** Sequencing within remove_pending_release: walk → trailing replay
→ sweep / rel re-apply (§2.4 site 2), with the sweep condition evaluated on
the POST-trailing queue.

## 3. Verdict summary

| Item | Verdict |
|---|---|
| §0 spec-vs-BUG-1.md rule mismatch (pure cap modeled, union rule documented) | NEEDS-DOCUMENTATION (no spec change; C++ v1 = pure cap; union term must not ship unverified) |
| 1.1 cur fast path | SOUND |
| 1.2 missing-ok replay frees | SOUND |
| 1.3 replay placement assert | SOUND (TLC adjudicates; C++ carries assert) |
| 1.4 fut on failed admission | SOUND |
| 1.5 ready `seq <= cap` funding | SOUND (v2 redistrict re-check noted) |
| 1.6 RPR rel handling / acks | SOUND |
| 1.7 strict sweep frees | SOUND (v2 duplicate-release caveat) |
| 1.8 append ≡ canonical | SOUND |
| 1.9 ARR interaction | SOUND |

No blocker for C++ work once the local fix-validation matrix is green.
