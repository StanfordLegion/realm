# Triage: INV_NoDupAlloc violation in SafetyFixed4 (sapling job 77810)

**Verdict: (a) SPEC ARTIFACT — a wiring bug in the FIX_RPR toggle's call
site, not a fix-design flaw and not a Realm bug.** The C++ blueprint is
unaffected in substance, but gains one normative sentence (below). The C++
gate re-opens after a two-line spec correction and re-verification.

Run: SafetyFixed4 (bundle CAP+SWEEP+RPR, 4 inst, H=4, sizes 2,1,1,2,
USER_POISON off, `-deadlock`), violated after 12.8B generated / 4.02B
distinct, trace depth 10 (graph depth 12), 14h10m.
Source: `slurm-77810-SafetyFixed4.out:900-1289`.

## Trace reconstruction (10 states)

| # | Action | Effect |
|---|--------|--------|
| 1 | Init | all empty |
| 2 | RequestCreate(**I2**, sz 1, ballistic pre) | CREATE_PENDING, `reqCap[2] = 0` |
| 3 | RequestCreate(**I1**, sz 2, triggered) | placed [0,2) |
| 4 | RequestCreate(**I4**, sz 2, triggered) | placed [2,4) — **heap full** |
| 5 | RequestDestroy(**I1**, untriggered) | `R1 = [inst 1, ¬ready, seq 1]` |
| 6 | RequestCreate(**I3**, sz 1, triggered) | cur full → capped path, cap = seqCtr = 1, funding {R1} → **DEFERRED**, lastSeq 1; canonical `fut = {3@[0,1), 4@[2,4)}`; `rel := cur` |
| 7 | RequestDestroy(**I2**) while CREATE_PENDING | DELAYEDDESTROY (`CREATE_PENDING_DESTROY`), preD = {eCreated(2)} (C2) |
| 8 | FireBallisticC(2) | I2's create precondition fires |
| 9 | TriggerCreate(**I2**) | FIX_CAP monotone guard: cap 0 < queue tail lastSeq 1 → **INSTANT_FAILURE**; eCreated(2) POISONED; dd-push `R2 = [inst 2, ¬ready, seq 2]` (cc:1146-1147, not applied to fut) |
| 10 | TriggerDestroy(**I2**) fires POISONED → `TriggerDestroyPoisoned(2)` | RPR rebuild — **dupAlloc = TRUE** (see below); `fut` bit-identical before/after |

State-10 internals (`remove_pending_release` model, queue = [R1, R2],
allocs = [I3 lastSeq 1]): rebuild `fut := cur = {1,4}`; walk R1 (survivor)
→ free I1 → inner loop places I3 @ [0,1) (kept); walk R2 (target) → erased.
Walk result: `L.fut = {3@0, 4@2}`, `L.paOut = <<I3>>`, `L.dup = FALSE`.

## Pinned DoAlloc site

`DeferredAlloc.tla:1018`: `tr == IF FIX_RPR THEN TrailingRPR(L.fut, L.paOut)`.

`RPRLoop` (line 533) returns `paOut = inner.kept \o r.paOut` — the **full
surviving queue**, i.e. walk-KEPT allocs concatenated with the true trailing
(never-examined) remainder (base case, line 522: `paOut |-> paA`). The
FIX_RPR call site feeds that whole list to `TrailingRPR` (lines 550-562),
which re-runs `CanAlloc`/`DoAlloc` on the kept prefix. In this trace the
kept I3 is already in `L.fut`, `CanAlloc` succeeds on the residual gap
[1,2), and `DoAlloc(fut, 3, 1)` sees `HasTag(fut, 3) = TRUE` →
`dup = TRUE`. The left-biased `@@` in `DoAlloc` keeps the old placement, so
`fut` is unchanged — exactly the observed state delta (only `dupAlloc`,
`pendingReleases`, `destroyWaiter` change).

Reconstruction re-verified twice against the trace and the operator text.

## Why (a) and not (b)/(c)

- **Not a fix-design flaw (b):** the agreed design and the C++ blueprint
  specify "after the outer walk ends at cc:1595, run the cc:1579-1594 inner
  loop once more with no seqid bound." In C++ the alloc cursor `it2` has
  already advanced **past** every kept alloc during the walk; the trailing
  pass continues from that cursor and can only see the never-examined
  remainder. The re-examination of kept allocs exists only in the spec's
  functional reconstruction, which lost the cursor position by reusing
  `paOut` (full survivors) instead of the trailing remainder.
- **Not a residual code bug (c):** current C++ has no trailing pass at all
  (that omission IS BUG-5), and in the base model `paOut` is only ever used
  to rebuild `pendingAllocs'` — it is never re-fed to `DoAlloc`. Consistent
  with every toggles-off run being dup-clean.

Model-only severity note: the witnessed flavor is benign (ghost flag only).
A second, worse flavor is reachable in-model: if `CanAlloc` FAILS for a
kept alloc (its own placement consumed the last fitting gap), `TrailingRPR`
spuriously EVENTUAL_FAILs an alloc the walk had successfully re-placed,
erasing it from the queue while its placement stays in `fut` — bogus poison
cascades and a stale fut tag. Both flavors vanish with the correction.

## Proposed spec correction (two edits)

1. `RPRLoop`: expose the trailing remainder separately — base case gains
   `trail |-> paA`, recursive case `trail |-> r.trail` (all other fields
   unchanged; `paOut` keeps its full-survivors meaning for the
   FIX_RPR = FALSE path).
2. Call site (`TriggerDestroyPoisoned`):
   `tr == IF FIX_RPR THEN TrailingRPR(L.fut, L.trail) ELSE ...` and
   `paF == IF FIX_RPR THEN KeptPrefix \o tr.kept ELSE L.paOut`, with
   `KeptPrefix == SubSeq(L.paOut, 1, Len(L.paOut) - Len(L.trail))`
   (exact, since `paOut = kept \o trail` by construction).

## Normative addition to the C++ blueprint (BLUEPRINT-REVIEW.md)

The trailing pass must **continue from the walk's final `it2` cursor** —
never restart from `pending_allocs.begin()`. This trace is the live
demonstration of the mis-scoped variant: in C++ a restart-from-begin would
re-run `allocate()` on an already-placed tag, inserting a second range and
overwriting `allocated[tag]` — a real #442-class leak/corruption, not a
ghost flag.

## Why the local matrix missed it

The shape needs **4 instances** (two triggered fillers to fill the heap, a
third create that defers with a kept-through-RPR placement, and a fourth
whose DELAYEDDESTROY entry poisons — SafetyMiniFixed's 3-instance space
provably cannot build it), AND `INV_NoDupAlloc` is absent from every local
bundle battery (only Safety/SafetyMini/Poison4/Big and the sapling Fixed
configs check it — notably NOT Inversion, SafetyMiniFixed, SmokeFixed,
EventLoopFixed, GCRipple, LivenessFixed). Recommendation for re-validation:
add `INV_NoDupAlloc` to all bundle configs so the corrected `TrailingRPR`
is actually observed locally.

## Sanity confirmation of the other two sapling violations (expected)

- **Safety (77808)**, 9-state trace: final state `pendingAllocs = <<>>`,
  `pendingReleases = [inst 4 ¬ready seq 2, inst 2 READY+defNote seq 3]`, no
  poison anywhere — the known **BUG-6 variant (a)** stranded-ready shape at
  4-instance scale. As expected.
- **Poison4 (77809)**, 9-state trace: final state `pendingAllocs = <<>>`,
  `pendingReleases = [inst 2 READY+defNote seq 2]`, `failedVia[4] = "RPR"`,
  `eCreated[4] = POISONED` — the queue was emptied by an in-walk RPR
  failure with a ready survivor: the known **BUG-6 variant (b)** poison
  stranding, i.e. precisely the third sweep site's shape. As expected.
- Both runs halted on these first (expected) violations at 1.44B / 273M
  distinct states, so the deep hunts (BUG-3; BUG-4-standalone/BUG-5-unfixed)
  were preempted — the already-created SafetyHunt/PoisonHunt configs (out of
  this triage's scope) are the vehicle for those re-runs.

## Implications for the C++ gate

The fix design itself took no damage: nothing in this violation touches the
capped-admission, sweep, or trailing-replay semantics as intended for C++.
Gate sequence: apply the two-line spec correction → SANY + local bundle
matrix re-run with `INV_NoDupAlloc` added to all bundle batteries (regression
toggles-off must stay exact) → resubmit SafetyFixed4 fresh on sapling (spec
changed; the checkpoint is not reusable). Big/BigFixed submissions are
unaffected by the correction only if resubmitted after it lands (BigFixed
checks `INV_NoDupAlloc` and runs FIX_RPR — a stale-spec run could hit the
same artifact).
