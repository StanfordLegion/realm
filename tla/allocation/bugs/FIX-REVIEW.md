# FIX-REVIEW: adversarial review of the recommended fixes in BUG-1.md and BUG-6.md

Scope: the bug **existence** claims are TLC-confirmed and are not in question here. This review
attacks the **recommended fixes** before they are proposed to the Realm maintainers. All reasoning
is on paper against the code (no TLC runs; a concurrent fork owns the model).

| Fix | Verdict |
|---|---|
| BUG-1 (a/c): request-time seqid cap + monotone-cap admission | **NEEDS-REFINEMENT** — as stated it breaks the canonical GC-ripple pattern; repaired rule below |
| BUG-6 (A): sweep stranded ready entries when the drain empties `pending_allocs` | **NEEDS-REFINEMENT** — mechanism sound, but coverage misses a third stranding site and the sweep body must be redistrict-aware |
| BUG-6 (B): tolerate-ready + re-apply at reset | SOUND as an alternative, **also closes the composite leak** (so A-vs-B is latency/complexity, not safety); B's re-apply must also cover cc:1557 |

---

## 1. BUG-1 fix: the request-time cap instant-fails the GC-ripple pattern (CRITICAL)

**The regression.** The pure request-time cap excludes every release requested in the
request→trigger window. But the *canonical legal* use of a create precondition — Sean/Mike's own
GC-ripple example (transcript ~24:03-24:39: "I don't want Realm to try to do the instance creation
until I know that the effects of the garbage collection have rippled and **all the destroy calls
have actually been done**") — puts the funding destroy *requests* in exactly that window. That is
the *point* of `e_pre` there: the create request may arrive at the memory before the victims'
destroy requests do (multi-node ripple, network reordering), and `e_pre` orders the create's
*consideration* after they land. Under the cap as written in BUG-1.md, ADA at trigger sees the
funding releases with `seqid > seqid_cap`, the capped test fails, and the create returns
`ALLOC_INSTANT_FAILURE` — a **false out-of-memory on the primary motivating pattern for
preconditioned creates**, in workloads near capacity, which are exactly the workloads deferred
allocation exists to serve. Legion (which blocks on `InstanceAllocResult`) would report OOM to the
mapper where today it correctly reports success.

**Why no arrival-order-only rule can do better.** Construct two clients with *identical* arrival
sequences at the memory: create(I2, pre=P) → destroy(I1, pre=Q) → trigger(P). In client 1
(GC-ripple), Q is a last-use event of I1, independent of I2. In client 2 (the BUG-1 cycle), Q
depends on `e_created(I2)`. Realm sees the same inputs in the same order; the difference lives
entirely in the event graph Realm cannot inspect. Any deterministic policy keyed on arrival order
alone must give the same answer to both — so it either hangs client 2 (today's behavior) or fails
client 1 (the proposed cap). **The fix as stated chooses to fail client 1 and BUG-1.md does not
say so.** This must be surfaced to the maintainers as the central trade-off, not discovered by
them.

**Repaired rule (strongest implementable form found).** At trigger time, the capped funding set is

> releases with `seqid <= seqid_cap` **UNION** releases whose own destroy precondition **has
> already triggered clean** (checked via `has_triggered_faultaware` on the stored precondition
> event at ADA time; poisoned-triggered excluded — those will be cancelled).

Soundness of the union term: the alloc's `e_created` has not fired (the alloc is only now being
attempted), so a release whose precondition has *already triggered* cannot be waiting on it —
counting it can never close a cycle through this alloc, regardless of request order. It is also
exactly as strong as existing semantics on the poison side: today a counted release that later
fires poisoned already EVENTUAL_FAILs its dependents (cc:1591), so excluding poisoned-triggered
and counting clean-triggered adds no new "yes then no" mode.

Re-attack of the repaired rule:
- **Queue-inversion induction still holds.** With the monotone-cap rule applied to the *cap* term
  only: caps are nondecreasing in queue order, so any *untriggered* funding release R with
  `seq(R) <= cap(A_j)` for some queued alloc was requested before every queued alloc whose cap
  admits it — `cap(A_m) < seq(R)` for exactly the allocs requested before R, and those sit
  *strictly earlier* in the monotone queue. Hence every dependence edge from an untriggered
  funding release points strictly earlier in queue order. Clean-triggered releases are dependency
  *sinks* (they wait on nothing). All edges point strictly earlier ⇒ the wait graph is a DAG ⇒ no
  hang. The union term does not participate in and does not weaken the monotone comparison.
- **GC coverage, honestly stated:** the repaired rule saves the GC pattern **iff** by the time
  `e_pre` fires the victims' destroy preconditions have themselves triggered (the natural wiring:
  `e_pre` downstream of, or merged with, the collection's completion conditions). If Legion's
  `e_pre` only guarantees the destroy *requests* landed while their preconditions are still
  unfired, the pattern still instant-fails. **Open question for Mike: which wiring does Legion
  use?** If the weak wiring exists, the client-side mitigation is a retry after collection
  completes (Legion already blocks on `InstanceAllocResult`, so it has the hook), or the API
  extension below.
- **Ballistic-lite as the complete answer (recommend mentioning):** a flag on
  `release_storage_deferrable` by which the client *declares* the precondition independent of
  unfired allocator-output events ("ballistic"). Realm counts flagged window releases without any
  event-graph visibility; unflagged ones fall under the cap. This is the transcript's ballistic
  direction (~35:34) implemented as a contract declaration rather than an inference, consistent
  with Realm's existing unverified-contract style (no-back-edges is already such a contract).
  Legion's GC destroys qualify trivially. Backward compatible (default = unflagged = capped).

**Soundness of the capped gate itself (question 1): CONFIRMED.** The capped test is an
*additional* admission guard; on success the alloc is still placed in the full `future_allocator`
as today, so every downstream determinism argument (drain `assert(ok)` cc:1668-1670, future
cross-check cc:1674-1691, ARR replay) is untouched. `can_allocate` is monotone in free-set
inclusion for first-fit, so capped-yes ⇒ full-yes. Caps are static per-alloc; ARR's erasure of
ready releases only ever removes *already-triggered* releases, which are in the universally-safe
class, so no "effective cap" drift arises across swaps. No "yes then no" scenario found.

**Failure cascades — reword the report's claim.** "Monotone-cap kills all cycles of this family"
overstates: it prevents *hangs*. A monotone-rule rejection poisons `e_created(A)`, which can
poison a dependent release, which can EVENTUAL_FAIL an already-admitted alloc B via
`remove_pending_release` (a chain today's code would have *hung* on instead). Within existing
poison semantics, and strictly better than a hang, but it is a failure *cascade*, not a clean
single failure — say so.

**Implementability (question 3): workable, with two deltas beyond BUG-1.md's sketch.**
(i) `PendingRelease` must store its precondition `Event` (captured at the push sites cc:859/885/
895/1147) to support the union term's `has_triggered_faultaware` check — today only the
instance's `deferred_destroy` holds it. (ii) The capped test on the `pending_allocs`-nonempty
path (cc:798) must be an *interleaved scratch replay*: current + capped-set releases + already-
queued allocs, applied in seqid/queue order (the cc:1258-1277 shape), not a bare
release-only rebuild — otherwise older queued allocs' space consumption is unaccounted. This is
well-defined precisely *because* the monotone rule guarantees older allocs' funding sets are
subsets of the newer alloc's capped set. Cost: O(pending list) once per *deferred-create trigger*
only; triggered-precondition creates (the dominant case) are untouched.

---

## 2. BUG-6 fix A: sound mechanism, incomplete coverage

**(4) Sweep safety: CONFIRMED, with a redistrict proviso.** Tag-keyed frees commute (final
free-set is order-independent; order only matters against interleaved *allocations*, and
`pending_allocs` is empty by hypothesis). Redistrict entries are also safe to sweep **because
`split_range` places children inside the old instance's own range** (mem_impl.inl:200-262), so
their placement is independent of other swept frees — but the sweep body must then replicate the
*full* drain side-effects for a redistrict entry: collect child offsets and fire
`ALLOC_EVENTUAL_SUCCESS/FAILURE` for the children (the `it->release(current_allocator, offsets)`
form, cc:1372/1459), not just `deallocate` + ack. BUG-6.md's one-line fix description covers only
the plain-destroy shape. A stranded ready entry *can* be a redistrict (the reuse path pushes
ready-with-defNote at cc:1037-1041), so this is required, not optional.

**Coverage gap (the real finding): a third stranding site is missed.** `pending_allocs` can also
transition to empty inside `remove_pending_release` — the inner replay loop erases allocs that no
longer fit (cc:1587-1592) and can drain the queue while *ready* entries survive in the walked
list (this is exactly BUG-6 variant (b) / Phase-1 reviewer-B finding 5, reachable at 3 instances
with one poison). Fix A as located in BUG-6.md (tails of the two oldest-drain paths) does not run
there, so the poison-path stranding — and its identical downstream consequences, including the
cc:772 abort and the composite-leak precondition — survives the fix. **Refinement: make the sweep
a shared helper (`sweep_stranded_ready_releases()`), invoked at every `pending_allocs` →
empty transition:** (1) `release_storage_immediate` oldest-drain tail (~cc:1702), (2)
`reuse_storage_immediate` oldest-drain tail (~cc:1433), (3) `remove_pending_release` tail
(~cc:1596). The fourth emptying site, ARR full-success (cc:1236-1252), already erases all ready
entries by construction and needs nothing.

**(5) Notify timing: CONFIRMED SAFE.** The sweep fires `notify_deallocation` at the moment the
tag leaves `current_allocator`, which is precisely the condition the `deferred_dealloc_notify`
contract demands (mem_impl.h:427-436: delay "until this entry is drained from pending_releases" —
the sweep *is* a drain). Earlier firing than status quo only shortens the ack latency the report
already flags; the #442 double-tracking guard is the tag-out-of-current condition, which holds.

**(6) Fix B closes the composite too: CONFIRMED.** With ready entries re-applied after the
`release_allocator := current` reset, an ARR full-success builds `test` from a rel that already
excludes the stranded tag; `current := test` drops the tag exactly when the entry is erased and
its notify fired (cc:1239-1250) — no leak. BUG-6.md already states this; the framing "A removes
the root cause, B keeps stranded entries alive" is fair. One addition: **B's re-apply must also
be added after the reset at cc:1557** (`remove_pending_release`), not only cc:787 — cc:1557 is
the BUG-4 site and has the same stale-rel shape. (cc:1309 needs nothing: ARR's partial path
erases all ready entries before assigning `release := current`.) With that, B is a complete
alternative; the A-vs-B choice is ack latency + "no ready entries at quiescence" hygiene vs.
smaller diff, **not** safety.

---

## 3. Overclaims found (maintainer-falsifiable statements)

1. **BUG-1.md ("Terminal state" paragraph): "InstanceAllocResult at trigger time — which never
   comes."** False as written: the `InstanceAllocResult` *does* come at DEFERRED admission with
   `success = true` (inst_impl.cc:1140-1142) — that is what unblocks Legion's mapper wait. What
   never comes is the eventual completion (`ALLOC_EVENTUAL_SUCCESS` / the `e_created` trigger).
   Reword; as written it also understates the bug (the mapper is affirmatively told "will
   succeed" and then hung, which is worse than "never told").
2. **BUG-1.md ("Concrete execution" tail): "the Smoke trace is the identical construction at heap
   size 2."** The Smoke config is HEAP_SIZE = 3 with two size-2 instances (Smoke.cfg:13-16,
   SizesSmoke). Same shape, wrong constants.
3. **BUG-1.md "no error of any kind is reported": stands, minor softening available.** Verified:
   no autonomous watchdog exists for stuck deferred allocations; the only related machinery is
   `deadlock_catch` (runtime_impl.cc:2119-2120), a SIGTERM/SIGINT handler that produces
   diagnostics only when an operator kills the process. Optionally say "no autonomous
   diagnostics; state is only dumped on external SIGTERM/SIGINT."
4. **BUG-6.md "nothing after cc:787 re-applies the stranded ready entry": VERIFIED CORRECT** by
   site enumeration (cc:787/1309/1557 assign without re-apply; cc:1438-1447/1707-1717 re-apply
   but require both queues nonempty; cc:871/1024/1367/1468/1636/1738 touch only the incoming
   entry). Not an overclaim — noting it here so the maintainers know it was independently
   re-checked.
5. **BUG-6.md composite ("#442 class"): stands and is if anything understated** — the
   `deferred_dealloc_notify` machinery (mem_impl.h:427-436) *is* the #442 guard, and the
   composite fires the notify while the tag is live, i.e. it bypasses that guard by construction;
   the "if the recycled ID re-enters" step is the documented #442 failure mode, not speculation.

## 4. Bottom line

- **BUG-1:** keep the monotone-cap architecture, but propose it in the **repaired form**
  (cap ∪ clean-triggered releases), present the GC-ripple trade-off explicitly, ask Mike which
  `e_pre` wiring Legion uses, and offer the ballistic-lite declaration flag as the complete
  long-term answer. Reword "kills all cycles" → "prevents hangs; rejections can cascade as
  poison-mediated failures."
- **BUG-6:** adopt fix A **as a shared helper at all three `pending_allocs`→empty transitions**,
  with a redistrict-aware sweep body; keep B's strengthened assert as documentation; note B is a
  safety-equivalent fallback if the sweep is judged too invasive, provided its re-apply also
  lands at cc:1557.
