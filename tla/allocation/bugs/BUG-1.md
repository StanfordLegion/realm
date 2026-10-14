# BUG-1: Deferred creates are ordered at precondition-trigger time, not request time — event-loop deadlock

**Status:** machine-confirmed by TLC (EventLoop config: deadlock in 7 states; Smoke config independently, 4,029 states).
**Traces:** `EventLoop.trace.txt` (primary), `traces/Smoke-run1.txt` (same shape at Smoke's constants).
**Fix review:** recommended fix revised per adversarial review — see `FIX-REVIEW.md` §1.
**Class:** liveness — silent permanent hang. No poison, no OOM abort, no error of any kind is reported.

## Summary

A deferred instance creation is inserted into the memory's release/alloc total order when its
**precondition triggers**, not when it was **requested**. `attempt_deferrable_allocation` associates
the new pending alloc with every pending release currently in the list (`last_release_seqid :=
cur_release_seqid`, mem_impl.cc:784/801), and tests fit against a future heap that applies all of
them (mem_impl.cc:768-781). But Realm handed out the instance's ready event `e_created` at *request*
time, so a release *requested after the create but before its trigger* may legally carry a
precondition that depends on `e_created` of this very instance. Realm then plans the allocation out
of a release that can only happen after the allocation completes: a cycle through the event graph
that Realm can never resolve. This is the bug Sean suspected in the deferred-allocation talk
("an instance that is created with a precondition doesn't actually necessarily get put in the right
spot in the overall ordering ... can cause an event loop", transcript ~24:47-25:35). The
conservatism rule he describes — "we don't ever want to associate an allocation with a release that
came chronologically after it" (~34:20) — is enforced against the wrong clock: trigger order instead
of request order.

## Concrete execution (from the TLC counterexample, EventLoop config: heap = 3 units, two instances of size 3)

Client call order (all from one control-plane context, in program order):

| # | Client call | Realm path | Heap state after |
|---|------------|-----------|------------------|
| 1 | create **I1** (size 3, no precondition) | `allocate_storage_deferrable` (cc:693) -> precondition already triggered (cc:703) -> `attempt_deferrable_allocation` (cc:734): `pending_allocs` empty, `current_allocator.allocate` succeeds at offset 0 (cc:754-757) -> **INSTANT_SUCCESS**; `e_created(I1)` fires clean (inst_impl.cc:1201-1202) | current = {I1@[0,3)} |
| 2 | create **I2** (size 3, precondition **P** = user event, untriggered) | cc:712-717: `inst_offset := INSTOFFSET_DELAYEDALLOC`, `deferred_create.defer`, return `ALLOC_DEFERRED`. **Note: the ready event `e_created(I2)` was already handed to the client by this call** (transcript ~15:56: "we immediately hand back an instance ID and an event"). Nothing is recorded in the allocator's ordering. | unchanged |
| 3 | destroy **I1** (precondition = merge(`e_created(I1)`, `e_created(I2)`)) | `release_storage_deferrable` (cc:810): precondition untriggered -> `pending_allocs` empty -> push `PendingRelease(I1, !ready, seqid=1)` (cc:857-859); `deferred_destroy.defer` (cc:920) | pending_releases = [R1(I1, seq 1)] |
| 4 | destroy **I2** (precondition = `e_created(I2)`) | cc:845-849: I2 is `INSTOFFSET_DELAYEDALLOC` -> marked `INSTOFFSET_DELAYEDDESTROY`, deferred. (Incidental to the cycle — this is just the client's eventual cleanup of I2.) | unchanged |
| 5 | client triggers **P** (ballistic: depends on nothing) | `DeferredCreate::event_triggered` (inst_impl.cc:49-56) -> `allocate_storage_immediate(I2)` (cc:1088): sees DELAYEDDESTROY (cc:1106), not poisoned -> `attempt_deferrable_allocation` (cc:1136): `current` full (cc:755 fails), releases exist (cc:762) -> **future := current − R1 = empty heap** (cc:768-779) -> `future.allocate(I2)` succeeds (cc:781) -> `pending_allocs = [PendingAlloc(I2, lastSeq = cur_release_seqid = 1)]` (cc:783-784), `release_allocator := current` (cc:787) -> **ALLOC_DEFERRED**. Then the queued destroy is pushed: `PendingRelease(I2, !ready, seqid=2)` (cc:1146-1147) and applied to future (cc:1150-1153). `e_created(I2)` stays unfired (inst_impl.cc:1126-1157). | pending_allocs = [A(I2, lastSeq=1)]; pending_releases = [R1(I1,1), R2(I2,2)] |

**Terminal state (the deadlock):**

- A(I2) is granted only when R1 drains (`release_storage_immediate` oldest-drain, cc:1631-1702, or a reorder — R1 is the only release that frees enough space).
- R1 drains only when merge(`e_created(I1)`, `e_created(I2)`) triggers. `e_created(I1)` is fired; **`e_created(I2)` is not**.
- `e_created(I2)` fires only when A(I2) is granted (`notify_allocation(ALLOC_EVENTUAL_SUCCESS)`, inst_impl.cc:1201-1202).

Cycle: A(I2) -> R1 -> `e_created(I2)` -> A(I2). Every waiter is parked; nothing times out. Worse
than a mere hang with no answer: at DEFERRED admission the `InstanceAllocResult` profiling response
**is delivered with `success = true`** (inst_impl.cc:1140-1142) — a Legion mapper blocked on it is
affirmatively told the allocation will succeed, unblocks, builds downstream work on that promise,
and *then* the system hangs. The eventual completion (`ALLOC_EVENTUAL_SUCCESS` / the `e_created`
trigger) is what never comes. The Smoke trace is the identical construction at Smoke's constants
(HEAP_SIZE = 3, two instances of size 2).

## Why the client is legal

- **Topological sort / no back edges:** the calls occur in the order create(I1), create(I2),
  destroy(I1), destroy(I2), trigger(P). Every event dependency points at an event handed out by an
  *earlier* call (`e_created(I1)`, `e_created(I2)`); P is triggered unconditionally, depending on no
  Realm result. No user event is ever triggered based on a deferred-allocation outcome.
- **Destroy-after-create:** both destroy preconditions include the respective instance's own
  `e_created`, exactly the discipline Legion enforces today.
- **The idiom is canonical:** "free the old instance only after the new one exists (and the copy
  from old to new has run)" — destroy(I1) gated on `e_created(I2)` is the standard copy/migration
  chain. Combine it with a create precondition (Sean/Mike's own example: creates gated on garbage-
  collection ripple, transcript ~24:03-24:39) and this is exactly the shape above.

The talk's protection (~30:40-34:45) is that each pending alloc remembers "the newest pending
release when it *showed up*" so that an alloc is never satisfied by a release that might depend on
its post-condition. But "showed up" is implemented as ADA time = trigger time. Between request and
trigger, Realm has already published `e_created`, and any release requested in that window — R1 here
— may depend on it. ADA then happily counts R1's space. The rule "never associate an allocation with
a release that came chronologically after it" is sound only if "chronologically" means request
order; the code implements trigger order.

## Impact

- Silent, permanent, distributed hang: the memory's pending lists never drain, every event
  downstream of the instance never triggers, and no error path fires (not the poisoned case, not an
  OOM `abort()`, not `ALLOC_INSTANT_FAILURE`). There are no autonomous diagnostics for stuck
  deferred allocations: the only related machinery is `deadlock_catch` (runtime_impl.cc:2119),
  which dumps state only when an operator externally sends SIGTERM/SIGINT.
- The severity is compounded by the affirmative promise: the mapper receives
  `InstanceAllocResult{success=true}` at DEFERRED admission (inst_impl.cc:1140-1142) before the
  hang, so the client has already committed downstream work to an allocation that will never
  materialize.
- Exposure requires a create with an untriggered precondition. Legion's current mapper path blocks
  on `InstanceAllocResult` before exposing the instance, which narrows the single-context window,
  but (a) Legion does issue preconditioned creates for GC ripple today, (b) destroy requests can
  arrive from other contexts/nodes during the window, and (c) Realm's contract does not require
  clients to serialize this way. Severity: **high** (hard hang, legal client, no diagnostics), with
  moderate likelihood today and growing likelihood as preconditioned creates get more use.

## The central trade-off (read this before the fix)

Any fix keyed on **arrival order alone** faces an impossibility: construct two clients with
*identical* arrival sequences at the memory — create(I2, pre=P), destroy(I1, pre=Q), trigger(P). In
the **GC-ripple client**, Q is a last-use event of I1, independent of I2: the whole point of P
(transcript ~24:03-24:39) is that the funding destroy *requests* land in the create's
request->trigger window, and the create must then use them. In the **cycle client**, Q depends on
`e_created(I2)`. Realm sees the same inputs in the same order; the difference lives entirely in the
event graph Realm cannot inspect. Any deterministic arrival-order-only policy must answer both the
same way — so it either hangs the cycle client (today's behavior) or **false-OOMs the canonical
GC-ripple pattern, the primary motivating use of preconditioned creates**, in exactly the
near-capacity workloads deferred allocation exists to serve. A pure request-time cap (our first
draft) makes the second choice silently. The repaired rule below uses one extra bit of information
that *is* visible to Realm — whether a release's precondition has already triggered — to save the
GC pattern in its natural wiring; whether that covers Legion's actual wiring is an open question
for Mike (below).

## Candidate fixes (no code changed yet)

**(a) = (c), REPAIRED FORM — recommended: request-time seqid cap, with clean-triggered releases
always fundable.** Record `cur_release_seqid` at request time in the deferral path (cc:712-717,
e.g. on `DeferredCreate` or the instance) as `seqid_cap`. At trigger time, ADA tests the fit
against a **capped funding set**:

> releases with `seqid <= seqid_cap` **UNION** releases whose own destroy precondition has
> **already triggered clean** by ADA time (checked via `has_triggered_faultaware` on the stored
> precondition event; poisoned-triggered excluded — those get cancelled/removed).

replayed in list order on a scratch allocator, instead of testing the full `future_allocator`
(cc:768-779). Admission also requires the **monotone-cap rule** on the cap term: admit only if
`seqid_cap >=` the cap of every alloc already in `pending_allocs` (caps monotone => compare the
newest entry only). If either check fails -> `ALLOC_INSTANT_FAILURE`. On success the alloc is
placed in the full `future_allocator` exactly as today, so the drain determinism machinery
(cc:1668-1670, cc:1674-1691, ARR replay) is untouched; first-fit `can_allocate` is monotone in
free-set inclusion, so capped-yes implies full-yes — "never say yes then no" is preserved.

*Soundness of the union term:* the alloc's `e_created` has not fired (it is only now being
attempted), so a release whose precondition has already triggered clean cannot be waiting on it —
counting it can never close a cycle through this alloc, regardless of request order. Excluding
poisoned-triggered adds no new failure mode: a counted release that fires poisoned already
EVENTUAL_FAILs its dependents today (cc:1587-1592).

*Why no hang survives (induction sketch, repaired form):* clean-triggered funding releases are
dependency **sinks** — they wait on nothing. Every *untriggered* funding release R used by a queued
alloc has `seq(R) <= cap` of that alloc, i.e. R was requested before that alloc's request; under
the monotone-cap rule the caps are nondecreasing in queue order, so any alloc whose cap does *not*
admit R (requested before R) sits strictly earlier in the queue than every alloc whose cap does.
Hence every wait edge — alloc-to-untriggered-release, release-to-`e_created`, alloc-to-queue-
predecessor — points strictly earlier in (queue order, request order), and the wait graph is a
DAG. **This prevents hangs; it does not make every outcome clean:** a capped rejection poisons
`e_created(A)`, which can poison a dependent release, which can EVENTUAL_FAIL an already-admitted
alloc via `remove_pending_release` — a poison-mediated failure *cascade*, within existing poison
semantics, and strictly better than the hang today's code produces on the same input — but a
cascade nonetheless. **Caveat (validation matrix): the cascade drains cleanly only with the BUG-5
fix also in place.** If a dependent alloc is *trailing* in `remove_pending_release`'s replay (its
`last_release_seqid` exceeds every surviving walked seqid), the pre-existing BUG-5 hole strands it
forever instead of failing it — the Inversion client deadlocked with FIX_CAP on and FIX_RPR off
(witness: `traces/Inversion-bug5-deadlock.txt`). FIX_RPR = `remove_pending_release` processes
trailing allocs after its walk (place-or-EVENTUAL_FAIL); see `bugs/BUG-5.md` for the full analysis.

*GC-ripple coverage, honestly stated:* the repaired rule saves the GC pattern **iff** by the time
`e_pre` fires, the victims' destroy preconditions have themselves already triggered (the natural
wiring: `e_pre` downstream of, or merged with, the collection's completion conditions). If Legion's
`e_pre` only guarantees the destroy *requests* have landed while their preconditions are still
unfired, the pattern still instant-fails under this rule.

> **OPEN QUESTION for Mike:** which wiring does Legion use for the GC-ripple `e_pre` — does it fire
> only after the victims' destroy preconditions have triggered (rule saves the pattern), or merely
> after the destroy *requests* have been issued (rule false-fails it)?

*Fallbacks if the wiring is unfavorable:* (i) client-side retry — Legion already blocks on
`InstanceAllocResult`, so on a capped `ALLOC_INSTANT_FAILURE` it can re-issue the create after
collection completes; (ii) **ballistic-lite**: a flag on `release_storage_deferrable` by which the
client *declares* the destroy precondition independent of unfired allocator-output events. Flagged
window releases are always fundable; unflagged ones fall under the cap. This is the transcript's
ballistic direction (~35:34) as a contract declaration rather than an inference — consistent with
Realm's existing unverified-contract style (no-back-edges is already such a contract), backward
compatible (default = unflagged = capped), and Legion's GC destroys qualify trivially.

*Implementation sketch:* (i) store `seqid_cap` at cc:712-717; (ii) **`PendingRelease` stores its
precondition `Event`** (captured at the push sites cc:859/885/895/1147) so ADA can evaluate the
union term — today only the instance's `deferred_destroy` holds it; (iii) ADA gains the capped
admission: on the `pending_allocs`-empty path a scratch rebuild mirroring cc:768-779 restricted to
the funding set; on the `pending_allocs`-**nonempty** path (cc:798) the test must be an
**interleaved scratch replay** — current + funding-set releases + already-queued allocs' placements
applied in seqid/queue order (the cc:1258-1277 shape) — not a bare release-only rebuild, otherwise
older queued allocs' space consumption is unaccounted; this is well-defined precisely because the
monotone rule makes older allocs' funding sets subsets of the newer alloc's. Cost: O(pending list)
once per *deferred-create trigger* only; triggered-precondition creates (the dominant case) are
untouched. Edge cases: a DELAYEDDESTROY self-release gets `seqid = cap + k > cap` and an unfired
precondition by construction, so an instance still never funds itself; ARR's erasures only remove
already-triggered releases, which are in the universally-fundable class, so no cap drift across
swaps.

**(b) Full ballistic-event tracking** (talk, ~35:34): only count releases whose preconditions are
known "scheduled" by inspecting event provenance. Strictly stronger — but requires event-graph
visibility Realm does not have; ballistic-lite above captures its value as a contract bit. Long-term
direction, not a near-term fix.

**(d) Cycle detection over the event graph at destroy-request time.** Rejected for the reasons the
talk itself gives (~33:38): Realm cannot see whether one event derives from another, and walking
the distributed graph is impractical.

**Recommendation: (a)/(c) in the repaired form (cap ∪ clean-triggered), with the monotone-cap
admission rule, plus the ballistic-lite flag if Legion's GC wiring turns out unfavorable.**

> **v1 scope as verified (see BLUEPRINT-REVIEW.md §0/§3).** The TLA+ fix that was actually landed
> and validated (`FIX_CAP`) implements the **pure request-time cap**: funding = surviving releases
> with `seqid <= seqid_cap` only. The "∪ clean-triggered releases" union term above is **not
> modeled and therefore not verified**. Per Mike's decision, spurious instant-failures are
> acceptable (correctness over compatibility), so:
> - **C++ v1 must implement the pure cap exactly as modeled** — no union term, and no
>   precondition-`Event` storage on `PendingRelease` (a cost v1 thereby avoids).
> - The union term is demoted to an optional later optimization. **Do not implement it without
>   first extending the spec's funding gate and re-running the full validation matrix.**
> - Practical consequence (accepted behavior change): the GC-ripple pattern succeeds when the
>   funding destroys have been *applied* before the create's trigger, and honestly
>   `ALLOC_INSTANT_FAILURE`s otherwise — instead of risking a hang.
> - **Shipping constraint: FIX_CAP must not land in C++ without FIX_RPR** (the BUG-5 fix) — the
>   capped rejection cascade only drains cleanly with FIX_RPR in place; without it a trailing
>   dependent alloc in `remove_pending_release` is stranded forever
>   (witness: `traces/Inversion-bug5-deadlock.txt`; full analysis in `bugs/BUG-5.md`).

## Verification plan (model v-next)

1. Add `seqid_cap` to the spec's deferred-create state; change ADA to the **repaired** rule: capped
   funding set = {seq <= cap} ∪ {releases whose precondition has fired clean}, scratch-replay
   admission (interleaved form on the nonempty path), monotone-cap check. Keep everything else
   identical.
2. Expected flips: EventLoop — deadlock disappears; I2's create resolves `ALLOC_INSTANT_FAILURE`,
   both destroys fire poisoned, run drains to Quiescent (green). Smoke — returns to a clean pass.
3. New inversion client (the A/R*/B shape): expected to **instant-fail cleanly** (possibly as a
   poison-mediated cascade) rather than hang; also run it against the naive pure cap to document
   that the naive form is insufficient.
4. New GC-ripple client: model `e_pre` firing only after the victims' destroy preconditions have
   resolved clean; expected to still **SUCCEED** under the repaired rule (and to false-fail under
   the pure cap — documenting the regression the review caught).
5. Expected non-flips: SafetyMini's BUG-6 violation persists (independent mechanism); all currently
   green invariants stay green (the funding set only restricts admissions; the union term only
   adds releases that are sinks).
