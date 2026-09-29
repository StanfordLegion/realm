--------------------------- MODULE DeferredAlloc ---------------------------
(***************************************************************************)
(* Protocol model of Realm's LocalManagedMemory deferred instance          *)
(* allocation/deletion logic.  Implements DESIGN.md (tla/allocation)       *)
(* sections 2-4 and the ghost/invariant machinery of section 6.            *)
(*                                                                         *)
(* Code citations:                                                         *)
(*   cc  = src/realm/mem_impl.cc                                           *)
(*   h   = src/realm/mem_impl.h                                            *)
(*   inl = src/realm/mem_impl.inl                                          *)
(*   ii  = src/realm/inst_impl.cc                                          *)
(*                                                                         *)
(* Each action corresponds to exactly one allocator_mutex-holding region   *)
(* (h:398, DESIGN.md section 1).  Client/environment wiring (preconditions,*)
(* contract C1-C3, fairness, Quiescent/Done) lives in MCDeferredAlloc.     *)
(*                                                                         *)
(* Actions take (trig, pois) parameters describing the state of the        *)
(* operation's precondition at the moment of the call; the MC module       *)
(* supplies them from its event layer.                                     *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
  HEAP_SIZE,   \* heap is 0..HEAP_SIZE-1 (DESIGN 1)
  INSTANCES,   \* finite set of instance ids (never reused, DESIGN 1)
  Size,        \* [INSTANCES -> 1..HEAP_SIZE] model-assigned sizes
  FIX_CAP,     \* BUG-1 fix variant: request-time seqid cap on admissions
               \* (bugs/BUG-1.md as amended; FALSE = current Realm behavior)
  FIX_SWEEP,   \* BUG-6 fix A + BUG-4-standalone re-apply (bugs/BUG-6.md;
               \* FALSE = current Realm behavior)
  FIX_RPR      \* BUG-5 fix: trailing alloc replay in remove_pending_release
               \* (bugs/BUG-5.md; FALSE = current Realm behavior).  Required
               \* in the FIX_CAP bundle: the cap raises poisoned-release
               \* frequency, and a trailing alloc stranded by the cc:1595
               \* walk end deadlocks in every trigger order
               \* (traces/Inversion-bug5-deadlock.txt)

ASSUME /\ HEAP_SIZE \in Nat \ {0}
       /\ Size \in [INSTANCES -> 1..HEAP_SIZE]
       /\ FIX_CAP \in BOOLEAN
       /\ FIX_SWEEP \in BOOLEAN
       /\ FIX_RPR \in BOOLEAN

\* instOffset sentinels (INSTOFFSET_* stand-ins, inst_impl.h:167-172)
OFF_NONE   == -1
OFF_FAILED == -2

\* AllocationResult values used internally (h:88-96)
\*  "IS" = ALLOC_INSTANT_SUCCESS, "IF" = ALLOC_INSTANT_FAILURE,
\*  "DEF" = ALLOC_DEFERRED, "CANC" = ALLOC_CANCELLED
Statuses == {"UNREQUESTED", "CREATE_PENDING", "CREATE_PENDING_DESTROY",
             "ALLOC_DEFERRED", "ALLOCATED", "FAILED", "DESTROYED"}

VARIABLES
  \* --- protocol state (DESIGN 3) ---
  cur,              \* current_allocator  (h:411)
  fut,              \* future_allocator   (h:411) - kept stale exactly as code does
  rel,              \* release_allocator  (h:411) - kept stale exactly as code does
  pendingAllocs,    \* Seq of [inst, size, lastSeq]          (h:413-419, 444)
  pendingReleases,  \* Seq of [inst, isReady, seq, defNote]  (h:420-443, 445)
  seqCtr,           \* cur_release_seqid (h:412); pushes use pre-increment
  reqCap,           \* [INSTANCES -> Nat] FIX_CAP only: cur_release_seqid
                    \* snapshot at create-REQUEST time (C++: new
                    \* DeferredCreate::seqid_cap field, set in the
                    \* allocate_storage_deferrable deferral path cc:712-717);
                    \* all-0 when FIX_CAP = FALSE
  instState,        \* [INSTANCES -> Statuses]
  instOffset,       \* [INSTANCES -> -2..HEAP_SIZE]
  eCreated,         \* [INSTANCES -> {"UNFIRED","CLEAN","POISONED"}] (ii:1121-1122, 1201-1202)
  \* --- ghost variables (DESIGN 6) ---
  allocatedEver,    \* insts whose tag ever entered cur
  curFreed,         \* insts whose tag left cur (explicit deallocate or realized swap-erasure)
  missingFree,      \* any missing_ok=FALSE free found no tag (inl:614 assert)
  structuralAssertFailed, \* cc:846, 1630, 1662, 1682, 1720-1723, 1548-1551
  unblockFailed,    \* cc:1670 assert(ok)
  futMismatch,      \* cc:1674-1691 DEBUG cross-check
  poisonReplayBad,  \* cc:1587 assert(found)
  readyAtRebuild,   \* cc:772 assert(!it->is_ready) at the ADA rebuild site
  dupAlloc,         \* a materialized DoAlloc hit a live tag (inl:500 leak class, #442)
  wasDeferred,      \* insts that ever got ALLOC_DEFERRED
  failedVia,        \* [INSTANCES -> {"NONE","INSTANT","CANCELLED","RPR"}]
  notifyCount       \* notify_deallocation count per inst (PROP_NotifyOnce)

protoVars == << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                reqCap, instState, instOffset, eCreated, allocatedEver, curFreed,
                missingFree, structuralAssertFailed, unblockFailed,
                futMismatch, poisonReplayBad, readyAtRebuild, dupAlloc,
                wasDeferred, failedVia, notifyCount >>

ghostVars == << allocatedEver, curFreed, missingFree, structuralAssertFailed,
                unblockFailed, futMismatch, poisonReplayBad, readyAtRebuild,
                dupAlloc, wasDeferred, failedVia, notifyCount >>

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 2: the deterministic first-fit allocator.                       *)
(* An allocator state is a function  tag -> [first, size]  over a subset   *)
(* of INSTANCES; free space is derived.  Faithful to BasicRangeAllocator's *)
(* address-ordered free list (inl:528-551, 424-435): first fit = smallest  *)
(* offset o such that [o, o+sz) is entirely free (DESIGN 2).               *)
(***************************************************************************)

EmptyAlloc == [t \in {} |-> [first |-> 0, size |-> 0]]

HasTag(a, t) == t \in DOMAIN a

IsFree(a, x) == \A t \in DOMAIN a :
                   ~(a[t].first <= x /\ x < a[t].first + a[t].size)

IsFreeRange(a, o, sz) == \A x \in o..(o+sz-1) : IsFree(a, x)

CanAlloc(a, sz) == \E o \in 0..(HEAP_SIZE - sz) : IsFreeRange(a, o, sz)

\* lowest-address placement (inl:424-435); minimal o is always a gap start
FirstFitOff(a, sz) ==
  CHOOSE o \in 0..(HEAP_SIZE - sz) :
    /\ IsFreeRange(a, o, sz)
    /\ \A o2 \in 0..(HEAP_SIZE - sz) : IsFreeRange(a, o2, sz) => o <= o2

\* REPRESENTATION LIMIT: the C++ allocate() on an already-allocated tag does
\* allocated[tag] = idx (inl:500) - the NEW range wins and the old range is
\* LEAKED while still linked (the #442 double-tracking class).  The partial-
\* function representation cannot express that leak (@@ is left-biased, so
\* the OLD placement would win here).  Call sites therefore flag any
\* materialized DoAlloc onto a live tag via the dupAlloc ghost
\* (INV_NoDupAlloc) instead of modeling the leak.
DoAlloc(a, t, sz) == a @@ (t :> [first |-> FirstFitOff(a, sz), size |-> sz])

DoFree(a, t) == [u \in (DOMAIN a) \ {t} |-> a[u]]

\* missing_ok=TRUE call sites (inl:610-614 tolerated-miss form)
FreeMissingOk(a, t) == IF HasTag(a, t) THEN DoFree(a, t) ELSE a

-----------------------------------------------------------------------------
(* small helpers *)

Min(S) == CHOOSE x \in S : \A y \in S : x <= y

ToSet(s) == {s[k] : k \in DOMAIN s}

RemoveAt(s, m) == SubSeq(s, 1, m-1) \o SubSeq(s, m+1, Len(s))

\* first pending_releases index for inst j, 0 if none (begin()-first search)
FirstRelIdx(pr, j) ==
  LET ks == {k \in 1..Len(pr) : pr[k].inst = j}
  IN IF ks = {} THEN 0 ELSE Min(ks)

\* indices of ready entries, and the notified/erased sets ARR produces
ReadySet(pr)        == {pr[k].inst : k \in {kk \in 1..Len(pr) : pr[kk].isReady}}
ReadyDefNoteSet(pr) == {pr[k].inst : k \in {kk \in 1..Len(pr) :
                                             pr[kk].isReady /\ pr[kk].defNote}}
NonReadyOnly(pr)    == SelectSeq(pr, LAMBDA e : ~e.isReady)

\* apply all pending releases with missing_ok=TRUE, in order (cc:769-779 replay)
RECURSIVE ApplyFreesMissingOk(_, _)
ApplyFreesMissingOk(a, s) ==
  IF s = <<>> THEN a
  ELSE ApplyFreesMissingOk(FreeMissingOk(a, Head(s).inst), Tail(s))

\* apply ready releases with missing_ok=FALSE (cc:1706-1717 rebuild);
\* returns [a, miss] where miss records any inl:614 assert(missing_ok) firing
RECURSIVE ApplyReadyFreesStrict(_, _)
ApplyReadyFreesStrict(a, s) ==
  IF s = <<>> THEN [a |-> a, miss |-> FALSE]
  ELSE LET e == Head(s)
       IN IF e.isReady
          THEN LET m == ~HasTag(a, e.inst)
                   r == ApplyReadyFreesStrict(FreeMissingOk(a, e.inst), Tail(s))
               IN [a |-> r.a, miss |-> m \/ r.miss]
          ELSE ApplyReadyFreesStrict(a, Tail(s))

\* FIX_SWEEP (BUG-6 fix A, bugs/BUG-6.md): when pending_allocs is empty,
\* ready entries must not strand in pending_releases - walk in list order,
\* apply each is_ready entry to cur, fire its deferred dealloc-notify, and
\* erase it.  C++: new LocalManagedMemory::sweep_ready_releases() helper
\* called (mutex held) from release_storage_immediate after the cc:1702
\* prefix erase, and from remove_pending_release after the cc:1562-1595
\* walk, whenever pending_allocs is empty.  attempt_release_reordering's
\* full-success path (cc:1236-1252) already erases every ready entry, so it
\* needs no sweep.  Running the sweep whenever pending_allocs is empty (not
\* only on the transition) is idempotent and establishes the invariant
\* inductively.  Returns [cur, pr, notified, freed, miss].
RECURSIVE Sweep(_, _)
Sweep(curA, s) ==
  IF s = <<>>
  THEN [cur |-> curA, pr |-> <<>>, notified |-> {}, freed |-> {},
        miss |-> FALSE]
  ELSE LET e == Head(s)
       IN IF e.isReady
          THEN LET m == ~HasTag(curA, e.inst)   \* C++ free is missing_ok=FALSE
                   r == Sweep(FreeMissingOk(curA, e.inst), Tail(s))
               IN [cur |-> r.cur, pr |-> r.pr,
                   notified |-> (IF e.defNote THEN {e.inst} ELSE {})
                                \cup r.notified,
                   freed |-> {e.inst} \cup r.freed,
                   miss |-> m \/ r.miss]
          ELSE LET r == Sweep(curA, Tail(s))
               IN [cur |-> r.cur, pr |-> <<e>> \o r.pr,
                   notified |-> r.notified, freed |-> r.freed,
                   miss |-> r.miss]

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.1: ADA - attempt_deferrable_allocation (cc:749-807).          *)
(* Pure helper; inlined into RequestCreate / TriggerCreate.                *)
(* Returns [res, cur, fut, rel, pa, ready772].                             *)
(***************************************************************************)
ADARes(i, sz, curA, futA, relA, paA, prA, seqNow) ==
  IF paA = <<>> THEN                                        \* cc:754
    IF CanAlloc(curA, sz) THEN                              \* cc:755-757
      [res |-> "IS", cur |-> DoAlloc(curA, i, sz), fut |-> futA,
       rel |-> relA, pa |-> paA, ready772 |-> FALSE,
       dup |-> HasTag(curA, i)]
    ELSE IF prA = <<>> THEN                                 \* cc:762-765
      \* (Config::deferred_instance_allocation is TRUE in v1, DESIGN 1)
      [res |-> "IF", cur |-> curA, fut |-> futA, rel |-> relA,
       pa |-> paA, ready772 |-> FALSE, dup |-> FALSE]
    ELSE
      \* rebuild future from scratch (cc:768-779); cc:772 assert(!is_ready)
      LET r772 == \E k \in 1..Len(prA) : prA[k].isReady
          f    == ApplyFreesMissingOk(curA, prA)            \* cc:778 missing_ok=TRUE
      IN IF CanAlloc(f, sz) THEN                            \* cc:781
           [res |-> "DEF",
            cur |-> curA,
            fut |-> DoAlloc(f, i, sz),
            rel |-> curA,                                   \* cc:787
            pa  |-> << [inst |-> i, size |-> sz, lastSeq |-> seqNow] >>, \* cc:783-784
            ready772 |-> r772, dup |-> HasTag(f, i)]
         ELSE
           [res |-> "IF", cur |-> curA,
            fut |-> f,                                      \* cc:790-791 stale, kept as code leaves it
            rel |-> relA, pa |-> paA, ready772 |-> r772, dup |-> FALSE]
  ELSE                                                      \* cc:795-806
    IF CanAlloc(futA, sz) THEN                              \* cc:798
      [res |-> "DEF", cur |-> curA, fut |-> DoAlloc(futA, i, sz),
       rel |-> relA,
       pa  |-> Append(paA, [inst |-> i, size |-> sz, lastSeq |-> seqNow]), \* cc:800-801
       ready772 |-> FALSE, dup |-> HasTag(futA, i)]
    ELSE
      [res |-> "IF", cur |-> curA, fut |-> futA, rel |-> relA,
       pa |-> paA, ready772 |-> FALSE, dup |-> FALSE]

-----------------------------------------------------------------------------
(***************************************************************************)
(* FIX_CAP variant of the admission test (BUG-1 fix blueprint,             *)
(* bugs/BUG-1.md as amended).                                              *)
(* C++ sites: the allocate_storage_deferrable deferral path (cc:712-717)   *)
(* gains a DeferredCreate::seqid_cap field (atomic snapshot of             *)
(* cur_release_seqid at REQUEST time); attempt_deferrable_allocation       *)
(* (cc:749-807) replaces the cc:768-779 arrival-order rebuild and the      *)
(* cc:798 append-test with the monotone-cap guard + capped canonical       *)
(* replay below; admission records last_release_seqid := cap instead of    *)
(* cur_release_seqid (cc:784/801).  Releases newer than the cap can no     *)
(* longer fund the allocation, so a release whose precondition depends on  *)
(* this instance's eCreated is never load-bearing for it - the BUG-1       *)
(* cycle is cut.  Drain, ARR and RPR are unchanged.                        *)
(***************************************************************************)

\* releases in prA with seq <= bound, applied in list order.  List order IS
\* seq order: every push uses ++cur_release_seqid, so seqs are strictly
\* increasing along pending_releases and erasures preserve that.
RECURSIVE ApplyCappedFrees(_, _, _)
ApplyCappedFrees(aA, prA, bound) ==
  IF prA = <<>> \/ Head(prA).seq > bound THEN aA
  ELSE ApplyCappedFrees(FreeMissingOk(aA, Head(prA).inst), Tail(prA), bound)

\* canonical replay: queued allocs placed at their lastSeq watermarks,
\* interleaved with surviving releases (a release funds only if its seq is
\* <= both the next alloc's watermark and the bound); trailing releases with
\* seq <= bound applied after the last alloc.  Frees are missing-ok
\* (survivors for FAILED instances have no tag - mirrors cc:778).
\* ok = FALSE if a queued alloc fails to place: prior admissions all tested
\* this same canonical state, so by determinism the C++ fix would assert
\* this cannot happen - a FALSE here is a fix bug (surfaced via
\* structuralAssertFailed at the call sites).
RECURSIVE CanonReplay(_, _, _, _)
CanonReplay(aA, prA, paA, bound) ==
  IF paA = <<>>
  THEN [a |-> ApplyCappedFrees(aA, prA, bound), ok |-> TRUE, dup |-> FALSE]
  ELSE LET al == Head(paA)
       IN IF prA /= <<>> /\ Head(prA).seq <= al.lastSeq
                         /\ Head(prA).seq <= bound
          THEN CanonReplay(FreeMissingOk(aA, Head(prA).inst),
                           Tail(prA), paA, bound)
          ELSE IF CanAlloc(aA, al.size)
               THEN LET r == CanonReplay(DoAlloc(aA, al.inst, al.size),
                                         prA, Tail(paA), bound)
                    IN [a |-> r.a, ok |-> r.ok,
                        dup |-> HasTag(aA, al.inst) \/ r.dup]
               ELSE [a |-> aA, ok |-> FALSE, dup |-> FALSE]

ADAResCap(i, sz, cap, curA, futA, relA, paA, prA) ==
  IF paA = <<>> /\ CanAlloc(curA, sz) THEN
    \* cc:754-757 current-allocator fast path, kept verbatim under FIX_CAP
    [res |-> "IS", cur |-> DoAlloc(curA, i, sz), fut |-> futA,
     rel |-> relA, pa |-> paA, ready772 |-> FALSE,
     dup |-> HasTag(curA, i), capAssert |-> FALSE]
  ELSE IF paA /= <<>> /\ cap < paA[Len(paA)].lastSeq THEN
    \* monotone-cap guard: keeps lastSeq non-decreasing along the queue
    \* (C++: instant failure; caller poisons eCreated)
    [res |-> "IF", cur |-> curA, fut |-> futA, rel |-> relA,
     pa |-> paA, ready772 |-> FALSE, dup |-> FALSE, capAssert |-> FALSE]
  ELSE
    \* capped canonical test on a copy of cur.  If cap < seq of the front
    \* survivor the funding set is empty and this degenerates to a cur-only
    \* test (the intended C++ fast path).  NO ready-fold in v1: is_ready
    \* entries with seq > cap do NOT fund (agreed: spurious instant-fail
    \* acceptable; ARR still helps opportunistically, unchanged).
    LET t == CanonReplay(curA, prA, paA, cap)
    IN IF t.ok /\ CanAlloc(t.a, sz)
       THEN LET newPa  == Append(paA,
                                 [inst |-> i, size |-> sz, lastSeq |-> cap])
                maxSeq == IF prA = <<>> THEN cap ELSE prA[Len(prA)].seq
                f      == CanonReplay(curA, prA, newPa,
                                      IF maxSeq > cap THEN maxSeq ELSE cap)
            IN [res |-> "DEF", cur |-> curA,
                \* fut maintained CANONICALLY (cur + ALL surviving releases
                \* + ALL queued allocs at their watermarks, incl. the new
                \* one at cap): INV_FutureOffsetConsistency is expected to
                \* HOLD under FIX_CAP - a canonical-vs-drain placement
                \* disagreement is a fix bug TLC must catch.
                fut |-> f.a,
                rel |-> IF paA = <<>> THEN curA ELSE relA,  \* cc:787 analog
                pa  |-> newPa, ready772 |-> FALSE,
                dup |-> f.dup, capAssert |-> ~f.ok]
       ELSE
            [res |-> "IF", cur |-> curA, fut |-> futA, rel |-> relA,
             pa |-> paA, ready772 |-> FALSE, dup |-> FALSE,
             capAssert |-> ~t.ok]

\* dispatcher: cap = reqCap[i] for trigger-deferred creates, seqCtr-now for
\* request-triggered ones (callers pass it); legacy path ignores cap.
ADAResv(i, sz, cap, curA, futA, relA, paA, prA, seqNow) ==
  IF FIX_CAP THEN ADAResCap(i, sz, cap, curA, futA, relA, paA, prA)
  ELSE ADARes(i, sz, curA, futA, relA, paA, prA, seqNow)

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.6: ARR - attempt_release_reordering (cc:1207-1324).           *)
(* Pure helper; called only with paA # <<>>.                               *)
(* Returns [changed, cur, fut, rel, pa, pr, placed, notified, erasedReady].*)
(* On changed = FALSE the caller keeps its own values (unwind, cc:1320).   *)
(***************************************************************************)

\* greedy prefix of pending allocs onto test allocator (cc:1220-1231)
RECURSIVE Greedy(_, _)
Greedy(test, pa) ==
  IF pa = <<>> \/ ~CanAlloc(test, Head(pa).size)
  THEN [test |-> test, n |-> 0, placed |-> <<>>, dup |-> FALSE]
  ELSE LET a   == Head(pa)
           off == FirstFitOff(test, a.size)                 \* placement-time offset (FIX 1)
           r   == Greedy(DoAlloc(test, a.inst, a.size), Tail(pa))
       IN [test |-> r.test, n |-> 1 + r.n,
           placed |-> <<[inst |-> a.inst, off |-> off]>> \o r.placed,
           dup |-> HasTag(test, a.inst) \/ r.dup]

\* trailing non-ready releases after all allocs replayed (cc:1284-1289)
RECURSIVE TrailingFrees(_, _, _)
TrailingFrees(tf, pr, idx) ==
  IF idx > Len(pr) THEN tf
  ELSE TrailingFrees(IF pr[idx].isReady THEN tf
                     ELSE FreeMissingOk(tf, pr[idx].inst),  \* cc:1286 missing_ok=TRUE
                     pr, idx + 1)

\* partial-path replay (cc:1258-1289): for each remaining alloc, first apply
\* non-ready releases with seq <= its lastSeq, then allocate; trailing
\* non-ready releases after the last alloc.  Ready entries advance the walk
\* but are NOT re-applied (already inside test, DESIGN 4.6 note).
RECURSIVE Replay(_, _, _, _)
Replay(tf, pr, idx, paRem) ==
  IF paRem = <<>> THEN
    [ok |-> TRUE, tf |-> TrailingFrees(tf, pr, idx), dup |-> FALSE]
  ELSE LET a == Head(paRem)
       IN IF idx <= Len(pr) /\ pr[idx].seq <= a.lastSeq     \* cc:1260-1266
          THEN Replay(IF pr[idx].isReady THEN tf
                      ELSE FreeMissingOk(tf, pr[idx].inst), \* cc:1263 missing_ok=TRUE
                      pr, idx + 1, paRem)
          ELSE IF CanAlloc(tf, a.size)                      \* cc:1270-1272
               THEN LET rr == Replay(DoAlloc(tf, a.inst, a.size),
                                     pr, idx, Tail(paRem))
                    IN [ok |-> rr.ok, tf |-> rr.tf,
                        dup |-> HasTag(tf, a.inst) \/ rr.dup]
               ELSE [ok |-> FALSE, tf |-> tf, dup |-> FALSE] \* cc:1274-1276 -> unwind

ARRNone == [changed |-> FALSE]

ARRFun(curA, futA, relA, paA, prA) ==
  \* front-only gate (cc:1211-1215); guarantees the cc:1233 assert(n >= 1)
  IF ~CanAlloc(relA, Head(paA).size)
  THEN ARRNone
  ELSE LET g == Greedy(relA, paA)
       IN IF g.n = Len(paA) THEN
            \* full success (cc:1236-1252): cur := test, clear allocs,
            \* erase ready releases; fut and rel left stale (cc:1252 note)
            [changed |-> TRUE,
             cur |-> g.test, fut |-> futA, rel |-> relA,
             pa  |-> <<>>,
             pr  |-> NonReadyOnly(prA),                     \* cc:1241-1250
             placed |-> g.placed,
             notified |-> ReadyDefNoteSet(prA),             \* cc:1244-1245
             erasedReady |-> ReadySet(prA),
             dup |-> g.dup]
          ELSE
            LET rp == Replay(g.test, prA, 1,
                             SubSeq(paA, g.n + 1, Len(paA)))  \* cc:1255-1289
            IN IF rp.ok THEN
                 [changed |-> TRUE,
                  cur |-> g.test,                           \* cc:1306
                  fut |-> rp.tf,                            \* cc:1307
                  rel |-> g.test,                           \* cc:1309 rel := cur'
                  pa  |-> SubSeq(paA, g.n + 1, Len(paA)),   \* cc:1304
                  pr  |-> NonReadyOnly(prA),                \* cc:1292-1301
                  placed |-> g.placed,
                  notified |-> ReadyDefNoteSet(prA),        \* cc:1295-1296
                  erasedReady |-> ReadySet(prA),
                  dup |-> g.dup \/ rp.dup]
               ELSE ARRNone                                 \* cc:1311-1321 unwind

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.5 oldest-path machinery: the cc:1640-1700 do-while drain and  *)
(* its nested unblock scan (cc:1649-1695).                                 *)
(***************************************************************************)

\* unblock scan at drain position k: cur already reflects the k-th free;
\* fut is the untouched future_allocator (read-only cross-checks).
RECURSIVE UnblockScan(_, _, _, _, _)
UnblockScan(paA, pr, k, curA, futA) ==
  IF paA = <<>>
  THEN [cur |-> curA, pa |-> paA, succ |-> <<>>,
        ubFail |-> FALSE, futBad |-> FALSE, structF |-> FALSE, dup |-> FALSE]
  ELSE LET a == Head(paA)
       IN IF k + 1 <= Len(pr) /\ a.lastSeq >= pr[k+1].seq   \* cc:1654-1657 break
          THEN [cur |-> curA, pa |-> paA, succ |-> <<>>,
                ubFail |-> FALSE, futBad |-> FALSE, structF |-> FALSE,
                dup |-> FALSE]
          ELSE
            LET orderBad == a.lastSeq < pr[k].seq           \* cc:1662 DEBUG assert
            IN IF ~CanAlloc(curA, a.size)
               THEN \* cc:1670 assert(ok) fails: flag and stop the scan
                    [cur |-> curA, pa |-> paA, succ |-> <<>>,
                     ubFail |-> TRUE, futBad |-> FALSE, structF |-> orderBad,
                     dup |-> FALSE]
               ELSE
                 LET off  == FirstFitOff(curA, a.size)      \* placement-time (cc:1668)
                     cur2 == DoAlloc(curA, a.inst, a.size)
                     fm   == FirstRelIdx(pr, a.inst)        \* cc:1679-1683 begin()-first
                     fb   == IF HasTag(futA, a.inst)        \* cc:1674-1677
                             THEN ~(futA[a.inst].first = off /\
                                    futA[a.inst].size  = a.size)
                             ELSE fm /= 0 /\ pr[fm].isReady \* cc:1687 must be !ready
                     sf   == orderBad \/
                             (~HasTag(futA, a.inst) /\ fm = 0) \* cc:1682 off-end
                     r    == UnblockScan(Tail(paA), pr, k, cur2, futA)
                 IN [cur |-> r.cur, pa |-> r.pa,
                     succ |-> <<[inst |-> a.inst, off |-> off]>> \o r.succ,
                     ubFail |-> r.ubFail,
                     futBad |-> fb \/ r.futBad,
                     structF |-> sf \/ r.structF,
                     dup |-> HasTag(curA, a.inst) \/ r.dup]

\* the do-while (cc:1640-1700): frees pr[k] from cur, runs the unblock scan,
\* continues while the next entry is ready.  Returns k' = first surviving
\* index (cc:1702 erases the prefix).
RECURSIVE DrainLoop(_, _, _, _, _, _)
DrainLoop(pr, k, i, curA, futA, paA) ==
  LET e    == pr[k]
      miss == ~HasTag(curA, e.inst)                         \* cc:1641 missing_ok=FALSE
      cur1 == FreeMissingOk(curA, e.inst)
      nf   == IF e.inst /= i /\ e.defNote THEN {e.inst} ELSE {} \* cc:1644-1645
      s    == UnblockScan(paA, pr, k, cur1, futA)
      cont == k + 1 <= Len(pr) /\ pr[k+1].isReady           \* cc:1700
  IN IF cont
     THEN LET r == DrainLoop(pr, k + 1, i, s.cur, futA, s.pa)
          IN [cur |-> r.cur, pa |-> r.pa, k |-> r.k,
              succ |-> s.succ \o r.succ,
              notified |-> nf \cup r.notified,
              curFreedSet |-> {e.inst} \cup r.curFreedSet,
              missFree |-> miss \/ r.missFree,
              ubFail |-> s.ubFail \/ r.ubFail,
              futBad |-> s.futBad \/ r.futBad,
              structF |-> s.structF \/ r.structF,
              dup |-> s.dup \/ r.dup]
     ELSE [cur |-> s.cur, pa |-> s.pa, k |-> k + 1,
           succ |-> s.succ, notified |-> nf,
           curFreedSet |-> {e.inst}, missFree |-> miss,
           ubFail |-> s.ubFail, futBad |-> s.futBad, structF |-> s.structF,
           dup |-> s.dup]

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.7 machinery: remove_pending_release (cc:1538-1597).           *)
(***************************************************************************)

\* inner alloc-replay loop (cc:1579-1593): consume allocs with
\* lastSeq <= seqid; successes stay in the list (it2 advances), failures are
\* erased and reported.  Returns [fut, kept, failed, rest].
RECURSIVE InnerAllocs(_, _, _)
InnerAllocs(futA, paA, seqid) ==
  IF paA = <<>> \/ Head(paA).lastSeq > seqid
  THEN [fut |-> futA, kept |-> <<>>, failed |-> {}, rest |-> paA,
        dup |-> FALSE]
  ELSE LET a == Head(paA)
       IN IF CanAlloc(futA, a.size)                         \* cc:1581
          THEN LET r == InnerAllocs(DoAlloc(futA, a.inst, a.size),
                                    Tail(paA), seqid)
               IN [fut |-> r.fut, kept |-> <<a>> \o r.kept,
                   failed |-> r.failed, rest |-> r.rest,
                   dup |-> HasTag(futA, a.inst) \/ r.dup]
          ELSE LET r == InnerAllocs(futA, Tail(paA), seqid) \* cc:1591-1592 erase
               IN [fut |-> r.fut, kept |-> r.kept,
                   failed |-> {a.inst} \cup r.failed, rest |-> r.rest,
                   dup |-> r.dup]

\* outer walk (cc:1562-1595).  The erased target still contributes its saved
\* seqid (cc:1564).  NOTE (BUG-5, DESIGN 4.7/8): there is no trailing alloc
\* replay after the loop - allocs with lastSeq above every walked seqid are
\* never re-placed into the rebuilt fut.  Faithful to the code.
RECURSIVE RPRLoop(_, _, _, _, _, _)
RPRLoop(pr, idx, i, found, futA, paA) ==
  IF idx > Len(pr)
  THEN [pr |-> <<>>, fut |-> futA, paOut |-> paA,
        trail |-> paA,   \* the never-examined remainder (it2's final
                         \* position onward) - the ONLY part FIX_RPR's
                         \* trailing pass may see (bugs/DUPALLOC-TRIAGE.md)
        failed |-> {}, bad |-> FALSE, foundOut |-> found, dup |-> FALSE]
  ELSE LET e     == pr[idx]
           isT   == e.inst = i /\ ~found                    \* cc:1567 first match only
           fut1  == IF isT THEN futA
                    ELSE FreeMissingOk(futA, e.inst)        \* cc:1573 missing_ok=TRUE
           inner == InnerAllocs(fut1, paA, e.seq)           \* cc:1579 saved seqid
           badH  == inner.failed /= {} /\ ~(found \/ isT)   \* cc:1587 assert(found)
           r     == RPRLoop(pr, idx + 1, i, found \/ isT, inner.fut, inner.rest)
       IN [pr |-> (IF isT THEN r.pr ELSE <<e>> \o r.pr),
           fut |-> r.fut,
           paOut |-> inner.kept \o r.paOut,   \* = full survivors: kept \o trail
           trail |-> r.trail,
           failed |-> inner.failed \cup r.failed,
           bad |-> badH \/ r.bad,
           foundOut |-> r.foundOut,
           dup |-> inner.dup \/ r.dup]

\* FIX_RPR (BUG-5 fix, bugs/BUG-5.md): trailing alloc replay.  C++: after
\* the outer walk in remove_pending_release ends at cc:1595, run the
\* cc:1579-1594 inner loop ONCE MORE with no seqid bound - allocs whose
\* lastSeq exceeds every walked seqid (possible after ARR-partial erased the
\* ready release they recorded) are otherwise neither refunded into fut nor
\* failed, and strand forever.  Placement mirrors InnerAllocs: success lands
\* in fut and the alloc stays queued (lastSeq unchanged); failure is
\* EVENTUAL_FAILUREd exactly like the in-walk path (failedVia "RPR",
\* eCreated poisoned, erased - downstream poison cascades are correct).
\* The cc:1587 assert(found) analog is trivially satisfied here: the target
\* erase happened before any trailing processing (no 'bad' field needed).
RECURSIVE TrailingRPR(_, _)
TrailingRPR(futA, paA) ==
  IF paA = <<>>
  THEN [fut |-> futA, kept |-> <<>>, failed |-> {}, dup |-> FALSE]
  ELSE LET a == Head(paA)
       IN IF CanAlloc(futA, a.size)
          THEN LET r == TrailingRPR(DoAlloc(futA, a.inst, a.size), Tail(paA))
               IN [fut |-> r.fut, kept |-> <<a>> \o r.kept,
                   failed |-> r.failed,
                   dup |-> HasTag(futA, a.inst) \/ r.dup]
          ELSE LET r == TrailingRPR(futA, Tail(paA))
               IN [fut |-> r.fut, kept |-> r.kept,
                   failed |-> {a.inst} \cup r.failed, dup |-> r.dup]

-----------------------------------------------------------------------------
(* batch-update helpers over the per-instance maps.  These read the        *)
(* current (unprimed) variables; actions apply them to compute primes.     *)

\* eCreated after firing succS clean (EVENTUAL/INSTANT success,
\* ii:1201-1202) and poisoning failS (ii:1121-1122)
ECreatedAfter(succS, failS) ==
  [j \in INSTANCES |-> IF j \in succS THEN "CLEAN"
                       ELSE IF j \in failS THEN "POISONED"
                       ELSE eCreated[j]]

\* instState after: failures -> FAILED; successful allocs -> ALLOCATED
\* (or DESTROYED if their dealloc notify fires in the same action);
\* notified ALLOCATED instances -> DESTROYED; others keep their state.
StatusAfter(succS, failS, notifS) ==
  [j \in INSTANCES |->
     IF j \in failS THEN "FAILED"
     ELSE IF j \in notifS
          THEN IF j \in succS \/ instState[j] = "ALLOCATED"
               THEN "DESTROYED" ELSE instState[j]
     ELSE IF j \in succS THEN "ALLOCATED"
     ELSE instState[j]]

\* instOffset after: successes read their placement from allocator a2
\* (all placed tags end up in the final cur - DESIGN 4.5/4.6)
OffsetsAfter(a2, succS, failS) ==
  [j \in INSTANCES |-> IF j \in succS THEN a2[j].first
                       ELSE IF j \in failS THEN OFF_FAILED
                       ELSE instOffset[j]]

NotifyAfter(S) ==
  [j \in INSTANCES |-> notifyCount[j] + (IF j \in S THEN 1 ELSE 0)]

InstsOf(pairs) == {p.inst : p \in pairs}

\* placement-time offsets (FIX 1): the C++ captures each unblocked alloc's
\* offset at placement time (cc:1668, cc:1693) and never re-reads the
\* allocator; re-reading the FINAL allocator is wrong when a later ready
\* entry drained in the same action frees the just-placed tag (reachable
\* with contract C2 off), and aborts TLC on the missing key.  pairs is a
\* set of [inst, off] records; an instance is placed at most once per action.
OffsetsAfterPairs(pairs, failS) ==
  [j \in INSTANCES |->
     IF \E p \in pairs : p.inst = j
     THEN (CHOOSE p \in pairs : p.inst = j).off
     ELSE IF j \in failS THEN OFF_FAILED ELSE instOffset[j]]

FailedViaAfter(S, tag) ==
  [j \in INSTANCES |-> IF j \in S THEN tag ELSE failedVia[j]]

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.2: RequestCreate(i, trig, pois)                               *)
(* allocate_storage_deferrable (cc:693-745).  trig/pois describe preC[i]   *)
(* at request time (supplied by MC).                                       *)
(***************************************************************************)
RequestCreate(i, trig, pois) ==
  /\ instState[i] = "UNREQUESTED"
  /\ IF trig /\ pois THEN
       \* ALLOC_CANCELLED (cc:703-708); eCreated poisoned (ii:1121-1122)
       /\ instState'  = [instState EXCEPT ![i] = "FAILED"]
       /\ instOffset' = [instOffset EXCEPT ![i] = OFF_FAILED]
       /\ eCreated'   = [eCreated EXCEPT ![i] = "POISONED"]
       /\ failedVia'  = [failedVia EXCEPT ![i] = "CANCELLED"]
       /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                       reqCap, allocatedEver, curFreed, missingFree,
                       structuralAssertFailed, unblockFailed, futMismatch,
                       poisonReplayBad, readyAtRebuild, dupAlloc, wasDeferred,
                       notifyCount >>
     ELSE IF ~trig THEN
       \* defer (cc:712-717): INSTOFFSET_DELAYEDALLOC, waiter registered
       \* FIX_CAP: snapshot cur_release_seqid at REQUEST time (C++: new
       \* DeferredCreate::seqid_cap field, set in the cc:712-717 deferral
       \* path under the atomicity the C++ needs for that read)
       /\ instState' = [instState EXCEPT ![i] = "CREATE_PENDING"]
       /\ reqCap' = IF FIX_CAP THEN [reqCap EXCEPT ![i] = seqCtr] ELSE reqCap
       /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                       instOffset, eCreated, allocatedEver, curFreed,
                       missingFree, structuralAssertFailed, unblockFailed,
                       futMismatch, poisonReplayBad, readyAtRebuild, dupAlloc,
                       wasDeferred, failedVia, notifyCount >>
     ELSE
       \* triggered clean: ADA (cc:734-736); request == trigger, so under
       \* FIX_CAP the cap is seqCtr-now (uniform with the deferred case)
       LET r == ADAResv(i, Size[i], seqCtr, cur, fut, rel,
                        pendingAllocs, pendingReleases, seqCtr)
       IN /\ cur' = r.cur  /\ fut' = r.fut  /\ rel' = r.rel
          /\ pendingAllocs' = r.pa
          /\ readyAtRebuild' = (readyAtRebuild \/ r.ready772)
          /\ dupAlloc' = (dupAlloc \/ r.dup)
          \* FIX_CAP: capAssert = canonical-replay-failed (fix-internal
          \* assert the intended C++ would carry)
          /\ structuralAssertFailed' =
               (structuralAssertFailed \/ (IF FIX_CAP THEN r.capAssert
                                           ELSE FALSE))
          /\ UNCHANGED reqCap
          /\ CASE r.res = "IS" ->
                    /\ instState'  = [instState EXCEPT ![i] = "ALLOCATED"]
                    /\ instOffset' = [instOffset EXCEPT ![i] = r.cur[i].first]
                    /\ eCreated'   = [eCreated EXCEPT ![i] = "CLEAN"]
                    /\ allocatedEver' = allocatedEver \cup {i}
                    /\ UNCHANGED << wasDeferred, failedVia >>
               [] r.res = "DEF" ->
                    /\ instState' = [instState EXCEPT ![i] = "ALLOC_DEFERRED"]
                    /\ wasDeferred' = wasDeferred \cup {i}
                    /\ UNCHANGED << instOffset, eCreated, allocatedEver,
                                    failedVia >>
               [] r.res = "IF" ->
                    /\ instState'  = [instState EXCEPT ![i] = "FAILED"]
                    /\ instOffset' = [instOffset EXCEPT ![i] = OFF_FAILED]
                    /\ eCreated'   = [eCreated EXCEPT ![i] = "POISONED"]
                    /\ failedVia'  = [failedVia EXCEPT ![i] = "INSTANT"]
                    /\ UNCHANGED << wasDeferred, allocatedEver >>
          /\ UNCHANGED << pendingReleases, seqCtr, curFreed, missingFree,
                          unblockFailed, futMismatch,
                          poisonReplayBad, notifyCount >>

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.3: TriggerCreate(i, pois)                                     *)
(* DeferredCreate::event_triggered -> allocate_storage_immediate           *)
(* (ii:49-56; cc:1087-1176).  pois = preC[i] fired poisoned.               *)
(***************************************************************************)
TriggerCreate(i, pois) ==
  /\ instState[i] \in {"CREATE_PENDING", "CREATE_PENDING_DESTROY"}
  /\ LET dd == instState[i] = "CREATE_PENDING_DESTROY"      \* cc:1106-1107
         r  == IF pois
               THEN [res |-> "CANC", cur |-> cur, fut |-> fut, rel |-> rel,
                     pa |-> pendingAllocs, ready772 |-> FALSE,
                     dup |-> FALSE, capAssert |-> FALSE]    \* cc:1113-1115
               ELSE ADAResv(i, Size[i], reqCap[i], cur, fut, rel,
                            pendingAllocs, pendingReleases, seqCtr) \* cc:1136-1138
                    \* FIX_CAP: cap = seqid_cap snapshot from request time
         seq2 == IF dd THEN seqCtr + 1 ELSE seqCtr          \* cc:1147 ++seqid
         pr2  == IF dd                                      \* cc:1146-1147 unconditional push
                 THEN Append(pendingReleases,
                             [inst |-> i, isReady |-> FALSE,
                              seq |-> seq2, defNote |-> FALSE])
                 ELSE pendingReleases
         fut2 == IF dd /\ r.res \in {"IS", "DEF"} /\ r.pa /= <<>>
                 THEN FreeMissingOk(r.fut, i)               \* cc:1150-1153
                 ELSE r.fut
     IN /\ cur' = r.cur  /\ fut' = fut2  /\ rel' = r.rel
        /\ pendingAllocs' = r.pa
        /\ pendingReleases' = pr2
        /\ seqCtr' = seq2
        /\ readyAtRebuild' = (readyAtRebuild \/ r.ready772)
        /\ dupAlloc' = (dupAlloc \/ r.dup)
        /\ structuralAssertFailed' =
             (structuralAssertFailed \/ (IF FIX_CAP THEN r.capAssert
                                         ELSE FALSE))
        \* FIX_CAP: the seqid_cap dies with the DeferredCreate object;
        \* reset to 0 purely to canonicalize the model state
        /\ reqCap' = IF FIX_CAP THEN [reqCap EXCEPT ![i] = 0] ELSE reqCap
        /\ CASE r.res = "IS" ->
                  /\ instState'  = [instState EXCEPT ![i] = "ALLOCATED"]
                  /\ instOffset' = [instOffset EXCEPT ![i] = r.cur[i].first]
                  /\ eCreated'   = [eCreated EXCEPT ![i] = "CLEAN"]
                  /\ allocatedEver' = allocatedEver \cup {i}
                  /\ UNCHANGED << wasDeferred, failedVia >>
             [] r.res = "DEF" ->
                  /\ instState' = [instState EXCEPT ![i] = "ALLOC_DEFERRED"]
                  /\ wasDeferred' = wasDeferred \cup {i}
                  /\ UNCHANGED << instOffset, eCreated, allocatedEver,
                                  failedVia >>
             [] r.res = "IF" ->
                  /\ instState'  = [instState EXCEPT ![i] = "FAILED"]
                  /\ instOffset' = [instOffset EXCEPT ![i] = OFF_FAILED]
                  /\ eCreated'   = [eCreated EXCEPT ![i] = "POISONED"]
                  /\ failedVia'  = [failedVia EXCEPT ![i] = "INSTANT"]
                  /\ UNCHANGED << wasDeferred, allocatedEver >>
             [] r.res = "CANC" ->
                  /\ instState'  = [instState EXCEPT ![i] = "FAILED"]
                  /\ instOffset' = [instOffset EXCEPT ![i] = OFF_FAILED]
                  /\ eCreated'   = [eCreated EXCEPT ![i] = "POISONED"]
                  /\ failedVia'  = [failedVia EXCEPT ![i] = "CANCELLED"]
                  /\ UNCHANGED << wasDeferred, allocatedEver >>
        /\ UNCHANGED << curFreed, missingFree,
                        unblockFailed, futMismatch, poisonReplayBad,
                        notifyCount >>

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.4: RequestDestroy(i, trig, pois)                              *)
(* release_storage_deferrable (cc:810-924).  trig/pois describe preD[i]    *)
(* at request time.  MC guarantees at most one destroy request per inst.   *)
(***************************************************************************)
RequestDestroy(i, trig, pois) ==
  /\ instState[i] \notin {"UNREQUESTED", "DESTROYED"}
  /\ UNCHANGED reqCap   \* release path never touches the FIX_CAP snapshot
  /\ IF trig /\ pois THEN
       \* silent cancel (cc:818-825)
       UNCHANGED protoVars
     ELSE IF instState[i] = "CREATE_PENDING" THEN
       \* DELAYEDALLOC -> DELAYEDDESTROY (cc:845-849); cc:846 assert(!triggered)
       \* (FIX 4: with trig, a release build proceeds past the assert and
       \*  still acks the triggered destroy at cc:915-917 - model as-code;
       \*  the state is already condemned by INV_StructuralAsserts)
       /\ instState' = [StatusAfter({}, {}, IF trig THEN {i} ELSE {})
                          EXCEPT ![i] = "CREATE_PENDING_DESTROY"]
       /\ structuralAssertFailed' = (structuralAssertFailed \/ trig)
       /\ notifyCount' = NotifyAfter(IF trig THEN {i} ELSE {})
       /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                       instOffset, eCreated, allocatedEver, curFreed,
                       missingFree, unblockFailed, futMismatch,
                       poisonReplayBad, readyAtRebuild, dupAlloc, wasDeferred,
                       failedVia >>
     ELSE IF pendingAllocs = <<>> THEN                      \* cc:851-860
       IF trig THEN
         \* apply directly to current state (cc:852-855), ack (cc:915-917)
         LET failedI == instState[i] = "FAILED"
             miss    == ~failedI /\ ~HasTag(cur, i)         \* cc:855 missing_ok=FALSE
         IN /\ cur' = IF failedI THEN cur ELSE FreeMissingOk(cur, i)
            /\ curFreed' = IF failedI THEN curFreed ELSE curFreed \cup {i}
            /\ missingFree' = (missingFree \/ miss)
            /\ notifyCount' = NotifyAfter({i})
            /\ instState' = StatusAfter({}, {}, {i})
            /\ UNCHANGED << fut, rel, pendingAllocs, pendingReleases, seqCtr,
                            instOffset, eCreated, allocatedEver,
                            structuralAssertFailed, unblockFailed,
                            futMismatch, poisonReplayBad, readyAtRebuild,
                            dupAlloc, wasDeferred, failedVia >>
       ELSE
         \* push, no future state yet (cc:857-859); waiter registered (cc:920)
         /\ seqCtr' = seqCtr + 1
         /\ pendingReleases' = Append(pendingReleases,
                                      [inst |-> i, isReady |-> FALSE,
                                       seq |-> seqCtr + 1, defNote |-> FALSE])
         /\ UNCHANGED << cur, fut, rel, pendingAllocs, instState, instOffset,
                         eCreated, allocatedEver, curFreed, missingFree,
                         structuralAssertFailed, unblockFailed, futMismatch,
                         poisonReplayBad, readyAtRebuild, dupAlloc,
                         wasDeferred, failedVia, notifyCount >>
     ELSE                                                   \* cc:861-897
       IF trig THEN
         IF instState[i] = "FAILED" THEN
           \* ready destruction of a failed alloc: skip heap (cc:866-868),
           \* still acked (cc:915-917)
           /\ notifyCount' = NotifyAfter({i})
           /\ instState' = StatusAfter({}, {}, {i})
           /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases,
                           seqCtr, instOffset, eCreated, allocatedEver,
                           curFreed, missingFree, structuralAssertFailed,
                           unblockFailed, futMismatch, poisonReplayBad,
                           readyAtRebuild, dupAlloc, wasDeferred, failedVia >>
         ELSE
           \* cc:871-872 missing_ok=FALSE frees, then ARR (cc:875-876)
           LET miss == ~HasTag(rel, i) \/ ~HasTag(fut, i)
               rel1 == FreeMissingOk(rel, i)
               fut1 == FreeMissingOk(fut, i)
               arr  == ARRFun(cur, fut1, rel1, pendingAllocs, pendingReleases)
           IN IF arr.changed THEN
                LET succP  == ToSet(arr.placed)
                    succS  == InstsOf(succP)
                    notifS == arr.notified \cup {i}         \* i: cc:915-917; defNote drains: cc:922-923
                IN /\ cur' = arr.cur /\ fut' = arr.fut /\ rel' = arr.rel
                   /\ pendingAllocs' = arr.pa
                   /\ pendingReleases' = arr.pr
                   /\ eCreated'   = ECreatedAfter(succS, {})
                   /\ instState'  = StatusAfter(succS, {}, notifS)
                   /\ instOffset' = OffsetsAfterPairs(succP, {})
                   /\ notifyCount' = NotifyAfter(notifS)
                   /\ allocatedEver' = allocatedEver \cup succS
                   /\ curFreed' = curFreed \cup arr.erasedReady \cup {i}
                   /\ missingFree' = (missingFree \/ miss)
                   /\ dupAlloc' = (dupAlloc \/ arr.dup)
                   /\ UNCHANGED << seqCtr, structuralAssertFailed,
                                   unblockFailed, futMismatch,
                                   poisonReplayBad, readyAtRebuild,
                                   wasDeferred, failedVia >>
              ELSE
                \* unwind failed: push ready entry, defer the ack (cc:884-887)
                /\ rel' = rel1 /\ fut' = fut1
                /\ seqCtr' = seqCtr + 1
                /\ pendingReleases' =
                     Append(pendingReleases,
                            [inst |-> i, isReady |-> TRUE,
                             seq |-> seqCtr + 1, defNote |-> TRUE])
                /\ missingFree' = (missingFree \/ miss)
                /\ UNCHANGED << cur, pendingAllocs, instState, instOffset,
                                eCreated, allocatedEver, curFreed,
                                structuralAssertFailed, unblockFailed,
                                futMismatch, poisonReplayBad, readyAtRebuild,
                                dupAlloc, wasDeferred, failedVia,
                                notifyCount >>
       ELSE
         \* untriggered with pending allocs (cc:890-896)
         /\ fut' = IF instState[i] = "FAILED" THEN fut
                   ELSE FreeMissingOk(fut, i)               \* cc:892-893 missing_ok=TRUE
         /\ seqCtr' = seqCtr + 1
         /\ pendingReleases' = Append(pendingReleases,
                                      [inst |-> i, isReady |-> FALSE,
                                       seq |-> seqCtr + 1, defNote |-> FALSE])
         /\ UNCHANGED << cur, rel, pendingAllocs, instState, instOffset,
                         eCreated, allocatedEver, curFreed, missingFree,
                         structuralAssertFailed, unblockFailed, futMismatch,
                         poisonReplayBad, readyAtRebuild, dupAlloc,
                         wasDeferred, failedVia, notifyCount >>

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 4.5: TriggerDestroy(i, pois)                                    *)
(* DeferredDestroy::event_triggered -> release_storage_immediate           *)
(* (ii:81-99; cc:1600-1794).  Enabled by MC once preD[i] has fired for a   *)
(* destroy that was deferred.  pois = fired poisoned.                      *)
(***************************************************************************)

\* clean, oldest path (cc:1634-1717 + tail ARR cc:1751-1753)
TriggerDestroyOldest(i) ==
  LET pr0     == pendingReleases
      pa0     == pendingAllocs
      relMiss == pa0 /= <<>> /\ ~HasTag(rel, i)             \* cc:1636 missing_ok=FALSE
      rel1    == IF pa0 /= <<>> THEN FreeMissingOk(rel, i) ELSE rel \* cc:1635-1637
      d       == DrainLoop(pr0, 1, i, cur, fut, pa0)
      pr1     == SubSeq(pr0, d.k, Len(pr0))                 \* cc:1702 erase prefix
      needRb  == d.succ /= <<>> /\ d.pa /= <<>>             \* cc:1704-1706
      rb      == IF needRb THEN ApplyReadyFreesStrict(d.cur, pr1) \* cc:1707-1717
                 ELSE [a |-> rel1, miss |-> FALSE]
      arrGo   == d.pa /= <<>>                               \* cc:1751-1753
      arr     == IF arrGo THEN ARRFun(d.cur, fut, rb.a, d.pa, pr1) ELSE ARRNone
      chg     == arrGo /\ arr.changed
      cur2    == IF chg THEN arr.cur ELSE d.cur
      fut2    == IF chg THEN arr.fut ELSE fut
      rel2    == IF chg THEN arr.rel ELSE rb.a
      pa2     == IF chg THEN arr.pa  ELSE d.pa
      pr2     == IF chg THEN arr.pr  ELSE pr1
      \* FIX_SWEEP: sweep_ready_releases() when pending_allocs is empty
      \* (C++: release_storage_immediate, after the cc:1702 prefix erase
      \* and the cc:1751 tail ARR).  If pa2 emptied via ARR full-success,
      \* arr.pr has no ready entries left and the sweep is a no-op.
      sw      == IF FIX_SWEEP /\ pa2 = <<>>
                 THEN Sweep(cur2, pr2)
                 ELSE [cur |-> cur2, pr |-> pr2, notified |-> {},
                       freed |-> {}, miss |-> FALSE]
      succP   == ToSet(d.succ) \cup (IF chg THEN ToSet(arr.placed) ELSE {})
      succS   == InstsOf(succP)
      notifS  == d.notified \cup (IF chg THEN arr.notified ELSE {})
                 \cup sw.notified
                 \cup {i}                                   \* i: cc:1789-1790; defNote drains: cc:1792-1793
      freedS  == d.curFreedSet \cup (IF chg THEN arr.erasedReady ELSE {})
                 \cup sw.freed
  IN /\ cur' = sw.cur /\ fut' = fut2 /\ rel' = rel2
     /\ pendingAllocs' = pa2
     /\ pendingReleases' = sw.pr
     /\ eCreated'   = ECreatedAfter(succS, {})
     /\ instState'  = StatusAfter(succS, {}, notifS)
     /\ instOffset' = OffsetsAfterPairs(succP, {})
     /\ notifyCount' = NotifyAfter(notifS)
     /\ allocatedEver' = allocatedEver \cup succS
     /\ curFreed' = curFreed \cup freedS
     /\ missingFree' = (missingFree \/ relMiss \/ d.missFree \/ rb.miss
                        \/ sw.miss)
     /\ unblockFailed' = (unblockFailed \/ d.ubFail)
     /\ futMismatch' = (futMismatch \/ d.futBad)
     /\ structuralAssertFailed' = (structuralAssertFailed \/ d.structF)
     /\ dupAlloc' = (dupAlloc \/ d.dup \/ (IF chg THEN arr.dup ELSE FALSE))
     /\ UNCHANGED << seqCtr, poisonReplayBad, readyAtRebuild, wasDeferred,
                     failedVia >>

\* clean, non-oldest path (cc:1718-1742 + tail ARR cc:1751-1753); the C++
\* find loop starts at the SECOND entry (cc:1719-1721 pre-increments)
TriggerDestroyNonOldest(i) ==
  LET pr0 == pendingReleases
      ks  == {k \in 2..Len(pr0) : pr0[k].inst = i}
  IN IF ks = {} THEN
       \* cc:1720-1723 off-end assert: flag, no other step is defined
       /\ structuralAssertFailed' = TRUE
       /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                       instState, instOffset, eCreated, allocatedEver,
                       curFreed, missingFree, unblockFailed, futMismatch,
                       poisonReplayBad, readyAtRebuild, dupAlloc, wasDeferred,
                       failedVia, notifyCount >>
     ELSE
       LET m   == Min(ks)
           pr1 == [pr0 EXCEPT ![m].isReady = TRUE]          \* cc:1724
       IN IF pendingAllocs = <<>> THEN
            \* apply to current state directly (cc:1726-1730)
            LET miss == ~HasTag(cur, i)                     \* cc:1728 missing_ok=FALSE
            IN /\ cur' = FreeMissingOk(cur, i)
               /\ pendingReleases' = RemoveAt(pr1, m)       \* cc:1730
               /\ curFreed' = curFreed \cup {i}
               /\ missingFree' = (missingFree \/ miss)
               /\ notifyCount' = NotifyAfter({i})           \* cc:1789-1790
               /\ instState' = StatusAfter({}, {}, {i})
               /\ UNCHANGED << fut, rel, pendingAllocs, seqCtr, instOffset,
                               eCreated, allocatedEver,
                               structuralAssertFailed, unblockFailed,
                               futMismatch, poisonReplayBad, readyAtRebuild,
                               dupAlloc, wasDeferred, failedVia >>
          ELSE
            \* apply to release allocator, defer the ack (cc:1731-1741),
            \* then tail ARR (cc:1751-1753)
            LET miss == ~HasTag(rel, i)                     \* cc:1738 missing_ok=FALSE
                rel1 == FreeMissingOk(rel, i)
                pr2  == [pr1 EXCEPT ![m].defNote = TRUE]    \* cc:1739
                arr  == ARRFun(cur, fut, rel1, pendingAllocs, pr2)
            IN IF arr.changed THEN
                 LET succP  == ToSet(arr.placed)
                     succS  == InstsOf(succP)
                     notifS == arr.notified                 \* i included iff its entry drained (cc:1792-1793)
                 IN /\ cur' = arr.cur /\ fut' = arr.fut /\ rel' = arr.rel
                    /\ pendingAllocs' = arr.pa
                    /\ pendingReleases' = arr.pr
                    /\ eCreated'   = ECreatedAfter(succS, {})
                    /\ instState'  = StatusAfter(succS, {}, notifS)
                    /\ instOffset' = OffsetsAfterPairs(succP, {})
                    /\ notifyCount' = NotifyAfter(notifS)
                    /\ allocatedEver' = allocatedEver \cup succS
                    /\ curFreed' = curFreed \cup arr.erasedReady
                    /\ missingFree' = (missingFree \/ miss)
                    /\ dupAlloc' = (dupAlloc \/ arr.dup)
                    /\ UNCHANGED << seqCtr, structuralAssertFailed,
                                    unblockFailed, futMismatch,
                                    poisonReplayBad, readyAtRebuild,
                                    wasDeferred, failedVia >>
               ELSE
                 /\ rel' = rel1
                 /\ pendingReleases' = pr2
                 /\ missingFree' = (missingFree \/ miss)
                 /\ UNCHANGED << cur, fut, pendingAllocs, seqCtr, instState,
                                 instOffset, eCreated, allocatedEver,
                                 curFreed, structuralAssertFailed,
                                 unblockFailed, futMismatch, poisonReplayBad,
                                 readyAtRebuild, dupAlloc, wasDeferred,
                                 failedVia, notifyCount >>

\* poisoned path: remove_pending_release (cc:1538-1597 via cc:1754-1755);
\* no notify_deallocation (cc:1789 guard)
TriggerDestroyPoisoned(i) ==
  IF pendingAllocs = <<>> THEN
    \* cc:1544-1553: erase first entry for i; off-end assert cc:1548-1551
    LET ks == {k \in 1..Len(pendingReleases) : pendingReleases[k].inst = i}
    IN IF ks = {} THEN
         /\ structuralAssertFailed' = TRUE
         /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases,
                         seqCtr, instState, instOffset, eCreated,
                         allocatedEver, curFreed, missingFree, unblockFailed,
                         futMismatch, poisonReplayBad, readyAtRebuild,
                         dupAlloc, wasDeferred, failedVia, notifyCount >>
       ELSE
         /\ pendingReleases' = RemoveAt(pendingReleases, Min(ks))
         /\ UNCHANGED << cur, fut, rel, pendingAllocs, seqCtr, instState,
                         instOffset, eCreated, allocatedEver, curFreed,
                         missingFree, structuralAssertFailed, unblockFailed,
                         futMismatch, poisonReplayBad, readyAtRebuild,
                         dupAlloc, wasDeferred, failedVia, notifyCount >>
  ELSE
    \* cc:1554-1596: rewrite future history; legacy: rel := cur WITHOUT
    \* ready releases (BUG-4, kept faithful when FIX_SWEEP = FALSE); no
    \* trailing alloc replay (BUG-5, unchanged by these fixes)
    LET L     == RPRLoop(pendingReleases, 1, i, FALSE, cur, pendingAllocs)
        \* FIX_RPR (BUG-5): trailing replay AFTER the outer walk ends
        \* (cc:1595) - the cc:1579-1594 inner loop once more, no seqid
        \* bound, against the rebuilt fut.  It CONTINUES from the walk's
        \* final alloc cursor: only L.trail (the never-examined remainder)
        \* is processed - feeding it the walk-KEPT prefix re-DoAllocs
        \* already-placed allocs (the sapling SafetyFixed4 dupAlloc
        \* artifact, bugs/DUPALLOC-TRIAGE.md).  FALSE = identity.
        tr    == IF FIX_RPR THEN TrailingRPR(L.fut, L.trail)
                 ELSE [fut |-> L.fut, kept |-> L.trail, failed |-> {},
                       dup |-> FALSE]
        \* L.paOut = KeptPrefix \o L.trail by construction, so this exactly
        \* reattaches the walk-kept prefix ahead of the trailing survivors
        \* (queue order preserved); with FIX_RPR = FALSE, paF = L.paOut.
        KeptPrefix == SubSeq(L.paOut, 1, Len(L.paOut) - Len(L.trail))
        paF   == IF FIX_RPR THEN KeptPrefix \o tr.kept ELSE L.paOut
        failS == L.failed \cup tr.failed
        \* FIX_SWEEP: sweep_ready_releases() when the queue is empty
        \* (C++: remove_pending_release, after the cc:1562-1595 walk AND
        \* after the FIX_RPR trailing pass).  Swept defNote acks fire like
        \* the cc:1792-1793 loop.  COMPOSITION POINT (verified): the sweep
        \* condition and input use paF, the post-trailing queue - if the
        \* trailing pass fails every remaining alloc and empties the queue,
        \* the sweep still runs; C++ must sequence sweep after the trailing
        \* replay for the same reason.
        sw    == IF FIX_SWEEP /\ paF = <<>>
                 THEN Sweep(cur, L.pr)
                 ELSE [cur |-> cur, pr |-> L.pr, notified |-> {},
                       freed |-> {}, miss |-> FALSE]
        \* FIX_SWEEP (BUG-4-standalone re-apply): when pending_allocs stays
        \* NONEMPTY (post-trailing), re-apply surviving is_ready entries to
        \* the rebuilt rel (mirrors cc:1713-1714) so rel = current + ready
        \* releases (h:399-405) holds again.  rel is invalid when the queue
        \* emptied, so the sweep branch leaves it at the legacy cc:1557
        \* value.
        relRe == IF FIX_SWEEP /\ paF /= <<>>
                 THEN ApplyReadyFreesStrict(cur, L.pr)
                 ELSE [a |-> cur, miss |-> FALSE]
    IN /\ fut' = tr.fut
       /\ rel' = relRe.a                                    \* cc:1557 (legacy: plain cur)
       /\ cur' = sw.cur
       /\ pendingReleases' = sw.pr
       /\ pendingAllocs' = paF
       /\ eCreated'   = ECreatedAfter({}, failS)            \* cc:1591; ii:1121-1122
       /\ instState'  = StatusAfter({}, failS, sw.notified)
       /\ instOffset' = OffsetsAfter(cur, {}, failS)
       /\ failedVia'  = FailedViaAfter(failS, "RPR")
       /\ notifyCount' = NotifyAfter(sw.notified)
       /\ curFreed' = curFreed \cup sw.freed
       /\ missingFree' = (missingFree \/ sw.miss \/ relRe.miss)
       /\ poisonReplayBad' = (poisonReplayBad \/ L.bad)
       \* FIX 2: cc:1542 assert(!pending_releases.empty()) on entry
       /\ structuralAssertFailed' =
            (structuralAssertFailed \/ pendingReleases = <<>>)
       /\ dupAlloc' = (dupAlloc \/ L.dup \/ tr.dup)
       /\ UNCHANGED << seqCtr, allocatedEver,
                       unblockFailed, futMismatch, readyAtRebuild,
                       wasDeferred >>

TriggerDestroy(i, pois) ==
  /\ UNCHANGED reqCap   \* release path never touches the FIX_CAP snapshot
  /\ (IF pois THEN TriggerDestroyPoisoned(i)
      ELSE IF pendingReleases = <<>> THEN
        \* cc:1630 assert(!pending_releases.empty()): flag, no defined step
        /\ structuralAssertFailed' = TRUE
        /\ UNCHANGED << cur, fut, rel, pendingAllocs, pendingReleases, seqCtr,
                        instState, instOffset, eCreated, allocatedEver,
                        curFreed, missingFree, unblockFailed, futMismatch,
                        poisonReplayBad, readyAtRebuild, dupAlloc,
                        wasDeferred, failedVia, notifyCount >>
      ELSE IF Head(pendingReleases).inst = i                \* cc:1634
           THEN TriggerDestroyOldest(i)
           ELSE TriggerDestroyNonOldest(i))

-----------------------------------------------------------------------------
(***************************************************************************)
(* Section 6: invariants (DESIGN 6).  MC selects which to check per config.*)
(***************************************************************************)

Cells(a, t) == a[t].first .. (a[t].first + a[t].size - 1)

UsedCells(a) == UNION {Cells(a, t) : t \in DOMAIN a}

RECURSIVE SumSizes(_)
SumSizes(a) == IF DOMAIN a = {} THEN 0
               ELSE LET t == CHOOSE t \in DOMAIN a : TRUE
                    IN a[t].size + SumSizes(DoFree(a, t))

INV_NoOverlap ==
  \A t1, t2 \in DOMAIN cur :
    t1 /= t2 => Cells(cur, t1) \cap Cells(cur, t2) = {}

INV_InBounds ==
  \A t \in DOMAIN cur :
    cur[t].first >= 0 /\ cur[t].first + cur[t].size <= HEAP_SIZE

\* spec-typo catcher: with no overlap and in-bounds this is an equality
INV_Conservation == Cardinality(UsedCells(cur)) = SumSizes(cur)

INV_CurrentMatchesGround == DOMAIN cur = allocatedEver \ curFreed

\* strengthened cc:772 assert; EXPECTED TO FAIL with FIX_SWEEP = FALSE
\* (BUG-6, DESIGN 6/8) and EXPECTED TO HOLD with FIX_SWEEP = TRUE (the
\* sweep drains every stranded ready entry - that is the fix's claim)
INV_NoReadyWhenNoPendingAllocs ==
  pendingAllocs = <<>> =>
    \A k \in 1..Len(pendingReleases) : ~pendingReleases[k].isReady

\* companion: the exact in-ADA rebuild-site check (cc:772); expected FAIL
\* with FIX_SWEEP = FALSE, expected HOLD with FIX_SWEEP = TRUE.  Note the
\* FIX_CAP admission path has no cc:768 rebuild, so under FIX_CAP this flag
\* can only be set by the legacy path (i.e. never when FIX_CAP = TRUE).
INV_NoReadyAtRebuild == ~readyAtRebuild

\* all missing_ok=FALSE frees found their tag (inl:614):
\* cc:855, cc:871-872, cc:1636, cc:1641, cc:1713-1714, cc:1728, cc:1738
INV_TriggeredDeallocPresent == ~missingFree

INV_StructuralAsserts == ~structuralAssertFailed

INV_NoOrphanTags ==
  \A t \in DOMAIN cur :
    \/ notifyCount[t] = 0
    \/ \E k \in 1..Len(pendingReleases) : pendingReleases[k].inst = t

INV_InOrderUnblockSucceeds == ~unblockFailed                \* cc:1670

\* cc:1674-1691; under FIX_CAP the canonically maintained fut is CLAIMED to
\* agree with drain placement - a violation with FIX_CAP = TRUE is a fix bug
INV_FutureOffsetConsistency == ~futMismatch

INV_PoisonReplayOnlyFailsAfterPoint == ~poisonReplayBad     \* cc:1587

\* FIX 3 detector: the C++ allocate() on a live tag leaks the old range
\* (allocated[tag] = idx, inl:500 - new range wins, old range stays linked:
\* the #442 double-tracking class).  The partial-function representation
\* cannot express the leak, so any materialized DoAlloc onto a live tag is
\* flagged instead of modeled.
INV_NoDupAlloc == ~dupAlloc

\* a DEFERRED promise fails only via RemovePendingRelease (DESIGN 6)
SAFETY_PromisesKept ==
  \A j \in wasDeferred : instState[j] = "FAILED" => failedVia[j] = "RPR"

PROP_NotifyOnceSafety == \A j \in INSTANCES : notifyCount[j] <= 1

-----------------------------------------------------------------------------
(* type and init *)

AllocatorType(a) ==
  /\ DOMAIN a \subseteq INSTANCES
  /\ \A t \in DOMAIN a : a[t] \in [first : 0..HEAP_SIZE, size : 1..HEAP_SIZE]

TypeOK ==
  /\ AllocatorType(cur) /\ AllocatorType(fut) /\ AllocatorType(rel)
  /\ \A k \in 1..Len(pendingAllocs) :
       pendingAllocs[k] \in [inst : INSTANCES, size : 1..HEAP_SIZE,
                             lastSeq : Nat]
  /\ \A k \in 1..Len(pendingReleases) :
       pendingReleases[k] \in [inst : INSTANCES, isReady : BOOLEAN,
                               seq : Nat, defNote : BOOLEAN]
  /\ seqCtr \in Nat
  /\ reqCap \in [INSTANCES -> Nat]
  /\ instState \in [INSTANCES -> Statuses]
  /\ instOffset \in [INSTANCES -> -2..HEAP_SIZE]
  /\ eCreated \in [INSTANCES -> {"UNFIRED", "CLEAN", "POISONED"}]
  /\ allocatedEver \subseteq INSTANCES /\ curFreed \subseteq INSTANCES
  /\ wasDeferred \subseteq INSTANCES
  /\ missingFree \in BOOLEAN /\ structuralAssertFailed \in BOOLEAN
  /\ unblockFailed \in BOOLEAN /\ futMismatch \in BOOLEAN
  /\ poisonReplayBad \in BOOLEAN /\ readyAtRebuild \in BOOLEAN
  /\ dupAlloc \in BOOLEAN
  /\ failedVia \in [INSTANCES -> {"NONE", "INSTANT", "CANCELLED", "RPR"}]
  /\ notifyCount \in [INSTANCES -> Nat]

InitProto ==
  /\ cur = EmptyAlloc /\ fut = EmptyAlloc /\ rel = EmptyAlloc  \* cc:680; fut/rel default-constructed, written before any read (DESIGN 3)
  /\ pendingAllocs = <<>> /\ pendingReleases = <<>>
  /\ seqCtr = 0                                             \* cc:675
  /\ reqCap = [j \in INSTANCES |-> 0]
  /\ instState = [j \in INSTANCES |-> "UNREQUESTED"]
  /\ instOffset = [j \in INSTANCES |-> OFF_NONE]
  /\ eCreated = [j \in INSTANCES |-> "UNFIRED"]
  /\ allocatedEver = {} /\ curFreed = {} /\ wasDeferred = {}
  /\ missingFree = FALSE /\ structuralAssertFailed = FALSE
  /\ unblockFailed = FALSE /\ futMismatch = FALSE
  /\ poisonReplayBad = FALSE /\ readyAtRebuild = FALSE /\ dupAlloc = FALSE
  /\ failedVia = [j \in INSTANCES |-> "NONE"]
  /\ notifyCount = [j \in INSTANCES |-> 0]

=============================================================================
