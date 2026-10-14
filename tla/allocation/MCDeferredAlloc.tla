--------------------------- MODULE MCDeferredAlloc ---------------------------
(***************************************************************************)
(* Model-checking harness for the Realm deferred instance allocation      *)
(* protocol (DeferredAlloc.tla).  Implements DESIGN.md sections 5-7:      *)
(*   - client actions: create/destroy request issuance with               *)
(*     nondeterministic dependency-term choice under contract toggles     *)
(*     C1 (topological sort), C2 (destroy-after-create), C3 (destroy      *)
(*     only after create attempted)                                       *)
(*   - environment actions: precondition firing (the exact DESIGN.md s5   *)
(*     rule: a term fires only when ALL deps have resolved AND its        *)
(*     ballistic user event has fired; poisoned iff >=1 dep poisoned or   *)
(*     the ballistic event was user-poisoned), ballistic event firing,    *)
(*     trigger delivery (TriggerCreate / TriggerDestroy)                  *)
(*   - intrinsic poison is ALWAYS ON (a failed/cancelled create resolves  *)
(*     eCreated(i) poisoned; C2 then propagates into preD[i]).  The       *)
(*     USER_POISON constant gates only client-poisoned ballistic events.  *)
(*   - Init/Next composition, Done self-loop + Quiescent (DESIGN.md s6),  *)
(*     fairness + LIVE_NoStuckAllocs for the Liveness config,             *)
(*     SeqCtrBound state constraint, per-config Size functions.           *)
(*                                                                         *)
(* Quiescence subtlety: a destroy whose precondition fires POISONED is    *)
(* silently cancelled (cc:818-825) or removed (cc:1754); the instance     *)
(* then stays ALLOCATED with its tag in cur forever - a documented,       *)
(* client-caused leak ("POSSIBLE LEAK", ii:87), not a protocol bug.       *)
(* Quiescent treats that as a resolved terminal state, and               *)
(* INV_QuiescentHeapEmpty permits exactly those tags, so deadlock-checked *)
(* configs stay green on legal traces and the BUG-4 detector stays sharp  *)
(* (a BUG-4-stranded tag belongs to a DESTROYED/notified instance and is  *)
(* still flagged).                                                        *)
(***************************************************************************)
EXTENDS DeferredAlloc

CONSTANTS
  USER_POISON,   \* BOOLEAN: client may poison ballistic user events
  C1_ENABLED,    \* BOOLEAN: dep sets restricted to earlier-requested creates
  C2_ENABLED,    \* BOOLEAN: i \in preD[i].deps forced
  C3_ENABLED,    \* BOOLEAN: destroy request only after create attempted
  CLIENT_MODE    \* "FREE" | "SCRIPTED_EVENTLOOP" | "NO_CROSS_DEPS"

ASSUME CLIENT_MODE \in {"FREE", "SCRIPTED_EVENTLOOP", "NO_CROSS_DEPS",
                        "SCRIPTED_COMPOSITE",
                        "SCRIPTED_GCRIPPLE", "SCRIPTED_INVERSION"}
ASSUME USER_POISON \in BOOLEAN /\ C1_ENABLED \in BOOLEAN
       /\ C2_ENABLED \in BOOLEAN /\ C3_ENABLED \in BOOLEAN

(***************************************************************************)
(* Client / event variables (DESIGN.md s5).                               *)
(***************************************************************************)
VARIABLES
  createRequested,  \* [INSTANCES -> BOOLEAN]
  destroyRequested, \* [INSTANCES -> BOOLEAN]
  createWaiter,     \* [INSTANCES -> BOOLEAN]  DeferredCreate registered (cc:715)
  destroyWaiter,    \* [INSTANCES -> BOOLEAN]  DeferredDestroy registered (cc:920)
  preC,             \* [INSTANCES -> Term]     create precondition term
  preD,             \* [INSTANCES -> Term]     destroy precondition term
  balC,             \* [INSTANCES -> BalState] ballistic event of preC[i]
  balD              \* [INSTANCES -> BalState] ballistic event of preD[i]

BalStates == {"NONE", "PENDING", "FIRED", "POISONED"}
NoTerm    == [deps |-> {}, ballistic |-> FALSE]

clientVars == <<createRequested, destroyRequested, createWaiter, destroyWaiter,
                preC, preD, balC, balD>>

mcVars == <<protoVars, clientVars>>

ClientInit ==
  /\ createRequested  = [i \in INSTANCES |-> FALSE]
  /\ destroyRequested = [i \in INSTANCES |-> FALSE]
  /\ createWaiter     = [i \in INSTANCES |-> FALSE]
  /\ destroyWaiter    = [i \in INSTANCES |-> FALSE]
  /\ preC = [i \in INSTANCES |-> NoTerm]
  /\ preD = [i \in INSTANCES |-> NoTerm]
  /\ balC = [i \in INSTANCES |-> "NONE"]
  /\ balD = [i \in INSTANCES |-> "NONE"]

MCInit == InitProto /\ ClientInit

(***************************************************************************)
(* The DESIGN.md s5 firing rule over the protocol's explicit eCreated.    *)
(***************************************************************************)
ECreatedResolved(j) == eCreated[j] # "UNFIRED"
ECreatedPoisoned(j) == eCreated[j] = "POISONED"

\* A term fires only when ALL deps have resolved (clean or poisoned) AND,
\* if ballistic, its user event has fired.  No early-fire on first poison.
TermFired(t, bal) ==
  /\ \A j \in t.deps : ECreatedResolved(j)
  /\ t.ballistic => bal \in {"FIRED", "POISONED"}

\* Poisoned iff at least one dep resolved poisoned, or the ballistic user
\* event was user-poisoned.  Meaningful only when TermFired holds.
TermPoisoned(t, bal) ==
  \/ \E j \in t.deps : ECreatedPoisoned(j)
  \/ (t.ballistic /\ bal = "POISONED")

(***************************************************************************)
(* Client choice sets (contract C1/C2/C3 + CLIENT_MODE).                  *)
(***************************************************************************)
EligibleDeps ==
  IF C1_ENABLED THEN {j \in INSTANCES : createRequested[j]} ELSE INSTANCES

AllowedCTerm(i) ==
  CASE CLIENT_MODE = "SCRIPTED_EVENTLOOP" ->
         \* the s5 worked example: I0=1 immediate, I1=2 ballistic-only
         IF i = 1 THEN {[deps |-> {}, ballistic |-> FALSE]}
                  ELSE {[deps |-> {}, ballistic |-> TRUE]}
    [] CLIENT_MODE = "SCRIPTED_COMPOSITE" ->
         \* Composite4 script (bugs/BUG-6.md item 6): every create is
         \* immediate-precondition; the interesting deferral (i2, i4) comes
         \* from ADA itself, not from create preconditions.
         {[deps |-> {}, ballistic |-> FALSE]}
    [] CLIENT_MODE = "SCRIPTED_GCRIPPLE" ->
         \* GC-ripple (bugs/BUG-1.md, FIX-REVIEW.md trade-off client 1):
         \* I1 fills the heap immediately; I2 (same size) is gated on
         \* ballistic B.  The environment fires B and the destroy's
         \* ballistic D in nondeterministic order.
         IF i = 1 THEN {[deps |-> {}, ballistic |-> FALSE]}
                  ELSE {[deps |-> {}, ballistic |-> TRUE]}
    [] CLIENT_MODE = "SCRIPTED_INVERSION" ->
         \* Trigger inversion (bugs/BUG-1.md A/R*/B): I1 immediate filler;
         \* A=2 and B=3 are both ballistic-gated creates whose triggers can
         \* fire in inverted request order.
         IF i = 1 THEN {[deps |-> {}, ballistic |-> FALSE]}
                  ELSE {[deps |-> {}, ballistic |-> TRUE]}
    [] OTHER ->
         {[deps |-> d, ballistic |-> b] :
            d \in SUBSET (INSTANCES \ {i}), b \in BOOLEAN}

AllowedDTerm(i) ==
  CASE CLIENT_MODE = "SCRIPTED_EVENTLOOP" ->
         IF i = 1 THEN {[deps |-> {1, 2}, ballistic |-> FALSE]}
                  ELSE {[deps |-> {2},    ballistic |-> FALSE]}
    [] CLIENT_MODE = "NO_CROSS_DEPS" ->
         \* excludes the BUG-1 shape: no destroy precondition may wait on
         \* another instance's eCreated
         {[deps |-> d, ballistic |-> b] : d \in SUBSET {i}, b \in BOOLEAN}
    [] CLIENT_MODE = "SCRIPTED_COMPOSITE" ->
         \* Composite4 script: destroy(1) is held open by a ballistic user
         \* event (it is the stranding drain trigger); every other destroy
         \* waits only on its own eCreated (C2-minimal).  destroy(3) and
         \* destroy(5) therefore arrive request-time-TRIGGERED once their
         \* instance is ALLOCATED - the cc:884-887 pushback (i3) and the
         \* stale-rel ARR invocation (i5) respectively.
         IF i = 1 THEN {[deps |-> {1}, ballistic |-> TRUE]}
                  ELSE {[deps |-> {i}, ballistic |-> FALSE]}
    [] CLIENT_MODE = "SCRIPTED_GCRIPPLE" ->
         \* destroy(I1) is gated on ballistic D (its dep {1} is resolved by
         \* then, so D alone controls firing); destroy(I2) is C2-minimal.
         \* DestroyOrderOK forces destroy(1) to be requested AFTER I2's
         \* create - the request lands inside I2's request->trigger window.
         IF i = 1 THEN {[deps |-> {1}, ballistic |-> TRUE]}
                  ELSE {[deps |-> {i}, ballistic |-> FALSE]}
    [] CLIENT_MODE = "SCRIPTED_INVERSION" ->
         \* destroy(I1) = R1 depends on eCreated(A=2) - the cycle edge; the
         \* cleanup destroys are C2-minimal.  CreateOrderOK forces B=3's
         \* create request AFTER destroy(1), so cap(B) >= seq(R1) while
         \* cap(A) < seq(R1): inverted triggers (B before A) then exercise
         \* the monotone-cap guard.
         IF i = 1 THEN {[deps |-> {1, 2}, ballistic |-> FALSE]}
                  ELSE {[deps |-> {i}, ballistic |-> FALSE]}
    [] OTHER ->
         {[deps |-> d, ballistic |-> b] :
            d \in SUBSET INSTANCES, b \in BOOLEAN}

\* Scripted-mode request-order constraints (TRUE in every other mode).
CreateOrderOK(i) ==
  CASE CLIENT_MODE = "SCRIPTED_INVERSION" /\ i = 3 ->
         destroyRequested[1]              \* B requested after destroy(I1)
    [] CLIENT_MODE = "SCRIPTED_GCRIPPLE" /\ i = 2 ->
         \* I1 fills the heap FIRST; without this guard TLC found the
         \* off-script order (I2 first, destroy(I2), then I1 allocates into
         \* the freed heap) violating INV_GCRippleSuccessFunded's intent
         \* (traces/GCRipple-orderguard-misfire.txt).
         createRequested[1]
    [] OTHER -> TRUE

DestroyOrderOK(i) ==
  IF CLIENT_MODE = "SCRIPTED_GCRIPPLE" /\ i = 1
  THEN createRequested[2]                 \* destroy(I1) after I2's create
  ELSE TRUE

(***************************************************************************)
(* Client actions.  Each wraps exactly one protocol action (which updates *)
(* all protocol+ghost variables) with the client-side bookkeeping.        *)
(* Status at request time (has_triggered_faultaware, cc:703 / cc:819):    *)
(* a just-issued ballistic conjunct is PENDING, so any ballistic term is  *)
(* untriggered at request; a deps-only term is triggered iff all deps     *)
(* have already resolved, poisoned iff one resolved poisoned.             *)
(***************************************************************************)
ClientRequestCreate(i) ==
  /\ ~createRequested[i]
  /\ CreateOrderOK(i)
  /\ \E t \in AllowedCTerm(i) :
       /\ t.deps \subseteq EligibleDeps                       \* C1
       /\ LET trig == (\A j \in t.deps : ECreatedResolved(j)) /\ ~t.ballistic
              pois == trig /\ (\E j \in t.deps : ECreatedPoisoned(j))
          IN /\ RequestCreate(i, trig, pois)
             /\ createWaiter' = [createWaiter EXCEPT ![i] = ~trig]  \* cc:715
       /\ preC' = [preC EXCEPT ![i] = t]
       /\ balC' = [balC EXCEPT ![i] = IF t.ballistic THEN "PENDING" ELSE "NONE"]
  /\ createRequested' = [createRequested EXCEPT ![i] = TRUE]
  /\ UNCHANGED <<destroyRequested, destroyWaiter, preD, balD>>

ClientRequestDestroy(i) ==
  /\ createRequested[i]
  /\ ~destroyRequested[i]
  /\ DestroyOrderOK(i)
  /\ C3_ENABLED =>
       instState[i] \notin {"CREATE_PENDING", "CREATE_PENDING_DESTROY"}
  /\ \E t \in AllowedDTerm(i) :
       /\ t.deps \subseteq EligibleDeps                       \* C1
       /\ C2_ENABLED => i \in t.deps                          \* C2
       /\ LET trig == (\A j \in t.deps : ECreatedResolved(j)) /\ ~t.ballistic
              pois == trig /\ (\E j \in t.deps : ECreatedPoisoned(j))
          IN /\ RequestDestroy(i, trig, pois)
             \* waiter registered iff deferred (cc:918-921); note the
             \* cc:846 structural-flag case (triggered destroy of a
             \* CREATE_PENDING instance, C2-off only) registers NO waiter,
             \* matching the release-build fallthrough to cc:915-917.
             /\ destroyWaiter' = [destroyWaiter EXCEPT ![i] = ~trig]
       /\ preD' = [preD EXCEPT ![i] = t]
       /\ balD' = [balD EXCEPT ![i] = IF t.ballistic THEN "PENDING" ELSE "NONE"]
  /\ destroyRequested' = [destroyRequested EXCEPT ![i] = TRUE]
  /\ UNCHANGED <<createRequested, createWaiter, preC, balC>>

(***************************************************************************)
(* Environment actions.                                                    *)
(***************************************************************************)
FireBallisticC(i) ==
  /\ balC[i] = "PENDING"
  /\ \E v \in ({"FIRED"} \cup (IF USER_POISON THEN {"POISONED"} ELSE {})) :
       balC' = [balC EXCEPT ![i] = v]
  /\ UNCHANGED protoVars
  /\ UNCHANGED <<createRequested, destroyRequested, createWaiter,
                 destroyWaiter, preC, preD, balD>>

FireBallisticD(i) ==
  /\ balD[i] = "PENDING"
  /\ \E v \in ({"FIRED"} \cup (IF USER_POISON THEN {"POISONED"} ELSE {})) :
       balD' = [balD EXCEPT ![i] = v]
  /\ UNCHANGED protoVars
  /\ UNCHANGED <<createRequested, destroyRequested, createWaiter,
                 destroyWaiter, preC, preD, balC>>

EnvTriggerCreate(i) ==
  /\ createWaiter[i]
  /\ TermFired(preC[i], balC[i])
  /\ TriggerCreate(i, TermPoisoned(preC[i], balC[i]))
  /\ createWaiter' = [createWaiter EXCEPT ![i] = FALSE]
  /\ UNCHANGED <<createRequested, destroyRequested, destroyWaiter,
                 preC, preD, balC, balD>>

\* Enabled purely on "waiter registered AND term fired": deliberately NOT
\* guarded on create-side state, so that C2-off configs can deliver a
\* destroy trigger while the create is still pending and reach the
\* cc:1630 / cc:1720-1723 / cc:1548-1551 structural asserts (BUG-2 hunt).
EnvTriggerDestroy(i) ==
  /\ destroyWaiter[i]
  /\ TermFired(preD[i], balD[i])
  /\ TriggerDestroy(i, TermPoisoned(preD[i], balD[i]))
  /\ destroyWaiter' = [destroyWaiter EXCEPT ![i] = FALSE]
  /\ UNCHANGED <<createRequested, destroyRequested, createWaiter,
                 preC, preD, balC, balD>>

(***************************************************************************)
(* Quiescence, Done self-loop, Next (DESIGN.md s6).                       *)
(***************************************************************************)
\* A destroy whose precondition fired poisoned was silently cancelled
\* (cc:818-825) or removed (cc:1754-1755): the instance legally stays
\* ALLOCATED, tag in cur, forever ("POSSIBLE LEAK", ii:87).  Terminal.
DestroyResolvedLeak(i) ==
  /\ instState[i] = "ALLOCATED"
  /\ destroyRequested[i]
  /\ ~destroyWaiter[i]
  /\ \A k \in 1..Len(pendingReleases) : pendingReleases[k].inst # i

Quiescent ==
  /\ \A i \in INSTANCES : createRequested[i] /\ destroyRequested[i]
  /\ pendingAllocs   = <<>>
  /\ pendingReleases = <<>>
  /\ \A i \in INSTANCES : ~createWaiter[i] /\ ~destroyWaiter[i]
  /\ \A i \in INSTANCES :
       instState[i] \in {"DESTROYED", "FAILED"} \/ DestroyResolvedLeak(i)
  /\ \A i \in INSTANCES : balC[i] # "PENDING" /\ balD[i] # "PENDING"

\* Clean completion self-loops so that a TLC deadlock report fires exactly
\* on stuck NON-quiescent states.
Done == Quiescent /\ UNCHANGED mcVars

MCNext ==
  \/ \E i \in INSTANCES : ClientRequestCreate(i)
  \/ \E i \in INSTANCES : ClientRequestDestroy(i)
  \/ \E i \in INSTANCES : FireBallisticC(i)
  \/ \E i \in INSTANCES : FireBallisticD(i)
  \/ \E i \in INSTANCES : EnvTriggerCreate(i)
  \/ \E i \in INSTANCES : EnvTriggerDestroy(i)
  \/ Done

Spec == MCInit /\ [][MCNext]_mcVars

(***************************************************************************)
(* Fairness and temporal properties (Liveness config).  WF on the client  *)
(* request actions encodes the full-cleanup client (DESIGN.md s5): without *)
(* it, a trace where the client simply never issues a destroy makes       *)
(* LIVE_NoStuckAllocs fail for a reason that is not a Realm bug.          *)
(***************************************************************************)
Fairness ==
  \A i \in INSTANCES :
    /\ WF_mcVars(ClientRequestCreate(i))
    /\ WF_mcVars(ClientRequestDestroy(i))
    /\ WF_mcVars(FireBallisticC(i))
    /\ WF_mcVars(FireBallisticD(i))
    /\ WF_mcVars(EnvTriggerCreate(i))
    /\ WF_mcVars(EnvTriggerDestroy(i))

LiveSpec == Spec /\ Fairness

LIVE_NoStuckAllocs ==
  \A i \in INSTANCES :
    (instState[i] = "ALLOC_DEFERRED") ~>
      (instState[i] \in {"ALLOCATED", "DESTROYED", "FAILED"})

(***************************************************************************)
(* Harness-side invariants and constraints.                                *)
(***************************************************************************)
\* Backstop for BUG-4-escalated (DESIGN.md s6/s8): at quiescence every tag
\* still in the heap must be a documented poisoned-destroy leak.  A
\* BUG-4-stranded tag belongs to a DESTROYED (already-notified) instance,
\* which DestroyResolvedLeak does not admit, so the detector stays sharp.
INV_QuiescentHeapEmpty ==
  Quiescent => \A t \in DOMAIN cur : DestroyResolvedLeak(t)

\* DESIGN.md s7 backstop; naturally bounded (one release per instance v1).
SeqCtrBound == seqCtr <= 2 * Cardinality(INSTANCES)

(***************************************************************************)
(* Fix-validation intent invariants (v-next; FIX_CAP / FIX_SWEEP are      *)
(* declared in DeferredAlloc.tla).  Mode-guarded: vacuous elsewhere.      *)
(***************************************************************************)
\* GC-ripple under the pure request-time cap: I2's funding set is empty
\* (the destroy of I1 is requested after I2's create), so ADA either
\* succeeds against current (destroy already APPLIED - "completed releases
\* fund via current regardless of cap") or INSTANT-FAILs.  Never deferred.
INV_GCRippleNoDefer ==
  (CLIENT_MODE = "SCRIPTED_GCRIPPLE" /\ FIX_CAP) =>
    instState[2] # "ALLOC_DEFERRED"

\* If I2 succeeded, it was funded by I1's completed release: I1's tag was
\* out of cur when I2 was placed, and (sizes = H each) can never return.
INV_GCRippleSuccessFunded ==
  CLIENT_MODE = "SCRIPTED_GCRIPPLE" =>
    (instState[2] \in {"ALLOCATED", "DESTROYED"} => ~HasTag(cur, 1))

\* Inversion under the cap: A=2 has cap < seq(R1) and, when B=3 is already
\* queued, a lower cap than B - the empty-funding-set test or the monotone-
\* cap guard must INSTANT-FAIL it.  A is never admitted as deferred.
INV_InversionCapped ==
  (CLIENT_MODE = "SCRIPTED_INVERSION" /\ FIX_CAP) =>
    instState[2] # "ALLOC_DEFERRED"

(***************************************************************************)
(* Per-config Size functions (cfg files substitute Size <- SizesX).       *)
(***************************************************************************)
SizesSmoke     == (1 :> 2) @@ (2 :> 2)                                  \* H=3
SizesEventLoop == (1 :> 3) @@ (2 :> 3)                                  \* H=3
SizesLiveness  == (1 :> 2) @@ (2 :> 2) @@ (3 :> 1)                      \* H=4
SizesSafety    == (1 :> 2) @@ (2 :> 1) @@ (3 :> 1) @@ (4 :> 2)          \* H=4
SizesBig       == (1 :> 2) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 1) @@ (5 :> 3) \* H=6
SizesComposite == (1 :> 2) @@ (2 :> 2) @@ (3 :> 1) @@ (4 :> 1) @@ (5 :> 1) \* H=4
SizesGCRipple  == (1 :> 3) @@ (2 :> 3)                                  \* H=3
SizesInversion == (1 :> 3) @@ (2 :> 1) @@ (3 :> 2)                      \* H=3

(***************************************************************************)
(* Hand-simulated BUG-1 trace (SCRIPTED_EVENTLOOP, INSTANCES={1,2}, H=3,  *)
(* Size=SizesEventLoop), verified against the action definitions above:   *)
(*                                                                         *)
(*  1. ClientRequestCreate(1): t=[deps={},bal=F] -> trig, clean ->        *)
(*     RequestCreate(1,T,F): ADA INSTANT_SUCCESS, cur={1@[0,3)},          *)
(*     eCreated[1]=CLEAN, instState[1]=ALLOCATED.                         *)
(*  2. ClientRequestCreate(2): t=[deps={},bal=T] -> balC[2]=PENDING,      *)
(*     untriggered -> RequestCreate(2,F,F): instState[2]=CREATE_PENDING,  *)
(*     createWaiter[2]=T.                                                 *)
(*  3. ClientRequestDestroy(1): t=[deps={1,2},bal=F] (C1 ok: both         *)
(*     requested; C2 ok: 1 in deps).  eCreated[2]=UNFIRED -> untrig ->    *)
(*     RequestDestroy(1,F,F): pendingAllocs empty -> push release         *)
(*     [1,ready=F,seq=1] (cc:858-859); destroyWaiter[1]=T.                *)
(*  4. ClientRequestDestroy(2): t=[deps={2},bal=F] -> untrig; create(2)   *)
(*     still pending -> DELAYEDDESTROY (cc:845-849):                      *)
(*     instState[2]=CREATE_PENDING_DESTROY, destroyWaiter[2]=T; NO        *)
(*     release entry pushed yet.  (full cleanup)                          *)
(*  5. FireBallisticC(2): balC[2]=FIRED -> preC[2] fired clean.           *)
(*  6. EnvTriggerCreate(2): TriggerCreate(2,F), dd=TRUE -> ADA(2,3):      *)
(*     cur full -> rebuild fut = cur - {rel 1} = empty -> fits -> DEF,    *)
(*     pendingAllocs=<<[2,3,lastSeq=1]>>, rel=cur; then cc:1146-1147      *)
(*     pushes [2,ready=F,seq=2] and cc:1150-1153 frees it from fut        *)
(*     (fut ends empty).  instState[2]=ALLOC_DEFERRED, createWaiter[2]=F. *)
(*  7. STUCK: EnvTriggerDestroy(1) needs eCreated[2] resolved;            *)
(*     eCreated[2] fires only on EVENTUAL_SUCCESS of 2, which needs the   *)
(*     release of 1.  EnvTriggerDestroy(2) likewise.  Nothing else        *)
(*     enabled; state is not Quiescent (pendingAllocs # <<>>) -> TLC      *)
(*     reports deadlock.  This is BUG-1.                                  *)
(***************************************************************************)

===============================================================================
