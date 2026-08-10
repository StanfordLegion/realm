-------------------------- MODULE BarrierArrival --------------------------
(***************************************************************************)
(* TLA+ model of the ARRIVAL half of the Realm scalable-barrier protocol    *)
(* described in SCALABLE_BARRIERS_IMPLEMENTATION_PLAN.md.                   *)
(*                                                                         *)
(* Modelled: sections 10 (deterministic radix overlay), 11 (dynamic         *)
(* arrival protocol: local accumulation, DIRTY propagation, COLLECT         *)
(* rounds returning cumulative values with revisions), 12 (learned fast     *)
(* path: per-source quota, per-relay expected child subtotals, silent       *)
(* underfull children), 13 (mismatch detection and pull-based recovery),    *)
(* 16.1 (freeze-on-first-use: a generation's mode is fixed when it starts), *)
(* plus the EAGER FLUSH extension.                                          *)
(*                                                                         *)
(* Deliberately NOT modelled: the adjustment ledger / Lamport stamps /      *)
(* causal DAG (section 15), the subscription and trigger-notification tree  *)
(* (section 14), multicast encoding (section 7), plan learning/commit       *)
(* traffic (the learned plan is supplied as a constant, which is exactly    *)
(* the plan generation 0 would have learned), node failure (the plan        *)
(* assumes reliable nodes and reliable transport).                          *)
(*                                                                         *)
(* See README.md for results, constants and the counterexample traces.      *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS
    N,              (* number of nodes                                      *)
    R,              (* radix of the routing overlay                         *)
    RootId,         (* node id of the logical root                          *)
    G,              (* number of generations                                *)
    Arr,            (* [Gens -> [Nodes -> Nat]] arrivals the app will issue  *)
    Expected,       (* [Gens -> Nat] persistent expected count               *)
    PlannedGens,    (* SUBSET Gens: generations frozen onto the fast path    *)
    Quota,          (* [Nodes -> Nat] predicted local arrival count          *)
    PlannedKids,    (* [Nodes -> SUBSET Nodes] predicted children            *)
    PTotal,         (* [Nodes -> Nat] predicted subtree total                *)
    EagerFlush,     (* BOOLEAN: enable the eager-flush extension             *)
    MaxCopies,      (* per-message cap in the network bag (>= 1)             *)
    MaxDups,        (* transport duplication budget                          *)
    MaxRev          (* revision ceiling (a saturation detector, not a rule)  *)

----------------------------------------------------------------------------
(*                          Small helpers                                  *)
----------------------------------------------------------------------------
Min2(a, b) == IF a < b THEN a ELSE b

(* Sum of f[x] over x in S.  S is always a small finite set here. *)
SumF(S, f) ==
    LET RECURSIVE H(_)
        H(T) == IF T = {} THEN 0
                ELSE LET x == CHOOSE y \in T : TRUE
                     IN f[x] + H(T \ {x})
    IN H(S)

----------------------------------------------------------------------------
(*             Section 10: deterministic owner-relative radix tree          *)
(*                                                                         *)
(* All protocol logic below is written in RELATIVE index space, where       *)
(* index 0 is the logical root.  NodeIdOf / RelOf are the plan's mapping    *)
(* back to absolute node ids; they are only exercised by an ASSUME.         *)
----------------------------------------------------------------------------
Nodes == 0 .. N - 1
Gens  == 0 .. G - 1

NodeIdOf(i) == (i + RootId) % N
RelOf(x)    == (x - RootId + N) % N

Parent(i) == (i - 1) \div R
Kids(i)   == { i * R + k : k \in 1 .. R } \cap Nodes

RECURSIVE SubtreeOf(_)
SubtreeOf(n) == {n} \cup UNION { SubtreeOf(c) : c \in Kids(n) }

(* A node participates in the learned plan iff it is the root or its
   parent predicted it.  Nodes with no plan are dynamic (section 12/13). *)
InPlan(n)      == (n = 0) \/ (n \in PlannedKids[Parent(n)])
HasPlan(g, n)  == (g \in PlannedGens) /\ InPlan(n)

ASSUME N \in Nat /\ N >= 2
ASSUME R \in Nat /\ R >= 1
ASSUME G \in Nat /\ G >= 1
ASSUME RootId \in 0 .. N - 1
ASSUME MaxCopies >= 1 /\ MaxDups >= 0 /\ MaxRev >= 1
ASSUME EagerFlush \in BOOLEAN
ASSUME PlannedGens \subseteq Gens
(* the relative/absolute mapping is a bijection *)
ASSUME \A i \in Nodes : RelOf(NodeIdOf(i)) = i
(* the overlay is a tree with bounded degree, rooted at relative index 0 *)
ASSUME \A i \in Nodes \ {0} : Parent(i) \in Nodes /\ i \in Kids(Parent(i))
ASSUME \A i \in Nodes : Cardinality(Kids(i)) <= R
(* the learned plan is a subtree of the overlay and its subtree totals add up *)
ASSUME \A n \in Nodes : PlannedKids[n] \subseteq Kids(n)
ASSUME \A n \in Nodes : PTotal[n] = Quota[n] + SumF(PlannedKids[n], PTotal)
(* the application supplies exactly the expected count in every generation *)
ASSUME \A g \in Gens : SumF(Nodes, Arr[g]) = Expected[g]

----------------------------------------------------------------------------
(*                            Messages                                     *)
(*                                                                         *)
(* Section 17.1.  All five message kinds share one record shape so that     *)
(* the network bag has a uniform element type.                             *)
(*   REP     child -> parent : cumulative value + revision (+ planned flag) *)
(*   MM      child -> parent : plan mismatch signal (section 13.1)          *)
(*   DIRTY   child -> parent : clean-to-dirty edge (section 11.1/11.2)      *)
(*   COLLECT parent -> child : collection request (section 11.3 / 13.3)     *)
(*   FLUSH   parent -> child : eager-flush mode propagation (extension)     *)
----------------------------------------------------------------------------
Mk(k, g, s, d, rv, v, p) ==
    [kind |-> k, gen |-> g, src |-> s, dst |-> d, rev |-> rv, val |-> v, pc |-> p]

Rep(g, s, rv, v, p) == Mk("REP",     g, s, Parent(s), rv, v, p)
MMsg(g, s)          == Mk("MM",      g, s, Parent(s),  0, 0, FALSE)
Dty(g, s)           == Mk("DIRTY",   g, s, Parent(s),  0, 0, FALSE)
Col(g, s, d)        == Mk("COLLECT", g, s, d,          0, 0, FALSE)
Flu(g, s, d)        == Mk("FLUSH",   g, s, d,          0, 0, FALSE)

EmptyBag == [x \in {} |-> 0]
Cnt(B, m)     == IF m \in DOMAIN B THEN B[m] ELSE 0
AddSet(B, S)  == [ m \in (DOMAIN B) \cup S |->
                     Min2(MaxCopies, Cnt(B, m) + (IF m \in S THEN 1 ELSE 0)) ]
Del(B, m)     == IF Cnt(B, m) <= 1
                 THEN [ x \in (DOMAIN B) \ {m} |-> B[x] ]
                 ELSE [ B EXCEPT ![m] = @ - 1 ]

----------------------------------------------------------------------------
(*                            State                                        *)
----------------------------------------------------------------------------
VARIABLES
    pending,         (* [g][n] arrivals the application has not issued yet   *)
    local,           (* [g][n] locally accumulated arrivals (section 11.1)   *)
    mode,            (* [g][n] in {"DYN","PLAN","REC","FLUSH"}               *)
    acc,             (* [g][n][c] highest accepted [rev,val] from child c    *)
    sentRev,         (* [g][n] revision of the last report sent upward       *)
    sentVal,         (* [g][n] cumulative value of the last report sent up   *)
    dirtySent,       (* [g][n] a DIRTY is outstanding for this dirty episode *)
    dirtyKids,       (* [g][n] children known dirty and not yet collected    *)
    awaiting,        (* [g][n] children this node's collect wave waits for   *)
    owedReply,       (* [g][n] this node owes its parent a collect reply     *)
    mmSeen,          (* [g][n] this node has already forwarded a mismatch    *)
    collectPending,  (* [g] root has an unserviced dirty/mismatch signal     *)
    triggered,       (* SUBSET Gens                                          *)
    trigCount,       (* [g] number of times g triggered (must stay <= 1)     *)
    trigTotal,       (* [g] root total observed at trigger time, -1 if none  *)
    rootMsgs,        (* [g] arrival-protocol messages SENT to the root, i.e.
                        the root's protocol fan-in (saturating at R+1; a
                        transport duplicate is not a protocol message)       *)
    fatal,           (* equal revision with conflicting content was seen     *)
    msgs,            (* network bag: message -> number of copies in flight   *)
    dups             (* transport duplications used so far                   *)

vars == << pending, local, mode, acc, sentRev, sentVal, dirtySent, dirtyKids,
           awaiting, owedReply, mmSeen, collectPending, triggered, trigCount,
           trigTotal, rootMsgs, fatal, msgs, dups >>

NoRep == [rev |-> -1, val |-> 0]

(* cumulative subtree value currently held by n for generation g *)
Subtot(g, n) ==
    local[g][n] + SumF(Kids(n), [c \in Kids(n) |-> acc[g][n][c].val])

(* arrivals the application has actually issued inside n's subtree so far *)
IssuedIn(g, n) ==
    SumF(SubtreeOf(n), [x \in SubtreeOf(n) |-> Arr[g][x] - pending[g][x]])

TypeOK ==
    /\ pending  \in [Gens -> [Nodes -> Nat]]
    /\ local    \in [Gens -> [Nodes -> Nat]]
    /\ mode     \in [Gens -> [Nodes -> {"DYN", "PLAN", "REC", "FLUSH"}]]
    /\ \A g \in Gens, n \in Nodes :
          /\ DOMAIN acc[g][n] = Kids(n)
          /\ \A c \in Kids(n) : acc[g][n][c].rev \in (-1) .. MaxRev
                                /\ acc[g][n][c].val \in Nat
    /\ sentRev  \in [Gens -> [Nodes -> (-1) .. MaxRev]]
    /\ sentVal  \in [Gens -> [Nodes -> Nat]]
    /\ dirtySent \in [Gens -> [Nodes -> BOOLEAN]]
    /\ \A g \in Gens, n \in Nodes : dirtyKids[g][n] \subseteq Kids(n)
                                    /\ awaiting[g][n] \subseteq Kids(n)
    /\ owedReply \in [Gens -> [Nodes -> BOOLEAN]]
    /\ mmSeen    \in [Gens -> [Nodes -> BOOLEAN]]
    /\ collectPending \in [Gens -> BOOLEAN]
    /\ triggered \subseteq Gens
    /\ trigCount \in [Gens -> Nat]
    /\ fatal \in BOOLEAN
    /\ dups \in 0 .. MaxDups
    /\ \A m \in DOMAIN msgs :
          /\ m.kind \in {"REP", "MM", "DIRTY", "COLLECT", "FLUSH"}
          /\ m.gen \in Gens /\ m.src \in Nodes /\ m.dst \in Nodes
          /\ msgs[m] \in 1 .. MaxCopies

Init ==
    /\ pending  = [g \in Gens |-> [n \in Nodes |-> Arr[g][n]]]
    /\ local    = [g \in Gens |-> [n \in Nodes |-> 0]]
    /\ mode     = [g \in Gens |-> [n \in Nodes |->
                      IF HasPlan(g, n) THEN "PLAN" ELSE "DYN"]]
    /\ acc      = [g \in Gens |-> [n \in Nodes |-> [c \in Kids(n) |-> NoRep]]]
    /\ sentRev  = [g \in Gens |-> [n \in Nodes |-> -1]]
    /\ sentVal  = [g \in Gens |-> [n \in Nodes |-> 0]]
    /\ dirtySent = [g \in Gens |-> [n \in Nodes |-> FALSE]]
    /\ dirtyKids = [g \in Gens |-> [n \in Nodes |-> {}]]
    /\ awaiting  = [g \in Gens |-> [n \in Nodes |-> {}]]
    /\ owedReply = [g \in Gens |-> [n \in Nodes |-> FALSE]]
    /\ mmSeen    = [g \in Gens |-> [n \in Nodes |-> FALSE]]
    /\ collectPending = [g \in Gens |-> FALSE]
    /\ triggered = {}
    /\ trigCount = [g \in Gens |-> 0]
    /\ trigTotal = [g \in Gens |-> -1]
    /\ rootMsgs  = [g \in Gens |-> 0]
    /\ fatal     = FALSE
    /\ msgs      = EmptyBag
    /\ dups      = 0

----------------------------------------------------------------------------
(*                        Predicates over plan state                       *)
----------------------------------------------------------------------------
(* Section 12.2: every expected child produced exactly its predicted
   subtotal, the local quota is met, and no unexpected child reported. *)
PlannedSat(g, n) ==
    /\ HasPlan(g, n)
    /\ local[g][n] = Quota[n]
    /\ \A c \in PlannedKids[n] : acc[g][n][c].val = PTotal[c]
    /\ \A c \in Kids(n) \ PlannedKids[n] : acc[g][n][c].rev < 0

(* Section 13.1: detectable deviations. *)
MismatchCond(g, n) ==
    /\ mode[g][n] = "PLAN"
    /\ \/ local[g][n] > Quota[n]                          (* source over quota *)
       \/ \E c \in Kids(n) \ PlannedKids[n] :             (* unexpected child  *)
             acc[g][n][c].rev >= 0 \/ c \in dirtyKids[g][n]
       \/ \E c \in PlannedKids[n] : acc[g][n][c].val > PTotal[c]  (* excess *)

(* Eager flush propagates down the plan tree and down any dynamic branch
   the node already knows about (section 13.3: "old active branches plus
   new dynamic branches, not every machine node"). *)
FlushTargets(g, n) == PlannedKids[n] \cup dirtyKids[g][n]

RecMode == IF EagerFlush THEN "FLUSH" ELSE "REC"

(* Root fan-in accounting (sections 6.6 and 23): count the arrival-protocol
   messages a node sends to the logical root, at send time.  Counting at
   delivery time would also count transport duplicates, which are not
   protocol messages.
   Only generations on the learned fast path are counted; the bound in 6.6
   is claimed for a stable plan only, and not tracking the dynamic
   generations keeps the state space smaller. *)
SendsToRoot(n) == (n # 0) /\ (Parent(n) = 0)
BumpSend(g, n, k) ==
    IF SendsToRoot(n) /\ k /\ (g \in PlannedGens)
    THEN [rootMsgs EXCEPT ![g] = Min2(@ + 1, R + 1)]
    ELSE rootMsgs

----------------------------------------------------------------------------
(*                             Local actions                               *)
----------------------------------------------------------------------------
(* Section 11.1: arrive() accumulates locally, always. *)
Arrive(n, g) ==
    /\ pending[g][n] > 0
    /\ pending' = [pending EXCEPT ![g][n] = @ - 1]
    /\ local'   = [local   EXCEPT ![g][n] = @ + 1]
    /\ UNCHANGED << mode, acc, sentRev, sentVal, dirtySent, dirtyKids,
                    awaiting, owedReply, mmSeen, collectPending, triggered,
                    trigCount, trigTotal, rootMsgs, fatal, msgs, dups >>

(* Section 13.1/13.3: a node notices a deviation from the plan, leaves the
   fast path, signals its parent, and -- with eager flush -- immediately
   propagates FLUSH down its planned branches. *)
EnterRecovery(n, g) ==
    /\ MismatchCond(g, n)
    /\ mode'   = [mode   EXCEPT ![g][n] = RecMode]
    /\ mmSeen' = [mmSeen EXCEPT ![g][n] = TRUE]
    /\ msgs'   = AddSet(msgs,
                    (IF n # 0 /\ ~mmSeen[g][n] THEN {MMsg(g, n)} ELSE {})
                    \cup (IF EagerFlush
                          THEN { Flu(g, n, c) : c \in FlushTargets(g, n) }
                          ELSE {}))
    /\ collectPending' = IF n = 0
                         THEN [collectPending EXCEPT ![g] = TRUE]
                         ELSE collectPending
    /\ rootMsgs' = BumpSend(g, n, ~mmSeen[g][n])
    /\ UNCHANGED << pending, local, acc, sentRev, sentVal, dirtySent,
                    dirtyKids, awaiting, owedReply, triggered, trigCount,
                    trigTotal, fatal, dups >>

(* Section 12.1/12.2 planned completion, and the push behaviour of a node
   that has left the fast path (recovering node announces its overflow;
   flushing node reports every arrival immediately). *)
PushUp(n, g) ==
    /\ n # 0
    /\ Subtot(g, n) > sentVal[g][n]
    /\ sentRev[g][n] < MaxRev
    /\ \/ (mode[g][n] = "PLAN" /\ PlannedSat(g, n))
       \/ mode[g][n] \in {"REC", "FLUSH"}
    /\ msgs' = AddSet(msgs, { Rep(g, n, sentRev[g][n] + 1, Subtot(g, n),
                                  mode[g][n] = "PLAN") })
    /\ sentRev' = [sentRev EXCEPT ![g][n] = @ + 1]
    /\ sentVal' = [sentVal EXCEPT ![g][n] = Subtot(g, n)]
    /\ dirtySent' = [dirtySent EXCEPT ![g][n] = FALSE]
    /\ rootMsgs' = BumpSend(g, n, TRUE)
    /\ UNCHANGED << pending, local, mode, acc, dirtyKids, awaiting,
                    owedReply, mmSeen, collectPending, triggered, trigCount,
                    trigTotal, fatal, dups >>

(* Section 11.1/11.2: one DIRTY per clean-to-dirty transition.  Only a
   node running the DYNAMIC protocol does this.  A node on the learned
   fast path stays silent until it reaches its quota (section 12.2). *)
SendDirty(n, g) ==
    /\ n # 0
    /\ mode[g][n] = "DYN"
    /\ ~dirtySent[g][n]
    /\ (Subtot(g, n) > sentVal[g][n]) \/ (dirtyKids[g][n] # {})
    /\ msgs' = AddSet(msgs, {Dty(g, n)})
    /\ dirtySent' = [dirtySent EXCEPT ![g][n] = TRUE]
    /\ rootMsgs' = BumpSend(g, n, TRUE)
    /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtyKids,
                    awaiting, owedReply, mmSeen, collectPending, triggered,
                    trigCount, trigTotal, fatal, dups >>

(* Section 11.3 step 3: a relay replies only after every child in its
   snapshot has replied.  The reply is cumulative and carries a fresh
   revision; it is sent even when the value did not change, otherwise the
   collection wave could never complete. *)
CompleteCollect(n, g) ==
    /\ n # 0
    /\ owedReply[g][n]
    /\ awaiting[g][n] = {}
    /\ sentRev[g][n] < MaxRev
    /\ msgs' = AddSet(msgs, { Rep(g, n, sentRev[g][n] + 1, Subtot(g, n), FALSE) })
    /\ sentRev' = [sentRev EXCEPT ![g][n] = @ + 1]
    /\ sentVal' = [sentVal EXCEPT ![g][n] = Subtot(g, n)]
    /\ owedReply' = [owedReply EXCEPT ![g][n] = FALSE]
    /\ dirtySent' = [dirtySent EXCEPT ![g][n] = FALSE]
    /\ rootMsgs' = BumpSend(g, n, TRUE)
    /\ UNCHANGED << pending, local, mode, acc, dirtyKids, awaiting, mmSeen,
                    collectPending, triggered, trigCount, trigTotal,
                    fatal, dups >>

(* Section 11.3: the root schedules a collection pass after a dirty
   notification, and section 13.3 step 4: on mismatch it collects every
   incomplete planned branch and every dynamic dirty branch.
   NOTE: the root does NOT spin collection rounds -- section 11.3 says it
   waits for another clean-to-dirty transition. *)
LaunchCollect(g) ==
    /\ collectPending[g]
    /\ g \notin triggered
    /\ LET tg == { c \in Kids(0) :
                     \/ c \in dirtyKids[g][0]
                     \/ /\ HasPlan(g, 0)
                        /\ mode[g][0] # "PLAN"
                        /\ c \in PlannedKids[0]
                        /\ acc[g][0][c].val < PTotal[c] }
       IN /\ tg # {}
          /\ msgs' = AddSet(msgs, { Col(g, 0, c) : c \in tg })
          /\ dirtyKids' = [dirtyKids EXCEPT ![g][0] = @ \ tg]
    /\ collectPending' = [collectPending EXCEPT ![g] = FALSE]
    /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                    awaiting, owedReply, mmSeen, triggered, trigCount,
                    trigTotal, rootMsgs, fatal, dups >>

(* Section 6.1: trigger exactly once, on an exact count, in order.  The
   >= test (rather than =) is deliberate: it lets TriggerCorrect actually
   detect a double-counted total instead of silently disabling the step. *)
Trigger(g) ==
    /\ g \notin triggered
    /\ (g = 0) \/ ((g - 1) \in triggered)
    /\ Subtot(g, 0) >= Expected[g]
    /\ triggered' = triggered \cup {g}
    /\ trigCount' = [trigCount EXCEPT ![g] = @ + 1]
    /\ trigTotal' = [trigTotal EXCEPT ![g] = Subtot(g, 0)]
    /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                    dirtyKids, awaiting, owedReply, mmSeen, collectPending,
                    rootMsgs, fatal, msgs, dups >>

----------------------------------------------------------------------------
(*                         Message handlers                                *)
----------------------------------------------------------------------------
Consume(m, S) == AddSet(Del(msgs, m), S)

(* Section 6.2: keep the highest accepted revision.  A higher revision
   REPLACES; an equal revision with different content is fatal; a lower
   revision is stale and ignored. *)
RcvRep(m) ==
    LET n == m.dst
        g == m.gen
        c == m.src
        cur == acc[g][n][c]
        newer == m.rev > cur.rev
    IN /\ c \in Kids(n)
       /\ acc' = IF newer
                 THEN [acc EXCEPT ![g][n][c] = [rev |-> m.rev, val |-> m.val]]
                 ELSE acc
       /\ fatal' = fatal \/ (m.rev = cur.rev /\ m.val # cur.val)
       (* the root owes nobody a reply, so it keeps no wave state *)
       /\ awaiting' = IF newer /\ n # 0
                      THEN [awaiting EXCEPT ![g][n] = @ \ {c}]
                      ELSE awaiting
       /\ msgs' = Del(msgs, m)
       /\ UNCHANGED << pending, local, mode, sentRev, sentVal, dirtySent,
                       dirtyKids, owedReply, mmSeen, collectPending,
                       triggered, trigCount, trigTotal, rootMsgs, dups >>

RcvMM(m) ==
    LET n == m.dst
        g == m.gen
        c == m.src
    IN /\ mode' = [mode EXCEPT ![g][n] =
                     IF EagerFlush THEN "FLUSH"
                     ELSE IF mode[g][n] = "PLAN" THEN "REC"
                     ELSE mode[g][n]]
       /\ mmSeen' = [mmSeen EXCEPT ![g][n] = TRUE]
       /\ msgs' = Consume(m,
                    (IF n # 0 /\ ~mmSeen[g][n] THEN {MMsg(g, n)} ELSE {})
                    \cup (IF EagerFlush /\ mode[g][n] # "FLUSH"
                          THEN { Flu(g, n, x) : x \in FlushTargets(g, n) \ {c} }
                          ELSE {}))
       /\ collectPending' = IF n = 0
                            THEN [collectPending EXCEPT ![g] = TRUE]
                            ELSE collectPending
       /\ rootMsgs' = BumpSend(g, n, ~mmSeen[g][n])
       /\ UNCHANGED << pending, local, acc, sentRev, sentVal, dirtySent,
                       dirtyKids, awaiting, owedReply, triggered, trigCount,
                       trigTotal, fatal, dups >>

RcvDirty(m) ==
    LET n == m.dst
        g == m.gen
        c == m.src
    IN /\ c \in Kids(n)
       /\ dirtyKids' = [dirtyKids EXCEPT ![g][n] = @ \cup {c}]
       /\ collectPending' = IF n = 0
                            THEN [collectPending EXCEPT ![g] = TRUE]
                            ELSE collectPending
       /\ msgs' = Del(msgs, m)
       /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                       awaiting, owedReply, mmSeen, triggered, trigCount,
                       trigTotal, rootMsgs, fatal, dups >>

(* Section 11.3 step 2 / 13.3 step 4: snapshot the branches that still owe
   something and recurse into them; the reply is owed until they answer. *)
RcvCollect(m) ==
    LET n == m.dst
        g == m.gen
        tg == { c \in Kids(n) :
                  \/ c \in dirtyKids[g][n]
                  \/ /\ HasPlan(g, n)
                     /\ c \in PlannedKids[n]
                     /\ acc[g][n][c].val < PTotal[c] }
    IN /\ n # 0
       /\ awaiting'  = [awaiting  EXCEPT ![g][n] = tg]
       /\ owedReply' = [owedReply EXCEPT ![g][n] = TRUE]
       /\ dirtyKids' = [dirtyKids EXCEPT ![g][n] = @ \ tg]
       /\ msgs' = Consume(m, { Col(g, n, c) : c \in tg })
       /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                       mmSeen, collectPending, triggered, trigCount,
                       trigTotal, rootMsgs, fatal, dups >>

(* EAGER FLUSH extension: switch this generation to flushing mode, which
   makes the node self-announcing from now on, and pass the mode further
   down the plan tree. *)
RcvFlush(m) ==
    LET n == m.dst
        g == m.gen
    IN /\ mode[g][n] # "FLUSH"
       /\ mode' = [mode EXCEPT ![g][n] = "FLUSH"]
       /\ msgs' = Consume(m, { Flu(g, n, c) : c \in FlushTargets(g, n) })
       /\ UNCHANGED << pending, local, acc, sentRev, sentVal, dirtySent,
                       dirtyKids, awaiting, owedReply, mmSeen, collectPending,
                       triggered, trigCount, trigTotal, rootMsgs, fatal, dups >>

RcvFlushDone(m) ==   (* already flushing: just consume the redundant copy *)
    /\ mode[m.gen][m.dst] = "FLUSH"
    /\ msgs' = Del(msgs, m)
    /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                    dirtyKids, awaiting, owedReply, mmSeen, collectPending,
                    triggered, trigCount, trigTotal, rootMsgs, fatal, dups >>

Deliver(m) ==
    /\ m \in DOMAIN msgs
    /\ CASE m.kind = "REP"     -> RcvRep(m)
         [] m.kind = "MM"      -> RcvMM(m)
         [] m.kind = "DIRTY"   -> RcvDirty(m)
         [] m.kind = "COLLECT" -> RcvCollect(m)
         [] m.kind = "FLUSH"   -> RcvFlush(m) \/ RcvFlushDone(m)

(* The transport is reliable but may duplicate (section 2 requires the
   protocol to tolerate overlapping reports regardless).  Duplication is
   an unfair action: liveness may never depend on it. *)
DupMsg ==
    /\ dups < MaxDups
    /\ \E m \in DOMAIN msgs :
          /\ Cnt(msgs, m) < MaxCopies
          /\ msgs' = AddSet(msgs, {m})
    /\ dups' = dups + 1
    /\ UNCHANGED << pending, local, mode, acc, sentRev, sentVal, dirtySent,
                    dirtyKids, awaiting, owedReply, mmSeen, collectPending,
                    triggered, trigCount, trigTotal, rootMsgs, fatal >>

----------------------------------------------------------------------------
(*                          Next-state relation                            *)
----------------------------------------------------------------------------
LocalStep(n, g) ==
    \/ Arrive(n, g)
    \/ EnterRecovery(n, g)
    \/ PushUp(n, g)
    \/ SendDirty(n, g)
    \/ CompleteCollect(n, g)

DeliverFrom(s) == \E m \in DOMAIN msgs : m.src = s /\ Deliver(m)

Next ==
    \/ \E n \in Nodes, g \in Gens : LocalStep(n, g)
    \/ \E g \in Gens : LaunchCollect(g) \/ Trigger(g)
    \/ \E s \in Nodes : DeliverFrom(s)
    \/ DupMsg

----------------------------------------------------------------------------
(*                        Fairness assumptions                             *)
(*                                                                         *)
(* These are the whole point of the exercise: the plan states no liveness   *)
(* theory, so the assumptions under which the barrier makes progress have   *)
(* to be written down here.                                                *)
(*                                                                         *)
(* FairLocal   -- every node eventually takes each of its enabled local     *)
(*                steps for each generation: it eventually issues a         *)
(*                pending arrive(), eventually notices a plan mismatch,     *)
(*                eventually sends a report it is willing to send,          *)
(*                eventually emits its clean-to-dirty DIRTY, and eventually *)
(*                completes a collection wave whose children have answered. *)
(*                (This is one WF per (node, generation) over the           *)
(*                disjunction; sound because every local step of a fixed    *)
(*                (n,g) can only occur finitely often, so a permanently     *)
(*                enabled step cannot be starved by the others forever.)    *)
(* FairDeliver -- weak fairness on message delivery, per sender.            *)
(* FairCollect -- weak fairness on the root's collection scheduling.        *)
(* FairTrigger -- weak fairness on the root publishing a trigger.           *)
(*                                                                         *)
(* Deliberately NOT assumed: fairness of DupMsg (duplication must never be  *)
(* load-bearing), and any form of timeout / polling / re-collection that    *)
(* is not caused by a protocol signal (section 11.3 forbids spinning        *)
(* collection rounds).                                                      *)
----------------------------------------------------------------------------
FairLocal   == \A n \in Nodes, g \in Gens : WF_vars(LocalStep(n, g))
FairDeliver == \A s \in Nodes : WF_vars(DeliverFrom(s))
FairCollect == \A g \in Gens : WF_vars(LaunchCollect(g))
FairTrigger == \A g \in Gens : WF_vars(Trigger(g))

Fairness == FairLocal /\ FairDeliver /\ FairCollect /\ FairTrigger

Spec == Init /\ [][Next]_vars /\ Fairness

----------------------------------------------------------------------------
(*                        Safety invariants (section 6)                    *)
----------------------------------------------------------------------------
(* 6.1 (1): a generation triggers at most once. *)
TriggerOnce == \A g \in Gens : trigCount[g] <= 1

(* 6.1 (1)+(2): it triggers only on the exact expected count. *)
TriggerCorrect == \A g \in triggered : trigTotal[g] = Expected[g]

(* 6.1 (4): generations trigger in order. *)
TriggerInOrder == \A g \in triggered : (g = 0) \/ ((g - 1) \in triggered)

(* 6.2: the receiver stores a value that some sender actually sent, never a
   sum of two reports; revisions never run ahead of the sender; an equal
   revision with different content is fatal. *)
NoDoubleCount ==
    /\ ~fatal
    /\ \A g \in Gens, n \in Nodes : \A c \in Kids(n) :
          /\ acc[g][n][c].rev <= sentRev[g][c]
          /\ acc[g][n][c].val <= sentVal[g][c]

(* 6.2 / 6.3: no arrival is lost and none is counted twice.  Every node's
   cumulative subtree value is bounded by the arrivals the application has
   actually issued in that subtree. *)
NoLostArrival ==
    \A g \in Gens, n \in Nodes :
        /\ Subtot(g, n) <= IssuedIn(g, n)
        /\ sentVal[g][n] <= IssuedIn(g, n)

(* 6.6 / 23: bounded root fan-in.
   (a) the overlay itself never gives any node more than R children;
   (b) at most R distinct children ever report to the root;
   (c) "For a STABLE learned plan, the logical root receives at most the
       configured radix's worth of arrival-plan completions" -- so the
       message bound is claimed only for a generation on the learned fast
       path in which nothing has yet deviated from the plan.  Section 6.6
       explicitly exempts bootstrap and recovery from the per-generation
       bound (they only have to keep each individual ROUND bounded, which
       (a) gives us since the root only ever talks to Kids(0)). *)
BoundedRootFanIn ==
    /\ \A n \in Nodes : Cardinality(Kids(n)) <= R
    /\ \A g \in Gens :
          Cardinality({c \in Kids(0) : acc[g][0][c].rev >= 0}) <= R
    /\ \A g \in Gens :
          ((g \in PlannedGens) /\ (\A n \in Nodes : ~mmSeen[g][n]))
             => (rootMsgs[g] <= R)

(* Saturation detector: if this ever fails the MaxRev ceiling truncated
   behaviours and the liveness result would be unsound. *)
RevNotSaturated == \A g \in Gens, n \in Nodes : sentRev[g][n] < MaxRev

Safety ==
    /\ TypeOK
    /\ TriggerOnce
    /\ TriggerCorrect
    /\ TriggerInOrder
    /\ NoDoubleCount
    /\ NoLostArrival
    /\ BoundedRootFanIn
    /\ RevNotSaturated

(* 6.2, as an action property: a higher revision replaces and never lowers
   the accepted value; a lower revision is ignored. *)
MonotoneAccept ==
    [][ \A g \in Gens, n \in Nodes : \A c \in Kids(n) :
           /\ acc'[g][n][c].val >= acc[g][n][c].val
           /\ acc'[g][n][c].rev >= acc[g][n][c].rev ]_vars

----------------------------------------------------------------------------
(*                              Liveness                                   *)
----------------------------------------------------------------------------
AllIssued    == \A g \in Gens, n \in Nodes : pending[g][n] = 0
AllTriggered == triggered = Gens

EventuallyTriggers == AllIssued ~> AllTriggered

(* A terminal state in which some generation has not triggered.  Nothing is
   enabled anywhere: every arrive() has been issued, the network is empty and
   no node has any work left, yet the barrier has not fired.  Checking
   NotStuck as an ordinary invariant produces a much shorter and more
   readable counterexample than the liveness lasso does. *)
Stuck    == (~AllTriggered) /\ (~ENABLED Next)
NotStuck == ~Stuck

----------------------------------------------------------------------------
(* Trace alias.  NOT referenced by the .cfg files: the tla2tools 2.19 build
   used for the recorded results does not accept the ALIAS keyword in a
   configuration file, so traces are printed with the default (full state)
   format.  Add "ALIAS Alias" to a .cfg if your tla2tools supports it.     *)
----------------------------------------------------------------------------
Alias ==
    [ mode      |-> mode,
      local     |-> local,
      pend      |-> pending,
      subtotal  |-> [g \in Gens |-> [n \in Nodes |-> Subtot(g, n)]],
      accepted  |-> [g \in Gens |-> [n \in Nodes |->
                        [c \in Kids(n) |-> acc[g][n][c]]]],
      sent      |-> [g \in Gens |-> [n \in Nodes |->
                        <<sentRev[g][n], sentVal[g][n]>>]],
      dirtyK    |-> dirtyKids,
      dirtyS    |-> dirtySent,
      await     |-> awaiting,
      owed      |-> owedReply,
      collPend  |-> collectPending,
      trig      |-> triggered,
      inflight  |-> DOMAIN msgs ]

============================================================================
