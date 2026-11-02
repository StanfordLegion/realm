\* Copyright 2026 Stanford University, NVIDIA Corporation
\* SPDX-License-Identifier: Apache-2.0
\*
\* Licensed under the Apache License, Version 2.0 (the "License");
\* you may not use this file except in compliance with the License.
\* You may obtain a copy of the License at
\*
\*     http://www.apache.org/licenses/LICENSE-2.0
\*
\* Unless required by applicable law or agreed to in writing, software
\* distributed under the License is distributed on an "AS IS" BASIS,
\* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
\* See the License for the specific language governing permissions and
\* limitations under the License.

------------------------------ MODULE BarrierArrive ------------------------------
(***************************************************************************)
(* Realm scalable barriers - arrival protocol, generalised to a SEQUENCE of *)
(* arrival plans so that several pattern changes can be in flight at once.  *)
(*                                                                         *)
(* BarrierV2 checked a single pattern change.  Everything it verified holds *)
(* here; this module adds epoch bookkeeping so a node can be two plans      *)
(* behind, which is the case pipelining actually produces.                  *)
(*                                                                         *)
(* THE PROTOCOL                                                             *)
(*                                                                         *)
(*  1. STEADY STATE - QUOTA-GATED FORWARDING.  A node forwards its          *)
(*     cumulative subtree total when it reaches (>=, never =) the SUBTREE   *)
(*     quota its plan predicts; below it, it stays SILENT and aggregates.   *)
(*     One report per edge per steady generation is the entire point: the   *)
(*     ungated design's owner fan-in measured ~N-1 messages per generation  *)
(*     under synchronized arrivals, at every radix (SCALE_TEST_PLAN.md      *)
(*     section 7).  Reports remain cumulative and REPLACE, and Trigger      *)
(*     demands exact equality, so a low total simply does not fire.         *)
(*                                                                         *)
(*     THE GATE IS ONLY SAFE BECAUSE OF ITS VALVES (rule 6): an earlier     *)
(*     design gated on the LOCAL quota precisely because subtree-waiting    *)
(*     strands when counts route around the waiter.  Six MCGate scenarios   *)
(*     each deadlock with one valve removed.                                *)
(*  2. OVER-ARRIVAL at a predicted node -> eager flush for that generation, *)
(*     report at once, announce the mode to its children.                   *)
(*  3. ARRIVAL AT A NODE OUTSIDE THE PLAN -> it cannot wait for a match, so *)
(*     it reports DIRECTLY to the owner, which flushes and announces.       *)
(*  4. PATTERN CHANGE.  The owner sends an INVALIDATION down the tree being *)
(*     retired and the NEW PLAN down the new tree.  A node that is still an *)
(*     un-invalidated member of the retiring plan PARKS the new plan; the   *)
(*     invalidation applies it on arrival.  That deferral is what keeps the *)
(*     two broadcasts from racing without any global knowledge: a node's    *)
(*     own membership is the flag, and it is self-clearing.                 *)
(*     On invalidation a node FORWARDS FIRST, then flushes EVERY open       *)
(*     generation, then switches.  Installing a plan RETURNS the node to    *)
(*     planned mode.  Flush MUST NOT be sticky: generations are unbounded   *)
(*     in the implementation, so a node that stayed eager after its first   *)
(*     invalidation would aggregate nothing ever again.  That is only safe  *)
(*     because rule 1 has no child-wait - with one, a re-parented           *)
(*     generation strands on a child that already reported elsewhere.       *)
(*  5. A cumulative total only ever increases; a report that does not       *)
(*     increase the stored value is stale and is discarded.                 *)
(*  6. THE GATE'S VALVES - every way counts can route around a gated relay  *)
(*     has a feed-forward detector whose response is the owner flushing the *)
(*     generation down the current tree:                                    *)
(*      - OWNER RECEIPT VALVE: a report from a sender the current plan does *)
(*        not list as an owner child (MCGateAhead), or whose value exceeds  *)
(*        the sender's predicted subtree quota (MCGateOver).                *)
(*      - OWNER SWITCH AUDIT: installing plan k re-judges every accepted    *)
(*        value against k's subtree quotas - a value accepted legally under *)
(*        k-1 can be deviant under k (MCGateMove) or orphaned entirely      *)
(*        (MCGateDemote).                                                   *)
(*      - INSTALL RE-FAN: flush fans are kid-list snapshots, so a plan      *)
(*        install re-announces every open flushed generation to the NEW     *)
(*        children (MCGateDemote for newplan, MCGatePark for a parked       *)
(*        install).                                                         *)
(*      - STALE-EDGE SIGNAL: the relay that receives a report over a pinned *)
(*        edge the current plan disowns is the one witness that counts are  *)
(*        routing irregularly; it forwards AND signals the owner            *)
(*        (MCGatePark).                                                     *)
(*     Under these valves the case-3 flush signals (immediate and           *)
(*     retroactive) are no longer load-bearing for safety - the mask probe  *)
(*     found no scenario that strands without them - but they REMAIN in the *)
(*     implementation: their flush traffic is the plan-learning evidence    *)
(*     (ARRIVAL_PROTOCOL.md section 11).                                    *)
(*                                                                         *)
(* Feed forward throughout: every action is caused by an arrival or by the  *)
(* receipt of a message.  No timers, no polling, no background sweep.       *)
(*                                                                         *)
(* NOT MODELLED, deliberately: message duplication (the transport is        *)
(* reliable), reduction barriers (not in the first implementation), node    *)
(* failure, and the subscription/trigger-notification tree.                 *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Nodes,          \* set of node ids
    Owner,          \* the fixed logical root
    MaxGen,         \* generations are 1..MaxGen
    Pattern,        \* [Gens -> [Nodes -> Nat]] arrivals the APPLICATION issues
    NumPlans,       \* how many successive arrival plans this run goes through
    Plans,          \* [1..NumPlans -> [Nodes -> [quota, inplan, kids]]]
    PlanStart,      \* [1..NumPlans -> Gens] first generation each plan applies to
    BaseCount,      \* [Gens -> Nat] expected arrival count BEFORE any alteration
    AlterOps        \* set of [node, gen, delta, ts, prev, anc]

\* AN ALTERATION RECORD.  'ts' is its causal timestamp - distinct and positive
\*  across the whole run, because the implementation mints them from a counter
\*  seeded with the node id.  'delta' MAY BE NEGATIVE (event.h permits it and
\*  the implementation applies it; see the terminal-negative note on Alter).
\*
\*  'prev' and 'anc' are the two PREDECESSORS, each 0 for "none":
\*    prev - the previous alteration THIS node issued on this barrier
\*           (barrier_impl.cc, 'last_alter_ts').  Same node, so it orders a
\*           chain that active messages may otherwise reorder.
\*    anc  - the alteration whose HANDLE this one was invoked on.  A handle can
\*           be passed BETWEEN NODES, so 'anc' may name an alteration issued
\*           somewhere else, and that is a dependency 'prev' cannot express.
\*           Applying a descendant before its ancestor moves the owner's
\*           'expected' to a value the application never asked for, and the
\*           arrivals gated on the descendant are then released against it.
\*
\*  The owner HOLDS an alteration until both are applied: RecvAlter below is
\*  simply disabled, which is the model's standard way of writing a queue.
\*
\*  'givento' is the node the application HANDS THE RETURNED HANDLE TO, or
\*   NoTarget for nobody.  It exists so the model can say the common shape of
\*   a positive alteration: the count goes up because a NEW participant will
\*   arrive, and that participant needs a handle which witnesses the
\*   alteration.  See ArriveOnGivenHandle.
\*
\*   RESTRICTION, and it is load-bearing: a node may be the 'givento' of AT
\*   MOST ONE alteration.  'tsSent[n][g]' is one cumulative counter per
\*   (node, generation), which is sound only while every bypassed arrival a
\*   node makes for a generation carries the SAME gate.  Two handles would
\*   give it two gates sharing one counter, and a message whose gate is open
\*   would then carry arrivals whose gates are still shut - TLC catches it
\*   immediately as a TriggerCorrect violation, which is how this restriction
\*   came to be written down.  (The implementation has no such restriction:
\*   it keys ts_issued and the owner's streams per (sender, gate).)

Gens   == 1..MaxGen
Gens0  == 0..MaxGen
Epochs == 1..NumPlans

\* PLAN WELL-FORMEDNESS was tried here as an ASSUME - "every child id exceeds
\*  its parent's", which is how the implementation gets acyclic pinned edges
\*  (it assigns heap parents over ascending node ids, so a parent's id is
\*  always lower).  IT IS NOT A PROPERTY OF THE SCENARIOS, and the attempt is
\*  recorded rather than repeated: MCGateDemote's second plan has node 3 as
\*  the parent of node 1, which is the demoted owner-child the scenario is
\*  named for, and MCGateMove and MCGatePark do the same.  TLC rejects the
\*  whole spec on a failed ASSUME, so all three stopped running.
\*
\*  Ascending ids are SUFFICIENT for what rule 10 needs but not NECESSARY.
\*  What is actually needed is that the UNION of the pinned edges every plan
\*  can produce is acyclic - MCGateDemote's union (1->0, 2->1, 3->0, 1->3)
\*  is, despite the descending edge.  Checking that needs a transitive
\*  closure over the plan constants; it is NOT done, and ARRIVAL_PROTOCOL
\*  section 8.9 still lists it as an assumption rather than a result.
\*
\*  Note also that these scenarios supply plans the implementation CANNOT
\*  build.  That is a safe over-approximation - it explores more - but it
\*  means a mutation they catch is not necessarily reachable in the code.

RECURSIVE SumFn(_, _)
SumFn(S, f) == IF S = {} THEN 0
               ELSE LET x == CHOOSE y \in S : TRUE
                    IN  f[x] + SumFn(S \ {x}, f)

RECURSIVE SumDeltas(_)
SumDeltas(S) == IF S = {} THEN 0
                ELSE LET x == CHOOSE y \in S : TRUE
                     IN  x.delta + SumDeltas(S \ {x})


VARIABLES
    unissued,     \* [Nodes -> [Gens -> Nat]]  arrivals not yet made
    localTotal,   \* [Nodes -> [Gens -> Nat]]  arrivals accumulated here
    reportedUp,   \* [Nodes -> [Gens -> Nat]]  cumulative total last sent to the parent
    childAcc,     \* [Nodes -> [Nodes -> [Gens -> Nat]]] highest accepted per child
    flushing,     \* [Nodes -> [Gens -> BOOLEAN]]
    curPlan,      \* [Nodes -> plan record]    the plan this node currently holds
    myEpoch,      \* [Nodes -> Epochs]         which plan index that is
    invalEpoch,   \* [Nodes -> 0..NumPlans]    highest plan index invalidated here
    deferEpoch,   \* [Nodes -> 0..NumPlans]    a parked new plan (0 = none)
    ownerAcc,     \* [Gens -> Nat]             owner's accepted total per generation
    watermark,    \* highest CONTIGUOUS triggered generation
    triggered,    \* [Gens -> BOOLEAN]
    msgs,         \* set of in-flight messages (a set, so delivery may reorder)
    expected,     \* [Gens -> Nat]   owner's CURRENT expected count (alterations applied)
    appliedTs,    \* SUBSET of alteration timestamps the owner has applied
    unaltered,    \* SUBSET AlterOps  alterations not yet issued
    myTs,         \* [Nodes -> [Gens -> Nat]] timestamp this node's arrivals carry
    reportTo,     \* [Nodes -> [Gens -> Nodes \cup {NoTarget}]] PINNED report edge
    \* [Nodes -> [Gens -> Nat]] arrivals this node BYPASSED the tree with.
    \*  ONE COUNTER PER (node, generation), which is sound only while every
    \*  bypassed arrival a node makes for a generation carries the SAME gate.
    \*  A node that alters TWICE over one generation has two - myTs moves from
    \*  the first timestamp to the second - and they would then share this
    \*  counter, so a message whose gate is open would carry arrivals whose
    \*  gates are still shut.  TLC reports that as a TriggerCorrect violation
    \*  in the UNMUTATED spec, which is how the limitation was found; see
    \*  MCGiven2's header.  The implementation does NOT share it: ts_issued is
    \*  keyed per gate and the owner's streams per (sender, gate).  Scenarios
    \*  must therefore give each node at most one gate per generation.
    tsSent,
    tsAcc         \* [Gens -> Nat]   owner's count of TIMESTAMPED arrivals

vars == << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
           myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered, msgs,
           expected, appliedTs, unaltered, myTs, tsAcc, tsSent, reportTo >>

\* Everything alter_arrival_count adds.  Existing actions leave all of it alone,
\*  so they gain a single conjunct rather than five entries in a frame list.
alterVars == << expected, appliedTs, unaltered, myTs, tsAcc >>

\* Ground truth for how many arrivals generation g will EVER see: the pattern's
\*  base arrivals plus the deltas of every alteration ISSUED so far (they are
\*  persistent, so an alteration at gen k counts for every g >= k).  An
\*  alteration that has not been issued has not created its arrival yet.
Total(g) == SumFn(Nodes, [n \in Nodes |-> Pattern[g][n]])
            + SumDeltas({ a \in (AlterOps \ unaltered) : a.gen <= g })

AppliedDelta(n, g) ==
    SumDeltas({ a \in (AlterOps \ unaltered) : (a.node = n) /\ (a.gen <= g) })

\* How many arrivals this node has BYPASSED the tree with, for g.  It used to be
\*  derived - issued minus localTotal - which silently assumed every arrival a
\*  node makes is booked against its OWN obligation.  That stops being true as
\*  soon as handles cross nodes (see ArriveForAncestor), so it is now kept.
\* THE GATE ON A BYPASSED ARRIVAL, in one place.  It has two independent
\*  components and either may be 0 for "no dependency on this axis", so the
\*  test is needed verbatim by everything that looks at a tsdirect message -
\*  the receive and the stale-drop alike.  Writing it twice is how the
\*  stale-drop came to test only one component and only for a non-zero value,
\*  which left an ungated stale message on the wire for ever and showed up as
\*  an EventuallyDrained counter-example rather than as any safety failure.
\*  This is barrier_impl.cc's ts_gate_open_locked().
TsGateOpen(m) == /\ ((m.ts = 0)       \/ (m.ts \in appliedTs))
                 /\ ((m.local_ts = 0) \/ (m.local_ts \in appliedTs))

\* HOW LONG A RECEIVED HANDLE GATES - the one operator the whole question
\*  turns on, so that comparing the two designs is a one-line change and
\*  cannot drift between the places that depend on it.
\*
\*  AS WRITTEN: only the generation the alteration NAMES.  That is the design
\*   under test - the recipient bypasses once and is back in the tree.  It is
\*   also the WEAKEST version of the rule: the implementation cannot know
\*   a.gen and would use the first generation it actually used the handle at,
\*   which is >= a.gen, so it gates a superset.  Clearing the weak version
\*   clears the strong one.
\*
\*  Mutation HANDLE_GATES_FOREVER rewrites this to 'g >= a.gen', which is the
\*   behaviour as shipped: advance_barrier preserves the timestamp, so every
\*   handle derived from it gates too and the recipient never returns to the
\*   tree.
HandleGateWindow(a, g) == g = a.gen

\* Is this node's arrival routed around the tree for this generation?  Either
\*  because it altered (its own arrival floor) or because it is arriving on a
\*  handle somebody gave it.
Bypassing(m, g) ==
    \/ myTs[m][g] # 0
    \/ \E a \in AlterOps : /\ a.givento = m
                          /\ a \notin unaltered
                          /\ HandleGateWindow(a, g)

\* The node set of a subtree under a supplied plan.  Terminates because the
\*  ASSUME above makes every child id exceed its parent's.
RECURSIVE SubNodes(_, _)
SubNodes(e, n) == {n} \cup UNION { SubNodes(e, c) : c \in Plans[e][n].kids }

\* The alteration a given timestamp names.  Timestamps are globally unique, so
\*  this is a function.  Guard every use with 'ts # 0'.
AlterOf(ts) == CHOOSE a \in AlterOps : a.ts = ts

\* The owner's timestamped total is a SUM OF CUMULATIVE PER-NODE VALUES, exactly
\*  like the tree's childAcc.  One direction, one semantics: every accumulator
\*  in the protocol is replace-if-higher and counts up.
TsTotal(g) == SumFn(Nodes, [n \in Nodes |-> tsAcc[n][g]])

Msgs ==
    [ kind : {"report","direct"}, from : Nodes, to : Nodes, gen : Gens, val : 0..64 ]
    \cup [ kind : {"flush"},      from : Nodes, to : Nodes, gen : Gens ]
    \cup [ kind : {"invalidate"}, from : Nodes, to : Nodes, epoch : 0..NumPlans ]
    \cup [ kind : {"newplan"},    from : Nodes, to : Nodes, epoch : Epochs ]
    \cup [ kind : {"alter"},      from : Nodes, to : Nodes, gen : Gens,
                                  delta : -8..8, ts : 1..8,
                                  prev : 0..8, anc : 0..8 ]
    \* THE GATE HAS TWO COMPONENTS, exactly as Generation::TsKey does: the
    \*  timestamp the HANDLE carries ('ts', which may name an alteration issued
    \*  on another node) and this node's own arrival floor ('local_ts', the
    \*  implementation's ts_floor).  Either may be 0 for "no dependency here",
    \*  and the owner may count the arrival only once BOTH are applied.
    \cup [ kind : {"tsdirect"},   from : Nodes, to : Nodes, gen : Gens,
                                  ts : 0..8, local_ts : 0..8, val : 0..64 ]

(***************************************************************************)
(* THE PINNED REPORT EDGE.  A node's report target for a generation is      *)
(* fixed the FIRST time it touches that generation and a later plan change  *)
(* does not move it.                                                        *)
(*                                                                         *)
(* Without this the owner DOUBLE COUNTS.  Reports are cumulative and only   *)
(* ever increase, so a contribution already folded into an old parent's     *)
(* subtree total can never be retracted; if the child is then re-parented   *)
(* and reports again down a different chain, the owner - which sums per     *)
(* SENDER - counts it twice.  NoOverCount catches it.                       *)
(*                                                                         *)
(* Pinning makes each node's contribution flow up exactly ONE chain per     *)
(* generation.  It assumes the pinned edges cannot form a cycle; the        *)
(* implementation gets that from building plan trees in ascending node id   *)
(* with the owner at the root, so every report edge strictly decreases.     *)
(***************************************************************************)
\* an INTEGER sentinel: TLC cannot evaluate an unbounded CHOOSE, and a string
\*  sentinel makes the # test compare mixed types and throw
NoTarget == -1

\* A node's children come from ITS OWN plan record - never inferred from what
\*  other nodes believe their parent to be.  Mixing those lets a reachability
\*  check conclude a node is reachable when nobody would ever forward to it.
KidsOf(n) == curPlan[n].kids

RECURSIVE Grow(_)
Grow(S) == LET S2 == S \cup UNION { KidsOf(n) : n \in S }
           IN  IF S2 = S THEN S ELSE Grow(S2)

\* Nodes an announcement can still get to.  Seeded from the owner AND from the
\*  target of every announcement already on the wire: a parent that has
\*  forwarded and dropped a child leaves that child in no tree at all, and an
\*  invalidation still travelling toward an ANCESTOR will reach the whole
\*  subtree under it.  Crediting only messages addressed to the holder itself
\*  makes this fire on perfectly healthy in-flight states.
Announced == { m.to : m \in { x \in msgs :
                 x.kind \in {"invalidate","flush","newplan"} } }
Reachable == Grow({Owner} \cup Announced)

SubtreeKnown(n, g) == localTotal[n][g] + SumFn(Nodes, [c \in Nodes |-> childAcc[n][c][g]])
Unreported(n, g)   == SubtreeKnown(n, g) - reportedUp[n][g]
Holding(n, g)      == Unreported(n, g) > 0

ReportWith(n, g, v) ==
    IF n = Owner THEN {}
    ELSE {[ kind |-> "report", from |-> n, to |-> Owner, gen |-> g, val |-> v ]}

\* NOTE: a report goes to the node's PARENT - the explicit parent field of
\*  the plan record THIS NODE installed (barrier_impl report_target_locked),
\*  modelled as a lookup in this node's own epoch's static plan.  A planless
\*  node has no parent and reports to the owner (case 3).  Never derived from
\*  other nodes' records: the implementation cannot see those.
ParentOf(n) == IF curPlan[n].inplan
                    /\ \E p \in Nodes : n \in Plans[myEpoch[n]][p].kids
                 THEN CHOOSE p \in Nodes : n \in Plans[myEpoch[n]][p].kids
                 ELSE Owner

TargetOf(n, g) == IF reportTo[n][g] # NoTarget THEN reportTo[n][g] ELSE ParentOf(n)

Send(n, g, v) ==
    IF n = Owner THEN {}
    ELSE {[ kind |-> "report", from |-> n, to |-> TargetOf(n, g), gen |-> g, val |-> v ]}

\* pin n's edge for every generation in S that is not pinned yet
Pin(f, n, S) == [f EXCEPT ![n] = [g \in Gens |-> IF g \in S THEN TargetOf(n, g)
                                                            ELSE f[n][g]]]

FlushFan(n, g) == { [ kind |-> "flush", from |-> n, to |-> c, gen |-> g ] : c \in KidsOf(n) }

PlanSatisfied(n, g) ==
    /\ curPlan[n].inplan
    /\ localTotal[n][g] = curPlan[n].quota

\* THE SUBTREE QUOTA - what the current plan predicts this node's whole
\*  subtree contributes per generation.  The implementation computes it at
\*  install time from the plan payload (tree shape is a deterministic
\*  function of the (node, quota) vector), so it costs nothing on the wire.
RECURSIVE SubQ(_, _)
SubQ(e, n) == Plans[e][n].quota +
              LET kids == Plans[e][n].kids
              IN  IF kids = {} THEN 0
                  ELSE SumFn(kids, [c \in kids |-> SubQ(e, c)])

SubQCur(n) == SubQ(myEpoch[n], n)

\* THE OWNER VALVE.  A gated relay can strand only when counts route AROUND
\*  it, and every such routing produces exactly one of two owner-visible
\*  symptoms: a report from a sender the current plan does not list as an
\*  owner child (the sender pinned direct while planless, or a retired
\*  relay's invalidation-time forward), or a child whose cumulative value
\*  EXCEEDS its predicted subtree quota (a stowaway count folded into a
\*  legitimate chain).  Either way the owner flushes the generation down the
\*  current tree, which un-gates every holder.
OwnerDeviant(m) == /\ m.to = Owner
                   /\ \/ m.from \notin KidsOf(Owner)
                      \/ m.val > SubQ(myEpoch[Owner], m.from)

Init ==
    /\ unissued   = [n \in Nodes |-> [g \in Gens |-> Pattern[g][n]]]
    /\ localTotal = [n \in Nodes |-> [g \in Gens |-> 0]]
    /\ reportedUp = [n \in Nodes |-> [g \in Gens |-> 0]]
    /\ childAcc   = [n \in Nodes |-> [c \in Nodes |-> [g \in Gens |-> 0]]]
    /\ flushing   = [n \in Nodes |-> [g \in Gens |-> FALSE]]
    /\ curPlan    = Plans[1]
    /\ myEpoch    = [n \in Nodes |-> 1]
    /\ invalEpoch = [n \in Nodes |-> 0]
    /\ deferEpoch = [n \in Nodes |-> 0]
    /\ ownerAcc   = [g \in Gens |-> 0]
    /\ watermark  = 0
    /\ triggered  = [g \in Gens |-> FALSE]
    /\ msgs       = {}
    /\ expected   = BaseCount
    /\ appliedTs  = {}
    /\ unaltered  = AlterOps
    /\ myTs       = [n \in Nodes |-> [g \in Gens |-> 0]]
    /\ reportTo   = [n \in Nodes |-> [g \in Gens |-> NoTarget]]
    /\ tsSent     = [n \in Nodes |-> [g \in Gens |-> 0]]
    /\ tsAcc      = [n \in Nodes |-> [g \in Gens |-> 0]]

(***************************************************************************)
(* ACTION 1 - the application issues an arrival.                           *)
(***************************************************************************)
Arrive(n, g) ==
    /\ unissued[n][g] > 0
    /\ unissued' = [unissued EXCEPT ![n][g] = @ - 1]
    /\ \/ \* This node has issued an alteration covering g, so its arrivals carry
          \*  a causal timestamp.  Tree aggregation collapses arrivals into a
          \*  single integer and would ERASE that timestamp, so a timestamped
          \*  arrival BYPASSES THE TREE and is reported to the owner on its own.
          /\ myTs[n][g] > 0
          /\ tsSent' = [tsSent EXCEPT ![n][g] = @ + 1]
          /\ msgs' = msgs \cup
               {[ kind |-> "tsdirect", from |-> n, to |-> Owner, gen |-> g,
                  ts |-> 0, local_ts |-> myTs[n][g], val |-> tsSent[n][g] + 1 ]}
          /\ UNCHANGED << localTotal, reportedUp, flushing >>
       \/ /\ myTs[n][g] = 0
          /\ UNCHANGED tsSent
          /\ localTotal' = [localTotal EXCEPT ![n][g] = @ + 1]
          /\ LET lt  == localTotal[n][g] + 1
                 sub == lt + SumFn(Nodes, [c \in Nodes |-> childAcc[n][c][g]])
             IN
             \* CASE 3 TAKES PRIORITY OVER THE FLUSHING FLAG.  An outsider's
             \*  arrival must SIGNAL, not merely report: a node that went
             \*  planless has flushing set, and if that branch wins, its count
             \*  reaches the owner as ordinary traffic - a count without a
             \*  signal - and nothing ever tells the owner the plan
             \*  mis-predicts this generation.
             \*
             \*  The COUNT still follows the pinned edge (Send): a cumulative
             \*  value sent straight to the owner would double-count whenever
             \*  an earlier report from this node is already folded into some
             \*  relay's aggregate.  The SIGNAL is count-free: a flush to the
             \*  owner, idempotent there, fanned down the current tree.
             \/ /\ ~curPlan[n].inplan                                \* case 3
                /\ msgs' = msgs \cup Send(n, g, sub)
                           \cup {[ kind |-> "flush", from |-> n, to |-> Owner, gen |-> g ]}
                /\ reportedUp' = [reportedUp EXCEPT ![n][g] = sub]
                /\ UNCHANGED flushing
             \/ /\ flushing[n][g] /\ curPlan[n].inplan              \* already eager
                /\ msgs' = msgs \cup Send(n, g, sub)
                /\ reportedUp' = [reportedUp EXCEPT ![n][g] = sub]
                /\ UNCHANGED flushing
             \/ /\ ~flushing[n][g] /\ curPlan[n].inplan
                /\ lt > curPlan[n].quota                            \* case 2: over-arrival
                /\ flushing' = [flushing EXCEPT ![n][g] = TRUE]
                /\ msgs' = msgs \cup Send(n, g, sub) \cup FlushFan(n, g)
                /\ reportedUp' = [reportedUp EXCEPT ![n][g] = sub]
             \/ /\ ~flushing[n][g] /\ curPlan[n].inplan
                /\ lt <= curPlan[n].quota
                /\ \/ /\ sub >= SubQCur(n)                          \* case 1: subtree done
                      /\ msgs' = msgs \cup Send(n, g, sub)
                      /\ reportedUp' = [reportedUp EXCEPT ![n][g] = sub]
                   \/ /\ sub < SubQCur(n)
                      /\ UNCHANGED << msgs, reportedUp >>            \* SILENCE (the GATE)
                /\ UNCHANGED flushing
    /\ reportTo' = Pin(reportTo, n, {g})
    /\ UNCHANGED << childAcc, curPlan, myEpoch, invalEpoch, deferEpoch,
                    ownerAcc, watermark, triggered >>
    \* 'tsSent' is NOT in alterVars, because THIS action writes it.  A frame
    \*  conjunct naming a variable the action also assigns is a contradiction,
    \*  and TLC reports a contradiction as nothing at all - the action simply
    \*  stops being enabled.  It silently did: with tsSent inside alterVars
    \*  every timestamped arrival disappeared from the model and the ancestor
    \*  bisection "passed" because the hazard had become unreachable.
    /\ UNCHANGED alterVars

(***************************************************************************)
(* ACTION 2 - a cumulative report arrives.  Higher REPLACES; equal or lower *)
(* is stale and discarded.  Accepting it may be what completes this relay.  *)
(***************************************************************************)
RecvReport(m) ==
    /\ m \in msgs /\ m.kind \in {"report","direct"}
    /\ m.val > childAcc[m.to][m.from][m.gen]
    /\ LET acc2 == [childAcc EXCEPT ![m.to][m.from][m.gen] = m.val]
           sub2 == localTotal[m.to][m.gen]
                   + SumFn(Nodes, [c \in Nodes |-> acc2[m.to][c][m.gen]])
           fwd  == /\ m.to # Owner
                   /\ sub2 > reportedUp[m.to][m.gen]
                   /\ \/ flushing[m.to][m.gen]
                      \* A STALE EDGE.  The sender pinned this edge for this
                      \*  generation and a plan change has since moved it out of
                      \*  our child list.  We are still the only route its
                      \*  contribution has, so it must be passed on AT ONCE -
                      \*  holding it behind our own quota strands it, because
                      \*  nothing will ever make that quota relevant again.
                      \/ (m.from \notin KidsOf(m.to))
                      \/ /\ curPlan[m.to].inplan
                         /\ sub2 >= SubQCur(m.to)
       IN  /\ childAcc' = acc2
           /\ msgs' = (msgs \ {m}) \cup (IF fwd THEN Send(m.to, m.gen, sub2) ELSE {})
                      \cup (IF (m.from \notin KidsOf(m.to)) /\ (m.to # Owner)
                              THEN {[ kind |-> "flush", from |-> m.to, to |-> Owner,
                                      gen |-> m.gen ]} ELSE {})
                      \cup (IF ((m.kind = "direct") \/ OwnerDeviant(m))
                              /\ ~flushing[Owner][m.gen]
                              THEN FlushFan(Owner, m.gen) ELSE {})
           /\ reportedUp' = IF fwd THEN [reportedUp EXCEPT ![m.to][m.gen] = sub2]
                                   ELSE reportedUp
           /\ ownerAcc' = IF m.to = Owner
                            THEN [ownerAcc EXCEPT ![m.gen] =
                                    @ - childAcc[Owner][m.from][m.gen] + m.val]
                            ELSE ownerAcc
           /\ flushing' = IF (m.kind = "direct") \/ OwnerDeviant(m)
                            THEN [flushing EXCEPT ![Owner][m.gen] = TRUE]
                            ELSE flushing
    /\ reportTo' = Pin(reportTo, m.to, {m.gen})
    /\ UNCHANGED << unissued, localTotal, curPlan, myEpoch, invalEpoch,
                    deferEpoch, watermark, triggered >>
    /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent

DropStale ==
    \E m \in msgs :
        /\ \/ /\ m.kind \in {"report","direct"}
              /\ m.val <= childAcc[m.to][m.from][m.gen]
           \/ /\ m.kind = "tsdirect"
              /\ TsGateOpen(m)
              /\ m.val <= tsAcc[m.from][m.gen]
        /\ msgs' = msgs \ {m}
        /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                        myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered >>
        /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 3 - a flush announcement arrives.  Idempotent per generation.     *)
(***************************************************************************)
RecvFlush(m) ==
    /\ m \in msgs /\ m.kind = "flush"
    /\ IF flushing[m.to][m.gen]
         THEN /\ msgs' = msgs \ {m}
              /\ UNCHANGED << flushing, reportedUp >>
         ELSE /\ flushing' = [flushing EXCEPT ![m.to][m.gen] = TRUE]
              /\ msgs' = (msgs \ {m}) \cup FlushFan(m.to, m.gen)
                         \cup (IF Unreported(m.to, m.gen) > 0
                                 THEN Send(m.to, m.gen, SubtreeKnown(m.to, m.gen)) ELSE {})
              /\ reportedUp' = [reportedUp EXCEPT ![m.to][m.gen] =
                                  IF Unreported(m.to, m.gen) > 0
                                    THEN SubtreeKnown(m.to, m.gen) ELSE @]
    /\ reportTo' = Pin(reportTo, m.to, {m.gen})
    /\ UNCHANGED << unissued, localTotal, childAcc, curPlan, myEpoch, invalEpoch,
                    deferEpoch, ownerAcc, watermark, triggered >>
    /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent

(***************************************************************************)
(* ACTION 4 - the owner triggers a generation, in order, and switches plans *)
(* when the next plan's start generation has arrived.                       *)
(***************************************************************************)
Trigger(g) ==
    /\ ~triggered[g]
    /\ g = watermark + 1
    /\ ownerAcc[g] + localTotal[Owner][g] + TsTotal(g) = expected[g]
    /\ triggered' = [triggered EXCEPT ![g] = TRUE]
    /\ watermark' = g
    /\ LET switch == { k \in Epochs : (k > 1) /\ (PlanStart[k] = g + 1) }
       IN  IF switch = {}
             THEN UNCHANGED << msgs, curPlan, myEpoch, invalEpoch, deferEpoch,
                               flushing >>
             ELSE LET k == CHOOSE x \in switch : TRUE
                      deviant == { h \in Gens :
                                     /\ h > g
                                     /\ \/ flushing[Owner][h]
                                        \/ \E s \in Nodes \ {Owner} :
                                             \/ childAcc[Owner][s][h] > SubQ(k, s)
                                             \/ /\ childAcc[Owner][s][h] > 0
                                                /\ s \notin Plans[k][Owner].kids }
                  IN  /\ msgs' = msgs
                                 \cup { [ kind |-> "invalidate", from |-> Owner,
                                          to |-> c, epoch |-> myEpoch[Owner] ] :
                                          c \in curPlan[Owner].kids }
                                 \cup { [ kind |-> "newplan", from |-> Owner,
                                          to |-> c, epoch |-> k ] :
                                          c \in Plans[k][Owner].kids }
                                 \cup UNION { { [ kind |-> "flush", from |-> Owner,
                                                  to |-> c, gen |-> h ] :
                                                  c \in Plans[k][Owner].kids } :
                                                h \in deviant }
                      /\ curPlan'    = [curPlan    EXCEPT ![Owner] = Plans[k][Owner]]
                      /\ myEpoch'    = [myEpoch    EXCEPT ![Owner] = k]
                      /\ invalEpoch' = [invalEpoch EXCEPT ![Owner] = k - 1]
                      /\ flushing'   = [flushing EXCEPT ![Owner] =
                                          [h \in Gens |-> IF h \in deviant THEN TRUE
                                                           ELSE flushing[Owner][h]]]
                      /\ UNCHANGED deferEpoch
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, ownerAcc >>
    /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 5 - invalidation.  FORWARD FIRST through the tree being retired,  *)
(* flush EVERY open generation, then apply any parked plan.                 *)
(***************************************************************************)
RecvInvalidate(m) ==
    /\ m \in msgs /\ m.kind = "invalidate"
    /\ IF invalEpoch[m.to] >= m.epoch
         THEN /\ msgs' = msgs \ {m}
              /\ UNCHANGED << reportedUp, flushing, curPlan, myEpoch,
                              invalEpoch, deferEpoch >>
         ELSE /\ LET kids == KidsOf(m.to)
                      held == { g \in Gens : ~triggered[g] /\ Unreported(m.to, g) > 0 }
                      dk   == deferEpoch[m.to]
                      \* A PARKED PLAN DELIVERED BY ITS OWN DEATH NOTICE IS NOT
                      \*  A PLAN.  This invalidation retires epoch m.epoch, so a
                      \*  parked plan with dk <= m.epoch is already retired.
                      \*  Installing it puts this node in planned mode on a dead
                      \*  plan having just consumed the only invalidation that
                      \*  will ever name it; forwarding it does the same to every
                      \*  descendant, who additionally never see the invalidation
                      \*  at all - the one node positioned to route it down the
                      \*  dead plan's edges spent it in this very action.
                      live == dk > m.epoch
                  IN  /\ msgs' = (msgs \ {m})
                                 \cup { [ kind |-> "invalidate", from |-> m.to,
                                          to |-> c, epoch |-> m.epoch ] : c \in kids }
                                 \cup UNION { Send(m.to, g, SubtreeKnown(m.to, g)) :
                                                g \in held }
                                 \cup (IF live
                                         THEN { [ kind |-> "newplan", from |-> m.to,
                                                  to |-> c, epoch |-> dk ] :
                                                  c \in Plans[dk][m.to].kids }
                                              \cup
                                              UNION { { [ kind |-> "flush", from |-> m.to,
                                                          to |-> c, gen |-> x ] :
                                                          c \in Plans[dk][m.to].kids } :
                                                        x \in { h \in Gens :
                                                                  ~triggered[h]
                                                                  /\ (flushing[m.to][h]
                                                                       \/ SubtreeKnown(m.to, h) > 0) } }
                                         ELSE {})
                                 \* RETROACTIVE CASE 3.  A node that ran ahead
                                 \*  arrived believing itself a plan member, so
                                 \*  the outsider rule never fired - its count
                                 \*  reached the owner, but a count is not a
                                 \*  signal, and nothing tells the owner the
                                 \*  plan mis-predicts this generation.  On
                                 \*  learning it is planless, the node delivers
                                 \*  case 3 late: a COUNT-FREE flush signal to
                                 \*  the owner for each open generation it has
                                 \*  arrivals on.  The owner fans the flush down
                                 \*  the current tree, which is what unsticks
                                 \*  contributions parked behind quotas the
                                 \*  retired plan over-predicted.
                                 \cup (IF ~live /\ (m.to # Owner)
                                         THEN { [ kind |-> "flush", from |-> m.to,
                                                  to |-> Owner, gen |-> g ] :
                                                  g \in { h \in Gens :
                                                            ~triggered[h]
                                                            /\ localTotal[m.to][h] > 0 } }
                                         ELSE {})
                      /\ reportedUp' = [reportedUp EXCEPT ![m.to] =
                                          [g \in Gens |-> IF g \in held
                                                            THEN SubtreeKnown(m.to, g)
                                                            ELSE reportedUp[m.to][g]]]
                      \* FLUSH IS PER GENERATION AND OUTLIVES PLAN INSTALLS.
                      \*  Set for every open generation this node has state on;
                      \*  cleared only by that generation triggering.  A plan
                      \*  install must NOT clear it: the owner's deviation
                      \*  flush and the newplan RACE, and if the install wins
                      \*  the flush is lost and the generation strands.
                      \*  Generations with no activity are governed by the plan
                      \*  record alone - planless nodes are outsiders (case 3),
                      \*  planned nodes start planned - so nothing here is
                      \*  sticky across future generations.
                      /\ flushing' = [flushing EXCEPT ![m.to] =
                                        [g \in Gens |->
                                           IF ~triggered[g] /\ SubtreeKnown(m.to, g) > 0
                                             THEN TRUE
                                             ELSE flushing[m.to][g]]]
                      \* A node left with no live plan IS AN OUTSIDER and must
                      \*  say so in its plan record: keeping the retired record
                      \*  (inplan = TRUE) makes later run-ahead arrivals report
                      \*  as ordinary member traffic - a count without a signal
                      \*  - instead of firing case 3.  The retroactive signal
                      \*  above covers arrivals from BEFORE this invalidation;
                      \*  becoming an outsider covers the ones after.
                      /\ curPlan' = IF live THEN [curPlan EXCEPT ![m.to] = Plans[dk][m.to]]
                                            ELSE [curPlan EXCEPT ![m.to] =
                                                    [quota |-> 0, inplan |-> FALSE,
                                                     kids |-> {}]]
                      /\ myEpoch' = IF live THEN [myEpoch EXCEPT ![m.to] = dk] ELSE myEpoch
                      /\ deferEpoch' = [deferEpoch EXCEPT ![m.to] = 0]
              /\ invalEpoch' = [invalEpoch EXCEPT ![m.to] = m.epoch]
    /\ reportTo' = Pin(reportTo, m.to,
                        { g \in Gens : ~triggered[g] /\ Unreported(m.to, g) > 0 })
    /\ UNCHANGED << unissued, localTotal, childAcc, ownerAcc, watermark, triggered >>
    /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent

(***************************************************************************)
(* ACTION 6 - the new plan.  PARK it while still an un-invalidated member   *)
(* of the retiring plan; otherwise apply and pass it on.                    *)
(***************************************************************************)
RecvNewPlan(m) ==
    /\ m \in msgs /\ m.kind = "newplan"
    \* A plan whose RETIREMENT this node has already witnessed is dead on
    \*  arrival, exactly as in RecvInvalidate's 'live' guard: messages reorder,
    \*  so a newplan can arrive after both its own broadcast's invalidations
    \*  have overtaken it.  Installing it would return this node to planned
    \*  mode under a dead plan - having already consumed the only invalidation
    \*  that will ever name it - and forwarding it would do the same to every
    \*  descendant.  One principle, two doors: install only if
    \*  m.epoch > myEpoch AND m.epoch > invalEpoch.
    /\ IF (myEpoch[m.to] >= m.epoch) \/ (invalEpoch[m.to] >= m.epoch)
         THEN /\ msgs' = msgs \ {m}
              /\ UNCHANGED << reportedUp, curPlan, myEpoch, deferEpoch, flushing >>
         ELSE IF curPlan[m.to].inplan /\ (invalEpoch[m.to] < myEpoch[m.to])
                THEN /\ msgs' = msgs \ {m}
                     /\ deferEpoch' = [deferEpoch EXCEPT ![m.to] = m.epoch]
                     /\ UNCHANGED << reportedUp, curPlan, myEpoch, flushing >>
                ELSE /\ LET held == { g \in Gens : ~triggered[g] /\ Unreported(m.to, g) > 0 }
                         IN  /\ msgs' = (msgs \ {m})
                                        \cup { [ kind |-> "newplan", from |-> m.to,
                                                 to |-> c, epoch |-> m.epoch ] :
                                                 c \in Plans[m.epoch][m.to].kids }
                                        \cup UNION { { [ kind |-> "flush", from |-> m.to,
                                                         to |-> c, gen |-> h ] :
                                                         c \in Plans[m.epoch][m.to].kids } :
                                                       h \in { x \in Gens :
                                                                 ~triggered[x]
                                                                 /\ flushing[m.to][x] } }
                                        \cup UNION { Send(m.to, g, SubtreeKnown(m.to, g)) :
                                                       g \in held }
                             /\ reportedUp' = [reportedUp EXCEPT ![m.to] =
                                                 [g \in Gens |-> IF g \in held
                                                                   THEN SubtreeKnown(m.to, g)
                                                                   ELSE reportedUp[m.to][g]]]
                             /\ curPlan' = [curPlan EXCEPT ![m.to] = Plans[m.epoch][m.to]]
                             /\ myEpoch' = [myEpoch EXCEPT ![m.to] = m.epoch]
                             \* a plan install NEVER clears flush: a flushed
                             \*  generation stays eager until it triggers (see
                             \*  RecvInvalidate).  Planned mode is what governs
                             \*  the generations that are NOT flushed.
                             /\ UNCHANGED flushing
                     /\ UNCHANGED deferEpoch
    /\ reportTo' = Pin(reportTo, m.to,
                        { g \in Gens : ~triggered[g] /\ Unreported(m.to, g) > 0 })
    /\ UNCHANGED << unissued, localTotal, childAcc, invalEpoch,
                    ownerAcc, watermark, triggered >>
    /\ UNCHANGED alterVars
    /\ UNCHANGED tsSent

(***************************************************************************)
(* ACTION 7 - alter_arrival_count.  The change is PERSISTENT: it applies to *)
(* this generation and every later one (event.h).  The call is nonblocking  *)
(* and does no round trip, so the alteration travels to the owner while the *)
(* application carries on.                                                  *)
(***************************************************************************)
Alter(a) ==
    /\ a \in unaltered
    /\ ~triggered[a.gen]
    \* AN ALTERATION FOLLOWS ITS PREDECESSORS AT THE ISSUING SIDE TOO.  'prev'
    \*  is the previous alteration this node issued, so it has to have been
    \*  issued; 'anc' is the alteration whose handle this one is invoked on, so
    \*  THAT handle has to exist - the application cannot hold a handle for an
    \*  alteration nobody has made yet.  Both are about ISSUE order, which is
    \*  program order; the owner's APPLY order is RecvAlter's problem.
    /\ \A b \in AlterOps : ((b.ts = a.prev) \/ (b.ts = a.anc)) => (b \notin unaltered)
    \* THE APPLICATION CONTRACT (event.h): before issuing an alteration the
    \*  caller must still hold at least one unissued arrival from the
    \*  pre-alteration count.  That RESERVED ARRIVAL is what stops the barrier
    \*  triggering before the owner has learned of the alteration - the safety
    \*  argument lives in the API contract, not in the protocol.
    /\ unissued[a.node][a.gen] > 0
    \* A NEGATIVE ALTERATION CANCELS AN OBLIGATION, and it can only cancel one
    \*  that is still outstanding: the application may not withdraw an arrival
    \*  it has already made.  For a positive delta this is vacuous.  When the
    \*  cancellation takes the count for a generation to exactly zero, that is
    \*  the TERMINAL NEGATIVE of event.h:293-297 - the generation completes with
    \*  no arrival on the returned handle at all, which is why this guard is a
    \*  '>= 0' and not a '> 0'.
    /\ \A g \in Gens : (g >= a.gen) => (unissued[a.node][g] + a.delta >= 0)
    \* RULE 9 - the expected count must stay positive for every generation the
    \*  alteration reaches, this one included.  Without this a scenario could
    \*  drive a barrier to zero expected arrivals, which event.h makes a fatal
    \*  application error rather than a protocol state.
    /\ \A g \in Gens : (g >= a.gen) => (Total(g) + a.delta > 0)
    /\ unaltered' = unaltered \ {a}
    \* PERSISTENT: the delta applies to this generation and every later one, and
    \*  it is a promise of real arrivals still to be issued by this node (or,
    \*  when negative, the withdrawal of one).
    /\ unissued' = [unissued EXCEPT ![a.node] =
                      [g \in Gens |-> IF g >= a.gen THEN unissued[a.node][g] + a.delta
                                                    ELSE unissued[a.node][g]]]
    \* THE HANDLE SETS THE RECIPIENT'S GATE TOO.  Not just the arrival that
    \*  discharges the obligation: the recipient holds a handle carrying the
    \*  timestamp, so EVERY arrival it makes on it is routed around the tree.
    \*  That is the whole cost of preserving the timestamp, and the model
    \*  could not say it before.
    /\ myTs' = [n \in Nodes |->
                  CASE n = a.node ->
                         [g \in Gens |-> IF g >= a.gen THEN a.ts ELSE myTs[n][g]]
                    [] n = a.givento ->
                         [g \in Gens |-> IF HandleGateWindow(a, g) THEN a.ts
                                                                   ELSE myTs[n][g]]
                    [] OTHER -> myTs[n]]
    \* Altering the count makes this node's plan quota wrong, and because its own
    \*  arrivals now bypass the tree its localTotal can never reach that quota -
    \*  so as a RELAY it would go silent and strand its children.  Enter EAGER
    \*  FLUSH for every affected open generation, exactly as an over-arrival
    \*  does, and announce it downward.
    /\ LET aff  == { g \in Gens : (g >= a.gen) /\ ~triggered[g] }
           held == { g \in Gens : (g >= a.gen) /\ ~triggered[g]
                                 /\ Unreported(a.node, g) > 0 }
       IN  /\ flushing' = [flushing EXCEPT ![a.node] =
                             [g \in Gens |-> IF g \in aff THEN TRUE
                                                          ELSE flushing[a.node][g]]]
           /\ reportedUp' = [reportedUp EXCEPT ![a.node] =
                               [g \in Gens |-> IF g \in held THEN SubtreeKnown(a.node, g)
                                                            ELSE reportedUp[a.node][g]]]
           /\ msgs' = msgs
                      \cup {[ kind |-> "alter", from |-> a.node, to |-> Owner,
                              gen |-> a.gen, delta |-> a.delta, ts |-> a.ts,
                              prev |-> a.prev, anc |-> a.anc ]}
                      \cup UNION { FlushFan(a.node, g) : g \in aff }
                      \cup UNION { Send(a.node, g, SubtreeKnown(a.node, g)) : g \in held }
    /\ UNCHANGED << localTotal, childAcc, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, tsAcc, tsSent >>
    /\ UNCHANGED reportTo

\* THE OWNER-SIDE HOLD (barrier_impl.cc, 'held_alters').  An alteration is
\*  applied only once BOTH its predecessors have been.  Writing it as a
\*  disabled action rather than an explicit queue is the same device
\*  RecvTsDirect uses: it needs no message ordering, and it means TLC explores
\*  every order in which the two could land.
\*
\*  'prev' alone is not enough.  It is per-node, and a handle crosses nodes: B
\*  can issue an alteration descended from A's while A's is still in flight, and
\*  nothing in B's own chain mentions A.  Dropping the 'anc' conjunct below is
\*  the mutation that reproduces that bug, and it reproduces it on MCHandoff -
\*  NOT on MCAncestor, where the ancestor's node arrives too and its own floor
\*  already gates the generation.
RecvAlter(m) ==
    /\ m \in msgs /\ m.kind = "alter"
    /\ ((m.prev = 0) \/ (m.prev \in appliedTs))
    /\ ((m.anc  = 0) \/ (m.anc  \in appliedTs))
    /\ expected' = [g \in Gens |-> IF g >= m.gen THEN expected[g] + m.delta
                                                 ELSE expected[g]]
    /\ appliedTs' = appliedTs \cup {m.ts}
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    unaltered, myTs, tsAcc, tsSent >>
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 7b - AN ARRIVAL MADE ON A DESCENDANT HANDLE, BY ANOTHER NODE.     *)
(*                                                                         *)
(* event.h: the handle an alteration returns must be used by an arrival on  *)
(* it, by a further alteration on it, or by "a descendant chain of          *)
(* alterations that ends in such an arrival" - and "an arrival on a         *)
(* descendant witnesses only the ancestors on its own causal path".  So an  *)
(* arrival made by node B, on a handle B obtained by altering a handle A    *)
(* gave it, DISCHARGES A's reserved arrival.  A itself may never arrive.    *)
(*                                                                         *)
(* THIS IS THE CASE THE PER-NODE MACHINERY CANNOT SEE.  A's own arrival     *)
(* floor gates A's arrivals; here A makes none.  B's arrival carries only   *)
(* B's timestamp, so on its own it witnesses only B's alteration.  What     *)
(* makes it witness A's too is the owner HOLDING B's alteration until A's   *)
(* is applied (the 'anc' conjunct of RecvAlter): after that, "B's timestamp *)
(* is applied" implies "A's is applied".  Delete that conjunct and this     *)
(* action triggers a generation one arrival short - see MCHandoff.          *)
(***************************************************************************)
ArriveForAncestor(a, g) ==
    /\ a \notin unaltered          \* the handle exists only once a is issued
    /\ a.anc # 0                   \* ... and only a descendant has an ancestor
    /\ g >= a.gen                  \* alterations are persistent, so the handle
                                   \*  covers this generation and every later one
    /\ ~triggered[g]
    /\ LET anc == AlterOf(a.anc)
       IN  /\ anc \notin unaltered
           /\ anc.node # a.node    \* a same-node chain is 'prev', not a handoff
           /\ unissued[anc.node][g] > 0
           /\ unissued' = [unissued EXCEPT ![anc.node][g] = @ - 1]
    \* RULE 8.1 - an arrival on a timestamped handle always bypasses the tree.
    \*  Its gate carries the HANDLE's timestamp and, independently, the
    \*  arriving node's own floor.
    /\ tsSent' = [tsSent EXCEPT ![a.node][g] = @ + 1]
    /\ msgs' = msgs \cup
         {[ kind |-> "tsdirect", from |-> a.node, to |-> Owner, gen |-> g,
            ts |-> a.ts, local_ts |-> myTs[a.node][g], val |-> tsSent[a.node][g] + 1 ]}
    /\ UNCHANGED << localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, unaltered, myTs, tsAcc >>
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 7c - AN ARRIVAL BY A NODE THAT WAS SIMPLY GIVEN THE HANDLE.       *)
(*                                                                         *)
(* The case the model had no way to say.  A rank joins a barrier BECAUSE of *)
(* an alteration - somebody raised the count on its behalf and handed it    *)
(* the returned handle - so it arrives on that handle without ever altering *)
(* anything itself.  ArriveForAncestor is NOT this: that one requires the   *)
(* arriving node to have issued a descendant alteration of its own.         *)
(*                                                                         *)
(* THE QUESTION THIS EXISTS TO ANSWER.  advance_barrier preserves the       *)
(* causal timestamp, so this node's handle carries it for every later       *)
(* generation and every arrival it makes bypasses the tree for ever.  The   *)
(* implementation used to zero it, which gated only the alteration's own    *)
(* generation.  Is the gate on the LATER generations load-bearing, or is    *)
(* the reserved arrival at the alteration's own generation already enough,  *)
(* given that generations trigger in watermark order?                       *)
(*                                                                         *)
(* Mutation GIVEN_HANDLE_UNGATED_AFTER rewrites the gate below to           *)
(*   ts |-> IF g = a.gen THEN a.ts ELSE 0                                   *)
(* which is the pre-change behaviour.  Caught => preservation is load-      *)
(* bearing on the arrival path and its cost is the price of correctness.    *)
(* Not caught => preservation is needed only so a later alter_arrival_count *)
(* can chain its ancestry, and the arrival path can be let back into the    *)
(* tree.                                                                    *)
(*                                                                         *)
(* OVER-APPROXIMATION, stated: an ungated arrival here still goes DIRECT to *)
(* the owner rather than through the tree, which is where the implementation*)
(* would route a zero gate.  Direct is strictly the more dangerous of the   *)
(* two for an early-trigger question - it is countable sooner - so a clean  *)
(* result here covers the tree route too.                                   *)
(***************************************************************************)
ArriveOnGivenHandle(a, g) ==
    /\ a \notin unaltered          \* the handle exists only once a is issued
    /\ a.givento # NoTarget        \* ... and was actually handed to somebody
    /\ LET n == a.givento IN
       /\ n # a.node               \* somebody else's handle
       /\ myTs[n][g] = 0           \* and that node has altered nothing itself
    /\ g >= a.gen                  \* alterations are persistent
    /\ ~triggered[g]
    \* it discharges an obligation the ALTERING node booked.  event.h allows
    \*  that: the returned handle needs one use, by anybody on its branch.
    /\ unissued[a.node][g] > 0
    /\ unissued' = [unissued EXCEPT ![a.node][g] = @ - 1]
    /\ tsSent' = [tsSent EXCEPT ![a.givento][g] = @ + 1]
    /\ msgs' = msgs \cup
         {[ kind |-> "tsdirect", from |-> a.givento, to |-> Owner, gen |-> g,
            ts |-> (IF HandleGateWindow(a, g) THEN a.ts ELSE 0), local_ts |-> 0,
            val |-> tsSent[a.givento][g] + 1 ]}
    /\ UNCHANGED << localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, unaltered, myTs, tsAcc >>
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 8 - a timestamped arrival reaches the owner.  THE GATE: it cannot *)
(* be counted until every alteration it witnessed has been applied.  This   *)
(* is barrier_impl.h's per-generation `pending` map keyed by timestamp,     *)
(* modelled as a disabled action - so it needs no message ordering.         *)
(***************************************************************************)
RecvTsDirect(m) ==
    /\ m \in msgs /\ m.kind = "tsdirect"
    /\ TsGateOpen(m)
    /\ m.val > tsAcc[m.from][m.gen]
    /\ tsAcc' = [tsAcc EXCEPT ![m.from][m.gen] = m.val]
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, unaltered, myTs, tsSent >>
    /\ UNCHANGED reportTo

(***************************************************************************)
(* A COMPLETED run stutters, so "finished" is never mistaken for "stuck".   *)
(* Any remaining state with no successor is a genuine deadlock.            *)
(***************************************************************************)
Done == (\A g \in Gens : triggered[g]) /\ UNCHANGED vars

Next ==
    \/ Done
    \/ DropStale
    \/ \E n \in Nodes, g \in Gens : Arrive(n, g)
    \/ \E m \in msgs : RecvReport(m)
    \/ \E m \in msgs : RecvFlush(m)
    \/ \E m \in msgs : RecvInvalidate(m)
    \/ \E m \in msgs : RecvNewPlan(m)
    \/ \E g \in Gens : Trigger(g)
    \/ \E a \in unaltered : Alter(a)
    \/ \E m \in msgs : RecvAlter(m)
    \/ \E m \in msgs : RecvTsDirect(m)
    \/ \E a \in AlterOps, g \in Gens : ArriveForAncestor(a, g)
    \/ \E a \in AlterOps, g \in Gens : ArriveOnGivenHandle(a, g)

\* Everything except the terminal stutter.  'Done' is in Next so that a
\*  finished run is not reported as a deadlock; it must stay OUT of the
\*  fairness condition, or "the run has finished" would discharge the
\*  obligation to make progress on a run that has not.
Active ==
    \/ DropStale
    \/ \E n \in Nodes, g \in Gens : Arrive(n, g)
    \/ \E m \in msgs : RecvReport(m)
    \/ \E m \in msgs : RecvFlush(m)
    \/ \E m \in msgs : RecvInvalidate(m)
    \/ \E m \in msgs : RecvNewPlan(m)
    \/ \E g \in Gens : Trigger(g)
    \/ \E a \in unaltered : Alter(a)
    \/ \E m \in msgs : RecvAlter(m)
    \/ \E m \in msgs : RecvTsDirect(m)
    \/ \E a \in AlterOps, g \in Gens : ArriveForAncestor(a, g)
    \/ \E a \in AlterOps, g \in Gens : ArriveOnGivenHandle(a, g)

Spec == Init /\ [][Next]_vars

\* THE FAIRNESS THIS PROTOCOL ACTUALLY ASSUMES, written down.  Weak fairness
\*  on each action family separately, and not WF_vars(Active) as a whole:
\*  WF of a disjunction is discharged by any one disjunct firing forever, so
\*  a run that delivers reports and never delivers the alteration would
\*  satisfy it.  Per-family is what "every handler eventually runs" means.
\*
\*  The application is fair too (Arrive and Alter): a barrier whose arrivals
\*  are never issued does not trigger, and that is not the protocol's fault.
\*  There is no fairness on 'Done' - it changes nothing.
\*
\*  SYMMETRY IS UNSOUND WITH THIS.  Liveness configs must not declare it.
\*  The delivery conjuncts quantify over 'msgs' INSIDE the WF, not over the
\*  message type outside it.  'Msgs' is a type with thousands of members and
\*  would become thousands of fairness conditions for TLC's liveness graph;
\*  'msgs' is the handful actually in flight.  The weaker reading - "if some
\*  report is continuously deliverable then some report is delivered" -
\*  suffices here because every budget in these scenarios is finite, so no
\*  infinite stream of other messages exists to starve one behind.
Fairness ==
    /\ WF_vars(DropStale)
    /\ \A n \in Nodes, g \in Gens : WF_vars(Arrive(n, g))
    /\ \A a \in AlterOps        : WF_vars(Alter(a))
    /\ \A a \in AlterOps, g \in Gens : WF_vars(ArriveForAncestor(a, g))
    /\ \A a \in AlterOps, g \in Gens : WF_vars(ArriveOnGivenHandle(a, g))
    /\ \A g \in Gens            : WF_vars(Trigger(g))
    /\ WF_vars(\E m \in msgs : RecvReport(m))
    /\ WF_vars(\E m \in msgs : RecvFlush(m))
    /\ WF_vars(\E m \in msgs : RecvInvalidate(m))
    /\ WF_vars(\E m \in msgs : RecvNewPlan(m))
    /\ WF_vars(\E m \in msgs : RecvAlter(m))
    /\ WF_vars(\E m \in msgs : RecvTsDirect(m))

FairSpec == Spec /\ Fairness

(***************************************************************************)
(* LIVENESS.  Safety plus deadlock detection does NOT establish these:      *)
(* deadlock detection finds a state with no successor, and an execution     *)
(* that keeps taking steps forever without ever triggering has successors   *)
(* at every step.  These need FairSpec.                                     *)
(***************************************************************************)

\* Every generation eventually triggers.  The headline property: this is what
\*  a barrier is for, and no invariant in this file implies it.
EventuallyTriggered == <>(\A g \in Gens : triggered[g])

\* And the protocol drains: no message is left on the wire forever.  A
\*  protocol can trigger everything and still leak - a report nobody will ever
\*  accept, an invalidation to a node that has moved on - and a leak is a
\*  retained record in the implementation.
EventuallyDrained == <>[](msgs = {})

\* Nothing is held for ever.  'Holding' is unreported work; this says every
\*  node's unreported work is eventually collected rather than merely that
\*  somebody could collect it, which is all ReachableWhileHolding claims.
EventuallyCollected ==
    \A n \in Nodes, g \in Gens : <>[](~Holding(n, g) \/ triggered[g])

(***************************************************************************)
(* SAFETY                                                                   *)
(***************************************************************************)
TypeOK == /\ watermark \in Gens0
          /\ msgs \subseteq Msgs
          /\ \A g \in Gens : ownerAcc[g] \in 0..64
          /\ \A n \in Nodes, g \in Gens : tsAcc[n][g] \in 0..64
          /\ \A g \in Gens : expected[g] \in 0..64
          /\ \A n \in Nodes, g \in Gens : unissued[n][g] \in 0..64
          /\ \A n \in Nodes, g \in Gens : tsSent[n][g] \in 0..64

TriggerInOrder == \A g \in Gens : triggered[g] => (\A h \in 1..(g-1) : triggered[h])
\* A triggered generation has accounted for EXACTLY every arrival the
\*  application issued - Total(g) is ground truth from Pattern, independent of
\*  what the owner currently believes 'expected' to be.  A trigger against a
\*  stale expected count (one alteration short) violates this immediately, so
\*  this is the check that owns alter_arrival_count.
TriggerCorrect == \A g \in Gens : triggered[g] =>
                      (ownerAcc[g] + localTotal[Owner][g] + TsTotal(g) = Total(g))
NoOverCount    == \A g \in Gens :
                      ownerAcc[g] + localTotal[Owner][g] + TsTotal(g) <= Total(g)

\* THE OWNER'S COUNT IS EXACTLY THE BASE PLUS WHAT IT HAS APPLIED.  This
\*  replaces an 'expected[g] <= Total(g)' bound that only held for NON-NEGATIVE
\*  alterations: with a negative delta in flight, Total(g) has already dropped
\*  (the application issued it) while the owner's count has not, so the old
\*  form reported a mis-specified scenario on a perfectly legal run.
\*
\*  The equality is also strictly stronger where the bound did hold.  It is
\*  violated by applying an alteration twice, by applying one to the wrong
\*  generation, and by applying one that is still being held - none of which
\*  an inequality in either direction can see.
ExpectedSane ==
    \A g \in Gens :
        expected[g] = BaseCount[g]
                      + SumDeltas({ a \in AlterOps : (a.gen <= g)
                                                    /\ (a.ts \in appliedTs) })

\* THE NO-EARLY-TRIGGER PROPERTY, stated directly.  'TriggerCorrect' says the
\*  accepted total EQUALS the arrivals the application will issue, which is an
\*  aggregate: counting one node's arrival twice while another node still owes
\*  one produces the right sum and passes.  This says the thing event.h
\*  actually promises - when a generation triggers, nobody still owes it an
\*  arrival - and it says it per node, so the two errors cannot cancel.
\*
\*  'unissued' is the application's obligation, not the protocol's belief: it
\*  drops only in Arrive (an arrival really made) and moves in Alter (an
\*  obligation created or withdrawn).  No action of the owner's can touch it,
\*  so this cannot be satisfied by the owner deciding it is satisfied.
NoEarlyTrigger ==
    \A g \in Gens : triggered[g] => (\A n \in Nodes : unissued[n][g] = 0)

\* No node may sit on unreported work with nothing that will ever collect it.
\*  The in-flight cases matter: a parent that has already forwarded an
\*  invalidation has discharged its duty and may drop the child from its list,
\*  so the child is briefly in no tree while the message is still on the wire.
ReachableWhileHolding ==
    \A n \in Nodes, g \in Gens :
        (~triggered[g] /\ Holding(n, g) /\ n # Owner) =>
            \/ flushing[n][g]
            \/ ~curPlan[n].inplan
            \/ (n \in Reachable)

\* Retained state is bounded: at most one parked plan per node.
BoundedRetention == \A n \in Nodes : deferEpoch[n] \in 0..NumPlans

\* ONE GATE PER NODE PER GENERATION, checked rather than remembered.
\*  'tsSent' is a single cumulative counter per (node, generation), which is
\*  sound only while every bypassed arrival a node makes for a generation
\*  carries the same gate.  Two gates sharing one counter means a message
\*  whose gate is OPEN carries arrivals whose gates are still SHUT, and the
\*  owner counts them early.  That shows up as a TriggerCorrect violation
\*  several steps later with a trace that says nothing about the cause - it
\*  cost two confusing debugging rounds before it was understood.  This says
\*  it at the point of the mistake.
\*
\*  The implementation has no such restriction: ts_issued is keyed per gate
\*  and the owner's streams per (sender, gate).  This is a MODEL fidelity
\*  bound, and a scenario that trips it is mis-specified, not a protocol bug.
\*  WHAT COUNTS AS "THE SAME GATE" IS THE SET OF TIMESTAMPS IT DEMANDS, not
\*  the two components.  TsGateOpen requires BOTH to be applied, so {0, T}
\*  and {T, T} open at the same instant and may share a counter safely - and
\*  a node that has altered sends both: {0, T} when it arrives on a clean
\*  handle and {T, T} on its own returned one.  The first draft of this
\*  compared components and fired on MCHandoff and MCAncestor, which are
\*  fine.  What must not share a counter is two gates demanding DIFFERENT
\*  timestamps, which is what a node altering twice produces.
GateSet(m) == {m.ts, m.local_ts} \ {0}

TsStreamSingleGate ==
    \A m1 \in msgs, m2 \in msgs :
        (/\ m1.kind = "tsdirect"
         /\ m2.kind = "tsdirect"
         /\ m1.from = m2.from
         /\ m1.gen = m2.gen) => (GateSet(m1) = GateSet(m2))

\* A RECIPIENT MUST NOT STILL BE BYPASSING AFTER THE GENERATION ITS HANDLE
\*  NAMED, while holding a quota somebody is silent on.
\*
\*  BE CLEAR ABOUT WHAT THIS IS.  With HandleGateWindow as written it is
\*  VACUOUS BY CONSTRUCTION - the window ends at a.gen, so "bypassing beyond
\*  a.gen" cannot arise.  It is a REGRESSION GUARD against the window being
\*  widened again, not independent evidence, and it must not be cited as the
\*  latter.
\*
\*  The first draft of it said "no gated node may have ANY bypasser holding a
\*  quota in its subtree", which fired in both arms: the bypass at the
\*  alteration's OWN generation is legitimate and happens under either
\*  design.  Making that version sound needs the owner's retirement on first
\*  bypass, which costs 4,365x on MCAlter (306M states, 55 minutes) and is
\*  not affordable here.  The real hazard - a node silent for ever - is what
\*  TLC's deadlock check already reports, and MCReadmit is paired with it.
NoLingeringBypasser ==
    \A n \in Nodes, g \in Gens :
        (/\ n # Owner
         /\ ~triggered[g]
         /\ curPlan[n].inplan
         /\ ~flushing[n][g]) =>
            (\A m \in SubNodes(myEpoch[n], n) :
                (/\ Plans[myEpoch[n]][m].quota > 0
                 /\ \E a \in AlterOps : /\ a.givento = m
                                       /\ a \notin unaltered
                                       /\ g > a.gen
                                       /\ HandleGateWindow(a, g)) => FALSE)

\* A gate names an alteration that exists.  Cheap, and it catches the class of
\*  modelling slip where an action invents a timestamp - which is how a new
\*  action can look correct and gate on nothing at all.
GateNamesRealAlteration ==
    \A m \in msgs :
        (m.kind = "tsdirect") =>
            (/\ (m.ts = 0)       \/ (\E a \in AlterOps : a.ts = m.ts)
             /\ (m.local_ts = 0) \/ (\E a \in AlterOps : a.ts = m.local_ts))

=============================================================================
