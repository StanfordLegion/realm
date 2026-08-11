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
(*  1. STEADY STATE.  A node forwards its cumulative subtree total when it  *)
(*     matches the quota its plan predicts.  Below quota it stays SILENT.   *)
(*     It does NOT additionally wait for its predicted children.  Reports   *)
(*     are cumulative and REPLACE, so a forward that omits a child is       *)
(*     superseded when that child reports, and Trigger demands exact        *)
(*     equality so a low total simply does not fire.  An earlier draft had  *)
(*     a child-wait and this file warned against removing it; seven         *)
(*     scenarios say otherwise - see rule 4.  Only nodes with a non-zero    *)
(*     expected contribution are in a plan at all.                          *)
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
    AlterOps        \* set of [node, gen, delta, ts] - ts distinct and positive

Gens   == 1..MaxGen
Gens0  == 0..MaxGen
Epochs == 1..NumPlans

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
    tsAcc         \* [Gens -> Nat]   owner's count of TIMESTAMPED arrivals

vars == << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
           myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered, msgs,
           expected, appliedTs, unaltered, myTs, tsAcc, reportTo >>

\* Everything alter_arrival_count adds.  Existing actions leave all of it alone,
\*  so they gain a single conjunct rather than five entries in a frame list.
alterVars == << expected, appliedTs, unaltered, myTs, tsAcc >>

\* Ground truth for how many arrivals generation g will EVER see: the pattern's
\*  base arrivals plus the deltas of every alteration ISSUED so far (they are
\*  persistent, so an alteration at gen k counts for every g >= k).  An
\*  alteration that has not been issued has not created its arrival yet.
Total(g) == SumFn(Nodes, [n \in Nodes |-> Pattern[g][n]])
            + SumDeltas({ a \in (AlterOps \ unaltered) : a.gen <= g })

\* Arrivals this node has issued for g, and how many of those BYPASSED the tree.
\*  A bypassed arrival never enters localTotal, so the difference is exact.
AppliedDelta(n, g) ==
    SumDeltas({ a \in (AlterOps \ unaltered) : (a.node = n) /\ (a.gen <= g) })
Issued(n, g)   == Pattern[g][n] + AppliedDelta(n, g) - unissued[n][g]
TsIssued(n, g) == Issued(n, g) - localTotal[n][g]

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
                                  delta : 0..8, ts : 1..8 ]
    \cup [ kind : {"tsdirect"},   from : Nodes, to : Nodes, gen : Gens, ts : 1..8,
                                  val : 0..64 ]

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

\* NOTE: a report goes to the node's PARENT.  The parent is whoever currently
\*  lists n as a child; modelled by addressing the report to that node.
ParentOf(n) == IF \E p \in Nodes : n \in curPlan[p].kids
                 THEN CHOOSE p \in Nodes : n \in curPlan[p].kids
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
          /\ msgs' = msgs \cup
               {[ kind |-> "tsdirect", from |-> n, to |-> Owner,
                  gen |-> g, ts |-> myTs[n][g], val |-> TsIssued(n, g) + 1 ]}
          /\ UNCHANGED << localTotal, reportedUp, flushing >>
       \/ /\ myTs[n][g] = 0
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
                /\ \/ /\ lt = curPlan[n].quota                      \* case 1: matched
                      /\ msgs' = msgs \cup Send(n, g, sub)
                      /\ reportedUp' = [reportedUp EXCEPT ![n][g] = sub]
                   \/ /\ ~( lt = curPlan[n].quota )
                      /\ UNCHANGED << msgs, reportedUp >>            \* SILENCE
                /\ UNCHANGED flushing
    /\ reportTo' = Pin(reportTo, n, {g})
    /\ UNCHANGED << childAcc, curPlan, myEpoch, invalEpoch, deferEpoch,
                    ownerAcc, watermark, triggered >>
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
                         /\ localTotal[m.to][m.gen] = curPlan[m.to].quota
       IN  /\ childAcc' = acc2
           /\ msgs' = (msgs \ {m}) \cup (IF fwd THEN Send(m.to, m.gen, sub2) ELSE {})
                      \cup (IF (m.kind = "direct") /\ ~flushing[Owner][m.gen]
                              THEN FlushFan(Owner, m.gen) ELSE {})
           /\ reportedUp' = IF fwd THEN [reportedUp EXCEPT ![m.to][m.gen] = sub2]
                                   ELSE reportedUp
           /\ ownerAcc' = IF m.to = Owner
                            THEN [ownerAcc EXCEPT ![m.gen] =
                                    @ - childAcc[Owner][m.from][m.gen] + m.val]
                            ELSE ownerAcc
           /\ flushing' = IF m.kind = "direct"
                            THEN [flushing EXCEPT ![Owner][m.gen] = TRUE]
                            ELSE flushing
    /\ reportTo' = Pin(reportTo, m.to, {m.gen})
    /\ UNCHANGED << unissued, localTotal, curPlan, myEpoch, invalEpoch,
                    deferEpoch, watermark, triggered >>
    /\ UNCHANGED alterVars

DropStale ==
    \E m \in msgs :
        /\ \/ /\ m.kind \in {"report","direct"}
              /\ m.val <= childAcc[m.to][m.from][m.gen]
           \/ /\ m.kind = "tsdirect"
              /\ m.ts \in appliedTs
              /\ m.val <= tsAcc[m.from][m.gen]
        /\ msgs' = msgs \ {m}
        /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                        myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered >>
        /\ UNCHANGED alterVars
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
             THEN UNCHANGED << msgs, curPlan, myEpoch, invalEpoch, deferEpoch >>
             ELSE LET k == CHOOSE x \in switch : TRUE
                  IN  /\ msgs' = msgs
                                 \cup { [ kind |-> "invalidate", from |-> Owner,
                                          to |-> c, epoch |-> myEpoch[Owner] ] :
                                          c \in curPlan[Owner].kids }
                                 \cup { [ kind |-> "newplan", from |-> Owner,
                                          to |-> c, epoch |-> k ] :
                                          c \in Plans[k][Owner].kids }
                      /\ curPlan'    = [curPlan    EXCEPT ![Owner] = Plans[k][Owner]]
                      /\ myEpoch'    = [myEpoch    EXCEPT ![Owner] = k]
                      /\ invalEpoch' = [invalEpoch EXCEPT ![Owner] = k - 1]
                      /\ UNCHANGED deferEpoch
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, ownerAcc >>
    /\ UNCHANGED alterVars
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

(***************************************************************************)
(* ACTION 7 - alter_arrival_count.  The change is PERSISTENT: it applies to *)
(* this generation and every later one (event.h).  The call is nonblocking  *)
(* and does no round trip, so the alteration travels to the owner while the *)
(* application carries on.                                                  *)
(***************************************************************************)
Alter(a) ==
    /\ a \in unaltered
    /\ ~triggered[a.gen]
    \* THE APPLICATION CONTRACT (event.h): before issuing an alteration the
    \*  caller must still hold at least one unissued arrival from the
    \*  pre-alteration count.  That RESERVED ARRIVAL is what stops the barrier
    \*  triggering before the owner has learned of the alteration - the safety
    \*  argument lives in the API contract, not in the protocol.
    /\ unissued[a.node][a.gen] > 0
    /\ unaltered' = unaltered \ {a}
    \* PERSISTENT: the delta applies to this generation and every later one, and
    \*  it is a promise of real arrivals still to be issued by this node.
    /\ unissued' = [unissued EXCEPT ![a.node] =
                      [g \in Gens |-> IF g >= a.gen THEN unissued[a.node][g] + a.delta
                                                    ELSE unissued[a.node][g]]]
    /\ myTs' = [myTs EXCEPT ![a.node] =
                  [g \in Gens |-> IF g >= a.gen THEN a.ts ELSE myTs[a.node][g]]]
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
                              gen |-> a.gen, delta |-> a.delta, ts |-> a.ts ]}
                      \cup UNION { FlushFan(a.node, g) : g \in aff }
                      \cup UNION { Send(a.node, g, SubtreeKnown(a.node, g)) : g \in held }
    /\ UNCHANGED << localTotal, childAcc, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, tsAcc >>
    /\ UNCHANGED reportTo

RecvAlter(m) ==
    /\ m \in msgs /\ m.kind = "alter"
    /\ expected' = [g \in Gens |-> IF g >= m.gen THEN expected[g] + m.delta
                                                 ELSE expected[g]]
    /\ appliedTs' = appliedTs \cup {m.ts}
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    unaltered, myTs, tsAcc >>
    /\ UNCHANGED reportTo

(***************************************************************************)
(* ACTION 8 - a timestamped arrival reaches the owner.  THE GATE: it cannot *)
(* be counted until every alteration it witnessed has been applied.  This   *)
(* is barrier_impl.h's per-generation `pending` map keyed by timestamp,     *)
(* modelled as a disabled action - so it needs no message ordering.         *)
(***************************************************************************)
RecvTsDirect(m) ==
    /\ m \in msgs /\ m.kind = "tsdirect"
    /\ m.ts \in appliedTs
    /\ m.val > tsAcc[m.from][m.gen]
    /\ tsAcc' = [tsAcc EXCEPT ![m.from][m.gen] = m.val]
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED << unissued, localTotal, reportedUp, childAcc, flushing, curPlan,
                    myEpoch, invalEpoch, deferEpoch, ownerAcc, watermark, triggered,
                    expected, appliedTs, unaltered, myTs >>
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

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* SAFETY                                                                   *)
(***************************************************************************)
TypeOK == /\ watermark \in Gens0
          /\ msgs \subseteq Msgs
          /\ \A g \in Gens : ownerAcc[g] \in 0..64
          /\ \A n \in Nodes, g \in Gens : tsAcc[n][g] \in 0..64
          /\ \A g \in Gens : expected[g] \in 0..64

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

\* Scenario sanity: with only non-negative alterations the owner's expected
\*  count climbs toward - never past - the arrivals the application will issue.
\*  Catches a mis-specified scenario rather than a protocol defect.
ExpectedSane == \A g \in Gens : expected[g] <= Total(g)

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

=============================================================================
