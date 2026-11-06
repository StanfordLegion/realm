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

------------------------------- MODULE MCReadmit ------------------------------
(***************************************************************************)
(* RE-ADMITTING A NODE THAT ONCE BYPASSED.  The scenario that answers       *)
(* whether the owner may stop excluding a handle recipient for ever.        *)
(*                                                                         *)
(*   plan 1 (gen 1)      0              plan 2 (gen 2)      0               *)
(*                       |- 1  q1                           |- 1  q1        *)
(*                       |- 2  q1                           |- 2  q1        *)
(*                     (3 is OUTSIDE)                          |- 3  q1     *)
(*                                                                         *)
(* NODE 3 IS OUTSIDE PLAN 1 AND INSIDE PLAN 2, which is what the owner      *)
(* actually does: a node whose arrivals bypassed contributes nothing to the *)
(* evidence a plan is built from, so the rebuild that follows leaves it     *)
(* out; a later one takes it back once its arrivals reappear in the tree.   *)
(*                                                                         *)
(* An earlier draft gave node 3 a quota in plan 1 as well.  That strands    *)
(* node 2 at generation 1 in BOTH arms - the bypass at the alteration's own *)
(* generation is legitimate and happens either way - so the scenario could  *)
(* not tell the two designs apart.  The question is about what happens      *)
(* AFTER that generation, and the plans have to say so.                      *)
(*                                                                         *)
(*   node 1  base 1/gen; alters +1 at gen 1 and hands the handle to node 3  *)
(*   node 2  base 1/gen; a RELAY, and the node at risk - it is silent until *)
(*           its subtree quota (itself plus node 3) is met                  *)
(*   node 3  base 1/gen; a leaf under node 2, and the recipient             *)
(*                                                                         *)
(* THE ALTERER IS A DIRECT CHILD OF THE OWNER, deliberately.  Alter makes   *)
(* the altering node flush itself, so it can never strand anyone above it;  *)
(* putting it under a non-owner relay would instead exercise rule 9's       *)
(* retirement, which this model does not have (see NoGatedWaitOnBypasser's  *)
(* precondition).                                                           *)
(*                                                                         *)
(* WHAT EACH OUTCOME MEANS.  Node 3 bypasses at generation 1.  At           *)
(* generation 2 plan 2 is installed and counts on it again:                 *)
(*                                                                         *)
(*   as written  - the handle gates only generation 1, so node 3 is back in *)
(*                 the tree by generation 2, node 2's subtree quota of two  *)
(*                 is met, and the run completes.  Re-admission is sound.   *)
(*   HANDLE_GATES_FOREVER - node 3 never returns, node 2's subtree quota of *)
(*                 two can never be met, node 2 is silent for ever and the  *)
(*                 run DEADLOCKS.  That is the behaviour as shipped, and    *)
(*                 the reason the owner has to exclude recipients           *)
(*                 permanently today.                                        *)
(*                                                                         *)
(* The discriminator is DEADLOCK, not an invariant.  That is the honest     *)
(* tool here: the hazard is a node going silent for ever, and TLC's         *)
(* deadlock check says exactly that without needing the owner's             *)
(* retirement-on-bypass, which costs 4,365x on MCAlter and is not           *)
(* affordable in this model.                                                 *)
(*                                                                         *)
(*   gen 1  base 3 arrivals, +1 from the alteration -> expected and issued 4*)
(*   gen 2  the alteration is persistent -> expected and issued 4           *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern == [ g \in 1..2 |-> [n \in MCNodes |-> IF n = 0 THEN 0 ELSE 1] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

\* plan 1: node 3 is not a member - it has not contributed through the tree
MCTree1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

\* plan 2: node 3 is taken back, under node 2
MCTree2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCTree1, MCTree2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..2 |-> 3]

MCAlterOps  == { [node |-> 1, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> 3] }

=============================================================================
