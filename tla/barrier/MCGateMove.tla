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

------------------------------ MODULE MCGateMove ------------------------------
(***************************************************************************)
(* THE HIDDEN-DISPLACEMENT STRAND: a leaf MOVES between relays across a     *)
(* switch and its run-ahead count ends up folded inside a LEGITIMATE        *)
(* owner-child's aggregate.  No stale edge ever fires (the pinned edge was  *)
(* live when used and never carries again); the owner never sees a non-kid  *)
(* sender.  The only symptom anywhere is ARITHMETIC: the owner-child's      *)
(* cumulative value EXCEEDS its predicted subtree quota, while the leaf's   *)
(* new relay gates on a subtree that can never complete.                    *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 1  q1     (keeps 1, takes 2 away from it)   *)
(*        |- 2  q1           |- 3  q1                                      *)
(*                              |- 2  q1                                   *)
(*                                                                         *)
(*   g2  node 2 runs ahead under plan 1: pins to node 1, count 1.           *)
(*       Node 1 switches, its own arrival completes its plan-2 subtree      *)
(*       (quota 1, no kids) PLUS the stowaway - it reports 2 where plan 2   *)
(*       predicts 1.  Node 3 gates on quota 2 (its own + node 2's) and      *)
(*       can never complete.  Exercise-proof for the owner's                *)
(*       value-exceeds-subtree-quota valve.                                 *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 3}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
