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

------------------------------ MODULE MCGateAhead ------------------------------
(***************************************************************************)
(* THE RUN-AHEAD MIXED-PIN STRAND against the gate.                         *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 3  q1                                      *)
(*        |- 2  q1              |- 2  q1                                   *)
(*                                                                         *)
(*   g2  node 2 runs ahead and arrives while still on plan 1: its count     *)
(*       pins to node 1 and reaches the owner through node 1's              *)
(*       invalidation-time forward - arriving at the owner FROM A SENDER    *)
(*       THE CURRENT PLAN DISOWNS.  Node 3 installs plan 2, arrives, and    *)
(*       gates on a subtree quota (its own 1 + node 2's 1) that can never   *)
(*       complete: node 2's count routed around it.  Node 3 is never        *)
(*       invalidated (it was not in plan 1's tree), so no switch-time       *)
(*       flush touches it.                                                  *)
(*                                                                         *)
(* The only party that can see the deviation is the OWNER: a report from a  *)
(* sender that is not a current-plan child.  This scenario is the           *)
(* exercise-proof for the owner-side valve.                                 *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 2 -> 1
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
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {3}]
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
