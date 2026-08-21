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

------------------------------ MODULE MCGateDemote ------------------------------
(***************************************************************************)
(* THE DEMOTED-RELAY STRAND: the case only the owner's NON-KID test can     *)
(* see.  A relay is demoted from owner-child to grandchild by the new plan  *)
(* but keeps its pinned edge to the owner for a straddling generation, so   *)
(* its reports keep arriving at the owner - value-legal (its cumulative     *)
(* never exceeds its own predicted subtree), never planless (so no case-3   *)
(* or retroactive flush), yet invisible to its NEW parent, which gates on   *)
(* a subtree that includes them.                                            *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 3  q1                                      *)
(*        |- 2  q1              |- 1  q1                                   *)
(*                                 |- 2  q1                                *)
(*                                                                         *)
(*   g2  node 2 runs ahead under plan 1 and reports to node 1; node 1,      *)
(*       still holding (gated), pins gen 2 to the owner and is then         *)
(*       switched to plan 2 - where it remains a relay over node 2 but      *)
(*       under node 3.  Its pinned forwards for gen 2 keep going to the     *)
(*       owner; node 3 gates on quota 3 and can never complete.             *)
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
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
