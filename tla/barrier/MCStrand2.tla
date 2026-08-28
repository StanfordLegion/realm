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

------------------------------ MODULE MCStrand2 ------------------------------
(***************************************************************************)
(* The STRANDING that per-generation pinning introduces, minimally.         *)
(* MCDeepSwitch shows it at millions of states; this shows it at thousands. *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 3  q1                                      *)
(*        |- 2  q1                                                         *)
(*                                                                         *)
(* Nodes 1 AND 2 are dropped entirely by plan 2.                            *)
(*                                                                         *)
(*   1. node 2 runs ahead and arrives on g2 while still on plan 1; its      *)
(*      quota is met so it reports to node 1 and PINS that edge for g2      *)
(*   2. the plan switches; node 1 is invalidated, flushes what it holds     *)
(*      (nothing for g2 yet) and gets no replacement plan                   *)
(*   3. node 2's report reaches node 1 afterwards.  Node 1 is below its own *)
(*      quota, is no longer flushing because the switch cleared it, and is  *)
(*      no longer reachable from the owner - plan 2's tree goes elsewhere   *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 3

MCPattern ==
    [ g \in 1..3 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ]
          [] g = 3 -> [n \in MCNodes |-> CASE n = 2 -> 1
                                           [] n = 3 -> 1   \* runs ahead, then orphaned
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

\* node 1 is still a RELAY here, with node 3 beneath it
MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

\* ...and only NOW is node 1 dropped, after node 3 has pinned to it
MCPlan3 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 3
MCPlans     == <<MCPlan1, MCPlan2, MCPlan3>>
MCPlanStart == <<1, 2, 3>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
