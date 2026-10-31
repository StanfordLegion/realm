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

------------------------------ MODULE MCGateJump ------------------------------
(***************************************************************************)
(* THE QUOTA-JUMP CASE: why the gate is >= and never =.  Reports are        *)
(* cumulative and REPLACE, so an accepted value can move a relay's subtree  *)
(* total PAST its quota in one step: the over-arriving leaf's val-2 report  *)
(* can be delivered before (or instead of) its val-1 report.  A gate that   *)
(* tests equality never opens and the relay's own count strands - and the   *)
(* owner valve is silent, because the aggregate it sees stays within the    *)
(* relay's predicted subtree quota.                                         *)
(*                                                                         *)
(*     0                pattern g1:  node 1 -> 1   (relay, quota 1)         *)
(*     |- 1  q1          node 2 -> 2   (over-arriving leaf)                 *)
(*        |- 2  q1                                                          *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2}
MCOwner  == 0
MCMaxGen == 1

MCPattern ==
    [ g \in 1..1 |->
        [n \in MCNodes |-> CASE n = 1 -> 1
                             [] n = 2 -> 2
                             [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
