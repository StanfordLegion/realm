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

------------------------------ MODULE MCGateOver ------------------------------
(***************************************************************************)
(* THE GATED OVER/UNDER STRAND, minimal form.  A conserved deviation pair   *)
(* that STRADDLES a gated relay: the under-arrival is inside the relay's    *)
(* subtree, the over-arrival is outside it (an owner-direct leaf), so the   *)
(* relay's subtree quota is unreachable and the over-arriver's flush fan    *)
(* (which goes DOWN from the deviator) never touches the relay.             *)
(*                                                                         *)
(*        0                pattern g1:  node 1 -> 1   (relay, quota 1)      *)
(*        |- 1  q1          node 2 -> 0   (under: relay's subtree short)    *)
(*        |  |- 2  q1       node 3 -> 2   (over: owner-direct leaf)         *)
(*        |- 3  q1                                                          *)
(*                                                                         *)
(* Without the over-arrival's count-free flush signal TO THE OWNER, nothing *)
(* un-gates node 1 and its own count strands: deadlock.  This is the        *)
(* exercise-proof scenario for that valve.                                  *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 1

MCPattern ==
    [ g \in 1..1 |->
        [n \in MCNodes |-> CASE n = 1 -> 1
                             [] n = 3 -> 2
                             [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 3}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
