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

----------------------------- MODULE MCDeviate -----------------------------
(***************************************************************************)
(* Exercises the DEVIATION paths, which the other scenarios never reach     *)
(* because there every generation's pattern matched the plan governing it.  *)
(*                                                                         *)
(* One plan throughout (no pattern change at all), so nothing here depends  *)
(* on the invalidation machinery:                                           *)
(*                                                                         *)
(*     0 = owner                                                            *)
(*     |- 1  quota 1                                                        *)
(*     |  |- 3  quota 2                                                     *)
(*     |- 2  quota 1                                                        *)
(*     4 is in no plan                                                      *)
(*                                                                         *)
(* Generation 1 matches the plan exactly - the steady-state case.           *)
(*                                                                         *)
(* Generation 2 DISAGREES with it in two ways at once:                      *)
(*   - node 2 arrives 3 times against a quota of 1  -> OVER-ARRIVAL, case 2 *)
(*   - node 4 arrives at all, and is in no plan     -> OUTSIDER,    case 3  *)
(* so the generation can only complete if the over-arriving node flushes    *)
(* and the outsider reports directly to the owner.                          *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        IF g = 1 THEN [n \in MCNodes |-> CASE n = 0 -> 0   \* matches the plan
                                           [] n = 1 -> 1
                                           [] n = 2 -> 1
                                           [] n = 3 -> 2
                                           [] n = 4 -> 0 ]
                 ELSE [n \in MCNodes |-> CASE n = 0 -> 0
                                           [] n = 1 -> 1
                                           [] n = 2 -> 3   \* over-arrives (quota 1)
                                           [] n = 3 -> 2
                                           [] n = 4 -> 1 ] ] \* outsider arrives

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

\* No alterations in this scenario: the base count is simply every arrival the
\*  pattern issues, so the model must reduce EXACTLY to the pre-alteration one.
MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
