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

------------------------------ MODULE MCChainSign -----------------------------
(***************************************************************************)
(* THE SAME-NODE CHAIN, WITH THE SIGNS THAT MAKE IT MATTER.                 *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1                                             *)
(*                     |- 2  q1                                             *)
(*                                                                         *)
(* Node 2 issues +1 (ts 1) on a base-count-2 barrier and then -1 (ts 2,     *)
(* 'prev' 1) on the handle that returned.  The net change is zero and the   *)
(* final expected count is 2, the same as the base.                         *)
(*                                                                         *)
(* THE HAZARD IS THE ORDER, NOT THE TOTAL.  Applied out of order the owner  *)
(* sees the -1 first and takes expected to ONE - a count the application    *)
(* was never in, since it went 2 -> 3 -> 2.  Node 1's single untimestamped  *)
(* arrival then satisfies it on its own and generation 1 triggers while     *)
(* node 2 still owes an arrival: a barrier that triggered too early.        *)
(*                                                                         *)
(* This is the scenario for the 'prev' conjunct of RecvAlter.  It needs     *)
(* BOTH signs: with two positive alterations the out-of-order apply only    *)
(* ever leaves 'expected' too LOW by the amount not yet applied and the     *)
(* trigger simply waits, and with the negative FIRST (MCNegative) the       *)
(* lowered count is the truth.  Only +1 then -1 puts the owner below the    *)
(* count the application actually asked for.                                *)
(*                                                                         *)
(* One generation: the hazard is an apply-order hazard at the owner and it  *)
(* is complete within the generation the alterations name.                  *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2}
MCOwner  == 0
MCMaxGen == 1

MCPattern == [ g \in 1..1 |-> [n \in MCNodes |-> IF n = 0 THEN 0 ELSE 1] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..1 |-> 2]

MCAlterOps  == { [node |-> 2, gen |-> 1, delta |-> 1,  ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> NoTarget],
                 [node |-> 2, gen |-> 1, delta |-> -1, ts |-> 2,
                  prev |-> 1, anc |-> 0, givento |-> NoTarget] }

=============================================================================
