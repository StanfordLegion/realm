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

------------------------------- MODULE MCHandoff ------------------------------
(***************************************************************************)
(* THE CROSS-NODE HANDOFF, which is the case 'prev' cannot describe.        *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1                                             *)
(*                     |- 2  q1                                             *)
(*                                                                         *)
(* Node 1 issues +1 (ts 1) on a base-count-2 barrier and passes the handle  *)
(* it returns to node 2 WHILE THE ALTERATION IS STILL ON THE WIRE.  Node 2  *)
(* issues a descendant +1 on that handle (ts 2, anc 1).  The final expected *)
(* count is four.                                                           *)
(*                                                                         *)
(* Node 1 then makes NO arrivals of its own.  event.h allows that: its      *)
(* reserved arrival may be discharged by "a descendant chain of alterations *)
(* that ends in such an arrival", and node 2's arrivals on its own returned *)
(* handle are that chain.  So all four arrivals are made by node 2, and     *)
(* every one of them carries ts 2 and nothing else.                         *)
(*                                                                         *)
(* NOTHING LOCAL CAN SAVE THIS.  Node 1's arrival floor gates node 1's      *)
(* arrivals, and node 1 makes none.  Node 2's floor is ts 2, which the      *)
(* owner can apply the moment the message lands.  If it does, expected goes *)
(* 2 -> 3, three arrivals gated on ts 2 are released against it, and        *)
(* generation 1 triggers one arrival short of the four it is owed.          *)
(*                                                                         *)
(* The 'anc' conjunct of RecvAlter is what prevents it: holding ts 2 until  *)
(* ts 1 is applied makes "ts 2 is applied" imply "ts 1 is applied", so a    *)
(* gate on ts 2 transitively gates on ts 1.  Deleting that conjunct is      *)
(* mutation ANCESTOR_HOLD and it MUST violate NoEarlyTrigger here.          *)
(*                                                                         *)
(* One generation only: the hazard is an APPLY-ORDER hazard at the owner    *)
(* and it is complete within the generation the alterations name.           *)
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

MCAlterOps  == { [node |-> 1, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> NoTarget],
                 [node |-> 2, gen |-> 1, delta |-> 1, ts |-> 2,
                  prev |-> 0, anc |-> 1, givento |-> NoTarget] }

=============================================================================
