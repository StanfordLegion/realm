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

------------------------------ MODULE MCNegative ------------------------------
(***************************************************************************)
(* NEGATIVE AND MIXED-SIGN ALTERATIONS, including the TERMINAL NEGATIVE.    *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1                                             *)
(*                     |- 2  q1                                             *)
(*                        |- 3  q1                                          *)
(*                                                                         *)
(* Node 2 alters -1 (ts 1) at generation 1 and then +1 (ts 2) at generation *)
(* 2, the second chained behind the first ('prev' 1).  TWO DEPENDENT        *)
(* ALTERATIONS OF OPPOSITE SIGN, from one node, across a plan switch at     *)
(* generation 2 - the combination the singleton positive scenario cannot    *)
(* reach.                                                                   *)
(*                                                                         *)
(* THE TERMINAL NEGATIVE (event.h:293-297).  Node 2's -1 cancels the one    *)
(* arrival it owed for generation 1, so after the alteration node 2 owes    *)
(* NOTHING for that generation: it completes with no arrival on the handle  *)
(* the alteration returned.  That is the one branch of the contract whose   *)
(* safety does not rest on a reserved arrival, and the implementation used  *)
(* to abort on reaching it.  Here it is just a run that has to come out     *)
(* right, and NoEarlyTrigger is what says so: generation 1 may trigger only *)
(* once nobody owes it anything, and after the cancellation nobody does.    *)
(*                                                                         *)
(* THE MIXED-SIGN ORDERING.  Generation 2 carries both deltas (alterations  *)
(* are persistent), and they move expected in opposite directions.  An      *)
(* owner that applied one of them twice, or applied one to the wrong        *)
(* generation, lands on an expected count that no prefix of the applied set *)
(* explains - which is exactly what ExpectedSane now states as an equality. *)
(*                                                                         *)
(* THE CHAIN.  'prev' is what stops the +1 being applied before the -1.     *)
(* Applied out of order the owner's expected count passes through 4 - a     *)
(* value the application was never in, since it went 3 -> 2 -> 3 - and      *)
(* ExpectedSane's equality is what says so.                                 *)
(*                                                                         *)
(*   gen 1  base 3, node 2's arrival cancelled   -> expected and issued 2    *)
(*   gen 2  base 3, -1 and +1 from node 2        -> expected and issued 3    *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |-> [n \in MCNodes |-> IF n = 0 THEN 0 ELSE 1] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

\* the plan the owner rebuilds after the alteration retires plan 1: node 2
\*  bypasses the tree from generation 1 on, so it gets no quota and node 3
\*  reports straight to the owner instead of through it
MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 3}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..2 |-> 3]

MCAlterOps  == { [node |-> 2, gen |-> 1, delta |-> -1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> NoTarget],
                 [node |-> 2, gen |-> 2, delta |-> 1,  ts |-> 2,
                  prev |-> 1, anc |-> 0, givento |-> NoTarget] }

=============================================================================
