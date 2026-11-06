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

-------------------------------- MODULE MCGiven -------------------------------
(***************************************************************************)
(* A RANK THAT JOINS THE BARRIER BECAUSE OF THE ALTERATION.                 *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1          node 2 is NOT in the plan: it has  *)
(*                                       no base arrivals to build one from *)
(*                                                                         *)
(* This is the common shape of a positive alteration.  Node 1 raises the    *)
(* count by one because a NEW participant - node 2 - is going to arrive,    *)
(* and hands node 2 the handle that came back.  Node 2 alters nothing; it   *)
(* only ever arrives, on the handle it was given and on handles derived     *)
(* from it.                                                                 *)
(*                                                                         *)
(* Three generations, because the question is about the generations AFTER   *)
(* the one the alteration names.  The alteration is at generation 1, so     *)
(* generations 2 and 3 are the ones whose gate is under test.               *)
(*                                                                         *)
(*   every generation  node 1 arrives once (its own base arrival, gated by  *)
(*                     its arrival floor) and node 2 arrives once on the    *)
(*                     given handle.  expected and issued are both 2.       *)
(*                                                                         *)
(* THE SAFETY ARGUMENT UNDER TEST.  Node 1's base arrival is the reserved   *)
(* arrival the contract requires, and it is gated on the alteration, so     *)
(* generation 1 cannot complete until the owner has applied it.  Generations*)
(* trigger in watermark order, so 2 and 3 cannot trigger before 1 does.  If *)
(* that is sufficient, node 2's arrivals at generations 2 and 3 need no     *)
(* gate of their own and it can stay in the tree.                           *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2}
MCOwner  == 0
MCMaxGen == 3

\* node 2 has NO base arrivals - it exists in this barrier only because of
\*  the alteration, which is what makes it the new participant
MCPattern == [ g \in 1..3 |-> [n \in MCNodes |-> IF n = 1 THEN 1 ELSE 0] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..3 |-> 1]

\* node 1 raises the count because node 2 will arrive, and hands node 2 the
\*  handle that came back
MCAlterOps  == { [node |-> 1, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> 2] }

=============================================================================
