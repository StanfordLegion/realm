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

------------------------------- MODULE MCGiven2 -------------------------------
(***************************************************************************)
(* REPEATED GROWTH - two alterations at DIFFERENT generations, each handing *)
(* its handle to a different new participant.                               *)
(*                                                                         *)
(* The adversarial version of MCGiven, built to break the claim that the    *)
(* reserved arrival at an alteration's own generation is enough.  With ONE  *)
(* alteration that claim holds trivially: generations trigger in watermark  *)
(* order, so the gate at the alteration's generation holds every later one  *)
(* behind it.  With TWO, the later alteration names a LATER generation, and *)
(* by then the first recipient's handle is - if the timestamp does not      *)
(* survive advance_barrier - ungated.                                       *)
(*                                                                         *)
(*   node 1  base 1/gen; alters +1 at gen 1 (ts 1), handle to node 2        *)
(*   node 4  base 1/gen; alters +1 at gen 2 (ts 2), handle to node 3        *)
(*   node 2  no base arrivals; arrives on node 1's handle, gens 1-3         *)
(*   node 3  no base arrivals; arrives on node 4's handle, gens 2-3         *)
(*                                                                         *)
(*   gen 1  expected and issued 3                                           *)
(*   gen 2  expected and issued 4                                           *)
(*   gen 3  expected and issued 4                                           *)
(*                                                                         *)
(* THE TWO ALTERATIONS COME FROM DIFFERENT NODES ON PURPOSE.  'tsSent' is   *)
(* one cumulative counter per (node, generation), which is sound only while *)
(* every bypassed arrival a node makes for a generation carries the SAME    *)
(* gate.  A node that alters TWICE over the same generation has two - myTs  *)
(* moves from the first timestamp to the second - and the two then share a  *)
(* counter, so a message whose gate is open carries arrivals whose gates    *)
(* are shut.  The first draft of this scenario had node 1 doing both        *)
(* alterations and violated TriggerCorrect in the UNMUTATED spec, which is  *)
(* how the limitation was found.  It is latent in Arrive for any scenario   *)
(* with one node altering twice; the implementation does not share it,      *)
(* because it keys ts_issued and the owner's streams per (sender, gate).    *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4}
MCOwner  == 0
MCMaxGen == 3

MCPattern ==
    [ g \in 1..3 |-> [n \in MCNodes |-> IF (n = 1) \/ (n = 4) THEN 1 ELSE 0] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 4}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 4 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..3 |-> 2]

MCAlterOps  == { [node |-> 1, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> 2],
                 [node |-> 4, gen |-> 2, delta |-> 1, ts |-> 2,
                  prev |-> 0, anc |-> 0, givento |-> 3] }

=============================================================================
