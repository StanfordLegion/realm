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

------------------------------ MODULE MCAncestor ------------------------------
(***************************************************************************)
(* CROSS-NODE ALTERATION ANCESTRY.  Two alterations, on two different       *)
(* nodes, where the second was invoked on a HANDLE RETURNED BY THE FIRST.   *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1                                             *)
(*                     |- 2  q1                                             *)
(*                        |- 3  q1                                          *)
(*                                                                         *)
(* Node 1 issues +1 (ts 1) and passes the returned handle to node 2, which  *)
(* issues a DESCENDANT +1 (ts 2, anc 1) on it.  Node 1's alteration is      *)
(* still on the wire to the owner while node 2's is issued and sent, and    *)
(* a set of messages reorders freely, so the owner may see ts 2 first.      *)
(*                                                                         *)
(* THE HAZARD.  'prev' cannot express this dependency: it is per node, and  *)
(* node 2 has issued nothing before.  If the owner applies ts 2 on arrival  *)
(* it moves expected from 3 to 4, a count the application was never in -    *)
(* the application went from 3 to 4 to 5 along node 1's branch - and every  *)
(* arrival gated on ts 2 is released against it.  Four arrivals then        *)
(* complete generation 1 whose true count is five: a barrier that triggered *)
(* too early, which is what event.h names as the failure mode.              *)
(*                                                                         *)
(* A POSITIVE SCENARIO, NOT A BISECTION.  Deleting the 'anc' conjunct of    *)
(* RecvAlter does NOT break this one, and the reason is worth recording:    *)
(* here the ancestor's node issues arrivals of its own, so its own arrival  *)
(* floor already gates the generation behind its own timestamp and the      *)
(* out-of-order apply cannot be cashed in.  What this scenario checks is    *)
(* that a descendant chain across two nodes WORKS - that the hold does not  *)
(* stall it.  The case where the hold is load-bearing is MCHandoff, where   *)
(* the ancestor's node makes no arrivals at all.                            *)
(*                                                                         *)
(*   gen 1  base 3 arrivals; +1 from node 1 and +1 from node 2, so expected *)
(*          and issued both reach 5                                         *)
(*   gen 2  the alterations are PERSISTENT, so both nodes owe one more here *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        [n \in MCNodes |-> CASE n = 1 -> 1
                             [] n = 2 -> 1
                             [] n = 3 -> 1
                             [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..2 |-> 3]

\* ts 2 names ts 1 as its ANCESTOR and has no 'prev' of its own: node 2 has
\*  issued no alteration before this one.  That is exactly the shape the
\*  per-node chain cannot describe.
MCAlterOps  == { [node |-> 1, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> NoTarget],
                 [node |-> 2, gen |-> 1, delta |-> 1, ts |-> 2,
                  prev |-> 0, anc |-> 1, givento |-> NoTarget] }

=============================================================================
