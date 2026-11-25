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

----------------------------- MODULE MCPlan4Tiny -----------------------------
EXTENDS BarrierArrive

\* Minimal 3-node, 4-plan, 4-arrival scenario reaching ALL R2-09 and R3-06
\* branches on node 1:
\*   1. R3-06 live/defer: invalEpoch(0) < m.epoch(2) < myEpoch(3) < deferEpoch(4)
\*   2. R2-09 inval keep: myEpoch(3) > m.epoch(2)
\*   3. R2-09 park keep:  m.epoch(3) <= deferEpoch(4) while holding plan 2
MCNodes  == {0, 1, 2}
MCOwner  == 0
MCMaxGen == 4

MCPattern ==
    [ g \in 1..4 |->
        CASE g = 1 -> [n \in MCNodes |-> IF n = 0 THEN 1 ELSE 0]
          [] g = 2 -> [n \in MCNodes |-> IF n = 0 THEN 1 ELSE 0]
          [] g = 3 -> [n \in MCNodes |-> IF n = 0 THEN 1 ELSE 0]
          [] g = 4 -> [n \in MCNodes |-> IF n = 2 THEN 1 ELSE 0] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 1, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan3 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 1, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan4 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 1, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 4
MCPlans     == <<MCPlan1, MCPlan2, MCPlan3, MCPlan4>>
MCPlanStart == <<1, 2, 3, 4>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
