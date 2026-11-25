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

---------------------------- MODULE MCValveSlack ----------------------------
EXTENDS BarrierArrive_Valve

\* Minimal 5-arrival scenario (2 in gen 1, 3 in gen 2) exercising BOTH halves of R3-07:
\*  - Plan 1 (gen 1): 0 -> 3(q1); node 3 alters +1 (retires Plan 1) -> expected(1)=2.
\*  - Plan 2 (gen 2): 0 -> 1(q1) -> 2(q1) (SubQ=2, expected(2)=3, slack=1; alterer 3 excluded).
\*  - At gen 2: leaf 2 makes 0 arrivals, alterer 3 makes 2 bypassed arrivals (1 base + 1 alter):
\*      tsAcc(2) + SubQ(2) = 4 > expected(2) = 3!
\*    * If tsdirect(val=2) lands AFTER Plan 2 installs -> receipt check_bypass_slack_locked fires.
\*    * If tsdirect(val=2) runs ahead BEFORE Plan 2 installs (while Plan 1 is retired!) ->
\*      switch-time over_slack in Trigger's deviant audit fires.
MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> IF n = 3 THEN 1 ELSE 0]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 3 -> 1 [] OTHER -> 0] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..2 |-> CASE g = 1 -> 1 [] OTHER -> 2]
MCAlterOps  == { [node |-> 3, gen |-> 1, delta |-> 1, ts |-> 1,
                  prev |-> 0, anc |-> 0, givento |-> NoTarget] }
=============================================================================
