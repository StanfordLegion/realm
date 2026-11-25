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
