------------------------------ MODULE MCSeven ------------------------------
(***************************************************************************)
(* Seven nodes, depth 3, three generations, three plans - sized so the full *)
(* mutation battery is still affordable.                                    *)
(*                                                                         *)
(* The state space is driven by ARRIVAL INTERLEAVINGS, not by node count on *)
(* its own: every individual arrive() is a separate action, so total        *)
(* arrivals across all generations is the number that matters.  This        *)
(* scenario uses 12 where MCBig3 used 16, which is what makes it tractable  *)
(* at one more node than MCLarge.                                           *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)        plan 3 (gen 3)             *)
(*     0                     0                     0                        *)
(*     |- 1  q1              |- 2  q1              |- 1  q1                 *)
(*     |  |- 3  q2           |  |- 5  q1           |  |- 6  q1              *)
(*     |- 2  q1                                                             *)
(*        |- 4  q1                                                          *)
(*                                                                         *)
(*   g1  matches plan 1                       -> steady state               *)
(*   g2  DISAGREES with plan 2: node 2 over-arrives (case 2) and node 6     *)
(*       arrives while in no plan (case 3)                                  *)
(*   g3  matches plan 3, except node 3 runs ahead while still on plan 1,    *)
(*       where its quota of 2 keeps it silent - so reaching it needs an     *)
(*       invalidation across two pattern changes and two hops.              *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4, 5, 6}
MCOwner  == 0
MCMaxGen == 3

MCPattern ==
    [ g \in 1..3 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 2 -> 1
                                           [] n = 3 -> 2
                                           [] n = 4 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 2 -> 2     \* over-arrival (quota 1)
                                           [] n = 5 -> 1
                                           [] n = 6 -> 1     \* outsider
                                           [] OTHER -> 0 ]
          [] g = 3 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 6 -> 1
                                           [] n = 3 -> 1     \* runs ahead
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {4}]
          [] n = 3 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] n = 4 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {5}]
          [] n = 5 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan3 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {6}]
          [] n = 6 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 3
MCPlans     == <<MCPlan1, MCPlan2, MCPlan3>>
MCPlanStart == <<1, 2, 3>>

\* No alterations in this scenario: the base count is simply every arrival the
\*  pattern issues, so the model must reduce EXACTLY to the pre-alteration one.
MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
