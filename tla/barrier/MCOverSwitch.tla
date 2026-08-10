---------------------------- MODULE MCOverSwitch ----------------------------
(***************************************************************************)
(* OVER-ARRIVAL combined with a PLAN SWITCH - the one structural            *)
(* combination that lived only in the scenarios that outgrew this machine   *)
(* (MCLarge/MCLate/MCDeepSwitch on the final spec).  MCStale-sized.         *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 1  q1                                      *)
(*        |- 2  q1                                                         *)
(*                                                                         *)
(*   g1  matches plan 1                                                    *)
(*   g2  node 1 OVER-ARRIVES its plan-2 quota (2 vs 1), and node 2 - kept   *)
(*       by no plan - runs ahead into g2 while still holding plan 1.        *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 2   \* over-arrival (quota 1)
                                           [] n = 2 -> 1   \* runs ahead, then dropped
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
