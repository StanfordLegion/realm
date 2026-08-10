---------------------------- MODULE MCDeepSwitch ----------------------------
(***************************************************************************)
(* The only scenario with BOTH a relay below a relay AND two successive     *)
(* pattern changes.  MCLarge has two switches but is owner -> relay -> leaf;*)
(* MCChain has the depth but one plan; MCSeven had neither the depth nor,   *)
(* once the child-wait was removed, affordability.                          *)
(*                                                                         *)
(*   plan 1 (gens 1-2)     plan 2 (gen 3)      plan 3 (gen 4)              *)
(*     0                     0                   0                         *)
(*     |- 1  q1              |- 1  q1            |- 2  q1                  *)
(*        |- 2  q1              |- 4  q1            |- 3  q1               *)
(*           |- 3  q1                                                      *)
(*                                                                         *)
(*   g1, g2  match plan 1 - the three-deep chain in steady state            *)
(*   g3      matches plan 2                                                 *)
(*   g4      matches plan 3 EXCEPT node 4 arrives while in no plan, and it  *)
(*           may run ahead into g4 while still holding plan 1 - two plans   *)
(*           behind, which is what MCSeven existed to cover                 *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4}
MCOwner  == 0
MCMaxGen == 4

MCPattern ==
    [ g \in 1..4 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ]
          [] g = 3 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 4 -> 1 [] OTHER -> 0 ]
          [] g = 4 -> [n \in MCNodes |-> CASE n = 2 -> 1
                                           [] n = 3 -> 1
                                           [] n = 4 -> 1   \* outsider, may run ahead
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {4}]
          [] n = 4 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan3 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 3
MCPlans     == <<MCPlan1, MCPlan2, MCPlan3>>
MCPlanStart == <<1, 3, 4>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
