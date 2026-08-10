------------------------------ MODULE MCLarge ------------------------------
(***************************************************************************)
(* Six nodes, three generations, TWO pattern changes - so a node can be two *)
(* plans behind while still holding work.                                   *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)        plan 3 (gen 3)             *)
(*     0                     0                     0                        *)
(*     |- 1  q1              |- 2  q3              |- 1  q2                 *)
(*     |  |- 3  q2                                    |- 4  q1              *)
(*     |- 2  q1                                                             *)
(*                                                                         *)
(* Node 3 is the interesting one: it expects 2 under plan 1, contributes 2  *)
(* in generation 1, then runs ahead and issues ONE generation-3 arrival.     *)
(* Under plan 1's quota of 2 that is silent, and plans 2 and 3 drop node 3  *)
(* entirely - so it can only be handed over by an invalidation reaching it   *)
(* through node 1, across two successive pattern changes.                    *)
(*                                                                         *)
(* Node 5 never participates at all, and node 4 appears only in plan 3.      *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4, 5}
MCOwner  == 0
MCMaxGen == 3

MCPattern ==
    [ g \in 1..3 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 0 -> 0   \* matches plan 1
                                           [] n = 1 -> 1
                                           [] n = 2 -> 1
                                           [] n = 3 -> 2
                                           [] n = 4 -> 0
                                           [] n = 5 -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 0 -> 0   \* matches plan 2
                                           [] n = 1 -> 0
                                           [] n = 2 -> 3
                                           [] n = 3 -> 0
                                           [] n = 4 -> 0
                                           [] n = 5 -> 0 ]
          [] g = 3 -> [n \in MCNodes |-> CASE n = 0 -> 0   \* node 3 runs ahead
                                           [] n = 1 -> 2
                                           [] n = 2 -> 0
                                           [] n = 3 -> 1
                                           [] n = 4 -> 1
                                           [] n = 5 -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 3, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan3 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 2, inplan |-> TRUE, kids |-> {4}]
          [] n = 4 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
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
