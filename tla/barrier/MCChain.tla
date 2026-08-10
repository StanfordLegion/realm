------------------------------- MODULE MCChain -------------------------------
(***************************************************************************)
(* A THREE-DEEP RELAY CHAIN - the shape the child-wait is supposed to need. *)
(* Neither MCLarge nor MCSeven has a relay below a relay: in both, every    *)
(* plan is owner -> relay -> leaf.  Here node 2 is a relay whose parent is  *)
(* also a relay, so a premature forward at node 2 has somewhere to go wrong.*)
(*                                                                         *)
(*     0                                                                   *)
(*     |- 1  q1                                                            *)
(*        |- 2  q1                                                         *)
(*           |- 3  q1                                                      *)
(*                                                                         *)
(*   g1  matches the plan exactly - the child-wait is the only gate         *)
(*   g2  deviates twice: node 3 over-arrives (case 2) at the BOTTOM of the  *)
(*       chain, and node 4 arrives while in no plan (case 3)                *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 2   \* over-arrival, deepest
                                           [] n = 4 -> 1   \* outsider
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
