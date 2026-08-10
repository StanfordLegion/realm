------------------------------ MODULE MCDouble ------------------------------
(***************************************************************************)
(* The DOUBLE-COUNT in its minimal form: four nodes, two generations, one   *)
(* plan switch.  MCDeepSwitch exhibits it too but costs millions of states; *)
(* this costs thousands, which is what makes iterating on a fix possible.   *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 2  q1                                      *)
(*        |- 2  q1              |- 3  q1                                   *)
(*           |- 3  q1                                                      *)
(*                                                                         *)
(* Node 2 is RE-PARENTED from node 1 to the owner, and it has a child of    *)
(* its own so its cumulative total can grow AFTER the move:                 *)
(*                                                                         *)
(*   1. node 2 runs ahead and arrives on g2 while still on plan 1; its      *)
(*      quota is met so it reports val=1 to node 1                          *)
(*   2. node 1 folds that into its subtree total and reports it onward      *)
(*   3. the plan switches; node 2's parent becomes the owner                *)
(*   4. node 3 reports to node 2, so node 2's total rises to 2 and it       *)
(*      reports 2 - now down a different chain                              *)
(*                                                                         *)
(* The owner sums per SENDER, so node 2's own arrival is counted once       *)
(* inside node 1's stale total and again in node 2's own report.            *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 2 -> 1
                                           [] n = 3 -> 1 [] OTHER -> 0 ] ]

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
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
