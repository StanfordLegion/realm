------------------------------ MODULE MCStrand ------------------------------
(***************************************************************************)
(* The STRANDING that per-generation pinning introduces, minimally.         *)
(* MCDeepSwitch shows it at millions of states; this shows it at thousands. *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 3  q1                                      *)
(*        |- 2  q1                                                         *)
(*                                                                         *)
(* Nodes 1 AND 2 are dropped entirely by plan 2.                            *)
(*                                                                         *)
(*   1. node 2 runs ahead and arrives on g2 while still on plan 1; its      *)
(*      quota is met so it reports to node 1 and PINS that edge for g2      *)
(*   2. the plan switches; node 1 is invalidated, flushes what it holds     *)
(*      (nothing for g2 yet) and gets no replacement plan                   *)
(*   3. node 2's report reaches node 1 afterwards.  Node 1 is below its own *)
(*      quota, is no longer flushing because the switch cleared it, and is  *)
(*      no longer reachable from the owner - plan 2's tree goes elsewhere   *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 2 -> 1   \* runs ahead, then orphaned
                                           [] n = 3 -> 1 [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
