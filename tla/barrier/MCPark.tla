------------------------------- MODULE MCPark -------------------------------
(***************************************************************************)
(* Why the DEFERRAL exists, isolated.  A node that installs a new plan      *)
(* before its invalidation arrives forwards that invalidation down the NEW  *)
(* kid list - so an old-plan descendant never receives it.  Only a          *)
(* descendant holding BELOW QUOTA exposes that: one whose quota is met      *)
(* reports on its own and needs no invalidation.  MCStrand2's descendants   *)
(* all meet their quotas, which is why it cannot catch the no-deferral and  *)
(* forget-before-forward mutations.  Here node 2 under-fills its quota and  *)
(* an outsider provides the offset.                                         *)
(*                                                                         *)
(*   plan 1 (gen 1)        plan 2 (gen 2)                                  *)
(*     0                     0                                             *)
(*     |- 1  q1              |- 1  q1                                      *)
(*        |- 2  q2              |- 3  q1                                   *)
(*                                                                         *)
(*   g1  matches plan 1 (node 2 arrives twice)                             *)
(*   g2  node 2 runs ahead with ONE arrival (quota 2 - silent), node 3     *)
(*       arrives as plan-2 member, node 1 arrives once.                    *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1 [] n = 2 -> 2
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 2 -> 1   \* below its plan-1 quota
                                           [] n = 3 -> 1
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {2}]
          [] n = 2 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCPlan2 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 2
MCPlans     == <<MCPlan1, MCPlan2>>
MCPlanStart == <<1, 2>>

MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
