------------------------------- MODULE MCOver -------------------------------
(***************************************************************************)
(* ISOLATES CASE 2 - the eager flush on over-arrival.                       *)
(*                                                                         *)
(* In MCDeviate the over-arrival and the outsider both happened in the same *)
(* generation, and the outsider's direct report made the OWNER announce a   *)
(* flush.  That flush reached the over-arriving node and let it proceed     *)
(* through the "already flushing" branch, so case 2 was never the thing     *)
(* actually under test.  Here the over-arrival is the only deviation.       *)
(*                                                                         *)
(*     0 = owner                                                            *)
(*     |- 1  quota 1                                                        *)
(*     |  |- 3  quota 2                                                     *)
(*     |- 2  quota 1   <- arrives 3 times in generation 2                   *)
(*     4 is in no plan and never arrives                                    *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3, 4}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        IF g = 1 THEN [n \in MCNodes |-> CASE n = 0 -> 0
                                           [] n = 1 -> 1
                                           [] n = 2 -> 1
                                           [] n = 3 -> 2
                                           [] n = 4 -> 0 ]
                 ELSE [n \in MCNodes |-> CASE n = 0 -> 0
                                           [] n = 1 -> 1
                                           [] n = 2 -> 3
                                           [] n = 3 -> 2
                                           [] n = 4 -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 3 -> [quota |-> 2, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

\* No alterations in this scenario: the base count is simply every arrival the
\*  pattern issues, so the model must reduce EXACTLY to the pre-alteration one.
MCBaseCount == [g \in 1..MCMaxGen |->
                  SumFn(MCNodes, [n \in MCNodes |-> MCPattern[g][n]])]
MCAlterOps  == {}

=============================================================================
