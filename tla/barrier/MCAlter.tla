------------------------------- MODULE MCAlter -------------------------------
(***************************************************************************)
(* alter_arrival_count.  Four nodes, two generations, one plan, one         *)
(* alteration - deliberately small, because what is being checked is a      *)
(* COUNTING hazard rather than a routing one and it shows up at minimum     *)
(* size.                                                                    *)
(*                                                                         *)
(*   plan 1            0                                                    *)
(*                     |- 1  q1                                             *)
(*                     |- 2  q1                                             *)
(*                        |- 3  q1                                          *)
(*                                                                         *)
(* Node 2 is a RELAY, not a leaf: it alters its own count while still       *)
(* having to forward node 3's reports.  That is the case where bypassing    *)
(* the tree with a timestamped arrival could strand a child.                *)
(*                                                                         *)
(*   gen 1  base 3 arrivals; node 2 alters +1 (ts 1), which CREATES a fourth *)
(*          arrival it must then issue - so expected and issued both reach 4 *)
(*   gen 2  base 3 arrivals; the alteration is PERSISTENT, so node 2 owes one*)
(*          more here too and expected is 4                                  *)
(*                                                                         *)
(* The hazard: if the owner counts node 2's arrivals before applying the    *)
(* alteration, it triggers generation 1 at 3 instead of 4 - a barrier that  *)
(* triggered too early, which is what event.h names as the failure mode.    *)
(***************************************************************************)
EXTENDS BarrierArrive

MCNodes  == {0, 1, 2, 3}
MCOwner  == 0
MCMaxGen == 2

MCPattern ==
    [ g \in 1..2 |->
        CASE g = 1 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 2 -> 1     \* base; +1 comes from the alteration
                                           [] n = 3 -> 1
                                           [] OTHER -> 0 ]
          [] g = 2 -> [n \in MCNodes |-> CASE n = 1 -> 1
                                           [] n = 2 -> 1
                                           [] n = 3 -> 1
                                           [] OTHER -> 0 ] ]

NoPlan == [quota |-> 0, inplan |-> FALSE, kids |-> {}]

MCPlan1 ==
    [n \in MCNodes |->
        CASE n = 0 -> [quota |-> 0, inplan |-> TRUE, kids |-> {1, 2}]
          [] n = 1 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] n = 2 -> [quota |-> 1, inplan |-> TRUE, kids |-> {3}]
          [] n = 3 -> [quota |-> 1, inplan |-> TRUE, kids |-> {}]
          [] OTHER  -> NoPlan ]

MCNumPlans  == 1
MCPlans     == <<MCPlan1>>
MCPlanStart == <<1>>

\* base + persistent delta must equal what the pattern issues:
\*   gen 1: 3 + 1 = 4   gen 2: 2 + 1 = 3
MCBaseCount == [g \in 1..2 |-> 3]
MCAlterOps  == { [node |-> 2, gen |-> 1, delta |-> 1, ts |-> 1] }

=============================================================================
