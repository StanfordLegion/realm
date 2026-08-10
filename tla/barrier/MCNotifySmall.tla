--------------------------- MODULE MCNotifySmall ---------------------------
(***************************************************************************)
(* Two remote nodes, three generations, one poisoned - small enough that a  *)
(* GUARD-REMOVAL mutation still terminates.  Removing a guard ENLARGES the  *)
(* reachable state space rather than shrinking it, so clearing a mutation   *)
(* as benign costs far more than catching one; the full MCNotify scenario   *)
(* is out of reach for that purpose.                                        *)
(*                                                                         *)
(* This is the scenario behind the battery's one deliberate negative        *)
(* control: removing Depart's waiting[n] = {} guard is NOT caught, because  *)
(* rule 6's re-subscribe recovery is the correctness rule and the guard is  *)
(* only an optimisation.  It has teeth - it catches the gap, no-recovery    *)
(* and discretionary-adds mutations - so its clean verdict is meaningful.   *)
(*                                                                         *)
(*   node 1  gens 1 and 3  - far-future waiter                             *)
(*   node 2  gens 2 and 3  - consults the poisoned generation              *)
(***************************************************************************)
EXTENDS BarrierNotify

MCNodes      == {1, 2}
MCMaxGen     == 3
MCPoisonGens == {2}
MCWaitPattern == [n \in MCNodes |-> CASE n = 1 -> {1, 3}
                                      [] OTHER -> {2, 3} ]

=============================================================================
