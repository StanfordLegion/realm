----------------------------- MODULE MCNotifyBig -----------------------------
(***************************************************************************)
(* Four remote nodes, three generations - one more node than MCNotify.         *)
(* them the LAST, so a poison snapshot has to survive a set change.         *)
(*                                                                         *)
(* NOTE: this scenario DOES NOT CONVERGE - it was still climbing past 47M   *)
(* distinct states with a growing queue and was abandoned.  Kept as the      *)
(* record of where the ceiling is: three remote nodes.                       *)
(*                                                                         *)
(*   node 1  gens 1 and 3  - far-future waiter across two generations       *)
(*   node 2  gens 2 and 3  - consults on both poisoned and clean gens       *)
(*   node 3  gen 4 only    - joins late, after the set has already shrunk   *)
(*   node 4  gens 1 and 2  - departs early, then comes back                 *)
(***************************************************************************)
EXTENDS BarrierNotify

MCNodes      == {1, 2, 3, 4}
MCMaxGen     == 3
MCPoisonGens == {2}
MCWaitPattern == [n \in MCNodes |-> CASE n = 1 -> {1, 3}
                                      [] n = 2 -> {2, 3}
                                      [] n = 3 -> {3}
                                      [] OTHER -> {1, 2} ]

=============================================================================
