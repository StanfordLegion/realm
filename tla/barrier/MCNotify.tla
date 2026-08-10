------------------------------ MODULE MCNotify ------------------------------
(***************************************************************************)
(* Baseline scenario for the subscription/notification protocol.            *)
(*                                                                         *)
(* Three remote nodes, three generations.  As in the arrival specs, the     *)
(* PATTERN is scripted and only the INTERLEAVINGS are nondeterministic -    *)
(* leaving the consultation pattern free explodes the state space without   *)
(* covering anything the invariants can distinguish.                        *)
(*                                                                         *)
(*   node 1  consults gen 1, and gen 3 - a FAR-FUTURE waiter, which is what *)
(*           makes a node look idle while it is still interested            *)
(*   node 2  consults gen 2 (the poisoned generation)                       *)
(*   node 3  consults gen 3 only                                            *)
(*                                                                         *)
(* Still nondeterministic, and deliberately so: when each node consults,    *)
(* when a node signals departure, which subset of collected intents the     *)
(* owner actually removes, and every message delivery order.  That covers   *)
(* any choice of K, M, or shrink policy.                                    *)
(***************************************************************************)
EXTENDS BarrierNotify

MCNodes      == {1, 2, 3}
MCMaxGen     == 3
MCPoisonGens == {2}
MCWaitPattern == [n \in MCNodes |-> CASE n = 1 -> {1, 3}
                                      [] n = 2 -> {2}
                                      [] OTHER -> {3} ]

=============================================================================
