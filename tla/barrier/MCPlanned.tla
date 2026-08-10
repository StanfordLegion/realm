----------------------------- MODULE MCPlanned -----------------------------
(***************************************************************************)
(* A ONE-GENERATION model of the same topology as MCArrival, used for the   *)
(* liveness runs and for the transport-duplication stress runs.             *)
(*                                                                         *)
(* Liveness checking in TLC costs roughly 10-100x what safety costs, so the *)
(* liveness configuration drops the second generation.  What is lost is     *)
(* only the pipelining / trigger-ordering dimension, which is a safety      *)
(* property (TriggerInOrder) and is checked at G=2 by the safety runs.      *)
(*                                                                         *)
(* Topology (N = 4, R = 2, RootId = 0), relative indices:                   *)
(*                                                                         *)
(*         0            <- logical root, Kids(0) = {1,2}                    *)
(*        / \                                                              *)
(*       1   2          <- Kids(1) = {3}, Kids(2) = {}                      *)
(*       |                                                                 *)
(*       3                                                                 *)
(*                                                                         *)
(* The single generation is frozen onto a learned plan whose quotas are     *)
(*     node 2 -> 1,  node 3 -> 2   (subtree totals 1 and 2, root total 3)   *)
(* but the application REDISTRIBUTES:                                       *)
(*     node 2 arrives twice,  node 3 arrives once                           *)
(* The total is unchanged at 3, so this is exactly the "valid              *)
(* redistribution" of plan section 13.2: node 2 overshoots its quota (a     *)
(* detectable positive deviation) and node 3 is underfull (silent).         *)
(*                                                                         *)
(* Setting PlannedGens = {} in the .cfg turns the same scenario into a      *)
(* purely DYNAMIC generation (plan section 11), which is how the dynamic    *)
(* protocol is liveness-checked on its own.                                 *)
(***************************************************************************)
EXTENDS BarrierArrival

ArrDef ==
    [ g \in Gens |-> [ n \in Nodes |-> CASE n = 2 -> 2 [] n = 3 -> 1
                                         [] OTHER -> 0 ] ]

ExpectedDef == [ g \in Gens |-> 3 ]

QuotaDef       == [ n \in Nodes |-> CASE n = 2 -> 1 [] n = 3 -> 2 [] OTHER -> 0 ]
PlannedKidsDef == [ n \in Nodes |-> CASE n = 0 -> {1, 2} [] n = 1 -> {3} [] OTHER -> {} ]
PTotalDef      == [ n \in Nodes |-> CASE n = 0 -> 3 [] n = 1 -> 2
                                      [] n = 2 -> 1 [] OTHER -> 2 ]

============================================================================
