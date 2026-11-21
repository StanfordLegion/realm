\* Copyright 2026 Stanford University, NVIDIA Corporation
\* SPDX-License-Identifier: Apache-2.0
\*
\* Licensed under the Apache License, Version 2.0 (the "License");
\* you may not use this file except in compliance with the License.
\* You may obtain a copy of the License at
\*
\*     http://www.apache.org/licenses/LICENSE-2.0
\*
\* Unless required by applicable law or agreed to in writing, software
\* distributed under the License is distributed on an "AS IS" BASIS,
\* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
\* See the License for the specific language governing permissions and
\* limitations under the License.

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
