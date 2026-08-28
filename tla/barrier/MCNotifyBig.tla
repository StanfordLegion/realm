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
