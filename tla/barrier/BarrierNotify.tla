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

---------------------------- MODULE BarrierNotify ----------------------------
(***************************************************************************)
(* Realm scalable barriers - SUBSCRIPTION / NOTIFICATION protocol.          *)
(*                                                                         *)
(* Companion to BarrierArrive (the arrival protocol).  This half does not need  *)
(* to be precise: a node that is not notified pulls, and the owner answers. *)
(* The pull is the correctness guarantee; the broadcast set is a cache.     *)
(*                                                                         *)
(* THE PROTOCOL                                                             *)
(*                                                                         *)
(*  1. The owner keeps a SET of subscribers.  There is no learned tree: the *)
(*     multicast layer plans a fresh forwarding tree from the encoded set   *)
(*     on every send, so a set change can never leave a stale tree.         *)
(*  2. A notification is a DELTA - the newly triggered range and whichever  *)
(*     of those generations are poisoned - so its size never grows with     *)
(*     the barrier's poison history.  Old poison is delivered once, on the  *)
(*     SUBSCRIBE REPLY, which is point to point and can be keyed to what    *)
(*     the subscriber says it already knows.  A node that receives a delta  *)
(*     it cannot apply (a gap below m.prev) discards it and pulls.          *)
(*  3. MEMBERSHIP IS PUBLISHED, NEVER SELF-DECIDED.  The owner's published  *)
(*     set is authoritative, and a shrink is published to the PRE-SHRINK    *)
(*     set, so a node being removed always hears about it.                  *)
(*  4. ADDS ARE MANDATORY, REMOVALS ARE DISCRETIONARY.  Refusing an add     *)
(*     strands a waiter; refusing a removal only costs bandwidth.           *)
(*  5. A node removed while it still holds an outstanding waiter RE-        *)
(*     SUBSCRIBES AT ONCE.  Declining to depart while holding a waiter is   *)
(*     NOT sufficient on its own - a node can register a waiter after its   *)
(*     departure intent has been collected but before the owner shrinks.    *)
(*                                                                         *)
(* Nodes here are the REMOTE nodes; the owner is the watermark/subSet/      *)
(* setVer/wantOut variables themselves.                                     *)
(*                                                                         *)
(* DELIBERATELY ABSTRACTED: the idle threshold K, the tracked-round         *)
(* interval M, and the cost test that decides whether a shrink pays are all *)
(* replaced by NONDETERMINISM - Depart may fire on any eligible node and    *)
(* the owner may shrink by ANY subset of collected intents.  That is        *)
(* strictly more general than any tuning, which is the point: the           *)
(* constants must not be able to affect correctness.                        *)
(*                                                                         *)
(* NOT MODELLED: message loss (the transport is reliable), reduction        *)
(* barriers, node failure, and the multicast tree itself - the forwarding   *)
(* layer delivers to exactly the encoded set, so a notification is modelled *)
(* as one message per member.                                               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Nodes,        \* the REMOTE nodes (the owner is the owner-side variables)
    MaxGen,       \* generations are 1..MaxGen
    PoisonGens,   \* SUBSET Gens - which generations the application poisons
    WaitPattern   \* [Nodes -> SUBSET Gens] - which generations each node consults

Gens  == 1..MaxGen
Gens0 == 0..MaxGen

VARIABLES
    watermark,   \* owner: highest triggered generation
    subSet,      \* owner: SUBSET Nodes, the current subscriber set
    setVer,      \* owner: version, bumped on EVERY change to subSet
    wantOut,     \* owner: SUBSET Nodes, departure intents collected so far
    known,       \* [Nodes -> Gens0]        node's known watermark
    knownPois,   \* [Nodes -> SUBSET Gens]  node's known poisoned set
    member,      \* [Nodes -> {"NO","PENDING","YES"}]  node's belief about membership
    myVer,       \* [Nodes -> Nat]  highest setVer this node has applied
    waiting,     \* [Nodes -> SUBSET Gens]  generations this node still needs
    msgs         \* set of in-flight messages (a set, so delivery may reorder)

vars == << watermark, subSet, setVer, wantOut, known, knownPois,
           member, myVer, waiting, msgs >>

Msgs ==
      [ kind : {"notify"},    to : Nodes, wm : Gens, prev : Gens0,
                              pois : SUBSET Gens, inset : BOOLEAN, sv : Nat ]
  \cup [ kind : {"subscribe"}, from : Nodes, lk : Gens0 ]
  \cup [ kind : {"reply"},     to : Nodes, wm : Gens0, pois : SUBSET Gens,
                              sv : Nat ]

Init ==
    /\ watermark = 0
    /\ subSet    = {}
    /\ setVer    = 0
    /\ wantOut   = {}
    /\ known     = [n \in Nodes |-> 0]
    /\ knownPois = [n \in Nodes |-> {}]
    /\ member    = [n \in Nodes |-> "NO"]
    /\ myVer     = [n \in Nodes |-> 0]
    /\ waiting   = [n \in Nodes |-> {}]
    /\ msgs      = {}

(***************************************************************************)
(* ACTION 1 - the owner triggers the next generation (possibly poisoning    *)
(* it) and may shrink the set by any subset of the intents it has           *)
(* collected.  The notification goes to the PRE-SHRINK set, which is what   *)
(* guarantees a departing node hears about its own removal.                 *)
(***************************************************************************)
Trigger ==
    /\ watermark + 1 \in Gens
    /\ \E R \in SUBSET wantOut :
        LET g   == watermark + 1
            sv2 == IF (R \cap subSet) = {} THEN setVer ELSE setVer + 1
        IN  /\ watermark' = g
            /\ subSet'    = subSet \ R
            /\ setVer'    = sv2
            /\ wantOut'   = wantOut \ R
            /\ msgs' = msgs \cup
                 { [ kind |-> "notify", to |-> c, wm |-> g, prev |-> watermark,
                     pois |-> (IF g \in PoisonGens THEN {g} ELSE {}),
                     inset |-> (c \notin R), sv |-> sv2 ] : c \in subSet }
    /\ UNCHANGED << known, knownPois, member, myVer, waiting >>

(***************************************************************************)
(* ACTION 2 - local code consults the barrier: a blocking wait, an explicit *)
(* subscribe, or registering an EventWaiter for deferred work.  A node not  *)
(* covered by the published set pulls.                                      *)
(***************************************************************************)
Consult(n, g) ==
    /\ g \in WaitPattern[n]
    /\ g \notin waiting[n]
    /\ known[n] < g
    /\ waiting' = [waiting EXCEPT ![n] = @ \cup {g}]
    /\ wantOut' = wantOut \ {n}          \* consulting resets the idle counter
    /\ IF member[n] = "NO"
         THEN /\ msgs'   = msgs \cup
                            {[ kind |-> "subscribe", from |-> n, lk |-> known[n] ]}
              /\ member' = [member EXCEPT ![n] = "PENDING"]
         ELSE UNCHANGED << msgs, member >>
    /\ UNCHANGED << watermark, subSet, setVer, known, knownPois, myVer >>

(***************************************************************************)
(* ACTION 3 - a notification arrives.  It is a DELTA, so it can only be    *)
(* applied if this node has already seen everything below m.prev; otherwise *)
(* it is discarded and the node pulls.  Membership is version gated and is  *)
(* applied even on a gap.  A node removed while it still holds a waiter     *)
(* re-subscribes at once.                                                   *)
(***************************************************************************)
RecvNotify(m) ==
    /\ m \in msgs /\ m.kind = "notify"
    /\ LET \* A DELTA notification is gap sensitive: if this node has not seen
           \*  everything below m.prev it cannot apply the range, because it
           \*  would advance its watermark over generations whose poison status
           \*  it does not know.  Discard and pull - the reply is exact.
           gap   == m.prev > known[m.to]
           fresh == (~gap) /\ (m.wm > known[m.to])
           nk    == IF fresh THEN m.wm ELSE known[m.to]
           np    == IF fresh THEN knownPois[m.to] \cup m.pois ELSE knownPois[m.to]
           w2    == { g \in waiting[m.to] : g > nk }
           newv  == m.sv > myVer[m.to]
           \* membership is applied even on a gap - the message may be this
           \*  node's only notice of its own removal
           mem0  == IF newv THEN (IF m.inset THEN "YES" ELSE "NO")
                            ELSE member[m.to]
           resub == (mem0 = "NO") /\ (w2 # {})
           \* at most one outstanding pull per node: an in-flight subscribe
           \*  already carries an lk at or below what a new one would, so its
           \*  reply is a superset of what this pull would ask for
           pull  == (gap \/ resub)
                    /\ ~(\E x \in msgs : (x.kind = "subscribe") /\ (x.from = m.to))
       IN  /\ known'     = [known     EXCEPT ![m.to] = nk]
           /\ knownPois' = [knownPois EXCEPT ![m.to] = np]
           /\ waiting'   = [waiting   EXCEPT ![m.to] = w2]
           /\ myVer'     = [myVer     EXCEPT ![m.to] = IF newv THEN m.sv ELSE @]
           /\ member'    = [member    EXCEPT ![m.to] =
                              IF resub THEN "PENDING" ELSE mem0]
           /\ msgs' = (msgs \ {m}) \cup
                      (IF pull THEN {[ kind |-> "subscribe", from |-> m.to,
                                       lk |-> nk ]}
                               ELSE {})
    /\ UNCHANGED << watermark, subSet, setVer, wantOut >>

(***************************************************************************)
(* ACTION 4 - the owner receives a pull.  The add is MANDATORY, and the     *)
(* reply carries the current snapshot: that is what covers the race where   *)
(* the generation triggers while the subscribe is still on the wire.        *)
(***************************************************************************)
RecvSubscribe(m) ==
    /\ m \in msgs /\ m.kind = "subscribe"
    /\ LET sv2 == IF m.from \in subSet THEN setVer ELSE setVer + 1
       IN  /\ subSet'  = subSet \cup {m.from}
           /\ setVer'  = sv2
           /\ wantOut' = wantOut \ {m.from}
           /\ msgs' = (msgs \ {m}) \cup
                      {[ kind |-> "reply", to |-> m.from, wm |-> watermark,
                         pois |-> { g \in PoisonGens : (g > m.lk) /\ (g <= watermark) },
                         sv |-> sv2 ]}
    /\ UNCHANGED << watermark, known, knownPois, member, myVer, waiting >>

RecvReply(m) ==
    /\ m \in msgs /\ m.kind = "reply"
    /\ LET fresh == m.wm > known[m.to]
           nk    == IF fresh THEN m.wm  ELSE known[m.to]
           \* the reply is a DELTA keyed on what this node said it knew, so it
           \*  is unioned in - substituting it would drop older poison
           np    == IF fresh THEN knownPois[m.to] \cup m.pois ELSE knownPois[m.to]
           newv  == m.sv > myVer[m.to]
       IN  /\ known'     = [known     EXCEPT ![m.to] = nk]
           /\ knownPois' = [knownPois EXCEPT ![m.to] = np]
           /\ waiting'   = [waiting   EXCEPT ![m.to] = { g \in @ : g > nk }]
           /\ myVer'     = [myVer     EXCEPT ![m.to] = IF newv THEN m.sv ELSE @]
           /\ member'    = [member    EXCEPT ![m.to] =
                              IF newv THEN "YES" ELSE member[m.to]]
           /\ msgs'      = msgs \ {m}
    /\ UNCHANGED << watermark, subSet, setVer, wantOut >>

(***************************************************************************)
(* ACTION 5 - a node signals that it wants out.  Nondeterministic: this     *)
(* stands in for the whole idle-counter / tracked-round mechanism, so no    *)
(* choice of K or M can be wrong.  The guard waiting[n] = {} is the         *)
(* PREVENTION half of rule 5 - an optimisation, not the safety rule.        *)
(***************************************************************************)
Depart(n) ==
    /\ n \in subSet
    /\ member[n] = "YES"
    /\ waiting[n] = {}
    /\ n \notin wantOut
    /\ wantOut' = wantOut \cup {n}
    /\ UNCHANGED << watermark, subSet, setVer, known, knownPois,
                    member, myVer, waiting, msgs >>

(***************************************************************************)
(* A settled run stutters, so "finished" is never mistaken for "stuck".     *)
(***************************************************************************)
Done ==
    /\ watermark = MaxGen
    /\ \A n \in Nodes : waiting[n] = {}
    /\ msgs = {}
    /\ UNCHANGED vars

Next ==
    \/ Done
    \/ Trigger
    \/ \E n \in Nodes, g \in Gens : Consult(n, g)
    \/ \E m \in msgs : RecvNotify(m)
    \/ \E m \in msgs : RecvSubscribe(m)
    \/ \E m \in msgs : RecvReply(m)
    \/ \E n \in Nodes : Depart(n)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* SAFETY                                                                   *)
(***************************************************************************)
TypeOK == /\ watermark \in Gens0
          /\ known  \in [Nodes -> Gens0]
          /\ member \in [Nodes -> {"NO","PENDING","YES"}]
          /\ \A n \in Nodes : myVer[n] <= setVer
          /\ msgs \subseteq Msgs

\* A node is never told a generation triggered before it did.
NeverOverstate == \A n \in Nodes : known[n] <= watermark

\* A notification is a complete snapshot, so a node's poison knowledge is
\*  always exactly the truth up to its own watermark - even after missing
\*  arbitrarily many notifications.  A delta-based notification breaks this.
PoisonAccurate ==
    \A n \in Nodes : knownPois[n] = { g \in PoisonGens : g <= known[n] }

\* THE liveness-critical one, encoded as safety: no node may hold an
\*  outstanding waiter with nothing that will ever satisfy it.  In-flight
\*  messages count - a node being removed is briefly outside the set while
\*  its own removal notice is still on the wire.
NoStranded ==
    \A n \in Nodes :
        (waiting[n] # {}) =>
            \/ n \in subSet
            \/ (\E m \in msgs : (m.kind = "notify" \/ m.kind = "reply") /\ m.to = n)
            \/ (\E m \in msgs : m.kind = "subscribe" /\ m.from = n)

\* Membership is published, never self-decided: a node believing it is
\*  covered is either in the set or has its correction already on the wire.
MembershipPublished ==
    \A n \in Nodes :
        (member[n] = "YES") =>
            \/ n \in subSet
            \/ (\E m \in msgs : m.kind = "notify" /\ m.to = n /\ ~m.inset
                                /\ m.sv > myVer[n])

=============================================================================
