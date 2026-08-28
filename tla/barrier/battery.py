#!/usr/bin/env python3
# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The MUTATION BATTERY for the scalable-barrier specifications.
#
# Both protocol documents end with the same instruction: "if you change any of
# these rules, re-run the battery."  This is the battery.  Each row disables
# one protocol rule and model-checks a scenario CHOSEN TO CATCH exactly that
# mutation - the pairing is load-bearing, not cosmetic: this project produced
# two false verification results from mutations paired with scenarios that
# lacked the structure to catch them (see ARRIVAL_PROTOCOL.md section 7, the
# masking note).
#
# Verdicts:
#   OK  caught      - the scenario rejects the mutated spec, as expected
#   OK  benign      - a deliberate negative control passed, as expected
#   XX  ...         - a mismatch: either a rule has silently become redundant
#                     (investigate before celebrating - see the child-wait and
#                     stale-edge histories, which went opposite ways) or a
#                     scenario has lost its teeth
#   EDIT-FAIL       - a mutation pattern no longer matches the spec exactly
#                     once; the battery REFUSES to run rather than silently
#                     testing the wrong site (this exact failure produced a
#                     false result once)
#
# Usage:  python3 battery.py [arrival|alter|notify|all]
# Needs:  java on PATH or at JAVA below; tools/tla2tools.jar (see README.md).
# Time:   ~15-30 min for 'all'.  Rows that catch stop at the violation and are
#         fast; NOT-CAUGHT rows and negative controls explore the whole state
#         space.  TLC scratch can reach tens of GB - it is confined to a
#         temp dir and removed per row.
#
# NEVER point this at the canonical .tla files in place: every row runs on a
# COPY in a scratch directory.  A killed run must not be able to leave a
# mutated spec on disk (that also happened once).

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
JAVA = os.environ.get("TLA_JAVA", "/opt/homebrew/opt/openjdk/bin/java")
if not os.path.exists(JAVA):
    JAVA = "java"
JAR = os.path.join(HERE, "tools", "tla2tools.jar")

# (name, spec, config, module, expect, [(old, new), ...])
#   expect: "CATCH" or "BENIGN"
ARRIVAL = [
 ("no deferral: park condition disabled", "Park.cfg", "MCPark", "CATCH",
  [("         ELSE IF curPlan[m.to].inplan /\\ (invalEpoch[m.to] < myEpoch[m.to])",
    "         ELSE IF FALSE")]),
 ("forget the child list BEFORE forwarding the invalidation", "Park.cfg", "MCPark", "CATCH",
  [("""                                 \\cup { [ kind |-> "invalidate", from |-> m.to,
                                          to |-> c, epoch |-> m.epoch ] : c \\in kids }""", "")]),
 ("accept stale reports (running total may go down)", "Double.cfg", "MCDouble", "CATCH",
  [("    /\\ m.val > childAcc[m.to][m.from][m.gen]\n", "")]),
 # SUBSUMED BY THE OWNER VALVE (BarrierArrive rule 6): under quota-gated
 #  forwarding the mask probe found NO scenario that strands without the
 #  case-3 flush signal - the owner's non-kid/value tests recover every one.
 #  The signal stays in the implementation regardless: its flush traffic is
 #  the plan-learning evidence (ARRIVAL_PROTOCOL.md section 11).
 ("case 3 sends its count but NO flush signal (subsumed)", "Stale.cfg", "MCStale", "BENIGN",
  [("""                /\\ msgs' = msgs \\cup Send(n, g, sub)
                           \\cup {[ kind |-> "flush", from |-> n, to |-> Owner, gen |-> g ]}""",
    """                /\\ msgs' = msgs \\cup Send(n, g, sub)""")]),
 ("no eager flush on over-arrival", "Over.cfg", "MCOver", "CATCH",
  [("                /\\ lt > curPlan[n].quota                            \\* case 2: over-arrival",
    "                /\\ FALSE")]),
 ("no pinning: reports follow the current plan", "Double.cfg", "MCDouble", "CATCH",
  [("TargetOf(n, g) == IF reportTo[n][g] # NoTarget THEN reportTo[n][g] ELSE ParentOf(n)",
    "TargetOf(n, g) == ParentOf(n)")]),
 ("no live guard: parked plan installed by its own retirement", "Strand2.cfg", "MCStrand2", "CATCH",
  [("                      live == dk > m.epoch", "                      live == dk > 0")]),
 ("no install guard: newplan installed after its retirement was seen", "Strand2.cfg", "MCStrand2", "CATCH",
  [("    /\\ IF (myEpoch[m.to] >= m.epoch) \\/ (invalEpoch[m.to] >= m.epoch)",
    "    /\\ IF myEpoch[m.to] >= m.epoch")]),
 # REPOINTED under quota-gated forwarding: the owner valve rescues MCStale's
 #  strand, but five other scenarios still catch this one.
 ("no planless-outsider", "Double.cfg", "MCDouble", "CATCH",
  [("""                                            ELSE [curPlan EXCEPT ![m.to] =
                                                    [quota |-> 0, inplan |-> FALSE,
                                                     kids |-> {}]]""",
    """                                            ELSE curPlan""")]),
 # SUBSUMED BY THE OWNER VALVE, same verdict and same caveat as the
 #  immediate case-3 signal above: retained for plan learning.
 ("no retroactive case 3 (subsumed)", "Stale.cfg", "MCStale", "BENIGN",
  [("""                                 \\cup (IF ~live /\\ (m.to # Owner)
                                         THEN { [ kind |-> "flush", from |-> m.to,
                                                  to |-> Owner, gen |-> g ] :
                                                  g \\in { h \\in Gens :
                                                            ~triggered[h]
                                                            /\\ localTotal[m.to][h] > 0 } }
                                         ELSE {})""", "")]),
 ("case 3 loses priority to the flushing flag", "Stale.cfg", "MCStale", "CATCH",
  [("""             \\/ /\\ ~curPlan[n].inplan                                \\* case 3""",
    """             \\/ /\\ ~flushing[n][g] /\\ ~curPlan[n].inplan            \\* case 3""")]),
 ("a plan install clears flush (the race)", "Stale.cfg", "MCStale", "CATCH",
  [("""                             \\* a plan install NEVER clears flush: a flushed
                             \\*  generation stays eager until it triggers (see
                             \\*  RecvInvalidate).  Planned mode is what governs
                             \\*  the generations that are NOT flushed.
                             /\\ UNCHANGED flushing""",
    """                             /\\ flushing' = [flushing EXCEPT ![m.to] =
                                               [g \\in Gens |-> IF ~triggered[g] THEN FALSE
                                                                        ELSE flushing[m.to][g]]]""")]),
 ("no stale-edge forwarding", "Double.cfg", "MCDouble", "CATCH",
  [("                      \\/ (m.from \\notin KidsOf(m.to))", "")]),
 # SUBSUMED, kept as a documented negative result (ARRIVAL_PROTOCOL rule 6):
 #  the rule-10 machinery covers what invalidate-time all-generation flushing
 #  used to.  If this row ever starts CATCHING, rule 6's honesty note is stale.
 ("flush only the switch generation at invalidate (subsumed)", "Strand2.cfg", "MCStrand2", "BENIGN",
  [("""                                           IF ~triggered[g] /\\ SubtreeKnown(m.to, g) > 0
                                             THEN TRUE""",
    """                                           IF (g = watermark + 1) /\\ ~triggered[g]
                                                /\\ SubtreeKnown(m.to, g) > 0
                                             THEN TRUE""")]),
]

# QUOTA-GATED FORWARDING (BarrierArrive rules 1 and 6).  Each row removes one
#  of the gate's mechanisms; the MCGate* scenarios were each built to strand
#  without exactly one of them (the iteration record is SCALE_TEST_PLAN.md).
#  MCGatePark catches two different mechanisms through its two interleavings:
#  counts arriving before the parked install need the re-fan, counts arriving
#  after it need the stale-edge signal.
GATE = [
 ("audit: switch-time value/orphan audit removed", "GateMove.cfg", "MCGateMove", "CATCH",
  [("""                                     /\\ \\/ flushing[Owner][h]
                                        \\/ \\E s \\in Nodes \\ {Owner} :
                                             \\/ childAcc[Owner][s][h] > SubQ(k, s)
                                             \\/ /\\ childAcc[Owner][s][h] > 0
                                                /\\ s \\notin Plans[k][Owner].kids }""",
    """                                     /\\ flushing[Owner][h] }""")]),
 ("valve: receipt-time value-exceeds test removed", "GateOver.cfg", "MCGateOver", "CATCH",
  [("""OwnerDeviant(m) == /\\ m.to = Owner
                   /\\ \\/ m.from \\notin KidsOf(Owner)
                      \\/ m.val > SubQ(myEpoch[Owner], m.from)""",
    """OwnerDeviant(m) == /\\ m.to = Owner
                   /\\ m.from \\notin KidsOf(Owner)""")]),
 ("valve: receipt-time non-kid test removed", "GateDemote.cfg", "MCGateDemote", "CATCH",
  [("""OwnerDeviant(m) == /\\ m.to = Owner
                   /\\ \\/ m.from \\notin KidsOf(Owner)
                      \\/ m.val > SubQ(myEpoch[Owner], m.from)""",
    """OwnerDeviant(m) == /\\ m.to = Owner
                   /\\ m.val > SubQ(myEpoch[Owner], m.from)""")]),
 ("audit: orphaned-sender branch removed", "GateDemote.cfg", "MCGateDemote", "CATCH",
  [("""                                             \\/ /\\ childAcc[Owner][s][h] > 0
                                                /\\ s \\notin Plans[k][Owner].kids }""",
    """                                             }""")]),
 ("re-fan: newplan install does not re-announce flush", "GateDemote.cfg", "MCGateDemote", "CATCH",
  [("""                                        \\cup UNION { { [ kind |-> "flush", from |-> m.to,
                                                         to |-> c, gen |-> h ] :
                                                         c \\in Plans[m.epoch][m.to].kids } :
                                                       h \\in { x \\in Gens :
                                                                 ~triggered[x]
                                                                 /\\ flushing[m.to][x] } }
""", "")]),
 ("re-fan: parked install does not re-announce flush", "GatePark.cfg", "MCGatePark", "CATCH",
  [("""                                              \\cup
                                              UNION { { [ kind |-> "flush", from |-> m.to,
                                                          to |-> c, gen |-> x ] :
                                                          c \\in Plans[dk][m.to].kids } :
                                                        x \\in { h \\in Gens :
                                                                  ~triggered[h]
                                                                  /\\ (flushing[m.to][h]
                                                                       \\/ SubtreeKnown(m.to, h) > 0) } }
""", "")]),
 ("stale-edge receipt sends no owner flush signal", "GatePark.cfg", "MCGatePark", "CATCH",
  [("""                      \\cup (IF (m.from \\notin KidsOf(m.to)) /\\ (m.to # Owner)
                              THEN {[ kind |-> "flush", from |-> m.to, to |-> Owner,
                                      gen |-> m.gen ]} ELSE {})
""", "")]),
 ("gate: closes on exact equality (>= becomes =)", "GateJump.cfg", "MCGateJump", "CATCH",
  [("""                /\\ \\/ /\\ sub >= SubQCur(n)                          \\* case 1: subtree done""",
    """                /\\ \\/ /\\ sub = SubQCur(n)                           \\* case 1: subtree done"""),
   ("""                   \\/ /\\ sub < SubQCur(n)""",
    """                   \\/ /\\ ~( sub = SubQCur(n) )"""),
   ("""                         /\\ sub2 >= SubQCur(m.to)""",
    """                         /\\ sub2 = SubQCur(m.to)""")]),
 # the gate itself is an OPTIMIZATION: removing it reverts to the previously
 #  verified eager design, so its removal is documented-benign.
 ("gate removed entirely (reverts to verified eager design)", "GateMove.cfg", "MCGateMove", "BENIGN",
  [("""                /\\ \\/ /\\ sub >= SubQCur(n)                          \\* case 1: subtree done""",
    """                /\\ \\/ /\\ TRUE                                       \\* eager: no gate"""),
   ("""                   \\/ /\\ sub < SubQCur(n)""",
    """                   \\/ /\\ FALSE"""),
   ("""                         /\\ sub2 >= SubQCur(m.to)""",
    """                         /\\ localTotal[m.to][m.gen] = curPlan[m.to].quota""")]),
]

ALTER = [
 ("owner counts a ts arrival WITHOUT applying its alteration", "Alter.cfg", "MCAlter", "CATCH",
  [("    /\\ m.ts \\in appliedTs\n    /\\ m.val > tsAcc[m.from][m.gen]\n",
    "    /\\ m.val > tsAcc[m.from][m.gen]\n")]),
 ("timestamped arrivals aggregate through the TREE (ts erased)", "Alter.cfg", "MCAlter", "CATCH",
  [("          /\\ myTs[n][g] > 0\n", "          /\\ FALSE\n"),
   ("       \\/ /\\ myTs[n][g] = 0\n", "       \\/ /\\ TRUE\n")]),
 ("altering node does NOT enter eager flush (stops relaying)", "Alter.cfg", "MCAlter", "CATCH",
  [("""       IN  /\\ flushing' = [flushing EXCEPT ![a.node] =
                             [g \\in Gens |-> IF g \\in aff THEN TRUE
                                                          ELSE flushing[a.node][g]]]""",
    """       IN  /\\ flushing' = flushing""")]),
 ("no reserved-arrival contract guard", "Alter.cfg", "MCAlter", "CATCH",
  [("    /\\ unissued[a.node][a.gen] > 0\n", "")]),
]

NOTIFY = [
 ("apply a delta notification DESPITE a gap", "Notify.cfg", "MCNotify", "CATCH",
  [("           gap   == m.prev > known[m.to]", "           gap   == FALSE")]),
 ("no recovery: removed node with a live waiter does NOT re-subscribe", "Notify.cfg", "MCNotify", "CATCH",
  [('           resub == (mem0 = "NO") /\\ (w2 # {})', "           resub == FALSE")]),
 ("no version gate: stale notify may resurrect membership", "Notify.cfg", "MCNotify", "CATCH",
  [('''           mem0  == IF newv THEN (IF m.inset THEN "YES" ELSE "NO")
                            ELSE member[m.to]''',
    '''           mem0  == IF m.inset THEN "YES" ELSE "NO"''')]),
 ("shrink published to the POST-shrink set", "Notify.cfg", "MCNotify", "CATCH",
  [("inset |-> (c \\notin R), sv |-> sv2 ] : c \\in subSet }",
    "inset |-> (c \\notin R), sv |-> sv2 ] : c \\in (subSet \\ R) }")]),
 ("adds are discretionary (owner may refuse a subscribe)", "Notify.cfg", "MCNotify", "CATCH",
  [("""    /\\ LET sv2 == IF m.from \\in subSet THEN setVer ELSE setVer + 1
       IN  /\\ subSet'  = subSet \\cup {m.from}""",
    """    /\\ \\E addit \\in BOOLEAN :
       LET sv2 == IF (m.from \\in subSet) \\/ ~addit THEN setVer ELSE setVer + 1
       IN  /\\ subSet'  = IF addit THEN subSet \\cup {m.from} ELSE subSet""")]),
 ("subscribe reply carries no watermark", "Notify.cfg", "MCNotify", "CATCH",
  [('{[ kind |-> "reply", to |-> m.from, wm |-> watermark,',
    '{[ kind |-> "reply", to |-> m.from, wm |-> 0,')]),
 # anchored on the RecvReply comment: RecvNotify contains an identically-worded
 #  merge line, and an unanchored replace once silently mutated the wrong site
 ("delta reply SUBSTITUTED rather than merged", "Notify.cfg", "MCNotify", "CATCH",
  [("""\\*  is unioned in - substituting it would drop older poison
           np    == IF fresh THEN knownPois[m.to] \\cup m.pois ELSE knownPois[m.to]""",
    """\\*  is unioned in - substituting it would drop older poison
           np    == IF fresh THEN m.pois ELSE knownPois[m.to]""")]),
 # NEGATIVE CONTROL: the depart guard is an optimisation; rule 6's recovery is
 #  the correctness rule.  Runs on the SMALL scenario - a guard removal grows
 #  the state space, and MCNotify cannot finish it.
 ("may depart while holding a waiter (control)", "Small.cfg", "MCNotifySmall", "BENIGN",
  [("    /\\ waiting[n] = {}\n", "")]),
]

SUITES = {
    "arrival": ("BarrierArrive.tla", ARRIVAL),
    "gate": ("BarrierArrive.tla", GATE),
    "alter": ("BarrierArrive.tla", ALTER),
    "notify": ("BarrierNotify.tla", NOTIFY),
}


def classify(out):
    # NOTE the 'Error:' prefix: grepping for the bare 'Invariant ... is
    #  violated' misses it, which once degraded every verdict to a fallback.
    if "Error: Invariant" in out:
        return "caught: " + out.split("Error: Invariant")[1].split("is violated")[0].strip()
    if "Deadlock reached" in out:
        return "caught: deadlock"
    if "No error has been found" in out:
        return "not caught"
    return "ERROR/UNPARSED"


def run_row(specfile, name, cfg, module, expect, edits):
    spec = open(os.path.join(HERE, specfile)).read()
    for old, new in edits:
        n = spec.count(old)
        if n != 1:
            return ("EDIT-FAIL", "pattern matches %d sites (need exactly 1)" % n)
        spec = spec.replace(old, new, 1)
    with tempfile.TemporaryDirectory(prefix="battery_") as d:
        os.makedirs(os.path.join(d, "jtmp"))
        shutil.copy(os.path.join(HERE, cfg), d)
        shutil.copy(os.path.join(HERE, module + ".tla"), d)
        open(os.path.join(d, specfile), "w").write(spec)
        r = subprocess.run(
            [JAVA, "-Xmx8g", "-XX:+UseParallelGC",
             "-Djava.io.tmpdir=" + os.path.join(d, "jtmp"),
             "-cp", JAR, "tlc2.TLC", "-workers", "8", "-config", cfg,
             module + ".tla"],
            capture_output=True, text=True, cwd=d, timeout=7200)
        v = classify(r.stdout)
    if expect == "CATCH":
        return ("OK" if v.startswith("caught") else "XX", v)
    return ("OK" if v == "not caught" else "XX", v + " (expected benign)")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(SUITES) if which == "all" else [which]
    failures = 0
    for suite in names:
        specfile, rows = SUITES[suite]
        print("== %s (%s) ==" % (suite, specfile))
        for name, cfg, module, expect, edits in rows:
            tag, verdict = run_row(specfile, name, cfg, module, expect, edits)
            if tag != "OK":
                failures += 1
            print("  %-3s %-13s %-36s %s" % (tag, module, verdict, name), flush=True)
    print("battery:", "PASS" if failures == 0 else "%d FAILURES" % failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
