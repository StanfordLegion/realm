#!/bin/sh
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

# ---------------------------------------------------------------------------
# Run TLC on the Realm deferred-instance-allocation specification.
#
#   ./run.sh                 run the default local sweep (increasing cost)
#   ./run.sh Safety          run one configuration by name
#   ./run.sh dist            run the DISTRIBUTED (BUG-8 / event-timestamp)
#                            matrix - Dist*.cfg against MCDistDeferredAlloc
#                            (see DIST-EXPECTED.md; deadlock checking stays
#                            ON for every dist config)
#   ./run.sh sany            parse-check the .tla modules
#
# Module is MCDeferredAlloc for the v1 configs and MCDistDeferredAlloc for
# the Dist* configs (selected by name prefix); each <name>.cfg selects
# constants, invariants, and client shape (DESIGN.md section 7 /
# DIST-EXPECTED.md).
#
# Configurations and expected outcomes (DESIGN.md sections 7-8):
#
#   Smoke      H=3, 2 insts            all INV_*, deadlock check   expect PASS
#                                      (BUG-1-shaped deadlock traces possible)
#   EventLoop  worked example, 2 insts deadlock check              expect FAIL (BUG-1)
#   Safety     H=4, 4 insts (2,1,1,2)  all INV_*/SAFETY_*, dlk     expect FAIL:
#                                      INV_NoReadyWhenNoPendingAllocs (BUG-6a)
#   Liveness   H=4, 3 insts, WF        LIVE_NoStuckAllocs          expect FAIL (BUG-1)
#
#   Poison4    4 insts + USER_POISON   hunts BUG-4esc/5/6b   sapling-targeted
#   Big        H=5-6, 4-5 insts        open hunt             sapling-targeted
#
# Poison4 and Big are excluded from the default sweep (projected > 1h locally);
# run them by name here at your own risk, or submit sapling_tlc.sbatch.
#
# Deadlock semantics: deadlock-check configs rely on TLC's built-in check plus
# the spec's Done self-loop (DESIGN.md section 6), so they must NOT pass
# -deadlock.  Temporal-liveness configs (Liveness) MUST pass -deadlock to
# suppress the check (clean termination would otherwise be reported).
#
# Requires tla2tools.jar (default: the copy in ../barrier/tools).
# Environment overrides: JAVA, JAR, WORKERS, HEAP, JTMP.
#
# TLC unpacks the TLA+ standard modules into java.io.tmpdir; this script
# points it at $JTMP (default ./jtmp).  Under a sandbox, set JTMP to a
# writable scratch directory.
# ---------------------------------------------------------------------------

HERE=$(cd "$(dirname "$0")" && pwd)

# prefer homebrew openjdk (the system /usr/bin/java stub has no runtime)
if [ -z "$JAVA" ]; then
    if [ -x /opt/homebrew/opt/openjdk/bin/java ]; then
        JAVA=/opt/homebrew/opt/openjdk/bin/java
    else
        JAVA=java
    fi
fi
JAR=${JAR:-$HERE/../barrier/tools/tla2tools.jar}
WORKERS=${WORKERS:-8}
HEAP=${HEAP:-4g}
JTMP=${JTMP:-$HERE/jtmp}

mkdir -p "$JTMP" "$HERE/states"

# Per-config extra TLC flags.  Deadlock checking stays ON only for
# Smoke/EventLoop (they own the BUG-1 deadlock class); every other config
# passes -deadlock so short deadlock traces don't preempt the deeper
# invariant/temporal hunts (EXPECTED.md).  NOTE: temporal configs
# (Liveness, LivenessNoCross) also require an UNSANDBOXED JVM - TLC's
# liveness checker binds a local RMI socket at startup.
extra_flags_for() {
    case $1 in
        Smoke|EventLoop) echo "" ;;
        # Fixed-model configs whose whole point is "the deadlock class is
        # gone" keep deadlock checking ON.  With the full bundle
        # (FIX_CAP+FIX_SWEEP+FIX_RPR) Inversion is GREEN deadlock-ON; the
        # historical two-toggle deadlock is kept as the BUG-5 witness in
        # traces/Inversion-bug5-deadlock.txt (see the cfg header).
        SmokeFixed|EventLoopFixed|EventLoopCapOnly|GCRipple|Inversion) echo "" ;;
        # Distributed (BUG-8) matrix: deadlock checking ON for ALL dist
        # and taint configs - the greens rely on the Done self-loop and the
        # expected-FAIL configs (DistBase, DistUEIllegal, DistRTUEIllegal,
        # TaintUEPoison-FALSE, TaintIllegalWait) exist to produce deadlock
        # counterexamples (DIST-EXPECTED.md).
        TaintOpenLive|TaintXMemOpenLive) echo "-deadlock" ;; # temporal
        # TaintXMemOpen*, TaintXMemOpen6, TaintOpenUE, TaintOpen3 are
        # SAPLING-BOUND (see cfg headers) - excluded from the local sweep,
        # runnable by name.
        TaintXMemOpen|TaintXMemOpen6|TaintXMemOpenRoot) echo "-gzip" ;;
        Dist*|Taint*)    echo "" ;;
        *)               echo "-deadlock" ;;
    esac
}

# v1 configs check MCDeferredAlloc; Dist* configs check MCDistDeferredAlloc.
module_for() {
    case $1 in
        Dist*|Taint*) echo "MCDistDeferredAlloc" ;;
        *)            echo "MCDeferredAlloc" ;;
    esac
}

sany_check() {
    cd "$HERE" || exit 1   # SANY resolves EXTENDS relative to the cwd
    mods="DeferredAlloc MCDeferredAlloc"
    # dist modules join the check once the spec module exists (the harness
    # cannot parse without it - see DIST-EXPECTED.md status note)
    if [ -f "$HERE/DistDeferredAlloc.tla" ]; then
        mods="$mods DistDeferredAlloc MCDistDeferredAlloc"
    fi
    for m in $mods; do
        echo "=== SANY $m.tla"
        "$JAVA" -Djava.io.tmpdir="$JTMP" -cp "$JAR" tla2sany.SANY "$m.tla" \
            || exit 1
    done
}

run_cfg() {
    cfg=$1
    if [ ! -f "$HERE/$cfg.cfg" ]; then
        echo "error: no such config: $HERE/$cfg.cfg" >&2
        exit 1
    fi
    case $cfg in
        Safety|Poison4|Big|SafetyFixed4|Poison4Fixed|BigFixed)
            echo "note: $cfg is sapling-targeted (see sapling_tlc.sbatch); running locally anyway." ;;
    esac
    mod=$(module_for "$cfg")
    echo "==========================================================="
    echo "=== $cfg   (module $mod)"
    echo "==========================================================="
    rm -rf "$HERE/states/$cfg"
    # shellcheck disable=SC2046
    "$JAVA" -XX:+UseParallelGC -Xmx"$HEAP" \
        -Djava.io.tmpdir="$JTMP" \
        -cp "$JAR" tlc2.TLC \
        -config "$HERE/$cfg.cfg" \
        -workers "$WORKERS" \
        -metadir "$HERE/states/$cfg" \
        $(extra_flags_for "$cfg") \
        "$HERE/$mod.tla"
    echo
}

if [ $# -ge 1 ]; then
    if [ "$1" = "sany" ]; then
        sany_check
        exit 0
    fi
    if [ "$1" = "dist" ]; then
        # Distributed (BUG-8 / event-timestamp) matrix, increasing cost
        # order; expectations in DIST-EXPECTED.md.  DistBase and
        # DistUEIllegal are EXPECTED to end in TLC deadlock reports (the
        # BUG-8 hole registration and the contract-violation demo);
        # TaintXMemBase and TaintPreRootBase are EXPECTED to violate
        # INV_NoFundingCycle (the F4 / F3 witnesses, DIST-EXPECTED.md).
        for c in DistLocal DistLocalUnion DistBase DistStamp DistStampOOB \
                 DistUELegal DistUEIllegal DistHandoff DistHandoffPB \
                 DistOpen \
                 DistRT DistRTOOB DistRTUEIllegal DistRTHandoff \
                 DistRTSameSrcRace DistRTOpen \
                 TaintBug8 TaintHold TaintHandoffRun TaintUEPoisonBase \
                 TaintUEPoison TaintUEGated TaintIllegalWait TaintUnion \
                 TaintOpen TaintOpenLive \
                 TaintXMemBase TaintXMem TaintPreRootBase TaintPreRoot \
                 TaintBug8Prefix TaintHandoffRunPrefix TaintUEGatedPrefix \
                 TaintUnionPrefix; do
            run_cfg "$c"
        done
        exit 0
    fi
    for c in "$@"; do run_cfg "$c"; done
else
    echo "note: Safety, Poison4, Big and their Fixed variants (SafetyFixed4,"
    echo "      Poison4Fixed, BigFixed) are sapling-targeted and excluded from this"
    echo "      sweep (see SAPLING_JOBS.md); run them by name or via"
    echo "      sapling_tlc.sbatch.  SafetyMini/Composite4 are the local"
    echo "      reproducers for BUG-6 and the BUG-6->BUG-4 composite; the"
    echo "      *Fixed/GCRipple/Inversion configs validate the CAP+SWEEP+RPR"
    echo "      fix bundle (all green, Inversion deadlock-ON included)."
    # increasing cost order; base model first, then the fix-validation matrix
    for c in Smoke EventLoop SafetyMini Composite4 Liveness LivenessNoCross \
             SmokeFixed EventLoopFixed EventLoopCapOnly GCRipple Inversion \
             Composite4Fixed SafetyMiniSweepOnly SafetyMiniFixed LivenessFixed; do
        run_cfg "$c"
    done
fi
