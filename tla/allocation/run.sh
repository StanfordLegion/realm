#!/bin/sh
# ---------------------------------------------------------------------------
# Run TLC on the Realm deferred-instance-allocation specification.
#
#   ./run.sh                 run the default local sweep (increasing cost)
#   ./run.sh Safety          run one configuration by name
#   ./run.sh sany            parse-check DeferredAlloc.tla / MCDeferredAlloc.tla
#
# Module is always MCDeferredAlloc; each <name>.cfg selects constants,
# invariants, and client shape (see DESIGN.md section 7).
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
        *)               echo "-deadlock" ;;
    esac
}

sany_check() {
    cd "$HERE" || exit 1   # SANY resolves EXTENDS relative to the cwd
    for m in DeferredAlloc MCDeferredAlloc; do
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
    echo "==========================================================="
    echo "=== $cfg   (module MCDeferredAlloc)"
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
        "$HERE/MCDeferredAlloc.tla"
    echo
}

if [ $# -ge 1 ]; then
    if [ "$1" = "sany" ]; then
        sany_check
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
