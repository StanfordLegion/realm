#!/bin/bash
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
# Slurm batch script: run TLC on the Realm deferred-allocation spec on sapling.
#
# Usage (submit FROM the tla/allocation directory so SLURM_SUBMIT_DIR is right):
#
#   cd <repo>/tla/allocation
#   sbatch sapling_tlc.sbatch Poison4        # config name as $1, default Big
#   sbatch sapling_tlc.sbatch Big
#   sbatch -t 48:00:00 --mem=256G sapling_tlc.sbatch Big   # override resources
#
# STORAGE LAYOUT: TLC's metadir (disk state queue + fingerprint set +
# checkpoints) lives on NODE-LOCAL /tmp (or $SLURM_TMPDIR when set), NOT on
# the shared filesystem.  The state queue is a huge, constantly-churning
# write load; putting it on shared storage is slow and can fill the
# filesystem for everyone (this killed all five round-2 jobs simultaneously
# when /scratch2 pressure hit).  To keep resumability, the newest COMPLETED
# checkpoint is periodically synced back to states/<config>/ in the submit
# directory (see SYNC below).
#
# Resuming an interrupted run: point RECOVER_FROM at the synced checkpoint
# directory on shared storage (it must contain vars.chkpt); the script stages
# it onto node-local disk and passes TLC the local path:
#   RECOVER_FROM=$PWD/states/<config>/<timestamp> \
#       sbatch sapling_tlc.sbatch <config>
# (Raw TLC_EXTRA="-recover <abs path>" still works but the path must then be
# reachable AND fast from the compute node — prefer RECOVER_FROM.)
#
# CHECKPOINT VALIDITY: recovery requires the BYTE-IDENTICAL .tla files that
# wrote the checkpoint.  Any spec edit — even one that provably does not
# change the next-state relation for this config's constants — shifts the
# string-intern table TLC serializes states against, and recovery fails with
# "ValueInputStream: Can not unpickle a value of kind <N>".  If the spec has
# changed since the checkpoint: delete states/<config> and start fresh.
#
# SYNC: OFF BY DEFAULT — hard rule: never put TLC data on /scratch on
# sapling.  With SYNC_CHECKPOINT=1 (explicit opt-in only), a background loop
# every CHECKPOINT_MIN/2 minutes rsyncs the newest quiescent checkpoint to
# states/<config>/ in the submit dir (atomic <ts>.partial rename, older
# checkpoints pruned) so an interrupted run can be resumed via RECOVER_FROM.
# With the default (off), an interrupted run restarts from zero — size the
# time limit so runs finish in one shot.
#
# ------------------------- SITE-SPECIFIC DEFAULTS ---------------------------
# CHECK THESE BEFORE FIRST SUBMISSION - they are guesses, not pinned to
# sapling's actual layout.  All are overridable on the sbatch command line
# (partition/time/cpus/mem) or via environment (JAVA, JAR, ALLOC_DIR, HEAP).
#SBATCH --job-name=tlc-deferred-alloc
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
# deliver TERM to the batch shell 10 min before the hard kill so the
# cleanup trap can finish - time-limit/scancel kills in taint rounds 1-2
# truncated rm -rf and left 13-127G stale dirs per node, which then
# tripped the MIN_AVAIL_G floor for every subsequent round
#SBATCH --signal=B:TERM@600
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
# ---------------------------------------------------------------------------

set -u

CONFIG=${1:-Big}

# locate the spec: $ALLOC_DIR override, else the directory we submitted from
ALLOC_DIR=${ALLOC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}

# module dispatch mirrors run.sh: v1 configs check MCDeferredAlloc;
# Dist*/Taint* configs check the distributed model
case $CONFIG in
    Dist*|Taint*) MODULE=MCDistDeferredAlloc ;;
    *)            MODULE=MCDeferredAlloc ;;
esac
if [ ! -f "$ALLOC_DIR/$MODULE.tla" ]; then
    echo "error: $MODULE.tla not found in $ALLOC_DIR" >&2
    echo "       submit from tla/allocation or set ALLOC_DIR=<path>" >&2
    exit 1
fi
if [ ! -f "$ALLOC_DIR/$CONFIG.cfg" ]; then
    echo "error: no such config: $ALLOC_DIR/$CONFIG.cfg" >&2
    exit 1
fi

# redirect output to a per-config log next to the catch-all slurm-%j.out
exec > "$ALLOC_DIR/slurm-${SLURM_JOB_ID:-nojob}-${CONFIG}.out" 2>&1

# ------------------------------ java discovery ------------------------------
# order: $JAVA env var, environment-modules java, PATH java
if [ -z "${JAVA:-}" ]; then
    if command -v module >/dev/null 2>&1; then
        module load java 2>/dev/null || module load openjdk 2>/dev/null || true
    fi
    JAVA=$(command -v java || true)
fi
if [ -z "${JAVA:-}" ] || ! "$JAVA" -version >/dev/null 2>&1; then
    echo "error: no working java found; set JAVA=<path> in the environment" >&2
    exit 1
fi

JAR=${JAR:-$ALLOC_DIR/../tla2tools.jar}
if [ ! -f "$JAR" ]; then
    echo "error: tla2tools.jar not found at $JAR (set JAR=<path>)" >&2
    exit 1
fi

# ------------------------------ resources -----------------------------------
WORKERS=${SLURM_CPUS_PER_TASK:-40}

# heap ~80% of the job's memory; SLURM_MEM_PER_NODE is in MB when --mem is used
if [ -z "${HEAP:-}" ]; then
    if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
        HEAP=$(( SLURM_MEM_PER_NODE * 80 / 100 ))m
    else
        HEAP=100g
    fi
fi

CHECKPOINT_MIN=${CHECKPOINT_MIN:-60}
# HARD RULE (Mike, 2026-08-27): never put TLC data on /scratch on sapling.
# Checkpoint sync-back to the submit dir therefore defaults OFF; opt in
# per-run with SYNC_CHECKPOINT=1 only if the single-checkpoint write to
# shared storage is judged acceptable for that run.
SYNC_CHECKPOINT=${SYNC_CHECKPOINT:-0}
TLC_EXTRA=${TLC_EXTRA:-}
RECOVER_FROM=${RECOVER_FROM:-}

# Deadlock checking is ON only for Smoke/EventLoop (they own the BUG-1
# deadlock class); every other config passes -deadlock so short deadlock
# traces don't preempt the deeper invariant/temporal hunts (EXPECTED.md).
case $CONFIG in
    Smoke|EventLoop) : ;;
    SmokeFixed|EventLoopFixed|EventLoopCapOnly|GCRipple|Inversion) : ;;
    TaintOpenLive|TaintXMemOpenLive) TLC_EXTRA="$TLC_EXTRA -deadlock" ;;  # temporal (MCTaintLiveSpec)
    Dist*|Taint*) : ;;  # distributed-model configs run deadlock check ON
                        # (Done disjunct makes clean completion a self-loop;
                        # a TLC deadlock = genuinely stuck state) - mirrors
                        # run.sh's dist flag table
    *) TLC_EXTRA="$TLC_EXTRA -deadlock" ;;
esac

# -gzip compresses the on-disk state queue (typically 5-10x): the queue is
# the dominant disk consumer at billion-state scale, and node-local disks
# are limited, so this is what makes big runs fit. Disable with TLC_GZIP=0.
# (Appended HERE, before the header echo, so the printed extra= is truthful.)
TLC_GZIP=${TLC_GZIP:-1}
[ "$TLC_GZIP" = "1" ] && TLC_EXTRA="$TLC_EXTRA -gzip"

# --------------------------- node-local storage ------------------------------
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp}
LOCAL_DIR="$LOCAL_ROOT/${USER}-tlc-${SLURM_JOB_ID:-nojob}-$CONFIG"
LOCAL_STATES="$LOCAL_DIR/states"
LOCAL_JTMP="$LOCAL_DIR/jtmp"
SHARED_STATES="$ALLOC_DIR/states/$CONFIG"
mkdir -p "$LOCAL_STATES" "$LOCAL_JTMP"
# shared-storage dir only exists if checkpoint sync is explicitly opted in
[ "$SYNC_CHECKPOINT" = "1" ] && mkdir -p "$SHARED_STATES"

echo "=== TLC on sapling: config=$CONFIG"
echo "    java=$JAVA  workers=$WORKERS  heap=$HEAP  extra='$TLC_EXTRA'"
echo "    spec=$ALLOC_DIR/$MODULE.tla"
if [ "$SYNC_CHECKPOINT" = "1" ]; then
    echo "    metadir=$LOCAL_STATES (node-local)  checkpoint-sync=$SHARED_STATES (every ~$(( CHECKPOINT_MIN / 2 ))min)"
else
    echo "    metadir=$LOCAL_STATES (node-local)  checkpoint-sync=OFF (nothing is written to shared storage)"
fi
df -h "$LOCAL_ROOT" | tail -1 | awk '{print "    node-local disk: size="$2" used="$3" avail="$4}'
AVAIL_G=$(df -BG --output=avail "$LOCAL_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
# HARD FLOOR (taint round 1 post-mortem): six of the 8 round-1 jobs died
# on 'No space left on device' - co-scheduled jobs shared a node's /tmp
# (XMem four started with 68/53/16/6.4G free on 183G nodes) or the node
# carried a 435G baseline of foreign data (733G c0001).
# Failing fast beats dying 36min-48h in with the bound short of the gate.
# Override per-run for small configs: MIN_AVAIL_G=<G> sbatch ...
MIN_AVAIL_G=${MIN_AVAIL_G:-300}
if [ -n "$AVAIL_G" ] && [ "$AVAIL_G" -lt "$MIN_AVAIL_G" ]; then
    echo "error: only ${AVAIL_G}G free on $LOCAL_ROOT (floor MIN_AVAIL_G=${MIN_AVAIL_G}G); refusing to start." >&2
    echo "       largest consumers on $LOCAL_ROOT:" >&2
    du -sh "$LOCAL_ROOT"/* 2>/dev/null | sort -h | tail -5 >&2
    echo "       fixes: submit with --exclusive (no shared /tmp); manually clean stale" >&2
    echo "       $LOCAL_ROOT/${USER}-tlc-* dirs from dead jobs; or target a big-disk node (-w)." >&2
    exit 75
fi
"$JAVA" -version 2>&1 | head -1

# stage a shared-storage checkpoint onto local disk for recovery
if [ -n "$RECOVER_FROM" ]; then
    if [ ! -f "$RECOVER_FROM/vars.chkpt" ]; then
        echo "error: RECOVER_FROM=$RECOVER_FROM has no vars.chkpt" >&2
        exit 1
    fi
    ts=$(basename "$RECOVER_FROM")
    echo "    staging checkpoint $ts to node-local disk..."
    rsync -a "$RECOVER_FROM/" "$LOCAL_STATES/$ts/" || { echo "error: checkpoint staging failed" >&2; exit 1; }
    TLC_EXTRA="$TLC_EXTRA -recover $LOCAL_STATES/$ts"
    echo "    staged; recovering from $LOCAL_STATES/$ts"
fi

# background checkpoint sync: newest quiescent checkpoint -> shared storage
sync_loop() {
    while sleep $(( CHECKPOINT_MIN * 30 )); do   # CHECKPOINT_MIN/2 in seconds
        newest=$(ls -dt "$LOCAL_STATES"/*/ 2>/dev/null | head -1)
        [ -n "$newest" ] || continue
        [ -f "$newest/vars.chkpt" ] || continue
        # quiescent = nothing written in the last minute (not mid-checkpoint)
        [ -z "$(find "$newest" -mmin -1 -print -quit 2>/dev/null)" ] || continue
        ts=$(basename "$newest")
        [ -d "$SHARED_STATES/$ts" ] && continue   # this checkpoint already synced
        rsync -a "$newest/" "$SHARED_STATES/$ts.partial/" || continue
        rm -rf "$SHARED_STATES/$ts" && mv "$SHARED_STATES/$ts.partial" "$SHARED_STATES/$ts"
        # prune older synced checkpoints, keep the newest one only
        for d in "$SHARED_STATES"/*/; do
            [ "$(basename "$d")" = "$ts" ] || rm -rf "$d"
        done
        echo "    [sync] checkpoint $ts synced to shared storage at $(date '+%F %T')"
    done
}
SYNC_PID=""
if [ "$SYNC_CHECKPOINT" = "1" ]; then
    sync_loop &
    SYNC_PID=$!
fi

cleanup() {
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    rm -rf "$LOCAL_DIR"
}
trap cleanup EXIT

# shellcheck disable=SC2086
"$JAVA" -XX:+UseParallelGC -Xmx"$HEAP" \
    -Djava.io.tmpdir="$LOCAL_JTMP" \
    -cp "$JAR" tlc2.TLC \
    -config "$ALLOC_DIR/$CONFIG.cfg" \
    -workers "$WORKERS" \
    -checkpoint "$CHECKPOINT_MIN" \
    -metadir "$LOCAL_STATES" \
    $TLC_EXTRA \
    "$ALLOC_DIR/$MODULE.tla" &
JAVA_PID=$!
# on early TERM (the --signal=B:TERM@600 pre-kill warning, or scancel):
# stop TLC first so it quits writing, then fall through to the normal
# tail (wait returns, EXIT trap removes LOCAL_DIR) inside the grace window
trap 'kill "$JAVA_PID" 2>/dev/null' TERM

# hung-JVM watchdog: TLC's disk-full RuntimeException (fp-set merge path)
# can kill the checking threads while worker threads hang, leaving a dead
# JVM holding the node - taint round 1 lost ~54 node-hours this way
# (78166: 5.7h, 78238: 8.5h, 78243: 40h). If the log reports disk-full
# and the JVM is still alive one poll later (10 min grace), end it. The
# clean-exit paths (StatePoolWriter failures, normal completion) are
# unaffected - the JVM is already gone when the poll fires.
JOB_LOG="$ALLOC_DIR/slurm-${SLURM_JOB_ID:-nojob}-${CONFIG}.out"
(
    seen=0
    while kill -0 "$JAVA_PID" 2>/dev/null; do
        sleep 600
        if grep -q "No space left on device" "$JOB_LOG" 2>/dev/null; then
            if [ "$seen" = "1" ]; then
                echo "=== watchdog: disk-full reported and JVM still alive 10min later; terminating TLC"
                kill "$JAVA_PID" 2>/dev/null
                sleep 60
                kill -9 "$JAVA_PID" 2>/dev/null
                break
            fi
            seen=1
        fi
    done
) &
WATCHDOG_PID=$!

wait "$JAVA_PID"
rc=$?
kill "$WATCHDOG_PID" 2>/dev/null

# final sync attempt: if TLC exited on its own (not killed), the last
# checkpoint is quiescent and worth keeping for a later resume
if [ "$SYNC_CHECKPOINT" = "1" ]; then
    newest=$(ls -dt "$LOCAL_STATES"/*/ 2>/dev/null | head -1)
    if [ -n "$newest" ] && [ -f "$newest/vars.chkpt" ]; then
        ts=$(basename "$newest")
        if [ ! -d "$SHARED_STATES/$ts" ]; then
            rsync -a "$newest/" "$SHARED_STATES/$ts.partial/" \
                && rm -rf "$SHARED_STATES/$ts" \
                && mv "$SHARED_STATES/$ts.partial" "$SHARED_STATES/$ts" \
                && echo "    [sync] final checkpoint $ts synced"
        fi
    fi
fi

echo "=== TLC exit code: $rc"
exit $rc
