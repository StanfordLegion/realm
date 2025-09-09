#!/bin/bash

# Copyright 2023 NVIDIA Corporation
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
#

# set -euo pipefail
set -e

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export REALM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/common.sh"

# Prepare output directory
function mk_output() {
    DATE="$(date +%Y/%m/%d)"
    mkdir -p "$REALM_DIR/$DATE"
    export HOST_OUT_DIR="$REALM_DIR/$DATE"
    echo "Redirecting stdout, stderr and logs to $HOST_OUT_DIR"
    export CMD_OUT_DIR="$HOST_OUT_DIR"
}

# Print usage if requested
if [[ $# -lt 2 || ! "$1" =~ ^(1/)?[0-9]+(:[0-9]+)?$ ]]; then
    echo "Usage: $(basename "${BASH_SOURCE[0]}") <num-nodes>[:<ranks-per-node>] <prog> <arg1> <arg2> ..."
    echo "Positional arguments:"
    echo "  <num-nodes> : positive integer or ratio < 1 (e.g. 1/4, for partial-node runs)"
    echo "  <ranks-per-node> : positive integer (default: 1)"
    echo "  <argI> : arguments to the program itself, Realm or Legion"
    echo "Arguments read from the environment:"
    echo "  DRY_RUN : don't submit the job, just print out the job scheduler command (default: 0)"
    echo "  JOB_NAME : the name of the job submitted, it must be set"
    echo "  PLATFORM : what machine we are executing on (default: auto-detected)"
    echo "  QUEUE : what queue/partition to submit the job to (default: depends on cluster)"
    echo "  TIMELIMIT : how much time to request for the job, in minutes (defaut: 60)"
    echo "  NUM_GPUS : number of gpus per node"
    echo "  ACCOUNT : account name"
    exit
fi

# Read arguments
if [ -z "$JOB_NAME" ]; then
    echo "\$JOB_NAME must be set"
    exit 1
fi
NODE_STR="$1"
if [[ "$1" == *":"* ]]; then
    RANKS_PER_NODE="${NODE_STR#*:}"
    NODE_STR="${NODE_STR%%:*}"
else
    RANKS_PER_NODE=1
fi
if [[ "$NODE_STR" =~ ^1/[0-9]+$ ]]; then
    NUM_NODES=1
    RATIO_OF_NODE_USED="$NODE_STR"
else
    NUM_NODES="$NODE_STR"
    RATIO_OF_NODE_USED=1
fi
NUM_RANKS=$(( $NUM_NODES * $RANKS_PER_NODE ))
export TIMELIMIT="${TIMELIMIT:-60}"
export DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == 1 ]]; then
    export NOWAIT=1
fi
export NOWAIT="${NOWAIT:-0}"

export BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
    echo "Please specify the BUILD_DIR"
    exit 1
fi

export NODE_LIST="${NODE_LIST:-}"

export ITERATIONS="${ITERATIONS:-20}"

export NUM_GPUS="${NUM_GPUS:-8}"

detect_platform

# remove the 1st arg from $@
shift

echo $@
export SCRIPT_NAME=$@

if [[ "$PLATFORM" == "other" ]]; then
    export NOWAIT=1
    
    mpirun_cmd=(mpirun -x LEGION_BACKTRACE=1 -n "$NUM_RANKS" --npernode "$RANKS_PER_NODE" --bind-to none "$@")
    echo "local run: ${mpirun_cmd[@]}"
    
    for i in {1..10}; do
        "${mpirun_cmd[@]}"
    done
elif [[ "$PLATFORM" == "computelab" || "$PLATFORM" == "oberon" ]]; then
    mk_output
    TIME="$(date +%H%M%S)"
    NUM_GPUS_PER_RANK=$(( $NUM_GPUS / $RANKS_PER_NODE ))

    mpirun_cmd=(mpirun -x LEGION_BACKTRACE=1 -x REALM_UCP_BOOTSTRAP_PLUGIN -x PATH -x LD_LIBRARY_PATH -x SCRIPT_NAME \
                -n "$NUM_RANKS" --npernode "$RANKS_PER_NODE" --bind-to none "$@")

    slurm_cmd=("$SCRIPT_DIR/job.slurm" "${mpirun_cmd[@]}")

    if [[ -z "${NODE_LIST}" ]]; then
        sbatch_cmd=(sbatch -p "$QUEUE" -t "$TIMELIMIT" --exclusive -N "$NUM_NODES" \
                    --gres=gpu:"$NUM_GPUS_PER_RANK" --ntasks-per-node="$RANKS_PER_NODE" \
                    -o "$HOST_OUT_DIR/$JOB_NAME-$TIME.txt")
    else
        sbatch_cmd=(sbatch -p "$QUEUE" -t "$TIMELIMIT" --exclusive -w "$NODE_LIST" -N "$NUM_NODES" \
                    --gres=gpu:"$NUM_GPUS_PER_RANK" --ntasks-per-node="$RANKS_PER_NODE" \
                    -o "$HOST_OUT_DIR/$JOB_NAME-$TIME.txt")
    fi

    if [[ -n "$ACCOUNT" ]]; then
        sbatch_cmd+=(-A "$ACCOUNT")
    fi

    sbatch_cmd+=("${slurm_cmd[@]}")
    submit "${sbatch_cmd[@]}"
fi

# Wait for batch job to start
if [[ "$NOWAIT" != 1 ]]; then
    echo "Waiting for job to start & piping stdout/stderr"
    sleep 1
    echo "Job started"
    # sed '/^Job finished/q' <( exec tail -n +0 -f "$HOST_OUT_DIR/out.txt" ) && kill $!
fi
