#!/bin/bash

# Copyright 2025 NVIDIA Corporation
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
#

NODES=8
NODE_STEP=2
CPUS=24
CPU_STEP=2
GPUS=8
GPU_STEP=1
SCRIPT_NAME="task_ubench"

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LEGION_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
    echo "Please specify the BUILD_DIR"
    exit 1
fi

n=1
while [ $n -le $NODES ]; do
    c=1
    while [ $c -le $CPUS ]; do
        echo "======== Running on $n nodes, $c cpus ========"
        if [ $n -eq 1 ]; then
            export JOB_NAME=${SCRIPT_NAME}_n1_c${c}
            CMD="${LEGION_DIR}/tools/bench/run.sh 1 ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu ${c} -s 40 -n 100000 -ll:networks none -warmup 5 -ll:util 0 -ll:io 0 -ll:bgwork 1"
        else
            export JOB_NAME=${SCRIPT_NAME}_n${n}_c${c}_remote
            CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu ${c} -s 40 -n 100000 -remote 1 -warmup 5 -ll:util 0 -ll:io 0 -ll:bgwork 1"
        fi
        echo "${JOB_NAME}, $CMD"
        bash $CMD
        if [ $c -eq 1 ]; then
          c=2
        else
          c=$(( $c + $CPU_STEP ))
        fi
    done
    g=1
    while [ $g -le $GPUS ]; do
        echo "======== Running on $n nodes, $g gpus ========"
        if [ $n -eq 1 ]; then
            export JOB_NAME=${SCRIPT_NAME}_n1_g${g}
            CMD="${LEGION_DIR}/tools/bench/run.sh 1 ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu 1 -ll:gpu ${g} -s 40 -n 100000 -ll:networks none -warmup 5 -gpu -ll:util 0 -ll:io 0 -ll:bgwork 1"
        else
            export JOB_NAME=${SCRIPT_NAME}_n${n}_g${g}_remote
            CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu 1 -ll:gpu ${g} -s 40 -n 100000 -remote 1 -warmup 5 -gpu -ll:util 0 -ll:io 0 -ll:bgwork 1"
        fi
        echo "${JOB_NAME}, $CMD"
        bash $CMD
        if [ $g -eq 1 ]; then
          g=2
        else
          g=$(( $g + $GPU_STEP ))
        fi
    done
    n=$(( $n * $NODE_STEP ))
done
