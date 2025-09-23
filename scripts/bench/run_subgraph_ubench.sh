#!/bin/bash

# Copyright 2023 NVIDIA Corporation
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
CPUS=16
CPU_STEP=2
SCRIPT_NAME="subgraph_ubench"

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
        # fan
        # export JOB_NAME=${SCRIPT_NAME}_n${n}_c${c}_fan
        # CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu ${c} -s 10 -size 10000 -t FAN"
        # echo "${JOB_NAME}, $CMD"
        # bash $CMD
        
        # chain
        export JOB_NAME=${SCRIPT_NAME}_n${n}_c${c}_chain
        CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:cpu ${c} -s 10 -size 10000 -t CHAIN"
        echo "${JOB_NAME}, $CMD"
        bash $CMD

        c=$(( $c * $CPU_STEP ))
    done 
    n=$(( $n * $NODE_STEP ))
done
