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

NODES=2
NODE_STEP=2
CPUS=4
CPU_STEP=2
SCRIPT_NAME="reservation_ubench"

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LEGION_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
    echo "Please specify the BUILD_DIR"
    exit 1
fi

n=2
while [ $n -le $NODES ]; do
    c=4
    while [ $c -le $CPUS ]; do
        for lpp in 16
        do
            # fan
            for tpppl in 100
            do
                echo "======== Running on $n nodes, $c cpus, tpppl ${tpppl} FAN ========"
                export JOB_NAME=${SCRIPT_NAME}_n${n}_c${c}_fan_tpppl${tpppl}
                CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:util 0 -ll:io 0 -ll:cpu ${c} -lpp ${lpp} -t FAN -tpppl ${tpppl} -s 60 -warmup 1"
                echo "${JOB_NAME}, $CMD"
                bash $CMD
            done
            # chain
            # for tpppl in 4
            # do
            #     echo "======== Running on $n nodes, $c cpus, tpppl ${tpppl} CHAIN ========"
            #     export JOB_NAME=${SCRIPT_NAME}_n${n}_c${c}_chain_tpppl${tpppl}
            #     CMD="${LEGION_DIR}/tools/bench/run.sh ${n} ${LEGION_DIR}/${BUILD_DIR}/bin/${SCRIPT_NAME} -ll:util 0 -ll:io 0 -ll:cpu ${c} -lpp ${lpp} -t CHAIN -tpppl ${tpppl} -s 10 -warmup 1"
            #     echo "${JOB_NAME}, $CMD"
            #     bash $CMD
            # done
        done
        c=$(( $c * $CPU_STEP ))
    done 
    n=$(( $n * $NODE_STEP ))
done
