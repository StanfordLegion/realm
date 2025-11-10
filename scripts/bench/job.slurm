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

set -u

echo "Job ID: $SLURM_JOB_ID"
echo "Submitted from: $(pwd)"
echo "Started on: $(date)"
echo "Running on: $SLURM_JOB_NODELIST"
if [[ -e "$SCRIPT_DIR/$PLATFORM.sh" ]]; then
    echo "source $SCRIPT_DIR/$PLATFORM.sh"
    source "$SCRIPT_DIR/$PLATFORM.sh"
fi
echo "Job cmd: $@"
echo "ITERATIONS: $ITERATIONS"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "REALM_UCP_BOOTSTRAP_PLUGIN: ${REALM_UCP_BOOTSTRAP_PLUGIN:-}"
echo "script=$SCRIPT_NAME"
echo "slurm_nodes=$SLURM_NNODES"
echo "slurm_tasks_per_node=$SLURM_NTASKS_PER_NODE"
for i in $(seq 1 $ITERATIONS)
    do
        "$@"
    done
echo "Job finished: $(date)"
