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

set -euo pipefail

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REALM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/common.sh"

# Print usage if requested
if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [extra build args]"
    echo "Arguments read from the environment:"
    echo "  CONDUIT : GASNet conduit to use (if applicable) (default: auto-detected)"
    echo "  GPU_ARCH : CUDA architecture to build for (default: auto-detected)"
    echo "  NETWORK : Realm networking backend to use (default: auto-detected)"
    echo "  PLATFORM : what machine to build for -- provides defaults for other options"
    echo "             (default: auto-detected)"
    echo "  QUEUE : what queue/partition to submit the job to (default: depends on cluster)"
    echo "  USE_CUDA : include CUDA support (default: auto-detected)"
    echo "  USE_OPENMP : include OpenMP support (default: auto-detected)"
    echo "  DEBUG : build in debug/release mode (default: 0)"
    echo "  BUILD_DIR : build directory (default: build)"
    echo "  NUM_GPUS : number of gpus per node"
    exit
fi

ENV_CMD=""
export USE_CUDA="${USE_CUDA:-}"
if [[ "$USE_CUDA" == 1 ]]; then
    ENV_CMD="${ENV_CMD}USE_CUDA=1,"
    if [ ${GPU_ARCH+x} ]; then 
        ENV_CMD="${ENV_CMD}GPU_ARCH=${GPU_ARCH},"
    fi
fi
export USE_OPENMP="${USE_OPENMP:-}"
if [[ "$USE_OPENMP" == 1 ]]; then
    ENV_CMD="${ENV_CMD}USE_OPENMP=1,"
fi
if [ ${NETWORK+x} ]; then 
    ENV_CMD="${ENV_CMD}NETWORK=${NETWORK},"
    if [ ${CONDUIT+x} ]; then 
        ENV_CMD="${ENV_CMD}CONDUIT=${CONDUIT},"
    fi
    if [[ "$NETWORK" == ucx ]]; then
        export ucx_ROOT="${ucx_ROOT:-}"
        if [[ -z "${ucx_ROOT}" ]]; then
            echo "Please specify the root of ucx by ucx_ROOT"
            exit 1
        fi
        ENV_CMD="${ENV_CMD}ucx_ROOT=${ucx_ROOT},"
    fi
fi

export DEBUG="${DEBUG:-0}"
ENV_CMD="${ENV_CMD}DEBUG=$DEBUG,"

if [ ${BUILD_DIR+x} ]; then 
    ENV_CMD="${ENV_CMD}BUILD_DIR=${BUILD_DIR},"
else
    ENV_CMD="${ENV_CMD}BUILD_DIR=build,"
fi

export NUM_GPUS="${NUM_GPUS:-8}"

detect_platform

if [[ "$PLATFORM" == other ]]; then
    echo "Unknown PLATFORM, so will run locally..."
    # Remove the trailing comma from ENV_CMD
    ENV_CMD="${ENV_CMD%,}"
    # Replace commas with spaces to create proper environment variable definitions
    ENV_CMD="${ENV_CMD//,/ }"
    # Run the command with the environment variables
    echo $ENV_CMD
    env $ENV_CMD bash "$SCRIPT_DIR/cmake_build.sh"
elif [[ "$PLATFORM" == computelab || "$PLATFORM" == "oberon" ]]; then
    echo "Platform:${PLATFORM} is detected, using srun to build Legion on ${QUEUE}"
    ENV_CMD="${ENV_CMD}PLATFORM=${PLATFORM}"
    # this is for building, so one GPU is enough
    srun -p $QUEUE --exclusive --nodes 1 --gres=gpu:1 --export="$ENV_CMD" "$SCRIPT_DIR/cmake_build.sh"
fi
