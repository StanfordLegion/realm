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

set -e

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export REALM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PLATFORM="${PLATFORM:-}"
export USE_CUDA="${USE_CUDA:-}"
export CUDA_ARCH="${CUDA_ARCH:-}"
export USE_OPENMP="${USE_OPENMP:-}"
export NETWORK="${NETWORK:-}"
export BUILD_DIR="${BUILD_DIR:-}"
export ucx_ROOT="${ucx_ROOT:-}"
echo "Start building Legion using the following configurations:"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "REALM_DIR: ${REALM_DIR}"
echo "PLATFORM: ${PLATFORM}"
echo "USE_CUDA: ${USE_CUDA}"
echo "USE_OPENMP: ${USE_OPENMP}"
echo "NETWORK: ${NETWORK}"
echo "CONDUIT: ${CONDUIT}"
echo "BUILD_DIR: ${BUILD_DIR}"
echo "ucx_ROOT: ${ucx_ROOT}"

if [[ -e "$SCRIPT_DIR/$PLATFORM.sh" ]]; then
    source "$SCRIPT_DIR/$PLATFORM.sh"
fi

# setup cmake options
if [[ "$USE_CUDA" == 1 ]]; then
    cmake_options="${cmake_options} -DREALM_ENABLE_CUDA=ON -DLegion_CUDA_ARCH=${CUDA_ARCH}"
fi
if [[ "$USE_OPENMP" == 1 ]]; then
    cmake_options="${cmake_options} -DREALM_ENABLE_OPENMP=ON"
fi
if [[ "$NETWORK" == gasnetex ]]; then
    if [ -z ${CONDUIT+x} ]; then 
        echo "CONDUIT is unset"
        exit 1
    fi
    cmake_options="${cmake_options} -DREALM_ENABLE_GASNETEX=ON -DGASNET_CONDUIT=${CONDUIT}"
elif [[ "$NETWORK" == ucx ]]; then
    cmake_options="${cmake_options} -DREALM_ENABLE_UCX=ON"
fi
if [[ "$DEBUG" == 0 ]]; then
    cmake_options="${cmake_options} -DCMAKE_BUILD_TYPE=Release"
else
    cmake_options="${cmake_options} -DCMAKE_BUILD_TYPE=Debug"
fi

cmake_options="${cmake_options} -DREALM_BUILD_BENCHMARKS=ON -DREALM_BUILD_TESTS=ON"

# build legion
mkdir -p "$REALM_DIR/$BUILD_DIR"
pushd "$REALM_DIR/$BUILD_DIR"
echo "CMake Options: ${cmake_options}"
# make clean
cmake ../ ${cmake_options}
make -j VERBOSE=1
popd
