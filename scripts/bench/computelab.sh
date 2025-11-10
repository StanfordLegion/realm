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

echo $REALM_DIR
# only for ucx, but it doe not break other network modules if it is set
if [[ -z "${REALM_UCP_BOOTSTRAP_PLUGIN}" ]]; then
  export REALM_UCP_BOOTSTRAP_PLUGIN=$REALM_DIR/$BUILD_DIR/lib/realm_ucp_bootstrap_mpi.so}
fi

# CUDA Toolkit Path
if [[ -z "${CUDA_PATH}" ]]; then
  export CUDA_PATH="/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/"
fi

if [[ -z "${MPI_PATH}" ]]; then
  export MPI_PATH="/home/scratch.svc_compute_arch/release/mpi/openmpi/v4.1.4-ucx-1.13.1-cuda11.5"
fi

# Update PATH
export PATH="${MPI_PATH}/bin:${CUDA_PATH}/bin:${PATH:-}"

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${MPI_PATH}/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# Other environment variables
export CUDA_ARCH="${CUDA_ARCH:-70}"
export CONDUIT="${CONDUIT:-ibv}"
export GASNET_HOST_DETECT="hostname"
