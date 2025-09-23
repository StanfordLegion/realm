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

function detect_platform {
    if [[ -n "${PLATFORM+x}" ]]; then
        return
    elif [[ "$(uname -n)" == *"r6515"* ]]; then
        export PLATFORM=computelab
        export QUEUE=v100-sxm2-16gb@cr+mp/dgx-1v@cr+mp/8gpu-80cpu-512gb
        export GPU_ARCH=70
    else
        export PLATFORM=other
    fi
}

function submit {
    echo -n "Submitted:"
    for TOK in "$@"; do printf " %q" "$TOK"; done
    echo
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "(dry run; no work was actually submitted)"
    else
        "$@"
    fi
}
