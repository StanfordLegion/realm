/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HIP_CUDA_H
#define HIP_CUDA_H

#include <hip/hip_runtime_api.h>

// types
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaSurfaceObject_t hipSurfaceObject_t

// functions
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree // hipFreeHost is deprecated
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc // hipMallocHost is deprecated
#define cudaMemAdvise hipMemAdvise
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy

// enum values
#define cudaBoundaryModeClamp hipBoundaryModeClamp
#define cudaBoundaryModeTrap hipBoundaryModeTrap
#define cudaBoundaryModeZero hipBoundaryModeZero
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSuccess hipSuccess

#endif
