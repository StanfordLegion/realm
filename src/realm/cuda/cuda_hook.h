/* Copyright 2024 Stanford University, NVIDIA Corporation
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

 #ifndef REALM_CUDA_HOOK_H
 #define REALM_CUDA_HOOK_H
 
 #include "realm/cuda/cuda_internal.h"
 
 namespace Realm {
 
   namespace Cuda {
 
     class CudaHook {
     public:
       CudaHook();
       ~CudaHook();
 
       void start_task(GPUStream *current_stream);
       void end_task(GPUStream *current_task_stream, Processor::TaskFuncID task_id);
 
     private:
       CUpti_SubscriberHandle cupti_subscriber;
     };
 
   }; // namespace Cuda
 
 }; // namespace Realm
 
 #endif