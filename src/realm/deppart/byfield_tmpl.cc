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

// per-dimension instantiator for byfield.cc

#undef REALM_TEMPLATES_ONLY
#include "./byfield.cc"

#ifndef INST_N1
  #error INST_N1 must be defined!
#endif
#ifndef INST_N2
  #error INST_N2 must be defined!
#endif

#define FOREACH_TT(__func__) \
  __func__(int,int) \
  __func__(int,unsigned) \
  __func__(int,long long) \
  __func__(unsigned,int) \
  __func__(unsigned,unsigned) \
  __func__(unsigned,long long) \
  __func__(long long,int) \
  __func__(long long,unsigned) \
  __func__(long long,long long)

namespace Realm {

#define N1 INST_N1
#define N2 INST_N2

#ifdef REALM_USE_CUDA
  #define GPU_BYFIELD_LINE(N, T, ...) template class GPUByFieldMicroOp<N,T,__VA_ARGS__>;
  #define DOIT_NT(N, T) \
      template void IndexSpace<N, T>::required_byfield_buffer_size(						     \
	const std::vector<DeppartEstimateInput<N,T>>&,							     \
	std::vector<DeppartBufferRequirements>&) const;

FOREACH_NT(DOIT_NT)

#else
  #define GPU_BYFIELD_LINE(N, T, ...) /* no CUDA */
#endif

#define DOIT(N,T,F) \
  template class ByFieldMicroOp<N,T,F>; \
  GPU_BYFIELD_LINE(N, T, F) \
  template class ByFieldOperation<N,T,F>; \
  template ByFieldMicroOp<N,T,F>::ByFieldMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,F> >&, \
							     const std::vector<F>&, \
							     std::vector<IndexSpace<N,T> >&, \
							     const ProfilingRequestSet &, \
							     Event) const;



  
FOREACH_NTF(DOIT)

#define ZP(N,T) Point<N,T>
#define ZR(N,T) Rect<N,T>
#define DOIT2(T1,T2) \
  DOIT(N1,T1,ZP(N2,T2))
  //  DOIT(N1,T1,ZR(N2,T2))

  FOREACH_TT(DOIT2)

};
