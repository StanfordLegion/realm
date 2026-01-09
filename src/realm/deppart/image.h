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

// image operations for Realm dependent partitioning

#ifndef REALM_DEPPART_IMAGE_H
#define REALM_DEPPART_IMAGE_H

#include "realm/deppart/partitions.h"
#include "realm/deppart/rectlist.h"

namespace Realm {
    template<int N, typename T, int N2, typename T2>
    class ImageMicroOp : public PartitioningMicroOp {
    public:
        static const int DIM = N;
        typedef T IDXTYPE;
        static const int DIM2 = N2;
        typedef T2 IDXTYPE2;

        ImageMicroOp(IndexSpace<N, T> _parent_space, IndexSpace<N2, T2> _inst_space,
                     RegionInstance _inst, size_t _field_offset, bool _is_ranged);

        virtual ~ImageMicroOp(void);

        void add_sparsity_output(IndexSpace<N2, T2> _source, SparsityMap<N, T> _sparsity);

        void add_sparsity_output_with_difference(IndexSpace<N2, T2> _source,
                                                 IndexSpace<N, T> _diff_rhs,
                                                 SparsityMap<N, T> _sparsity);

        void add_sparsity_output_with_intersection(IndexSpace<N2, T2> _source,
                                                   IndexSpace<N, T> _diff_rhs,
                                                   SparsityMap<N, T> _sparsity);

        void add_approx_output(int index, PartitioningOperation *op);

        virtual void execute(void);

        void dispatch(PartitioningOperation *op, bool inline_ok);

    protected:
        friend struct RemoteMicroOpMessage<ImageMicroOp<N, T, N2, T2> >;
        static ActiveMessageHandlerReg<RemoteMicroOpMessage<ImageMicroOp<N, T, N2, T2> > > areg;

        friend class PartitioningMicroOp;

        template<typename S>
        REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

        // construct from received packet
        template<typename S>
        ImageMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S &s);

        template<typename BM>
        void populate_bitmasks_ptrs(std::map<int, BM *> &bitmasks);

        template<typename BM>
        void populate_bitmasks_ranges(std::map<int, BM *> &bitmasks);

        template<typename BM>
        void populate_approx_bitmask_ptrs(BM &bitmask);

        template<typename BM>
        void populate_approx_bitmask_ranges(BM &bitmask);

        IndexSpace<N, T> parent_space;
        IndexSpace<N2, T2> inst_space;
        RegionInstance inst;
        size_t field_offset;
        bool is_ranged;
        bool is_intersection;
        std::vector<IndexSpace<N2, T2> > sources;
        std::vector<IndexSpace<N, T> > diff_rhss;
        std::vector<SparsityMap<N, T> > sparsity_outputs;
        int approx_output_index;
        intptr_t approx_output_op;
    };

    template<int N, typename T, int N2, typename T2>
    class ImageOperation : public PartitioningOperation {
    public:
        ImageOperation(const IndexSpace<N, T> &_parent,
                       const DomainTransform<N, T, N2, T2> &_domain_transform,
                       const ProfilingRequestSet &reqs, GenEventImpl *_finish_event,
                       EventImpl::gen_t _finish_gen);

        virtual ~ImageOperation(void);

        IndexSpace<N, T> add_source(const IndexSpace<N2, T2> &source);

        IndexSpace<N, T> add_source_with_difference(
            const IndexSpace<N2, T2> &source, const IndexSpace<N, T> &diff_rhs);

        IndexSpace<N, T> add_source_with_intersection(
            const IndexSpace<N2, T2> &source, const IndexSpace<N, T> &diff_rhs);

        virtual void execute(void);

        virtual void print(std::ostream &os) const;

        virtual void set_overlap_tester(void *tester);

    protected:
        IndexSpace<N, T> parent;
        DomainTransform<N, T, N2, T2> domain_transform;
        std::vector<IndexSpace<N2, T2> > sources;
        std::vector<IndexSpace<N, T> > diff_rhss;
        std::vector<SparsityMap<N, T> > images;
        bool is_intersection;
    };

    template<int N, typename T, int N2, typename T2>
    class StructuredImageMicroOp : public PartitioningMicroOp {
    public:
        StructuredImageMicroOp(
            const IndexSpace<N, T> &_parent,
            const StructuredTransform<N, T, N2, T2> &_transform);

        virtual ~StructuredImageMicroOp(void);

        virtual void execute(void);

        virtual void populate(std::map<int, HybridRectangleList<N, T> *> &bitmasks);

        void dispatch(PartitioningOperation *op, bool inline_ok);

        void add_sparsity_output(IndexSpace<N2, T2> _source,
                                 SparsityMap<N, T> _sparsity);

    protected:
        IndexSpace<N, T> parent_space;
        StructuredTransform<N, T, N2, T2> transform;
        std::vector<IndexSpace<N2, T2> > sources;
        std::vector<SparsityMap<N, T> > sparsity_outputs;
    };

    template<int N, typename T, int N2, typename T2>
      class GPUImageOperation : public PartitioningOperation {
    public:
      GPUImageOperation(const IndexSpace<N, T> &_parent,
                     const DomainTransform<N, T, N2, T2> &_domain_transform,
                     const ProfilingRequestSet &reqs,
                     size_t _buffer_size,
                     RegionInstance _buffer,
                     GenEventImpl *_finish_event,
                     EventImpl::gen_t _finish_gen);

      virtual ~GPUImageOperation(void);

      IndexSpace<N, T> add_source(const IndexSpace<N2, T2> &source);

      virtual void execute(void);

      virtual void print(std::ostream &os) const;

    protected:
      IndexSpace<N, T> parent;
      DomainTransform<N, T, N2, T2> domain_transform;
      std::vector<IndexSpace<N2, T2> > sources;
      std::vector<SparsityMap<N, T> > images;
      size_t buffer_size;
      RegionInstance buffer;
    };

    template<int N, typename T, int N2, typename T2>
    class GPUImageMicroOp : public GPUMicroOp<N, T> {
    public:
        GPUImageMicroOp(
            const IndexSpace<N, T> &_parent,
            const DomainTransform<N, T, N2, T2> &_domain_transform,
            bool _exclusive, size_t fixed_buffer_size = 0, RegionInstance buffer = RegionInstance::NO_INST);

        virtual ~GPUImageMicroOp(void);

        virtual void execute(void);

        virtual void gpu_populate_ptrs();

        virtual void gpu_populate_rngs();

        void dispatch(PartitioningOperation *op, bool inline_ok);

        void add_sparsity_output(IndexSpace<N2, T2> _source,
                                 SparsityMap<N, T> _sparsity);

    protected:
        IndexSpace<N, T> parent_space;
        DomainTransform<N, T, N2, T2> domain_transform;
        std::vector<IndexSpace<N2, T2> > sources;
        std::vector<SparsityMap<N, T> > sparsity_outputs;
        size_t fixed_buffer_size;
        RegionInstance buffer;
    };
}; // namespace Realm

#endif // REALM_DEPPART_IMAGE_H
