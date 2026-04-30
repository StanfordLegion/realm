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

// Realm inter-node networking abstractions

#ifndef REALM_NETWORK_H
#define REALM_NETWORK_H

#include "realm/realm_config.h"
#include "realm/module.h"
#include "realm/nodeset.h"
#include "realm/memory.h"
#include "realm/bytearray.h"

#include <map>

namespace Realm {

  // NodeID defined in nodeset.h

  class NetworkModule;
  class MemoryImpl;
  class IBMemory;
  class ByteArray;
  class ActiveMessageImpl;
  class IncomingMessageManager;
  class NetworkSegment;

  // a RemoteAddress is used to name the target of an RDMA operation - in some
  //  cases it's as simple as a pointer, but in others additional info is needed
  //  (hopefully we won't need more than 16B anywhere though)
  struct RemoteAddress {
    union {
      struct {
        uintptr_t ptr;
        uintptr_t extra;
      };
      unsigned char raw_bytes[384];
    };
  };

  // a LocalAddress is used to name the local source of an RDMA write or target
  //  of an RDMA read
  struct LocalAddress {
    const NetworkSegment *segment;
    uintptr_t offset;
  };

  namespace Network {
    // a few globals for efficiency
    extern NodeID my_node_id;
    extern NodeID max_node_id;
    extern NodeSet all_peers;
    // all peers that can access shared memory from this node
    // NOTE: This is an over-estimation.  Users should be robust to the fact that this may
    //       include peers that are not able to access shared memory.
    extern NodeSet shared_peers;

    // in most cases, there will be a single network module - if so, we set
    //  this so we don't have to do a per-node lookup
    extern NetworkModule *single_network;

    // gets the network for a given node
    NetworkModule *get_network(NodeID node);

    // and a few "global" operations that abstract over any/all networks
    void barrier(void);

    // result of a single quiescence-check round.  The runtime's shutdown loop
    //  calls Network::check_for_quiescence repeatedly and uses this to decide
    //  whether to keep iterating, declare termination, or abort
    enum class QuiescenceStatus
    {
      // confirmed quiescent: two consecutive rounds agreed on the global state
      //  AND the state is "no in-flight messages, no queued work, no pending
      //  completions".  The runtime can safely proceed with detach.
      DONE,
      // counters changed since the previous round (or this is the first call) -
      //  the system is making progress, the runtime should call again
      PROGRESSING,
      // counters have been frozen for several consecutive rounds but the global
      //  state is not quiet - this indicates a real bug (a leak in the message
      //  accounting or a permanently-stuck queue), not just a slow operation
      STUCK,
    };

    // a quiescence check across all nodes - returns whether the global system
    //  is quiescent, still progressing, or stuck.  The implementation is a
    //  Mattern's-shaped two-round stability check; correctness requires that
    //  the application has already promised it will not initiate new work
    //  (e.g., post-precondition during shutdown).
    QuiescenceStatus check_for_quiescence(IncomingMessageManager *message_manager);

    // collective communication across all nodes (TODO: subcommunicators?)
    template <typename T>
    T broadcast(NodeID root, T val);

    template <typename T>
    void gather(NodeID root, T val, std::vector<T> &result);
    template <typename T>
    void gather(NodeID root, T val); // for non-root participants

    // untyped versions
    void broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes);
    void gather(NodeID root, const void *val_in, void *vals_out, size_t bytes);

    // for sending active messages
    ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const RemoteAddress &dest_payload_addr, void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        const NodeSet &targets, unsigned short msgid, size_t header_size,
        size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, void *storage_base, size_t storage_size);

    size_t recommended_max_payload(NodeID target, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(NodeID target, const RemoteAddress &dest_payload_addr,
                                   bool with_congestion, size_t header_size);
    size_t recommended_max_payload(NodeID target, const void *data, size_t bytes_per_line,
                                   size_t lines, size_t line_stride, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(const NodeSet &targets, const void *data,
                                   size_t bytes_per_line, size_t lines,
                                   size_t line_stride, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(NodeID target, const LocalAddress &src_payload_addr,
                                   size_t bytes_per_line, size_t lines,
                                   size_t line_stride,
                                   const RemoteAddress &dest_payload_addr,
                                   bool with_congestion, size_t header_size);
  }; // namespace Network

  // a network module provides additional functionality on top of a normal Realm
  //  module
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NetworkModule : public Module {
  protected:
    NetworkModule(const std::string &_name);

  public:
    // all subclasses should define this (static) method - its responsibilities
    // are:
    // 1) determine if the network module should even be loaded
    // 2) fix the command line if the spawning system hijacked it
    // static NetworkModule *create_network_module(RuntimeImpl *runtime,
    //                                            int *argc, const char ***argv);

    // Enumerates all the peers that the current node could potentially share memory with
    virtual void get_shared_peers(NodeSet &shared_peers) = 0;

    // actual parsing of the command line should wait until here if at all
    //  possible
    virtual void parse_command_line(RuntimeImpl *runtime,
                                    std::vector<std::string> &cmdline);

    // "attaches" to the network, if that is meaningful - attempts to
    //  bind/register/(pick your network-specific verb) the requested memory
    //  segments with the network
    virtual void attach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) = 0;

    // detaches from the network
    virtual void detach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) = 0;

    // collective communication within this network
    virtual void barrier(void) = 0;
    virtual void broadcast(NodeID root, const void *val_in, void *val_out,
                           size_t bytes) = 0;
    virtual void gather(NodeID root, const void *val_in, void *vals_out,
                        size_t bytes) = 0;
    virtual void allgatherv(const char *val_in, size_t bytes, std::vector<char> &vals_out,
                            std::vector<size_t> &lengths) = 0;

    // Per-rank state used by the quiescence-detection algorithm in
    //  Network::check_for_quiescence.  Each backend exposes its current local
    //  state, the algorithm sums/ORs across ranks, and termination is declared
    //  when two consecutive rounds agree on a quiet state.
    //
    // Correctness depends on every source of "future messages from this rank"
    //  being captured by either:
    //   - any_queue_nonempty (anything queued that could spontaneously
    //     produce a send when it runs), or
    //   - packets_reserved exceeding packets_received globally (anything
    //     in flight on the wire), or
    //   - pending_completions > 0 (any local-completion bookkeeping that
    //     could fire and produce comp replies)
    struct QuiescenceState {
      // 1 if any work-item queue, message-manager queue, or pending in-flight
      //  GASNet/UCX/MPI request is non-empty at sample time; 0 otherwise.
      //  Reduced via SUM (any rank reporting nonzero leaves the sum nonzero)
      uint64_t any_queue_nonempty;
      // cumulative count of network messages this rank has originated.
      //  At quiescence, sum across ranks must equal sum of packets_received
      uint64_t packets_reserved;
      // cumulative count of network messages this rank has received and
      //  counted - sampled post-drain so the count reflects everything the
      //  drain in Network::check_for_quiescence delivered to handlers
      uint64_t packets_received;
      // count of pending remote completions on this rank waiting for replies
      //  to arrive; reduced via SUM
      uint64_t pending_completions;
    };

    // Sample the current quiescence state of this network.  Called after the
    //  IncomingMessageManager has been drained, so 'packets_received' reflects
    //  every received message that has been dispatched as of the call.
    virtual void sample_quiescence_state(QuiescenceState &state) = 0;

    // Sum-reduce a small array of uint64_t across all ranks.  Used by the
    //  Mattern's-shaped Network::check_for_quiescence loop to combine
    //  per-rank QuiescenceState samples into a global tally.
    virtual void quiescence_allreduce_sum(const uint64_t *local_counts,
                                          uint64_t *total_counts, size_t count) = 0;

    // used to create a remote proxy for a memory
    virtual MemoryImpl *create_remote_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                             Memory::Kind kind,
                                             const ByteArray &rdma_info) = 0;
    virtual IBMemory *create_remote_ib_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                              Memory::Kind kind,
                                              const ByteArray &rdma_info) = 0;

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size) = 0;

    virtual ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size) = 0;

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size,
                               const RemoteAddress &dest_payload_addr, void *storage_base,
                               size_t storage_size) = 0;

    virtual ActiveMessageImpl *create_active_message_impl(
        const NodeSet &targets, unsigned short msgid, size_t header_size,
        size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, void *storage_base, size_t storage_size) = 0;

    virtual size_t recommended_max_payload(NodeID target, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(const NodeSet &targets, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target,
                                           const LocalAddress &src_payload_addr,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size) = 0;
  };

  namespace NetworkSegmentInfo {
    // "enum" (using a namespace so that they can be extended in other
    //  headers) describing the different kind of memories that a network
    //  segment can live in
    typedef unsigned MemoryType;

    // each memory type gets to define what the extra data means for itself
    typedef uintptr_t MemoryTypeExtraData;

    static const MemoryType Unknown = 0;

    // generic memory that is read/write-able by the host CPUs
    static const MemoryType HostMem = 1;

    // optional flags for a network segment
    typedef unsigned FlagsType;
    struct OptionFlags {
      // registration should be performed on-demand rather than eagerly
      static const FlagsType OnDemandRegistration = 1U << 0;
    };
  }; // namespace NetworkSegmentInfo

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NetworkSegment {
  public:
    NetworkSegment();

    // normally a request will just be for a particular size
    void request(NetworkSegmentInfo::MemoryType _memtype, size_t _bytes,
                 size_t _alignment, NetworkSegmentInfo::MemoryTypeExtraData _memextra = 0,
                 NetworkSegmentInfo::FlagsType _flags = 0);

    // but it can also be for a pre-allocated chunk of memory with a fixed address
    void assign(NetworkSegmentInfo::MemoryType _memtype, void *_base, size_t _bytes,
                NetworkSegmentInfo::MemoryTypeExtraData _memextra = 0,
                NetworkSegmentInfo::FlagsType _flags = 0);

    void *base; // once this is non-null, it cannot be changed
    size_t bytes, alignment;
    NetworkSegmentInfo::MemoryType memtype;
    NetworkSegmentInfo::MemoryTypeExtraData memextra;
    NetworkSegmentInfo::FlagsType flags;

    // again, a single network puts itself here in addition to adding to the map
    NetworkModule *single_network;
    ByteArray *single_network_data;

    // a map from each of the networks that successfully bound the segment to
    //  whatever data (if any) that network needs to track the binding
    std::map<NetworkModule *, ByteArray> networks;

    void add_rdma_info(NetworkModule *network, const void *data, size_t len);
    const ByteArray *get_rdma_info(NetworkModule *network) const;

    // returns whether the segment is registered for all networks,
    //  or for a specific network
    bool is_registered() const;
    bool is_registered(NetworkModule *network) const;

    // tests whether an address range is in segment
    bool in_segment(const void *range_base, size_t range_bytes) const;
    bool in_segment(uintptr_t range_base, size_t range_bytes) const;
  };

}; // namespace Realm

#include "realm/network.inl"

#endif
