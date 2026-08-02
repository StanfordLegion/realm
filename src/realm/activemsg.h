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

// hopefully a more user-friendly C++ template wrapper for GASNet active
//  messages...

#ifndef ACTIVEMSG_H
#define ACTIVEMSG_H

#include "realm/realm_config.h"
#include "realm/fragmented_message.h"
#include "realm/mutex.h"
#include "realm/serialize.h"
#include "realm/nodeset.h"
#include "realm/multicast.h"
#include "realm/network.h"
#include "realm/atomics.h"
#include "realm/threads.h"
#include "realm/bgwork.h"
#include <atomic>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <mutex>
#include <vector>

#include <optional>

namespace Realm {

  namespace Config {
    // if true, the number and min/max/avg/stddev duration of handler per
    //  message type is recorded and printed
    extern bool profile_activemsg_handlers;

    // the maximum time we're willing to spend on inline message
    //  handlers
    extern long long max_inline_message_time;
  }; // namespace Config

  enum
  {
    PAYLOAD_NONE,    // no payload in packet
    PAYLOAD_KEEP,    // use payload pointer, guaranteed to be stable
    PAYLOAD_FREE,    // take ownership of payload, free when done
    PAYLOAD_COPY,    // make a copy of the payload
    PAYLOAD_SRCPTR,  // payload has been copied to the src data pool
    PAYLOAD_PENDING, // payload needs to be copied, but hasn't yet
    PAYLOAD_KEEPREG, // use payload pointer, AND it's registered!
    PAYLOAD_EMPTY,   // message can have payload, but this one is 0 bytes
  };

  class ActiveMessageImpl;

  template <typename T, size_t INLINE_STORAGE = 256>
  class ActiveMessage {
  public:
    // constructs an INACTIVE message object - call init(...) as needed
    ActiveMessage();

    // construct a new active message for a single recipient
    // in addition to the header struct (T), a message can include a variable
    //  payload which can be delivered to a particular destination address
    // NOTE: there is deliberately no NodeSet ("send this to many nodes")
    //  constructor - a multi-target send fans out one message per target at
    //  the SOURCE, which does not scale.  Use multicast_message<T>(...) from
    //  realm/activemsg.h instead: it forwards over a bounded-radix
    //  tree of ordinary unicast messages (plan sections 7.1-7.6).
    ActiveMessage(NodeID _target, size_t _max_payload_size = 0);
    ActiveMessage(NodeID _target, size_t _max_payload_size,
                  const RemoteAddress &_dest_payload_addr);

    // providing the payload (as a 1D reference, which must be PAYLOAD_KEEP)
    //  up front can avoid a copy if the source location is directly accessible
    //  by the networking hardware
    //  Per the semantics of PAYLOAD_KEEP, you must keep the payload buffer
    //  alive and unmodified until the call to commit or cancel returns
    ActiveMessage(NodeID _target, const void *_data, size_t _datalen);
    ActiveMessage(NodeID _target, const LocalAddress &_src_payload_addr, size_t _datalen,
                  const RemoteAddress &_dest_payload_addr);
    ActiveMessage(NodeID _target, const LocalAddress &_src_payload_addr,
                  size_t _bytes_per_line, size_t _lines, size_t _line_stride,
                  const RemoteAddress &_dest_payload_addr);

    ~ActiveMessage(void);

    // a version of `init` for each constructor above
    void init(NodeID _target, size_t _max_payload_size = 0);
    void init(NodeID _target, size_t _max_payload_size,
              const RemoteAddress &_dest_payload_addr);
    void init(NodeID _target, const void *_data, size_t _datalen);
    void init(NodeID _target, const LocalAddress &_src_payload_addr, size_t _datalen,
              const RemoteAddress &_dest_payload_addr);
    void init(NodeID _target, const LocalAddress &_src_payload_addr,
              size_t _bytes_per_line, size_t _lines, size_t _line_stride,
              const RemoteAddress &_dest_payload_addr);

    // large messages may need to be fragmented, so use cases that can
    //  handle the fragmentation at a higher level may want to know the
    //  largest size that is fragmentation-free - the answer can depend
    //  on whether the data is to be delivered to a known RemoteAddress
    //  and/or whether the source data location is known
    // a call that sets `with_congestion` may get a smaller value (maybe
    //  even 0) if the path to the named target(s) is getting full
    static size_t recommended_max_payload(NodeID target, bool with_congestion);
    static size_t recommended_max_payload(NodeID target,
                                          const RemoteAddress &dest_payload_addr,
                                          bool with_congestion);
    static size_t recommended_max_payload(NodeID target, const void *data,
                                          size_t bytes_per_line, size_t lines,
                                          size_t line_stride, bool with_congestion);
    static size_t
    recommended_max_payload(NodeID target, const LocalAddress &src_payload_addr,
                            size_t bytes_per_line, size_t lines, size_t line_stride,
                            const RemoteAddress &dest_payload_addr, bool with_congestion);

    // operator-> gives access to the header structure
    T *operator->(void);
    T &operator*(void);

    // variable payload can be written to in three ways:
    //  (a) Realm-style serialization (currently eager)
    template <typename T2>
    bool operator<<(const T2 &to_append);
    //  (b) old memcpy-like behavior (using the various payload modes)
    void add_payload(const void *data, size_t datalen, int payload_mode = PAYLOAD_COPY);
    void add_payload(const void *data, size_t bytes_per_line, size_t lines,
                     size_t line_stride, int payload_mode = PAYLOAD_COPY);
    //  (c) request for a pointer to write into (writes must be completed before
    //       call to commit or cancel)
    void *payload_ptr(size_t datalen);

    // register callbacks to be called upon:
    //  a) local completion - i.e. all source data has been read and can now
    //      be safely overwritten
    //  b) remote completion - message has been received AND HANDLED by target
    //
    // callbacks need to be "lightweight" - for heavier work, the message
    //  handler on the target can send an explicit response message
    template <typename CALLABLE>
    void add_local_completion(const CALLABLE &callable);
    template <typename CALLABLE>
    void add_remote_completion(const CALLABLE &callable);

    // every active message must eventually be commit()'ed or cancel()'ed
    void commit(void);
    void cancel(void);

  protected:
    ActiveMessageImpl *impl;
    T *header;
    Realm::Serialization::FixedBufferSerializer fbs;
    alignas(alignof(T) > alignof(uint64_t) ? alignof(T) : alignof(uint64_t)) uint64_t
        inline_capacity[INLINE_STORAGE / sizeof(uint64_t)];

  private:
    // chunked-mode state: only used when the requested payload exceeds the
    //  network's hard limit for a single message
    size_t network_max_payload_{0}; // 0 = normal (non-chunked) mode
    NodeID chunk_target_{-1};
    const void *chunk_src_data_{nullptr};
    size_t chunk_src_datalen_{0};
    std::vector<std::byte>
        chunk_alloc_; // owned buffer for network-allocated chunked mode

    void init_chunked(NodeID _target, size_t _max_payload_size);
    void init_chunked_data(NodeID _target, const void *_data, size_t _datalen);
    void commit_chunked(void);
    static uint64_t next_chunk_message_id(NodeID node_id);
  };

  // type-erased wrappers for completion callbacks
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE CompletionCallbackBase {
  public:
    virtual ~CompletionCallbackBase();
    virtual void invoke() = 0;
    virtual size_t size() const = 0;
    virtual CompletionCallbackBase *clone_at(void *p) const = 0;

    static const size_t ALIGNMENT = 8;

    // helper functions for invoking/cloning/destroying a collection of callbacks
    static void invoke_all(void *start, size_t bytes);
    static void clone_all(void *dst, const void *src, size_t bytes);
    static void destroy_all(void *start, size_t bytes);
  };

  template <typename CALLABLE>
  class CompletionCallback : public CompletionCallbackBase {
  public:
    CompletionCallback(const CALLABLE &_callable);
    virtual void invoke();
    virtual size_t size() const;
    virtual CompletionCallbackBase *clone_at(void *p) const;

  protected:
    CALLABLE callable;
  };

  // per-network active message implementations are mostly opaque, but a few
  //  fields are exposed to avoid virtual function calls
  class ActiveMessageImpl {
  public:
    virtual ~ActiveMessageImpl() {}

    // reserves space for a local/remote completion - caller will
    //  placement-new the completion at the provided address
    virtual void *add_local_completion(size_t size) = 0;
    virtual void *add_remote_completion(size_t size) = 0;

    virtual void commit(size_t act_payload_size) = 0;
    virtual void cancel() = 0;

    void *header_base;
    void *payload_base;
    size_t payload_size;
  };

  class ActiveMessageHandlerRegBase;

  struct ActiveMessageHandlerStats {
    atomic<size_t> count, sum, sum2, minval, maxval;

    ActiveMessageHandlerStats(void);
    void record(long long t_start, long long t_end);
  };

  struct FragmentInfo {
    uint32_t chunk_id{0};
    uint32_t total_chunks{0};
    uint64_t msg_id{0};
  };

  // singleton class that can convert message type->ID and ID->handler
  class ActiveMessageHandlerTable {
  public:
    ActiveMessageHandlerTable(void);
    ~ActiveMessageHandlerTable(void);

    typedef unsigned short MessageID;
    typedef void (*MessageHandler)(NodeID sender, const void *header, const void *payload,
                                   size_t payload_size, TimeLimit work_until);
    typedef void (*MessageHandlerNoTimeout)(NodeID sender, const void *header,
                                            const void *payload, size_t payload_size);
    typedef bool (*MessageHandlerInline)(NodeID sender, const void *header,
                                         const void *payload, size_t payload_size,
                                         TimeLimit work_until);

    template <typename T>
    MessageID lookup_message_id(void) const;

    const char *lookup_message_name(MessageID id);
    void record_message_handler_call(MessageID id, long long t_start, long long t_end);
    void report_message_handler_stats();

    static void append_handler_reg(ActiveMessageHandlerRegBase *new_reg);

    void construct_handler_table(void);

    typedef unsigned TypeHash;

    struct HandlerEntry {
      TypeHash hash;
      const char *name;
      MessageHandler handler;
      MessageHandlerNoTimeout handler_notimeout;
      MessageHandlerInline handler_inline;
      ActiveMessageHandlerStats stats;

      std::optional<const FragmentInfo &(*)(const void *)> extract_frag_info;
    };

    HandlerEntry *lookup_message_handler(MessageID id);

  protected:
    static ActiveMessageHandlerRegBase *pending_handlers;

    std::vector<HandlerEntry> handlers;
  };

  extern ActiveMessageHandlerTable activemsg_handler_table;

  class ActiveMessageHandlerRegBase {
  public:
    virtual ~ActiveMessageHandlerRegBase(void) {}
    virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const = 0;
    virtual ActiveMessageHandlerTable::MessageHandlerNoTimeout
    get_handler_notimeout(void) const = 0;
    virtual ActiveMessageHandlerTable::MessageHandlerInline
    get_handler_inline(void) const = 0;

    ActiveMessageHandlerTable::TypeHash hash;
    const char *name;
    bool must_free;
    ActiveMessageHandlerRegBase *next_handler;
    std::optional<const FragmentInfo &(*)(const void *)> extract_frag_info;
  };

  template <typename T, typename T2 = T>
  class ActiveMessageHandlerReg : public ActiveMessageHandlerRegBase {
  public:
    ActiveMessageHandlerReg(void);
    ~ActiveMessageHandlerReg(void);

    // when registering an active message handler, the following three methods
    //  are looked for in class T2
    // (a) void handle_message(NodeID, const T&, const void *, size_t, TimeLimit)
    // (b) void handle_message(NodeID, const T&, const void *, size_t)
    // (c) bool handle_inline(NodeID, const T&, const void *, size_t, TimeLimit)
    //
    // at least one of (a) or (b) must be present, with (a) being preferred
    //
    // if (c) is present, it will be used to attempt inline handling of
    //  active messages as they arrive, with the following constraints:
    //   (i) the handler must not block on any mutexes (trylocks are ok)
    //   (ii) the handler must not perform dynamic memory allocation/frees
    //   (iii) the handler must try very hard to stay within the specified
    //          time limit
    // if the inline handler is unable to satisfy these requirements, it should
    //  not attempt to handle the message, returning 'false' and letting it be
    //  queued as normal

    // returns either the requested kind of handler or a null pointer if
    //  it doesn't exist
    virtual ActiveMessageHandlerTable::MessageHandler get_handler(void) const;
    virtual ActiveMessageHandlerTable::MessageHandlerNoTimeout
    get_handler_notimeout(void) const;
    virtual ActiveMessageHandlerTable::MessageHandlerInline
    get_handler_inline(void) const;

    // this method does nothing, but can be called to force the instantiation
    //  of a handler registration object (needed when things are inside templates)
    void force_instantiation(void) {}
  };

  namespace ThreadLocal {
    // this flag will be true when we are running a message handler
    extern thread_local bool in_message_handler;
  }; // namespace ThreadLocal

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE IncomingMessageManager
    : public BackgroundWorkItem {
  public:
    IncomingMessageManager(int _nodes, int _dedicated_threads,
                           Realm::CoreReservationSet &crs);
    ~IncomingMessageManager(void);

    typedef uintptr_t CallbackData;
    typedef void (*CallbackFnptr)(NodeID, CallbackData, CallbackData);

    // adds an incoming message to the queue
    // returns true if the call was handled immediately (in which case the
    //  callback, if present, will NOT be called), or false if the message
    //  will be processed later
    bool add_incoming_message(NodeID sender, ActiveMessageHandlerTable::MessageID msgid,
                              const void *hdr, size_t hdr_size, int hdr_mode,
                              const void *payload, size_t payload_size, int payload_mode,
                              CallbackFnptr callback_fnptr, CallbackData callback_data1,
                              CallbackData callback_data2, TimeLimit work_until);

    void start_handler_threads(size_t stack_size);

    // stalls caller until all incoming messages have been handled (and at
    //  least 'min_messages_handled' in total)
    void drain_incoming_messages(size_t min_messages_handled);

    void shutdown(void);

    virtual bool do_work(TimeLimit work_until);

    void handler_thread_loop(void);

  protected:
    struct MessageBlock;

    struct Message {
      MessageBlock *block;
      Message *next_msg;
      NodeID sender;
      ActiveMessageHandlerTable::HandlerEntry *handler;
      void *hdr;
      size_t hdr_size;
      bool hdr_needs_free;
      void *payload;
      size_t payload_size;
      bool payload_needs_free;
      CallbackFnptr callback_fnptr;
      CallbackData callback_data1, callback_data2;
    };

    struct MessageBlock {
      static MessageBlock *new_block(size_t _total_size);
      static void free_block(MessageBlock *block);

      void reset();

      // called with message manager lock held
      Message *append_message(size_t hdr_bytes_needed, size_t payload_bytes_needed);

      // called _without_ message manager lock held
      void recycle_message(Message *msg, IncomingMessageManager *manager);

      size_t total_size, size_used;
      atomic<unsigned> use_count;
      MessageBlock *next_free;
    };

    int get_messages(Message *&head, Message **&tail, bool wait);
    bool return_messages(int sender, size_t num_handled, Message *head, Message **tail);

    int nodes, dedicated_threads, sleeper_count;
    atomic<bool> bgwork_requested;
    int shutdown_flag;
    Message **heads;
    Message ***tails;
    bool *in_handler;
    int *todo_list; // list of nodes with non-empty message lists
    int todo_oldest, todo_newest;
    int handlers_active;
    bool drain_pending;
    size_t drain_min_count;
    size_t total_messages_handled;
    Mutex mutex;
    Mutex::CondVar condvar, drain_condvar;
    CoreReservation *core_rsrv;
    std::vector<Thread *> handler_threads;
    MessageBlock *current_block;
    MessageBlock *available_blocks;
    size_t num_available_blocks;
    size_t cfg_max_available_blocks, cfg_message_block_size;

    struct PairHash {
      std::size_t operator()(const std::pair<NodeID, uint64_t> &p) const
      {
        return std::hash<NodeID>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
      }
    };

    std::unordered_map<std::pair<NodeID, uint64_t>, std::unique_ptr<FragmentedMessage>,
                       PairHash>
        frag_message;
  };

  template <typename UserHdr>
  struct WrappedWithFragInfo {
    FragmentInfo frag_info;
    UserHdr user;

    UserHdr *operator->() { return &user; }
    const UserHdr *operator->() const { return &user; }
    UserHdr &operator*() { return user; }
    const UserHdr &operator*() const { return user; }
  };

  // trait to detect WrappedWithFragInfo types
  template <typename T>
  struct is_wrapped_with_frag_info : std::false_type {};
  template <typename U>
  struct is_wrapped_with_frag_info<WrappedWithFragInfo<U>> : std::true_type {};

  ////////////////////////////////////////////////////////////////////////
  //
  // multicast envelope and bounded-radix forwarding (plan sections 7.3-7.5)
  //
  // Layered strictly above backend unicast: a multicast is an ordinary unicast
  //  ActiveMessage carrying an envelope, which each relay repartitions and forwards
  //  BEFORE delivering the original typed message locally.  The target set and its wire
  //  encoding live in realm/multicast.h, which deliberately has no dependency on this
  //  file, so that codec stays unit-testable on its own.
  //

  // forwarding radix used when no core module config is available (matches the
  //  -ll:barrier_radix / barrier_broadcast_radix default)
  static const size_t MULTICAST_DEFAULT_RADIX = 4;

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastEnvelopeMessage
  //

  // Flags carried in the envelope.  A fire-and-forget multicast sets none of these and
  //  carries no completion metadata at all (plan section 7.5).
  namespace MulticastEnvelopeFlags {
    enum : uint32_t
    {
      // the envelope carries completion metadata after the original payload and the
      //  parent expects exactly one acknowledgement from this subtree
      COMPLETION_TRACKED = 1U << 0,

      // every bit that is defined - anything else in 'flags' is a protocol violation
      ALL_KNOWN = COMPLETION_TRACKED,
    };
  }; // namespace MulticastEnvelopeFlags

  // The multicast envelope (plan section 7.4).  The variable portion that follows is,
  //  in order:
  //
  //     [target_encoding_size]  encoded target slice (realm/multicast.h wire form)
  //     [original_header_size]  original typed header bytes
  //     [original_payload_size] original payload bytes
  //     [completion_size]       optional completion metadata - a single varint holding
  //                             the node this subtree must acknowledge to, present if
  //                             and only if COMPLETION_TRACKED is set
  //
  // This type deliberately does NOT declare a `handle_inline`: plan section 22 requires
  //  that child sends not be issued recursively from an inline handler, so an envelope
  //  is always queued and forwarded from the normal active-message path.  It also does
  //  not declare a FragmentInfo member, which means an oversized envelope is handled by
  //  the existing WrappedWithFragInfo fragmentation machinery on every hop and is
  //  reassembled before the relay repartitions it (plan section 7.5).
  struct MulticastEnvelopeMessage {
    // (origin_node, multicast_id) is the globally unique multicast identifier
    uint64_t multicast_id = 0;
    // the node that ORIGINATED the multicast - this, and never the previous hop, is
    //  what the original handler must see as its sender
    NodeID origin_node = 0;

    uint32_t original_payload_size = 0;
    uint32_t target_encoding_size = 0;
    uint32_t completion_size = 0;
    uint32_t flags = 0;
    // hops from the origin: the origin's own first-hop envelopes carry 1.  This is 32
    //  bits rather than 16 because with a radix of 1 the tree degenerates to a chain
    //  one hop long per target.
    uint32_t depth = 0;

    // message ID of the ORIGINAL typed message, looked up in the same
    //  ActiveMessageHandlerTable at both ends
    unsigned short original_message_id = 0;
    uint16_t original_header_size = 0;
    // redundant copy of the first payload byte, so that a receiver can report the
    //  claimed encoding in a fatal diagnostic even if the payload is truncated
    unsigned char target_encoding_kind = 0;

    static void handle_message(NodeID sender, const MulticastEnvelopeMessage &hdr,
                               const void *payload, size_t payload_size,
                               TimeLimit work_until);
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastAckMessage
  //

  // The single acknowledgement a completion-tracked subtree sends to its parent once it
  //  has delivered locally AND collected an acknowledgement from every child (plan
  //  section 7.5).  These exist only for the lifetime of one explicitly
  //  completion-tracked multicast; a fire-and-forget multicast never sends one.
  //
  // Like the envelope, this deliberately has no inline handler: handling an
  //  acknowledgement can send the next acknowledgement up the tree, and plan section 22
  //  requires that not to happen recursively out of an inline handler.
  struct MulticastAckMessage {
    // (origin_node, multicast_id) identifies which multicast is being acknowledged; the
    //  child that sent it is the ordinary active-message sender
    uint64_t multicast_id = 0;
    NodeID origin_node = 0;

    static void handle_message(NodeID sender, const MulticastAckMessage &hdr,
                               const void *payload, size_t payload_size,
                               TimeLimit work_until);
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // fatal diagnostics (plan section 21.1)
  //

  // Everything plan section 21.1 requires of a multicast fatal error.  Fields that are
  //  not knowable in a given failure (e.g. a header that did not decode) stay at their
  //  defaults.
  struct MulticastFatalContext {
    NodeID local_node = 0;
    // the PREVIOUS HOP the envelope arrived from, which is not in general the origin
    NodeID sender = 0;
    NodeID origin_node = 0;
    uint64_t multicast_id = 0;
    unsigned original_message_id = 0;
    unsigned target_encoding_kind = 0;
    size_t target_encoding_size = 0;
    size_t original_header_size = 0;
    size_t original_payload_size = 0;
    size_t received_payload_size = 0;
    unsigned depth = 0;
    MulticastDecodeStatus status = MulticastDecodeStatus::OK;
    // plain-language statement of the protocol rule that was violated
    const char *rule = "";

    void describe(std::ostream &os) const;
    std::string to_string(void) const;
  };

  // Injectable fatal-error hook, mirroring BarrierFatalReporter.  The default reporter
  //  logs the full context and aborts; a test installs its own so it can assert on the
  //  diagnostic without dying.  A reporter that returns normally means "this envelope
  //  is dropped", so every call site unwinds cleanly without delivering or forwarding.
  class MulticastFatalReporter {
  public:
    virtual ~MulticastFatalReporter(void) {}
    virtual void report(const MulticastFatalContext &ctx) = 0;
  };

  // returns the previously installed reporter (null means "the default"), so tests can
  //  restore it
  MulticastFatalReporter *set_multicast_fatal_reporter(MulticastFatalReporter *reporter);
  MulticastFatalReporter *get_multicast_fatal_reporter(void);
  void report_multicast_fatal(const MulticastFatalContext &ctx);

  ////////////////////////////////////////////////////////////////////////
  //
  // aggregate remote completion (plan section 7.5)
  //

  class MulticastTransport;

  // What the origin of a completion-tracked multicast wants run - exactly once, after
  //  the message has been received AND HANDLED by every target.  The forwarding layer
  //  takes ownership: the object is deleted as soon as it has been invoked.
  class MulticastCompletionCallback {
  public:
    virtual ~MulticastCompletionCallback(void);
    virtual void invoke(void) = 0;
  };

  // (origin, multicast_id) is globally unique, and partitions are disjoint, so a node
  //  is part of at most one subtree per multicast and this is a sufficient key
  struct MulticastCompletionKey {
    NodeID origin = 0;
    uint64_t multicast_id = 0;

    bool operator<(const MulticastCompletionKey &rhs) const
    {
      if(origin != rhs.origin)
        return (origin < rhs.origin);
      return (multicast_id < rhs.multicast_id);
    }
    bool operator==(const MulticastCompletionKey &rhs) const
    {
      return (origin == rhs.origin) && (multicast_id == rhs.multicast_id);
    }
  };

  // The TRANSIENT acknowledgement state of plan section 7.5.  There is exactly one of
  //  these per node (the runtime transport owns a process-global one; a unit test owns
  //  one per simulated node), and it is empty except while a completion-tracked
  //  multicast is actually in flight.  Nothing here is a reusable multicast plan: no
  //  target set, no encoding and no routing information is retained, and every record is
  //  reclaimed the instant its subtree is complete (plan section 2, final bullet).
  class MulticastCompletionState {
  public:
    MulticastCompletionState(void);
    ~MulticastCompletionState(void);

    // What the caller must do once a record's outstanding count reaches zero.  The
    //  record is already gone from the table by then - state is reclaimed before the
    //  acknowledgement is sent, never after.
    struct Notification {
      enum Action
      {
        NOTHING,         // still waiting on other children or on local delivery
        ACK_PARENT,      // relay: send exactly one acknowledgement to 'parent'
        INVOKE_CALLBACK, // origin: run 'callback' once, then delete it
        UNKNOWN,         // no such record - a protocol violation
      };
      Action action = NOTHING;
      NodeID parent = 0;
      MulticastCompletionCallback *callback = nullptr;
    };

    // Origin side: retain the callback under (origin, multicast_id).  'outstanding' is
    //  the number of first-hop envelopes plus one if the origin is itself a target.
    void begin_origin(NodeID origin, uint64_t multicast_id, size_t outstanding,
                      MulticastCompletionCallback *callback);

    // Relay side: retain the parent to acknowledge and the number of outstanding
    //  units, which is one per child envelope plus one for this node's own delivery.
    void begin_relay(NodeID origin, uint64_t multicast_id, NodeID parent,
                     size_t outstanding);

    // one unit of progress - either this node's local delivery finished being HANDLED,
    //  or one child acknowledged its whole subtree
    Notification note_completion(NodeID origin, uint64_t multicast_id);

    // number of multicasts currently being tracked here - zero except while a
    //  completion-tracked multicast is in flight
    size_t num_pending(void) const;

    // high-water mark since construction or the last reset_peak(), so a test can prove
    //  that a fire-and-forget multicast created no acknowledgement state at all
    size_t peak_pending(void) const;
    void reset_peak(void);

  protected:
    struct Record {
      NodeID parent = 0;
      size_t outstanding = 0;
      MulticastCompletionCallback *callback = nullptr;
      bool is_origin = false;
    };

    mutable Mutex mutex;
    std::map<MulticastCompletionKey, Record> pending;
    size_t peak = 0;
  };

  // Everything the deferred half of a local delivery needs in order to settle one unit
  //  of completion.  It is captured explicitly rather than rederived from the transport
  //  because the notification can run on a handler thread long after the forwarding call
  //  that created it returned.
  struct MulticastCompletionToken {
    MulticastTransport *transport = nullptr;
    MulticastCompletionState *state = nullptr;
    NodeID local = 0; // the node whose record this settles
    NodeID origin = 0;
    uint64_t multicast_id = 0;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastTransport
  //

  // Everything the forwarding algorithm needs from the rest of the runtime.  The
  //  production implementation (get_runtime_multicast_transport()) sends ordinary
  //  unicast active messages and delivers through the runtime's IncomingMessageManager;
  //  unit tests supply an in-process implementation that captures sends so that a whole
  //  forwarding tree can be exercised without a real network.
  class MulticastTransport {
  public:
    virtual ~MulticastTransport(void);

    virtual NodeID my_node_id(void) const = 0;
    // configured node count - both ends of an encoding must agree on this
    virtual NodeID num_nodes(void) const = 0;
    // bounded forwarding radix R (plan section 7.3); must be at least 1
    virtual size_t radix(void) const = 0;

    // Sends one multicast envelope to 'relay', which is by construction the first node
    //  of the slice the envelope carries.  'payload' is the entire variable portion and
    //  must be copied before this returns.
    virtual void send_envelope(NodeID relay, const MulticastEnvelopeMessage &env,
                               const void *payload, size_t payload_bytes) = 0;

    // May the ORIGINAL message be handed to a single target by ordinary unicast?  The
    //  answer is no when the payload would need fragmentation, because fragmentation is
    //  driven by the compile-time message type and this layer only has a message ID -
    //  such a send goes through an envelope instead and gets fragmented there.
    virtual bool can_send_original(size_t hdr_size, size_t payload_size) const = 0;

    // Sends the ORIGINAL typed message to 'target' by ordinary unicast.  Only legal
    //  when the local node is the origin, because the receiver sees the local node as
    //  the sender (plan section 7.4).
    //
    // A non-null 'completion' must be settled once the target has received AND HANDLED
    //  the message - i.e. it is an ordinary active-message remote completion.
    virtual void send_original(NodeID target, ActiveMessageHandlerTable::MessageID msgid,
                               const void *hdr, size_t hdr_size, const void *payload,
                               size_t payload_size,
                               const MulticastCompletionToken *completion) = 0;

    // Sends one acknowledgement for 'multicast_id' from 'from' to 'parent'.  'from' is
    //  passed explicitly because an acknowledgement can be produced by a deferred local
    //  delivery, i.e. from a context that is no longer "on" the sending node.
    virtual void send_ack(NodeID from, NodeID parent, NodeID origin,
                          uint64_t multicast_id) = 0;

    // Final local delivery.  'origin' MUST be presented to the original handler as its
    //  sender; a relay never becomes the apparent sender just because it transmitted
    //  the final hop (plan section 7.4).
    //
    // A non-null 'completion' must be settled once the original handler has actually
    //  RUN, which for a message that could not be handled inline is later than this
    //  call returns; MulticastForwarder::dispatch_local does this correctly.
    virtual void deliver_local(NodeID origin, ActiveMessageHandlerTable::MessageID msgid,
                               const void *hdr, size_t hdr_size, const void *payload,
                               size_t payload_size, TimeLimit work_until,
                               const MulticastCompletionToken *completion) = 0;

    // The transient acknowledgement state of this node (plan section 7.5).  Empty
    //  except while a completion-tracked multicast is in flight.
    virtual MulticastCompletionState &completion_state(void) = 0;
  };

  // The transport used by the registered envelope handler and by the typed helpers
  //  below.  Its radix comes from the core module config value that also backs
  //  -ll:barrier_radix, read once and cached.
  MulticastTransport &get_runtime_multicast_transport(void);

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastForwarder
  //

  // The bounded-radix forwarding algorithm of plan section 7.3.  It is expressed purely
  //  in terms of MulticastTransport, so the whole tree is exercisable in one process.
  class MulticastForwarder {
  public:
    // Origin-side entry point.  'targets' is the complete logical target set; the local
    //  node may or may not be a member.  'hdr'/'payload' are copied before this returns
    //  so a PAYLOAD_KEEP-style caller buffer may be reused immediately (plan 7.5) -
    //  equivalently, LOCAL completion has already happened when this returns and this
    //  layer no longer references any caller-owned data.
    //
    // An empty target set is a successful no-op, and a single remote target uses the
    //  ordinary unicast fast path.
    //
    // 'on_remote_complete', if given, is invoked EXACTLY ONCE after the message has
    //  been received and handled by every target, and is then deleted.  Ownership
    //  transfers to this call.  Passing null - the normal fire-and-forget case - means
    //  no acknowledgement metadata is put on the wire and no acknowledgement state is
    //  created anywhere (plan section 7.5).
    static void send(MulticastTransport &transport, const MulticastTargetSet &targets,
                     ActiveMessageHandlerTable::MessageID msgid, const void *hdr,
                     size_t hdr_size, const void *payload, size_t payload_size,
                     TimeLimit work_until = TimeLimit(),
                     MulticastMetricsSink *metrics = 0,
                     MulticastCompletionCallback *on_remote_complete = 0);

    // Relay-side entry point, called by MulticastEnvelopeMessage::handle_message.
    //  'sender' is the previous hop and is used only for diagnostics.
    static void forward(MulticastTransport &transport, NodeID sender,
                        const MulticastEnvelopeMessage &env, const void *payload,
                        size_t payload_size, TimeLimit work_until,
                        MulticastMetricsSink *metrics = 0);

    // Acknowledgement-side entry point, called by MulticastAckMessage::handle_message.
    static void handle_ack(MulticastTransport &transport, NodeID sender,
                           const MulticastAckMessage &ack);

    // Records one unit of progress against a retained completion record and performs
    //  whatever that completes: acknowledging a parent, or invoking (and deleting) the
    //  origin's callback.  Public because it is also the deferred local-delivery
    //  notification and the unicast fast path's remote completion.
    static void settle(const MulticastCompletionToken &token);

    // Invokes the ALREADY REGISTERED handler for 'msgid' with 'sender' as the apparent
    //  sender, reusing the ordinary incoming-message machinery so that inline-handler
    //  and TimeLimit behavior is identical to a directly received message.  Handler
    //  signature detection is never duplicated here (plan section 7.4).
    //
    // If 'completion' is non-null it is settled once the handler has actually run -
    //  immediately if the message was handled inline, and otherwise from the ordinary
    //  post-handler callback so that "handled by every target" really means handled.
    //  The original message type must not be one that carries its own FragmentInfo,
    //  because an incomplete fragment never reaches a handler and so could never be
    //  observed as handled (checked in debug builds).
    static bool dispatch_local(IncomingMessageManager *manager, NodeID sender,
                               ActiveMessageHandlerTable::MessageID msgid,
                               const void *hdr, size_t hdr_size, const void *payload,
                               size_t payload_size, TimeLimit work_until,
                               const MulticastCompletionToken *completion = 0);

    // local half of the globally unique (origin_node, counter) multicast ID
    static uint64_t next_multicast_id(void);
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // typed helpers
  //

  // Multicasts one already-built typed header (and an optional copied 1-D payload) to
  //  'targets' over the runtime transport.  This REPLACES the old ActiveMessage(NodeSet)
  //  constructor, which fanned out one message per target at the source; that
  //  constructor, Network::create_active_message_impl(NodeSet, ...), the NodeSet
  //  recommended_max_payload overloads and every backend's multicast source loop are all
  //  gone (plan section 7.6).
  //
  // A caller holding a Realm::NodeSet converts explicitly with
  //  MulticastTargetSet(nodes) - NodeSet itself stays as a general in-memory set.
  //
  // The payload is always copied, so this is the PAYLOAD_COPY/PAYLOAD_KEEP equivalent;

} // namespace Realm

#include "realm/activemsg.inl"

#endif
