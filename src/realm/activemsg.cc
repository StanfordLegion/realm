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

#include "realm/realm_config.h"
#include "realm/atomics.h"

#include "realm/activemsg.h"
#include "realm/module_config.h"
#include "realm/multicast.h"
#include "realm/runtime_impl.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"
#include "realm/logging.h"

#include <math.h>

namespace Realm {

  Realm::Logger log_amhandler("amhandler");

  namespace Config {
    // if true, the number and min/max/avg/stddev duration of handler per
    //  message type is recorded and printed
    bool profile_activemsg_handlers = false;

    // the maximum time we're willing to spend on inline message
    //  handlers
    long long max_inline_message_time = 5000 /* nanoseconds*/;
  }; // namespace Config

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionCallbackBase
  //

  CompletionCallbackBase::~CompletionCallbackBase() {}

  /*static*/ void CompletionCallbackBase::invoke_all(void *start, size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      CompletionCallbackBase *cc = static_cast<CompletionCallbackBase *>(start);
      cc->invoke();
      size_t step = cc->size();
      start = static_cast<char *>(start) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }

  /*static*/ void CompletionCallbackBase::clone_all(void *dst, const void *src,
                                                    size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      const CompletionCallbackBase *cc = static_cast<const CompletionCallbackBase *>(src);
      cc->clone_at(dst);
      size_t step = cc->size();
      src = static_cast<const char *>(src) + step;
      dst = static_cast<char *>(dst) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }

  /*static*/ void CompletionCallbackBase::destroy_all(void *start, size_t bytes)
  {
    size_t ofs = 0;
    while(ofs < bytes) {
      CompletionCallbackBase *cc = static_cast<CompletionCallbackBase *>(start);
      size_t step = cc->size();
      cc->~CompletionCallbackBase();
      start = static_cast<char *>(start) + step;
      ofs += step;
    }
    assert(ofs == bytes);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // struct ActiveMessageHandlerStats
  //

  ActiveMessageHandlerStats::ActiveMessageHandlerStats(void)
    : count(0)
    , sum(0)
    , sum2(0)
    , minval(~size_t(0))
    , maxval(0)
  {}

  void ActiveMessageHandlerStats::record(long long t_start, long long t_end)
  {
    long long delta = t_end - t_start;
    size_t val = (delta > 0) ? delta : 0;
    count.fetch_add(1);
    minval.fetch_min(val);
    maxval.fetch_max(val);
    sum.fetch_add(val);
    sum2.fetch_add(val * val); // TODO: smarter math to avoid overflow
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ActiveMessageHandlerTable
  //

  ActiveMessageHandlerTable::ActiveMessageHandlerTable(void) {}

  ActiveMessageHandlerTable::~ActiveMessageHandlerTable(void) {}

  ActiveMessageHandlerTable::HandlerEntry *
  ActiveMessageHandlerTable::lookup_message_handler(MessageID id)
  {
    assert(id < handlers.size());
    return &handlers[id];
  }

  const char *
  ActiveMessageHandlerTable::lookup_message_name(ActiveMessageHandlerTable::MessageID id)
  {
    assert(id < handlers.size());
    return handlers[id].name;
  }

  void ActiveMessageHandlerTable::record_message_handler_call(MessageID id,
                                                              long long t_start,
                                                              long long t_end)
  {
    assert(id < handlers.size());
    handlers[id].stats.record(t_start, t_end);
  }

  void ActiveMessageHandlerTable::report_message_handler_stats()
  {
    if(Config::profile_activemsg_handlers) {
      for(size_t i = 0; i < handlers.size(); i++) {
        const ActiveMessageHandlerStats &stats = handlers[i].stats;
        size_t count = stats.count.load();
        if(count == 0)
          continue;

        size_t sum = stats.sum.load();
        size_t sum2 = stats.sum2.load();
        size_t minval = stats.minval.load();
        size_t maxval = stats.maxval.load();
        double avg = double(sum) / double(count);
        double stddev = sqrt((double(sum2) / double(count)) - (avg * avg));
        log_amhandler.print() << "handler " << i << ": " << handlers[i].name
                              << " count=" << count << " avg=" << avg << " dev=" << stddev
                              << " min=" << minval << " max=" << maxval;
      }
    }
  }

  /*static*/ void
  ActiveMessageHandlerTable::append_handler_reg(ActiveMessageHandlerRegBase *new_reg)
  {
    new_reg->next_handler = pending_handlers;
    pending_handlers = new_reg;
  }

  static inline bool hash_less(const ActiveMessageHandlerTable::HandlerEntry &a,
                               const ActiveMessageHandlerTable::HandlerEntry &b)
  {
    return (a.hash < b.hash);
  }

  void ActiveMessageHandlerTable::construct_handler_table(void)
  {
    for(ActiveMessageHandlerRegBase *nextreg = pending_handlers; nextreg;
        nextreg = nextreg->next_handler) {
      HandlerEntry e;
      e.hash = nextreg->hash;
      e.name = nextreg->name;
      e.handler = nextreg->get_handler();
      e.extract_frag_info = nextreg->extract_frag_info;
      e.handler_notimeout = nextreg->get_handler_notimeout();
      // at least one of the two above must be non-null
      assert((e.handler != 0) || (e.handler_notimeout != 0));
      e.handler_inline = nextreg->get_handler_inline();
      handlers.push_back(e);
    }

    std::sort(handlers.begin(), handlers.end(), hash_less);

    // handler ids are the same everywhere, so only log on node 0
    if(Network::my_node_id == 0)
      for(size_t i = 0; i < handlers.size(); i++)
        log_amhandler.info() << "handler " << i << ": " << handlers[i].name
                             << (handlers[i].handler ? " (timeout)" : "")
                             << (handlers[i].handler_inline ? " (inline)" : "");
  }

  /*static*/ ActiveMessageHandlerRegBase *ActiveMessageHandlerTable::pending_handlers = 0;

  /*extern*/ ActiveMessageHandlerTable activemsg_handler_table;

  ////////////////////////////////////////////////////////////////////////
  //
  // class IncomingMessageManager::MessageBlock
  //

  /*static*/ IncomingMessageManager::MessageBlock *
  IncomingMessageManager::MessageBlock::new_block(size_t _total_size)
  {
    void *ptr = malloc(_total_size);
    assert(ptr != 0);
    MessageBlock *block = new(ptr) MessageBlock;
    block->total_size = _total_size;
    block->next_free = 0;
    block->reset();
    log_amhandler.info() << "creating message block: " << block;
    return block;
  }

  /*static*/ void IncomingMessageManager::MessageBlock::free_block(MessageBlock *block)
  {
    while(block) {
      log_amhandler.info() << "freeing message block: " << block;
      MessageBlock *next = block->next_free;
      block->~MessageBlock();
      free(block);
      block = next;
    }
  }

  void IncomingMessageManager::MessageBlock::reset()
  {
    size_used = sizeof(MessageBlock);
    size_used = (size_used + 15) & ~size_t(15); // 16B alignment
    use_count.store(1);
  }

  IncomingMessageManager::Message *
  IncomingMessageManager::MessageBlock::append_message(size_t hdr_bytes_needed,
                                                       size_t payload_bytes_needed)
  {
    size_t msg_ofs = size_used;
    size_t new_used = msg_ofs + sizeof(Message);
    new_used = (new_used + 15) & ~size_t(15); // 16B alignment

    size_t hdr_ofs;
    if(hdr_bytes_needed > 0) {
      hdr_ofs = new_used;
      new_used += hdr_bytes_needed;
      new_used = (new_used + 15) & ~size_t(15); // 16B alignment
    } else
      hdr_ofs = 0;

    size_t payload_ofs;
    if(payload_bytes_needed > 0) {
      payload_ofs = new_used;
      new_used += payload_bytes_needed;
      new_used = (new_used + 15) & ~size_t(15); // 16B alignment
    } else
      payload_ofs = 0;

    // does it fit?
    if(new_used <= total_size) {
      use_count.fetch_add(1);
      size_used = new_used;

      uintptr_t base = reinterpret_cast<uintptr_t>(this);
      Message *msg = reinterpret_cast<Message *>(base + msg_ofs);
      msg->block = this;
      msg->hdr = ((hdr_ofs > 0) ? reinterpret_cast<void *>(base + hdr_ofs) : 0);
      msg->payload =
          ((payload_ofs > 0) ? reinterpret_cast<void *>(base + payload_ofs) : 0);
      return msg;
    } else {
      // would it have ever fit?
      assert((new_used - size_used) <= (total_size - sizeof(MessageBlock)));

      // return failure - caller will find a new block
      return 0;
    }
  }

  void IncomingMessageManager::MessageBlock::recycle_message(
      IncomingMessageManager::Message *msg, IncomingMessageManager *manager)
  {
    // first, free any hdr/payload pointer we were borrowing
    if(msg->hdr_needs_free)
      free(msg->hdr);
    if(msg->payload_needs_free)
      free(msg->payload);

    // now decrement our use_count
    unsigned prev_count = use_count.fetch_sub(1);

    // if it was 1 (now 0), take the manager's lock and add ourselves to
    //  the available list (or delete if there's already enough)
    if(prev_count == 1) {
      bool delete_me = false;
      {
        AutoLock<> al(manager->mutex);
        if(manager->num_available_blocks < manager->cfg_max_available_blocks) {
          reset();
          next_free = manager->available_blocks;
          manager->available_blocks = this;
          manager->num_available_blocks++;
        } else
          delete_me = true;
      }

      if(delete_me)
        free_block(this);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class IncomingMessageManager
  //

  namespace ThreadLocal {
    // this flag will be true when we are running a message handler
    thread_local bool in_message_handler = false;
  }; // namespace ThreadLocal

  IncomingMessageManager::IncomingMessageManager(int _nodes, int _dedicated_threads,
                                                 Realm::CoreReservationSet &crs)
    : BackgroundWorkItem("activemsg handler")
    , nodes(_nodes)
    , dedicated_threads(_dedicated_threads)
    , sleeper_count(0)
    , bgwork_requested(false)
    , shutdown_flag(0)
    , handlers_active(0)
    , drain_pending(false)
    , drain_min_count(0)
    , total_messages_handled(0)
    , condvar(mutex)
    , drain_condvar(mutex)
    , available_blocks(0)
    , num_available_blocks(0)
    , cfg_max_available_blocks(10)
    , cfg_message_block_size(1048576 - 32) // 1MB - space for heap metadata
  {
    heads = new Message *[nodes];
    tails = new Message **[nodes];
    in_handler = new bool[nodes];
    for(int i = 0; i < nodes; i++) {
      heads[i] = 0;
      tails[i] = 0;
      in_handler[i] = false;
    }
    todo_list = new int[nodes + 1]; // an extra entry to distinguish full from empty
    todo_oldest = todo_newest = 0;

    if(dedicated_threads > 0)
      core_rsrv = new Realm::CoreReservation("AM handlers", crs,
                                             Realm::CoreReservationParameters());
    else
      core_rsrv = 0;

    current_block = MessageBlock::new_block(cfg_message_block_size);
  }

  IncomingMessageManager::~IncomingMessageManager(void)
  {
    delete core_rsrv;
    delete[] heads;
    delete[] tails;
    delete[] in_handler;
    delete[] todo_list;

    MessageBlock::free_block(current_block);
    if(available_blocks)
      MessageBlock::free_block(available_blocks);
  }

  bool IncomingMessageManager::add_incoming_message(
      NodeID sender, ActiveMessageHandlerTable::MessageID msgid, const void *hdr,
      size_t hdr_size, int hdr_mode, const void *payload, size_t payload_size,
      int payload_mode, CallbackFnptr callback_fnptr, CallbackData callback_data1,
      CallbackData callback_data2, TimeLimit work_until)
  {
#ifdef DEBUG_INCOMING
    printf("adding incoming message from %d\n", sender);
#endif

    // look up which message this is
    ActiveMessageHandlerTable::HandlerEntry *handler =
        activemsg_handler_table.lookup_message_handler(msgid);

    std::vector<char> message;

    if(handler && handler->extract_frag_info.has_value()) {
      AutoLock<> al(mutex);

      const FragmentInfo &frag_info = handler->extract_frag_info.value()(hdr);

      if(frag_info.total_chunks > 1) {
        auto key = std::make_pair(sender, frag_info.msg_id);
        auto it = frag_message.find(key);

        if(it == frag_message.end()) {
          it = frag_message
                   .emplace(key,
                            std::make_unique<FragmentedMessage>(frag_info.total_chunks))
                   .first;
        }

        bool ok = it->second->add_chunk(frag_info.chunk_id, payload, payload_size);
        assert(ok);

        if(!it->second->is_complete()) {
          total_messages_handled += 1;
          return false;
        }

        message = it->second->reassemble();

        frag_message.erase(it);
      }
    }

    if(!message.empty()) {
      payload = message.data();
      payload_size = message.size();
      payload_mode = PAYLOAD_COPY;
    }

    // if we have an inline handler and enough time to run it, give it
    //  a go
    if((handler->handler_inline != 0) && (Config::max_inline_message_time > 0) &&
       !work_until.will_expire(Config::max_inline_message_time)) {
      long long t_start = 0;
      if(Config::profile_activemsg_handlers)
        t_start = Clock::current_time_in_nanoseconds();

      if((handler->handler_inline)(
             sender, hdr, payload, payload_size,
             TimeLimit::relative(Config::max_inline_message_time))) {
        if(Config::profile_activemsg_handlers) {
          long long t_end = Clock::current_time_in_nanoseconds();
          handler->stats.record(t_start, t_end);
        }
        if(payload_mode == PAYLOAD_FREE)
          free(const_cast<void *>(payload));
        // see if we need to wake up a thread waiting on a drain
        {
          AutoLock<> al(mutex);
          total_messages_handled += 1;
          if(drain_pending && (todo_oldest == todo_newest) && (handlers_active == 0) &&
             (total_messages_handled >= drain_min_count)) {
            drain_pending = false;
            drain_condvar.broadcast();
          }
        }
        return true;
      }
    }

    // can't handle inline - need to create a Message object for it

    mutex.lock();

    Message *msg = 0;
    size_t hdr_bytes_needed = ((hdr_mode == PAYLOAD_COPY) ? hdr_size : 0);
    size_t payload_bytes_needed = ((payload_mode == PAYLOAD_COPY) ? payload_size : 0);
    while(true) {
      // try to stick this message in the current block
      msg = current_block->append_message(hdr_bytes_needed, payload_bytes_needed);
      if(msg != 0)
        break;

      // do we have a new block we can use?
      if((available_blocks != 0) || (current_block->use_count.load() == 1)) {
        // first release our hold on the current block
        unsigned prev_count = current_block->use_count.fetch_sub(1);

        if(prev_count == 1) {
          // in the (highly unlikely) case that all of its messages have
          //  been deleted, we can just reset it and reuse it
          current_block->reset();
          log_amhandler.debug() << "reusing message block: " << current_block;
        } else {
          assert(available_blocks != 0);
          current_block = available_blocks;
          available_blocks = available_blocks->next_free;
          current_block->next_free = 0;
          num_available_blocks--;
          log_amhandler.debug() << "switching to message block: " << current_block;
        }

        // either way, this must now succeed
        msg = current_block->append_message(hdr_bytes_needed, payload_bytes_needed);
        assert(msg != 0);
        break;
      }

      // no available blocks - drop the mutex while we allocate a new one
      mutex.unlock();
      MessageBlock *block = MessageBlock::new_block(cfg_message_block_size);
      mutex.lock();
      // we don't know what changed while we weren't holding the lock,
      //  so just stick this on the available list and restart
      block->next_free = available_blocks;
      available_blocks = block;
      num_available_blocks++;
    }

    // fill in message structure
    // TODO: let go of lock if copying a large payload?
    {
      msg->next_msg = 0;
      msg->sender = sender;
      msg->handler = handler;
      msg->callback_fnptr = callback_fnptr;
      msg->callback_data1 = callback_data1;
      msg->callback_data2 = callback_data2;

      if(hdr_mode == PAYLOAD_COPY)
        memcpy(msg->hdr, hdr, hdr_size);
      else
        msg->hdr = const_cast<void *>(hdr);
      msg->hdr_size = hdr_size;
      msg->hdr_needs_free = (hdr_mode == PAYLOAD_FREE);

      if(payload_size > 0) {
        if(payload_mode == PAYLOAD_COPY)
          memcpy(msg->payload, payload, payload_size);
        else
          msg->payload = const_cast<void *>(payload);
      }
      msg->payload_size = payload_size;
      msg->payload_needs_free = (payload_mode == PAYLOAD_FREE);
    }

    if(heads[sender]) {
      // tack this on to the existing list
      assert(tails[sender]);
      *(tails[sender]) = msg;
      tails[sender] = &(msg->next_msg);
    } else {
      // this starts a list, and the node needs to be added to the todo list
      heads[sender] = msg;
      tails[sender] = &(msg->next_msg);

      // enqueue if this sender isn't currently being handled
      if(!in_handler[sender]) {
        bool was_empty = todo_oldest == todo_newest;

        todo_list[todo_newest] = sender;
        todo_newest++;
        if(todo_newest > nodes)
          todo_newest = 0;
        assert(todo_newest != todo_oldest); // should never wrap around
        if(sleeper_count > 0)
          condvar.broadcast(); // wake up any sleepers

        if(was_empty && !bgwork_requested.load()) {
          bgwork_requested.store(true);
          make_active();
        }
      }
    }
    mutex.unlock();

    return false; // not handled right away
  }

  void IncomingMessageManager::start_handler_threads(size_t stack_size)
  {
    handler_threads.resize(dedicated_threads);

    Realm::ThreadLaunchParameters tlp;
    tlp.set_stack_size(stack_size);

    for(int i = 0; i < dedicated_threads; i++)
      handler_threads[i] = Realm::Thread::create_kernel_thread<
          IncomingMessageManager, &IncomingMessageManager::handler_thread_loop>(
          this, tlp, *core_rsrv);
  }

  // stalls caller until all incoming messages have been handled
  void IncomingMessageManager::drain_incoming_messages(size_t min_messages_handled)
  {
    AutoLock<> al(mutex);

    while((todo_oldest != todo_newest) || (handlers_active > 0) ||
          (total_messages_handled < min_messages_handled)) {
      drain_min_count = min_messages_handled;
      drain_pending = true;
      drain_condvar.wait();
    }
  }

  void IncomingMessageManager::shutdown(void)
  {
#ifdef DEBUG_REALM
    shutdown_work_item();
#endif

    mutex.lock();
    if(!shutdown_flag) {
      shutdown_flag = true;
      condvar.broadcast(); // wake up any sleepers
    }
    mutex.unlock();

    for(std::vector<Realm::Thread *>::iterator it = handler_threads.begin();
        it != handler_threads.end(); it++) {
      (*it)->join();
      delete(*it);
    }
    handler_threads.clear();
  }

  int IncomingMessageManager::get_messages(IncomingMessageManager::Message *&head,
                                           IncomingMessageManager::Message **&tail,
                                           bool wait)
  {
    AutoLock<> al(mutex);

    while(todo_oldest == todo_newest) {
      // todo list is empty
      if(shutdown_flag || !wait)
        return -1;

#ifdef DEBUG_INCOMING
      printf("incoming message list is empty - sleeping\n");
#endif
      sleeper_count += 1;
      condvar.wait();
      sleeper_count -= 1;
    }

    // pop the oldest entry off the todo list
    int sender = todo_list[todo_oldest];
    todo_oldest++;
    if(todo_oldest > nodes)
      todo_oldest = 0;
    head = heads[sender];
    tail = tails[sender];
    heads[sender] = 0;
    tails[sender] = 0;
    in_handler[sender] = true;
    handlers_active++;
#ifdef DEBUG_INCOMING
    printf("handling incoming messages from %d\n", sender);
#endif
    // if there are other senders with messages waiting, we can request more
    //  background workers right away
    if((todo_oldest != todo_newest) && !bgwork_requested.load()) {
      bgwork_requested.store(true);
      make_active();
    }

    return sender;
  }

  bool IncomingMessageManager::return_messages(int sender, size_t num_handled,
                                               IncomingMessageManager::Message *head,
                                               IncomingMessageManager::Message **tail)
  {
    AutoLock<> al(mutex);
    total_messages_handled += num_handled;
    in_handler[sender] = false;
    handlers_active--;

    bool enqueue_needed = false;
    if(heads[sender] != 0) {
      // list was non-empty
      if(head != 0) {
        // prepend on list
        *tail = heads[sender];
        heads[sender] = head;
      }
      // in in-order mode, we hadn't enqueued this sender, so do that now
      enqueue_needed = true;
    } else {
      if(head != 0) {
        heads[sender] = head;
        tails[sender] = tail;
        enqueue_needed = true;
      }
    }

    bool now_active = false;
    if(enqueue_needed) {
      bool was_empty = todo_oldest == todo_newest;

      todo_list[todo_newest] = sender;
      todo_newest++;
      if(todo_newest > nodes)
        todo_newest = 0;
      assert(todo_newest != todo_oldest); // should never wrap around
      if(sleeper_count > 0)
        condvar.broadcast(); // wake up any sleepers

      if(was_empty && !bgwork_requested.load()) {
        bgwork_requested.store(true);
        now_active = true;
      }
    }

    // was somebody waiting for the queue to go (perhaps temporarily) empty?
    if(drain_pending && (todo_oldest == todo_newest) && (handlers_active == 0) &&
       (total_messages_handled >= drain_min_count)) {
      drain_pending = false;
      drain_condvar.broadcast();
    }

    return now_active;
  }

  bool IncomingMessageManager::do_work(TimeLimit work_until)
  {
    // now that we've been called, our previous request for bgwork has been
    //  granted and we will need another one if/when more work comes
    // it's ok if this races with other threads that are adding/getting messages
    //  because we'll do the request ourselves below in that case
    bgwork_requested.store(false);

    Message *current_msg = 0;
    Message **current_tail = 0;
    int sender = get_messages(current_msg, current_tail, false /*!wait*/);

    // we're here because there was work to do, so an empty list is bad unless
    //  there are also dedicated threads that might have grabbed it
    if(sender == -1) {
      assert(dedicated_threads > 0);
      return false;
    }

    ThreadLocal::in_message_handler = true;

    Message *skipped_messages = 0;
    Message **skipped_tail = &skipped_messages;
    size_t num_handled = 0;

    while(current_msg) {
      Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
      int timing_idx = detailed_message_timing
                           .get_next_index(); // grab this while we still hold the lock
      CurrentTime start_time;
#endif
      long long t_start = 0;
      bool do_profile = Config::profile_activemsg_handlers;

      // do we have a handler that understands time limits?
      if(current_msg->handler->handler != 0) {
        if(do_profile)
          t_start = Clock::current_time_in_nanoseconds();

        (current_msg->handler->handler)(current_msg->sender, current_msg->hdr,
                                        current_msg->payload, current_msg->payload_size,
                                        work_until);
      } else {
        // estimate how long this handler will take, clamping at a
        //  semi-arbitrary 20us
        long long t_estimate = 20000;
        {
          size_t num = current_msg->handler->stats.sum.load();
          size_t den = current_msg->handler->stats.count.load();
          if(num < (den * t_estimate))
            t_estimate = num / den;
        }
        if(work_until.will_expire(t_estimate)) {
          // skip this message instead of handling it now
          *skipped_tail = current_msg;
          skipped_tail = &current_msg->next_msg;
          current_msg = current_msg->next_msg;
          // skipping things can take time too, so check if we're
          //  completely out of time
          if(work_until.is_expired())
            break;
          continue;
        }

        // always profile notimeout handlers
        do_profile = true;
        t_start = Clock::current_time_in_nanoseconds();

        (current_msg->handler->handler_notimeout)(current_msg->sender, current_msg->hdr,
                                                  current_msg->payload,
                                                  current_msg->payload_size);
      }

      long long t_end = 0;
      if(do_profile)
        t_end = Clock::current_time_in_nanoseconds();

      if(current_msg->callback_fnptr)
        (current_msg->callback_fnptr)(current_msg->sender, current_msg->callback_data1,
                                      current_msg->callback_data2);

      if(do_profile)
        current_msg->handler->stats.record(t_start, t_end);
#ifdef DETAILED_MESSAGE_TIMING
      detailed_message_timing.record(timing_idx, current_msg->get_peer(),
                                     current_msg->get_msgid(),
                                     -4, // 0xc - flagged as an incoming message,
                                     current_msg->get_msgsize(),
                                     count++, // how many messages we handle in a batch
                                     start_time, CurrentTime());
#endif
      // recycle message
      current_msg->block->recycle_message(current_msg, this);

      current_msg = next_msg;
      num_handled += 1;

      // do we need to stop early?
      if(current_msg && work_until.is_expired())
        break;
    }

    ThreadLocal::in_message_handler = false;

    // anything we didn't get to goes on the end of the skipped list
    if(current_msg) {
      *skipped_tail = current_msg;
      skipped_tail = current_tail;
    } else
      *skipped_tail = 0;

    // put back whatever we had left, if anything - request requeue if needed
    return return_messages(sender, num_handled, skipped_messages, skipped_tail);
  }

  void IncomingMessageManager::handler_thread_loop(void)
  {
    // this thread is ALWAYS in a handler
    ThreadLocal::in_message_handler = true;

    while(true) {
      Message *current_msg = 0;
      Message **current_tail = 0;
      int sender = get_messages(current_msg, current_tail, true /*wait*/);
      if(sender == -1) {
#ifdef DEBUG_INCOMING
        printf("received empty list - assuming shutdown!\n");
#endif
        break;
      }
#ifdef DETAILED_MESSAGE_TIMING
      int count = 0;
#endif
      size_t num_handled = 0;
      while(current_msg) {
        Message *next_msg = current_msg->next_msg;
#ifdef DETAILED_MESSAGE_TIMING
        int timing_idx = detailed_message_timing
                             .get_next_index(); // grab this while we still hold the lock
        CurrentTime start_time;
#endif
        long long t_start = 0;
        if(Config::profile_activemsg_handlers)
          t_start = Clock::current_time_in_nanoseconds();

        if(current_msg->handler->handler != 0)
          (current_msg->handler->handler)(current_msg->sender, current_msg->hdr,
                                          current_msg->payload, current_msg->payload_size,
                                          TimeLimit());
        else
          (current_msg->handler->handler_notimeout)(current_msg->sender, current_msg->hdr,
                                                    current_msg->payload,
                                                    current_msg->payload_size);

        long long t_end = 0;
        if(Config::profile_activemsg_handlers)
          t_end = Clock::current_time_in_nanoseconds();

        if(current_msg->callback_fnptr)
          (current_msg->callback_fnptr)(current_msg->sender, current_msg->callback_data1,
                                        current_msg->callback_data2);

        if(Config::profile_activemsg_handlers)
          current_msg->handler->stats.record(t_start, t_end);
#ifdef DETAILED_MESSAGE_TIMING
        detailed_message_timing.record(timing_idx, current_msg->get_peer(),
                                       current_msg->get_msgid(),
                                       -4, // 0xc - flagged as an incoming message,
                                       current_msg->get_msgsize(),
                                       count++, // how many messages we handle in a batch
                                       start_time, CurrentTime());
#endif
        // recycle message
        current_msg->block->recycle_message(current_msg, this);

        current_msg = next_msg;
        num_handled += 1;
      }
      // we always handle all the messages, but still indicate we're done
      return_messages(sender, num_handled, 0, 0);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // multicast envelope and bounded-radix forwarding (plan sections 7.3 and 7.4)
  //

  Logger log_multicast("multicast");

  MulticastMetricsSink::~MulticastMetricsSink(void) {}

  MulticastTransport::~MulticastTransport(void) {}

  MulticastCompletionCallback::~MulticastCompletionCallback(void) {}

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastFatalContext
  //

  void MulticastFatalContext::describe(std::ostream &os) const
  {
    os << rule << " (local=" << local_node << " sender=" << sender
       << " origin=" << origin_node << " multicast_id=" << multicast_id
       << " message_id=" << original_message_id
       << " encoding_kind=" << target_encoding_kind
       << " encoding_size=" << target_encoding_size
       << " header_size=" << original_header_size
       << " payload_size=" << original_payload_size
       << " received_bytes=" << received_payload_size << " depth=" << depth
       << " decode_status=" << status << ")";
  }

  std::string MulticastFatalContext::to_string(void) const
  {
    std::ostringstream os;
    describe(os);
    return os.str();
  }

  namespace {

    class DefaultMulticastFatalReporter : public MulticastFatalReporter {
    public:
      virtual void report(const MulticastFatalContext &ctx)
      {
        log_multicast.fatal() << "multicast protocol violation: " << ctx.to_string();
        abort();
      }
    };

    DefaultMulticastFatalReporter default_multicast_fatal_reporter;

    // installed only by tests, and only while no multicast traffic is in flight, so a
    //  plain pointer is sufficient here
    MulticastFatalReporter *installed_multicast_fatal_reporter = nullptr;

  }; // namespace

  MulticastFatalReporter *set_multicast_fatal_reporter(MulticastFatalReporter *reporter)
  {
    MulticastFatalReporter *previous = installed_multicast_fatal_reporter;
    installed_multicast_fatal_reporter = reporter;
    return previous;
  }

  MulticastFatalReporter *get_multicast_fatal_reporter(void)
  {
    return installed_multicast_fatal_reporter;
  }

  void report_multicast_fatal(const MulticastFatalContext &ctx)
  {
    if(installed_multicast_fatal_reporter != nullptr) {
      installed_multicast_fatal_reporter->report(ctx);
      return;
    }
    default_multicast_fatal_reporter.report(ctx);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastCompletionState
  //

  MulticastCompletionState::MulticastCompletionState(void) {}

  MulticastCompletionState::~MulticastCompletionState(void)
  {
    // plan section 2 (final bullet) and 7.1: nothing may outlive the multicasts it
    //  belongs to, so a nonempty table here means a subtree never acknowledged
    assert(pending.empty() &&
           "multicast completion state outlived the multicasts it was tracking");
  }

  void MulticastCompletionState::begin_origin(NodeID origin, uint64_t multicast_id,
                                              size_t outstanding,
                                              MulticastCompletionCallback *callback)
  {
    assert(outstanding > 0);
    assert(callback != nullptr);
    MulticastCompletionKey key;
    key.origin = origin;
    key.multicast_id = multicast_id;

    Record rec;
    rec.parent = origin;
    rec.outstanding = outstanding;
    rec.callback = callback;
    rec.is_origin = true;

    AutoLock<> al(mutex);
    bool inserted = pending.insert(std::make_pair(key, rec)).second;
    assert(inserted && "duplicate multicast completion record at the origin");
    (void)inserted;
    if(pending.size() > peak)
      peak = pending.size();
  }

  void MulticastCompletionState::begin_relay(NodeID origin, uint64_t multicast_id,
                                             NodeID parent, size_t outstanding)
  {
    assert(outstanding > 0);
    MulticastCompletionKey key;
    key.origin = origin;
    key.multicast_id = multicast_id;

    Record rec;
    rec.parent = parent;
    rec.outstanding = outstanding;
    rec.callback = nullptr;
    rec.is_origin = false;

    AutoLock<> al(mutex);
    bool inserted = pending.insert(std::make_pair(key, rec)).second;
    assert(inserted && "duplicate multicast completion record at a relay");
    (void)inserted;
    if(pending.size() > peak)
      peak = pending.size();
  }

  MulticastCompletionState::Notification
  MulticastCompletionState::note_completion(NodeID origin, uint64_t multicast_id)
  {
    MulticastCompletionKey key;
    key.origin = origin;
    key.multicast_id = multicast_id;

    Notification result;
    {
      AutoLock<> al(mutex);
      std::map<MulticastCompletionKey, Record>::iterator it = pending.find(key);
      if(it == pending.end()) {
        result.action = Notification::UNKNOWN;
        return result;
      }
      assert(it->second.outstanding > 0);
      if(--it->second.outstanding > 0)
        return result; // NOTHING

      // the subtree is complete: reclaim the record BEFORE telling anyone, so that no
      //  state survives the acknowledgement (plan section 7.5)
      result.action = (it->second.is_origin ? Notification::INVOKE_CALLBACK
                                            : Notification::ACK_PARENT);
      result.parent = it->second.parent;
      result.callback = it->second.callback;
      pending.erase(it);
    }
    return result;
  }

  size_t MulticastCompletionState::num_pending(void) const
  {
    AutoLock<> al(mutex);
    return pending.size();
  }

  size_t MulticastCompletionState::peak_pending(void) const
  {
    AutoLock<> al(mutex);
    return peak;
  }

  void MulticastCompletionState::reset_peak(void)
  {
    AutoLock<> al(mutex);
    peak = pending.size();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // forwarding helpers
  //

  namespace {

#ifdef DEBUG_REALM
    // Plan section 21.1 requires partition overlap (or a slice that is not actually a
    //  subset of what we were asked to forward) to be fatal in debug builds.  The check
    //  is exact and linear in the number of runs: walking the slices in order, every
    //  run must start after the previous one ended and must fit inside a run of the
    //  source, and the cardinalities must add up.  Containment plus disjointness plus
    //  equal cardinality is exactly "the slices partition the source".
    void validate_partition(const MulticastTargetSet &source,
                            const std::vector<MulticastTargetSet> &slices)
    {
      typedef MulticastTargetSet::Range Range;
      size_t src_idx = 0;
      size_t total = 0;
      bool have_prev = false;
      NodeID prev_last = 0;
      for(size_t i = 0; i < slices.size(); i++) {
        assert(!slices[i].empty() && "multicast partition produced an empty slice");
        const std::vector<Range> &runs = slices[i].ranges();
        for(size_t j = 0; j < runs.size(); j++) {
          assert((!have_prev || (runs[j].first > prev_last)) &&
                 "multicast partition slices overlap");
          while((src_idx < source.num_ranges()) &&
                (source.ranges()[src_idx].last < runs[j].first))
            src_idx++;
          assert((src_idx < source.num_ranges()) &&
                 "multicast partition slice is not part of the target set");
          assert((runs[j].first >= source.ranges()[src_idx].first) &&
                 (runs[j].last <= source.ranges()[src_idx].last) &&
                 "multicast partition slice is not part of the target set");
          prev_last = runs[j].last;
          have_prev = true;
        }
        total += slices[i].size();
      }
      assert((total == source.size()) && "multicast partition dropped targets");
    }
#endif

    // Everything that is identical for every child envelope of one forwarding step -
    //  the original message, the multicast identity, and whether this multicast is
    //  completion tracked.
    struct OutboundMulticast {
      NodeID origin = 0;
      uint64_t multicast_id = 0;
      ActiveMessageHandlerTable::MessageID msgid = 0;
      const void *hdr = nullptr;
      size_t hdr_size = 0;
      const void *payload = nullptr;
      size_t payload_size = 0;
      uint32_t flags = 0;
      unsigned depth = 0;
      // node the child subtree must acknowledge to, i.e. the node doing the sending -
      //  only meaningful when COMPLETION_TRACKED is set in 'flags'
      NodeID completion_parent = 0;

      bool tracked(void) const
      {
        return ((flags & MulticastEnvelopeFlags::COMPLETION_TRACKED) != 0);
      }
    };

    // Builds one envelope for 'slice' and hands it to the transport.  The relay is the
    //  first node of the slice (plan section 7.3 step 3).
    void send_one_slice(MulticastTransport &transport, const MulticastTargetSet &slice,
                        const OutboundMulticast &out, MulticastMetricsSink *metrics)
    {
      assert(!slice.empty());

      EncodedMulticastTargets enc =
          EncodedMulticastTargets::encode(slice, transport.num_nodes());
      if(metrics != nullptr)
        metrics->record_encoding_choice(enc.kind());

      // a fire-and-forget multicast carries no completion metadata whatsoever
      const size_t comp_size =
          (out.tracked()
               ? MulticastWire::varint_size(static_cast<uint64_t>(out.completion_parent))
               : 0);

      MulticastEnvelopeMessage env;
      env.multicast_id = out.multicast_id;
      env.origin_node = out.origin;
      env.original_payload_size = static_cast<uint32_t>(out.payload_size);
      env.target_encoding_size = static_cast<uint32_t>(enc.bytes());
      env.completion_size = static_cast<uint32_t>(comp_size);
      env.flags = out.flags;
      env.original_message_id = out.msgid;
      env.original_header_size = static_cast<uint16_t>(out.hdr_size);
      env.depth = out.depth;
      env.target_encoding_kind = static_cast<unsigned char>(enc.kind());

      // the variable portion is copied here, which is what makes the caller's
      //  PAYLOAD_KEEP lifetime guarantee hold across commit() (plan section 7.5)
      std::vector<unsigned char> buf;
      buf.reserve(enc.bytes() + out.hdr_size + out.payload_size + comp_size);
      buf.insert(buf.end(), enc.wire_bytes().begin(), enc.wire_bytes().end());
      if(out.hdr_size > 0) {
        const unsigned char *hdr_bytes = static_cast<const unsigned char *>(out.hdr);
        buf.insert(buf.end(), hdr_bytes, hdr_bytes + out.hdr_size);
      }
      if(out.payload_size > 0) {
        const unsigned char *body = static_cast<const unsigned char *>(out.payload);
        buf.insert(buf.end(), body, body + out.payload_size);
      }
      if(comp_size > 0)
        MulticastWire::append_varint(buf, static_cast<uint64_t>(out.completion_parent));

      transport.send_envelope(slice.first_node(), env, buf.data(), buf.size());
    }

    // Partitions 'remaining' into at most R slices WITHOUT sending anything.  Splitting
    //  planning from sending matters for completion tracking: the number of children
    //  has to be known (and the record retained) before the first child can possibly
    //  acknowledge.
    void plan_children(MulticastTransport &transport, const MulticastTargetSet &remaining,
                       std::vector<MulticastTargetSet> &slices)
    {
      slices.clear();
      if(remaining.empty())
        return;

      const size_t radix = transport.radix();
      assert(radix >= 1);

      remaining.partition(radix, slices);
#ifdef DEBUG_REALM
      validate_partition(remaining, slices);
#endif
      assert(slices.size() <= radix);
    }

  }; // namespace

  ////////////////////////////////////////////////////////////////////////
  //
  // class MulticastForwarder
  //

  /*static*/ uint64_t MulticastForwarder::next_multicast_id(void)
  {
    // only the local half of (origin_node, counter) - the origin node is carried
    //  separately in the envelope, so this only has to be unique per origin
    static atomic<uint64_t> counter(0);
    return counter.fetch_add(1) + 1;
  }

  namespace {

    // Post-handler notification for a local delivery that could not be handled inline.
    //  The token is heap allocated because IncomingMessageManager only carries two
    //  uintptr_t of callback data, and it is captured rather than rederived because
    //  this runs on a handler thread, potentially long after dispatch_local returned.
    void deferred_local_delivery_callback(NodeID /*sender*/,
                                          IncomingMessageManager::CallbackData data1,
                                          IncomingMessageManager::CallbackData /*data2*/)
    {
      MulticastCompletionToken *token =
          reinterpret_cast<MulticastCompletionToken *>(data1);
      MulticastCompletionToken copy = *token;
      delete token;
      MulticastForwarder::settle(copy);
    }

  }; // namespace

  /*static*/ void MulticastForwarder::settle(const MulticastCompletionToken &token)
  {
    assert(token.state != nullptr);
    MulticastCompletionState::Notification note =
        token.state->note_completion(token.origin, token.multicast_id);

    switch(note.action) {
    case MulticastCompletionState::Notification::NOTHING:
      break;

    case MulticastCompletionState::Notification::ACK_PARENT:
    {
      // exactly one acknowledgement per subtree, sent after the record was reclaimed
      assert(token.transport != nullptr);
      token.transport->send_ack(token.local, note.parent, token.origin,
                                token.multicast_id);
      break;
    }

    case MulticastCompletionState::Notification::INVOKE_CALLBACK:
    {
      // the origin's callback runs exactly once, after every target has HANDLED the
      //  message, and the state is already gone (plan section 7.5)
      assert(note.callback != nullptr);
      note.callback->invoke();
      delete note.callback;
      break;
    }

    case MulticastCompletionState::Notification::UNKNOWN:
    {
      MulticastFatalContext ctx;
      ctx.local_node = token.local;
      ctx.sender = token.local;
      ctx.origin_node = token.origin;
      ctx.multicast_id = token.multicast_id;
      ctx.rule = "multicast acknowledgement does not match any multicast in flight";
      report_multicast_fatal(ctx);
      break;
    }
    }
  }

  /*static*/ void MulticastForwarder::handle_ack(MulticastTransport &transport,
                                                 NodeID sender,
                                                 const MulticastAckMessage &ack)
  {
    MulticastCompletionToken token;
    token.transport = &transport;
    token.state = &transport.completion_state();
    token.local = transport.my_node_id();
    token.origin = ack.origin_node;
    token.multicast_id = ack.multicast_id;
    (void)sender;
    settle(token);
  }

  /*static*/ bool MulticastForwarder::dispatch_local(
      IncomingMessageManager *manager, NodeID sender,
      ActiveMessageHandlerTable::MessageID msgid, const void *hdr, size_t hdr_size,
      const void *payload, size_t payload_size, TimeLimit work_until,
      const MulticastCompletionToken *completion)
  {
    assert(manager != nullptr);

    IncomingMessageManager::CallbackFnptr fnptr = nullptr;
    IncomingMessageManager::CallbackData data1 = 0;
    MulticastCompletionToken *token = nullptr;
    if(completion != nullptr) {
#ifdef DEBUG_REALM
      // A message type that carries its own FragmentInfo is reassembled inside
      //  add_incoming_message, and an incomplete fragment returns without ever reaching
      //  a handler - so "handled" would be unobservable and the notification would be
      //  lost.  Such a type must not be the ORIGINAL message of a completion-tracked
      //  multicast; multicast its unfragmented form and let the envelope be chunked.
      ActiveMessageHandlerTable::HandlerEntry *entry =
          activemsg_handler_table.lookup_message_handler(msgid);
      assert((entry != nullptr) && !entry->extract_frag_info.has_value() &&
             "completion-tracked multicast of a fragment-carrying message type");
#endif
      token = new MulticastCompletionToken(*completion);
      fnptr = &deferred_local_delivery_callback;
      data1 = reinterpret_cast<IncomingMessageManager::CallbackData>(token);
    }

    // Reuse of the ordinary incoming path is deliberate: it looks the handler up by
    //  message ID in the same ActiveMessageHandlerTable, honors an inline handler when
    //  there is time for one, and otherwise queues the message for a handler thread
    //  with the same TimeLimit semantics as a message that arrived off the wire.  No
    //  handler-signature detection is duplicated here (plan section 7.4).
    bool handled = manager->add_incoming_message(
        sender, msgid, hdr, hdr_size, PAYLOAD_COPY, payload, payload_size, PAYLOAD_COPY,
        fnptr, data1, 0, work_until);

    if(token != nullptr) {
      if(handled) {
        // handled inline, so the post-handler callback will NOT run - settle here
        MulticastCompletionToken copy = *token;
        delete token;
        settle(copy);
      }
      // otherwise deferred_local_delivery_callback owns and frees it
    }

    return handled;
  }

  /*static*/ void MulticastForwarder::send(
      MulticastTransport &transport, const MulticastTargetSet &targets,
      ActiveMessageHandlerTable::MessageID msgid, const void *hdr, size_t hdr_size,
      const void *payload, size_t payload_size, TimeLimit work_until,
      MulticastMetricsSink *metrics, MulticastCompletionCallback *on_remote_complete)
  {
    // the envelope's length fields are 16/32 bits wide
    assert(hdr_size <= 0xffff);
    assert(payload_size <= 0xffffffffULL);

    // plan section 7.5: an empty target set is a successful no-op.  Every one of the
    //  zero targets has trivially already handled the message, so a requested remote
    //  completion fires immediately and no state is created.
    if(targets.empty()) {
      if(on_remote_complete != nullptr) {
        on_remote_complete->invoke();
        delete on_remote_complete;
      }
      return;
    }

    const NodeID local = transport.my_node_id();
    const NodeID num_nodes = transport.num_nodes();

    if(!targets.fits_node_count(num_nodes)) {
      MulticastFatalContext ctx;
      ctx.local_node = local;
      ctx.sender = local;
      ctx.origin_node = local;
      ctx.original_message_id = msgid;
      ctx.original_header_size = hdr_size;
      ctx.original_payload_size = payload_size;
      ctx.status = MulticastDecodeStatus::NODE_OUT_OF_RANGE;
      ctx.rule = "multicast target set contains a node outside the configured node "
                 "count";
      report_multicast_fatal(ctx);
      // nothing was sent, so the callback must not claim that everyone handled it
      delete on_remote_complete;
      return;
    }

    // 1. if the origin is a target, arrange local delivery and remove it from the
    //    forwarding set (plan section 7.3)
    MulticastTargetSet remaining(targets);
    const bool deliver_here = remaining.remove(local);

    OutboundMulticast out;
    out.origin = local;
    out.msgid = msgid;
    out.hdr = hdr;
    out.hdr_size = hdr_size;
    out.payload = payload;
    out.payload_size = payload_size;
    out.depth = 1;
    out.completion_parent = local;
    // every multicast gets the globally unique (origin_node, counter) identity plan
    //  section 7.4 requires, whether or not it is completion tracked - it is what a
    //  fatal diagnostic names, and what an acknowledgement is keyed on
    out.multicast_id = next_multicast_id();

    // A fire-and-forget multicast sets no flag, puts no completion metadata on the wire
    //  and creates no state anywhere (plan section 7.5).
    const bool tracked = (on_remote_complete != nullptr);
    if(tracked)
      out.flags |= MulticastEnvelopeFlags::COMPLETION_TRACKED;

    // decide the whole shape of the first hop before anything is transmitted, so that
    //  the origin's record can be retained before a child can possibly acknowledge
    const bool unicast_fast_path =
        ((remaining.size() == 1) && transport.can_send_original(hdr_size, payload_size));
    std::vector<MulticastTargetSet> slices;
    if(!remaining.empty() && !unicast_fast_path)
      plan_children(transport, remaining, slices);

    const size_t first_hops = (unicast_fast_path ? 1 : slices.size());

    MulticastCompletionToken token;
    if(tracked) {
      token.transport = &transport;
      token.state = &transport.completion_state();
      token.local = local;
      token.origin = local;
      token.multicast_id = out.multicast_id;
      // one unit per first hop, plus one for our own delivery if we are a target
      token.state->begin_origin(local, out.multicast_id,
                                first_hops + (deliver_here ? 1 : 0), on_remote_complete);
    }

    if(unicast_fast_path) {
      // plan section 7.5: a singleton target uses the ordinary unicast fast path.  This
      //  is only safe at the origin, where the local node already IS the sender the
      //  handler must see.  A tracked send rides the ordinary active-message remote
      //  completion, which by definition fires once the target has handled it.
      transport.send_original(remaining.first_node(), msgid, hdr, hdr_size, payload,
                              payload_size, (tracked ? &token : nullptr));
    } else {
      // 2-4. one envelope per slice, addressed to that slice's first node
      for(size_t i = 0; i < slices.size(); i++)
        send_one_slice(transport, slices[i], out, metrics);
    }
    if((metrics != nullptr) && (first_hops > 0))
      metrics->record_first_hops(first_hops);

    // forward-before-deliver (plan section 7.3 step 4) applies at the origin too: a
    //  handler such as runtime shutdown may stop progress
    if(deliver_here)
      transport.deliver_local(local, msgid, hdr, hdr_size, payload, payload_size,
                              work_until, (tracked ? &token : nullptr));
  }

  /*static*/ void MulticastForwarder::forward(MulticastTransport &transport,
                                              NodeID sender,
                                              const MulticastEnvelopeMessage &env,
                                              const void *payload, size_t payload_size,
                                              TimeLimit work_until,
                                              MulticastMetricsSink *metrics)
  {
    const NodeID local = transport.my_node_id();
    const NodeID num_nodes = transport.num_nodes();

    MulticastFatalContext ctx;
    ctx.local_node = local;
    ctx.sender = sender;
    ctx.origin_node = env.origin_node;
    ctx.multicast_id = env.multicast_id;
    ctx.original_message_id = env.original_message_id;
    ctx.target_encoding_kind = env.target_encoding_kind;
    ctx.target_encoding_size = env.target_encoding_size;
    ctx.original_header_size = env.original_header_size;
    ctx.original_payload_size = env.original_payload_size;
    ctx.received_payload_size = payload_size;
    ctx.depth = env.depth;

    // the four variable-length pieces must account for exactly the bytes we received -
    //  nothing below is sized from an unvalidated remote length (plan section 22)
    const size_t targets_bytes = env.target_encoding_size;
    const size_t hdr_bytes = env.original_header_size;
    const size_t body_bytes = env.original_payload_size;
    const size_t comp_bytes = env.completion_size;
    if((targets_bytes + hdr_bytes + body_bytes + comp_bytes) != payload_size) {
      ctx.status = MulticastDecodeStatus::TRUNCATED;
      ctx.rule = "multicast envelope length fields do not match the received payload";
      report_multicast_fatal(ctx);
      return;
    }

    if((env.flags & ~static_cast<uint32_t>(MulticastEnvelopeFlags::ALL_KNOWN)) != 0) {
      ctx.rule = "multicast envelope carries unknown flags";
      report_multicast_fatal(ctx);
      return;
    }

    const unsigned char *base = static_cast<const unsigned char *>(payload);

    // 0. completion metadata is present if and only if this multicast is tracked, and
    //    names the node this whole subtree must acknowledge to (plan section 7.5)
    const bool tracked = ((env.flags & MulticastEnvelopeFlags::COMPLETION_TRACKED) != 0);
    NodeID parent = 0;
    if(tracked) {
      size_t pos = 0;
      uint64_t raw = 0;
      MulticastDecodeStatus cstat = MulticastWire::read_varint(
          base + targets_bytes + hdr_bytes + body_bytes, comp_bytes, pos, raw);
      if((cstat != MulticastDecodeStatus::OK) || (pos != comp_bytes) ||
         (raw >= static_cast<uint64_t>(num_nodes))) {
        ctx.status = ((cstat != MulticastDecodeStatus::OK)
                          ? cstat
                          : MulticastDecodeStatus::NODE_OUT_OF_RANGE);
        ctx.rule = "malformed multicast completion metadata";
        report_multicast_fatal(ctx);
        return;
      }
      parent = static_cast<NodeID>(raw);
      if(parent != sender) {
        ctx.rule = "multicast completion metadata names a parent that is not the sender";
        report_multicast_fatal(ctx);
        return;
      }
    } else if(comp_bytes != 0) {
      ctx.rule = "multicast envelope carries completion metadata without the completion "
                 "flag";
      report_multicast_fatal(ctx);
      return;
    }

    // 1a. decode and fully validate the target slice
    MulticastTargetSet slice;
    MulticastDecodeStatus status =
        EncodedMulticastTargets::decode(base, targets_bytes, num_nodes, slice);
    if(status != MulticastDecodeStatus::OK) {
      ctx.status = status;
      ctx.rule = "malformed multicast target encoding";
      report_multicast_fatal(ctx);
      return;
    }
    if(base[0] != env.target_encoding_kind) {
      ctx.status = MulticastDecodeStatus::UNKNOWN_KIND;
      ctx.rule = "multicast envelope encoding kind disagrees with its payload";
      report_multicast_fatal(ctx);
      return;
    }

    // 1b. validate that the relay is included in the received slice (plan sections 7.3
    //     and 21.1)
    if(!slice.contains(local)) {
      ctx.rule = "multicast relay is not a member of the slice it was sent";
      report_multicast_fatal(ctx);
      return;
    }

    // 2. save the original sender from the envelope - a relay must never become the
    //    apparent sender just because it transmitted the final hop (plan section 7.4)
    const NodeID origin = env.origin_node;

    if(metrics != nullptr)
      metrics->record_tree_depth(env.depth);

    // 3. remove ourselves from the slice
    bool was_present = slice.remove(local);
    assert(was_present);
    (void)was_present;

    OutboundMulticast out;
    out.origin = origin;
    out.multicast_id = env.multicast_id;
    out.msgid = env.original_message_id;
    out.hdr = base + targets_bytes;
    out.hdr_size = hdr_bytes;
    out.payload = (body_bytes > 0) ? (base + targets_bytes + hdr_bytes) : nullptr;
    out.payload_size = body_bytes;
    out.flags = env.flags;
    out.depth = env.depth + 1;
    // a child acknowledges to US, not to whoever sent us this envelope
    out.completion_parent = local;

    // 4. partition and enqueue child envelopes BEFORE invoking the original local
    //    handler.  Forward-before-deliver matters for handlers such as runtime shutdown
    //    whose handler may stop progress.  We are on the ordinary (non-inline) handler
    //    path, so these sends are not recursive forwarding out of an inline handler
    //    (plan section 22).
    std::vector<MulticastTargetSet> slices;
    plan_children(transport, slice, slices);

    // The relay's transient record: its parent, its own (not yet finished) local
    //  delivery, and one outstanding acknowledgement per child.  It has to exist before
    //  the first child envelope goes out, because a child can acknowledge at any time
    //  after that (plan section 7.5).
    MulticastCompletionToken token;
    if(tracked) {
      token.transport = &transport;
      token.state = &transport.completion_state();
      token.local = local;
      token.origin = origin;
      token.multicast_id = env.multicast_id;
      token.state->begin_relay(origin, env.multicast_id, parent, slices.size() + 1);
    }

    for(size_t i = 0; i < slices.size(); i++)
      send_one_slice(transport, slices[i], out, metrics);

    // 5. deliver the original typed message locally exactly once
    transport.deliver_local(origin, env.original_message_id, out.hdr, out.hdr_size,
                            out.payload, out.payload_size, work_until,
                            (tracked ? &token : nullptr));
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastEnvelopeMessage
  //

  /*static*/ void MulticastEnvelopeMessage::handle_message(
      NodeID sender, const MulticastEnvelopeMessage &hdr, const void *payload,
      size_t payload_size, TimeLimit work_until)
  {
    MulticastForwarder::forward(get_runtime_multicast_transport(), sender, hdr, payload,
                                payload_size, work_until);
  }

  ActiveMessageHandlerReg<MulticastEnvelopeMessage> multicast_envelope_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MulticastAckMessage
  //

  /*static*/ void MulticastAckMessage::handle_message(NodeID sender,
                                                      const MulticastAckMessage &hdr,
                                                      const void * /*payload*/,
                                                      size_t /*payload_size*/,
                                                      TimeLimit /*work_until*/)
  {
    MulticastForwarder::handle_ack(get_runtime_multicast_transport(), sender, hdr);
  }

  ActiveMessageHandlerReg<MulticastAckMessage> multicast_ack_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // the runtime's multicast transport
  //

  namespace {

    class RuntimeMulticastTransport : public MulticastTransport {
    public:
      virtual NodeID my_node_id(void) const { return Network::my_node_id; }

      virtual NodeID num_nodes(void) const { return Network::max_node_id + 1; }

      virtual size_t radix(void) const
      {
        size_t cached = cached_radix.load();
        if(cached != 0)
          return cached;

        // initially the existing -ll:barrier_radix / barrier_broadcast_radix value
        //  (plan section 7.3).  The answer is only cached once the core module config
        //  actually exists, so a call made before the runtime is up falls back to the
        //  default without poisoning the cache.
        RuntimeImpl *runtime = get_runtime();
        if(runtime != nullptr) {
          ModuleConfig *core = runtime->get_module_config("core");
          if(core != nullptr) {
            int configured = 0;
            if((core->get_property("barrier_broadcast_radix", configured) ==
                REALM_SUCCESS) &&
               (configured >= 1)) {
              cached_radix.store(static_cast<size_t>(configured));
              return static_cast<size_t>(configured);
            }
          }
        }
        return MULTICAST_DEFAULT_RADIX;
      }

      virtual void send_envelope(NodeID relay, const MulticastEnvelopeMessage &env,
                                 const void *payload, size_t payload_bytes)
      {
        // an oversized envelope is fragmented here by the ordinary ActiveMessage
        //  machinery and reassembled before the relay repartitions it
        ActiveMessage<MulticastEnvelopeMessage> amsg(relay, payload_bytes);
        *amsg = env;
        if(payload_bytes > 0)
          amsg.add_payload(payload, payload_bytes);
        amsg.commit();
      }

      virtual bool can_send_original(size_t hdr_size, size_t payload_size) const
      {
        if(payload_size == 0)
          return true;
        // fragmentation of the original message would need its compile-time type, and
        //  all we have here is a message ID - such a send goes through an envelope
        //  instead, which the backend can fragment
        return payload_size <= Network::max_payload_size(hdr_size, nullptr);
      }

      virtual void send_original(NodeID target,
                                 ActiveMessageHandlerTable::MessageID msgid,
                                 const void *hdr, size_t hdr_size, const void *payload,
                                 size_t payload_size,
                                 const MulticastCompletionToken *completion)
      {
        // same sequence ActiveMessage<T> performs, except that the message ID and the
        //  header bytes are chosen at run time rather than by type
        uint64_t storage[32];
        ActiveMessageImpl *impl = Network::create_active_message_impl(
            target, msgid, hdr_size, payload_size, 0, 0, 0, &storage, sizeof(storage));
        memcpy(impl->header_base, hdr, hdr_size);
        if(payload_size > 0)
          memcpy(impl->payload_base, payload, payload_size);
        if(completion != nullptr) {
          // an ordinary remote completion already means "received AND HANDLED by the
          //  target", which for a single target is exactly the aggregate we want
          UnicastCompletionNotifier notifier;
          notifier.token = *completion;
          size_t bytes = sizeof(CompletionCallback<UnicastCompletionNotifier>);
          bytes = (((bytes - 1) / CompletionCallbackBase::ALIGNMENT) + 1) *
                  CompletionCallbackBase::ALIGNMENT;
          void *ptr = impl->add_remote_completion(bytes);
          new(ptr) CompletionCallback<UnicastCompletionNotifier>(notifier);
        }
        impl->commit(payload_size);
        impl->~ActiveMessageImpl();
      }

      virtual void send_ack(NodeID from, NodeID parent, NodeID origin,
                            uint64_t multicast_id)
      {
        assert(from == Network::my_node_id);
        (void)from;
        ActiveMessage<MulticastAckMessage> amsg(parent);
        amsg->multicast_id = multicast_id;
        amsg->origin_node = origin;
        amsg.commit();
      }

      virtual void deliver_local(NodeID origin,
                                 ActiveMessageHandlerTable::MessageID msgid,
                                 const void *hdr, size_t hdr_size, const void *payload,
                                 size_t payload_size, TimeLimit work_until,
                                 const MulticastCompletionToken *completion)
      {
        RuntimeImpl *runtime = get_runtime();
        assert((runtime != nullptr) && (runtime->message_manager != nullptr));
        MulticastForwarder::dispatch_local(runtime->message_manager, origin, msgid, hdr,
                                           hdr_size, payload, payload_size, work_until,
                                           completion);
      }

      virtual MulticastCompletionState &completion_state(void) { return completion; }

    protected:
      // callable form of "one unicast fast-path target has handled the message"
      struct UnicastCompletionNotifier {
        MulticastCompletionToken token;
        void operator()(void) const { MulticastForwarder::settle(token); }
      };

      mutable atomic<size_t> cached_radix{0};
      MulticastCompletionState completion;
    };

    RuntimeMulticastTransport runtime_multicast_transport;

  }; // namespace

  MulticastTransport &get_runtime_multicast_transport(void)
  {
    return runtime_multicast_transport;
  }

}; // namespace Realm
