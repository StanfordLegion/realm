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

#include "realm/event_impl.h"
#include "realm/barrier_impl.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"

#include <algorithm>
#include <cstring>
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/activemsg.h"
#include "realm/multicast.h"

namespace Realm {

  Logger log_barrier("barrier");

  // used in places that don't currently propagate poison but should
  static const bool POISON_FIXME = false;

  ////////////////////////////////////////////////////////////////////////
  //
  // class Barrier
  //

  /*static*/ const Barrier Barrier::NO_BARRIER = {/* zero-initialization */};

  /*static*/ const ::realm_event_gen_t Barrier::MAX_PHASES =
      (::realm_event_gen_t(1) << REALM_EVENT_GENERATION_BITS) - 1;

  /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
                                             ReductionOpID redop_id /*= 0*/,
                                             const void *initial_value /*= 0*/,
                                             size_t initial_value_size /*= 0*/)
  {
    BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id,
                                                    initial_value, initial_value_size);
    Barrier b = impl->current_barrier();

    return b;
  }

  void Barrier::destroy_barrier(void)
  {
    log_barrier.info() << "barrier destruction request: " << *this;
  }

  Barrier Barrier::advance_barrier(void) const
  {
    ID nextid(id);
    EventImpl::gen_t gen = ID(id).barrier_generation() + 1;
#ifdef DEBUG_REALM
    assert(MAX_PHASES <= nextid.barrier_generation().MAXVAL);
#endif
    // return NO_BARRIER if the count overflows
    if(gen > MAX_PHASES) {
      return Barrier::NO_BARRIER;
    }
    nextid.barrier_generation() = ID(id).barrier_generation() + 1;

    Barrier nextgen = nextid.convert<Barrier>();
    nextgen.timestamp = 0;

    return nextgen;
  }

  /*static*/ atomic<Barrier::timestamp_t> BarrierImpl::barrier_adjustment_timestamp(0);

  Barrier Barrier::alter_arrival_count(int delta) const
  {
    // a no-op alteration changes nothing and opens no causal branch, so it
    //  keeps whatever branch this handle was already on
    if(delta == 0) {
      return *this;
    }

    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    // ARRIVAL_PROTOCOL rules 8 and 9, action AL.  The causal timestamp is
    //  minted INSIDE the one critical section that also installs the arrival
    //  floor, enters eager flush and records the message (section 12); it comes
    //  back out so it can go on the returned handle.
    timestamp_t timestamp = impl->alter_arrival_count(ID(id).barrier_generation(), delta);

    Barrier with_ts;
    with_ts.id = id;
    with_ts.timestamp = timestamp;

    return with_ts;
  }

  Barrier Barrier::get_previous_phase(void) const
  {
    ID previd(id);
    EventImpl::gen_t gen = ID(id).barrier_generation();
    // can't back up before generation 0
    previd.barrier_generation() = ((gen > 0) ? (gen - 1) : gen);

    Barrier prevgen = previd.convert<Barrier>();
    prevgen.timestamp = 0;

    return prevgen;
  }

  void Barrier::arrive(unsigned count /*= 1*/, Event wait_on /*= Event::NO_EVENT*/,
                       const void *reduce_value /*= 0*/,
                       size_t reduce_value_size /*= 0*/) const
  {
    // arrival uses the timestamp stored in this barrier object
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(ID(id).barrier_generation(), -int(count), timestamp, wait_on,
                         Network::my_node_id, reduce_value, reduce_value_size,
                         TimeLimit::responsive());
  }

  bool Barrier::get_result(void *value, size_t value_size) const
  {
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    return impl->get_result(ID(id).barrier_generation(), value, value_size);
  }

  /*static*/ BarrierImpl *BarrierImpl::create_barrier(unsigned expected_arrivals,
                                                      ReductionOpID redopid,
                                                      const void *initial_value /*= 0*/,
                                                      size_t initial_value_size /*= 0*/)
  {
    BarrierImpl *impl = get_runtime()->local_barrier_free_list->alloc_entry();
    assert(impl);
    assert(ID(impl->me).is_barrier());

    // set the arrival count
    impl->base_arrival_count = expected_arrivals;

    if(redopid == 0) {
      assert(initial_value_size == 0);
      impl->redop_id = 0;
      impl->redop = 0;
      impl->initial_value = 0;
      impl->value_capacity = 0;
      impl->final_values.clear();
      impl->legacy.reset();
    } else {
      // reduction barriers stay on the legacy (non-scalable) path
      impl->legacy.reset(new BarrierImpl::LegacyReductionState);
      impl->redop_id = redopid; // keep the ID too so we can share it
      impl->redop = get_runtime()->reduce_op_table.get(redopid, 0);
      if(impl->redop == 0) {
        log_barrier.fatal() << "no reduction op registered for ID " << redopid;
        abort();
      }

      assert(initial_value != 0);
      assert(initial_value_size == impl->redop->sizeof_lhs);

      impl->initial_value = std::make_unique<char[]>(initial_value_size);
      memcpy(impl->initial_value.get(), initial_value, initial_value_size);

      impl->value_capacity = 0;
      impl->final_values.clear();
    }

    // and let the barrier rearm as many times as necessary without being released
    // impl->free_generation = (unsigned)-1;

    log_barrier.info() << "barrier created: " << impl->me << "/"
                       << impl->generation.load()
                       << " base_count=" << impl->base_arrival_count
                       << " redop=" << redopid;
    return impl;
  }

  // active messages

  namespace {
    // NOTIFICATION_PROTOCOL rule 5 - THE PULL.  One message type, two reply
    //  shapes forked at the OWNER (BarrierSubscribeReplyMessage on the scalable
    //  path, BarrierTriggerMessage on the legacy one), which is exactly why a
    //  remote node never has to know in advance whether this is a reduction
    //  barrier (tla/STATE_AND_LOCKING.md D2).
    struct BarrierSubscribeMessage {
      NodeID subscriber;
      ID::IDType barrier_id;
      // what the subscriber NEEDS - used only by the legacy path's one-shot
      //  'remote_subscribe_gens'
      EventImpl::gen_t subscribe_gen;
      // 'lk' - what the subscriber ALREADY KNOWS.  The reply's poison list is
      //  the delta above this, which is what a point-to-point answer can do and
      //  a multicast notification cannot, and it is what replaces the owner's
      //  per-node 'remote_trigger_gens' lookup on the pull path.
      EventImpl::gen_t last_known;

      static void handle_message(NodeID sender, const BarrierSubscribeMessage &args,
                                 const void *data, size_t datalen)
      {
        ID id(args.barrier_id);
        id.barrier_generation() = args.subscribe_gen;
        Barrier b = id.convert<Barrier>();
        BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

        impl->handle_remote_subscription(args.subscriber, args.subscribe_gen,
                                         args.last_known, data, datalen);
      }

      static void send_request(NodeID target, ID::IDType barrier_id,
                               EventImpl::gen_t subscribe_gen,
                               EventImpl::gen_t last_known, NodeID subscriber)
      {
        ActiveMessage<BarrierSubscribeMessage> amsg(target);
        amsg->subscriber = subscriber;
        amsg->barrier_id = barrier_id;
        amsg->subscribe_gen = subscribe_gen;
        amsg->last_known = last_known;
        amsg.commit();
      }
    };

    struct BarrierAdjustMessage {
      NodeID sender;
      int delta;
      Barrier barrier;
      Event wait_on;

      static void handle_message(NodeID sender, const BarrierAdjustMessage &args,
                                 const void *data, size_t datalen, TimeLimit work_until)
      {
        log_barrier.info() << "received barrier arrival: delta=" << args.delta
                           << " in=" << args.wait_on << " out=" << args.barrier << " ("
                           << args.barrier.timestamp << ")";
        BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
        EventImpl::gen_t gen = ID(args.barrier).barrier_generation();
        impl->adjust_arrival(gen, args.delta, args.barrier.timestamp, args.wait_on,
                             args.sender, datalen ? data : 0, datalen, work_until);
      }

      static void send_request(NodeID target, Barrier barrier, int delta, Event wait_on,
                               NodeID sender, const void *data, size_t datalen)
      {
        ActiveMessage<BarrierAdjustMessage> amsg(target, datalen);
        amsg->barrier = barrier;
        amsg->delta = delta;
        amsg->wait_on = wait_on;
        amsg->sender = sender;
        amsg.add_payload(data, datalen);
        amsg.commit();
      }
    };
  } // namespace

  // NOTE: no handle_inline - every barrier action takes 'mutex' and touches
  //  maps, and the inline contract forbids both (tla/STATE_AND_LOCKING.md D10).
  /*static*/ void BarrierReportMessage::handle_message(NodeID sender,
                                                       const BarrierReportMessage &args,
                                                       const void *data, size_t datalen,
                                                       TimeLimit work_until)
  {
    ID id(args.barrier_id);
    id.barrier_generation() = args.gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_report(args.from, args.gen, args.val, args.is_direct,
                               args.poisoned, data, datalen, work_until);
  }

  /*static*/ void BarrierReportMessage::send_request(NodeID target, ID::IDType barrier_id,
                                                     EventImpl::gen_t gen, int64_t val,
                                                     NodeID from, bool is_direct,
                                                     bool poisoned, const void *data,
                                                     size_t datalen)
  {
    // Q6: no hand-rolled chunking - an eager-flush map is O(subtree) and the
    //  active message layer fragments it for us.
    ActiveMessage<BarrierReportMessage> amsg(target, datalen);
    amsg->barrier_id = barrier_id;
    amsg->gen = gen;
    amsg->val = val;
    amsg->from = from;
    amsg->is_direct = is_direct;
    amsg->poisoned = poisoned;
    amsg.add_payload(data, datalen);
    amsg.commit();
  }

  // NOTE: no handle_inline, for the same reason as the report handler above -
  //  this one takes 'mutex' and may create a generation record (D10).
  /*static*/ void BarrierFlushMessage::handle_message(NodeID sender,
                                                      const BarrierFlushMessage &args,
                                                      const void *data, size_t datalen)
  {
    ID id(args.barrier_id);
    id.barrier_generation() = args.gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_flush(args.gen);
  }

  /*static*/ void BarrierFlushMessage::send_request(NodeID target, ID::IDType barrier_id,
                                                    EventImpl::gen_t gen)
  {
    ActiveMessage<BarrierFlushMessage> amsg(target);
    amsg->barrier_id = barrier_id;
    amsg->gen = gen;
    amsg.commit();
  }

  // NOTE: no handle_inline (D10) - this one takes 'mutex', walks every open
  //  generation and may install a parked plan.  It needs no TimeLimit: an
  //  invalidation moves no counts, so it can neither complete a generation nor
  //  wake a waiter.
  /*static*/ void
  BarrierInvalidateMessage::handle_message(NodeID sender,
                                           const BarrierInvalidateMessage &args,
                                           const void *data, size_t datalen)
  {
    ID id(args.barrier_id);
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_invalidate(args.epoch);
  }

  /*static*/ void BarrierInvalidateMessage::send_request(NodeID target,
                                                         ID::IDType barrier_id,
                                                         uint32_t epoch)
  {
    ActiveMessage<BarrierInvalidateMessage> amsg(target);
    amsg->barrier_id = barrier_id;
    amsg->epoch = epoch;
    amsg.commit();
  }

  // NOTE: no handle_inline (D10) - takes 'mutex' and copies the payload.  Like
  //  the invalidation it moves no counts, so it needs no TimeLimit.
  /*static*/ void BarrierNewPlanMessage::handle_message(NodeID sender,
                                                        const BarrierNewPlanMessage &args,
                                                        const void *data, size_t datalen)
  {
    ID id(args.barrier_id);
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    // the node that sent us our plan record IS our parent in the new tree
    //  (ARRIVAL_PROTOCOL section 8.1 - the parent is stored, not derived)
    impl->handle_remote_new_plan(sender, args.epoch, data, datalen);
  }

  /*static*/ void BarrierNewPlanMessage::send_request(NodeID target,
                                                      ID::IDType barrier_id,
                                                      uint32_t epoch, const void *data,
                                                      size_t datalen)
  {
    ActiveMessage<BarrierNewPlanMessage> amsg(target, datalen);
    amsg->barrier_id = barrier_id;
    amsg->epoch = epoch;
    amsg.add_payload(data, datalen);
    amsg.commit();
  }

  // NOTE: no handle_inline (D10) - this takes 'mutex', walks the open
  //  generations and may complete one, so it needs a TimeLimit.
  /*static*/ void BarrierAlterMessage::handle_message(NodeID sender,
                                                      const BarrierAlterMessage &args,
                                                      const void *data, size_t datalen,
                                                      TimeLimit work_until)
  {
    ID id(args.barrier_id);
    id.barrier_generation() = args.gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_alter(sender, args.gen, args.delta, args.ts, args.prev_ts,
                              work_until);
  }

  /*static*/ void BarrierAlterMessage::send_request(NodeID target, ID::IDType barrier_id,
                                                    EventImpl::gen_t gen, int delta,
                                                    Barrier::timestamp_t ts,
                                                    Barrier::timestamp_t prev_ts)
  {
    ActiveMessage<BarrierAlterMessage> amsg(target);
    amsg->barrier_id = barrier_id;
    amsg->gen = gen;
    amsg->delta = delta;
    amsg->ts = ts;
    amsg->prev_ts = prev_ts;
    amsg.commit();
  }

  // NOTE: no handle_inline (D10) - this takes 'mutex' and can complete a
  //  generation, so it needs a TimeLimit.
  /*static*/ void BarrierTsArrivalMessage::handle_message(
      NodeID sender, const BarrierTsArrivalMessage &args, const void *data,
      size_t datalen, TimeLimit work_until)
  {
    ID id(args.barrier_id);
    id.barrier_generation() = args.gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_ts_arrival(sender, args.gen, args.val, args.ts, args.local_ts,
                                   args.poisoned, work_until);
  }

  /*static*/ void BarrierTsArrivalMessage::send_request(
      NodeID target, ID::IDType barrier_id, EventImpl::gen_t gen, int64_t val,
      Barrier::timestamp_t ts, Barrier::timestamp_t local_ts, bool poisoned)
  {
    ActiveMessage<BarrierTsArrivalMessage> amsg(target);
    amsg->barrier_id = barrier_id;
    amsg->gen = gen;
    amsg->val = val;
    amsg->ts = ts;
    amsg->local_ts = local_ts;
    amsg->poisoned = poisoned;
    amsg.commit();
  }

  /*static*/ void BarrierTriggerMessage::send_request(
      NodeID target, ID::IDType barrier_id, EventImpl::gen_t trigger_gen,
      EventImpl::gen_t previous_gen, EventImpl::gen_t first_generation,
      ReductionOpID redop_id, const void *data, size_t datalen)
  {
    ActiveMessage<BarrierTriggerMessage> amsg(target, datalen);
    amsg->barrier_id = barrier_id;
    amsg->trigger_gen = trigger_gen;
    amsg->previous_gen = previous_gen;
    amsg->first_generation = first_generation;
    amsg->redop_id = redop_id;
    amsg.add_payload(data, datalen);
    amsg.commit();
  }

  /*static*/ void BarrierTriggerMessage::handle_message(NodeID sender,
                                                        const BarrierTriggerMessage &args,
                                                        const void *data, size_t datalen,
                                                        TimeLimit work_until)
  {
    log_barrier.info("received remote barrier trigger: " IDFMT "/%d -> %d",
                     args.barrier_id, args.previous_gen, args.trigger_gen);
    EventImpl::gen_t trigger_gen = args.trigger_gen;

    ID id(args.barrier_id);
    id.barrier_generation() = trigger_gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    // LEGACY / REDUCTION PATH ONLY.  The owner sends this message only when
    //  'redop_id != 0' - the scalable path's notification is
    //  BarrierNotifyMessage - so receiving one is itself the proof that this is
    //  a reduction barrier (D2).  The discriminator has to be the message's
    //  redop_id and not impl->redop_id, because this may be the very first
    //  thing this node ever hears about the barrier.
    const bool is_reduction = (args.redop_id != 0);

    // we'll probably end up with a list of local waiters to notify
    EventWaiter::EventWaiterList local_notifications;
    // this message doubles as the LEGACY path's subscribe reply, so it may have
    //  to re-subscribe (see below); that send is recorded here and emitted
    //  after the unlock (S2)
    BarrierImpl::PendingSends sends;
    {
      AutoLock<> a(impl->mutex);

      bool generation_updated = false;

      // it's theoretically possible for multiple trigger messages to arrive out
      //  of order, so check if this message triggers the oldest possible range.
      if(args.previous_gen <= impl->generation.load()) {
        if(is_reduction) {
          // LEGACY ONLY: see if we can pick up any of the held triggers too
          BarrierImpl::LegacyReductionState &ls = impl->legacy_state();
          while(!ls.held_triggers.empty()) {
            std::map<EventImpl::gen_t, EventImpl::gen_t>::iterator it =
                ls.held_triggers.begin();
            // if it's not contiguous, we're done
            if(it->first != trigger_gen)
              break;
            // it is contiguous, so absorb it into this message and remove the held
            // trigger
            log_barrier.info("collapsing future trigger: " IDFMT "/%d -> %d -> %d",
                             args.barrier_id, args.previous_gen, trigger_gen, it->second);
            trigger_gen = it->second;
            ls.held_triggers.erase(it);
          }
        }

        if(trigger_gen > impl->generation.load()) {
          impl->generation.store_release(trigger_gen);
          generation_updated = true;
          // ARRIVAL_PROTOCOL section 8.8 - a non-owner reclaims its own
          //  alteration state here, for the same reason the owner reclaims its
          //  in check_triggers_locked: an arrival can only ever name a
          //  breakpoint above the watermark
          impl->prune_alter_state_locked(trigger_gen);
        }

        // now iterate through any generations up to and including the latest triggered
        //  generation, and accumulate local waiters to notify
        while(!impl->generations.empty()) {
          std::map<EventImpl::gen_t, BarrierImpl::Generation *>::iterator it =
              impl->generations.begin();
          if(it->first > trigger_gen)
            break;

          local_notifications.absorb_append(it->second->local_waiters);
          delete it->second;
          impl->generations.erase(it);
        }
      } else if(is_reduction) {
        // LEGACY ONLY: hold this trigger until we get messages for the earlier
        //  generation(s)
        log_barrier.info("holding future trigger: " IDFMT "/%d (%d -> %d)",
                         args.barrier_id, impl->generation.load(), args.previous_gen,
                         trigger_gen);
        impl->legacy_state().held_triggers[args.previous_gen] = trigger_gen;
      } else {
        // A trigger message for a NON-reduction barrier should not exist: the
        //  scalable path notifies with BarrierNotifyMessage.  Discard rather
        //  than buffer, which is what NOTIFICATION_PROTOCOL rule 4 would do
        //  anyway.
        log_barrier.info("discarding out-of-order trigger: " IDFMT "/%d (%d -> %d)",
                         args.barrier_id, impl->generation.load(), args.previous_gen,
                         trigger_gen);
      }

      // is there any data we need to store?
      if(datalen) {
        assert(args.redop_id != 0);

        // TODO: deal with invalidation of previous instance of a barrier
        impl->redop_id = args.redop_id;
        impl->redop = get_runtime()->reduce_op_table.get(args.redop_id, 0);
        if(impl->redop == 0) {
          log_barrier.fatal() << "no reduction op registered for ID " << args.redop_id;
          abort();
        }
        impl->first_generation = args.first_generation;

        int rel_gen = trigger_gen - impl->first_generation;
        assert(rel_gen > 0);
        if(impl->value_capacity < (size_t)rel_gen) {
          size_t new_capacity = rel_gen;
          impl->final_values.resize(new_capacity * impl->redop->sizeof_lhs);
          // no need to initialize new entries - we'll overwrite them now or when data
          // does show up
          impl->value_capacity = new_capacity;
        }
        assert(args.trigger_gen <= trigger_gen);
        // trigger_gen might have changed so make sure you use args.trigger_gen here
        assert(datalen ==
               (impl->redop->sizeof_lhs * (args.trigger_gen - args.previous_gen)));
        assert(args.previous_gen >= impl->first_generation);
        memcpy(impl->final_values.data() + ((args.previous_gen - impl->first_generation) *
                                            impl->redop->sizeof_lhs),
               data, datalen);
      }

      // THE LEGACY PATH'S SUBSCRIBE REPLY.  This message is what the owner
      //  sends back to a subscribe on a reduction barrier, so it closes the
      //  rule-7 pull window.  A legacy subscription is ONE SHOT - the owner
      //  erases 'remote_subscribe_gens[node]' as soon as it is fulfilled - so
      //  membership goes back to NO, and a node that still holds a waiter has
      //  to ask again.  That is the same recovery NOTIFICATION_PROTOCOL rule 6
      //  makes on the scalable path, and it is needed here for the same reason:
      //  a consultation for a HIGHER generation, made while this pull was in
      //  flight, was suppressed and left no other trace.
      impl->pull_outstanding = false;
      impl->pull_deferred = false;
      impl->member = BarrierImpl::MEMBER_NO;
      if(impl->has_waiters_locked()) {
        impl->consult_subscribe_locked(sends);
      }

      // external waiters need to be signalled inside the lock
      if(generation_updated && impl->has_external_waiters) {
        impl->has_external_waiters = false;
        // also need external waiter mutex
        AutoLock<KernelMutex> al2(impl->external_waiter_mutex);
        impl->external_waiter_condvar.broadcast();
      }
    }

    impl->emit_pending_sends(sends);

    // with lock released, perform any local notifications.  The legacy path
    //  does not carry poison (Q4 is implemented on the scalable path only), so
    //  POISON_FIXME stays here.  The only other survivors are external_wait and
    //  external_timedwait, which decision Q9 puts out of scope entirely.
    if(!local_notifications.empty())
      get_runtime()->event_triggerer.trigger_event_waiters(local_notifications,
                                                           POISON_FIXME, work_until);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL rules 1, 2, 4 and 5 - the notify and reply messages.
  //
  // Both carry their poison list as a plain ascending array of generations in
  //  the payload.  Neither list can grow without bound: a notification's is
  //  confined to the range it announces, and a reply's to what the subscriber
  //  says it is missing, both capped by POISONED_GENERATION_LIMIT.
  //

  namespace {
    // Shared payload validation for both messages: 'num_poisoned' generations
    //  followed by 'trailing' bytes of something else (the departing set, or
    //  nothing).  The length is checked here rather than trusted, so a
    //  corrupt header can never make the reader walk off the buffer.
    //
    // NOTE the empty case is a NULL payload pointer, not an error: the common
    //  notification carries no poison and no membership at all.  That is why
    //  this returns a status and the pointer separately.
    bool poison_payload_ok(size_t datalen, uint32_t num_poisoned, size_t trailing)
    {
      return datalen == ((num_poisoned * sizeof(EventImpl::gen_t)) + trailing);
    }
  } // namespace

  // NOTE: no handle_inline (D10) - this takes 'mutex', frees generation records
  //  and wakes waiters, so it needs a TimeLimit.
  /*static*/ void BarrierNotifyMessage::handle_message(NodeID sender,
                                                       const BarrierNotifyMessage &args,
                                                       const void *data, size_t datalen,
                                                       TimeLimit work_until)
  {
    log_barrier.info("received barrier notify: " IDFMT "/%d -> %d sv=%llu",
                     args.barrier_id, args.prev, args.wm,
                     (unsigned long long)args.set_ver);

    ID id(args.barrier_id);
    id.barrier_generation() = args.wm;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    if(!poison_payload_ok(datalen, args.num_poisoned, args.departing_bytes)) {
      log_barrier.fatal() << "malformed barrier notification payload: " << b
                          << " bytes=" << datalen << " npois=" << args.num_poisoned
                          << " dep=" << args.departing_bytes;
      std::abort();
    }
    const EventImpl::gen_t *poison =
        (args.num_poisoned > 0) ? static_cast<const EventImpl::gen_t *>(data) : nullptr;

    // RULE 1, the wire detail.  A per-recipient membership flag cannot survive
    //  a multicast, so the message names the DEPARTING SET and each recipient
    //  computes its own membership.  This is EXACT because the notification
    //  goes only to the PRE-SHRINK set: every recipient is in it, so "still a
    //  member" is precisely "not in R".
    //
    // Decoded OUTSIDE the critical section - it is a pure byte codec, and a
    //  payload has no business lengthening an action (ARRIVAL_PROTOCOL 12).
    bool inset = true;
    if(args.departing_bytes > 0) {
      MulticastTargetSet departing;
      const unsigned char *dep = static_cast<const unsigned char *>(data) +
                                 (args.num_poisoned * sizeof(EventImpl::gen_t));
      if(EncodedMulticastTargets::decode(dep, args.departing_bytes,
                                         Network::max_node_id + 1,
                                         departing) != MulticastDecodeStatus::OK) {
        log_barrier.fatal() << "malformed barrier notification departing set: " << b
                            << " bytes=" << args.departing_bytes;
        std::abort();
      }
      inset = !departing.contains(Network::my_node_id);
    }

    impl->handle_remote_notify(args.wm, args.prev, args.gather_gen, args.set_ver, poison,
                               args.num_poisoned, inset, (args.shrink_hint != 0),
                               work_until);
  }

  /*static*/ void BarrierNotifyMessage::send_request(
      const MulticastTargetSet &targets, ID::IDType barrier_id, EventImpl::gen_t wm,
      EventImpl::gen_t prev, EventImpl::gen_t gather_gen, uint64_t set_ver,
      bool shrink_hint, const std::vector<EventImpl::gen_t> &poison,
      const std::vector<unsigned char> &departing)
  {
    if(targets.empty()) {
      return;
    }

    std::vector<unsigned char> payload;
    payload.resize((poison.size() * sizeof(EventImpl::gen_t)) + departing.size());
    if(!poison.empty()) {
      memcpy(payload.data(), poison.data(), poison.size() * sizeof(EventImpl::gen_t));
    }
    if(!departing.empty()) {
      memcpy(payload.data() + (poison.size() * sizeof(EventImpl::gen_t)),
             departing.data(), departing.size());
    }

    BarrierNotifyMessage hdr;
    hdr.barrier_id = barrier_id;
    hdr.wm = wm;
    hdr.prev = prev;
    hdr.gather_gen = gather_gen;
    hdr.set_ver = set_ver;
    hdr.num_poisoned = static_cast<uint32_t>(poison.size());
    hdr.departing_bytes = static_cast<uint32_t>(departing.size());
    // rule 8's advisory byte.  It rides the notification that was going out
    //  anyway, so publishing it costs nothing on the wire.
    hdr.shrink_hint = (shrink_hint ? 1 : 0);

    // The forwarding tree is planned FRESH from the encoded set on every send
    //  (NOTIFICATION_PROTOCOL section 1), which is why this protocol needs none
    //  of the tree-invalidation machinery the arrival one does.
    multicast_message<BarrierNotifyMessage>(
        targets, hdr, payload.empty() ? nullptr : payload.data(), payload.size());
  }

  // NOTE: no handle_inline (D10) - takes 'mutex' and wakes waiters.
  /*static*/ void BarrierSubscribeReplyMessage::handle_message(
      NodeID sender, const BarrierSubscribeReplyMessage &args, const void *data,
      size_t datalen, TimeLimit work_until)
  {
    log_barrier.info("received barrier subscribe reply: " IDFMT " wm=%d sv=%llu",
                     args.barrier_id, args.wm, (unsigned long long)args.set_ver);

    ID id(args.barrier_id);
    id.barrier_generation() = args.wm;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    if(!poison_payload_ok(datalen, args.num_poisoned, /*trailing=*/0)) {
      log_barrier.fatal() << "malformed barrier subscribe reply payload: " << b
                          << " bytes=" << datalen << " npois=" << args.num_poisoned;
      std::abort();
    }
    const EventImpl::gen_t *poison =
        (args.num_poisoned > 0) ? static_cast<const EventImpl::gen_t *>(data) : nullptr;

    impl->handle_remote_subscribe_reply(args.wm, args.set_ver, poison, args.num_poisoned,
                                        work_until);
  }

  /*static*/ void
  BarrierSubscribeReplyMessage::send_request(NodeID target, ID::IDType barrier_id,
                                             EventImpl::gen_t wm, uint64_t set_ver,
                                             const std::vector<EventImpl::gen_t> &poison)
  {
    const size_t bytes = poison.size() * sizeof(EventImpl::gen_t);
    ActiveMessage<BarrierSubscribeReplyMessage> amsg(target, bytes);
    amsg->barrier_id = barrier_id;
    amsg->wm = wm;
    amsg->set_ver = set_ver;
    amsg->num_poisoned = static_cast<uint32_t>(poison.size());
    if(bytes > 0) {
      amsg.add_payload(poison.data(), bytes);
    }
    amsg.commit();
  }

  // NOTE: no handle_inline (D10) - this takes 'mutex'.  It needs no TimeLimit:
  //  a departure intent moves no counts and no watermark, so it can neither
  //  complete a generation nor wake a waiter.  It cannot even change what the
  //  owner will publish next, because the shrink is not decided here - it is
  //  decided by the cost test at the next trigger.
  /*static*/ void BarrierDepartMessage::handle_message(NodeID sender,
                                                       const BarrierDepartMessage &args,
                                                       const void *data, size_t datalen)
  {
    ID id(args.barrier_id);
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    impl->handle_remote_depart(args.departing);
  }

  /*static*/ void BarrierDepartMessage::send_request(NodeID target, ID::IDType barrier_id,
                                                     NodeID departing)
  {
    ActiveMessage<BarrierDepartMessage> amsg(target);
    amsg->barrier_id = barrier_id;
    amsg->departing = departing;
    amsg.commit();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class BarrierImpl
  //

  BarrierImpl::BarrierImpl(void)
    : external_waiter_condvar(external_waiter_mutex)
  {
    remote_subscribe_gens.clear();
  }

  BarrierImpl::~BarrierImpl(void)
  {
    dump_counters("destroyed");
    delete[] poisoned_generations;
  }

  bool BarrierImpl::BarrierCounters::any(void) const
  {
    return (departs_sent | departs_suppressed | leave_rejoin_cycles | churn_backoffs |
            subscribe_fan_in | departs_received | shrinks_applied | shrinks_declined |
            nodes_removed | flush_episodes | flush_report_bytes | plan_rebuilds |
            plan_rebuilds_declined | gathering_declared | identical_plans_skipped |
            plans_parked | parked_plans_applied | dead_plans_discarded |
            retro_flushes_sent | stale_edge_forwards | report_edges_pinned |
            pin_conflicts_avoided | gap_pulls) != 0;
  }

  // IMPLEMENTATION_PLAN section 5 - the counters' way out.  Everything in
  //  BarrierCounters is written under 'mutex' beside the state it describes and
  //  nothing branches on any of it, so without this it would be write-only.
  //  One line, at 'info' level, only for a barrier that actually exercised the
  //  deviation or hysteresis paths - a barrier that behaved is silent.
  void BarrierImpl::dump_counters(const char *why) const
  {
    if(!counters.any()) {
      return;
    }
    log_barrier.info() << "barrier counters (" << why << "): " << me
                       << " departs_sent=" << counters.departs_sent
                       << " departs_suppressed=" << counters.departs_suppressed
                       << " leave_rejoin=" << counters.leave_rejoin_cycles
                       << " churn_backoffs=" << counters.churn_backoffs
                       << " depart_K=" << depart_K
                       << " subscribe_fan_in=" << counters.subscribe_fan_in
                       << " departs_received=" << counters.departs_received
                       << " shrinks_applied=" << counters.shrinks_applied
                       << " shrinks_declined=" << counters.shrinks_declined
                       << " nodes_removed=" << counters.nodes_removed
                       << " flush_episodes=" << counters.flush_episodes
                       << " flush_report_bytes=" << counters.flush_report_bytes
                       << " flush_report_bytes_max=" << counters.flush_report_bytes_max
                       << " plan_rebuilds=" << counters.plan_rebuilds
                       << " plan_rebuilds_declined=" << counters.plan_rebuilds_declined
                       << " gathering_declared=" << counters.gathering_declared
                       << " identical_plans_skipped=" << counters.identical_plans_skipped
                       << " agg_peak_entries=" << counters.agg_peak_entries
                       << " plans_parked=" << counters.plans_parked
                       << " parked_plans_applied=" << counters.parked_plans_applied
                       << " dead_plans_discarded=" << counters.dead_plans_discarded
                       << " retro_flushes_sent=" << counters.retro_flushes_sent
                       << " stale_edge_forwards=" << counters.stale_edge_forwards
                       << " report_edges_pinned=" << counters.report_edges_pinned
                       << " pin_conflicts_avoided=" << counters.pin_conflicts_avoided
                       << " gap_pulls=" << counters.gap_pulls;
  }

#ifdef DEBUG_REALM
  // DEBUG ONLY.  See the declaration: taking 'mutex' here proves NOBODY held
  //  it, and therefore that this caller did not.  Failing to take it proves
  //  nothing (another thread may hold it), so this never fires falsely.
  void BarrierImpl::assert_locked(const char *who) const
  {
    Mutex &m = const_cast<Mutex &>(mutex);
    if(m.trylock()) {
      m.unlock();
      log_barrier.fatal() << "barrier " << me << ": " << who
                          << "() called without holding 'mutex'";
      abort();
    }
  }
#endif

  void BarrierImpl::init(ID _me, unsigned _init_owner)
  {
    // the previous life of this slot is over - say what it did before the
    //  counters are cleared below
    dump_counters("recycled");
    me = _me;
    owner = _init_owner;
    gen_subscribed.store(0);
    first_generation = /*free_generation =*/0;
    // decision Q3 - the poison array is per-lifetime.  A stale entry would make
    //  has_triggered() report the NEW barrier's generation poisoned.
    delete[] poisoned_generations;
    poisoned_generations = nullptr;
    num_poisoned_generations.store(0);
    // NOTIFICATION_PROTOCOL: the node's belief about its own membership, and
    //  (on the owner) the published set itself.  Both are per-lifetime: a
    //  surviving 'member == MEMBER_YES' would stop the new barrier ever
    //  subscribing, and a surviving 'sub_set' would multicast the new
    //  barrier's notifications to nodes that never asked.
    member = MEMBER_NO;
    my_set_ver = 0;
    pull_outstanding = false;
    pull_deferred = false;
    sub_set.clear();
    set_ver = 0;
    want_out.clear();
    // rule 8 - the hysteresis state is per-lifetime too.  A surviving idle
    //  counter would make the new barrier's first notification look like K
    //  generations of silence and retire a node that has never been asked for
    //  anything, and a surviving 'depart_outstanding' would suppress the one
    //  request the new barrier is entitled to.  K itself restarts at its
    //  initial value: the churn the previous life observed says nothing about
    //  this one.
    last_consult_wm = 0;
    last_depart_wm = 0;
    depart_K = DEPART_K_INITIAL;
    depart_outstanding = false;
    shrink_hint = true;
    shrink_pays = true;
    counters = BarrierCounters();
    next_free = 0;
    remote_subscribe_gens.clear();
    base_arrival_count = 0;
    redop = 0;
    initial_value = 0;
    value_capacity = 0;
    final_values.clear();
    // drops the legacy per-node trigger map and any held triggers left over
    //  from a previous life of this slot
    legacy.reset();
    // the plan record and the epochs are per-lifetime state: a stale parked
    //  plan surviving into the next life of this slot would be installed by the
    //  first invalidation the new barrier ever saw
    cur_plan = ArrivalPlan();
    my_epoch = 0;
    inval_epoch = 0;
    defer_epoch = 0;
    defer_parent = -1;
    std::vector<unsigned char>().swap(deferred_plan_payload);
    // epochs must restart at 1 for the new life of this slot: a node still
    //  holding a plan record from the previous one would otherwise compare the
    //  new epochs against a stale, higher 'my_epoch' and drop them all
    next_epoch = 1;
    plan_rebuild_pending = false;
    // the alteration state is per-lifetime too: a stale arrival floor would
    //  make the new barrier's arrivals bypass the tree waiting on an
    //  alteration that belonged to the previous one, and a stale applied-set
    //  would open a gate that has not actually been opened
    ts_floor.clear();
    last_alter_ts = 0;
    alter_floor = 0;
    alter_steps.clear();
    applied_ts.clear();
    held_alters.clear();
    // a node that altered the PREVIOUS life of this slot has no arrival floor
    //  for this one, so it is a legitimate plan member again
    ts_bypass_nodes.clear();
    generation.store_release(0);
  }

  // like strdup, but works on arbitrary byte arrays
  static void *bytedup(const void *data, size_t datalen)
  {
    if(datalen == 0) {
      return 0;
    }
    void *dst = malloc(datalen);
    assert(dst != 0);
    memcpy(dst, data, datalen);
    return dst;
  }

  class DeferredBarrierArrival : public EventWaiter {
  public:
    DeferredBarrierArrival(Barrier _barrier, int _delta, NodeID _sender,
                           const void *_data, size_t _datalen)
      : barrier(_barrier)
      , delta(_delta)
      , sender(_sender)
      , data(bytedup(_data, _datalen))
      , datalen(_datalen)
    {}

    virtual ~DeferredBarrierArrival(void)
    {
      if(data) {
        free(data);
      }
    }

    virtual void event_triggered(bool poisoned, TimeLimit work_until)
    {
      // DECISION Q4 - A POISONED PRECONDITION POISONS THE GENERATION.  The
      //  arrival still happens: withholding it would hang the barrier instead
      //  of poisoning it, and every waiter would then be told nothing at all
      //  rather than told the truth.
      log_barrier.info() << "deferred barrier arrival: " << barrier << " ("
                         << barrier.timestamp << "), delta=" << delta
                         << ", poisoned=" << poisoned;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
      impl->adjust_arrival(ID(barrier).barrier_generation(), delta, barrier.timestamp,
                           Event::NO_EVENT, sender, data, datalen, work_until, poisoned);
      // not attached to anything, so delete ourselves when we're done
      delete this;
    }

    virtual void print(std::ostream &os) const
    {
      os << "deferred arrival: barrier=" << barrier << " (" << barrier.timestamp << ")"
         << ", delta=" << delta << " datalen=" << datalen;
    }

    virtual Event get_finish_event(void) const { return barrier; }

  protected:
    Barrier barrier;
    int delta;
    NodeID sender;
    void *data;
    size_t datalen;
  };

  // NOTE: the per-node timestamp-ordering machinery that used to live here
  //  (PerNodeUpdates, with its 'ts <= pn->last_ts' test and its own
  //  "TODO: really need two timestamps to properly order increments") is gone.
  //  It is replaced by the exact-set gate of ARRIVAL_PROTOCOL rule 8: an
  //  arrival names the alteration(s) it depends on, and the owner counts it
  //  only when every one of them IS IN 'applied_ts'.  A "<= the highest
  //  timestamp seen from that node" comparison admits an arrival whose own
  //  alteration has not been applied yet, which is a barrier that triggers too
  //  early - the one failure mode this rule exists to prevent.

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL section 11.1 - the eager-flush report's (node,count) map
  //
  // These are pure byte-level codecs: they touch no barrier state, take no
  //  lock, and are deliberately callable OUTSIDE the critical section so that
  //  decoding an inbound payload does not lengthen an action (section 12).
  //

  namespace {
    // The number of nodes both sides of the multicast codec must agree on.
    NodeID plan_node_count(void) { return Network::max_node_id + 1; }

    // See BarrierReportMessage for the format.  Produces nothing at all for an
    //  empty map: datalen == 0 is what tells the receiver "this report carries
    //  no map", which is NOT the same as "this report says the map is empty".
    //  The distinction matters because maps merge cumulatively and must never
    //  be seen to shrink.
    void encode_flush_map(const std::map<NodeID, int64_t> &counts,
                          std::vector<unsigned char> &out)
    {
      out.clear();

      // D8 / plan section 7.2: the node SET goes through the multicast codec,
      //  which picks the smallest of its eight representations - a contiguous
      //  block of nodes costs one run, not one entry per node.
      MulticastTargetSet nodes;
      for(std::map<NodeID, int64_t>::const_iterator it = counts.begin();
          it != counts.end(); ++it) {
        if(it->second <= 0) {
          continue;
        }
        // std::map iterates in ascending key order, which is exactly the
        //  strictly-increasing order this wants
        if(!nodes.append_increasing_node(it->first)) {
          return;
        }
      }
      if(nodes.empty()) {
        return;
      }

      EncodedMulticastTargets enc =
          EncodedMulticastTargets::encode(nodes, plan_node_count());

      MulticastWire::append_varint(out, enc.bytes());
      const unsigned char *encp = static_cast<const unsigned char *>(enc.data());
      out.insert(out.end(), encp, encp + enc.bytes());

      // the counts, in the set's own (sorted) iteration order - no node id is
      //  repeated on the wire
      for(std::map<NodeID, int64_t>::const_iterator it = counts.begin();
          it != counts.end(); ++it) {
        if(it->second <= 0) {
          continue;
        }
        MulticastWire::append_varint(out, static_cast<uint64_t>(it->second));
      }
    }

    // Rejects anything malformed rather than aborting, and never sizes an
    //  allocation from an unvalidated length: the cardinality comes from the
    //  DECODED set, not from a transmitted count.
    bool decode_flush_map(const void *data, size_t datalen,
                          std::map<NodeID, int64_t> &out)
    {
      out.clear();
      const unsigned char *p = static_cast<const unsigned char *>(data);
      if((p == 0) || (datalen == 0)) {
        return false;
      }

      size_t pos = 0;
      uint64_t set_bytes = 0;
      if(MulticastWire::read_varint(p, datalen, pos, set_bytes) !=
         MulticastDecodeStatus::OK) {
        return false;
      }
      if(set_bytes > (datalen - pos)) {
        return false;
      }

      MulticastTargetSet nodes;
      if(EncodedMulticastTargets::decode(p + pos, static_cast<size_t>(set_bytes),
                                         plan_node_count(),
                                         nodes) != MulticastDecodeStatus::OK) {
        return false;
      }
      pos += static_cast<size_t>(set_bytes);

      for(MulticastTargetSet::const_iterator it = nodes.begin(); it != nodes.end();
          ++it) {
        uint64_t count = 0;
        if(MulticastWire::read_varint(p, datalen, pos, count) !=
           MulticastDecodeStatus::OK) {
          out.clear();
          return false;
        }
        // a count that cannot be a plan quota is a corrupt payload, not a
        //  merely large one
        if(count > static_cast<uint64_t>(UINT32_MAX)) {
          out.clear();
          return false;
        }
        out.insert(out.end(), std::make_pair(*it, static_cast<int64_t>(count)));
      }

      // trailing bytes are malformed, not merely generous
      if(pos != datalen) {
        out.clear();
        return false;
      }
      return true;
    }
  } // namespace

  ////////////////////////////////////////////////////////////////////////
  //
  // scalable arrival helpers - ALL of these require 'mutex'
  //

  BarrierImpl::Generation *BarrierImpl::get_generation_locked(gen_t gen)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    std::map<gen_t, Generation *>::iterator it = generations.find(gen);
    if(it != generations.end()) {
      return it->second;
    }
    Generation *g = new Generation;
    generations[gen] = g;
    log_barrier.info() << "added tracker for barrier " << me << ", generation " << gen;
    return g;
  }

  // ARRIVAL_PROTOCOL rule 1 - the steady state, and BOTH of its halves are
  //  required.  A node forwards its cumulative subtree total only when its
  //  local arrival count equals the quota its plan predicts AND every child the
  //  plan predicts has reported at least once.
  //
  // The child-wait is a CORRECTNESS requirement, not an optimisation.  Early
  //  forwarding does NOT self-correct just because reports are cumulative: at
  //  depth >= 3, where a relay sits below another relay, dropping the child-wait
  //  is a mutation TLC catches.  Do not simplify it away.
  bool BarrierImpl::plan_satisfied_locked(const Generation &g) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // MEMBERSHIP: a node with no plan has no completion condition at all
    if(!cur_plan.inplan) {
      return false;
    }
    if(g.local_total != static_cast<int64_t>(cur_plan.quota)) {
      return false;
    }
    // NO CHILD-WAIT (ARRIVAL_PROTOCOL rule 1).  A relay does NOT additionally
    //  wait for every predicted child.  Reports are cumulative and REPLACE, so
    //  a forward that omits an absent child is superseded when that child
    //  reports, and a generation only triggers on exact equality with the
    //  expected count, so a low total cannot fire early.
    //
    //  This is load-bearing for rule 5: the child-wait is the only reason a
    //  relay cares WHICH child reports to it, and therefore the only reason a
    //  re-parented generation can strand on a child that already reported by
    //  the retiring tree.  Removing it is what lets flush stop being sticky.
    //  Do not reintroduce it.
    return true;
  }

  // ARRIVAL_PROTOCOL section 8.1: the parent is STORED in the plan record, not
  //  derived by searching for whoever lists us as a child.  A node outside the
  //  plan has no parent, so its reports go straight to the owner (rule 3).
  NodeID BarrierImpl::report_target_locked(void) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(cur_plan.inplan && (cur_plan.parent != static_cast<NodeID>(-1))) {
      return cur_plan.parent;
    }
    return owner;
  }

  // Has the plan this node holds been RETIRED, with no replacement having
  //  arrived yet?  my_epoch == 0 means "no plan has ever been installed here",
  //  which cur_plan.inplan already reports, so it is excluded.
  //
  // This is the same self-clearing shape as rule 5's deferral guard: applying a
  //  new plan raises my_epoch above inval_epoch, and nothing else has to
  //  remember that a switch happened.
  bool BarrierImpl::plan_retired_locked(void) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    return (my_epoch != 0) && (inval_epoch >= my_epoch);
  }

  // Is this node reporting EAGERLY for 'g' rather than aggregating?  The three
  //  cases below are the same three that make should_report_locked() true
  //  without consulting the plan, and they are also exactly the cases in which
  //  a report carries the (node,count) map of section 11.1: a node that is not
  //  aggregating is a node whose plan does not describe reality, and the map is
  //  how the owner learns what reality is.
  bool BarrierImpl::flush_mode_locked(gen_t gen, const Generation &g) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // eager-flush mode: every arrival reports immediately (rules 2, 3 and 4).
    //  This is also what carries rule 2 past the moment of over-arrival - once
    //  the flag is on, this node never goes back to aggregating for this
    //  generation, which is why there is no 'local_total > quota' test here.
    //  BarrierArrive's RecvReport guard is exactly 'flushing \/ PlanSatisfied'.
    if(g.flushing) {
      return true;
    }
    // rule 3 / MEMBERSHIP (section 3): a node outside the plan has no
    //  completion condition it could ever satisfy, so it is permanently in
    //  eager-flush mode and speaks at once.  BarrierArrive's RecvReport writes this
    //  as 'flushing \/ PlanSatisfied' rather than naming it, because in the
    //  model every un-planned node has already been invalidated and so has
    //  'flushing' set; here the state is reachable before any plan exists at
    //  all, and holding would strand the arrival.
    if(!cur_plan.inplan) {
      return true;
    }
    // A node whose plan has been RETIRED but not yet replaced is in exactly the
    //  same position as rule 3's outsider: the completion condition its record
    //  describes belongs to a tree that no longer exists, so waiting on it is
    //  the silence this protocol exists to prevent.
    //
    // BarrierArrive's RecvInvalidate covers this by setting flushing[n][g] for
    //  every untriggered g the node HAS STATE ON (SubtreeKnown > 0, :452-456)
    //  and making the node an OUTSIDER (inplan = FALSE, :464-467), so its
    //  later arrivals fire case 3 natively; nothing clears the flag again
    //  before the trigger (RecvNewPlan leaves 'flushing' UNCHANGED at
    //  :510-514).
    //
    // Here the outsider transition is DERIVED rather than stored - retiring
    //  the record would lose the old child list, which step 2 of the
    //  invalidation still needs - and the deviation signal is kept alive by:
    //
    //   * this test, which covers the window between the invalidation and the
    //     replacement (and IS self-clearing - installing a plan raises
    //     'my_epoch' above 'inval_epoch');
    //   * 'saw_direct' below, which makes the deviation signal travel THROUGH a
    //     below-quota relay rather than being swallowed by it;
    //   * the retroactive case-3 flush (handle_remote_invalidate step 5),
    //     which signals the owner for arrivals made BEFORE the invalidation
    //     that this test can no longer speak for; and
    //   * the owner's flush of every straddling generation when it installs a
    //     new plan (build_new_plan_locked step 5), which reaches them across
    //     the NEW tree.
    if(plan_retired_locked()) {
      return true;
    }
    // RULE 3, PROPAGATED.  A 'direct' passing through says "the plan does not
    //  describe reality for this generation", and it says it about the path as
    //  much as about the node that raised it.  Without this the flag rides on a
    //  report that a below-quota relay never sends: the signal dies one hop
    //  above the deviation and the owner never learns it has to flush.  That is
    //  the silence BarrierArrive avoids by keeping 'flushing' set forever after an
    //  invalidation, and it is the reason this is a mode and not just a flag to
    //  be copied onto the next outgoing report.
    if(g.saw_direct) {
      return true;
    }
    // RULE 8.3 - this node has issued an alteration covering 'gen', so its own
    //  arrivals BYPASS the tree (rule 8.1) and never reach 'local_total'.  Its
    //  quota is therefore unreachable, and a relay waiting on an unreachable
    //  quota goes silent and strands its children - a caught mutation
    //  (deadlock).  Like plan_retired_locked() above, this is derived rather
    //  than flagged, because the flag would have to be set on generations this
    //  node has no record for yet.
    if(ts_floor_locked(gen) != 0) {
      return true;
    }
    return false;
  }

  // The forwarding decision of action A step 5 and action R step 6.
  bool BarrierImpl::should_report_locked(gen_t gen, const Generation &g) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // the owner has no parent - its subtree total is the answer, not a report
    if(owner == Network::my_node_id) {
      return false;
    }
    // nothing we know that we have not already passed on
    if(!g.holding()) {
      return false;
    }
    // rules 2, 3, 4, 6 and 8 - not aggregating, so speak at once
    if(flush_mode_locked(gen, g)) {
      return true;
    }
    // rule 1 - both halves
    return plan_satisfied_locked(g);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL rules 8 and 9 - alter_arrival_count.  ALL of these
  //  require 'mutex'.
  //

  // The model's myTs[n][g], stored as a step function because alterations are
  //  PERSISTENT: the entry installed at generation k covers k and everything
  //  after it, until a later alteration installs a new one.
  Barrier::timestamp_t BarrierImpl::ts_floor_locked(gen_t gen) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(ts_floor.empty()) {
      return 0;
    }
    // the greatest entry with key <= gen
    std::map<gen_t, Barrier::timestamp_t>::const_iterator it = ts_floor.upper_bound(gen);
    if(it == ts_floor.begin()) {
      return 0;
    }
    --it;
    return it->second;
  }

  // OWNER ONLY.  expected(gen) - the arrival count this generation is waiting
  //  for, with every alteration that applies to it folded in.  PERSISTENCE
  //  (D5) is exactly the "key <= gen" here: an alteration issued for an earlier
  //  generation still counts for this one.
  int64_t BarrierImpl::expected_locked(gen_t gen) const
  {
    int64_t total = static_cast<int64_t>(base_arrival_count) + alter_floor;
    for(std::map<gen_t, int64_t>::const_iterator it = alter_steps.begin();
        (it != alter_steps.end()) && (it->first <= gen); ++it) {
      total += it->second;
    }
    return total;
  }

  // OWNER ONLY.  EXACT SET MEMBERSHIP, and deliberately not a comparison
  //  against the highest timestamp applied for the sending node: alterations
  //  can be applied out of order, and a "<=" gate would count an arrival whose
  //  own alteration is still in flight.
  bool BarrierImpl::ts_gate_open_locked(const Generation::TsKey &gate) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if((gate.ts != 0) && (applied_ts.count(gate.ts) == 0)) {
      return false;
    }
    if((gate.local_ts != 0) && (applied_ts.count(gate.local_ts) == 0)) {
      return false;
    }
    return true;
  }

  // OWNER ONLY.  'count' arrivals have been accounted for.  On the scalable
  //  path they count UP into the generation's bypassed-arrival total (D4); on
  //  the legacy path they count DOWN into 'unguarded_delta', which is that
  //  path's convention and stays there.
  void BarrierImpl::credit_arrivals_locked(Generation &g, int64_t count) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(redop_id != 0) {
      g.unguarded_delta -= static_cast<int>(count);
    } else {
      g.ts_acc += count;
    }
  }

  // OWNER ONLY - action TS.  One bypass stream per (sender, gate), cumulative
  //  and replace-if-higher exactly like child_acc (rule 7): a value that does
  //  not strictly exceed what we hold is stale and is dropped.
  void BarrierImpl::apply_ts_arrival_locked(NodeID from, Generation &g,
                                            const Generation::TsKey &gate, int64_t val)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    Generation::TsStreamKey key;
    key.from = from;
    key.gate = gate;

    Generation::TsStream &st = g.ts_streams[key];
    if(val <= st.seen) {
      log_barrier.info() << "dropping stale barrier ts arrival: " << me
                         << " from=" << from << " val=" << val << " have=" << st.seen;
      return;
    }
    const int64_t added = val - st.seen;
    st.seen = val;

    if(st.counted) {
      credit_arrivals_locked(g, added);
      return;
    }
    // THE GATE (rule 8.2): the owner may not count a timestamped arrival until
    //  every alteration it witnessed has been applied.  Until then the stream
    //  simply sits here - the model expresses this as RecvTsDirect being
    //  DISABLED, which is why it needs no message ordering at all.
    if(!ts_gate_open_locked(gate)) {
      log_barrier.info() << "parking barrier ts arrival: " << me << " from=" << from
                         << " val=" << val << " ts=" << gate.ts
                         << " local_ts=" << gate.local_ts;
      return;
    }
    st.counted = true;
    credit_arrivals_locked(g, st.seen);
  }

  // OWNER ONLY.  An alteration has just been applied, so every stream that was
  //  parked behind it may now be counted.  Fusing this with the application of
  //  the alteration is a legal refinement: the model expresses it as
  //  RecvTsDirect becoming ENABLED, and merging an enabling action with the
  //  action it enables never exposes a state the model does not have.
  void BarrierImpl::open_ts_gates_locked(Barrier::timestamp_t ts)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    const gen_t watermark = generation.load();
    for(std::map<gen_t, Generation *>::iterator it = generations.begin();
        it != generations.end(); ++it) {
      if(it->first <= watermark) {
        continue;
      }
      Generation &g = *(it->second);
      for(std::map<Generation::TsStreamKey, Generation::TsStream>::iterator sit =
              g.ts_streams.begin();
          sit != g.ts_streams.end(); ++sit) {
        if(sit->second.counted) {
          continue;
        }
        // only a stream this timestamp is part of can have been unblocked
        if((sit->first.gate.ts != ts) && (sit->first.gate.local_ts != ts)) {
          continue;
        }
        if(!ts_gate_open_locked(sit->first.gate)) {
          continue;
        }
        sit->second.counted = true;
        credit_arrivals_locked(g, sit->second.seen);
      }
    }
  }

  // OWNER ONLY - ARRIVAL_PROTOCOL rule 9, "a negative alteration ... invalidates
  //  the current arrival plan", applied here to alterations OF EITHER SIGN.
  //
  // Rule 9 states the negative case, whose plan defect is an unreachable quota.
  //  A POSITIVE alteration produces the same defect by a different route, and
  //  it is the one rule 8.3 is about: the altering node's arrivals now BYPASS
  //  the tree, so its 'local_total' stops growing and the quota the plan gave
  //  it can never be met.  Rule 8.3 keeps that node reporting for its own
  //  children, but nothing in it helps the node's PARENT, which is holding its
  //  whole subtree back on rule 1's child-wait ("every predicted child has
  //  reported at least once").  A leaf that alters strands its parent exactly
  //  as an unreachable quota strands a relay's children.  BarrierArrive does not
  //  see this because MCAlter's altering node reports to the owner, which never
  //  waits on a child - so this is a case the model does not cover rather than
  //  one it rules out.
  //
  // Retiring the plan is the protocol's own answer to "the plan is wrong", it
  //  is rule 6's verified machinery, and it covers the generations this node
  //  has no record for yet - which a per-generation flush cannot.  The cost is
  //  that a barrier altered every generation never settles into a plan and
  //  stays on the eager path; that is today's behaviour for every barrier, and
  //  it is the safe direction.
  void BarrierImpl::retire_plan_locked(PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // nothing has ever been distributed, so there is nothing to retire
    if(my_epoch == 0) {
      return;
    }
    // already retired and waiting for a replacement
    if(inval_epoch >= my_epoch) {
      return;
    }

    // RULE 6, FORWARD BEFORE FORGETTING: the targets come from the CURRENT
    //  child list and are resolved into 'sends' inside this section (S3).
    sends.fwd_inval_epoch = my_epoch;
    sends.inval_to.insert(sends.inval_to.end(), cur_plan.kids.begin(),
                          cur_plan.kids.end());
    inval_epoch = my_epoch;
    plan_rebuild_pending = true;

    log_barrier.info() << "retiring barrier plan: " << me << " epoch=" << my_epoch;
  }

  // OWNER ONLY - action RA, ARRIVAL_PROTOCOL rules 8 and 9.
  //
  // The alteration is PERSISTENT: it lands as a STEP in 'alter_steps', so it
  //  raises (or lowers) expected(g) for the generation it names and for every
  //  generation after it.  Nothing here is per-generation, which is precisely
  //  the bug this replaces.
  void BarrierImpl::apply_alter_locked(gen_t alter_gen, int delta,
                                       Barrier::timestamp_t ts,
                                       Barrier::timestamp_t prev_ts, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // step 1 - ORDERING.  One node's alterations are applied in the order it
    //  issued them: an alteration whose predecessor has not landed yet is held
    //  right here.  That is what makes a single timestamp on an arrival a
    //  sound gate for the entire chain behind it, and it is checked by exact
    //  membership rather than by comparing timestamps.
    if((prev_ts != 0) && (applied_ts.count(prev_ts) == 0)) {
      HeldAlter ha;
      ha.gen = alter_gen;
      ha.delta = delta;
      ha.ts = ts;
      held_alters.insert(std::make_pair(prev_ts, ha));
      log_barrier.info() << "holding barrier alteration: " << me << "/" << alter_gen
                         << " delta=" << delta << " ts=" << ts << " needs=" << prev_ts;
      return;
    }

    // a worklist rather than recursion: applying this one may release others
    std::vector<HeldAlter> ready;
    {
      HeldAlter ha;
      ha.gen = alter_gen;
      ha.delta = delta;
      ha.ts = ts;
      ready.push_back(ha);
    }

    while(!ready.empty()) {
      const HeldAlter cur = ready.back();
      ready.pop_back();

      // an alteration is applied EXACTLY ONCE.  The transport does not
      //  duplicate messages, so this is belt and braces - but a delta applied
      //  twice is a barrier that never triggers, which is expensive to debug.
      //  A timestamp of 0 is not an identity: it means "this alteration opens
      //  no causal branch", which several of them can do independently.
      if((cur.ts != 0) && (applied_ts.count(cur.ts) != 0)) {
        log_barrier.info() << "ignoring duplicate barrier alteration: " << me << "/"
                           << cur.gen << " ts=" << cur.ts;
        continue;
      }

      gen_t apply_gen = cur.gen;
      const gen_t watermark = generation.load();
      if(apply_gen <= watermark) {
        // CONTRACT VIOLATION (event.h:302-306): the generation the application
        //  named has already triggered, so it triggered without this
        //  alteration - "a barrier that triggered too early".  It is reported
        //  rather than fatal because the damage is already done, and the delta
        //  is still applied persistently from the next open generation on, so
        //  that every later count stays consistent.
        log_barrier.fatal() << "barrier alteration for a triggered generation: " << me
                            << "/" << cur.gen << " delta=" << cur.delta
                            << " watermark=" << watermark;
        apply_gen = watermark + 1;
      }

      alter_steps[apply_gen] += cur.delta;
      // 0 is the "no causal branch" sentinel and never goes in the applied set:
      //  ts_gate_open_locked() already reads a zero in a gate as "no dependency
      //  here", so an entry for it would mean nothing and would collide with
      //  the next untimestamped alteration.
      if(cur.ts != 0) {
        applied_ts[cur.ts] = apply_gen;
      }

      // RULE 9 - a base arrival count of zero is an error.  It is also what
      //  makes 0 a safe "not yet known" sentinel elsewhere.  'expected' is a
      //  step function, so checking it at the first open generation and at
      //  every breakpoint above the watermark covers every generation.
      {
        bool bad = (expected_locked(watermark + 1) <= 0);
        for(std::map<gen_t, int64_t>::const_iterator it = alter_steps.begin();
            !bad && (it != alter_steps.end()); ++it) {
          if(it->first > watermark) {
            bad = (expected_locked(it->first) <= 0);
          }
        }
        if(bad) {
          log_barrier.fatal() << "barrier arrival count driven to zero or below: " << me
                              << "/" << apply_gen << " delta=" << cur.delta;
          abort();
        }
      }

      // step 2 - the gate this timestamp was holding shut is now open
      if(cur.ts != 0) {
        open_ts_gates_locked(cur.ts);
      }

      // step 3 - RULE 9.  The plan is now wrong whichever way the count moved
      //  (see retire_plan_locked), so it is retired; the next trigger builds a
      //  replacement out of what the resulting eager reports gather.
      retire_plan_locked(sends);

      if(cur.delta < 0) {
        // RULE 9 - a NEGATIVE alteration also behaves as an ARRIVAL: it reduces
        //  the count still outstanding, which on a count-up accumulator is
        //  exactly what lowering 'expected' does.  ONE effect, applied once -
        //  crediting an arrival as well would count it twice.
        //
        // Eager flush over the affected open generations is what the state and
        //  locking document asks for here.  It overlaps the invalidation above,
        //  which flushes every open generation at each node it reaches, but
        //  rule 4 is idempotent so the overlap costs nothing.
        enter_flush_from_locked(apply_gen, sends);

        // THE TERMINAL-NEGATIVE CASE (event.h:293-297): a negative alteration
        //  that drives the remaining count to exactly zero needs no arrival on
        //  the returned handle, so it is the one branch of the contract whose
        //  safety argument does NOT rest on a reserved arrival.  It has no
        //  formal coverage (ARRIVAL_PROTOCOL section 8.6), so it is gated here
        //  rather than implemented speculatively.
        for(std::map<gen_t, Generation *>::iterator it = generations.begin();
            it != generations.end(); ++it) {
          if((it->first <= watermark) || (it->first < apply_gen)) {
            continue;
          }
          if(generation_complete_locked(it->first, *(it->second))) {
            log_barrier.fatal()
                << "terminal-negative alter_arrival_count is not implemented: " << me
                << "/" << it->first << " delta=" << cur.delta
                << " (a negative alteration that completes a generation by itself"
                << " has no formal coverage - see ARRIVAL_PROTOCOL section 8.6)";
            abort();
          }
        }
      }

      log_barrier.info() << "applied barrier alteration: " << me << "/" << apply_gen
                         << " delta=" << cur.delta << " ts=" << cur.ts
                         << " expected=" << expected_locked(apply_gen);

      // step 3 - release anything that was waiting for THIS alteration
      std::pair<std::multimap<Barrier::timestamp_t, HeldAlter>::iterator,
                std::multimap<Barrier::timestamp_t, HeldAlter>::iterator>
          range = held_alters.equal_range(cur.ts);
      for(std::multimap<Barrier::timestamp_t, HeldAlter>::iterator it = range.first;
          it != range.second; ++it) {
        ready.push_back(it->second);
      }
      held_alters.erase(range.first, range.second);
    }
  }

  // ARRIVAL_PROTOCOL section 8.8 - retained state must be reclaimable.
  //
  // 'alter_steps' entries at or below the watermark can never be un-applied, so
  //  they fold into the floor.  'ts_floor' answers "the greatest entry with key
  //  <= gen" and arrivals only ever happen above the watermark, so every
  //  breakpoint strictly below the greatest one that is <= watermark + 1 is
  //  unreachable.
  //
  // NOT pruned here: 'applied_ts'.  Dropping an entry would make an alteration
  //  chained behind it (BarrierAlterMessage::prev_ts) wait forever, so it needs
  //  a companion "already pruned, therefore applied" test to be safe.  What is
  //  retained is one entry per alteration ever issued on this barrier, which
  //  tla/STATE_AND_LOCKING.md section 7.3 already calls the residual of this
  //  design; shrinking it to one entry per altering node is deferred.
  void BarrierImpl::prune_alter_state_locked(gen_t watermark)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    while(!alter_steps.empty() && (alter_steps.begin()->first <= watermark)) {
      alter_floor += alter_steps.begin()->second;
      alter_steps.erase(alter_steps.begin());
    }

    if(!ts_floor.empty()) {
      std::map<gen_t, Barrier::timestamp_t>::iterator it =
          ts_floor.upper_bound(watermark + 1);
      if(it != ts_floor.begin()) {
        --it; // the newest breakpoint an arrival can still name
        ts_floor.erase(ts_floor.begin(), it);
      }
    }
  }

  // ARRIVAL_PROTOCOL section 11.1 - the (node,count) map this node speaks for.
  //
  // Merging is per-node REPLACE-IF-HIGHER and never addition.  Each entry is
  //  the count of arrivals that happened AT THAT NODE, so the same node showing
  //  up under two different children - which happens mid-switch, when it has
  //  reported to its old parent and then to its new one, and both are children
  //  of ours - must contribute ONCE, at its highest observed value.  Summing
  //  would inflate the count, and an inflated count becomes an unreachable
  //  quota, which is the one plan defect that deadlocks (see
  //  build_new_plan_locked).
  void BarrierImpl::merge_counts_locked(const Generation &g,
                                        std::map<NodeID, int64_t> &out) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    out.clear();
    if(g.local_total > 0) {
      out[Network::my_node_id] = g.local_total;
    }
    for(std::map<NodeID, Generation::ChildReport>::const_iterator it =
            g.child_acc.begin();
        it != g.child_acc.end(); ++it) {
      for(std::map<NodeID, int64_t>::const_iterator cit = it->second.counts.begin();
          cit != it->second.counts.end(); ++cit) {
        int64_t &slot = out[cit->first];
        if(cit->second > slot) {
          slot = cit->second;
        }
      }
    }
  }

  // A report is CUMULATIVE (rule 7): 'val' is this node's running subtree total
  //  for the generation, never an increment, so a later report supersedes an
  //  earlier one at the receiver and duplicate or out-of-order delivery is
  //  harmless.  'reported_up' is what makes the next Unreported() test exact.
  void BarrierImpl::record_report_locked(Generation &g, gen_t gen, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // build straight into the outbound slot - the map is a vector, and copying
    //  a PendingReport around would copy it with them
    PendingReport *r;
    if(sends.report.to == static_cast<NodeID>(-1)) {
      // the scalar slot first, so nothing on the steady-state path allocates
      r = &sends.report;
    } else {
      sends.more_reports.push_back(PendingReport());
      r = &sends.more_reports.back();
    }

    r->gen = gen;
    // RULE 7 AND THE PINNED TARGET (Generation::report_to).  The receiver keys
    //  its accumulator on the SENDER and REPLACES that sender's previous
    //  contribution, so this node's contribution to this generation has to
    //  reach the owner down exactly ONE chain.  The first report fixes which
    //  one; a plan installed later replaces 'cur_plan.parent' but must not
    //  re-aim a generation that has already been spoken for, or the same
    //  arrivals arrive at the common ancestor under two sender keys and are
    //  summed - with no way to retract the first, since a cumulative total may
    //  never go down.
    //
    // This is the general form of the hazard the note below describes for a
    //  RETIRED node.  It bites on the FIRST plan too: before any plan exists
    //  every node reports 'direct' to the owner, and the first plan puts most
    //  of them under a relay.
    const NodeID target = report_target_locked();
    if(g.report_to == static_cast<NodeID>(-1)) {
      g.report_to = target;
      counters.report_edges_pinned++;
    } else if(g.report_to != target) {
      // the pin is doing its job: the current plan would have re-aimed this
      //  generation's chain, and rule 10.1 says it must not
      counters.pin_conflicts_avoided++;
    }
    r->to = g.report_to;
    r->val = g.subtree_known();
    // decision Q4 - the poison bit is STICKY for the generation and is OR-ed
    //  into every report this node sends, so it needs no separate message and
    //  no acknowledgement.  It always rides the same report as the arrival
    //  COUNT that set it, which the generation cannot complete without, so it
    //  cannot be lost to the receiver's staleness test.
    r->poisoned = g.poisoned;

    // rule 3: a node holding no plan record - or one whose plan has been
    //  retired - has no completion condition the owner's plan predicted, so its
    //  report is a 'direct': "your plan did not account for me".
    //
    // 'saw_direct' propagates that upward.  The target of a report is the
    //  PINNED one above, never re-aimed mid-generation, because re-aiming would
    //  let the same arrivals reach the owner down two chains and be counted
    //  twice.  Keeping the routing fixed means the signal has to travel as a
    //  flag instead, or the first relay swallows it and the owner never learns
    //  its plan is wrong.  (Not modelled: BarrierArrive's scenarios never have a
    //  previously-planned node deviate.  Over-signalling a flush is always safe
    //  - it only ever causes MORE eager reporting - so this errs in the safe
    //  direction.)
    //
    // Two more cases raise it, and in both a node the plan DID predict is
    //  telling the owner that the plan is wrong:
    //
    //   * an OVER-ARRIVAL (rule 2).  The plan under-predicted this node, and
    //     only the owner can fix that - but rule 2's own fan-out goes DOWNWARD
    //     only, so without this the owner never hears about it and any
    //     below-quota relay between here and there stays silent forever.
    //     BarrierArrive does not need the signal because an invalidated node is
    //     flushing for the rest of the run and forwards regardless; this
    //     implementation lets aggregation resume, so the signal has to be real.
    //   * a report travelling a STALE EDGE, i.e. one whose pinned target is no
    //     longer the parent the current plan gives us.  Its receiver may have
    //     no reason of its own to speak, and everything behind that report
    //     would then be stuck behind ITS completion condition.
    r->is_direct =
        !cur_plan.inplan || plan_retired_locked() || g.saw_direct ||
        (g.report_to != target) ||
        (cur_plan.inplan && (g.local_total > static_cast<int64_t>(cur_plan.quota)));

    // section 11.1: only an EAGER-FLUSH report carries the per-node map.  A
    //  steady-state rule-1 report stays O(1) - which is the entire reason the
    //  tree exists - so the owner learns the arrival pattern only from the
    //  deviation path, and only for as long as a deviation is in progress.
    if(flush_mode_locked(gen, g)) {
      std::map<NodeID, int64_t> merged;
      merge_counts_locked(g, merged);
      encode_flush_map(merged, r->flush_map);

      // IMPLEMENTATION_PLAN section 5 - EAGER-FLUSH REPORT SIZES.  This is the
      //  only O(subtree) payload the protocol emits, and the peak is what says
      //  whether Q6's "let active-message fragmentation handle it" is still the
      //  right answer at scale.
      counters.flush_report_bytes += r->flush_map.size();
      if(r->flush_map.size() > counters.flush_report_bytes_max) {
        counters.flush_report_bytes_max = r->flush_map.size();
      }
    }

    g.reported_up = r->val;
  }

  // BarrierArrive's 'held' set, shared by RecvInvalidate and RecvNewPlan:
  //
  //   held == { g \in Gens : ~triggered[g] /\ Unreported(m.to, g) > 0 }
  //
  //  Note that this is UNCONDITIONAL - it does not consult 'flushing' or the
  //  plan-satisfied test.  Both rules report everything the node is sitting on,
  //  whatever mode it is in, because the tree that work was being aggregated
  //  for is going away.  The only guard is the model's Send(), which is empty
  //  at the owner: the owner has no parent, and its subtree total is the answer
  //  rather than a report.
  //
  // The target of each report is resolved HERE, inside the critical section and
  //  while cur_plan is still the OLD record - which is what BarrierArrive's
  //  unprimed Send()/ParentOf() means, and what S3 guarantees for free.
  void BarrierImpl::report_held_locked(PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(owner == Network::my_node_id) {
      return;
    }

    const gen_t watermark = generation.load();
    for(std::map<gen_t, Generation *>::iterator it = generations.begin();
        it != generations.end(); ++it) {
      // a generation this node already knows has triggered has nothing left to
      //  collect (the model's ~triggered[g])
      if(it->first <= watermark) {
        continue;
      }
      if(it->second->holding()) {
        record_report_locked(*(it->second), it->first, sends);
      }
    }
  }

  // ARRIVAL_PROTOCOL rule 4 - FLUSH IS PER GENERATION AND IDEMPOTENT.
  //
  // This is the shared core of rules 2, 3 and 4, which differ only in what
  //  causes them (an over-arrival here, a 'direct' arriving at the owner, or a
  //  flush announcement from our parent).  In every case:
  //
  //   * a node ALREADY flushing this generation changes nothing and sends
  //     nothing.  THAT IS WHAT TERMINATES THE FAN-OUT - without the early out
  //     the announcement would circulate around the tree forever.
  //   * otherwise the flag goes on for THIS GENERATION ONLY, the announcement
  //     is re-fanned to this node's children, and anything this node is
  //     currently HOLDING for the generation is reported at once.  Reporting
  //     held work on entry is the critical half: a node that has gone quiet
  //     waiting on a completion condition its stale plan will never meet is
  //     exactly the silence this protocol exists to break.
  void BarrierImpl::enter_flush_locked(Generation &g, gen_t gen, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(g.flushing) {
      return;
    }
    g.flushing = true;

    // IMPLEMENTATION_PLAN section 5 - EAGER-FLUSH EPISODES.  One per generation
    //  this node actually entered flush for; the early out above is what makes
    //  it an episode count rather than a message count.
    counters.flush_episodes++;

    // ARRIVAL_PROTOCOL section 11.1 - GATHERING STARTS HERE.  At the owner,
    //  entering flush is the one and only thing that says "my plan does not
    //  describe reality": it happens on a 'direct' from a node the plan did not
    //  predict (rule 3) and on an over-arrival at the owner itself (rule 2),
    //  and never in steady state.  From this moment every report that reaches
    //  the owner for this generation carries a (node,count) map, and the next
    //  trigger turns those maps into the next plan.
    if(owner == Network::my_node_id) {
      plan_rebuild_pending = true;
    }

    // fan out to the children of the plan record AS IT IS NOW.  The targets are
    //  resolved here, inside the critical section, so nothing that later
    //  replaces 'cur_plan' can strand this subtree (S3).
    // One fan PER GENERATION: rule 8's alteration path enters flush for every
    //  affected open generation in a single action, so these accumulate.
    if(!cur_plan.kids.empty()) {
      sends.flushes.push_back(PendingFlush());
      PendingFlush &pf = sends.flushes.back();
      pf.gen = gen;
      pf.to.insert(pf.to.end(), cur_plan.kids.begin(), cur_plan.kids.end());
    }

    // report whatever we are holding.  With 'flushing' now set, this test is
    //  exactly the model's "not the owner and Unreported > 0".
    if(should_report_locked(gen, g)) {
      record_report_locked(g, gen, sends);
    }
  }

  // ARRIVAL_PROTOCOL rule 8.3 - ISSUING AN ALTERATION PUTS THE NODE INTO EAGER
  //  FLUSH FOR EVERY AFFECTED OPEN GENERATION.  This is the easy one to miss,
  //  and removing it is a caught mutation (deadlock, not a safety violation):
  //  a node whose arrivals bypass the tree can never satisfy its own plan
  //  quota, so as a RELAY it would go silent and strand its children.
  //
  // Only generations this node already has a record for can have a flag set
  //  here.  Later ones are covered by flush_mode_locked() consulting the
  //  arrival floor, which is the same trick rule 6 uses for the generations it
  //  cannot flag either.
  void BarrierImpl::enter_flush_from_locked(gen_t from_gen, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    const gen_t watermark = generation.load();
    for(std::map<gen_t, Generation *>::iterator it = generations.begin();
        it != generations.end(); ++it) {
      if((it->first < from_gen) || (it->first <= watermark)) {
        continue;
      }
      enter_flush_locked(*(it->second), it->first, sends);
    }
  }

  namespace {
    // The decoded view of one 'newplan' payload - see BarrierNewPlanMessage for
    //  the wire format.  Each child's record is a contiguous slice of the SAME
    //  buffer, so forwarding is a copy of those bytes and never a re-encode.
    struct PlanSubtree {
      uint32_t quota = 0;
      std::vector<NodeID> kids;
      std::vector<std::pair<const unsigned char *, size_t>> kid_payload;
    };

    bool decode_plan_subtree(const void *data, size_t datalen, PlanSubtree &out)
    {
      const unsigned char *p = static_cast<const unsigned char *>(data);
      if((p == 0) || (datalen < (2 * sizeof(uint32_t)))) {
        return false;
      }

      uint32_t quota = 0;
      uint32_t num_kids = 0;
      memcpy(&quota, p, sizeof(uint32_t));
      memcpy(&num_kids, p + sizeof(uint32_t), sizeof(uint32_t));
      size_t off = 2 * sizeof(uint32_t);

      // each child costs at least its own (node, length) header plus an empty
      //  record, so a count above that is a truncated or corrupt payload and
      //  must be rejected BEFORE it is used to size anything
      if(num_kids > ((datalen - off) / (4 * sizeof(uint32_t)))) {
        return false;
      }

      out.quota = quota;
      out.kids.clear();
      out.kid_payload.clear();
      out.kids.reserve(num_kids);
      out.kid_payload.reserve(num_kids);

      for(uint32_t i = 0; i < num_kids; i++) {
        if((datalen - off) < (sizeof(int32_t) + sizeof(uint32_t))) {
          return false;
        }
        int32_t kid = 0;
        uint32_t kid_bytes = 0;
        memcpy(&kid, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        memcpy(&kid_bytes, p + off, sizeof(uint32_t));
        off += sizeof(uint32_t);
        if((datalen - off) < kid_bytes) {
          return false;
        }
        out.kids.push_back(static_cast<NodeID>(kid));
        out.kid_payload.push_back(
            std::make_pair(p + off, static_cast<size_t>(kid_bytes)));
        off += kid_bytes;
      }

      // a payload with trailing bytes is malformed, not merely generous
      return (off == datalen);
    }

    // Fan-out of a constructed plan.  This is a shape choice, not a protocol
    //  rule - the model verifies that ANY plan is safe to adopt - and 8 keeps a
    //  few-thousand-node barrier at depth 4.
    //
    // It is overridable (-ll:barrier_plan_radix) because the tree DEPTH is what
    //  makes the rule-10 race windows reachable: below ~radix+2 ranks every
    //  plan is a flat owner->all tree with no relays, so parks, dead-plan
    //  discards and stale edges are structurally impossible.  Radix 2 puts
    //  relay trees at 4 ranks, which is what lets one workstation exercise the
    //  paths TLC verified (tla/SCALE_TEST_PLAN.md).  Values below 2 are
    //  clamped: radix 1 would make a chain of every node, and 0 is meaningless.
    static size_t barrier_plan_radix(void)
    {
      static const size_t radix = []() -> size_t {
        int configured = 0;
        RuntimeImpl *runtime = get_runtime();
        if(runtime != nullptr) {
          ModuleConfig *core = runtime->get_module_config("core");
          if((core != nullptr) &&
             (core->get_property("barrier_plan_radix", configured) == REALM_SUCCESS) &&
             (configured >= 2)) {
            return static_cast<size_t>(configured);
          }
        }
        return 8;
      }();
      return radix;
    }

    void append_u32(std::vector<unsigned char> &out, uint32_t v)
    {
      const unsigned char *p = reinterpret_cast<const unsigned char *>(&v);
      out.insert(out.end(), p, p + sizeof(uint32_t));
    }

    // ARRIVAL_PROTOCOL section 11.4 - the other half of decode_plan_subtree.
    //
    // The tree is held as a flat array in k-ary heap order: 'tv[0]' is the
    //  owner and the children of index i are i*radix+1 .. i*radix+radix.  That
    //  makes "the subtree rooted at i" a pure index computation, so the
    //  pre-order encoding below needs no auxiliary structure at all.
    //
    // Each child's record is written as a CONTIGUOUS, length-prefixed slice, so
    //  the relay that receives it forwards its own children's slices verbatim -
    //  it never re-encodes, and never has to understand anything below its own
    //  depth.  The length is back-patched because it is not known until the
    //  child's whole subtree has been written.
    void encode_plan_subtree(size_t idx, const std::vector<NodeID> &tv,
                             const std::vector<uint32_t> &quota, size_t radix,
                             std::vector<unsigned char> &out)
    {
      const size_t first_kid = (idx * radix) + 1;
      size_t num_kids = 0;
      while((num_kids < radix) && ((first_kid + num_kids) < tv.size())) {
        num_kids++;
      }

      append_u32(out, quota[idx]);
      append_u32(out, static_cast<uint32_t>(num_kids));

      for(size_t k = 0; k < num_kids; k++) {
        const size_t kid_idx = first_kid + k;
        append_u32(out, static_cast<uint32_t>(static_cast<int32_t>(tv[kid_idx])));
        const size_t len_pos = out.size();
        append_u32(out, 0); // back-patched below
        const size_t start = out.size();
        encode_plan_subtree(kid_idx, tv, quota, radix, out);
        const uint32_t kid_bytes = static_cast<uint32_t>(out.size() - start);
        memcpy(&out[len_pos], &kid_bytes, sizeof(uint32_t));
      }
    }
  } // namespace

  // ARRIVAL_PROTOCOL rule 5 - install a plan record.  Reached from action P
  //  (the plan arrived and this node was free to take it) and from action I
  //  step 5 (a plan that was PARKED is applied once the invalidation lands).
  //
  // The order inside here is load-bearing:
  //
  //   1. the fan-out to the NEW children is recorded from the payload, and each
  //      child's slice is COPIED, because the active-message buffer is dead the
  //      moment the handler returns;
  //   2. held work is reported while cur_plan is still the OLD record.
  //      BarrierArrive evaluates Send()/ParentOf() in the unprimed state, so those
  //      reports are owed to the OLD parent - the node that has been collecting
  //      for this subtree all along and whose childAcc already holds part of
  //      it.  Re-aiming them at the new parent would let the same arrivals
  //      reach the owner down two paths at once;
  //   3. only then is the record replaced.
  //
  // 'flushing' is deliberately NOT set (BarrierArrive RecvNewPlan leaves it alone):
  //  the new plan is authoritative, so the node can go back to aggregating.
  void BarrierImpl::apply_plan_locked(uint32_t epoch, const void *data, size_t datalen,
                                      NodeID parent, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    PlanSubtree sub;
    if(!decode_plan_subtree(data, datalen, sub)) {
      log_barrier.fatal() << "malformed barrier plan payload: " << me
                          << " epoch=" << epoch << " bytes=" << datalen;
      abort();
    }

    // (1) record the fan-out to the new children, copying each subtree slice
    sends.fwd_plan_epoch = epoch;
    for(size_t i = 0; i < sub.kids.size(); i++) {
      PendingPlan pp;
      pp.to = sub.kids[i];
      pp.payload.assign(sub.kid_payload[i].first,
                        sub.kid_payload[i].first + sub.kid_payload[i].second);
      sends.plan_to.push_back(pp);
    }

    // (2) report held work with the OLD plan still in place
    report_held_locked(sends);

    // (3) switch
    cur_plan.quota = sub.quota;
    // MEMBERSHIP (section 3): only nodes with a non-zero expected contribution
    //  are in a plan at all, so being sent a record IS membership.
    cur_plan.inplan = true;
    cur_plan.parent = parent;
    cur_plan.kids = sub.kids;
    my_epoch = epoch;

    // A PLAN INSTALL NEVER CLEARS FLUSH (BarrierArrive, a caught mutation on
    //  MCStale - installing a plan used to return open generations to planned
    //  mode right here, and TLC kills it): the owner's deviation flush and the
    //  newplan RACE, and when the install wins the flush is lost and the
    //  generation strands behind a quota the new plan cannot meet.  A flushed
    //  generation stays eager until IT TRIGGERS, which is naturally bounded -
    //  the flag lives on the Generation record and dies with it.  New
    //  generations start planned under whatever plan this node holds, so
    //  nothing is sticky across the barrier's future.

    // (4) RE-FAN THE FLUSH DOWN THE NEW CHILD LIST.  A flush announcement is
    //  re-fanned by each recipient through the child list IT holds at the
    //  moment it arrives, so an announcement that overtakes this plan on the
    //  wire is fanned into the OLD subtree and never reaches the children this
    //  node has just acquired.  Those children would then go back to
    //  aggregating for a generation the owner is trying to drain.  The flush is
    //  idempotent per generation (rule 4), so a child that already has it drops
    //  this one and the fan-out still terminates.
    if(!cur_plan.kids.empty()) {
      const gen_t watermark = generation.load();
      for(std::map<gen_t, Generation *>::const_iterator it = generations.begin();
          it != generations.end(); ++it) {
        if((it->first <= watermark) || !it->second->flushing) {
          continue;
        }
        sends.flushes.push_back(PendingFlush());
        PendingFlush &pf = sends.flushes.back();
        pf.gen = it->first;
        pf.to.insert(pf.to.end(), cur_plan.kids.begin(), cur_plan.kids.end());
      }
    }

    log_barrier.info() << "installed barrier plan: " << me << " epoch=" << epoch
                       << " quota=" << cur_plan.quota << " parent=" << parent
                       << " kids=" << cur_plan.kids.size();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL section 11.2 and 11.4 - CONSTRUCTION and DISTRIBUTION.
  //  OWNER ONLY, called from inside the trigger critical section.
  //
  // 'agg' is the merged node -> arrival-count structure of section 11.2, and it
  //  is a LOCAL of the caller: it was merged out of the generation record that
  //  is about to be freed and it dies with the stack frame.  Section 11.3 is
  //  satisfied by there being nowhere for it to persist, which is decision D9 -
  //  a leak here would not fail loudly, it would just cost O(N) per barrier
  //  forever, and that is the exact blow-up this design exists to avoid.
  //
  // THE ONE INVARIANT PLAN CONSTRUCTION MUST NOT BREAK:
  //
  //   sum of quotas <= the arrival count the owner will be waiting for
  //
  // Under-estimating a node's quota is SAFE - it over-arrives, rule 2 puts it
  //  into eager flush, and everything it is holding comes out.  Over-estimating
  //  is FATAL: a node below its quota is silent by design (that silence is the
  //  aggregation), and if the plan asks for arrivals that never happen it stays
  //  silent forever and the generation never triggers.  With the sum bounded,
  //  total arrivals reaching the expected count forces at least one node either
  //  over its quota or outside the plan entirely, and both of those are
  //  deviation signals that break the silence.
  //
  // Because every entry in 'agg' is one node's own cumulative local total for a
  //  generation that has just triggered, their sum is at most that generation's
  //  arrival count - so the invariant holds by construction.  It is still
  //  checked, because it is the one place a construction bug turns into a
  //  hang instead of an assertion.
  void BarrierImpl::build_new_plan_locked(const std::map<NodeID, int64_t> &agg,
                                          int64_t tree_arrivals, PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    const NodeID me_node = Network::my_node_id;

    // IMPLEMENTATION_PLAN section 5 - PEAK AGGREGATION SIZE.  'agg' is a LOCAL
    //  of the trigger path (D9), merged out of the generation record that is
    //  about to be freed, so O(N) barriers each holding an O(N) participant map
    //  is structurally impossible rather than merely avoided.  Nothing would
    //  fail loudly if that ever stopped being true - it would just scale badly
    //  - so this is the number that would show it.  Recorded before the
    //  declines below, because a rebuild that is turned down still had to build
    //  the map.
    if(agg.size() > counters.agg_peak_entries) {
      counters.agg_peak_entries = agg.size();
    }

    // (1) MEMBERSHIP (sections 3 and 11.2): ONLY nodes with a non-zero count
    //     appear.  A node with no predicted arrivals is not a leaf, not a
    //     relay, not anything - it is left permanently in eager-flush mode, and
    //     that is what stops a relay ever waiting on a child that will never
    //     speak.
    //
    //     The owner is the one exception: it is always the root, even with a
    //     quota of zero, because it has to hold the top-level child list or
    //     nothing can fan an invalidation or a flush into the tree at all.
    //     (BarrierArrive's own scenarios do this - MCSeven.tla:52 gives node 0
    //     'quota |-> 0, inplan |-> TRUE'.)
    std::vector<NodeID> tv; // tv[0] is the owner; k-ary heap order
    std::vector<uint32_t> quota;
    int64_t total = 0;
    // every arrival the maps ACCOUNT FOR, including the ones belonging to nodes
    //  deliberately left out below.  This is what the completeness test weighs;
    //  'total' is what the over-prediction test weighs.
    int64_t gathered = 0;

    tv.push_back(me_node);
    quota.push_back(0);

    for(std::map<NodeID, int64_t>::const_iterator it = agg.begin(); it != agg.end();
        ++it) {
      if(it->second <= 0) {
        continue;
      }
      gathered += it->second;
      // RULE 8.1 / MEMBERSHIP.  A node that has altered this barrier bypasses
      //  the tree from its alteration's generation on, FOREVER (the alteration
      //  is persistent), so its 'local_total' never moves again and any quota
      //  this plan gave it would be unreachable.  'agg' is one PAST
      //  generation's counts, so a node that altered for a LATER generation is
      //  still in that evidence and would otherwise be re-admitted by every
      //  rebuild - as a leaf, it strands its parent on rule 1's child-wait
      //  forever; as a relay, it goes silent and strands its whole subtree.
      //  Leaving it out costs only that its arrivals stay on the bypass path,
      //  which is where rule 8.1 puts them anyway.
      if(ts_bypass_nodes.count(it->first) != 0) {
        log_barrier.info() << "excluding altered node from barrier plan: " << me
                           << " node=" << it->first;
        continue;
      }
      if(it->second > static_cast<int64_t>(UINT32_MAX)) {
        counters.plan_rebuilds_declined++;
        log_barrier.info() << "declining barrier plan rebuild (count overflow): " << me
                           << " node=" << it->first << " count=" << it->second;
        return;
      }
      total += it->second;
      if(it->first == me_node) {
        quota[0] = static_cast<uint32_t>(it->second);
      } else {
        // LOAD-BEARING ORDERING: 'agg' is a std::map, so this appends in
        //  ASCENDING NodeID order, and the heap parent of index i is
        //  (i-1)/radix < i.  Every non-owner therefore gets a parent that is
        //  either the owner or a node with a smaller NodeID, in this plan and
        //  in every other one.  That is what keeps a report edge - including a
        //  STALE one, left behind by a plan switch (Generation::report_to, and
        //  a retired node still reporting to its old parent) - strictly
        //  decreasing, and so what makes a cycle of them impossible.
        tv.push_back(it->first);
        quota.push_back(static_cast<uint32_t>(it->second));
      }
    }

    if(total <= 0) {
      return;
    }

    // COMPLETENESS (section 11.2, "from that it has both the participant set
    //  and the expected count per node, which is exactly a plan").  The merged
    //  maps are the participant set only when EVERY participant was eager when
    //  it reported, because only an eager-flush report carries a map (section
    //  11.1).  A node that met its quota under a valid plan reports a bare
    //  count and is then no longer holding anything, so the owner's flush
    //  announcement - which is at least two hops behind a locally satisfied
    //  quota, and is not even emitted until this critical section unlocks -
    //  finds nothing left to re-report and that node contributes no (node,
    //  count) pair at all.
    //
    // Building from partial evidence is SAFE (every omitted node is left
    //  outside the plan and therefore permanently eager) but it is not a plan:
    //  it invalidates nodes that were behaving correctly, and since those nodes
    //  then report eagerly - with maps - while the ones just admitted report
    //  bare counts, the NEXT rebuild gathers the complement and swaps the two
    //  halves back.  The set can oscillate for the life of the barrier, paying
    //  an invalidate and a newplan fan-out every generation and never reaching
    //  the steady state the tree exists to provide.
    //
    // So: build only from evidence that accounts for every arrival that went
    //  through the tree.  Declining costs nothing - the current plan survives
    //  untouched (nothing has been retired at this point), the deviating node
    //  stays on the eager path where rule 3 already puts it, and the next
    //  deviation re-arms the rebuild.  A generation that genuinely cannot
    //  complete under the stale plan forces a full flush before it triggers,
    //  and that flush is exactly what makes the gathering complete.
    if((tree_arrivals > 0) && (gathered != tree_arrivals)) {
      counters.plan_rebuilds_declined++;
      log_barrier.info() << "declining barrier plan rebuild (partial evidence): " << me
                         << " gen=" << (generation.load()) << " gathered=" << gathered
                         << " of " << tree_arrivals;
      // THE GATHERING GENERATION (ARRIVAL_PROTOCOL section 11.5).  Partial
      //  evidence is STRUCTURAL, not transient: a deviation discovered
      //  mid-generation can never yield complete maps, because the traffic
      //  that flowed before the flush reached its senders was bare rule-1
      //  counts.  Declining alone therefore never converges - under a
      //  PERSISTENT pattern shift every generation declines the same way and
      //  the barrier stays eager forever (the P_STEP probe measures exactly
      //  this).  So the owner DECLARES the next generation a gathering
      //  generation: it enters flush for it now, and the fan-out below makes
      //  every node eager for that generation from its first arrival.  Its
      //  evidence is then complete BY CONSTRUCTION, the rebuild at its
      //  trigger succeeds, and the plan it produces governs the generation
      //  after.  Cost: one fully-eager generation per pattern shift.
      //  enter_flush_locked is idempotent and re-arms plan_rebuild_pending,
      //  so repeated declines while the gathering generation is already
      //  declared collapse into one declaration.
      if(declines_until_gather > 0) {
        // backing off: the last gathering(s) reproduced the current plan, so
        //  this deviation pattern has no better plan to learn right now
        declines_until_gather--;
      } else {
        const gen_t gather_gen = generation.load() + 1;
        Generation *ng = get_generation_locked(gather_gen);
        if(!ng->flushing) {
          counters.gathering_declared++;
          log_barrier.info() << "declaring gathering generation: " << me
                             << " gen=" << gather_gen;
        }
        enter_flush_locked(*ng, gather_gen, sends);
      }
      return;
    }

    // THE INVARIANT, measured against the count this plan's FIRST generation
    //  will actually be waiting for - not against 'base_arrival_count', which a
    //  NEGATIVE alteration (rule 9) puts above the truth.  A later positive
    //  alteration only raises expected(), which is the safe direction; a later
    //  negative one invalidates the plan outright, which is what rule 9 is for.
    const gen_t plan_from = generation.load() + 1;
    const int64_t expected = expected_locked(plan_from);
    if(total > expected) {
      counters.plan_rebuilds_declined++;
      log_barrier.info() << "declining barrier plan rebuild (would over-predict): " << me
                         << " sum=" << total << " expected=" << expected;
      return;
    }

    // IMPLEMENTATION_PLAN section 5 - PLAN REBUILD FREQUENCY.  Counted past the
    //  declines, so it measures plans actually installed: a steady workload
    //  should build one and then stop.
    // THE IDENTICAL-PLAN SKIP (section 11.5).  Under an ALTERNATING pattern -
    //  no single plan fits, e.g. every odd generation deviates the same way -
    //  each gathering generation reproduces the plan already installed, and
    //  installing it would buy an invalidate+newplan broadcast per rebuild for
    //  nothing (measured: 65 rebuilds per 128 generations on the OVER probe).
    //  A 64-bit hash stands in for the last plan; on a match the install is
    //  skipped and the GATHERING BACKOFF doubles, so an adversarial pattern
    //  decays to the pre-gathering behaviour (deviating generations flush,
    //  conforming ones aggregate) instead of gathering forever.  A pattern
    //  shift that produces a DIFFERENT plan resets the backoff, so a genuine
    //  step-change still converges immediately.
    {
      uint64_t h = 0x9E3779B97F4A7C15ull;
      for(size_t i = 0; i < tv.size(); i++) {
        h = (h ^ static_cast<uint64_t>(tv[i])) * 0x100000001B3ull;
        h = (h ^ static_cast<uint64_t>(quota[i])) * 0x100000001B3ull;
      }
      if((my_epoch != 0) && (h == last_plan_hash)) {
        counters.identical_plans_skipped++;
        gather_backoff = (gather_backoff == 0)
                             ? 1
                             : ((gather_backoff < 32) ? (gather_backoff * 2) : 32);
        declines_until_gather = gather_backoff;
        log_barrier.info() << "skipping identical barrier plan: " << me
                           << " gen=" << generation.load()
                           << " backoff=" << gather_backoff;
        return;
      }
      last_plan_hash = h;
      gather_backoff = 0;
      declines_until_gather = 0;
    }

    counters.plan_rebuilds++;

    const uint32_t new_epoch = next_epoch++;

    // (2) RULE 6, FORWARD BEFORE FORGETTING.  The invalidation goes down the
    //     tree BEING RETIRED, so its targets come from the CURRENT child list
    //     and are resolved into 'sends' here, before step 3 replaces that list.
    //     'my_epoch' is the epoch being retired, which is what the recipients
    //     compare against.  With no plan yet there is nothing to retire, and a
    //     plan an alteration already retired (rule 9) is not retired twice -
    //     the second invalidation would be dropped as stale anyway.
    if((my_epoch != 0) && (inval_epoch < my_epoch)) {
      sends.fwd_inval_epoch = my_epoch;
      sends.inval_to.insert(sends.inval_to.end(), cur_plan.kids.begin(),
                            cur_plan.kids.end());
    }

    // (3) install the owner's own record.  'inval_epoch = new_epoch - 1' is
    //     what BarrierArrive's Trigger does (:322-324): the owner is never sent its
    //     own invalidation, so it retires the old epoch itself, which leaves
    //     plan_retired_locked() false and lets it aggregate under the new plan
    //     immediately.
    cur_plan.quota = quota[0];
    cur_plan.inplan = true;
    cur_plan.parent = static_cast<NodeID>(-1);
    cur_plan.kids.clear();
    for(size_t k = 1; (k <= barrier_plan_radix()) && (k < tv.size()); k++) {
      cur_plan.kids.push_back(tv[k]);
    }
    my_epoch = new_epoch;
    inval_epoch = new_epoch - 1;

    // (4) DISTRIBUTION (section 11.4): each child is sent ITS OWN SUBTREE, not
    //     just an epoch number - its quota, its child list, and recursively
    //     everything below it - because it has to be able to forward its own
    //     children's records without asking anyone.
    if(tv.size() > 1) {
      sends.fwd_plan_epoch = new_epoch;
      for(size_t k = 1; (k <= barrier_plan_radix()) && (k < tv.size()); k++) {
        sends.plan_to.push_back(PendingPlan());
        PendingPlan &pp = sends.plan_to.back();
        pp.to = tv[k];
        encode_plan_subtree(k, tv, quota, barrier_plan_radix(), pp.payload);
      }
    }

    // (5) FLUSH EVERY GENERATION THAT STRADDLES THE SWITCH.
    //
    //     BarrierArrive's RecvInvalidate puts every untriggered generation the
    //     node HAS STATE ON into eager flush at every node it reaches
    //     (SubtreeKnown > 0, :452-456), and nothing takes it off again before
    //     the trigger (RecvNewPlan leaves 'flushing' UNCHANGED, :510-514).
    //     What that buys is the generations already IN FLIGHT when the tree
    //     changed:
    //     their counts were accumulated for the old shape, and the node the new
    //     shape gives them to may have no record of them, no arrival of its own
    //     yet, and a child-wait on a child that has already discharged that
    //     generation to its OLD parent and will never speak again.
    //
    //     The invalidation covers that at each node for the generations that
    //     node has a record for.  This covers it for the generations the OWNER
    //     has a record for - which is where a subtree total that was forwarded
    //     BEFORE the switch ends up - and it fans down the NEW child list, so
    //     it reaches nodes that were nowhere near the old tree.  Together with
    //     'saw_direct' (which makes a straddling report announce itself all the
    //     way here rather than dying at a below-quota relay) that is every
    //     generation any node can be sitting on.
    //
    //     This deliberately does NOT go through enter_flush_locked().  That
    //     function is idempotent per generation - which is what terminates the
    //     fan-out - so for a generation the owner was ALREADY flushing (the
    //     usual case, since a 'direct' for it is often what motivated this
    //     rebuild in the first place) it would do nothing at all, and the
    //     announcement it did send went to the child list that has just been
    //     replaced.  The whole point here is to re-announce down the NEW one.
    //     It also avoids enter_flush_locked's plan_rebuild_pending side effect:
    //     these flushes are the CONSEQUENCE of a rebuild, not a reason for
    //     another one.  The owner never reports, so there is nothing else in
    //     that function this needs.
    if(!cur_plan.kids.empty()) {
      const gen_t watermark = generation.load();
      for(std::map<gen_t, Generation *>::iterator it = generations.begin();
          it != generations.end(); ++it) {
        if(it->first <= watermark) {
          continue;
        }
        // ONLY GENERATIONS WITH STATE OR ALREADY IN FLUSH.  The spec's only
        //  plan-switch flush is per-node at RecvInvalidate, conditioned on
        //  SubtreeKnown > 0 (BarrierArrive :452-456); a no-activity generation is
        //  governed by the plan record alone (:448-451), so converting it to
        //  eager mode here forfeits its aggregation for nothing.  A record CAN
        //  exist with zero state - add_waiter creates one for any future
        //  generation a local task waits on - and it is skipped.
        //
        // A record that is ALREADY FLUSHING is re-fanned even with no counts
        //  yet: a count-free flush signal (retroactive case 3) can reach the
        //  owner before the counts it announces, and this loop is the ONLY
        //  re-announcement down the NEW child list - enter_flush_locked's
        //  idempotence means no later signal for this generation will ever
        //  fan again.
        if((it->second->subtree_known() <= 0) && !it->second->flushing) {
          continue;
        }
        if(!it->second->flushing) {
          it->second->flushing = true;
          counters.flush_episodes++;
        }
        sends.flushes.push_back(PendingFlush());
        PendingFlush &pf = sends.flushes.back();
        pf.gen = it->first;
        pf.to.insert(pf.to.end(), cur_plan.kids.begin(), cur_plan.kids.end());
      }
    }

    log_barrier.info() << "built barrier plan: " << me << " epoch=" << new_epoch
                       << " nodes=" << tv.size() << " arrivals=" << total
                       << " own_quota=" << cur_plan.quota
                       << " kids=" << cur_plan.kids.size();
  }

  // 'mutex' must NOT be held (S2).  Reads only its argument and immutable
  //  members (S3).
  void BarrierImpl::emit_pending_sends(const PendingSends &sends)
  {
    if(sends.report.to != static_cast<NodeID>(-1)) {
      emit_report(sends.report);
    }
    for(std::vector<PendingReport>::const_iterator it = sends.more_reports.begin();
        it != sends.more_reports.end(); ++it) {
      emit_report(*it);
    }

    for(std::vector<PendingFlush>::const_iterator fit = sends.flushes.begin();
        fit != sends.flushes.end(); ++fit) {
      for(std::vector<NodeID>::const_iterator it = fit->to.begin(); it != fit->to.end();
          ++it) {
        log_barrier.info() << "sending barrier flush: " << me << "/" << fit->gen
                           << " dest=" << *it;
        BarrierFlushMessage::send_request(*it, me.id, fit->gen);
      }
    }

    // rule 8 - the alteration and the bypassed arrival both go STRAIGHT TO THE
    //  OWNER.  'owner' is write-once, so reading it here honours S3.
    if(sends.has_alter) {
      log_barrier.info() << "sending barrier alter: " << me << "/" << sends.alter.gen
                         << " delta=" << sends.alter.delta << " ts=" << sends.alter.ts
                         << " prev_ts=" << sends.alter.prev_ts << " dest=" << owner;
      BarrierAlterMessage::send_request(owner, me.id, sends.alter.gen, sends.alter.delta,
                                        sends.alter.ts, sends.alter.prev_ts);
    }

    if(sends.has_ts_arrival) {
      log_barrier.info() << "sending barrier ts arrival: " << me << "/"
                         << sends.ts_arrival.gen << " val=" << sends.ts_arrival.val
                         << " ts=" << sends.ts_arrival.ts
                         << " local_ts=" << sends.ts_arrival.local_ts
                         << " dest=" << owner;
      BarrierTsArrivalMessage::send_request(
          owner, me.id, sends.ts_arrival.gen, sends.ts_arrival.val, sends.ts_arrival.ts,
          sends.ts_arrival.local_ts, sends.ts_arrival.poisoned);
    }

    if(sends.fwd_inval_epoch != 0) {
      for(std::vector<NodeID>::const_iterator it = sends.inval_to.begin();
          it != sends.inval_to.end(); ++it) {
        log_barrier.info() << "sending barrier invalidate: " << me
                           << " epoch=" << sends.fwd_inval_epoch << " dest=" << *it;
        BarrierInvalidateMessage::send_request(*it, me.id, sends.fwd_inval_epoch);
      }
    }

    if(sends.fwd_plan_epoch != 0) {
      for(std::vector<PendingPlan>::const_iterator it = sends.plan_to.begin();
          it != sends.plan_to.end(); ++it) {
        log_barrier.info() << "sending barrier newplan: " << me
                           << " epoch=" << sends.fwd_plan_epoch << " dest=" << it->to
                           << " bytes=" << it->payload.size();
        BarrierNewPlanMessage::send_request(it->to, me.id, sends.fwd_plan_epoch,
                                            it->payload.data(), it->payload.size());
      }
    }

    // NOTIFICATION_PROTOCOL action T - the multicast notification.  Every byte
    //  of it, INCLUDING THE TARGET SET, was materialised inside the critical
    //  section (S3): the target set is the pre-shrink snapshot, so nothing that
    //  has happened to 'sub_set' since can strand a departing node.
    if(sends.notify.valid) {
      log_barrier.info() << "sending barrier notify: " << me << "/" << sends.notify.prev
                         << " -> " << sends.notify.wm << " sv=" << sends.notify.set_ver
                         << " targets=" << sends.notify.targets.size()
                         << " pois=" << sends.notify.poison.size()
                         << " dep=" << sends.notify.departing.size();
      BarrierNotifyMessage::send_request(sends.notify.targets, me.id, sends.notify.wm,
                                         sends.notify.prev, sends.notify.gather_gen,
                                         sends.notify.set_ver, sends.notify.shrink_hint,
                                         sends.notify.poison, sends.notify.departing);
    }

    // NOTIFICATION_PROTOCOL rule 8 - the departure intent.  UNICAST to the
    //  owner, deliberately not aggregated up the multicast ack tree (that tree
    //  carries no payload today, and adding one for a single speculative user
    //  is out of scope).  'owner' is write-once, so reading it here honours S3.
    if(sends.has_depart) {
      log_barrier.info() << "sending barrier depart: " << me << " dest=" << owner;
      BarrierDepartMessage::send_request(owner, me.id, Network::my_node_id);
    }

    // NOTIFICATION_PROTOCOL rule 5 - the pull.  'owner' is write-once, so
    //  reading it here honours S3.
    if(sends.has_subscribe) {
      log_barrier.info() << "subscribing to barrier "
                         << make_barrier(sends.subscribe_need)
                         << " lk=" << sends.subscribe_lk << " dest=" << owner;
      BarrierSubscribeMessage::send_request(owner, me.id, sends.subscribe_need,
                                            sends.subscribe_lk, Network::my_node_id);
    }
  }

  void BarrierImpl::emit_report(const PendingReport &r)
  {
    log_barrier.info() << "sending barrier " << (r.is_direct ? "direct" : "report")
                       << ": " << me << "/" << r.gen << " val=" << r.val
                       << " dest=" << r.to << " map=" << r.flush_map.size()
                       << " poisoned=" << r.poisoned;
    BarrierReportMessage::send_request(r.to, me.id, r.gen, r.val, Network::my_node_id,
                                       r.is_direct, r.poisoned, r.flush_map.data(),
                                       r.flush_map.size());
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL - poison, the watermark, and the pull.
  //

  // LOCK FREE (decision Q3/D7), and identical to
  //  GenEventImpl::is_generation_poisoned.  The acquire load of the count
  //  synchronises with the release store in add_poison_locked, so every slot
  //  below the observed count was written before that store; slots are
  //  append-only and never rewritten, so a concurrent later append can only
  //  extend the array beyond where this scan stops.
  bool BarrierImpl::is_generation_poisoned(gen_t gen) const
  {
    const int npg_cached = num_poisoned_generations.load_acquire();
    if(REALM_LIKELY(npg_cached == 0)) {
      return false;
    }
    for(int i = 0; i < npg_cached; i++) {
      if(poisoned_generations[i] == gen) {
        return true;
      }
    }
    return false;
  }

  // Caller MUST hold 'mutex'.  Idempotent - a notification and a subscribe
  //  reply can both name the same generation, and rule 5's "merge, never
  //  substitute" makes duplicates the normal case rather than the exception.
  void BarrierImpl::add_poison_locked(gen_t gen)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(is_generation_poisoned(gen)) {
      return;
    }
    const int npg = num_poisoned_generations.load();
    if(npg >= POISONED_GENERATION_LIMIT) {
      // decision Q3: there is no representation that stays lock-free past the
      //  cap, and silently forgetting a poisoned generation would wake a waiter
      //  with the wrong poison status - one of the three failure modes this
      //  protocol exists to prevent.
      log_barrier.fatal() << "barrier poisoned more than " << POISONED_GENERATION_LIMIT
                          << " times: " << me << "/" << gen;
      std::abort();
    }
    if(poisoned_generations == nullptr) {
      // allocated at full size ONCE and never reallocated, so a lock-free
      //  reader never sees a dangling array
      poisoned_generations = new gen_t[POISONED_GENERATION_LIMIT];
    }
    // plain store - this slot is not published until the count is
    poisoned_generations[npg] = gen;
    num_poisoned_generations.store_release(npg + 1);
  }

  // Caller MUST hold 'mutex'.  tla/STATE_AND_LOCKING.md section 3.5: the poison
  //  slots are published BEFORE the watermark, so a lock-free reader that sees
  //  the new watermark necessarily sees the poison that goes with it.
  void BarrierImpl::publish_watermark_locked(gen_t wm, const gen_t *poison,
                                             size_t num_poisoned,
                                             EventWaiter::EventWaiterList &clean,
                                             EventWaiter::EventWaiterList &poisoned)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    for(size_t i = 0; i < num_poisoned; i++) {
      add_poison_locked(poison[i]);
    }
    generation.store_release(wm);

    // ARRIVAL_PROTOCOL section 8.8 - a non-owner reclaims its own alteration
    //  state here, for the same reason the owner reclaims its in
    //  check_triggers_locked: an arrival can only ever name a breakpoint above
    //  the watermark.
    prune_alter_state_locked(wm);

    while(!generations.empty()) {
      std::map<gen_t, Generation *>::iterator it = generations.begin();
      if(it->first > wm) {
        break;
      }
      if(is_generation_poisoned(it->first)) {
        poisoned.absorb_append(it->second->local_waiters);
      } else {
        clean.absorb_append(it->second->local_waiters);
      }
      delete it->second;
      generations.erase(it);
    }

    // external waiters are signalled INSIDE the outer section - that is what
    //  pairs with the hand-over-hand handoff in external_wait
    if(has_external_waiters) {
      has_external_waiters = false;
      AutoLock<KernelMutex> al2(external_waiter_mutex);
      external_waiter_condvar.broadcast();
    }
  }

  // the model's 'waiting[n] # {}'
  //
  // BarrierNotify evaluates rule 6 over the waiter set AFTER the watermark has
  //  moved - 'w2 == { g \in waiting[m.to] : g > nk }' (:146) - and 'waiting[n]'
  //  loses a generation only when 'known[n]' actually REACHES it.  Every test
  //  below is therefore against the CURRENT watermark, and every consultation
  //  has to leave a trace that survives one.
  //
  // 'gen_subscribed' is that trace, and it is the only one an external waiter
  //  or a bare Event::subscribe() leaves:
  //
  //   * external_wait / external_timedwait push nothing into any
  //     'local_waiters' list; they set 'has_external_waiters', which is a
  //     WAKE-UP LATCH that publish_watermark_locked CLEARS when it broadcasts.
  //     Both rule-6 sites run publish_watermark_locked earlier in the SAME
  //     critical section, and the woken thread cannot re-arm the latch until
  //     that section releases 'mutex' - so reading the latch there is a
  //     guaranteed false negative, and the node is dropped from the subscriber
  //     set holding a waiter that will never be told (NoStranded, :261).
  //   * subscribe() records nothing else at all: a node that subscribes and
  //     then polls has_triggered() - which is the documented use of
  //     Event::subscribe, and is the lock-free carve-out that deliberately
  //     records no consultation - would otherwise be invisible here.
  //
  // It is an OVER-approximation: remove_waiter does not lower it, so a node can
  //  stay subscribed longer than it strictly needs to.  That costs bandwidth
  //  and never correctness, which is the same asymmetry rule 3 rests on.
  bool BarrierImpl::has_waiters_locked(void) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(has_external_waiters) {
      return true;
    }
    const gen_t watermark = generation.load();
    if(gen_subscribed.load() > watermark) {
      return true;
    }
    for(std::map<gen_t, Generation *>::const_iterator it = generations.begin();
        it != generations.end(); ++it) {
      if(it->first <= watermark) {
        continue;
      }
      if(!it->second->local_waiters.empty()) {
        return true;
      }
    }
    return false;
  }

  BarrierImpl::gen_t BarrierImpl::subscribe_need_locked(void) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    const gen_t watermark = generation.load();
    const gen_t needed = gen_subscribed.load();
    return (needed > watermark) ? needed : (watermark + 1);
  }

  // NOTIFICATION_PROTOCOL action C step 4.  No rule-7 guard: the model's
  //  Consult (BarrierNotify:122-126) has none, because 'member' moving to
  //  PENDING is what suppresses the duplicate.  Guarding this on
  //  'pull_outstanding' instead would let a consultation for a higher
  //  generation be swallowed by an in-flight pull that does not cover it.
  void BarrierImpl::consult_subscribe_locked(PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // RULE 8 step 3 - THE CONSULTATION SIGNAL, recorded BEFORE any early out.
    //  A consultation that needs no message is still evidence that this node
    //  cares about the barrier, and the idle counter is the only thing keeping
    //  it in the owner's set.  This is also the whole of the signal for
    //  external_wait / external_timedwait, which reach it through subscribe().
    note_consultation_locked();

    if(owner == Network::my_node_id) {
      return;
    }
    if(member != MEMBER_NO) {
      return;
    }
    // rule 8 - coming back after having asked to leave is the churn signal
    note_rejoin_locked();
    member = MEMBER_PENDING;
    pull_outstanding = true;
    pull_deferred = false;
    sends.has_subscribe = true;
    sends.subscribe_lk = generation.load();
    sends.subscribe_need = subscribe_need_locked();
  }

  // NOTIFICATION_PROTOCOL action N step 5 - the RECOVERY pull, rule 7 gated.
  void BarrierImpl::record_pull_locked(PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(owner == Network::my_node_id) {
      return;
    }
    if(pull_outstanding) {
      // RULE 7 - at most one outstanding pull.  The in-flight subscribe carries
      //  an 'lk' at or below what a new one would, so its reply is a superset
      //  of what this pull would ask for... but only until the OWNER consumes
      //  it.  'pull_outstanding' stays set until the reply lands, which is
      //  longer than the model's window, so the request is remembered and
      //  re-issued there rather than dropped.
      pull_deferred = true;
      return;
    }
    pull_outstanding = true;
    pull_deferred = false;
    sends.has_subscribe = true;
    sends.subscribe_lk = generation.load();
    sends.subscribe_need = subscribe_need_locked();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL rule 8 - DEPARTURE HYSTERESIS.
  //
  // PERFORMANCE ONLY, and the one part of either protocol that is deliberately
  //  NOT MODELLED (NOTIFICATION_PROTOCOL section 8.2).  BarrierNotify lets
  //  'Depart' fire on any eligible node at any time and lets 'Trigger' apply
  //  ANY subset of the collected requests, which is strictly more general than
  //  any tuning - so no choice of K, J or shrink policy can be wrong, and
  //  nothing below can produce anything worse than wasted bandwidth.  What
  //  carries correctness is rule 6 (a node removed while holding a waiter
  //  re-subscribes at once, verified) and the pull path, which has to exist
  //  anyway.
  //

  // Action C step 3 - THE CONSULTATION SIGNAL.
  //
  // Idleness is measured as the WATERMARK DELTA since this node last consulted
  //  the barrier, NOT as a count of notifications received.  That distinction
  //  is the point: the owner coalesces a whole contiguous run of triggers into
  //  a single delta notification whenever it can (action T), so counting
  //  messages would make a busy barrier look idle exactly when it is busiest.
  //
  // Consulting is add_waiter, subscribe, and external_wait / external_timedwait
  //  (which reach this through subscribe).  It is emphatically NOT
  //  has_triggered(): that stays one lock-free acquire load with no member
  //  write of any kind, and putting this signal on it is the regression
  //  tla/STATE_AND_LOCKING.md section 4 spells out.  Every caller of this
  //  function is already holding 'mutex' for other reasons.
  void BarrierImpl::note_consultation_locked(void)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    last_consult_wm = generation.load();
    // whatever this node asked for while it believed itself idle, it is not
    //  idle now.  This is also what makes a DECLINED request eventually
    //  retryable without any acknowledgement from the owner.
    depart_outstanding = false;
  }

  // A node that had asked to leave wants back in.  This is the first metric
  //  IMPLEMENTATION_PLAN section 5 asks for ("leave->rejoin cycles per
  //  barrier; near zero is the expectation"), and it is the same signal the
  //  adaptation uses: a rejoin within DEPART_CHURN_WINDOW generations means K
  //  was too small for this barrier's access pattern, so K doubles.
  //
  // 'last_depart_wm' is the guard, so a node that has never asked to leave -
  //  including one subscribing for the very first time - is never counted.
  void BarrierImpl::note_rejoin_locked(void)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // 0 is an unambiguous "never asked": a departure cannot fire below
    //  watermark DEPART_K_INITIAL, because the eligibility threshold is at
    //  least K generations of watermark movement.
    if(last_depart_wm == 0) {
      return;
    }

    const gen_t wm = generation.load();
    counters.leave_rejoin_cycles++;
    if((wm - last_depart_wm) <= DEPART_CHURN_WINDOW) {
      counters.churn_backoffs++;
      if(depart_K < DEPART_K_MAX) {
        depart_K = ((depart_K * 2) < DEPART_K_MAX) ? (depart_K * 2) : DEPART_K_MAX;
      }
      log_barrier.info() << "barrier departure churn: " << me << " wm=" << wm
                         << " asked_to_leave_at=" << last_depart_wm
                         << " new_K=" << depart_K;
    }
    last_depart_wm = 0;
  }

  // Action N step 6 / action RP - the eligibility test.  It lives where the
  //  WATERMARK MOVES, because that is the unit the idle counter is measured in;
  //  it is not a timer and there is nothing to poll.
  //
  // Every guard here is an optimisation.  The "do not depart while holding a
  //  waiter" one in particular is the model's single NEGATIVE CONTROL: removing
  //  it was checked and found benign, so it is here to reduce churn and for no
  //  other reason.  It is NOT sufficient on its own either - a waiter can be
  //  registered after this fires and before the owner applies the shrink -
  //  which is exactly why rule 6's recovery is the correctness rule.
  void BarrierImpl::consider_departure_locked(PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(owner == Network::my_node_id) {
      // the owner is never in its own subscriber set
      return;
    }
    if(member != MEMBER_YES) {
      // not covered, or a pull is still in flight - either way there is nothing
      //  to give up, and asking to leave while PENDING would race the reply
      //  that is about to add us
      return;
    }
    if(pull_outstanding) {
      // A node that is asking the owner for the watermark is not idle, and
      //  action N step 5 can record a GAP pull without ever moving 'member' off
      //  MEMBER_YES - so without this guard one critical section could emit
      //  both a subscribe and a departure, and whichever the owner happened to
      //  process second would win.  Cleared by the reply, where the test is
      //  re-run, so nothing is lost by waiting.
      return;
    }
    if(depart_outstanding) {
      // one request per idle episode.  A DECLINED request is simply never
      //  retried until this node consults again, which is what keeps a machine
      //  full of idle nodes from re-asking every generation.
      return;
    }
    if(has_waiters_locked()) {
      return;
    }

    const gen_t wm = generation.load();
    // THE STAGGER, and it is expressed in GENERATIONS rather than in time, so
    //  the protocol stays feed-forward and needs no timer.  Phase changes
    //  retire many nodes at once, so this is load bearing rather than
    //  defensive: without it the owner takes an O(N) unicast burst inside a
    //  single generation.  J is sized for spike smoothing and not for
    //  asymptotics - the subscribe path is ALREADY O(N) unicast, so an O(N)
    //  unsubscribe burst is not a new complexity class.
    const gen_t threshold =
        depart_K + static_cast<gen_t>(Network::my_node_id % DEPART_STAGGER_J);
    if((wm - last_consult_wm) < threshold) {
      return;
    }

    if(!shrink_hint) {
      // rule 8's advisory byte.  The owner's cost test declined the last shrink
      //  it weighed, so this request would only be declined as well; suppressing
      //  it saves the round trip and keeps this node's single request for a
      //  moment when it might actually be granted.
      counters.departs_suppressed++;
      return;
    }

    depart_outstanding = true;
    last_depart_wm = wm;
    counters.departs_sent++;
    sends.has_depart = true;
  }

  namespace {
    // A nominal per-delivery cost, in bytes, for the cost test below.  Only its
    //  RATIO to a target encoding's size matters, and the whole comparison is a
    //  heuristic: NOTIFICATION_PROTOCOL section 8.4 is explicit that what is
    //  verified is that ANY shrink the owner chooses is safe, not that it
    //  chooses well.
    const size_t MULTICAST_ENVELOPE_COST_BYTES = 64;

    // What EncodedMulticastTargets::encode() would produce, WITHOUT producing
    //  it: encode() evaluates every representation that can express the set and
    //  keeps the smallest, and encoded_size() answers that per kind in
    //  O(num_ranges) with no allocation at all.  This runs inside 'mutex', so
    //  not allocating is worth the loop.
    size_t best_encoded_size(const MulticastTargetSet &targets, NodeID num_nodes)
    {
      size_t best = 0;
      for(size_t k = 0; k < MULTICAST_ENCODING_KINDS; k++) {
        const size_t sz = EncodedMulticastTargets::encoded_size(
            targets, num_nodes, static_cast<MulticastTargetEncoding>(k));
        // 0 means "this kind cannot represent this set"
        if((sz != 0) && ((best == 0) || (sz < best))) {
          best = sz;
        }
      }
      return best;
    }

    // The scalar rule 3's cost test compares:
    //
    //    cost(S) = |S| * (ENVELOPE + encoded_size(S))
    //
    // The two factors are the two quantities the rule names - "bytes and
    //  deliveries" - and the PRODUCT is not an accident.  The multicast layer
    //  plans a fresh forwarding tree from the encoded set on every send and
    //  every hop carries a target encoding, so the encoding is paid ONCE PER
    //  DELIVERY rather than once per multicast.  That is precisely why dropping
    //  scattered nodes from an ALL_NODES set can LOSE: it trades away a handful
    //  of deliveries and buys a per-hop bitmap paid |S| times over.
    size_t multicast_cost(const MulticastTargetSet &targets, NodeID num_nodes)
    {
      if(targets.empty()) {
        return 0;
      }
      return targets.size() *
             (MULTICAST_ENVELOPE_COST_BYTES + best_encoded_size(targets, num_nodes));
    }
  } // namespace

  // OWNER ONLY - RULE 3's COST TEST.  "Adds are mandatory, removals are
  //  discretionary": refusing an add strands a waiter, refusing a removal only
  //  costs bandwidth, and that asymmetry is the entire licence for this
  //  function to say no.  A declined removal is simply a publication in which
  //  the node still appears - no message, no state, no recovery needed.
  //
  // All-or-nothing on the requested set.  Rule 3 permits ANY subset R, so
  //  taking either the whole of 'departing' or none of it is a legal choice and
  //  is the simple one; searching for the best-paying subset is not justified
  //  by anything measured.
  bool BarrierImpl::shrink_pays_locked(const MulticastTargetSet &departing) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    if(departing.empty()) {
      return false;
    }

    MulticastTargetSet after = sub_set;
    for(MulticastTargetSet::const_iterator it = departing.begin(); it != departing.end();
        ++it) {
      after.remove(*it);
    }

    const NodeID num_nodes = Network::max_node_id + 1;
    return multicast_cost(after, num_nodes) < multicast_cost(sub_set, num_nodes);
  }

  // 'mutex' must NOT be held (S2)
  /*static*/ void BarrierImpl::deliver_waiters(EventWaiter::EventWaiterList &clean,
                                               EventWaiter::EventWaiterList &poisoned,
                                               TimeLimit work_until)
  {
    if(!clean.empty()) {
      get_runtime()->event_triggerer.trigger_event_waiters(clean, false, work_until);
    }
    if(!poisoned.empty()) {
      get_runtime()->event_triggerer.trigger_event_waiters(poisoned, true, work_until);
    }
  }

  // OWNER ONLY.  D4: on the scalable path every accumulator counts UP, so the
  //  test is "have we accounted for exactly the arrivals we expect", never the
  //  legacy path's count-down.  The owner's accepted total is child_sum - the
  //  model's ownerAcc[g] - which the report handler maintains by REPLACING each
  //  child's previous contribution.
  bool BarrierImpl::generation_complete_locked(gen_t gen, const Generation &g) const
  {
    REALM_BARRIER_ASSERT_LOCKED();
    // expected(gen) is base_arrival_count until somebody alters the barrier,
    //  so both tests below are exactly what they were before stage C6 for a
    //  barrier nobody alters.
    if(redop_id != 0) {
      // legacy/reduction path: the count-down convention stays here (D4)
      return (expected_locked(gen) + g.unguarded_delta) == 0;
    }
    // the model's ownerAcc[g] + localTotal[Owner][g] + TsTotal(g) = expected[g]
    return (g.local_total + g.child_sum + g.ts_acc) == expected_locked(gen);
  }

  // OWNER ONLY.  Drains the contiguous run of complete generations.  Nothing is
  //  sent and no waiter is run here (S2) - the caller emits after the unlock.
  EventImpl::gen_t BarrierImpl::check_triggers_locked(
      EventWaiter::EventWaiterList &local_notifications,
      EventWaiter::EventWaiterList &poisoned_notifications,
      std::vector<RemoteNotification> &remote_notifications, gen_t &oldest_previous,
      PendingSends &sends)
  {
    REALM_BARRIER_ASSERT_LOCKED();
    gen_t trigger_gen = 0;

    // BarrierNotify's 'Trigger' shares this section (action T).  'prev' has to
    //  be the watermark BEFORE the whole contiguous chain, so it is captured
    //  here rather than inside the drain: coalescing several triggers into one
    //  delta notification is explicitly supported.
    const gen_t old_watermark = generation.load();
    // the generations in (old_watermark, trigger_gen] that were poisoned, which
    //  is exactly what the notification will carry (rule 4: the poison WITHIN
    //  the announced range, so the message never grows with the barrier's
    //  poison history)
    std::vector<gen_t> newly_poisoned;

    // ARRIVAL_PROTOCOL section 11.2/11.3 - the aggregation structure, and it is
    //  a LOCAL.  It is merged out of each drained generation's flush maps at
    //  the last moment those maps exist, and it is destroyed when this function
    //  returns.  There is deliberately no member to forget (D9).
    std::map<NodeID, int64_t> agg;
    bool have_agg = false;
    // the arrivals the gathered generation took THROUGH THE TREE, which is what
    //  a complete set of maps has to add up to (section 11.2).  Bypassed
    //  arrivals (rule 8.1) never touch a 'local_total' and so are in no map.
    int64_t agg_tree_arrivals = 0;

    std::map<gen_t, Generation *>::iterator it = generations.begin();
    while((it != generations.end()) && (it->first == (generation.load() + 1)) &&
          generation_complete_locked(it->first, *(it->second))) {
      // Gather from a generation the owner was in FLUSH mode for - those are
      //  the only ones whose reports carried per-node maps.  The highest one
      //  that yields anything wins OUTRIGHT rather than being merged in:
      //  counts are per generation, so combining two generations' maps would
      //  describe an arrival pattern that never happened.
      // 'flush_mode_locked' and not the bare 'flushing' flag: an alteration
      //  retires the plan (rule 9) rather than setting a per-generation flag,
      //  and a retired plan is exactly the state in which every node - and so
      //  every report reaching us - is eager and carrying a map.
      if((redop_id == 0) && plan_rebuild_pending &&
         flush_mode_locked(it->first, *(it->second))) {
        std::map<NodeID, int64_t> merged;
        merge_counts_locked(*(it->second), merged);
        if(!merged.empty()) {
          agg.swap(merged);
          have_agg = true;
          // this generation completed, so its accounting is exact:
          //  local_total + child_sum + ts_acc == expected(gen).  Everything but
          //  'ts_acc' came up the tree and is therefore describable by the maps.
          agg_tree_arrivals = expected_locked(it->first) - it->second->ts_acc;
        }
      }

      // decision Q4 - the generation is poisoned if any arrival on it had a
      //  poisoned precondition.  PUBLISHED BEFORE THE WATERMARK
      //  (tla/STATE_AND_LOCKING.md section 3.5), so a lock-free has_triggered()
      //  that sees the new watermark necessarily sees the poison with it.
      if(it->second->poisoned) {
        add_poison_locked(it->first);
        newly_poisoned.push_back(it->first);
        poisoned_notifications.absorb_append(it->second->local_waiters);
      } else {
        // keep the list of local waiters to wake up once we release the lock
        local_notifications.absorb_append(it->second->local_waiters);
      }
      trigger_gen = it->first;
      generation.store_release(it->first);

      // AND HERE THE GATHERED MAPS DIE, with the generation record that owns
      //  them (section 11.3)
      delete it->second;
      generations.erase(it);
      it = generations.begin();
    }

    if(trigger_gen == 0) {
      return 0;
    }

    // ARRIVAL_PROTOCOL section 8.8 - the alteration state a triggered
    //  generation put beyond reach is reclaimed here, while the watermark that
    //  makes it unreachable has just moved.
    prune_alter_state_locked(trigger_gen);

    // RULE 5 - the plan switch, in the SAME critical section as the trigger
    //  that motivated it (section 12: "count comparison and the plan switch
    //  that follows it").  At most one switch per section, which keeps this a
    //  faithful refinement of BarrierArrive's one-switch-per-Trigger.
    if(have_agg) {
      // consumed whether or not the build is accepted: retrying against the
      //  same evidence would reach the same answer, and the next deviation
      //  sets the flag again
      plan_rebuild_pending = false;
      build_new_plan_locked(agg, agg_tree_arrivals, sends);
    }

    if(redop_id != 0) {
      // LEGACY / REDUCTION PATH: one message per subscriber, sliced out of
      //  'final_values' by that node's own 'previous_gen', which is why the
      //  owner has to keep the O(N) per-node map here.  Left exactly as it was.
      std::map<unsigned, gen_t>::iterator it2 = remote_subscribe_gens.begin();
      while(it2 != remote_subscribe_gens.end()) {
        RemoteNotification rn;
        rn.node = it2->first;
        if(it2->second <= trigger_gen) {
          // we have fulfilled the entire subscription
          rn.trigger_gen = it2->second;
          std::map<unsigned, gen_t>::iterator to_nuke = it2++;
          remote_subscribe_gens.erase(to_nuke);
        } else {
          // subscription remains valid
          rn.trigger_gen = trigger_gen;
          ++it2;
        }
        // also figure out what the previous generation this node knew about was
        LegacyReductionState &ls = legacy_state();
        std::map<unsigned, gen_t>::iterator it3 = ls.remote_trigger_gens.find(rn.node);
        if(it3 != ls.remote_trigger_gens.end()) {
          rn.previous_gen = it3->second;
          it3->second = rn.trigger_gen;
        } else {
          rn.previous_gen = first_generation;
          ls.remote_trigger_gens[rn.node] = rn.trigger_gen;
        }

        if(remote_notifications.empty() || (rn.previous_gen < oldest_previous)) {
          oldest_previous = rn.previous_gen;
        }

        remote_notifications.push_back(rn);
      }
    } else {
      // ---- NOTIFICATION_PROTOCOL action T, the notification half -----------
      //
      // Step 1: choose the shrink.  RULE 3 - removals are DISCRETIONARY, so ANY
      //  subset of the collected intents is a legal choice and the model checks
      //  exactly that ("the owner may shrink by ANY subset").
      MulticastTargetSet departing;
      for(MulticastTargetSet::const_iterator wit = want_out.begin();
          wit != want_out.end(); ++wit) {
        // only a node that is actually IN the set is being removed; anything
        //  else in 'want_out' is a stale intent and changes no version
        if(sub_set.contains(*wit)) {
          departing.append_increasing_node(*wit);
        }
      }

      // Step 1b: RULE 3's COST TEST, and rule 8's advisory byte.  A shrink is
      //  applied only if it actually reduces what a notification costs; if it
      //  does not, 'departing' is emptied and the removal is DECLINED, which
      //  needs no protocol at all - it is just the next publication with the
      //  node still in it.
      //
      // The verdict is published as the hint so the rest of the machine stops
      //  asking for something that is not going to be granted, and it is
      //  STICKY: a trigger with nothing to weigh leaves it alone, because a
      //  verdict does not go stale until the set it was about changes shape.
      //  An add is the thing that changes shape, so action S is what puts the
      //  hint back to optimistic.  Resetting it here instead would make the
      //  suppression last exactly one notification and buy nothing.
      if(!departing.empty()) {
        if(shrink_pays_locked(departing)) {
          shrink_pays = true;
          counters.shrinks_applied++;
          counters.nodes_removed += departing.size();
        } else {
          shrink_pays = false;
          counters.shrinks_declined++;
          log_barrier.info() << "declining barrier subscriber shrink: " << me
                             << " set=" << sub_set.size()
                             << " departing=" << departing.size();
          departing.clear();
        }
      }

      // Step 2: SNAPSHOT THE PRE-SHRINK SET.  RULE 1 - "any shrink must be
      //  published to the PRE-SHRINK set", so a node being removed always hears
      //  about its own removal.  This is the one thing in this protocol that
      //  message reordering cannot repair, which is why it is handled by
      //  snapshotting inside the section rather than by ordering (S5).
      sends.notify.targets = sub_set;

      // Step 3: apply the shrink and bump the version.  RULE 2 - 'set_ver'
      //  moves on EVERY change to the set and is stamped on every publication;
      //  when nothing changes it does not move, every recipient's version gate
      //  says "stale", and no membership is applied at all.
      if(!departing.empty()) {
        for(MulticastTargetSet::const_iterator dit = departing.begin();
            dit != departing.end(); ++dit) {
          sub_set.remove(*dit);
        }
        set_ver += 1;
      }
      want_out.clear();

      // Step 4: compose the delta.  There is nobody to tell if the set is
      //  empty - and a node that is not in it pulls, which is the correctness
      //  guarantee; the set is only a cache.
      if(!sends.notify.targets.empty()) {
        sends.notify.valid = true;
        sends.notify.wm = trigger_gen;
        sends.notify.prev = old_watermark;
        // section 11.5: if the rebuild declined and declared the next
        //  generation a gathering generation, the declaration RIDES THIS
        //  NOTIFICATION - the receiver marks it eager in the same critical
        //  section that wakes its waiters, which is the only ordering that
        //  beats the wake->arrive race (P_STEP measured the flush-message
        //  version losing on nearly every generation).
        {
          std::map<gen_t, Generation *>::iterator git = generations.find(trigger_gen + 1);
          sends.notify.gather_gen = ((git != generations.end()) && git->second->flushing)
                                        ? (trigger_gen + 1)
                                        : 0;
        }
        sends.notify.set_ver = set_ver;
        // rule 8's one-byte hint rides the notification that was going out
        //  anyway, so it costs nothing on the wire
        sends.notify.shrink_hint = shrink_pays;
        sends.notify.poison.swap(newly_poisoned);
        if(!departing.empty()) {
          // only carried when 'set_ver' moved: with an unchanged version no
          //  recipient would apply membership, so the bytes would be waste
          EncodedMulticastTargets enc =
              EncodedMulticastTargets::encode(departing, Network::max_node_id + 1);
          const unsigned char *encp = static_cast<const unsigned char *>(enc.data());
          sends.notify.departing.assign(encp, encp + enc.bytes());
        }
      }
    }

    // external waiters need to be signalled inside the lock
    if(has_external_waiters) {
      has_external_waiters = false;
      // also need external waiter mutex
      AutoLock<KernelMutex> al2(external_waiter_mutex);
      external_waiter_condvar.broadcast();
    }

    return trigger_gen;
  }

  // 'mutex' must NOT be held (S2).  Everything this needs was materialised
  //  inside the critical section; the only members it touches are immutable
  //  ones (S3).
  void BarrierImpl::emit_trigger_notifications(
      gen_t trigger_gen, const std::vector<RemoteNotification> &remote_notifications,
      gen_t oldest_previous, const void *final_values_copy, size_t sizeof_lhs)
  {
    for(std::vector<RemoteNotification>::const_iterator it = remote_notifications.begin();
        it != remote_notifications.end(); it++) {
      // send each remote waiter data up to the generation they asked for
      gen_t tgt_trigger_gen = (*it).trigger_gen;
      log_barrier.info() << "sending remote trigger notification: " << me << "/"
                         << (*it).previous_gen << " -> " << tgt_trigger_gen
                         << ", dest=" << (*it).node;
      const void *data = 0;
      size_t datalen = 0;
      if(final_values_copy) {
        data = static_cast<const char *>(final_values_copy) +
               (((*it).previous_gen - oldest_previous) * sizeof_lhs);
        datalen = (tgt_trigger_gen - (*it).previous_gen) * sizeof_lhs;
      }
      BarrierTriggerMessage::send_request((*it).node, me.id, tgt_trigger_gen,
                                          (*it).previous_gen, first_generation, redop_id,
                                          data, datalen);
    }
  }

  ////////////////////////////////////////////////////////////////////////

  // used to adjust a barrier's arrival count either up or down
  // if delta > 0, timestamp is current time (on requesting node)
  // if delta < 0, timestamp says which positive adjustment this arrival must wait for
  void BarrierImpl::adjust_arrival(gen_t barrier_gen, int delta,
                                   Barrier::timestamp_t timestamp, Event wait_on,
                                   NodeID sender, const void *reduce_value,
                                   size_t reduce_value_size, TimeLimit work_until,
                                   bool poisoned /*= false*/)
  {
    Barrier b = make_barrier(barrier_gen, timestamp);

    if(!wait_on.has_triggered()) {
      // deferred arrival

      // only forward deferred arrivals if the precondition is not one that looks like
      // it'll
      //  trigger here first
      //
      // ARRIVAL_PROTOCOL rule 8: an arrival that has to be GATED may not be
      //  relocated.  Forwarding it makes it an arrival AT THE OWNER, which
      //  knows neither the handle's causal timestamp nor this node's arrival
      //  floor, so the alteration it depends on would stop holding it back -
      //  and an arrival counted before its alteration lands is a barrier that
      //  triggers too early.  The floor is read here, at the moment the
      //  application issued the arrival, which is the moment that decides
      //  whether it witnessed an alteration.
      if(owner != Network::my_node_id) {
        bool must_gate = (timestamp != 0);
        if(!must_gate) {
          AutoLock<> a(mutex);
          must_gate = (ts_floor_locked(barrier_gen) != 0);
        }
        if(!must_gate) {
          ID wait_id(wait_on);
          int wait_node;
          if(wait_id.is_event()) {
            wait_node = wait_id.event_creator_node();
          } else {
            wait_node = wait_id.barrier_creator_node();
          }
          if(wait_node != (int)Network::my_node_id) {
            // let deferral happen on owner node (saves latency if wait_on event
            //   gets triggered there)
            log_barrier.info() << "forwarding deferred barrier arrival: delta=" << delta
                               << " in=" << wait_on << " out=" << b << " (" << timestamp
                               << ")";
            BarrierAdjustMessage::send_request(owner, b, delta, wait_on, sender,
                                               reduce_value, reduce_value_size);
            return;
          }
        }
      }

      log_barrier.info() << "deferring barrier arrival: delta=" << delta
                         << " in=" << wait_on << " out=" << b << " (" << timestamp << ")";
      EventImpl::add_waiter(
          wait_on,
          new DeferredBarrierArrival(b, delta, sender, reduce_value, reduce_value_size));
      return;
    }

    log_barrier.info() << "barrier adjustment: event=" << b << " delta=" << delta
                       << " ts=" << timestamp;

#ifdef DEBUG_BARRIER_REDUCTIONS
    if(reduce_value_size) {
      char buffer[129];
      for(size_t i = 0; (i < reduce_value_size) && (i < 64); i++) {
        snprintf(buffer + 2 * i, sizeof buffer - 2 * i, "%02x",
                 ((const unsigned char *)reduce_value)[i]);
      }
      log_barrier.info("barrier reduction: event=" IDFMT "/%d size=%zd data=%s", me.id(),
                       barrier_gen, reduce_value_size, buffer);
    }
#endif

    // can't actually trigger while holding the lock, so remember which generation(s),
    //  if any, to trigger and do it at the end
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    std::vector<RemoteNotification> remote_notifications;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    size_t sizeof_lhs = 0;
    NodeID forward_to_node = (NodeID)-1;
    // reports and flush announcements are RECORDED under the lock and emitted
    //  after it (S2); the emit phase reads only these locals (S3)
    PendingSends sends;

    do { // so we can use 'break' from the middle
      AutoLock<> a(mutex);

      // ROUTE DECISION (tla/STATE_AND_LOCKING.md D2).  A non-owner cannot know
      //  whether this is a reduction barrier - it learns 'redop_id' only from a
      //  trigger message that carries reduction data - so it takes the scalable
      //  path only when everything about this adjustment says that is safe.
      //  Reduction barriers and alterations (delta > 0, which only ever reaches
      //  here from the legacy path now that alter_arrival_count has its own
      //  entry point) stay on the legacy eager path.  A TIMESTAMPED ARRIVAL no
      //  longer disqualifies the scalable path: rule 8.1 gives it its own
      //  bypass of the tree instead.
      const bool scalable_route =
          (redop_id == 0) && (reduce_value_size == 0) && (delta < 0);

      // ARRIVAL_PROTOCOL rule 8.1 - THE GATE THIS ARRIVAL CARRIES.  Two
      //  independent dependencies, either or both of which may be absent:
      //  the handle's own causal timestamp, and this node's arrival floor
      //  (the model's myTs[n][g], set by every alteration this node has issued
      //  for this generation or an earlier one).  A non-zero gate means the
      //  arrival BYPASSES THE TREE - a relay would collapse it into a single
      //  integer and erase the timestamps with it.
      Generation::TsKey gate;
      if(scalable_route) {
        gate.ts = timestamp;
        gate.local_ts = ts_floor_locked(barrier_gen);
      }
      const bool bypass = ((gate.ts != 0) || (gate.local_ts != 0));

      // barriers are owned by their creator node for their whole life, but a
      //  non-owner still has to hand the arrival on
      if(owner != Network::my_node_id) {
        if(!scalable_route) {
          forward_to_node = owner;
          break;
        }

        // ---- ARRIVAL_PROTOCOL action A, on a non-owner --------------------
        // Guard evaluation, the state update and the send DECISION are all in
        //  this one critical section (section 12).
        // Arriving on a generation that has already triggered is an
        //  APPLICATION error (action A step 1).  That is a different thing from
        //  a stale MESSAGE, which action R drops silently.
        assert(barrier_gen > generation.load());
        Generation *g = get_generation_locked(barrier_gen);

        // decision Q4 - a POISONED PRECONDITION poisons the generation.  The
        //  arrival itself still counts: refusing to count it would hang the
        //  barrier rather than poison it, and every waiter would then be told
        //  nothing at all instead of being told the truth.  The bit is sticky
        //  and travels with the count on whichever of the two paths below this
        //  arrival takes.
        if(poisoned) {
          g->poisoned = true;
        }

        if(bypass) {
          // RULE 8.1 - straight to the owner, carrying this node's CUMULATIVE
          //  count of arrivals issued for this generation under this gate (D4:
          //  replace-if-higher, exactly like every other accumulator here).
          //  'local_total' is deliberately NOT touched - a bypassed arrival
          //  never enters the tree's accounting.
          int64_t &issued = g->ts_issued[gate];
          issued += -static_cast<int64_t>(delta);
          sends.has_ts_arrival = true;
          sends.ts_arrival.gen = barrier_gen;
          sends.ts_arrival.val = issued;
          sends.ts_arrival.ts = gate.ts;
          sends.ts_arrival.local_ts = gate.local_ts;
          sends.ts_arrival.poisoned = g->poisoned;

          // RULE 8.3, generalised to the arrival rather than the alteration:
          //  an arrival that bypasses the tree never reaches 'local_total', so
          //  this node's quota just became that much less reachable.  A relay
          //  waiting on an unreachable quota goes silent and strands its
          //  children, so it stops aggregating for this generation and reports
          //  what it is already holding.  (Over-signalling a flush is always
          //  safe - it only ever causes MORE eager reporting.)
          enter_flush_locked(*g, barrier_gen, sends);
          break;
        }

        g->local_total += -static_cast<int64_t>(delta);

        // the '!flushing' test here is load-bearing, not just idempotence: once
        //  this node IS flushing, every arrival has to fall through to the
        //  reporting branch below instead of being absorbed by the (idempotent,
        //  and therefore silent) flush entry.
        if(!g->flushing && cur_plan.inplan &&
           (g->local_total > static_cast<int64_t>(cur_plan.quota))) {
          // RULE 2 - OVER-ARRIVAL.  A node the plan DID predict has just taken
          //  more arrivals than its quota, so the plan is wrong: the
          //  completion condition this node would otherwise wait for can never
          //  be met, and waiting is exactly the silence that deadlocks the
          //  generation.  It enters eager-flush mode, reports its cumulative
          //  total upward IMMEDIATELY and fans a flush announcement out to its
          //  children - they are aggregating under the same wrong plan.  From
          //  then on every arrival for this generation reports immediately,
          //  which falls out of the 'flushing' test in should_report_locked().
          enter_flush_locked(*g, barrier_gen, sends);
        } else if(should_report_locked(barrier_gen, *g)) {
          // already flushing, rule 3 (no plan record - report DIRECT to the
          //  owner), or rule 1's steady state (quota met and every predicted
          //  child has spoken)
          record_report_locked(*g, barrier_gen, sends);
        }
        // otherwise: SILENCE.  Below quota, or missing a child, this node says
        //  nothing at all - that is the aggregation that makes this scalable.
        break;
      }

      // sanity checks - is this a valid barrier?
      // assert(generation < free_generation);
      assert(base_arrival_count > 0);

      // update whatever generation we're told to.  NOTE: the owner DOES know
      //  whether this is a reduction barrier, so its accumulator choice is
      //  'redop_id == 0' and not the (deliberately conservative) route test
      //  above - otherwise a timestamped arrival on a scalable barrier would
      //  land in the legacy count-down and be lost.
      {
        assert(barrier_gen > generation.load());
        Generation *g = get_generation_locked(barrier_gen);

        // decision Q4, at the owner.  The scalable path publishes this when the
        //  generation triggers; the legacy path has no poison plumbing at all,
        //  so the bit is simply carried and never read there.
        if(poisoned && (redop_id == 0)) {
          g->poisoned = true;
        }

        if(delta > 0) {
          // An ALTERATION that came in through the arrival path.  Nothing
          //  generates this on the scalable path any more - alter_arrival_count
          //  has its own entry point and its own message - but the legacy
          //  forward of a positive delta still lands here, and applying it
          //  persistently (rules 8 and 9) is the whole point of stage C6.
          //
          // rule 8.1, as in handle_remote_alter: an altering node never belongs
          //  in a plan again.  Leaving a node out is always safe, so this errs
          //  towards excluding one that might not have needed it.
          if(redop_id == 0) {
            ts_bypass_nodes.insert(sender);
          }
          apply_alter_locked(barrier_gen, delta, timestamp, /*prev_ts=*/0, sends);
        } else if(bypass) {
          // RULE 8.1 at the owner: the arrival still bypasses the tree, but
          //  the "message" is applied inline rather than sent to ourselves.
          //  Action A step 4 and action TS, fused into one section (S1).
          int64_t &issued = g->ts_issued[gate];
          issued += -static_cast<int64_t>(delta);
          apply_ts_arrival_locked(Network::my_node_id, *g, gate, issued);
          enter_flush_locked(*g, barrier_gen, sends);
        } else if((redop_id != 0) && (timestamp != 0)) {
          // A LEGACY arrival that names an alteration.  Reduction barriers stay
          //  on the eager path, but the ordering question is the same one rule
          //  8.2 answers, so they get the same EXACT-SET gate rather than the
          //  "ts <= the highest timestamp from that node" comparison that used
          //  to live here.  The value is accumulated rather than replaced
          //  because a legacy adjustment message carries an INCREMENT - only
          //  the scalable path's own message is cumulative.
          Generation::TsKey lgate;
          lgate.ts = timestamp;
          Generation::TsStreamKey lkey;
          lkey.from = sender;
          lkey.gate = lgate;
          const int64_t sofar = g->ts_streams[lkey].seen;
          apply_ts_arrival_locked(sender, *g, lgate,
                                  sofar + -static_cast<int64_t>(delta));
        } else if(redop_id != 0) {
          // legacy/reduction path: arrivals count DOWN (D4)
          g->unguarded_delta += delta;
        } else {
          g->local_total += -static_cast<int64_t>(delta);
        }

        // The owner arriving on its own barrier is still action A, and the
        //  owner is a node in the tree like any other.  The model writes the
        //  two deviation cases as a message the owner sends to ITSELF - a
        //  'direct' when it is outside the plan (rule 3), a flush fan when it
        //  is in the plan and over quota (rule 2) - and its RecvReport step 5
        //  then puts the owner into flush mode.  We apply that inline rather
        //  than round-tripping a message to ourselves, and deliberately do NOT
        //  route the arrival through 'child_acc' the way the model's literal
        //  self-message does: it is already in 'local_total', and BarrierArrive's
        //  Trigger guard adds localTotal[Owner] to ownerAcc separately, so
        //  doing both would double-count it.
        //
        // Either way the owner has just learned its plan does not describe
        //  reality, and the flush fan-out is how the rest of the tree finds
        //  out.  The owner itself never reports (it has no parent), so
        //  enter_flush_locked() records only the announcements - and it is
        //  idempotent, so it does nothing at all after the first one.
        if((redop_id == 0) && (!cur_plan.inplan ||
                               (g->local_total > static_cast<int64_t>(cur_plan.quota)))) {
          enter_flush_locked(*g, barrier_gen, sends);
        }
      }

      // this may have completed one or more generations - and, with them, a
      //  plan rebuild and the notification, both recorded into 'sends' like
      //  everything else
      trigger_gen = check_triggers_locked(local_notifications, poisoned_notifications,
                                          remote_notifications, oldest_previous, sends);

      // do we have reduction data to apply?  we can do this even if the actual
      // adjustment is
      //  being held - no need to have lots of reduce values lying around
      if(reduce_value_size > 0) {
        assert(redop != 0);
        assert(redop->sizeof_rhs == reduce_value_size);

        // do we have space for this reduction result yet?
        int rel_gen = barrier_gen - first_generation;
        assert(rel_gen > 0);

        if(value_capacity < static_cast<size_t>(rel_gen)) {
          size_t new_capacity = rel_gen;
          size_t old_capacity = value_capacity;
          final_values.resize(new_capacity * redop->sizeof_lhs);
          for(size_t i = old_capacity; i < new_capacity; ++i) {
            std::memcpy(&final_values[i * redop->sizeof_lhs], initial_value.get(),
                        redop->sizeof_lhs);
          }

          value_capacity = new_capacity;
        }

        (redop->cpu_apply_excl_fn)(final_values.data() +
                                       ((rel_gen - 1) * redop->sizeof_lhs),
                                   0, reduce_value, 0, 1, redop->userdata);
      }

      // do this AFTER we actually update the reduction value above :)
      // if any remote notifications are going to occur and we have reduction values,
      // make a copy so
      //  we have something stable after we let go of the lock
      if(trigger_gen && redop) {
        int rel_gen = oldest_previous + 1 - first_generation;
        assert(rel_gen > 0);
        int count = trigger_gen - oldest_previous;
        final_values_copy =
            bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                    count * redop->sizeof_lhs);
        sizeof_lhs = redop->sizeof_lhs;
      }
    } while(0);

    if(forward_to_node != (NodeID)-1) {
      // decision Q8 - straight to the active message interface, no communicator
      //  indirection
      Barrier b = make_barrier(barrier_gen, timestamp);
      BarrierAdjustMessage::send_request(forward_to_node, b, delta, Event::NO_EVENT,
                                         sender, reduce_value, reduce_value_size);
      return;
    }

    // everything recorded inside the critical section goes out now (S2)
    emit_pending_sends(sends);

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;

      // notify local waiters first
      deliver_waiters(local_notifications, poisoned_notifications, work_until);

      emit_trigger_notifications(trigger_gen, remote_notifications, oldest_previous,
                                 final_values_copy, sizeof_lhs);
    }

    // free our copy of the final values, if we had one
    if(final_values_copy) {
      free(final_values_copy);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action R - RecvReport / DropStale (BarrierArrive RecvReport).
  //  One critical section covers the staleness test, the accumulator update,
  //  the recomputed subtree total and the forwarding decision (section 12).
  //
  void BarrierImpl::handle_remote_report(NodeID from, gen_t report_gen, int64_t val,
                                         bool is_direct, bool poisoned, const void *data,
                                         size_t datalen, TimeLimit work_until)
  {
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    std::vector<RemoteNotification> remote_notifications;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    size_t sizeof_lhs = 0;
    PendingSends sends;

    // ARRIVAL_PROTOCOL section 11.1 - decode the eager-flush (node,count) map
    //  BEFORE taking the lock.  It touches nothing but the message buffer, and
    //  a payload that is O(subtree) has no business lengthening a critical
    //  section (section 12).  This is also the copy out of the active-message
    //  buffer, which is dead the moment this handler returns.
    std::map<NodeID, int64_t> counts;
    bool have_counts = false;
    if(datalen > 0) {
      have_counts = decode_flush_map(data, datalen, counts);
      if(!have_counts) {
        log_barrier.fatal() << "malformed barrier flush map: " << me << "/" << report_gen
                            << " from=" << from << " bytes=" << datalen;
        abort();
      }
    }

    do {
      AutoLock<> a(mutex);

      // step 1: a report for a generation that has already triggered is dropped
      //  SILENTLY.  This is exactly what makes freeing a triggered generation's
      //  record safe: without it the absent child_acc entry would make a late
      //  report look like an increase and it would be counted, and forwarded,
      //  all over again.
      if(report_gen <= generation.load()) {
        log_barrier.info() << "dropping barrier report for triggered generation: " << me
                           << "/" << report_gen << " from=" << from
                           << " direct=" << is_direct;
        break;
      }

      Generation *g = get_generation_locked(report_gen);

      // decision Q4 - the poison bit is OR-ed in BEFORE the staleness test.
      //  It is monotone, so it cannot go backwards and a duplicate costs
      //  nothing; doing it after would let the sticky bit be thrown away with a
      //  report whose COUNT was stale but whose poison was news.
      if(poisoned) {
        g->poisoned = true;
      }

      // step 3 - RULE 7.  Cumulative totals only ever increase: a report whose
      //  value does not STRICTLY exceed what we have stored for this sender is
      //  stale and is discarded.  Never treat a report as an increment -
      //  accepting stale reports lets the accepted count go DOWN, which is a
      //  mutation TLC catches.
      //
      // NOTE: the accumulator is keyed on the SENDER and is never checked
      //  against cur_plan.kids.  A receiver MUST accept a report from a node it
      //  does not list as a child (ARRIVAL_PROTOCOL section 8.2): a node whose
      //  parent has already switched plans and dropped it still reports to that
      //  old parent, and the defensive check that looks worth adding here
      //  silently loses those arrivals.
      std::map<NodeID, Generation::ChildReport>::iterator it = g->child_acc.find(from);
      const int64_t prev = (it == g->child_acc.end()) ? 0 : it->second.total;
      if(val <= prev) {
        log_barrier.info() << "dropping stale barrier report: " << me << "/" << report_gen
                           << " from=" << from << " val=" << val << " have=" << prev;
        break;
      }
      if(it == g->child_acc.end()) {
        it = g->child_acc.insert(std::make_pair(from, Generation::ChildReport())).first;
      }
      it->second.total = val;

      // step 4, second half (section 11.1): the (node,count) map is cumulative
      //  in exactly the same way as the count - a relay re-sends its WHOLE
      //  subtree map, so a later report supersedes an earlier one and
      //  reordering stays harmless.  Replace it WHOLESALE when one is carried;
      //  leave the stored map alone when one is not, because "no map" is what a
      //  cheap steady-state report looks like and is not a claim that the
      //  subtree is empty.
      if(have_counts) {
        it->second.counts.swap(counts);
      }

      // rule 3, propagated (see record_report_locked): 'direct' means "your
      //  plan did not account for me", and a relay has to carry that flag on
      //  because the node that raised it still reports through the retired tree
      //  rather than straight to the owner.
      if(is_direct) {
        g->saw_direct = true;
      }

      // D2 fallback: the sender could not know this was a reduction barrier (it
      //  learns redop_id only from a trigger message carrying data), so it
      //  reported cumulatively.  The report is exact, so the increment it
      //  represents goes into the legacy count-down accumulator that path uses,
      //  and 'child_sum' - which only the scalable test reads - is left alone.
      const bool legacy_fold = (owner == Network::my_node_id) && (redop_id != 0);
      if(legacy_fold) {
        g->unguarded_delta -= static_cast<int>(val - prev);
      } else {
        // REPLACE this child's contribution, never add to it.  'child_sum' is
        //  the incremental maintenance of the model's
        //  ownerAcc' = @ - childAcc[Owner][from][gen] + val.
        g->child_sum += val - prev;
      }

      if(owner == Network::my_node_id) {
        // step 5 - RULE 3.  A 'direct' is a node telling the owner "your plan
        //  did not predict me": the sender holds no plan record, so it has no
        //  parent and no completion condition it could ever wait for, and it
        //  reported straight here.  The owner treats that as proof its plan is
        //  wrong, enters eager-flush mode FOR THIS GENERATION and fans a flush
        //  out through its children, so that every node still aggregating
        //  under the stale plan stops waiting and speaks.
        //
        // This runs BEFORE the trigger check below, because a trigger frees
        //  '*g'.  Fanning a flush for a generation that then triggers in the
        //  same section is harmless - the recipients' own reports come back
        //  for an already-triggered generation and action R step 1 drops them.
        if(is_direct && (redop_id == 0)) {
          enter_flush_locked(*g, report_gen, sends);
        }

        trigger_gen = check_triggers_locked(local_notifications, poisoned_notifications,
                                            remote_notifications, oldest_previous, sends);
        if(trigger_gen && redop) {
          int rel_gen = oldest_previous + 1 - first_generation;
          assert(rel_gen > 0);
          int count = trigger_gen - oldest_previous;
          final_values_copy =
              bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                      count * redop->sizeof_lhs);
          sizeof_lhs = redop->sizeof_lhs;
        }
      } else {
        // step 6, first half - RULE 10.5, A STALE EDGE (BarrierArrive RecvReport
        //  :293-304, 'm.from \notin KidsOf(m.to)').  The sender pinned this
        //  edge for this generation and a plan change has since moved it out
        //  of our child list.  We are still the only route its contribution
        //  has, so it must be passed on AT ONCE - holding it behind our own
        //  quota strands it, because nothing will ever make that quota
        //  relevant again.  Caught by MCDouble, MCStrand and MCStrand2
        //  (deadlock); ARRIVAL_PROTOCOL rule 10.5 records that it evaded five
        //  scenarios and looked deletable, and is load-bearing.
        //
        // The SENDER-side half of this (record_report_locked flagging
        //  'is_direct' when its pinned target differs from its current parent)
        //  is not enough on its own: it fires only once the sender has SEEN
        //  the newer plan, and the receiver always switches first - the
        //  receiver is the one that forwards the sender's invalidation - so a
        //  report already in flight across that window arrives unflagged.
        //  Only the receiver can see that its own child list has moved on,
        //  which is why the spec puts this disjunct at RecvReport.
        const bool stale_edge =
            g->holding() && (std::find(cur_plan.kids.begin(), cur_plan.kids.end(),
                                       from) == cur_plan.kids.end());
        // step 6, second half: accepting this report may be exactly what
        //  completes this relay
        if(stale_edge) {
          counters.stale_edge_forwards++;
        }
        if(stale_edge || should_report_locked(report_gen, *g)) {
          record_report_locked(*g, report_gen, sends);
        }
      }
    } while(0);

    // everything recorded inside the critical section goes out now (S2)
    emit_pending_sends(sends);

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;

      deliver_waiters(local_notifications, poisoned_notifications, work_until);

      emit_trigger_notifications(trigger_gen, remote_notifications, oldest_previous,
                                 final_values_copy, sizeof_lhs);
    }

    if(final_values_copy) {
      free(final_values_copy);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action F - RecvFlush (BarrierArrive RecvFlush).  Rule 4:
  //  flush is PER GENERATION and IDEMPOTENT.  One critical section covers the
  //  idempotence test, the flag, the re-fan and the report of held work
  //  (section 12).
  //
  void BarrierImpl::handle_remote_flush(gen_t flush_gen)
  {
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // A flush for a generation that has already triggered HERE has nothing
      //  left to collect: this node's watermark only moves when the owner has
      //  accounted for every arrival in that generation.  Dropping it also
      //  keeps the generation record's lifetime simple - the record is created
      //  once and freed exactly once, when the generation triggers - whereas
      //  creating one below the watermark would leave a record nothing ever
      //  frees.  Action R step 1 drops stale reports for the same reason.
      if(flush_gen <= generation.load()) {
        log_barrier.info() << "dropping barrier flush for triggered generation: " << me
                           << "/" << flush_gen;
      } else {
        // find OR CREATE.  The record has to exist even when no arrival has
        //  happened here yet: without it a later arrival for this generation
        //  would not know it is in flush mode and would go back to
        //  aggregating, which is the silence the announcement came to break.
        Generation *g = get_generation_locked(flush_gen);
        enter_flush_locked(*g, flush_gen, sends);
      }
    }

    // recorded under the lock, emitted now (S2).  Nothing is triggered here -
    //  a flush moves no counts, so it can never complete a generation.
    emit_pending_sends(sends);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action I - RecvInvalidate (BarrierArrive RecvInvalidate,
  //  rule 6).  ONE critical section, and THE ORDER INSIDE IT IS LOAD-BEARING:
  //  two of the three steps below have their own mutation in the battery, and
  //  TLC catches both.
  //
  void BarrierImpl::handle_remote_invalidate(uint32_t epoch)
  {
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // step 1 - EPOCH NUMBERING.  An invalidation that is not newer than what
      //  this node has already retired is stale: it is consumed and ignored.
      //  Nothing is forwarded, which is what terminates the fan-out.
      if(inval_epoch >= epoch) {
        log_barrier.info() << "dropping stale barrier invalidate: " << me
                           << " epoch=" << epoch << " have=" << inval_epoch;
      } else {
        // step 2 - FORWARD FIRST, through the CURRENT (old) child list, before
        //  anything below can replace it.  Dropping the child list first
        //  strands the entire subtree underneath - a caught mutation.  The
        //  targets are resolved into 'sends' here, inside the section, so S3
        //  makes this survive step 5 for free.
        sends.fwd_inval_epoch = epoch;
        sends.inval_to.insert(sends.inval_to.end(), cur_plan.kids.begin(),
                              cur_plan.kids.end());

        // step 3 - FLUSH EVERY OPEN GENERATION, not just the one that caused
        //  the switch.  A node may be holding arrivals for generations beyond
        //  it (a node running ahead of the owner), and every one of them was
        //  being aggregated for a tree that is going away.  Flushing only the
        //  switch generation is a caught mutation.
        //
        // Generations this node has no record for yet are covered by
        //  plan_retired_locked() rather than by a flag - see
        //  should_report_locked().
        //
        // 'saw_direct' goes on with it, so that everything reported out of here
        //  is flagged.  Each of these generations STRADDLES the switch: its
        //  counts were accumulated for a tree that is going away, and the node
        //  the new tree gives them to may have no record of them at all and no
        //  reason of its own to speak.  The flag is what carries that fact to
        //  the OWNER, which flushes the generation across the NEW tree.
        //
        // ONLY GENERATIONS THIS NODE HAS STATE ON (BarrierArrive RecvInvalidate
        //  :452-456, 'IF ~triggered[g] /\ SubtreeKnown(m.to, g) > 0').
        //  Generations with no activity are governed by the plan record alone
        //  (:448-451) - planless nodes are outsiders, planned nodes start
        //  planned - so nothing here is sticky across future generations.  A
        //  record CAN exist with zero state: add_waiter creates one for any
        //  future generation a local task waits on, and flagging it would
        //  convert a perfectly predicted generation to eager mode until it
        //  triggers - every arrival reporting an O(subtree) map, and the owner
        //  rebuilding an identical plan under a new epoch.
        const gen_t watermark = generation.load();
        for(std::map<gen_t, Generation *>::iterator it = generations.begin();
            it != generations.end(); ++it) {
          if((it->first > watermark) && (it->second->subtree_known() > 0)) {
            it->second->flushing = true;
            it->second->saw_direct = true;
          }
        }
        report_held_locked(sends);

        // step 4
        inval_epoch = epoch;

        // step 5 - and only NOW apply any parked plan.  This is the other half
        //  of rule 5's deferral: the node did its invalidation work first, so
        //  the retiring tree was fully served before the new one is adopted.
        //
        // THE LIVE GUARD (BarrierArrive RecvInvalidate 'live', a caught mutation on
        //  MCStrand2): a parked plan whose epoch does not exceed THIS
        //  invalidation's is being delivered by its own death notice.
        //  Installing it would return this node to planned mode under a dead
        //  plan, having just consumed the only invalidation that will ever
        //  name it; forwarding it would do the same to every descendant, who
        //  additionally never see that invalidation at all - the one node
        //  positioned to route it down the dead plan's edges spent it in this
        //  very action.  A dead parked plan is DISCARDED: not installed, not
        //  forwarded.  plan_retired_locked() then keeps this node eager and
        //  its reports flagged 'direct' until a genuinely newer plan arrives.
        if(defer_epoch > epoch) {
          counters.parked_plans_applied++;
          apply_plan_locked(defer_epoch, deferred_plan_payload.data(),
                            deferred_plan_payload.size(), defer_parent, sends);
        } else {
          if(defer_epoch != 0) {
            counters.dead_plans_discarded++;
            log_barrier.info() << "discarding dead parked barrier plan: " << me
                               << " parked=" << defer_epoch << " retired-by=" << epoch;
          }
          // RETROACTIVE CASE 3 (BarrierArrive RecvInvalidate :418-437, rule 10.4).
          //  This node has been invalidated with NO live replacement.  A node
          //  that ran ahead arrived believing itself a plan member, so the
          //  outsider rule never fired: its count reached the owner, but a
          //  count is not a signal, and nothing tells the owner the plan
          //  mis-predicts this generation.  On learning it is planless, the
          //  node delivers case 3 late - a COUNT-FREE flush signal to the
          //  owner for each open generation it has arrivals on.  The owner
          //  fans the flush down the CURRENT tree, which is what unsticks
          //  contributions parked behind quotas the retired plan
          //  over-predicted.  Caught by MCStale (deadlock).
          //
          // report_held_locked above is NOT a substitute: it only speaks when
          //  this node is HOLDING (Unreported > 0), and the run-ahead node
          //  that already reported its full count holds nothing - the
          //  'saw_direct' flag set in step 3 has no report left to ride on.
          //  This is the spec's condition exactly: localTotal > 0, independent
          //  of whether anything is unreported.  The signal is count-free (a
          //  flush, idempotent at the owner); the count itself stays on the
          //  pinned edge, because a cumulative value sent off-pin
          //  double-counts.
          if(owner != Network::my_node_id) {
            for(std::map<gen_t, Generation *>::iterator it = generations.begin();
                it != generations.end(); ++it) {
              if((it->first <= watermark) || (it->second->local_total <= 0)) {
                continue;
              }
              counters.retro_flushes_sent++;
              sends.flushes.push_back(PendingFlush());
              PendingFlush &pf = sends.flushes.back();
              pf.gen = it->first;
              pf.to.push_back(owner);
            }
          }
        }
        if(defer_epoch != 0) {
          defer_epoch = 0;
          defer_parent = -1;
          // BoundedRetention: give the bytes back, do not just clear()
          std::vector<unsigned char>().swap(deferred_plan_payload);
        }
      }
    }

    // recorded under the lock, emitted now (S2).  An invalidation moves no
    //  counts, so nothing can have triggered and no waiter can be woken here.
    emit_pending_sends(sends);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action P - RecvNewPlan (BarrierArrive RecvNewPlan, rule 5).
  //
  // The owner sends the invalidation down the tree being retired and the new
  //  plan down the new tree at the same moment, and the two RACE.  The
  //  resolution is entirely local and needs no global knowledge:
  //
  //   A node that receives a new plan while it is still an UN-INVALIDATED
  //   MEMBER of the retiring plan PARKS it.
  //
  // The condition is the node's OWN membership plus 'my invalidation has not
  //  arrived yet', and it is self-clearing: once the invalidation has passed,
  //  the node looks like it was never in the old plan and the new plan installs
  //  immediately.  Disabling the deferral is the first mutation in the battery
  //  and TLC reports ReachableWhileHolding violated in about 46 seconds.
  //
  void BarrierImpl::handle_remote_new_plan(NodeID parent, uint32_t epoch,
                                           const void *data, size_t datalen)
  {
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // step 1 - EPOCH NUMBERING.  A plan that is not newer than the one this
      //  node holds is stale: consumed and ignored.
      //
      // THE INSTALL GUARD (BarrierArrive RecvNewPlan, a caught mutation on
      //  MCStrand2): 'newer than what I hold' is NOT enough.  Messages
      //  reorder, so a newplan can arrive after both of its own broadcast's
      //  invalidations have overtaken it - and a plan whose RETIREMENT this
      //  node has already witnessed (inval_epoch >= epoch) is dead on
      //  arrival.  Installing it would re-enter planned mode under a dead
      //  plan with no future signal coming, and forwarding it would strand
      //  every descendant the same way.  One principle, two doors - see the
      //  live guard in handle_remote_invalidate.
      if((my_epoch >= epoch) || (inval_epoch >= epoch)) {
        if((my_epoch < epoch) && (inval_epoch >= epoch)) {
          // genuinely DEAD (install guard), not merely stale: this plan would
          //  have installed but its retirement was already witnessed here
          counters.dead_plans_discarded++;
        }
        log_barrier.info() << "dropping dead barrier newplan: " << me
                           << " epoch=" << epoch << " have=" << my_epoch
                           << " inval=" << inval_epoch;
      } else if(cur_plan.inplan && (inval_epoch < my_epoch)) {
        counters.plans_parked++;
        // step 2 - PARK IT.  At most ONE parked plan per node, so a second one
        //  simply replaces the first; that is safe because epochs are monotone
        //  and the newer plan is the one that will be current when the
        //  invalidation finally lands (BarrierArrive BoundedRetention).
        //
        // The payload is COPIED: the active-message buffer is dead once this
        //  handler returns, and the node has to be able to forward each child's
        //  slice of it later.
        log_barrier.info() << "parking barrier newplan: " << me << " epoch=" << epoch
                           << " (holding epoch " << my_epoch << ", inval " << inval_epoch
                           << ")";
        defer_epoch = epoch;
        defer_parent = parent;
        deferred_plan_payload.clear();
        if(datalen > 0) {
          const unsigned char *p = static_cast<const unsigned char *>(data);
          deferred_plan_payload.assign(p, p + datalen);
        }
      } else {
        // step 3 - install it, and pass it on
        apply_plan_locked(epoch, data, datalen, parent, sends);
      }
    }

    // recorded under the lock, emitted now (S2).  Like the invalidation, a plan
    //  carries no counts, so nothing can trigger here.
    emit_pending_sends(sends);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action AL - alter_arrival_count, rules 8 and 9.
  //
  // ONE critical section covers the whole action (section 12): minting the
  //  causal timestamp, installing the arrival floor that makes this node's
  //  subsequent arrivals bypass the tree, entering eager flush for every
  //  affected open generation, and recording the message.  The timestamp used
  //  to be minted by Barrier::alter_arrival_count and consumed much later, in a
  //  different section - one spec action split in two, which is a defect even
  //  though both halves were locked.
  //
  // THE RESERVED-ARRIVAL CONTRACT (event.h:290-292) is what makes this safe and
  //  the runtime cannot check it: before altering, the application must still
  //  hold at least one unissued arrival from the pre-alteration count.  That
  //  arrival is what keeps the generation open until the alteration lands.  The
  //  runtime's half of the bargain is the arrival floor below - once this node
  //  has altered, EVERY arrival it issues for an affected generation is gated,
  //  including one made on a pre-alteration handle, which is exactly the
  //  reserved arrival.
  //
  Barrier::timestamp_t BarrierImpl::alter_arrival_count(gen_t barrier_gen, int delta)
  {
    Barrier::timestamp_t ts = 0;
    PendingSends sends;

    // only reachable when this node is the owner and applies the alteration
    //  inline, but the trigger bookkeeping is the same either way
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    std::vector<RemoteNotification> remote_notifications;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    size_t sizeof_lhs = 0;

    {
      AutoLock<> a(mutex);

      // step 1 - MINT THE TIMESTAMP, inside the section.  The counter is seeded
      //  with (my_node_id << 48) + 1, so a timestamp is globally unique, never
      //  zero, and monotone per node.
      ts = barrier_adjustment_timestamp.fetch_add(1);

      // step 2 - the chain.  'prev_ts' names the alteration this one follows at
      //  this node, and the owner will not apply this one before that one.  It
      //  is what lets a single timestamp on an arrival stand for every
      //  alteration this node issued before it.
      const Barrier::timestamp_t prev_ts = last_alter_ts;
      last_alter_ts = ts;

      // step 3 - THE ARRIVAL FLOOR (the model's myTs[n][g]).  Alterations are
      //  persistent, so this entry covers 'barrier_gen' and every generation
      //  after it; entries above it named earlier alterations, all of which are
      //  behind this one in the chain, so this timestamp supersedes them and
      //  they go.
      ts_floor.erase(ts_floor.upper_bound(barrier_gen), ts_floor.end());
      ts_floor[barrier_gen] = ts;

      // step 4 - RULE 8.3, eager flush for every affected open generation
      enter_flush_from_locked(barrier_gen, sends);

      // step 5 - to the owner, or applied right here if we are the owner
      //  (action RA fused into this section, which S1 permits: a sequence of
      //  whole actions, never a fraction of one)
      if(owner == Network::my_node_id) {
        // rule 8.1 - from here on this node's arrivals bypass the tree, so it
        //  must never be given a quota in a plan again (build_new_plan_locked)
        if(redop_id == 0) {
          ts_bypass_nodes.insert(Network::my_node_id);
        }
        apply_alter_locked(barrier_gen, delta, ts, prev_ts, sends);
        trigger_gen = check_triggers_locked(local_notifications, poisoned_notifications,
                                            remote_notifications, oldest_previous, sends);
        if(trigger_gen && redop) {
          int rel_gen = oldest_previous + 1 - first_generation;
          assert(rel_gen > 0);
          int count = trigger_gen - oldest_previous;
          final_values_copy =
              bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                      count * redop->sizeof_lhs);
          sizeof_lhs = redop->sizeof_lhs;
        }
      } else {
        sends.has_alter = true;
        sends.alter.gen = barrier_gen;
        sends.alter.delta = delta;
        sends.alter.ts = ts;
        sends.alter.prev_ts = prev_ts;
      }

      log_barrier.info() << "barrier alteration: " << make_barrier(barrier_gen, ts)
                         << " delta=" << delta;
    }

    emit_pending_sends(sends);

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;
      deliver_waiters(local_notifications, poisoned_notifications,
                      TimeLimit::responsive());
      emit_trigger_notifications(trigger_gen, remote_notifications, oldest_previous,
                                 final_values_copy, sizeof_lhs);
    }

    if(final_values_copy) {
      free(final_values_copy);
    }

    return ts;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action RA - an alteration reaches the owner (rules 8, 9).
  //
  void BarrierImpl::handle_remote_alter(NodeID from, gen_t alter_gen, int delta,
                                        Barrier::timestamp_t ts,
                                        Barrier::timestamp_t prev_ts,
                                        TimeLimit work_until)
  {
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    std::vector<RemoteNotification> remote_notifications;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    size_t sizeof_lhs = 0;
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // barriers are owned by their creator for their whole life (migration is
      //  gone), so an alteration can only ever arrive here
      if(owner != Network::my_node_id) {
        log_barrier.fatal() << "barrier alteration delivered to a non-owner: " << me
                            << "/" << alter_gen << " from=" << from;
        abort();
      }

      // RULE 8.1 - the sender's arrivals BYPASS THE TREE from 'alter_gen' on,
      //  and an alteration is persistent, so that is true of every generation
      //  after it too.  Its 'local_total' will never move again, so any quota a
      //  future plan gave it would be unreachable and its parent would wait on
      //  rule 1's child-wait forever.  The owner is the only node that learns
      //  this, and 'applied_ts' is keyed by timestamp and so cannot answer
      //  "which nodes" - hence the set.  (Not modelled: MCAlter deliberately
      //  makes its altering node a relay whose parent is the Owner, which has
      //  no child-wait.)
      if(redop_id == 0) {
        ts_bypass_nodes.insert(from);
      }

      apply_alter_locked(alter_gen, delta, ts, prev_ts, sends);

      // applying it may have raised the count this generation waits for, or
      //  lowered it, or released a pile of arrivals that were parked behind it
      trigger_gen = check_triggers_locked(local_notifications, poisoned_notifications,
                                          remote_notifications, oldest_previous, sends);
      if(trigger_gen && redop) {
        int rel_gen = oldest_previous + 1 - first_generation;
        assert(rel_gen > 0);
        int count = trigger_gen - oldest_previous;
        final_values_copy =
            bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                    count * redop->sizeof_lhs);
        sizeof_lhs = redop->sizeof_lhs;
      }
    }

    emit_pending_sends(sends);

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;
      deliver_waiters(local_notifications, poisoned_notifications, work_until);
      emit_trigger_notifications(trigger_gen, remote_notifications, oldest_previous,
                                 final_values_copy, sizeof_lhs);
    }

    if(final_values_copy) {
      free(final_values_copy);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // ARRIVAL_PROTOCOL action TS - a bypassed arrival reaches the owner (rule
  //  8.1).  It is a report like any other: cumulative, replace-if-higher, and
  //  dropped outright once the generation has triggered - that last test is
  //  what makes freeing a triggered generation's record safe.
  //
  void BarrierImpl::handle_remote_ts_arrival(NodeID from, gen_t arrival_gen, int64_t val,
                                             Barrier::timestamp_t ts,
                                             Barrier::timestamp_t local_ts, bool poisoned,
                                             TimeLimit work_until)
  {
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    std::vector<RemoteNotification> remote_notifications;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    size_t sizeof_lhs = 0;
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      if(owner != Network::my_node_id) {
        log_barrier.fatal() << "barrier ts arrival delivered to a non-owner: " << me
                            << "/" << arrival_gen << " from=" << from;
        abort();
      }

      if(arrival_gen <= generation.load()) {
        log_barrier.info() << "dropping barrier ts arrival for triggered generation: "
                           << me << "/" << arrival_gen << " from=" << from;
      } else {
        Generation *g = get_generation_locked(arrival_gen);
        // decision Q4 - as on the report path, OR-ed in before anything can
        //  discard the message as stale
        if(poisoned) {
          g->poisoned = true;
        }
        Generation::TsKey gate;
        gate.ts = ts;
        gate.local_ts = local_ts;
        apply_ts_arrival_locked(from, *g, gate, val);

        trigger_gen = check_triggers_locked(local_notifications, poisoned_notifications,
                                            remote_notifications, oldest_previous, sends);
        if(trigger_gen && redop) {
          int rel_gen = oldest_previous + 1 - first_generation;
          assert(rel_gen > 0);
          int count = trigger_gen - oldest_previous;
          final_values_copy =
              bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                      count * redop->sizeof_lhs);
          sizeof_lhs = redop->sizeof_lhs;
        }
      }
    }

    emit_pending_sends(sends);

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;
      deliver_waiters(local_notifications, poisoned_notifications, work_until);
      emit_trigger_notifications(trigger_gen, remote_notifications, oldest_previous,
                                 final_values_copy, sizeof_lhs);
    }

    if(final_values_copy) {
      free(final_values_copy);
    }
  }

  bool BarrierImpl::has_triggered(gen_t needed_gen, bool &poisoned)
  {
    // TIER 0 ONLY.  No lock, no allocation, no message, no member write.  This
    //  is the ONE carve-out from ARRIVAL_PROTOCOL section 12
    //  (tla/STATE_AND_LOCKING.md section 4), and it is why the consultation
    //  signal of NOTIFICATION_PROTOCOL rule 8 is explicitly NOT recorded here.
    //
    // The acquire load below synchronises with the release store of the
    //  watermark, and the poison slots were published BEFORE that store
    //  (section 3.5), so anything that could have poisoned a generation at or
    //  below 'wm' is already visible to is_generation_poisoned().
    if(needed_gen <= generation.load_acquire()) {
      poisoned = is_generation_poisoned(needed_gen);
      return true;
    }

    poisoned = false;
    return false;
  }

  // NOTIFICATION_PROTOCOL action C, the explicit-subscribe entry point.  One
  //  critical section; the send is recorded in it and emitted after (S2).
  void BarrierImpl::subscribe(gen_t subscribe_gen)
  {
    PendingSends sends;
    {
      AutoLock<> a(mutex);
      if(subscribe_gen > gen_subscribed.load()) {
        gen_subscribed.store(subscribe_gen);
      }
      // step 4: a node not covered by the owner's published set pulls.  It is
      //  'member' that decides this, NOT a per-generation watermark: membership
      //  on the scalable path is persistent, so a node already in the set needs
      //  nothing at all for a later generation.
      consult_subscribe_locked(sends);
    }

    emit_pending_sends(sends);
  }

  void BarrierImpl::external_wait(gen_t gen_needed, bool &poisoned)
  {
    poisoned = POISON_FIXME;

    // early out for now without taking lock (TODO: fix for poisoning)
    if(gen_needed <= generation.load_acquire()) {
      return;
    }

    // make sure we're subscribed to a (potentially-remote) trigger
    this->subscribe(gen_needed);

    {
      AutoLock<> a(mutex);

      // wait until the generation has advanced far enough
      while(gen_needed > generation.load()) {
        has_external_waiters = true;
        // must wait on external_waiter_condvar with external_waiter_mutex
        //  but NOT with base mutex - hand-over-hand lock on the way in,
        //  and then release external_waiter mutex before retaking main
        //  mutex
        external_waiter_mutex.lock();
        mutex.unlock();
        external_waiter_condvar.wait();
        external_waiter_mutex.unlock();
        mutex.lock();
      }
    }
  }

  bool BarrierImpl::external_timedwait(gen_t gen_needed, bool &poisoned, long long max_ns)
  {
    poisoned = POISON_FIXME;

    // early out for now without taking lock (TODO: fix for poisoning)
    if(gen_needed <= generation.load_acquire()) {
      return true;
    }

    // make sure we're subscribed to a (potentially-remote) trigger
    this->subscribe(gen_needed);

    long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
    {
      AutoLock<> a(mutex);

      // wait until the generation has advanced far enough
      while(gen_needed > generation.load()) {
        long long now = Clock::current_time_in_nanoseconds();
        if(now >= deadline) {
          return false; // trigger has not occurred
        }
        has_external_waiters = true;
        // we don't actually care what timedwait returns - we'll recheck
        //  the generation ourselves
        // must wait on external_waiter_condvar with external_waiter_mutex
        //  but NOT with base mutex - hand-over-hand lock on the way in,
        //  and then release external_waiter mutex before retaking main
        //  mutex
        external_waiter_mutex.lock();
        mutex.unlock();
        external_waiter_condvar.timedwait(deadline - now);
        external_waiter_mutex.unlock();
        mutex.lock();
      }
    }
    return true;
  }

  // NOTIFICATION_PROTOCOL action C.  This is a CONSULTATION - one of the three
  //  the departure hysteresis of rule 8 will count (the others are 'subscribe'
  //  and 'external_wait'), and explicitly not 'has_triggered'.
  bool BarrierImpl::add_waiter(gen_t needed_gen,
                               EventWaiter *waiter /*, bool pre_subscribed = false*/)
  {
    bool trigger_now = false;
    bool trigger_poisoned = false;
    PendingSends sends;
    {
      AutoLock<> a(mutex);

      // RULE 8 step 3 - THE CONSULTATION SIGNAL, and it is recorded on BOTH
      //  branches below.  Action C records it even when the generation has
      //  already triggered ("skip to 3"): a node repeatedly asking about
      //  generations it has already seen is still using this barrier, and
      //  retiring it would only cost it a pull the next time it waits.
      note_consultation_locked();

      if(needed_gen > generation.load()) {
        Generation *g;
        std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
        if(it != generations.end()) {
          g = it->second;
        } else {
          g = new Generation;
          generations[needed_gen] = g;
          log_barrier.info() << "added tracker for barrier " << make_barrier(needed_gen);
        }
        g->local_waiters.push_back(waiter);

        if(needed_gen > gen_subscribed.load()) {
          gen_subscribed.store(needed_gen);
        }
        // step 4: pull if this node is not covered.  A node already in the
        //  owner's published set is covered for EVERY generation, so needing a
        //  higher one is not on its own a reason to speak.
        consult_subscribe_locked(sends);
      } else {
        // needed generation has already occurred - trigger this waiter once we let go
        // of lock.  Its poison status is whatever we know, which
        // PoisonAccurate says is exactly the truth up to our own watermark.
        trigger_now = true;
        trigger_poisoned = is_generation_poisoned(needed_gen);
      }
    }

    emit_pending_sends(sends);

    if(trigger_now) {
      // the CALLER's budget, never a manufactured one
      //  (tla/STATE_AND_LOCKING.md section 5 R2)
      waiter->event_triggered(trigger_poisoned, TimeLimit::responsive());
    }

    return true;
  }

  bool BarrierImpl::remove_waiter(gen_t needed_gen, EventWaiter *waiter)
  {
    AutoLock<> a(mutex);

    if(needed_gen <= generation.load()) {
      // already triggered, so nothing to remove
      return false;
    }

    // find the right generation - this should not fail
    std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
    assert(it != generations.end());
    bool ok = it->second->local_waiters.erase(waiter);
    assert(ok);
    return true;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL action S - a pull reaches the owner (rules 3 and 5).
  //  One critical section covers the set mutation and the version stamp,
  //  because rule 2 requires the 'set_ver' a recipient gates on to be the one
  //  that describes the set it is gating against.
  //
  void BarrierImpl::handle_remote_subscription(NodeID subscriber,
                                               EventImpl::gen_t subscribe_gen,
                                               EventImpl::gen_t last_known,
                                               const void *data, size_t datalen)
  {
    // take the lock and add the subscribing node - notice if they need to be notified
    // for
    //  any generations that have already triggered
    EventImpl::gen_t trigger_gen = 0;
    EventImpl::gen_t previous_gen = 0;
    void *final_values_copy = 0;
    size_t final_values_size = 0;
    NodeID forward_to_node = (NodeID)-1;
    // the scalable reply (rule 5).  Composed here, sent after the unlock (S2).
    bool send_reply = false;
    EventImpl::gen_t reply_wm = 0;
    uint64_t reply_sv = 0;
    std::vector<gen_t> reply_poison;

    do {
      AutoLock<> a(mutex);

      EventImpl::gen_t active_generation = generation.load();

      // first check - are we even the current owner?
      if(owner != Network::my_node_id) {
        forward_to_node = owner;
        break;
      }

      if(redop_id == 0) {
        // ---- THE SCALABLE PATH ------------------------------------------
        //
        // RULE 3 - THE ADD IS MANDATORY.  Refusing it strands a waiter;
        //  refusing a REMOVAL only costs bandwidth, which is the asymmetry the
        //  whole discretionary-shrink cost test rests on.  A subscribe also
        //  retracts any departure intent this node had collected - the model's
        //  'wantOut' = 'wantOut \ {m.from}'.
        const bool was_in = sub_set.contains(subscriber);
        sub_set.add(subscriber);
        if(!was_in) {
          // RULE 2 - the version moves on EVERY change to the set
          set_ver += 1;
          // rule 8 - the set just changed SHAPE, so whatever the cost test
          //  last concluded about shrinking it no longer describes anything.
          //  This is the only thing that clears a declined verdict, which is
          //  what makes the hint sticky enough to be worth publishing.
          shrink_pays = true;
        }
        want_out.remove(subscriber);

        // RULE 5 - the reply is a DELTA keyed on what the subscriber said it
        //  already knew.  A notification cannot do this: it goes to nodes with
        //  different watermarks, and keying it per recipient is precisely the
        //  per-node map this design deleted.
        //
        // THE REPLY MUST CARRY THE WATERMARK, and it is sent even when nothing
        //  has triggered.  That is what covers the trigger-during-subscribe
        //  race, and it is what closes the subscriber's rule-7 pull window.
        // IMPLEMENTATION_PLAN section 5 - SUBSCRIBE FAN-IN.  This is the last
        //  unaggregated O(N) path in the design, so it is the number that says
        //  whether the escape hatch is ever needed.
        counters.subscribe_fan_in++;

        send_reply = true;
        reply_wm = active_generation;
        reply_sv = set_ver;
        const int npg = num_poisoned_generations.load();
        for(int i = 0; i < npg; i++) {
          const gen_t pg = poisoned_generations[i];
          if((pg > last_known) && (pg <= reply_wm)) {
            reply_poison.push_back(pg);
          }
        }
        break;
      }

      // ---- THE LEGACY / REDUCTION PATH, unchanged ------------------------
      // One subscribe message type, two reply shapes, forked HERE at the owner
      //  - which is exactly why a remote node never has to know in advance
      //  whether this is a reduction barrier (D2).

      // make sure the subscription is for this "lifetime" of the barrier
      assert(subscribe_gen > first_generation);

      bool already_subscribed = false;
      {
        std::map<unsigned, EventImpl::gen_t>::iterator it =
            remote_subscribe_gens.find(subscriber);
        if(it != remote_subscribe_gens.end()) {
          // a valid subscription should always be for a generation that hasn't
          //  triggered yet
          assert(it->second > active_generation);
          if(it->second >= subscribe_gen) {
            already_subscribed = true;
          } else {
            it->second = subscribe_gen;
          }
        } else {
          // new subscription - the node may have been subscribed in the past
          // NOTE: remote_subscribe_gens should only hold subscriptions for
          //  generations that haven't triggered, so if we're subscribing to
          //  an old generation, don't add it
          if(subscribe_gen > active_generation) {
            remote_subscribe_gens[subscriber] = subscribe_gen;
          }
        }
      }

      // as long as we're not already subscribed to this generation, check to see if
      //  any trigger notifications are needed
      if(!already_subscribed && (active_generation > first_generation)) {
        // the reply has to carry the reduction values the subscriber has not
        //  seen, so the owner needs a per-node record of what it last told each
        //  node.  This is the map the scalable path replaced with 'lk'.
        LegacyReductionState &ls = legacy_state();
        std::map<unsigned, EventImpl::gen_t>::iterator it =
            ls.remote_trigger_gens.find(subscriber);
        if((it == ls.remote_trigger_gens.end()) || (it->second < active_generation)) {
          previous_gen =
              ((it == ls.remote_trigger_gens.end()) ? first_generation : it->second);
          trigger_gen = active_generation;
          ls.remote_trigger_gens[subscriber] = active_generation;

          int rel_gen = previous_gen + 1 - first_generation;
          assert(rel_gen > 0);
          final_values_size = (trigger_gen - previous_gen) * redop->sizeof_lhs;
          final_values_copy =
              bytedup(final_values.data() + ((rel_gen - 1) * redop->sizeof_lhs),
                      final_values_size);
        }
      }
    } while(0);

    if(forward_to_node != (NodeID)-1) {
      BarrierSubscribeMessage::send_request(forward_to_node, me.id, subscribe_gen,
                                            last_known, subscriber);
    }

    if(send_reply) {
      log_barrier.info() << "replying to barrier subscribe: " << me
                         << " dest=" << subscriber << " lk=" << last_known
                         << " wm=" << reply_wm << " sv=" << reply_sv
                         << " pois=" << reply_poison.size();
      BarrierSubscribeReplyMessage::send_request(subscriber, me.id, reply_wm, reply_sv,
                                                 reply_poison);
    }

    // send trigger message outside of lock, if needed
    if(trigger_gen > 0) {
      BarrierTriggerMessage::send_request(subscriber, me.id, trigger_gen, previous_gen,
                                          first_generation, redop_id, final_values_copy,
                                          final_values_size);
    }

    if(final_values_copy) {
      free(final_values_copy);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL action D, OWNER SIDE - rule 8.
  //
  // One critical section, and it does exactly one thing: remember that this
  //  node would like to be dropped.  Nothing is decided here.  The shrink is
  //  chosen - or declined - inside action T, which is the only place it can be,
  //  because rule 1 requires the shrink to be published to the PRE-SHRINK set
  //  and action T is where that snapshot is taken.
  //
  void BarrierImpl::handle_remote_depart(NodeID from)
  {
    AutoLock<> a(mutex);

    if(owner != Network::my_node_id) {
      // 'owner' is write-once now that migration is gone, so a departure
      //  reaching a non-owner is not a state this protocol can produce.  Drop
      //  it rather than forward: an intent is worth nothing, rule 8 is free to
      //  lose them, and the node's own idle counter will offer another.
      return;
    }
    if(redop_id != 0) {
      // reduction barriers stay on the legacy eager path (D1) and have no
      //  subscriber set to shrink
      return;
    }

    counters.departs_received++;

    if(!sub_set.contains(from)) {
      // a stale intent: the node has already been removed, or never joined.
      //  Recording it would make the next cost test weigh a removal that is not
      //  actually on offer, and 'want_out' is filtered against 'sub_set' again
      //  at action T anyway.
      return;
    }

    want_out.add(from);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL action N - a notification arrives (BarrierNotify
  //  RecvNotify).  ONE critical section, in this order.
  //
  void BarrierImpl::handle_remote_notify(gen_t wm, gen_t prev, gen_t gather_gen,
                                         uint64_t sv, const gen_t *poison,
                                         size_t num_poisoned, bool inset, bool hint,
                                         TimeLimit work_until)
  {
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // STEP 0 - RULE 8's advisory byte.  Not version gated and not ordered
      //  against anything: it is a hint, the worst a stale one can do is waste
      //  or withhold one unicast, and both are bandwidth.
      shrink_hint = hint;

      // STEP 1 - MEMBERSHIP, VERSION GATED, AND APPLIED EVEN ON A GAP.
      //  RULE 2: only a strictly newer version may change what this node
      //  believes, or a stale in-flight notification resurrects membership the
      //  owner has already dropped and the node hangs on its next wait.
      //  RULE 1: this message may be this node's ONLY notice of its own
      //  removal, so the membership half runs before - and independently of -
      //  the delta half below.
      if(sv > my_set_ver) {
        my_set_ver = sv;
        member = (inset ? MEMBER_YES : MEMBER_NO);
        if(member == MEMBER_NO) {
          // rule 8 - the request this node made has been granted, so the latch
          //  that kept it from asking twice has done its job
          depart_outstanding = false;
        }
      }

      // STEP 2.5 (ordered BEFORE the wake in step 3b) - THE GATHERING
      //  GENERATION (ARRIVAL_PROTOCOL section 11.5).  The owner declared the
      //  generation after this trigger fully eager so that the next rebuild's
      //  evidence completes by construction.  The declaration rides THIS
      //  message precisely so it can be applied in the same critical section
      //  that wakes this node's waiters: a woken waiter's next arrival then
      //  finds 'flushing' already set and reports WITH a map.  A separate
      //  flush message loses that race on nearly every generation (measured
      //  by the P_STEP probe: gathered=7-of-8 declines, forever).
      if(gather_gen > generation.load()) {
        Generation *ng = get_generation_locked(gather_gen);
        enter_flush_locked(*ng, gather_gen, sends);
      }

      // STEP 2 - the gap test.  A delta is only applicable if this node has
      //  already seen everything below 'prev'.
      const gen_t known = generation.load();
      const bool gap = (prev > known);
      const bool fresh = (!gap) && (wm > known);

      if(gap) {
        // STEP 3a - RULE 4: DISCARD THE DELTA ENTIRELY.  Applying it would
        //  advance the watermark over generations whose poison status this node
        //  cannot know, which is a waiter woken with the wrong poison status.
        //  It is NOT buffered: 'held_triggers' is not coming back, because the
        //  pull path has to exist anyway and is strictly simpler.
        counters.gap_pulls++;
        log_barrier.info() << "discarding barrier notify with gap: " << me
                           << " known=" << known << " (" << prev << " -> " << wm << ")";
      } else if(fresh) {
        // STEP 3b - poison first, then the watermark (section 3.5)
        publish_watermark_locked(wm, poison, num_poisoned, local_notifications,
                                 poisoned_notifications);
      }

      // STEP 4 - RULE 6.  A node removed while it still holds an outstanding
      //  waiter RE-SUBSCRIBES AT ONCE.  This is the correctness rule: the idle
      //  counter advances with the watermark, so a node waiting on a far-future
      //  generation LOOKS IDLE and can be dropped.  Declining to depart while
      //  holding a waiter is only an optimisation and is not sufficient on its
      //  own, because a waiter can be registered after the departure intent was
      //  collected and before the owner applies the shrink.
      const bool resub = (member == MEMBER_NO) && has_waiters_locked();
      if(resub) {
        // rule 8 - being dropped and immediately needing back in is the
        //  clearest churn signal there is, and it is exactly the case rule 6
        //  describes: a node parked on a far-future generation looks idle
        //  while the watermark climbs underneath it
        note_rejoin_locked();
        member = MEMBER_PENDING;
      }

      // STEP 5 - the pull, rule 7 gated
      if(gap || resub) {
        record_pull_locked(sends);
      }

      // STEP 6 - RULE 8, THE DEPARTURE HYSTERESIS.  It is evaluated here
      //  because this is where the watermark advances, and the idle counter is
      //  measured in watermark generations rather than in messages or in time.
      consider_departure_locked(sends);
    }

    emit_pending_sends(sends);
    deliver_waiters(local_notifications, poisoned_notifications, work_until);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // NOTIFICATION_PROTOCOL action RP - the answer to a pull (BarrierNotify
  //  RecvReply).
  //
  void BarrierImpl::handle_remote_subscribe_reply(gen_t wm, uint64_t sv,
                                                  const gen_t *poison,
                                                  size_t num_poisoned,
                                                  TimeLimit work_until)
  {
    EventWaiter::EventWaiterList local_notifications, poisoned_notifications;
    PendingSends sends;

    {
      AutoLock<> a(mutex);

      // STEP 1 - the rule-7 window closes
      pull_outstanding = false;

      // STEP 2 - RULE 2 again.  The SAME version gate is required here: a stale
      //  reply resurrects membership exactly the way a stale notify does.
      if(sv > my_set_ver) {
        my_set_ver = sv;
        member = MEMBER_YES;
      }

      // STEP 3 - RULE 5: the poison list is a DELTA above the 'lk' this node
      //  sent, so it is MERGED (add_poison_locked is idempotent), never
      //  substituted.  Substituting would drop the poison this node already
      //  knew about below 'lk'.
      if(wm > generation.load()) {
        publish_watermark_locked(wm, poison, num_poisoned, local_notifications,
                                 poisoned_notifications);
      }
      // STEP 4 - if the reply is not fresh, do not merge.  Safe because this
      //  node's poison knowledge is already exactly the truth up to its own
      //  (higher) watermark.

      // STEP 5 - re-issue whatever rule 7 suppressed while this pull was
      //  outstanding.  The model only suppresses while a SUBSCRIBE is in
      //  flight; from the owner's consumption of it to the reply landing here
      //  the model WOULD have pulled, so anything recorded in 'pull_deferred'
      //  is issued now.  A node removed while still holding a waiter also
      //  recovers here, for the same reason it does in action N.
      const bool resub = (member == MEMBER_NO) && has_waiters_locked();
      if(resub) {
        note_rejoin_locked(); // rule 8 - see action N step 4
        member = MEMBER_PENDING;
      }
      if(pull_deferred || resub) {
        pull_deferred = false;
        record_pull_locked(sends);
      }

      // STEP 6 - RULE 8.  The reply advances the watermark too - often by a
      //  long way, since it is what a node that had fallen behind catches up
      //  on - so the eligibility test belongs here for the same reason it
      //  belongs in action N.
      consider_departure_locked(sends);
    }

    emit_pending_sends(sends);
    deliver_waiters(local_notifications, poisoned_notifications, work_until);
  }

  bool BarrierImpl::get_result(gen_t result_gen, void *value, size_t value_size)
  {
    // generation hasn't triggered yet?
    if(result_gen > generation.load_acquire()) {
      return false;
    }

    // take the lock so we can safely see how many results (if any) are on hand
    AutoLock<> al(mutex);

    // if it has triggered, we should have the data
    int rel_gen = result_gen - first_generation;
    assert(rel_gen > 0);
    assert((size_t)rel_gen <= value_capacity);

    assert(redop != 0);
    assert(value_size == redop->sizeof_lhs);
    assert(value != 0);

    std::memcpy(value, &final_values[(rel_gen - 1) * redop->sizeof_lhs],
                redop->sizeof_lhs);
    return true;
  }

  ActiveMessageHandlerReg<BarrierTriggerMessage> barrier_handler_trigger;
  ActiveMessageHandlerReg<BarrierReportMessage> barrier_report_message_handler;
  ActiveMessageHandlerReg<BarrierFlushMessage> barrier_flush_message_handler;
  ActiveMessageHandlerReg<BarrierInvalidateMessage> barrier_invalidate_message_handler;
  ActiveMessageHandlerReg<BarrierNewPlanMessage> barrier_newplan_message_handler;
  ActiveMessageHandlerReg<BarrierAlterMessage> barrier_alter_message_handler;
  ActiveMessageHandlerReg<BarrierTsArrivalMessage> barrier_ts_arrival_message_handler;
  ActiveMessageHandlerReg<BarrierAdjustMessage> barrier_adjust_message_handler;
  ActiveMessageHandlerReg<BarrierSubscribeMessage> barrier_subscribe_message_handler;
  ActiveMessageHandlerReg<BarrierNotifyMessage> barrier_notify_message_handler;
  ActiveMessageHandlerReg<BarrierSubscribeReplyMessage> barrier_subscribe_reply_handler;
  ActiveMessageHandlerReg<BarrierDepartMessage> barrier_depart_message_handler;
}; // namespace Realm
