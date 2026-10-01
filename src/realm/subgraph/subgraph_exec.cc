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

// Realm subgraph implementation

// Compiled subgraphs: execution. SubgraphExecutionState is one instantiation
// in flight; ProcSubgraphExecutor feeds its tasks to a processor's scheduler loop.

#include "realm/subgraph/subgraph_impl.h"
#include "realm/event_impl.h"
#include "realm/network.h"
#include "realm/numa/numasysif.h"
#include "realm/proc_impl.h"
#include "realm/runtime_impl.h"
#include "realm/tasks.h"
#include "realm/timers.h"
#include "realm/transfer/transfer.h"
#include "realm/idx_impl.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef __linux__
#include <sched.h>
#endif

namespace Realm {
  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphExecutionState
  //

  SubgraphExecutionState::SubgraphExecutionState(SubgraphImpl *_subgraph,
                                                 Event _finish_event, int _priority,
                                                 span<const Event> _postconditions)
    : subgraph(_subgraph)
    , finish_counter(int64_t(_subgraph->compiled.procs.size()) +
                     int64_t(_subgraph->compiled.num_direct_ops))
    , finish_event(_finish_event)
    , postconditions(_postconditions.data(), _postconditions.data() + _postconditions.size())
    , poisoned_inputs(_subgraph->compiled.input_words)
    , poisoned(false)
    , priority(_priority)
  {
    for(auto &w : poisoned_inputs)
      w.store(0);
    subgraph->acquire_blocks(blocks);
    const CompiledSubgraph &c = subgraph->compiled;
    for(size_t dm = 0; dm < c.domains.size(); dm++)
      memcpy(blocks[dm], c.domains[dm].image.data(), c.domains[dm].bytes);
    if(c.num_async_ops > 0) {
      async_ops.resize(c.num_async_ops);
      tokens.assign(c.num_async_ops, nullptr);
      for(size_t i = 0; i < c.ops.size(); i++)
        if(c.ops[i].async_index >= 0) {
          async_ops[c.ops[i].async_index].state = this;
          async_ops[c.ops[i].async_index].op = uint32_t(i);
        }
    }
  }

  SubgraphExecutionState::~SubgraphExecutionState()
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(const CompiledSubgraph::Proc &p : c.procs)
      if(p.num_async > 0)
        p.impl->release_subgraph_tokens(tokens.data() + p.first_async, p.num_async);
    subgraph->release_blocks(blocks);
  }

  atomic<int64_t> &SubgraphExecutionState::counter(uint32_t op) const
  {
    const CompiledSubgraph::Op &o = subgraph->compiled.ops[op];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[o.counter_domain] +
                                                o.counter_offset);
  }

  atomic<int64_t> *SubgraphExecutionState::queue(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.queue_offset);
  }

  atomic<uint64_t> &SubgraphExecutionState::tail(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<uint64_t> *>(blocks[p.domain] + p.tail_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::pending_inputs(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.inputs_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::remaining(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.remaining_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::postcond_counter(uint32_t pc) const
  {
    const CompiledSubgraph::Postcond &p = subgraph->compiled.postconds[pc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.counter_domain] +
                                                p.counter_offset);
  }

  ByteArrayRef SubgraphExecutionState::op_args(uint32_t op) const
  {
    const CompiledSubgraph::Op &o = subgraph->compiled.ops[op];
    if(o.args_domain >= 0)
      return ByteArrayRef(blocks[o.args_domain] + o.args_offset, o.args_size);
    if(o.kind == SubgraphDefinition::OPKIND_TASK) {
      const ByteArray &a = subgraph->defn->tasks[o.index].args;
      return ByteArrayRef(a.base(), a.size());
    }
    return ByteArrayRef(nullptr, 0);
  }

  void SubgraphExecutionState::interpolate(const void *args, size_t arglen)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(const CompiledSubgraph::Interp &it : c.interps) {
      if((it.src_offset + it.bytes) > arglen)
        SUBGRAPH_FATAL(subgraph->me, "interpolation reads " << it.src_offset << "+" << it.bytes
                                                            << " bytes of the instantiation "
                                                               "arguments, but only "
                                                            << arglen << " were given");
      const CompiledSubgraph::Op &o = c.ops[it.op];
      char *dst = blocks[o.args_domain] + o.args_offset + it.dst_offset;
      const char *src = static_cast<const char *>(args) + it.src_offset;
      if(it.redop_id == 0) {
        memcpy(dst, src, it.bytes);
      } else {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(it.redop_id, nullptr);
        (redop->cpu_apply_excl_fn)(dst, 0, src, 0, 1 /*count*/, redop->userdata);
      }
    }
  }

  void SubgraphExecutionState::setup_profiling(
      const SubgraphInstantiationProfiling &iprof)
  {
    const SubgraphDefinition &d = *subgraph->defn;
    const CompiledSubgraph &c = subgraph->compiled;
    if(!iprof.copies.empty()) {
      // copies profile through Realm's transfer operations: hand each one
      // the definition's requests merged with this instantiation's
      copy_prs.resize(d.copies.size());
      for(size_t i = 0; i < d.copies.size(); i++)
        copy_prs[i].import_requests(d.copies[i].prs);
      for(const auto &kv : iprof.copies) {
        if(kv.first >= d.copies.size())
          SUBGRAPH_FATAL(subgraph->me, "profiling requested for copy "
                                           << kv.first << ", which does not exist");
        copy_prs[kv.first].import_requests(kv.second);
      }
    }
    if(!c.any_task_profiling && iprof.tasks.empty())
      return;
    prof_index.assign(c.ops.size(), -1);
    auto entry_for_task = [&](unsigned task) -> OpProfiling & {
      if(task >= d.tasks.size())
        SUBGRAPH_FATAL(subgraph->me, "profiling requested for task " << task
                                                                     << ", which does not exist");
      int32_t &idx = prof_index[c.task_ops[task]];
      if(idx < 0) {
        idx = int32_t(profiling.size());
        profiling.emplace_back(new OpProfiling);
        profiling.back()->requests.import_requests(d.tasks[task].prs);
      }
      return *profiling[idx];
    };
    for(size_t t = 0; t < d.tasks.size(); t++)
      if(!d.tasks[t].prs.empty())
        entry_for_task(unsigned(t));
    for(const auto &kv : iprof.tasks)
      entry_for_task(kv.first).requests.import_requests(kv.second);
    for(auto &p : profiling) {
      p->measurements.import_requests(p->requests);
      p->wants_timeline =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>();
      p->wants_proc =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationProcessorUsage>();
      p->wants_status =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationStatus>();
      p->wants_fevent =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationFinishEvent>();
      if(p->wants_timeline)
        p->timeline.record_create_time();
    }
  }

  bool SubgraphExecutionState::op_poisoned(uint32_t op) const
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const uint64_t *bits = c.op_inputs.data() + size_t(op) * c.input_words;
    for(size_t w = 0; w < c.input_words; w++)
      if(bits[w] & poisoned_inputs[w].load_acquire())
        return true;
    return false;
  }

  void SubgraphExecutionState::input_triggered(uint32_t input, bool is_poisoned)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Input &in = c.inputs[input];
    if(is_poisoned) {
      // Published before any dependent operation can become ready.
      poisoned_inputs[input / 64].fetch_or_acqrel(uint64_t(1) << (input % 64));
      poisoned.store_release(true);
    }
    for(uint32_t p : in.procs)
      pending_inputs(p).fetch_sub_acqrel(1);
    for(uint32_t t : in.targets)
      if(counter(t).fetch_sub_acqrel(1) == 1)
        op_ready(t);
  }

  void SubgraphExecutionState::start(void)
  {
    for(uint32_t r : subgraph->compiled.roots)
      if(counter(r).fetch_sub_acqrel(1) == 1)
        op_ready(r);
  }

  void SubgraphExecutionState::op_ready(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Op &o = c.ops[op];
    if(OpProfiling *pf = prof(op))
      if(pf->wants_timeline)
        pf->timeline.record_ready_time();
    if(o.proc >= 0) {
      const uint64_t slot = tail(o.proc).fetch_add_acqrel(1);
      queue(o.proc)[slot].store_release(int64_t(op));
      c.procs[o.proc].impl->notify_scheduler_of_new_work();
      return;
    }
    // Directly launched operation: performed by whoever made it ready.
    switch(o.kind) {
    case SubgraphDefinition::OPKIND_ARRIVAL:
    {
      if(!op_poisoned(op)) {
        const SubgraphDefinition::ArrivalDesc &ad = subgraph->defn->arrivals[o.index];
        Barrier b = ad.barrier;
        const void *value = ad.reduce_value.base();
        size_t value_size = ad.reduce_value.size();
        if(o.args_domain >= 0) {
          ByteArrayRef a = op_args(op);
          memcpy(&b, a.base(), sizeof(Barrier));
          value = static_cast<const char *>(a.base()) + sizeof(Barrier);
          value_size = a.size() - sizeof(Barrier);
        }
        b.arrive(ad.count, Event::NO_EVENT, value, value_size);
      }
      break;
    }
    case SubgraphDefinition::OPKIND_COPY:
      if(!op_poisoned(op)) {
        launch_copy(op); // completes asynchronously
        return;
      }
      break;
    default:
      SUBGRAPH_FATAL(subgraph->me, "internal error: operation " << op
                                                                << " of kind "
                                                                << op_kind_name(o.kind)
                                                                << " cannot be launched directly");
    }
    op_body_done(op);
    op_finished(op);
  }

  namespace {
    // A transfer on a compiled plan that reports back to the instantiation
    // when the data has moved. Realm's own bookkeeping (profiling responses,
    // the operation's finish event) runs first.
    class SubgraphTransferOperation : public TransferOperation {
    public:
      SubgraphTransferOperation(TransferDesc &desc, GenEventImpl *finish_event,
                                EventImpl::gen_t finish_gen, int priority,
                                const ProfilingRequestSet &prs,
                                SubgraphExecutionState *_state, uint32_t _op)
        : TransferOperation(desc, Event::NO_EVENT, finish_event, finish_gen, priority, prs)
        , state(_state)
        , op(_op)
      {}

    protected:
      virtual void mark_completed(void) override
      {
        SubgraphExecutionState *s = state;
        const uint32_t o = op;
        TransferOperation::mark_completed(); // may delete this operation
        s->op_body_done(o);
        s->op_finished(o);
      }

      SubgraphExecutionState *state;
      uint32_t op;
    };
  } // namespace

  void SubgraphExecutionState::launch_copy(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Op &o = c.ops[op];
    const SubgraphDefinition::CopyDesc &cd = subgraph->defn->copies[o.index];
    TransferDesc *plan = c.copy_plans[o.index];
    // every transfer operation needs a finish event of its own
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();
    const ProfilingRequestSet &prs = copy_prs.empty() ? cd.prs : copy_prs[o.index];
    SubgraphTransferOperation *top = new SubgraphTransferOperation(
        *plan, finish_event, ID(ev).event_generation(), priority + cd.priority, prs, this,
        op);
    // the plan is analyzed and the graph satisfied the preconditions: this
    // allocates intermediate buffers and creates the transfer descriptors
    top->start_or_defer();
  }

  void SubgraphExecutionState::op_body_done(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(uint64_t i = c.successors.offsets[op]; i < c.successors.offsets[op + 1]; i++) {
      const uint32_t sx = c.successors.data[i];
      if(counter(sx).fetch_sub_acqrel(1) == 1)
        op_ready(sx);
    }
  }

  void SubgraphExecutionState::op_finished(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(uint64_t i = c.postconds_of.offsets[op]; i < c.postconds_of.offsets[op + 1]; i++) {
      const uint32_t pc = c.postconds_of.data[i];
      if(postcond_counter(pc).fetch_sub_acqrel(1) != 1)
        continue;
      bool pc_poisoned = false;
      const uint64_t *bits = c.postcond_inputs.data() + size_t(pc) * c.input_words;
      for(size_t w = 0; w < c.input_words; w++)
        pc_poisoned = pc_poisoned || ((bits[w] & poisoned_inputs[w].load_acquire()) != 0);
      GenEventImpl::trigger(postconditions[pc], pc_poisoned);
    }
    for(uint64_t i = c.late_successors.offsets[op]; i < c.late_successors.offsets[op + 1];
        i++) {
      const uint32_t sx = c.late_successors.data[i];
      if(counter(sx).fetch_sub_acqrel(1) == 1)
        op_ready(sx);
    }
    // Last: a processor's share of the finish counter goes when its last
    // operation completes, whichever one that is (tasks may block or
    // complete asynchronously), and the state may be freed right after.
    const int32_t proc = c.ops[op].proc;
    if(proc < 0) {
      contributor_finished();
    } else if(remaining(uint32_t(proc)).fetch_sub_acqrel(1) == 1) {
      contributor_finished();
    }
  }

  void SubgraphExecutionState::complete_op_profiling(uint32_t op, OpProfiling *pf,
                                                     Event finish_event, bool skipped)
  {
    if(!pf)
      return;
    if(pf->wants_timeline) {
      if(!skipped)
        pf->timeline.record_complete_time();
      pf->measurements.add_measurement(pf->timeline);
    }
    if(pf->wants_proc) {
      ProfilingMeasurements::OperationProcessorUsage usage;
      usage.proc = subgraph->compiled.procs[subgraph->compiled.ops[op].proc].proc;
      pf->measurements.add_measurement(usage);
    }
    if(pf->wants_status) {
      ProfilingMeasurements::OperationStatus status;
      status.result = skipped ? ProfilingMeasurements::OperationStatus::CANCELLED
                              : ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY;
      status.error_code = 0;
      pf->measurements.add_measurement(status);
    }
    if(pf->wants_fevent) {
      ProfilingMeasurements::OperationFinishEvent fe;
      fe.finish_event = finish_event;
      pf->measurements.add_measurement(fe);
    }
    pf->measurements.send_responses(pf->requests);
  }

  void SubgraphExecutionState::AsyncOp::async_completed(void)
  {
    // The work this task launched has completed (called from the
    // processor's completion machinery, e.g. the GPU worker).
    if(finish_event.exists())
      GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    state->complete_op_profiling(op, state->prof(op), finish_event, false /*!skipped*/);
    state->op_finished(op);
  }

  void SubgraphExecutionState::contributor_finished(void)
  {
    // Copy out what is needed first: once the counter reaches zero and the
    // event triggers, this state may be released at any moment.
    Event fe = finish_event;
    bool p = poisoned.load_acquire();
    if(finish_counter.fetch_sub_acqrel(1) == 1)
      GenEventImpl::trigger(fe, p);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcSubgraphExecutor
  //

  /*static*/ int ProcSubgraphExecutor::poll_budget_us = 100;

  ProcSubgraphExecutor::ProcSubgraphExecutor(Processor _proc)
    : proc(_proc)
    , numa_node_(-2)
    , pending_count(0)
    , scan_start(0)
    , peeked_cursor(0)
    , peeked_op(SUBGRAPH_EMPTY_QUEUE_ENTRY)
    , activity_epoch(0)
    , polled_epoch(0)
    , poll_deadline_ns(0)
  {}

  ProcSubgraphExecutor::~ProcSubgraphExecutor() {}

  void ProcSubgraphExecutor::note_worker_started(void)
  {
    if(numa_node_ != -2)
      return;
    int node = -1;
#ifdef __linux__
    int cpu = sched_getcpu();
    if((cpu >= 0) && numasysif_numa_available()) {
      const HardwareTopology &topo = get_runtime()->host_topology;
      for(int dm = 0; (dm < 1024) && (node < 0); dm++)
        if(topo.numa_domain_has_processors(dm) &&
           topo.get_processors_by_domain(dm).count(cpu))
          node = dm;
    }
#endif
    numa_node_ = node;
  }

  void ProcSubgraphExecutor::enqueue_subgraph(SubgraphExecutionState *state)
  {
    AutoLock<> al(pending_mutex);
    pending.push_back(state);
    pending_count.store_release(int64_t(pending.size()));
  }

  void ProcSubgraphExecutor::absorb_pending(void)
  {
    {
      AutoLock<> al(pending_mutex);
      pending.swap(pending_scratch);
      pending_count.store(0);
    }
    for(SubgraphExecutionState *state : pending_scratch) {
      const CompiledSubgraph &c = state->subgraph->compiled;
      auto it = c.proc_index.find(proc);
      assert(it != c.proc_index.end());
      Cursor cur;
      cur.state = state;
      cur.proc = it->second;
      cur.queue = state->queue(cur.proc);
      cur.front = 0;
      cur.end = c.procs[cur.proc].num_ops;
      cur.priority = state->get_priority();
      assert(cur.end > 0);
      active.push_back(cur);
    }
    pending_scratch.clear();
    activity_epoch++;
  }

  bool ProcSubgraphExecutor::peek(int &priority)
  {
    if(pending_count.load_acquire() > 0)
      absorb_pending();

    // Highest-priority ready instantiation wins; ties go round robin.
    // Instantiations below the floor wait, like normal tasks do, even if
    // the ones holding the processor have nothing ready right now.
    const int floor = active_floor(INT_MIN);
    const size_t n = active.size();
    bool found = false;
    for(size_t k = 0; k < n; k++) {
      size_t i = scan_start + k;
      if(i >= n)
        i -= n;
      const Cursor &c = active[i];
      if(c.priority < floor)
        continue;
      if(found && (c.priority <= priority))
        continue;
      int64_t op = c.queue[c.front].load_acquire();
      if(op != SUBGRAPH_EMPTY_QUEUE_ENTRY) {
        peeked_cursor = i;
        peeked_op = op;
        priority = c.priority;
        found = true;
      }
    }
    return found;
  }

  int ProcSubgraphExecutor::active_floor(int none) const
  {
    int floor = none;
    bool any = false;
    for(const Cursor &c : active) {
      if(c.state->pending_inputs(c.proc).load_acquire() != 0)
        continue; // may still be waiting on work this processor would block
      if(!any || (c.priority > floor))
        floor = c.priority;
      any = true;
    }
    return floor;
  }

  void ProcSubgraphExecutor::dequeue(ReadyEntry &entry)
  {
    assert((peeked_cursor < active.size()) && (peeked_op != SUBGRAPH_EMPTY_QUEUE_ENTRY));
    Cursor &c = active[peeked_cursor];
    entry.state = c.state;
    entry.op = uint32_t(peeked_op);
    entry.priority = c.priority;
    c.front++;
    if(c.front == c.end) {
      // Drop the cursor now: once this processor's operations have all
      // completed, its finish decrement may let the state be released at
      // any time.
      active[peeked_cursor] = active.back();
      active.pop_back();
      scan_start = peeked_cursor;
    } else {
      scan_start = peeked_cursor + 1;
    }
    if(scan_start >= active.size())
      scan_start = 0;
    peeked_op = SUBGRAPH_EMPTY_QUEUE_ENTRY;
    activity_epoch++;
  }

  void ProcSubgraphExecutor::execute(const ReadyEntry &entry)
  {
    SubgraphExecutionState *state = entry.state;
    SubgraphImpl *impl = state->subgraph;
    const CompiledSubgraph &c = impl->compiled;
    const CompiledSubgraph::Op &op = c.ops[entry.op];
    assert((op.kind == SubgraphDefinition::OPKIND_TASK) && (op.proc >= 0));

    LocalTaskProcessor *proc_impl = c.procs[op.proc].impl;
    SubgraphExecutionState::OpProfiling *pf = state->prof(entry.op);

    if(state->op_poisoned(entry.op)) {
      // A task depending on a poisoned input is skipped; its successors
      // still drain and the finish event ends up poisoned.
      state->op_body_done(entry.op);
      state->complete_op_profiling(entry.op, pf, Event::NO_EVENT, true /*skipped*/);
      state->op_finished(entry.op);
      return;
    }

    const SubgraphDefinition::TaskDesc &task_desc = impl->defn->tasks[op.index];

    // Tokens of the deferred-effects predecessors this task's work must follow.
    const uint64_t tw0 = c.token_waits.offsets[entry.op];
    const size_t num_tokens = size_t(c.token_waits.offsets[entry.op + 1] - tw0);
    const void *token_buf[16];
    std::vector<const void *> token_vec;
    const void **wait_tokens = token_buf;
    if(num_tokens > 16) {
      token_vec.resize(num_tokens);
      wait_tokens = token_vec.data();
    }
    for(size_t k = 0; k < num_tokens; k++)
      wait_tokens[k] = state->tokens[c.token_waits.data[tw0 + k]];

    // Run the task function on this thread. There is no Operation: the flag
    // lets Processor::get_current_finish_event create a finish event on
    // demand, triggered once the task is complete. Context managers are not
    // applied; the processor's subgraph hooks provide the equivalent.
    Thread *thread = Thread::self();
    thread->subgraph_finish_event() =
        (pf && pf->wants_fevent) ? UserEvent::create_user_event().id : 0;
    if(pf && pf->wants_timeline)
      pf->timeline.record_start_time();
    void *context = proc_impl->begin_subgraph_task(wait_tokens, num_tokens);
    ThreadLocal::current_processor = proc;
    thread->start_subgraph_task_execution();
    proc_impl->execute_task(task_desc.task_id, state->op_args(entry.op));
    thread->stop_subgraph_task_execution();
    ThreadLocal::current_processor = Processor::NO_PROC;
    void *token = proc_impl->end_subgraph_task(context, op.deferred);
    if(pf && pf->wants_timeline)
      pf->timeline.record_end_time();
    Event finish_event;
    finish_event.id = thread->subgraph_finish_event();
    thread->subgraph_finish_event() = 0;

    if(!op.async) {
      if(finish_event.exists())
        GenEventImpl::trigger(finish_event, false /*!poisoned*/);
      state->op_body_done(entry.op);
      state->complete_op_profiling(entry.op, pf, finish_event, false /*!skipped*/);
      state->op_finished(entry.op);
      return;
    }

    // The function returned but the work goes on. Successors that may start
    // now are released first: arming the completion may finish the
    // instantiation, and free the state, before it returns.
    SubgraphExecutionState::AsyncOp &ao = state->async_ops[op.async_index];
    ao.finish_event = finish_event;
    state->tokens[op.async_index] = token;
    state->op_body_done(entry.op);
    proc_impl->arm_subgraph_task_completion(context, token, &ao);
  }

  bool ProcSubgraphExecutor::keep_polling(void)
  {
    if((poll_budget_us <= 0) || (activity_epoch == 0))
      return false;
    // While an instantiation holds this processor, nothing else may run
    // here anyway: keep polling until it is done.
    if(!active.empty())
      return true;
    long long now = Clock::current_time_in_nanoseconds();
    if(activity_epoch != polled_epoch) {
      polled_epoch = activity_epoch;
      poll_deadline_ns = now + 1000LL * poll_budget_us;
    }
    return now < poll_deadline_ns;
  }


}; // namespace Realm
