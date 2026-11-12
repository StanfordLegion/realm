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

// implementation of profiling stuff for Realm

#include "realm/profiling.h"

#include "realm/profiling_batch.h"
#include "realm/id.h"
#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/mutex.h"
#include "realm/network.h"
#include "realm/timers.h"

#include <atomic>
#include <cstdlib>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // batched delivery of profiling responses
  //
  // Responses to requests that allow it (ProfilingRequest::set_batched_delivery)
  //  and whose response processor is local are queued per (processor, task,
  //  priority) on a lock-free list. One delivery task, spawned only when the
  //  queue has no delivery pending or running, runs the response task's body
  //  for each queued response in turn, so most responses cost a push instead
  //  of a task spawn (with its event, scheduler notifications and possibly a
  //  thread wake-up).

  namespace ProfilingBatch {

    static Logger log_prof_batch("profbatch");

    struct Queue;

    struct Response {
      Response *next;
      size_t size;
      char *data(void) { return reinterpret_cast<char *>(this + 1); }
    };

    struct Queue {
      Processor proc;
      Processor::TaskFuncID task_id;
      int priority;
      std::atomic<Response *> head{nullptr}; // pushed in LIFO order
      std::atomic<bool> scheduled{false};    // a delivery task is pending or running
      Response *carry = nullptr;             // left by a delivery task that ran out
                                             //  of time (owned by the next one)
      Queue *next = nullptr;
    };

    // queues are created on first use and freed by cleanup() at shutdown
    static std::atomic<Queue *> queues{nullptr};
    static Mutex queues_mutex;

    // a delivery task runs responses for at most this long before handing the
    //  rest to a new delivery task, so other work at the same priority on the
    //  processor is not held up for long
    static const long long DELIVERY_TIME_SLICE_NS = 5000;

    static bool is_local(Processor p)
    {
      ID id(p);
      if(id.is_processor())
        return (NodeID(id.proc_owner_node()) == Network::my_node_id);
      if(id.is_procgroup())
        return (NodeID(id.pgroup_owner_node()) == Network::my_node_id);
      return false;
    }

    static Queue *find_queue(Processor p, Processor::TaskFuncID tid, int prio)
    {
      for(Queue *q = queues.load(std::memory_order_acquire); q != nullptr; q = q->next)
        if((q->proc == p) && (q->task_id == tid) && (q->priority == prio))
          return q;
      AutoLock<> al(queues_mutex);
      for(Queue *q = queues.load(std::memory_order_acquire); q != nullptr; q = q->next)
        if((q->proc == p) && (q->task_id == tid) && (q->priority == prio))
          return q;
      Queue *q = new Queue;
      q->proc = p;
      q->task_id = tid;
      q->priority = prio;
      q->next = queues.load(std::memory_order_relaxed);
      queues.store(q, std::memory_order_release);
      return q;
    }

    static void push(Queue *q, Response *r)
    {
      Response *h = q->head.load(std::memory_order_relaxed);
      do {
        r->next = h;
      } while(!q->head.compare_exchange_weak(h, r, std::memory_order_seq_cst));
      // (seq_cst: pairs with the delivery task's store of 'scheduled' and load
      //  of 'head', so that one of the two always sees the other's update)
      if(!q->scheduled.exchange(true, std::memory_order_seq_cst))
        Task::spawn_profiling_delivery(q->proc, q, q->priority);
    }

    void deliver(void *queue, ExecFn exec, void *ctx)
    {
      Queue *q = static_cast<Queue *>(queue);
      const long long t_end =
          Clock::current_time_in_nanoseconds() + DELIVERY_TIME_SLICE_NS;
      // this task owns the queue's delivery ('scheduled' is set) until it
      //  either empties the queue or hands it to a new delivery task
      Response *list = q->carry;
      q->carry = nullptr;
      while(true) {
        if(list == nullptr) {
          Response *lifo = q->head.exchange(nullptr, std::memory_order_seq_cst);
          if(lifo == nullptr) {
            q->scheduled.store(false, std::memory_order_seq_cst);
            // a push that saw 'scheduled' still set may have raced with the
            //  exchange above - if so, take its response or let the delivery
            //  task it spawned do it
            if(q->head.load(std::memory_order_seq_cst) == nullptr)
              return;
            if(q->scheduled.exchange(true, std::memory_order_seq_cst))
              return;
            continue;
          }
          // reverse into arrival order
          while(lifo != nullptr) {
            Response *n = lifo->next;
            lifo->next = list;
            list = lifo;
            lifo = n;
          }
        }
        Response *r = list;
        list = r->next;
        (*exec)(ctx, q->task_id, r->data(), r->size);
        free(r);
        if((Clock::current_time_in_nanoseconds() >= t_end) &&
           ((list != nullptr) || (q->head.load(std::memory_order_relaxed) != nullptr))) {
          // out of time: hand the rest to a new delivery task (we still own
          //  'scheduled', so no push will spawn one)
          q->carry = list;
          Task::spawn_profiling_delivery(q->proc, q, q->priority);
          return;
        }
      }
    }

    void cleanup(void)
    {
      // every processor, DMA and background thread has stopped, so nothing
      //  pushes or delivers any more; drop what was never delivered
      size_t dropped = 0;
      Queue *q = queues.exchange(nullptr);
      while(q != nullptr) {
        for(Response *r : {q->carry, q->head.exchange(nullptr)}) {
          while(r != nullptr) {
            Response *n = r->next;
            free(r);
            r = n;
            dropped++;
          }
        }
        Queue *n = q->next;
        delete q;
        q = n;
      }
      if(dropped > 0)
        log_prof_batch.info() << dropped
                              << " batched profiling responses dropped at shutdown";
    }

  }; // namespace ProfilingBatch

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequest
  //

  ProfilingRequest::ProfilingRequest(Processor _response_proc,
                                     Processor::TaskFuncID _response_task_id,
                                     int _priority /*= 0*/,
                                     bool _report_if_empty /*= false*/)
    : response_proc(_response_proc)
    , response_task_id(_response_task_id)
    , priority(_priority)
    , report_if_empty(_report_if_empty)
    , batched_delivery(false)
  {}

  ProfilingRequest::ProfilingRequest(const ProfilingRequest &to_copy)
    : response_proc(to_copy.response_proc)
    , response_task_id(to_copy.response_task_id)
    , priority(to_copy.priority)
    , report_if_empty(to_copy.report_if_empty)
    , batched_delivery(to_copy.batched_delivery)
    , user_data(to_copy.user_data)
    , requested_measurements(to_copy.requested_measurements)
  {}

  ProfilingRequest::~ProfilingRequest(void) {}

  ProfilingRequest &ProfilingRequest::operator=(const ProfilingRequest &rhs)
  {
    response_proc = rhs.response_proc;
    response_task_id = rhs.response_task_id;
    priority = rhs.priority;
    report_if_empty = rhs.report_if_empty;
    batched_delivery = rhs.batched_delivery;
    requested_measurements = rhs.requested_measurements;
    user_data = rhs.user_data;
    return *this;
  }

  ProfilingRequest &ProfilingRequest::add_user_data(const void *payload,
                                                    size_t payload_size)
  {
    user_data.set(payload, payload_size);
    return *this;
  }

  ProfilingRequest &ProfilingRequest::set_batched_delivery(bool batched /*= true*/)
  {
    batched_delivery = batched;
    return *this;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequestSet
  //

  ProfilingRequestSet::ProfilingRequestSet(void) {}

  ProfilingRequestSet::ProfilingRequestSet(const ProfilingRequestSet &to_copy)
  {
    // deep copy
    for(std::vector<ProfilingRequest *>::const_iterator it = to_copy.requests.begin();
        it != to_copy.requests.end(); it++)
      requests.push_back(new ProfilingRequest(**it));
  }

  ProfilingRequestSet::~ProfilingRequestSet(void)
  {
    // destroy all of our set members
    for(std::vector<ProfilingRequest *>::iterator it = requests.begin();
        it != requests.end(); it++)
      delete *it;
  }

  ProfilingRequestSet &ProfilingRequestSet::operator=(const ProfilingRequestSet &rhs)
  {
    for(std::vector<ProfilingRequest *>::iterator it = requests.begin();
        it != requests.end(); it++)
      delete *it;
    requests.clear();
    // deep copy
    for(std::vector<ProfilingRequest *>::const_iterator it = rhs.requests.begin();
        it != rhs.requests.end(); it++)
      requests.push_back(new ProfilingRequest(**it));
    return *this;
  }

  ProfilingRequest &ProfilingRequestSet::add_request(
      Processor response_proc, Processor::TaskFuncID response_task_id,
      const void *payload /*= 0*/, size_t payload_size /*= 0*/, int priority /*= 0*/,
      bool report_if_empty /*= false*/)
  {
    ProfilingRequest *pr =
        new ProfilingRequest(response_proc, response_task_id, priority, report_if_empty);

    if(payload)
      pr->add_user_data(payload, payload_size);

    requests.push_back(pr);

    return *pr;
  }

  size_t ProfilingRequestSet::request_count(void) const { return requests.size(); }

  bool ProfilingRequestSet::empty(void) const { return requests.empty(); }

  void ProfilingRequestSet::clear(void)
  {
    for(std::vector<ProfilingRequest *>::iterator it = requests.begin();
        it != requests.end(); it++) {
      delete(*it);
    }
    requests.clear();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingMeasurementCollection
  //

  ProfilingMeasurementCollection::ProfilingMeasurementCollection(void)
    : completed_requests_present(false)
  {}

  ProfilingMeasurementCollection::~ProfilingMeasurementCollection(void) {}

  void ProfilingMeasurementCollection::import_requests(const ProfilingRequestSet &prs)
  {
    requests.reserve(requests.size() + prs.requests.size());
    // iterate over the individual requests, recording the measurements needed
    for(std::vector<ProfilingRequest *>::const_iterator it = prs.requests.begin();
        it != prs.requests.end(); it++) {
      if((*it)->requested_measurements.empty()) {
        // send responses to empty requests immediately (no point in waiting)
        send_response(**it);
      } else {
        requests.push_back(
            PendingRequest{*it, int((*it)->requested_measurements.size()), true});
        wanted.insert_all((*it)->requested_measurements);
      }
    }
  }

  const ProfilingMeasurementCollection::Measurement *
  ProfilingMeasurementCollection::find_measurement(ProfilingMeasurementID id) const
  {
    for(const Measurement &m : measurements)
      if(m.id == id)
        return &m;
    return nullptr;
  }

  void ProfilingMeasurementCollection::send_response(const ProfilingRequest &pr) const
  {
    // a task ID of 0 means that the measurement is being ignored - don't even
    //  bother forming it
    if(pr.response_task_id == Processor::TASK_ID_PROCESSOR_NOP)
      return;

    // for each request, find the intersection of the measurements it wants and the ones
    // we have, in increasing ID order
    const Measurement *local_found[16];
    std::vector<const Measurement *> more_found;
    int count = 0;

    // at the very least, we need a count of measurements and the offset of the user data
    size_t bytes_needed = 2 * sizeof(int) + pr.user_data.size();

    pr.requested_measurements.for_each([&](ProfilingMeasurementID id) {
      const Measurement *m = find_measurement(id);
      if(m == nullptr)
        return;
      if(count < 16)
        local_found[count] = m;
      else
        more_found.push_back(m);
      count++;
      // we'll pad each measurement to an 8 byte boundary
      size_t msize_padded = (m->size + 7) & ~7ULL;

      bytes_needed += 2 * sizeof(int); // to store ID and offset of data
      bytes_needed += msize_padded;    // to store actual data
    });

    // not everybody wants an empty report
    if((count == 0) && !pr.report_if_empty)
      return;

    // a batched response is built in place in its queue entry
    const bool batched =
        pr.batched_delivery && ProfilingBatch::is_local(pr.response_proc);
    ProfilingBatch::Response *bresp = nullptr;
    char *payload = nullptr;
    if(batched) {
      bresp = static_cast<ProfilingBatch::Response *>(
          malloc(sizeof(ProfilingBatch::Response) + bytes_needed));
      assert(bresp != 0);
      bresp->size = bytes_needed;
      payload = bresp->data();
    } else {
      payload = (char *)malloc(bytes_needed);
      assert(payload != 0);
    }

    int *header = (int *)payload; // first bunch of stuff is a big int array
    char *data = payload + (2 + 2 * count) * sizeof(int);

    *header++ = count;
    for(int i = 0; i < count; i++) {
      const Measurement *m = (i < 16) ? local_found[i] : more_found[i - 16];
      *header = (int)(m->id);
      *(header + count) = data - payload; // offset of data start
      header++;

      if(m->size > 0) {
        memcpy(data, this->data.data() + m->offset, m->size);

        size_t msize_padded = (m->size + 7) & ~7ULL;
        data += msize_padded;
      }
    }

    // offset of user data start is always provided (if it equals the response size,
    // there's no data)
    *(header + count) = data - payload;
    if(pr.user_data.size() > 0) {
      memcpy(data, pr.user_data.base(), pr.user_data.size());
      data += pr.user_data.size();
    }

    assert((size_t)(data - payload) == bytes_needed);

    if(batched) {
      ProfilingBatch::push(
          ProfilingBatch::find_queue(pr.response_proc, pr.response_task_id, pr.priority),
          bresp);
      return;
    }

    pr.response_proc.spawn(pr.response_task_id, payload, bytes_needed, Event::NO_EVENT,
                           pr.priority);

    free(payload);
  }

  void ProfilingMeasurementCollection::send_responses(const ProfilingRequestSet &prs)
  {
    // print raw data right now
#ifdef DEBUG_PROFILING
    printf("raw profiling results:\n");
    for(const Measurement &m : measurements) {
      printf("[%d] = %zd (", (int)(m.id), m.size);
      for(size_t i = 0; i < m.size; i++)
        printf(" %02x", (unsigned char)data[m.offset + i]);
      printf(" )\n");
    }
#endif

    // only send responses for things that we didn't send eagerly on completion
    //  ('requests' holds the requests of 'prs' as imported, in order)
    assert(prs.requests.size() >= requests.size());
    for(const PendingRequest &pr : requests)
      if(pr.pending)
        send_response(*pr.request);
  }

  void ProfilingMeasurementCollection::clear_measurements(void)
  {
    measurements.clear();
    data.clear();

    // also have to restore the counts
    for(PendingRequest &pr : requests) {
      pr.measurements_left = int(pr.request->requested_measurements.size());
      pr.pending = true;
    }
    completed_requests_present = false;
  }

  void ProfilingMeasurementCollection::clear(void)
  {
    measurements.clear();
    data.clear();
    requests.clear();
    wanted.clear();
    completed_requests_present = false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingResponse
  //

  ProfilingResponse::ProfilingResponse(const void *_data, size_t _data_size)
    : data(static_cast<const char *>(_data))
    , data_size(_data_size)
  {
    const int *idata = static_cast<const int *>(_data);

    measurement_count = idata[0];
    ids = &(idata[1]);
    user_data_offset = idata[2 * measurement_count + 1];
  }

  ProfilingResponse::~ProfilingResponse(void)
  {
    // nothing to free - we didn't own the data
  }

  const void *ProfilingResponse::user_data(void) const
  {
    if(user_data_offset < data_size)
      return data + user_data_offset;
    else
      return 0;
  }

  size_t ProfilingResponse::user_data_size(void) const
  {
    return data_size - user_data_offset;
  }

  bool ProfilingResponse::find_id(int id, int &offset, int &size) const
  {
    // binary search on ids
    int lo = 0;
    int hi = measurement_count - 1;

    while(lo <= hi) {
      int mid = (lo + hi) >> 1;
      if(id < ids[mid]) {
        hi = mid - 1;
      } else if(id > ids[mid]) {
        lo = mid + 1;
      } else {
        offset = ids[mid + measurement_count];
        size = ids[mid + measurement_count + 1] - offset;
        return true;
      }
    }

    return false;
  }

}; // namespace Realm
