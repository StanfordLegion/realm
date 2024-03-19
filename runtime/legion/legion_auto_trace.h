/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_AUTO_TRACE_H__
#define __LEGION_AUTO_TRACE_H__

#include "legion.h"
#include "legion/legion_context.h"
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/suffix_tree.h"
#include "legion/trie.h"

#include <limits>
#include <queue>

template<>
struct std::hash<Legion::Internal::Murmur3Hasher::Hash> {
  std::size_t operator()(const Legion::Internal::Murmur3Hasher::Hash& h) const noexcept {
    return h.x ^ (h.y << 1);
  }
};

template<>
struct std::equal_to<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(const Legion::Internal::Murmur3Hasher::Hash& lhs,
                            const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
      return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

template<>
struct std::less<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(const Legion::Internal::Murmur3Hasher::Hash& lhs,
                            const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
    return lhs.x < rhs.x && lhs.y < rhs.y;
  }
};

namespace Legion {
  namespace Internal {

    // Declare all of the necessary loggers.
    LEGION_EXTERN_LOGGER_DECLARATIONS

    // Forward declarations.
    class BatchedTraceIdentifier;
    class TraceOccurrenceWatcher;
    class TraceReplayer;

    // TraceHashHelper is a utility class to hash Operations.
    class TraceHashHelper {
    public:
      TraceHashHelper();
      Murmur3Hasher::Hash hash(Operation* op);
      void hash(TaskOp* op);
      void hash(FillOp* op);
      void hash(FenceOp* op);
      void hash(CopyOp* op);
      void hash(AllReduceOp* op);
      void hash(AcquireOp* op);
      void hash(ReleaseOp* op);
      void hash(const RegionRequirement& req);
      void hash(const LogicalRegion& region);
    private:
      Murmur3Hasher hasher;
    };

    // TraceProcessingJobExecutor is an interface that hides the details
    // of executing and waiting for async trace processing jobs. It is
    // implemented by the InnerContext and ReplicateContext to handle
    // standard and control-replicated executions separately.
    class TraceProcessingJobExecutor {
    public:
      virtual RtEvent enqueue_task(
        const InnerContext::AutoTraceProcessRepeatsArgs& args,
        uint64_t opidx,
        bool wait,
        RtEvent precondition = RtEvent::NO_RT_EVENT
      ) = 0;
      virtual RtEvent poll_pending_tasks(
        uint64_t opidx,
        bool must_pop
      ) = 0;
    };

    // TraceIdentifier is a virtual class for implementing trace
    // identification algorithms.
    class TraceIdentifier {
    public:
       TraceIdentifier(
         TraceProcessingJobExecutor* executor,
         TraceOccurrenceWatcher& watcher,
         NonOverlappingAlgorithm repeats_alg,
         uint64_t max_add, // Maximum number of traces to add to the watcher at once.
         uint64_t max_inflight_requests, // Maximum number of async jobs in flight
         bool wait_on_async_job, // Whether to wait on concurrent meta tasks
         uint64_t min_trace_length, // Minimum trace length to identify.
         uint64_t max_trace_length // Maximum trace length to replay.
       );
      void process(Murmur3Hasher::Hash hash, uint64_t opidx);
      virtual ~TraceIdentifier() = default;
      static constexpr Murmur3Hasher::Hash SENTINEL = {};
      enum Algorithm {
        BATCHED = 0,
        MULTI_SCALE = 1,
        NO_ALG = 2,
      };
      static Algorithm parse_algorithm(const std::string& str);
      static const char* algorithm_to_string(Algorithm alg);
    private:
      // maybe_add_trace maybe adds a recorded trace into the
      // TraceOccurrenceWatcher's trie. It returns a boolean as
      // to whether or not a trace was successfully added.
      bool maybe_add_trace(
          const std::vector<Murmur3Hasher::Hash>& hashes,
          uint64_t opidx,
          uint64_t start,
          uint64_t end
      );
    protected:
      // issue_repeats_job performs the logic of actually launching
      // an asynchronous string processing job and updating the internal
      // state of the identifier to reflect the submitted job. Implementers
      // of this method must the jobs_in_flight data structure with the
      // issued job. Implementers of the TraceIdentifier interface should
      // override this method.
      virtual void maybe_issue_repeats_job(uint64_t opidx) = 0;
    protected:
      TraceProcessingJobExecutor* executor;
      std::vector<Murmur3Hasher::Hash> hashes;
      TraceOccurrenceWatcher& watcher;
      NonOverlappingAlgorithm repeats_alg;
      uint64_t max_add;
      uint64_t min_trace_length;
      // max_trace_length is the maximum length trace that we
      // will attempt to replay. We may still _identify_ traces that
      // have length longer than max_trace_length (as those are the actual
      // loops in the source program), but in order to not require buffering
      // too many tasks in the source application, we may want to limit the
      // amount of tasks in traces we replay.
      uint64_t max_trace_length;
      // InFlightProcessingRequest represents a currently executing
      // offline string processing request. When the TraceIdentifier
      // launches a new meta task, it will register it inside the
      // in_flight_requests queue.
      struct InFlightProcessingRequest {
        std::vector<Murmur3Hasher::Hash> hashes;
        RtEvent finish_event;
        // Where the meta task should place the result of
        // the offline computation.
        std::vector<NonOverlappingRepeatsResult> result;
        bool completed = false;
      };
      std::list<InFlightProcessingRequest> jobs_in_flight;
      uint64_t max_in_flight_requests;
      bool wait_on_async_job;
    };

    // BatchedTraceIdentifier batches up operations until a given
    // size is hit, and then computes repeated substrings within
    // the batch of operations.
    class BatchedTraceIdentifier : public TraceIdentifier {
    public:
      BatchedTraceIdentifier(
        uint64_t batchsize,  // Number of operations batched at once.
        // Remaining arguments are for the TraceIdentifier.
        TraceProcessingJobExecutor* executor,
        TraceOccurrenceWatcher& watcher,
        NonOverlappingAlgorithm repeats_alg,
        uint64_t max_add,
        uint64_t max_inflight_requests,
        bool wait_on_async_job,
        uint64_t min_trace_length,
        uint64_t max_trace_length
      );
      ~BatchedTraceIdentifier() override = default;
    protected:
      void maybe_issue_repeats_job(uint64_t opidx) override;
      uint64_t batchsize;
    };

    // MultiScaleBatchedTraceIdentifier implements a multi-scale version
    // of the BatchedTraceIdentifier that uses the ruler function to adaptively
    // issue analysis on increasing pieces of the buffered stream of operations.
    // In particular, for a buffer of size 16 and scale of 1, it would perform
    // analyses at the following intervals:
    // 1 2 1 4 1 2 8 1 2 1 4 1 2 16
    // Different values of scale increase the minimum analysis size. It can be
    // shown that with an O(nlog(n)) string processing algorithm, the total
    // runtime increases with a log(n) factor.
    class MultiScaleBatchedTraceIdentifier : public BatchedTraceIdentifier {
    public:
      MultiScaleBatchedTraceIdentifier(
        uint64_t batchsize,  // Number of operations batched at once.
        uint64_t scale, // Minimum size of the analysis.
        // Remaining arguments are for the TraceIdentifier.
        TraceProcessingJobExecutor* executor,
        TraceOccurrenceWatcher& watcher,
        NonOverlappingAlgorithm repeats_alg,
        uint64_t max_add,
        uint64_t max_inflight_requests,
        bool wait_on_async_job,
        uint64_t min_trace_length,
        uint64_t max_trace_length
      );
      ~MultiScaleBatchedTraceIdentifier() override = default;
    protected:
      void maybe_issue_repeats_job(uint64_t opidx) override;
    private:
      uint64_t scale;
      RtEvent prev_job_completion = RtEvent::NO_RT_EVENT;
    };

    // TraceOccurrenceWatcher tracks how many times inserted traces
    // have occured in the operation stream.
    class TraceOccurrenceWatcher {
    public:
      TraceOccurrenceWatcher(TraceReplayer& replayer, uint64_t visit_threshold);
      void process(Murmur3Hasher::Hash hash, uint64_t opidx);

      template<typename T>
      void insert(T start, T end, uint64_t opidx);
      template<typename T>
      TrieQueryResult query(T start, T end);

      // Clear invalidates all active pointers in the watcher.
      void clear() { this->active_pointers.clear(); }
    private:
      // Reference to a TraceReplayer to dump traces into.
      TraceReplayer& replayer;

      struct TraceMeta {
        // Needs to be default constructable.
        TraceMeta() : opidx(0) { }
        TraceMeta(uint64_t opidx_) : opidx(opidx_) { }
        // The opidx that this trace was inserted at.
        uint64_t opidx;
        // The occurrence watcher will only maintain the number
        // of visits. I don't think that we need to do decaying visits
        // here, though we might want to lower the amount of traces that
        // get committed to the replayer.
        uint64_t visits = 0;
        // completed marks whether this trace has moved
        // from the "watched" state to the "committed" state.
        // Once a trace has been completed, it will not be
        // returned from complete() anymore.
        bool completed = false;
        // The opidx that this trace was previously visited at.
        uint64_t previous_visited_opidx = 0;
      };
      Trie<Murmur3Hasher::Hash, TraceMeta> trie;
      uint64_t visit_threshold;

      // TriePointer maintains an active trace being
      // traversed in the watcher's trie.
      class TriePointer {
      public:
        TriePointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        bool complete();
      public:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        uint64_t opidx;
        uint64_t depth;
      };
      // All currently active pointers that need advancing.
      std::vector<TriePointer> active_pointers;
    };

    // OperationExecutor is a virtual class to be used by the
    // TraceReplayer to shim out internal logic for actually
    // starting and ending traces and issuing operations.
    class OperationExecutor {
    public:
      virtual TraceID get_fresh_trace_id() = 0;
      virtual void issue_begin_trace(TraceID id) = 0;
      virtual void issue_end_trace(TraceID id) = 0;
      virtual bool issue_operation(
        Operation* op,
        const std::vector<StaticDependence>* dependences = NULL,
        bool unordered = false,
        bool outermost = true
      ) = 0;
    };

    // TraceReplayer handles actually buffering and replaying committed traces.
    class TraceReplayer {
    public:
      TraceReplayer(OperationExecutor* executor_)
        : executor(executor_), operation_start_idx(0) { }

      // Enqueue a new operation, which has the given hash.
      void process(
        Operation* op,
        const std::vector<StaticDependence>* dependences,
        Murmur3Hasher::Hash hash,
        uint64_t opidx
      );
      void process_trace_noop(Operation* op);
      // Flush all pending operations out of the TraceReplayer. Accepts
      // the current opidx, used for scoring potentially replayed traces.
      void flush(uint64_t opidx);

      // Insert a new trace into the TraceReplayer.
      template<typename T>
      void insert(T start, T end, uint64_t opidx);
      // See if the chosen string is a prefix of a string contained
      // in the TraceReplayer.
      template<typename T>
      bool prefix(T start, T end);
    private:
      // Indirection layer to issue operations down to the underlying
      // runtime system.
      OperationExecutor* executor;

      struct TraceMeta {
        // TraceMeta's need to be default constructable.
        TraceMeta() {}
        TraceMeta(uint64_t opidx_, uint64_t length_)
          : opidx(opidx_), length(length_), last_visited_opidx(0),
            decaying_visits(0), replays(0),
            last_idempotent_visit_opidx(0),
            decaying_idempotent_visits(0.0), tid(0) { }
        // opidx that this trace was inserted at.
        uint64_t opidx;
        // length of the trace. This is used for scoring only.
        uint64_t length;
        // Fields for maintaining a decaying visit count.
        uint64_t last_visited_opidx;
        double decaying_visits;
        // Number of times the trace has been replayed.
        uint64_t replays;
        // Number of times the trace has been visited in
        // an idempotent manner (tracked in a decaying manner).
        uint64_t last_idempotent_visit_opidx;
        double decaying_idempotent_visits;
        // ID for the trace. It is unset if replays == 0.
        TraceID tid;

        // visit updates the TraceMeta's decaying visit count when visited
        // at opidx.
        void visit(uint64_t opidx);
        // score computes the TraceMeta's score when observed at opidx.
        double score(uint64_t opidx) const;
        // R is the exponential rate of decay for a trace.
        static constexpr double R = 0.99;
        // SCORE_CAP_MULT is the multiplier for how large the score
        // of a particular trace can ever get.
        static constexpr double SCORE_CAP_MULT = 10;
        // REPLAY_SCALE is at most how much a score should be increased
        // to favor replays.
        static constexpr double REPLAY_SCALE = 1.75;
        // IDEMPOTENT_VISIT_SCALE is at most how much a score should
        // be increased to favor idempotent replays.
        static constexpr double IDEMPOTENT_VISIT_SCALE = 2.0;
      };
      Trie<Murmur3Hasher::Hash, TraceMeta> trie;

      // For watching and maintaining decaying visit counts
      // of pointers for scoring.
      class WatchPointer {
      public:
        WatchPointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, uint64_t opidx_)
            : node(node_), opidx(opidx_) { }
        // This pointer only has an advance function, as there's nothing
        // to do on commit.
        bool advance(Murmur3Hasher::Hash token);
        uint64_t get_opidx() const { return this->opidx; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        uint64_t opidx;
      };
      std::vector<WatchPointer> active_watching_pointers;

      // For the actual committed trie.
      class CommitPointer {
      public:
        CommitPointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, uint64_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        void advance_for_trace_noop() { this->depth++; }
        bool complete();
        TraceID replay(OperationExecutor* executor);
        double score(uint64_t opidx);
        uint64_t get_opidx() const { return this->opidx; }
        uint64_t get_length() { return this->depth; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        uint64_t opidx;
        // depth is the number of operations (traceable and trace no-ops)
        // contained within the trace.
        uint64_t depth;
      };
      std::vector<CommitPointer> active_commit_pointers;
      // FrozenCommitPointer is a commit pointer with a frozen score
      // so that it can be maintained in-order inside completed_commit_pointers.
      // We use a separate type here so that CommitPointers do not get
      // accidentally ordered by the metric below.
      class FrozenCommitPointer : public CommitPointer {
      public:
        // We make these sort keys (score, -opidx) so that the highest
        // scoring, earliest opidx is the first entry in the ordering.
        FrozenCommitPointer(CommitPointer& p, uint64_t opidx)
          : CommitPointer(p), score(p.score(opidx), -int64_t(p.get_opidx())) {}
        friend bool operator<(const FrozenCommitPointer& a, const FrozenCommitPointer& b) {
          // Use > instead of < so that we get descending order.
          return a.score > b.score;
        }
      private:
        std::pair<double, int64_t> score;
      };
      // completed_commit_pointers is a _sorted_ vector of completed
      // commit pointers. All operations on it must preserve the sortedness.
      std::vector<FrozenCommitPointer> completed_commit_pointers;

      // Fields for the management of pending operations.
      struct PendingOperation {
        PendingOperation(Operation* op, const std::vector<StaticDependence>* deps)
          : operation(op), dependences(deps) { }
        Operation* operation;
        const std::vector<StaticDependence>* dependences;
      };
      std::queue<PendingOperation> operations;
      uint64_t operation_start_idx;

      // flush_buffer executes operations until opidx, or flushes
      // the entire operation buffer if no opidx is provided.
      void flush_buffer();
      void flush_buffer(uint64_t opidx);
      // replay_trace executes operations under the trace tid
      // until opidx, after which it inserts an end trace.
      void replay_trace(uint64_t opidx, TraceID tid);
    };

    template <typename T>
    class AutomaticTracingContext : public T,
                                    public OperationExecutor,
                                    public TraceProcessingJobExecutor {
    public:
      template <typename ... Args>
      AutomaticTracingContext(Args&& ... args)
        : T(std::forward<Args>(args) ... ),
          opidx(0),
          identifier(),
          watcher(this->replayer, this->runtime->auto_trace_commit_threshold),
          replayer(this)
        {
          switch (TraceIdentifier::Algorithm(this->runtime->auto_trace_identifier_alg)) {
            case TraceIdentifier::Algorithm::BATCHED: {
              identifier = std::unique_ptr<TraceIdentifier>(new BatchedTraceIdentifier(
                this->runtime->auto_trace_batchsize,
                this,
                this->watcher,
                this->runtime->auto_trace_repeats_alg,
                this->runtime->auto_trace_max_start_watch,
                this->runtime->auto_trace_in_flight_jobs,
                this->runtime->auto_trace_wait_async_jobs,
                this->runtime->auto_trace_min_trace_length,
                this->runtime->auto_trace_max_trace_length
              ));
              break;
            }
            case TraceIdentifier::Algorithm::MULTI_SCALE: {
              identifier = std::unique_ptr<TraceIdentifier>(new MultiScaleBatchedTraceIdentifier(
                  this->runtime->auto_trace_batchsize,
                  this->runtime->auto_trace_multi_scale_factor,
                  this,
                  this->watcher,
                  this->runtime->auto_trace_repeats_alg,
                  this->runtime->auto_trace_max_start_watch,
                  this->runtime->auto_trace_in_flight_jobs,
                  this->runtime->auto_trace_wait_async_jobs,
                  this->runtime->auto_trace_min_trace_length,
                  this->runtime->auto_trace_max_trace_length
              ));
              break;
            }
            default:
              assert(false);
          }
          // Perform any initialization for async trace analysis needed.
          T::initialize_async_trace_analysis(this->runtime->auto_trace_in_flight_jobs);
        }
        virtual ~AutomaticTracingContext() {
          // Report some statistics about the efficiency of the auto-tracer if
          // we've seen enough operations.
          if (this->executed_ops >= this->runtime->auto_trace_batchsize) {
            double pct = double(this->traced_ops) / double(this->executed_ops);
            log_auto_trace.info() << "Traced " << this->traced_ops << "/"
                                  << this->executed_ops << " = "
                                  << (100.0 * pct) << " percent.";
          } else {
            log_auto_trace.info() << "Context didn't execute enough operations.";
          }
        }
    public:
      bool add_to_dependence_queue(Operation *op,
                                   const std::vector<StaticDependence>* dependences = NULL,
                                   bool unordered = false,
                                   bool outermost = true) override;
      // get_new_unique_hash() generates a hash value that
      // will not repeat. This is used to represent operations
      // or events that are not traceable, so that the trace
      // identification analysis does not identify repeats that
      // cross over untraceable operations.
      Murmur3Hasher::Hash get_new_unique_hash();
      // If the application performs a blocking operation, we need to know
      // about that, so override TaskContext::record_blocking_call().
      void record_blocking_call(uint64_t future_coordinate) override;
    public:
      // Overrides for OperationExecutor.
      TraceID get_fresh_trace_id() override;
      void issue_begin_trace(TraceID id) override;
      void issue_end_trace(TraceID id) override;
      bool issue_operation(
        Operation* op,
        const std::vector<StaticDependence>* dependences = NULL,
        bool unordered = false,
        bool outermost = false
      ) override;
    public:
      // Overrides for TraceJobProcessingExecutor.
      RtEvent enqueue_task(
        const InnerContext::AutoTraceProcessRepeatsArgs& args,
        uint64_t opidx,
        bool wait,
        RtEvent precondition = RtEvent::NO_RT_EVENT
      ) override;
      RtEvent poll_pending_tasks(uint64_t opidx, bool must_pop) override;
    private:
      uint64_t opidx;
      std::unique_ptr<TraceIdentifier> identifier = nullptr;
      TraceOccurrenceWatcher watcher;
      TraceReplayer replayer;
      // unique_hash_idx_counter maintains a counter of non-traceable
      // operations seen so far, used to generate unique hashes for
      // those operations.
      uint64_t unique_hash_idx_counter = 0;
      // Maintain whether or not we are currently replaying a trace.
      bool started_auto_trace = false;
      // Counters for statistics about tracing efficiency.
      uint64_t traced_ops = 0;
      uint64_t executed_ops = 0;
    };

    void auto_trace_process_repeats(const void* args);

    // Utility functions.
    bool is_operation_traceable(Operation* op);
    bool is_operation_ignorable_in_traces(Operation* op);


    // TODO (rohany): Can we move these declarations to another file?

    template <typename T>
    bool AutomaticTracingContext<T>::add_to_dependence_queue(
      Operation* op,
      const std::vector<StaticDependence>* dependences,
      bool unordered,
      bool outermost
    ) {
      // If we have an unordered operation, just forward it directly without
      // getting it involved in the tracing infrastructure.
      if (unordered) {
        return this->issue_operation(op, dependences, unordered, outermost);
      }

      // TODO (rohany): In the future, if we allow automatic and explicit tracing
      //  in the same program, we're going to need to be able to know whether
      //  trace operations have been issued by the AutomaticTracingContext
      //  or the application. It's a bit of plumbing right now to get this
      //  through to all of the applications, so we'll start with the ones
      //  where it's present, and then disallow trace operations to be issued
      //   by the application.
      // Trace operations that we recognize go straight through the context into
      // the dependence queue, as we're only going to see them as a callback
      // that we issue while we're already in the dependence queue. This because
      // the callback structure is
      // AutoTracingContext::add_to_dependence_queue ->
      // AutoTracingContext::replay_trace ->
      // InnerContext::begin_trace ->
      // AutoTracingContext::add_to_dependence_queue -> HERE.
      switch (op->get_operation_kind()) {
        case Operation::OpKind::TRACE_BEGIN_OP_KIND: {
          assert(op->get_trace()->tid >= LEGION_MAX_APPLICATION_TRACE_ID && op->get_trace()->tid < LEGION_INITIAL_LIBRARY_ID_OFFSET);
          assert(dependences == NULL);
          return this->issue_operation(op);
        }
        case Operation::OpKind::TRACE_RECURRENT_OP_KIND: // Fallthrough.
        case Operation::OpKind::TRACE_COMPLETE_OP_KIND: {
          assert(dependences == NULL);
          return this->issue_operation(op);
        }
        default: {
          break;
        }
      }

      // If we encounter a traceable operation, then it's time to start
      // analyzing it and adding it the corresponding operation processors.
      if (is_operation_traceable(op)) {
        Murmur3Hasher::Hash hash = TraceHashHelper{}.hash(op);
        // TODO (rohany): Have to have a hash value that can be used as the sentinel $
        //  token for the suffix tree processing algorithms.
        assert(!(hash.x == 0 && hash.y == 0));
        this->identifier->process(hash, this->opidx);
        this->watcher.process(hash, this->opidx);
        this->replayer.process(op, dependences, hash, this->opidx);
        this->opidx++;
        return true;
      } else if (is_operation_ignorable_in_traces(op)) {
        // If the operation we are processing is "ignorable" in traces
        // then we won't consider it for trace identification or counting
        // watches in the identifier and watcher. The idea here is to thread
        // these operations through the pipeline unless we are replaying a
        // trace, in which case they will be dropped.
        assert(dependences == NULL);
        this->replayer.process_trace_noop(op);
        this->opidx++;
        return true;
      } else {
        // When encountering a non-traceable operation, insert a
        // dummy hash value into the trace identifier so that the
        // traces it finds don't span across these operations.
        this->identifier->process(this->get_new_unique_hash(), this->opidx);

        // When encountering a non-traceable operation, invalidate
        // all active pointers from the TraceOccurrenceWatcher, as
        // this operation has broken any active traces.
        this->watcher.clear();

        // If we see a non-traceable operation, then we need to flush
        // all of the pending operations sitting in the replayer (as
        // a trace is no longer possible to replay) before issuing
        // the un-traceable operation.
        this->replayer.flush(this->opidx);
        log_auto_trace.debug() << "Encountered untraceable operation: "
                               << Operation::get_string_rep(op->get_operation_kind());
        return this->issue_operation(op, dependences, unordered, outermost);
      }
    }


    template <typename T>
    TraceID AutomaticTracingContext<T>::get_fresh_trace_id() {
      return this->generate_dynamic_trace_id();
    }

    template <typename T>
    void AutomaticTracingContext<T>::issue_begin_trace(Legion::TraceID id) {
      T::begin_trace(
        id,
        false /* logical_only */,
        false /* static_trace */,
        nullptr /* managed */,
        false /* dep */,
        nullptr /* provenance */,
        false /* from application */
      );
      this->started_auto_trace = true;
    }

    template <typename T>
    void AutomaticTracingContext<T>::issue_end_trace(Legion::TraceID id) {
      this->started_auto_trace = false;
      T::end_trace(id, false /* deprecated */, nullptr /* provenance */, false /* from application */);
    }

    template <typename T>
    bool AutomaticTracingContext<T>::issue_operation(
      Legion::Internal::Operation *op,
      const std::vector<StaticDependence>* dependences,
      bool unordered,
      bool outermost
    ) {
      // If we're tracing and this operation is a trace no-op, then no-op.
      if (this->started_auto_trace && is_operation_ignorable_in_traces(op) && !unordered) {
        return true;
      }
      // Update counters for issued operations.
      if (!unordered) {
        // Only count operations that can be traced (i.e. random discards
        // don't count toward the trace success rate).
        if (!is_operation_ignorable_in_traces(op)) this->executed_ops++;
        if (this->started_auto_trace) this->traced_ops++;
      }
      return T::add_to_dependence_queue(op, dependences, unordered, outermost);
    }

    template <typename T>
    void AutomaticTracingContext<T>::record_blocking_call(uint64_t future_coordinate) {
      if (future_coordinate != InnerContext::NO_FUTURE_COORDINATE) {
        // Handling waits from the application is very similar
        // to the case in add_to_dependence_queue when we encounter an
        // operation that is not traceable. We interrupt traces in
        // the identifier, and flush the watcher and replayer. We identify
        // whether a wait is coming from the application by seeing if the
        // future being waited on has a valid coordinate.
        // TODO (rohany): I think that this is a little busted right now for
        //  inline mappings.
        this->identifier->process(this->get_new_unique_hash(), this->opidx);
        this->watcher.clear();
        this->replayer.flush(this->opidx);
      }
      // Need to also do whatever the base context was going to do.
      T::record_blocking_call(future_coordinate);
    }

    template <typename T>
    Murmur3Hasher::Hash AutomaticTracingContext<T>::get_new_unique_hash() {
      uint64_t idx = this->unique_hash_idx_counter;
      this->unique_hash_idx_counter++;
      Murmur3Hasher hasher;
      hasher.hash(Operation::OpKind::LAST_OP_KIND);
      hasher.hash(idx);
      Murmur3Hasher::Hash result;
      hasher.finalize(result);
      return result;
    }

    template <typename T>
    RtEvent AutomaticTracingContext<T>::enqueue_task(
        const InnerContext::AutoTraceProcessRepeatsArgs& args,
        uint64_t opidx,
        bool wait,
        RtEvent precondition
    ) {
      return T::enqueue_trace_analysis_meta_task(args, opidx, wait, precondition);
    }

    template <typename T>
    RtEvent AutomaticTracingContext<T>::poll_pending_tasks(
        uint64_t opidx,
        bool must_pop
    ) {
      return T::poll_pending_trace_analysis_tasks(opidx, must_pop);
    }

    template <typename T>
    void TraceOccurrenceWatcher::insert(T start, T end, uint64_t opidx) {
      this->trie.insert(start, end, TraceMeta(opidx));
    }

    template <typename T>
    TrieQueryResult TraceOccurrenceWatcher::query(T start, T end) {
      return this->trie.query(start, end);
    }

    template <typename T>
    void TraceReplayer::insert(T start, T end, uint64_t opidx) {
      return this->trie.insert(start, end, TraceMeta(opidx, std::distance(start, end)));
    }

    template <typename T>
    bool TraceReplayer::prefix(T start, T end) {
      return this->trie.prefix(start, end);
    }
  };
};

#endif // __LEGION_AUTO_TRACE_H__
