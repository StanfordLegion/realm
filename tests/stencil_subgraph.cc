#include "realm.h"
#include "stencil_subgraph.h"

#include <iomanip>
#include <iostream>
#include <map>

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  STENCIL_TASK,
  INCREMENT_TASK,
  VERIFY_TASK,
};

struct VerifyArgs {
  RegionInstance golden = RegionInstance::NO_INST;
  RegionInstance local_buffer = RegionInstance::NO_INST;
  IndexSpace<2> local_space = IndexSpace<2>();
};

struct TLTArgs {
  int64_t nx = 100;
  int64_t ny = 100;
  int64_t px = 2;
  int64_t py = 2;
  int64_t steps = 50;
  bool use_subgraph = false;
  bool verify = true;
};

void stencil_task(const void *_args, size_t arglen,
                  const void *userdata, size_t userlen, Processor p) {
  StencilArgs* args = (StencilArgs*)(_args);
  AffineAccessor<float, 2> finput_acc(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> foutput_acc(args->buffer, FID_OUTPUT);

  // 5-point stencil.
  auto bounds = args->local_space.bounds;
  for (int64_t i = bounds.lo[0]; i <= bounds.hi[0]; i++) {
    for (int64_t j = bounds.lo[1]; j <= bounds.hi[1]; j++) {
      float center = finput_acc[{i, j}];
      float north = i > 0 ? finput_acc[{i - 1, j}] : 0.0f;
      float south = i < args->hx - 1 ? finput_acc[{i + 1, j}] : 0.0f;
      float west = j > 0 ? finput_acc[{i, j - 1}] : 0.0f;
      float east = j < args->hy -1 ? finput_acc[{i, j + 1}] : 0.0f;
      foutput_acc[{i, j}] = (center + north + south + west + east) / 5.f;
    }
  }
}

void increment_task(const void *_args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p) {
  IncrementArgs* args = (IncrementArgs*)(_args);
  AffineAccessor<float, 2> finput_acc(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> foutput_acc(args->buffer, FID_OUTPUT);
  // Increment copies from output to input, and adds 1 to each element.
  auto bounds = args->local_space.bounds;
  for (int64_t i = bounds.lo[0]; i <= bounds.hi[0]; i++) {
    for (int64_t j = bounds.lo[1]; j <= bounds.hi[1]; j++) {
      finput_acc[{i, j}] = foutput_acc[{i, j}] + 1.0f;
    }
  }
}

void verify_task(const void *_args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p) {
  VerifyArgs* args = (VerifyArgs*)(_args);
  AffineAccessor<float, 2> input(args->local_buffer, FID_INPUT);
  AffineAccessor<float, 2> golden(args->golden, FID_INPUT);
  auto bounds = args->local_space.bounds;
  for (int64_t i = bounds.lo[0]; i <= bounds.hi[0]; i++) {
    for (int64_t j = bounds.lo[1]; j <= bounds.hi[1]; j++) {
      if (input[{i, j}] != golden[{i, j}]) {
        std::cout << "(" << i << "," << j << ") found: " << std::fixed << std::setprecision(8) << input[{i, j}] << " expected: " << golden[{i, j}] << std::endl;
      }
      assert((input[{i, j}] == golden[{i, j}]));
    }
  }
}

void run_stencil_direct(
  int64_t nx, int64_t ny,
  int64_t px, int64_t py, int64_t steps,
  std::map<std::pair<int64_t, int64_t>, Processor> procs,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> north_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> south_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> west_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> east_spaces,
  std::map<std::pair<int64_t, int64_t>, RegionInstance> instances,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> local_spaces
) {
  std::map<std::pair<int64_t, int64_t>, Event> task_postconds;
  std::map<std::pair<int64_t, int64_t>, Event> incoming_north;
  std::map<std::pair<int64_t, int64_t>, Event> incoming_south;
  std::map<std::pair<int64_t, int64_t>, Event> incoming_east;
  std::map<std::pair<int64_t, int64_t>, Event> incoming_west;

  for (int64_t i = 0; i < px; i++) {
    for (int64_t j = 0; j < py; j++) {
      task_postconds[{i, j}] = Event::NO_EVENT;
      incoming_north[{i, j}] = Event::NO_EVENT;
      incoming_south[{i, j}] = Event::NO_EVENT;
      incoming_east[{i, j}] = Event::NO_EVENT;
      incoming_west[{i, j}] = Event::NO_EVENT;
    }
  }

  for (int64_t step = 0; step < steps; step++) {
    // Before launching tasks issue copies from each neighbor's
    // local space into ours. Technically we don't need to do this
    // on the first iteration, but it also doesn't hurt anything.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        // For each direction, copy from remote input into local input.
        std::vector<CopySrcDstField> src(1), dst(1);
        Event cur = task_postconds[{i, j}];
        dst[0].set_field(instances[{i, j}], FID_INPUT, sizeof(float));
        if (i > 0) {
          src[0].set_field(instances[{i - 1, j}], FID_INPUT, sizeof(float));
          incoming_north[{i, j}] = north_spaces[{i, j}].copy(src, dst, ProfilingRequestSet(), Event::merge_events(cur, task_postconds[{i - 1, j}]));
        }
        if (i < px - 1) {
          src[0].set_field(instances[{i + 1, j}], FID_INPUT, sizeof(float));
          incoming_south[{i, j}] = south_spaces[{i, j}].copy(src, dst, ProfilingRequestSet(), Event::merge_events(cur, task_postconds[{i + 1, j}]));
        }
        if (j > 0) {
          src[0].set_field(instances[{i, j - 1}], FID_INPUT, sizeof(float));
          incoming_west[{i, j}] = west_spaces[{i, j}].copy(src, dst, ProfilingRequestSet(), Event::merge_events(cur, task_postconds[{i, j - 1}]));
        }
        if (j < py - 1) {
          src[0].set_field(instances[{i, j + 1}], FID_INPUT, sizeof(float));
          incoming_east[{i, j}] = east_spaces[{i, j}].copy(src, dst, ProfilingRequestSet(), Event::merge_events(cur, task_postconds[{i, j + 1}]));
        }
      }
    }

    std::vector<Event> deps;
    // Launch stencil tasks.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        deps.push_back(task_postconds[{i, j}]);
        deps.push_back(incoming_north[{i, j}]);
        deps.push_back(incoming_south[{i, j}]);
        deps.push_back(incoming_east[{i, j}]);
        deps.push_back(incoming_west[{i, j}]);
        Event pred = Event::merge_events(deps);
        StencilArgs args;
        args.buffer = instances[{i, j}];
        args.local_space = local_spaces[{i, j}];
        args.hx = nx;
        args.hy = ny;
        task_postconds[{i, j}] = procs[{i, j}].spawn(STENCIL_TASK, &args, sizeof(StencilArgs), pred);

        deps.clear();
      }
    }

    // Launch increment tasks.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
	// Make sure all outgoing copies are done before starting the task.
	deps.push_back(task_postconds[{i, j}]);
        if (i > 0) {
	  deps.push_back(incoming_south[{i - 1, j}]);
        }
        if (i < px - 1) {
	  deps.push_back(incoming_north[{i + 1, j}]);
        }
        if (j > 0) {
	  deps.push_back(incoming_east[{i, j - 1}]);
        }
        if (j < py - 1) {
	  deps.push_back(incoming_west[{i, j + 1}]);
        }

	Event pred = Event::merge_events(deps);
        IncrementArgs args;
        args.buffer = instances[{i, j}];
        args.local_space = local_spaces[{i, j}];
        task_postconds[{i, j}] = procs[{i, j}].spawn(INCREMENT_TASK, &args, sizeof(IncrementArgs), pred);

	deps.clear();
      }
    }
  }

  // Wait for all increment tasks to finish.
  std::vector<Event> events;
  events.reserve(task_postconds.size());
  for (auto& e : task_postconds) {
    events.push_back(e.second);
  }
  Event::merge_events(events).wait();
}

Subgraph compile_stencil(
  int64_t nx, int64_t ny,
  int64_t px, int64_t py, int64_t steps,
  int64_t subgraph_steps,
  std::map<std::pair<int64_t, int64_t>, Processor> procs,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> north_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> south_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> west_spaces,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> east_spaces,
  std::map<std::pair<int64_t, int64_t>, RegionInstance> instances,
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> local_spaces
) {
  SubgraphDefinition sd;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> stencil_tasks;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> increment_tasks;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> north_copies;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> south_copies;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> west_copies;
  std::map<std::tuple<int64_t, int64_t, int64_t>, unsigned> east_copies;

  // Add all tasks and copies.
  for (int64_t step = 0; step < subgraph_steps; step++) {
    // Copies.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        std::vector<CopySrcDstField> src(1), dst(1);
        dst[0].set_field(instances[{i, j}], FID_INPUT, sizeof(float));
        if (i > 0) {
          src[0].set_field(instances[{i - 1, j}], FID_INPUT, sizeof(float));
          SubgraphDefinition::CopyDesc copy;
          copy.space = north_spaces[{i, j}];
          copy.srcs = src;
          copy.dsts = dst;
          north_copies[{i, j, step}] = sd.copies.size();
          sd.copies.push_back(copy);
        }
        if (i < px - 1) {
          src[0].set_field(instances[{i + 1, j}], FID_INPUT, sizeof(float));
          SubgraphDefinition::CopyDesc copy;
          copy.space = south_spaces[{i, j}];
          copy.srcs = src;
          copy.dsts = dst;
          south_copies[{i, j, step}] = sd.copies.size();
          sd.copies.push_back(copy);
        }
        if (j > 0) {
          src[0].set_field(instances[{i, j - 1}], FID_INPUT, sizeof(float));
          SubgraphDefinition::CopyDesc copy;
          copy.space = west_spaces[{i, j}];
          copy.srcs = src;
          copy.dsts = dst;
          west_copies[{i, j, step}] = sd.copies.size();
          sd.copies.push_back(copy);
        }
        if (j < py - 1) {
          src[0].set_field(instances[{i, j + 1}], FID_INPUT, sizeof(float));
          SubgraphDefinition::CopyDesc copy;
          copy.space = east_spaces[{i, j}];
          copy.srcs = src;
          copy.dsts = dst;
          east_copies[{i, j, step}] = sd.copies.size();
          sd.copies.push_back(copy);
        }
      }
    }

    // Stencil tasks.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        SubgraphDefinition::TaskDesc task;
        task.proc = procs[{i, j}];
        task.task_id = STENCIL_TASK;
        StencilArgs args;
        args.buffer = instances[{i, j}];
        args.local_space = local_spaces[{i, j}];
        args.hx = nx;
        args.hy = ny;
        task.args = ByteArray(&args, sizeof(StencilArgs));
        stencil_tasks[{i, j, step}] = sd.tasks.size();
        sd.tasks.push_back(task);
      }
    }

    // Increment tasks.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        SubgraphDefinition::TaskDesc task;
        task.proc = procs[{i, j}];
        task.task_id = INCREMENT_TASK;
        IncrementArgs args;
        args.buffer = instances[{i, j}];
        args.local_space = local_spaces[{i, j}];
        task.args = ByteArray(&args, sizeof(IncrementArgs));
        increment_tasks[{i, j, step}] = sd.tasks.size();
        sd.tasks.push_back(task);
      }
    }
  }

  // Now add the dependencies.
  for (int64_t step = 0; step < subgraph_steps; step++) {
    // Copies have preconditions if they are after the first step.
    if (step > 0) {
      for (int64_t i = 0; i < px; i++) {
        for (int64_t j = 0; j < py; j++) {
          if (i > 0) {
            SubgraphDefinition::Dependency d1;
            d1.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d1.src_op_index = increment_tasks[{i, j, step - 1}];
            d1.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d1.tgt_op_index = north_copies[{i, j, step}];
            sd.dependencies.push_back(d1);
            SubgraphDefinition::Dependency d2;
            d2.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d2.src_op_index = increment_tasks[{i - 1, j, step - 1}];
            d2.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d2.tgt_op_index = north_copies[{i, j, step}];
            sd.dependencies.push_back(d2);
          }
          if (i < px - 1) {
            SubgraphDefinition::Dependency d1;
            d1.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d1.src_op_index = increment_tasks[{i, j, step - 1}];
            d1.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d1.tgt_op_index = south_copies[{i, j, step}];
            sd.dependencies.push_back(d1);
            SubgraphDefinition::Dependency d2;
            d2.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d2.src_op_index = increment_tasks[{i + 1, j, step - 1}];
            d2.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d2.tgt_op_index = south_copies[{i, j, step}];
            sd.dependencies.push_back(d2);
          }
          if (j > 0) {
            SubgraphDefinition::Dependency d1;
            d1.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d1.src_op_index = increment_tasks[{i, j, step - 1}];
            d1.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d1.tgt_op_index = west_copies[{i, j, step}];
            sd.dependencies.push_back(d1);
            SubgraphDefinition::Dependency d2;
            d2.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d2.src_op_index = increment_tasks[{i, j - 1, step - 1}];
            d2.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d2.tgt_op_index = west_copies[{i, j, step}];
            sd.dependencies.push_back(d2);
          }
          if (j < py - 1) {
            SubgraphDefinition::Dependency d1;
            d1.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d1.src_op_index = increment_tasks[{i, j, step - 1}];
            d1.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d1.tgt_op_index = east_copies[{i, j, step}];
            sd.dependencies.push_back(d1);
            SubgraphDefinition::Dependency d2;
            d2.src_op_kind = SubgraphDefinition::OPKIND_TASK;
            d2.src_op_index = increment_tasks[{i, j + 1, step - 1}];
            d2.tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
            d2.tgt_op_index = east_copies[{i, j, step}];
            sd.dependencies.push_back(d2);
          }
        }
      }
    }

    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        // Depend on the previous task at this point.
        if (step > 0) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.src_op_index = increment_tasks[{i, j, step - 1}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = stencil_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        // Depend on the stencil neighbors.
        if (i > 0) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = north_copies[{i, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = stencil_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (i < px - 1) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = south_copies[{i, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = stencil_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (j > 0) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = west_copies[{i, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = stencil_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (j < py - 1) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = east_copies[{i, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = stencil_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
      }
    }

    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
	{
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.src_op_index = stencil_tasks[{i, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = increment_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
	}
	// Anti-dependencies.
        if (i > 0) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = south_copies[{i - 1, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = increment_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (i < px - 1) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = north_copies[{i + 1, j, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = increment_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (j > 0) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = east_copies[{i, j - 1, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = increment_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
        if (j < py - 1) {
          SubgraphDefinition::Dependency d;
          d.src_op_kind = SubgraphDefinition::OPKIND_COPY;
          d.src_op_index = west_copies[{i, j + 1, step}];
          d.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          d.tgt_op_index = increment_tasks[{i, j, step}];
          sd.dependencies.push_back(d);
        }
      }
    }
  }
  sd.concurrency_mode = Realm::SubgraphDefinition::INSTANTIATION_ORDER;
  Subgraph stencil_subgraph;
  Subgraph::create_subgraph(stencil_subgraph, sd, ProfilingRequestSet()).wait();
  return stencil_subgraph;
}

void run_stencil_subgraph(int64_t steps, int64_t subgraph_steps, Subgraph& subgraph) {
  Event e = Event::NO_EVENT;
  for (int64_t step = 0; step < steps; step++) {
    if (step % subgraph_steps == 0) {
      e = subgraph.instantiate(nullptr, 0, ProfilingRequestSet(), SubgraphInstantiationProfilingRequestsDesc(), e);
    }
  }
  e.wait();
}

void top_level_task(const void *_args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p) {
  log_app.print() << "Realm subgraphs test";
  auto pq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  std::vector<Processor> cpus(pq.begin(), pq.end());
  auto gq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::TOC_PROC);
  std::vector<Processor> gpus(gq.begin(), gq.end());
  auto mq = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM).has_capacity(1);
  std::vector<Memory> sysmems(mq.begin(), mq.end());


  TLTArgs* args = (TLTArgs*)(_args);
  int64_t nx = args->nx;
  int64_t ny = args->ny;
  int64_t px = args->px;
  int64_t py = args->py;
  int64_t steps = args->steps;

  std::map<std::pair<int64_t, int64_t>, Processor> procs;
  std::map<std::pair<int64_t, int64_t>, Memory> mems;

  // TODO (rohany): This will change for GPUs ...
  // Construct a processor and memory mapping.
  for (int64_t i = 0; i < px; i++) {
    for (int64_t j = 0; j < py; j++) {
      if (gpus.empty()) {
        assert(px * py <= cpus.size());
        procs[{i, j}] = cpus[(i * py + j) % cpus.size()];
        mems[{i, j}] = sysmems[(i * py + j) % sysmems.size()];
      } else {
        procs[{i, j}] = gpus[(i * py + j) % gpus.size()];
        Machine::MemoryQuery query(Machine::get_machine());
        query.only_kind(Memory::GPU_FB_MEM);
        query.local_address_space();
        query.best_affinity_to(procs[{i, j}]);
        assert(query.count() == 1);
        mems[{i, j}] = query.first();
      }
    }
  }

  assert(nx % px == 0);
  assert(ny % py == 0);

  // Break the grid into pieces.
  int64_t lxsize = nx / px, lysize = ny / py;
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> local_spaces;
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> bloated_spaces;
  for (int64_t i = 0; i < px; i++) {
    for (int64_t j = 0; j < py; j++) {
      int64_t lx = i * lxsize;
      int64_t hx = (i + 1) * lxsize;
      int64_t ly = j * lysize;
      int64_t hy = (j + 1) * lysize;
      local_spaces[{i, j}] = Rect<2>({lx, ly}, {hx - 1, hy - 1});
      // Construct the bloated space by bloating all the dimensions
      // unless they are the boundary.
      bloated_spaces[{i, j}] = Rect<2>({i == 0 ? lx : lx - 1, j == 0 ? ly : ly - 1}, {i == px - 1 ? hx - 1 : hx, j == py - 1 ? hy - 1 : hy});
    }
  }

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_INPUT] = sizeof(float);
  field_sizes[FID_OUTPUT] = sizeof(float);

  // Construct all bloated spaces.
  std::map<std::pair<int64_t, int64_t>, RegionInstance> instances;
  {
    Event e = Event::NO_EVENT;
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        auto& inst = instances[{i, j}];
        e = RegionInstance::create_instance(inst, mems[{i, j}], bloated_spaces[{i, j}], field_sizes, 0 /* SOA */, ProfilingRequestSet());
        // Fill it with 0's.
        std::vector<CopySrcDstField> info(1);
        info[0].set_field(inst, FID_INPUT, sizeof(float));
        float value = 0;
        e = local_spaces[{i, j}].fill(info, ProfilingRequestSet(), &value, sizeof(float), e);
      }
    }
    e.wait();
  }

  // Construct index spaces for the copies in each direction.
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> north_spaces;
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> south_spaces;
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> east_spaces;
  std::map<std::pair<int64_t, int64_t>, IndexSpace<2>> west_spaces;
  {
    Event e = Event::NO_EVENT;
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        if (i > 0) {
          e = IndexSpace<2>::compute_intersection(bloated_spaces[{i, j}], local_spaces[{i - 1, j}], north_spaces[{i, j}], ProfilingRequestSet(), e);
        }
        if (j > 0) {
          e = IndexSpace<2>::compute_intersection(bloated_spaces[{i, j}], local_spaces[{i, j - 1}], west_spaces[{i, j}], ProfilingRequestSet(), e);
        }
        if (i < px - 1) {
          e = IndexSpace<2>::compute_intersection(bloated_spaces[{i, j}], local_spaces[{i + 1, j}], south_spaces[{i, j}], ProfilingRequestSet(), e);
        }
        if (j < py - 1) {
          e = IndexSpace<2>::compute_intersection(bloated_spaces[{i, j}], local_spaces[{i, j + 1}], east_spaces[{i, j}], ProfilingRequestSet(), e);
        }
      }
    }
    e.wait();
  }

  uint64_t us_start, us_end;
  // For each benchmark, run twice and disregard the first to warm up the CUDA driver.
  if (args->use_subgraph) {
    log_app.print() << "Using subgraph!";
    // Now, define the stencil as a subgraph.
    int64_t subgraph_steps = 10;
    assert(steps % subgraph_steps == 0);
    Subgraph stencil_subgraph = compile_stencil(nx, ny, px, py, steps, subgraph_steps, procs, north_spaces, south_spaces, west_spaces, east_spaces, instances, local_spaces);
    run_stencil_subgraph(steps, subgraph_steps, stencil_subgraph);

    // Instantiate the subgraph the correct number of times.
    us_start = Clock::current_time_in_microseconds();
    run_stencil_subgraph(steps, subgraph_steps, stencil_subgraph);
    us_end = Clock::current_time_in_microseconds();
  } else {
    log_app.print() << "Direct realm launch.";
    run_stencil_direct(nx, ny, px, py, steps, procs, north_spaces, south_spaces, west_spaces, east_spaces, instances, local_spaces);
    us_start = Clock::current_time_in_microseconds();
    run_stencil_direct(nx, ny, px, py, steps, procs, north_spaces, south_spaces, west_spaces, east_spaces, instances, local_spaces);
    us_end = Clock::current_time_in_microseconds();
  }

  log_app.print() << "Took: " << (us_end - us_start) << " microseconds.";


  // Now, check that the computation was correct.
  if (args->verify) {
    IndexSpace<2> full = Rect<2>({0, 0}, {nx - 1, ny - 1});
    RegionInstance golden, computed;
    RegionInstance::create_instance(golden, sysmems[0], full, field_sizes, 0 /* SOA */, ProfilingRequestSet()).wait();
    RegionInstance::create_instance(computed, sysmems[0], full, field_sizes, 0 /* SOA */, ProfilingRequestSet()).wait();
    {
      std::vector<CopySrcDstField> info(1);
      info[0].set_field(golden, FID_INPUT, sizeof(float));
      float value = 0;
      full.fill(info, ProfilingRequestSet(), &value, sizeof(float)).wait();
    }
    {
      AffineAccessor<float, 2> input(golden, FID_INPUT);
      AffineAccessor<float, 2> output(golden, FID_OUTPUT);
      for (int64_t step = 0; step < steps * 2; step++) {
        for (int64_t i = 0; i < nx; i++) {
          for (int64_t j = 0; j < ny; j++) {
            float center = input[{i, j}];
            float north = i > 0 ? input[{i - 1, j}] : 0.0f;
            float south = i < nx - 1 ? input[{i + 1, j}] : 0.0f;
            float west = j > 0 ? input[{i, j - 1}] : 0.0f;
            float east = j < ny - 1 ? input[{i, j + 1}] : 0.0f;
            output[{i, j}] = (center + north + south + west + east) / 5.f;
          }
        }

        for (int64_t i = 0; i < nx; i++) {
          for (int64_t j = 0; j < ny; j++) {
            input[{i, j}] = output[{i, j}] + 1.0f;
          }
        }
      }
    }

    // Copy each of the pieces into a CPU instance.
    for (int64_t i = 0; i < px; i++) {
      for (int64_t j = 0; j < py; j++) {
        std::vector<CopySrcDstField> src(1), dst(1);
        src[0].set_field(instances[{i, j}], FID_INPUT, sizeof(float));
        dst[0].set_field(computed, FID_INPUT, sizeof(float));
        local_spaces[{i, j}].copy(src, dst, ProfilingRequestSet()).wait();
      }
    }

    VerifyArgs vargs;
    vargs.local_buffer = computed;
    vargs.golden = golden;
    vargs.local_space = full;
    p.spawn(VERIFY_TASK, &vargs, sizeof(VerifyArgs)).wait();
  }

  log_app.print() << "Success!";

  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  TLTArgs args;

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-nx")) {
      args.nx = strtoll(argv[++i], 0, 10);
      continue;
    }
    if(!strcmp(argv[i], "-ny")) {
      args.ny = strtoll(argv[++i], 0, 10);
      continue;
    }
    if(!strcmp(argv[i], "-px")) {
      args.px = strtoll(argv[++i], 0, 10);
      continue;
    }
    if(!strcmp(argv[i], "-py")) {
      args.py = strtoll(argv[++i], 0, 10);
      continue;
    }
    if(!strcmp(argv[i], "-steps")) {
      args.steps = strtoll(argv[++i], 0, 10);
      continue;
    }
    if(!strcmp(argv[i], "-subgraph")) {
      args.use_subgraph = true;
      continue;
    }
    if(!strcmp(argv[i], "-no-check")) {
      args.verify = false;
      continue;
    }
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(VERIFY_TASK, verify_task);

  Processor::register_task_by_kind(
    Processor::LOC_PROC,
    false /*!global*/,
    STENCIL_TASK,
    CodeDescriptor(stencil_task),
    ProfilingRequestSet()
  ).external_wait();
  Processor::register_task_by_kind(
    Processor::LOC_PROC,
    false /*!global*/,
    INCREMENT_TASK,
    CodeDescriptor(increment_task),
    ProfilingRequestSet()
  ).external_wait();

#ifdef REALM_USE_CUDA
  Processor::register_task_by_kind(
    Processor::TOC_PROC,
    false /*!global*/,
    STENCIL_TASK,
    CodeDescriptor(stencil_task_gpu),
    ProfilingRequestSet()
  ).external_wait();
  Processor::register_task_by_kind(
    Processor::TOC_PROC,
    false /*!global*/,
    INCREMENT_TASK,
    CodeDescriptor(increment_task_gpu),
    ProfilingRequestSet()
  ).external_wait();
#endif


  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
      .only_kind(Processor::LOC_PROC)
      .first();
  assert(p.exists());

  // collective launch of a single main task
  rt.collective_spawn(p, TOP_LEVEL_TASK, &args, sizeof(TLTArgs));

  // main task will call shutdown - wait for that and return the exit code
  return rt.wait_for_shutdown();
}
