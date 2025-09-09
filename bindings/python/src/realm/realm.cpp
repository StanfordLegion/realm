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

#include <realm.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

namespace nb = nanobind;
using namespace nb::literals;

static Realm::Runtime runtime;

struct FunctionRecord {
  nb::callable fn;
  nb::object user_data;
  typedef std::unordered_map<Realm::Processor::TaskFuncID, FunctionRecord>
      FunctionRecordMap;
  static nb::ft_mutex mutex;
  static FunctionRecordMap records;

  static void task_body(const void *arg_buffer, size_t arglen, const void *userdata,
                        size_t userlen, Realm::Processor proc)
  {
    const FunctionRecord *rec = reinterpret_cast<const FunctionRecord *>(userdata);
    // Make sure to re-acquire the gil before we start messing with python again
    nb::gil_scoped_acquire gil;
    nb::tuple arg_tuple;
    {
      nb::bytes arg_bytes(arg_buffer, arglen);
      arg_tuple =
          nb::cast<nb::tuple>(nb::module_::import_("pickle").attr("loads")(arg_bytes));
    }
    nb::args args = nb::cast<nb::args>(arg_tuple[0]);
    nb::kwargs kwargs = nb::cast<nb::kwargs>(arg_tuple[1]);
    rec->fn(proc, rec->user_data, *args, **kwargs);
  }

  static FunctionRecord &register_function(Realm::Processor::TaskFuncID func_id,
                                           nb::callable &fn, nb::object &obj)
  {
    nb::ft_lock_guard lg(mutex);
    std::pair<FunctionRecordMap::iterator, bool> it =
        records.emplace(std::make_pair(func_id, FunctionRecord{fn, obj}));
    return it.first->second;
  }
};

nb::ft_mutex FunctionRecord::mutex;
FunctionRecord::FunctionRecordMap FunctionRecord::records;

template <typename T>
void add_id_type_definitions(nb::class_<T> &cls)
{
  cls.def(nb::init<>())
      .def(nb::init<realm_id_t>())
      .def_rw("id", &T::id)
      .def(nb::self == nb::self)
      .def(nb::self < nb::self)
      .def("exists", &T::exists)
      .def("__int__", [](const T &mem) { return mem.id; })
      .def("__getstate__", [](const T &obj) { return obj.id; })
      .def("__setstate__", [](T &obj, realm_id_t state) { obj.id = state; })
      .def("__str__", [](const T &obj) {
        std::ostringstream ss;
        ss << obj;
        return ss.str();
      });
}

NB_MODULE(realm, m)
{
  m.doc() = "A distributed, event-based tasking runtime for building high-performance "
            "applications that span clusters of CPUs, GPUs, and other accelerators.";
  m.attr("__version__") = REALM_VERSION;
  nb::class_<Realm::Runtime> py_runtime(m, "Runtime");
  nb::class_<Realm::Memory> py_memory(m, "Memory");
  nb::class_<Realm::Processor> py_processor(m, "Processor");
  nb::class_<Realm::ProcessorGroup, Realm::Processor> py_processor_group(
      m, "ProcessorGroup");
  nb::class_<Realm::Event> py_event(m, "Event");
  nb::class_<Realm::UserEvent, Realm::Event> py_user_event(m, "UserEvent");
  nb::class_<Realm::Barrier, Realm::Event> py_barrier(m, "Barrier");
  nb::class_<Realm::RegionInstance> py_instance(m, "RegionInstance");
  nb::class_<Realm::Reservation> py_reservation(m, "Reservation");
  nb::class_<Realm::CompletionQueue> py_completion_queue(m, "CompletionQueue");
  // Templated
  // nb::class_<Realm::SparsityMap> py_sparsity_map(m, "SparsityMap");
  // Doesn't implement operator<<
  // nb::class_<Realm::Subgraph> py_subgraph(m, "Subgraph");

  py_runtime.def(nb::init<>())
      .def_ro_static("runtime", &runtime)
      .def("init", [](Realm::Runtime &r) {
        int argc = 0;
        char **argv = nullptr;
        // TODO: pass in the arguments
        return r.init(&argc, &argv);
      });

  add_id_type_definitions(py_memory);
  add_id_type_definitions(py_event);
  add_id_type_definitions(py_processor);
  add_id_type_definitions(py_instance);
  add_id_type_definitions(py_reservation);
  add_id_type_definitions(py_completion_queue);
  // add_id_type_definitions(py_sparsity_map);
  // add_id_type_definitions(py_subgraph);
  py_memory.def_ro_static("NO_MEMORY", &Realm::Memory::NO_MEMORY);
  py_event.def_ro_static("NO_EVENT", &Realm::Event::NO_EVENT);
  py_processor.def_ro_static("NO_PROC", &Realm::Processor::NO_PROC);
  py_instance.def_ro_static("NO_INST", &Realm::RegionInstance::NO_INST);
  py_reservation.def_ro_static("NO_RESERVATION", &Realm::Reservation::NO_RESERVATION);
  py_completion_queue.def_ro_static("NO_QUEUE", &Realm::CompletionQueue::NO_QUEUE);

  nb::enum_<Realm::Memory::Kind>(py_memory, "Kind")
#define DEFINE_MEMORY_KINDS(name, desc) .value(#name, Realm::Memory::name)
      REALM_MEMORY_KINDS(DEFINE_MEMORY_KINDS)
#undef DEFINE_MEMORY_KINDS
          .export_values();

  py_memory.def_prop_ro("address_space", &Realm::Memory::address_space)
      .def_prop_ro("kind", &Realm::Memory::kind)
      .def_prop_ro("capacity", &Realm::Memory::capacity);

  py_event
      .def("has_triggered", &Realm::Event::has_triggered)
      // Make sure to _release_ the gil before we call wait here, so we can reacquire it
      // later
      .def("wait", &Realm::Event::wait, nb::call_guard<nb::gil_scoped_release>())
      .def("subscribe", &Realm::Event::subscribe)
      .def("cancel_operation",
           [](Realm::Event &event) { event.cancel_operation(nullptr, 0); })
      .def_static("merge_events", [](std::vector<Realm::Event> &events) {
        return Realm::Event::merge_events(events.data(), events.size());
      });

  py_user_event.def_static("create", &Realm::UserEvent::create_user_event)
      .def("trigger", &Realm::UserEvent::trigger, "wait_on"_a = Realm::Event::NO_EVENT,
           "ignore_faults"_a = false)
      .def("cancel", &Realm::UserEvent::cancel);

  py_barrier
      .def_static(
          "create",
          [](unsigned arrivals) { return Realm::Barrier::create_barrier(arrivals); })
      .def("destroy", &Realm::Barrier::destroy_barrier)
      .def("advance", &Realm::Barrier::advance_barrier)
      .def(
          "arrive",
          [](Realm::Barrier &bar, unsigned count, Realm::Event &event) {
            bar.arrive(count, event);
          },
          nb::arg() = 1, nb::arg() = Realm::Event::NO_EVENT)
      .def("__getstate__",
           [](const Realm::Barrier &obj) {
             return std::make_tuple(obj.id, obj.timestamp);
           })
      .def("__setstate__",
           [](Realm::Barrier &obj,
              std::tuple<realm_id_t, Realm::Barrier::timestamp_t> &state) {
             obj.id = std::get<0>(state);
             obj.timestamp = std::get<1>(state);
           });

  nb::enum_<Realm::Processor::Kind>(py_processor, "Kind")
#define DEFINE_PROC_KINDS(name, desc) .value(#name, Realm::Processor::name)
      REALM_PROCESSOR_KINDS(DEFINE_PROC_KINDS)
#undef DEFINE_PROC_KINDS
          .export_values();

  py_processor.def_prop_ro("kind", &Realm::Processor::kind)
      .def_prop_ro("address_space", &Realm::Processor::address_space)
      .def_prop_ro("cores", &Realm::Processor::get_num_cores)
      .def_prop_ro("current_finish_event", &Realm::Processor::get_current_finish_event)
      .def("spawn",
           [](const Realm::Processor &proc, Realm::Processor::TaskFuncID func_id,
              nb::args args, nb::kwargs &kwargs) -> Realm::Event {
             // TODO: only do this if remote, otherwise just pass the tuple and some
             // indication
             nb::bytes serialized_args =
                 nb::cast<nb::bytes>(nb::module_::import_("pickle").attr("dumps")(
                                         nb::make_tuple(args, kwargs)),
                                     false);
             return proc.spawn(func_id, serialized_args.data(), serialized_args.size());
           })
      .def("register_task",
           [](const Realm::Processor &proc, Realm::Processor::TaskFuncID func_id,
              nb::callable fn, nb::object user_data) -> Realm::Event {
             Realm::CodeDescriptor codedesc(&FunctionRecord::task_body);
             Realm::ProfilingRequestSet prs;
             FunctionRecord &rec =
                 FunctionRecord::register_function(func_id, fn, user_data);
             return proc.register_task(func_id, codedesc, prs, &rec, sizeof(rec));
           });
}