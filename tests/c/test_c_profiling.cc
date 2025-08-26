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

#include "common.h"

#include "realm.h"
#include "realm/profiling.h"
#include "realm/utils.h"

#include <stdio.h>
#include <assert.h>

Realm::Logger log_app("app");

using namespace Realm::ProfilingMeasurements;

enum
{
  MAIN_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  CHILD_TASK,
  OP_PROFILING_RESPONSE_TASK,
};

enum
{
  FID_BASE = 44,
};

struct ChildTaskArgs {
  bool inject_fault = false;
  bool hang = false;
  int sleep_useconds = 100000;
  Realm::Event wait_on = Realm::Event::NO_EVENT;
};

void REALM_FNPTR child_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  log_app.print("child_task on proc " IDFMT, proc);
  assert(arglen == sizeof(ChildTaskArgs));
  const ChildTaskArgs &cargs = *(const ChildTaskArgs *)args;
  if(cargs.wait_on.exists()) {
    cargs.wait_on.wait();
  }
}

void REALM_FNPTR op_profiling_response_task(const void *args, size_t arglen,
                                            const void *userdata, size_t userlen,
                                            realm_processor_t proc)
{
  log_app.print("op_profiling_response_task on proc " IDFMT, proc);
  realm_profiling_response_t response;
  response.data = args;
  response.data_size = arglen;

  realm_status_t status = REALM_SUCCESS;

  if(realm_profiling_response_get_measurement(&response, PMID_OP_TIMELINE, nullptr,
                                              nullptr) == REALM_SUCCESS) {
    realm_profiling_measurement_operation_timeline_t op_timeline;
    status = realm_profiling_response_get_measurement(&response, PMID_OP_TIMELINE,
                                                      &op_timeline, nullptr);
    assert(status == REALM_SUCCESS);
    printf("op timeline = %lld %lld %lld %lld (%lld %lld %lld)\n", op_timeline.ready_time,
           op_timeline.start_time, op_timeline.end_time, op_timeline.complete_time,
           (((op_timeline.start_time >= 0) && (op_timeline.ready_time >= 0))
                ? (op_timeline.start_time - op_timeline.ready_time)
                : -1),
           (((op_timeline.end_time >= 0) && (op_timeline.start_time >= 0))
                ? (op_timeline.end_time - op_timeline.start_time)
                : -1),
           (((op_timeline.complete_time >= 0) && (op_timeline.end_time >= 0))
                ? (op_timeline.complete_time - op_timeline.end_time)
                : -1));
  } else {
    printf("no timeline\n");
  }

  if(realm_profiling_response_get_measurement(&response, PMID_OP_EVENT_WAITS, nullptr,
                                              nullptr) == REALM_SUCCESS) {
    realm_profiling_measurement_operation_event_wait_interval_t *op_waits = nullptr;
    size_t count = 0;
    status = realm_profiling_response_get_measurement(&response, PMID_OP_EVENT_WAITS,
                                                      op_waits, &count);
    assert(status == REALM_SUCCESS);
    printf("op waits = %zd", count);
    if(count > 0) {
      op_waits = new realm_profiling_measurement_operation_event_wait_interval_t[count];
      status = realm_profiling_response_get_measurement(&response, PMID_OP_EVENT_WAITS,
                                                        op_waits, &count);
      printf(" [");
      for(size_t i = 0; i < count; i++) {
        printf(" (%lld %lld %lld %llx)", op_waits[i].wait_start, op_waits[i].wait_ready,
               op_waits[i].wait_end, op_waits[i].wait_event);
      }
      printf(" ]\n");
      delete[] op_waits;
    }
  } else {
    printf("no event wait data\n");
  }

  if(realm_profiling_response_get_measurement(&response, PMID_OP_COPY_INFO, nullptr,
                                              nullptr) == REALM_SUCCESS) {
    size_t count = 0;
    status = realm_profiling_response_get_measurement(&response, PMID_OP_COPY_INFO,
                                                      nullptr, &count);
    assert(status == REALM_SUCCESS);
    Realm::ProfilingResponse pr(args, arglen);
    OperationCopyInfo *cxx_copy_info = pr.get_measurement<OperationCopyInfo>();
    printf("copy info = %zd\n", count);
    for(size_t i = 0; i < count; i++) {
      size_t inst_count = 0;
      Realm::RegionInstance *insts = nullptr;
      // src insts
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_SRC_INST_ID(i), nullptr, &inst_count);
      assert(status == REALM_SUCCESS);
      insts = new Realm::RegionInstance[inst_count];
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_SRC_INST_ID(i), insts, &inst_count);
      assert(status == REALM_SUCCESS);
      log_app.print() << "src_insts idx " << i << " = "
                      << Realm::PrettyVector<Realm::RegionInstance>(
                             std::vector<Realm::RegionInstance>(insts,
                                                                insts + inst_count));
      delete[] insts;

      // dst insts
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_DST_INST_ID(i), nullptr, &inst_count);
      assert(status == REALM_SUCCESS);
      insts = new Realm::RegionInstance[inst_count];
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_DST_INST_ID(i), insts, &inst_count);
      assert(status == REALM_SUCCESS);
      log_app.print() << "dst_insts idx " << i << " = "
                      << Realm::PrettyVector<Realm::RegionInstance>(
                             std::vector<Realm::RegionInstance>(insts,
                                                                insts + inst_count));
      delete[] insts;

      // src fields
      realm_field_id_t *field_ids = nullptr;
      size_t field_count = 0;
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_SRC_FIELD_ID(i), nullptr, &field_count);
      assert(status == REALM_SUCCESS);
      field_ids = new realm_field_id_t[field_count];
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_SRC_FIELD_ID(i), field_ids, &field_count);
      assert(status == REALM_SUCCESS);
      log_app.print() << "src_fields idx " << i << " = "
                      << Realm::PrettyVector<Realm::FieldID>(std::vector<Realm::FieldID>(
                             field_ids, field_ids + field_count));
      delete[] field_ids;

      // dst fields
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_DST_FIELD_ID(i), nullptr, &field_count);
      assert(status == REALM_SUCCESS);
      field_ids = new realm_field_id_t[field_count];
      status = realm_profiling_response_get_measurement(
          &response, MAKE_OP_COPY_INFO_DST_FIELD_ID(i), field_ids, &field_count);
      assert(status == REALM_SUCCESS);
      log_app.print() << "dst_fields idx " << i << " = "
                      << Realm::PrettyVector<Realm::FieldID>(std::vector<Realm::FieldID>(
                             field_ids, field_ids + field_count));
      delete[] field_ids;
    }
    for(Realm::ProfilingMeasurements::OperationCopyInfo::InstInfo &inst_info :
        cxx_copy_info->inst_info) {
      log_app.print() << "copy type " << inst_info.request_type << ", src_insts ("
                      << Realm::PrettyVector<Realm::RegionInstance>(inst_info.src_insts)
                      << ")"
                      << ", dst_insts ("
                      << Realm::PrettyVector<Realm::RegionInstance>(inst_info.dst_insts)
                      << ")"
                      << ", src_fid "
                      << Realm::PrettyVector<Realm::FieldID>(inst_info.src_fields)
                      << ", dst_fid "
                      << Realm::PrettyVector<Realm::FieldID>(inst_info.dst_fields);
    }
    delete cxx_copy_info;
  } else {
    printf("no copy info\n");
  }
  assert(status == REALM_SUCCESS);
}

void REALM_FNPTR main_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, realm_processor_t proc)
{
  log_app.info("main_task on proc " IDFMT, proc);
  realm_event_t event;
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);

  // test timeline and status
  {
    realm_profiling_measurement_id_t measurement_ids[2];
    measurement_ids[0] = PMID_OP_TIMELINE;
    measurement_ids[1] = PMID_OP_STATUS;

    realm_profiling_request_t request[1];
    request[0].response_proc = proc;
    request[0].response_task_id = OP_PROFILING_RESPONSE_TASK;
    request[0].priority = 0;
    request[0].report_if_empty = 0;
    request[0].measurement_ids = measurement_ids;
    request[0].num_measurements = 2;
    request[0].payload = NULL;
    request[0].payload_size = 0;

    ChildTaskArgs cargs;
    status = realm_processor_spawn(runtime, proc, CHILD_TASK, &cargs, sizeof(cargs),
                                   request, 1, REALM_NO_EVENT, 0, &event);
    assert(status == REALM_SUCCESS);
    status = realm_event_wait(runtime, event, nullptr);
    assert(status == REALM_SUCCESS);
  }

  // test event waits
  {
    Realm::Processor task_proc = Realm::Processor(proc);
    Realm::ProfilingRequestSet prs;
    Realm::ProfilingRequest &pr =
        prs.add_request(task_proc, OP_PROFILING_RESPONSE_TASK, NULL, 0);
    pr.add_measurement<OperationEventWaits>();

    ChildTaskArgs cargs;
    Realm::UserEvent u = Realm::UserEvent::create_user_event();
    cargs.wait_on = u;
    Realm::Event e4 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    cargs.wait_on = Realm::Event::NO_EVENT;
    cargs.sleep_useconds = 500000;
    Realm::Event e5 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    u.trigger(e5);
    e4.wait();
  }

  // test copy info
  {
    Realm::Machine machine = Realm::Machine::get_machine();
    Realm::Memory system_memory = Realm::Machine::MemoryQuery(machine)
                                      .local_address_space()
                                      .only_kind(Realm::Memory::SYSTEM_MEM)
                                      .first();
    int num_elements = 1024;
    Realm::IndexSpace<1> idx_space = Realm::Rect<1>(0, num_elements - 1);
    Realm::RegionInstance src_inst_1, src_inst_2, dst_inst_1, dst_inst_2;
    std::map<Realm::FieldID, size_t> field_sizes;
    field_sizes[FID_BASE] = sizeof(int);
    field_sizes[FID_BASE + 1] = sizeof(int);
    Realm::RegionInstance::create_instance(src_inst_1, system_memory, idx_space,
                                           field_sizes, 0 /*SOA*/,
                                           Realm::ProfilingRequestSet())
        .wait();
    Realm::RegionInstance::create_instance(src_inst_2, system_memory, idx_space,
                                           field_sizes, 0 /*SOA*/,
                                           Realm::ProfilingRequestSet())
        .wait();
    Realm::RegionInstance::create_instance(dst_inst_1, system_memory, idx_space,
                                           field_sizes, 0 /*SOA*/,
                                           Realm::ProfilingRequestSet())
        .wait();
    Realm::RegionInstance::create_instance(dst_inst_2, system_memory, idx_space,
                                           field_sizes, 0 /*SOA*/,
                                           Realm::ProfilingRequestSet())
        .wait();

    std::vector<Realm::CopySrcDstField> srcs(4);
    srcs[0].set_field(src_inst_1, FID_BASE, sizeof(int));
    srcs[1].set_field(src_inst_1, FID_BASE + 1, sizeof(int));
    srcs[2].set_field(src_inst_2, FID_BASE, sizeof(int));
    srcs[3].set_field(src_inst_2, FID_BASE + 1, sizeof(int));
    std::vector<Realm::CopySrcDstField> dsts(4);
    dsts[0].set_field(dst_inst_1, FID_BASE, sizeof(int));
    dsts[1].set_field(dst_inst_1, FID_BASE + 1, sizeof(int));
    dsts[2].set_field(dst_inst_2, FID_BASE, sizeof(int));
    dsts[3].set_field(dst_inst_2, FID_BASE + 1, sizeof(int));
    Realm::ProfilingRequestSet prs;
    Realm::Processor task_proc = Realm::Processor(proc);
    Realm::ProfilingRequest &pr =
        prs.add_request(task_proc, OP_PROFILING_RESPONSE_TASK, NULL, 0);
    pr.add_measurement<OperationCopyInfo>();

    idx_space.copy(srcs, dsts, prs).wait();

    src_inst_1.destroy();
    src_inst_2.destroy();
    dst_inst_1.destroy();
    dst_inst_2.destroy();
  }

}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_create(&runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_init(runtime, &argc, &argv);
  assert(status == REALM_SUCCESS);

  realm_event_t register_task_event;

  status = realm_processor_register_task_by_kind(runtime, LOC_PROC,
                                                 REALM_REGISTER_TASK_DEFAULT, MAIN_TASK,
                                                 main_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event, nullptr);
  assert(status == REALM_SUCCESS);

  status = realm_processor_register_task_by_kind(runtime, LOC_PROC,
                                                 REALM_REGISTER_TASK_DEFAULT, CHILD_TASK,
                                                 child_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event, nullptr);
  assert(status == REALM_SUCCESS);

  status = realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, OP_PROFILING_RESPONSE_TASK,
      op_profiling_response_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event, nullptr);
  assert(status == REALM_SUCCESS);

  realm_processor_query_t proc_query;
  status = realm_processor_query_create(runtime, &proc_query);
  assert(status == REALM_SUCCESS);
  status = realm_processor_query_restrict_to_kind(proc_query, LOC_PROC);
  assert(status == REALM_SUCCESS);
  realm_processor_t proc;
  realm_processor_query_first(proc_query, &proc);
  status = realm_processor_query_destroy(proc_query);
  assert(status == REALM_SUCCESS);
  assert(proc != REALM_NO_PROC);

  realm_event_t e;
  status = realm_runtime_collective_spawn(runtime, proc, MAIN_TASK, 0, 0, 0, 0, &e);
  assert(status == REALM_SUCCESS);

  status = realm_runtime_signal_shutdown(runtime, e, 0);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_wait_for_shutdown(runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_destroy(runtime);
  assert(status == REALM_SUCCESS);

  return 0;
}