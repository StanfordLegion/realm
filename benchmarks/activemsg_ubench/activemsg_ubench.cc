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

#include "realm.h"
#include "realm/activemsg.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#include "realm/network.h"
#include "realm/mutex.h"

#include <time.h>
#include <unordered_map>

/*
  In this benchmark, we will benchmark the performance of realm active message.
  The communication pattern is all to all, and we adopt the SPMD fashion by
  launching a shard_task on each CPU processor. Inside the shard_task, we
  issue an active message (ping) to Network::all_peers, and when the peers receive
  the ping, they issue another active meesage (pong) back to the original sender.
  The timing starts from ping and stops after pong is received.
  When "-alltoall" is set to 0, we only issue the ping from the processor 0 shard_task
  instead of all shard_tasks.
*/

using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

// when enabled, we will measure the latency of each pingpong active message. 
// however, since active messages are queued, the start time will be the timestamp
// when the ping is queued, but the one when ping is sent out. 
// #define MEASURE_EACH_AM 

// Task IDs, some IDs are reserved so start at first available number
enum
{
  BENCHMARK_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  SHARD_TASK,
  TIME_REPORT_TASK,
};

struct ShardArgs {
  size_t id;
  Barrier barrier;
  UserEvent trigger_event;
};

namespace TestConfig {
  int output_config = 1;
  size_t payload_size = 32;
  int num_samples = 10;
  int num_pingpongs_per_sample = 10;
  int num_warmup_samples = 1;
  bool all_to_all = true;
  bool verify = false;
  int num_bgwork = 2;
  bool use_multicast = true;
}; // namespace TestConfig

std::unordered_map<realm_id_t, Mutex *> am_mutex;
std::unordered_map<realm_id_t, Mutex::CondVar *> am_condvar;
std::unordered_map<realm_id_t, atomic<int>> am_responses_needed;
#ifdef ENABLE_EXTRA_CHECK
std::unordered_map<realm_id_t, bool> am_completed; // this is only used for debugging
#endif
unsigned char *ping_payload_buff = nullptr;
unsigned char *pong_payload_buff = nullptr;
#ifdef MEASURE_EACH_AM
std::unordered_map<realm_id_t, Stat> stat_map_measure_each_am;
std::unordered_map<realm_id_t, Mutex*> update_stat_mutex_map;
#endif
std::unordered_map<realm_id_t, Stat> stat_map_measure_each_sample;

struct PingRequest {
  bool is_warmup;
  Processor ping_proc;
#ifdef MEASURE_EACH_AM
  long long ping_start_time;
#endif
  static void handle_message(int sender, const PingRequest &args, const void *data,
                             size_t datalen);
};
ActiveMessageHandlerReg<PingRequest> ping_request_handler;

struct PongRequest {
  bool is_warmup;
  Processor ping_proc;
#ifdef MEASURE_EACH_AM
  long long ping_start_time;
#endif
  static void handle_message(int sender, const PongRequest &args, const void *data,
                             size_t datalen);
};
ActiveMessageHandlerReg<PongRequest> pong_request_handler;

/*static*/ void PingRequest::handle_message(NodeID sender, const PingRequest &args,
                                            const void *data, size_t datalen)
{
  assert(datalen == TestConfig::payload_size);
  assert(pong_payload_buff != nullptr);
  log_app.info() << "ping request from node:" << sender << " proc:" << args.ping_proc;
  if(TestConfig::verify) {
    const unsigned char *payload = static_cast<const unsigned char *>(data);
    for(size_t i = 0; i < TestConfig::payload_size; i++) {
      int expected = sender % 256;
      if(payload[i] != expected) {
        log_app.error("ping expected %d, received %d at %zu", expected, payload[i], i);
      }
    }
  }
  ActiveMessage<PongRequest> amsg(sender, TestConfig::payload_size);
  amsg->is_warmup = args.is_warmup;
  amsg->ping_proc = args.ping_proc;
#ifdef MEASURE_EACH_AM
  amsg->ping_start_time = args.ping_start_time;
#endif
  amsg.add_payload(pong_payload_buff, TestConfig::payload_size);
  amsg.commit();
}

/*static*/ void PongRequest::handle_message(NodeID sender, const PongRequest &args,
                                            const void *data, size_t datalen)
{
  log_app.info() << "pong request from node:" << sender
                  << " ping proc:" << args.ping_proc;
  // record the timestamp
#ifdef MEASURE_EACH_AM
  long long end_time = Clock::current_time_in_microseconds();
  //log_app.print() << "ping ping latency:" << end_time - args.ping_start_time;
#endif
  // pid_t x = syscall(__NR_gettid);
  // log_app.print() << "proc:" << args.ping_proc << ", each ping pong latency:" << end_time - args.ping_start_time << " thread:" << x;
  assert(datalen == TestConfig::payload_size);
  if(TestConfig::verify) {
    const unsigned char *payload = static_cast<const unsigned char *>(data);
    for(size_t i = 0; i < TestConfig::payload_size; i++) {
      int expected = (sender + 1) % 256;
      if(payload[i] != expected) {
        log_app.error("pong expected %d, received %d at %zu", expected, payload[i], i);
      }
    }
  }
  // decrement the number of responses needed and wake the requestor if
  //  we're done
  {
#ifdef MEASURE_EACH_AM
    AutoLock<> al(*(update_stat_mutex_map[args.ping_proc.id]));
    assert(stat_map_measure_each_am.find(args.ping_proc.id) != stat_map_measure_each_am.end());
    if (!args.is_warmup) {
      stat_map_measure_each_am[args.ping_proc.id].sample(end_time - args.ping_start_time);
    }
#endif
    int prev = am_responses_needed[args.ping_proc.id].fetch_sub(1);
    if(prev == 1) {
      AutoLock<> al(*(am_mutex[args.ping_proc.id]));
#ifdef ENABLE_EXTRA_CHECK
      assert(am_completed[args.ping_proc.id] == false);
      am_completed[args.ping_proc.id] = true;
#endif
      am_condvar[args.ping_proc.id]->broadcast();
    }
  }
}

// one task per processor
void shard_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  assert(arglen == sizeof(ShardArgs));
  const ShardArgs &a = *reinterpret_cast<const ShardArgs *>(args);

  // pid_t pid = syscall(__NR_gettid);
  // log_app.print() << "shard_task on proc:" << p << ", thread:" << pid;

  // warmup
  NodeSet peers = Network::all_peers;
  for (int s = 0; s < TestConfig::num_warmup_samples; s++) {
    am_responses_needed[p.id].exchange(0);
    am_responses_needed[p.id].fetch_add(peers.size() * TestConfig::num_pingpongs_per_sample);
    for(int i = 0; i < TestConfig::num_pingpongs_per_sample; i++) {
      if (TestConfig::use_multicast) {
        ActiveMessage<PingRequest> amsg(peers, TestConfig::payload_size);
        amsg->is_warmup = true;
        amsg->ping_proc = p;
        amsg.add_payload(ping_payload_buff, TestConfig::payload_size);
        amsg.commit();
      } else {
        for (NodeSetIterator node_it = peers.begin(); node_it != peers.end(); node_it++) {
          ActiveMessage<PingRequest> amsg(*node_it, TestConfig::payload_size);
          amsg->is_warmup = true;
          amsg->ping_proc = p;
          amsg.add_payload(ping_payload_buff, TestConfig::payload_size);
          amsg.commit();
        }
      }
    }
    // wait for responses
    {
      AutoLock<> al(*(am_mutex[p.id]));
      while(am_responses_needed[p.id].load_acquire() > 0)
        am_condvar[p.id]->wait();
#ifdef ENABLE_EXTRA_CHECK
      assert(am_completed[p.id]);
      am_completed[p.id] = false;
#endif
    }
  }

  Barrier barrier = a.barrier;
  barrier.arrive(1);
  // barrier.wait();
  // barrier = barrier.advance_barrier();
  a.trigger_event.wait();

  for (int s = 0; s < TestConfig::num_samples; s++) {
    am_responses_needed[p.id].exchange(0);
    am_responses_needed[p.id].fetch_add(peers.size() * TestConfig::num_pingpongs_per_sample);
    long long total_start_time = Clock::current_time_in_microseconds();
    for(int i = 0; i < TestConfig::num_pingpongs_per_sample; i++) {
      if (TestConfig::use_multicast) {
        ActiveMessage<PingRequest> amsg(peers, TestConfig::payload_size);
        amsg->is_warmup = false;
        amsg->ping_proc = p;
        amsg.add_payload(ping_payload_buff, TestConfig::payload_size);
  #ifdef MEASURE_EACH_AM
        amsg->ping_start_time = Clock::current_time_in_microseconds();
  #endif
        amsg.commit();
      } else {
        for (NodeSetIterator node_it = peers.begin(); node_it != peers.end(); node_it++) {
          ActiveMessage<PingRequest> amsg(*node_it, TestConfig::payload_size);
          amsg->is_warmup = true;
          amsg->ping_proc = p;
          amsg.add_payload(ping_payload_buff, TestConfig::payload_size);
          amsg.commit();
          //log_app.print("shard %zu, send to %d", a.id, *node_it);
        }
      }
    }

    // wait for responses
    {
      AutoLock<> al(*(am_mutex[p.id]));
      while(am_responses_needed[p.id].load_acquire() > 0)
        am_condvar[p.id]->wait();
#ifdef ENABLE_EXTRA_CHECK
      assert(am_completed[p.id]);
      am_completed[p.id] = false;
#endif
    }
    long long total_end_time = Clock::current_time_in_microseconds();
  #ifdef MEASURE_EACH_AM
    log_app.info() << "proc:" << p << ", ping pong latency (measured each am):" << stat_map_measure_each_am[p.id];
  #endif
    double avg_time_per_am = (total_end_time - total_start_time) / static_cast<double>(TestConfig::num_pingpongs_per_sample * peers.size() * 2);
    log_app.info() << "proc:" << p << ", ping pong latency (measured each sample):" << avg_time_per_am << ", total time:" << total_end_time - total_start_time;
    stat_map_measure_each_sample[p.id].sample(avg_time_per_am);
  }
}

// the top level task
void benchmark_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  // output configuration
  if (TestConfig::output_config) {
    output_machine_config();
    printf("BENCHMARK_CONFIGURATION {alltoall:%d, payload_size:%zu, num_samples:%d, num_pingpong_per_sample:%d, num_bgwork:%d, multicast:%d}\n",
           TestConfig::all_to_all, TestConfig::payload_size, TestConfig::num_samples, TestConfig::num_pingpongs_per_sample, TestConfig::num_bgwork, TestConfig::use_multicast);
  }

  Machine machine = Machine::get_machine();

  std::vector<Processor> cpu_procs;
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC);
    cpu_procs.insert(cpu_procs.end(), query.begin(), query.end());
  }

  // if alltoall, then we launch one shard task per CPU processor,
  // otherwise in 1toall, we launch one shard task onto CPU processor 0
  size_t num_shard_tasks = 1;
  if(TestConfig::all_to_all) {
    num_shard_tasks = cpu_procs.size();
  }
  Barrier barrier = Barrier::create_barrier(num_shard_tasks);

  // Launch shard tasks
  {
    std::vector<Event> end_events;
    UserEvent start_event = UserEvent::create_user_event();
    for(size_t i = 0; i < num_shard_tasks; i++) {
      ShardArgs args;
      args.id = i;
      args.barrier = barrier;
      args.trigger_event = start_event;
      end_events.push_back(cpu_procs[i].spawn(SHARD_TASK, &args, sizeof(args)));
    }
    Event end_event = Event::merge_events(end_events);
    barrier.wait();
    long long start_time = Clock::current_time_in_microseconds();
    start_event.trigger();
    end_event.wait();
    long long end_time = Clock::current_time_in_microseconds();
    Machine machine = Machine::get_machine();
    //size_t num_cpus = Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC).count();
    size_t num_messages = 0;
    num_messages = TestConfig::num_pingpongs_per_sample * TestConfig::num_samples * Network::all_peers.size() * num_shard_tasks * 2; // * 2 because of pingpong
    log_app.print("num messages %zu", num_messages);
    std::cout << "RESULT {name:am_pingpong_overall_throughput, value:" << static_cast<double>(num_messages)/ (end_time - start_time) * 1e3 << ", unit:+messages/ms}" << std::endl;
    std::cout << "RESULT {name:latency_per_pingpong_message, value:" << (end_time - start_time) / static_cast<double>(num_messages) << ", unit:-us}" << std::endl;
    std::cout << "RESULT {name:total_time, value:" << end_time - start_time << ", unit:-us}" << std::endl;
  }

  barrier.destroy_barrier();
}

// one task per rank
void time_report_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                      Processor p)
{
  // first, gather all stats from local processors
#ifdef MEASURE_EACH_AM
  Stat local_stat_measure_each_am;
  for (const std::pair<const realm_id_t, Stat>& kv : stat_map_measure_each_am) {
    local_stat_measure_each_am.accumulate(kv.second);
  }
  log_app.print() << "rank:" << Network::my_node_id << ", ping pong latency (measure each am):" << local_stat_measure_each_am;
#endif
  Stat local_stat;
  for (const std::pair<const realm_id_t, Stat>& kv : stat_map_measure_each_sample) {
    local_stat.accumulate(kv.second);
  }
  log_app.print() << "rank:" << Network::my_node_id << ", ping pong latency:" << local_stat;
  if (p.address_space() == 0) {
    std::vector<Stat> all_stats;
    Network::gather<Stat>(0, local_stat, all_stats);
    Stat global_stat;
    for (const Stat &stat : all_stats) {
      global_stat.accumulate(stat);
    }
#ifdef BENCHMARK_USE_JSON_FORMAT
    std::cout << "RESULT {name:am_pingpong_latency (measured from shard_task), " << global_stat << ", unit:-us}" << std::endl;
#else
    std::cout <<"RESULT am_pingpong_latency(measured from shard_task)=/" << global_stat << " -us}" << std::endl;
#endif
  } else {
    Network::gather<Stat>(0, local_stat);
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  
  // TODO: we capture the bgwork before runtime init, because it is not accessible via API
  {
    CommandLineParser cp;
    cp.add_option_int("-ll:bgwork", TestConfig::num_bgwork, true);
    bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
    assert(ok);
  }

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-config", TestConfig::output_config)
      .add_option_int("-b", TestConfig::payload_size)
      .add_option_int("-npp", TestConfig::num_pingpongs_per_sample)
      .add_option_int("-s", TestConfig::num_samples)
      .add_option_int("-warmup", TestConfig::num_warmup_samples)
      .add_option_int("-alltoall", TestConfig::all_to_all)
      .add_option_bool("-verify", TestConfig::verify)
      .add_option_int("-multicast", TestConfig::use_multicast);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  // if there is only one node, exit the program
  if (Network::max_node_id == 0) {
    log_app.error("Please run the benchmark with at least two ranks");
    return 0;
  }

  // init the stat map
  std::vector<Processor> local_cpu_procs;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::LOC_PROC).local_address_space();
    local_cpu_procs.insert(local_cpu_procs.end(), query.begin(), query.end());
  }
  for(std::vector<Processor>::iterator it = local_cpu_procs.begin();
      it != local_cpu_procs.end(); it++) {
#ifdef ENABLE_EXTRA_CHECK
    am_completed.insert(std::pair<realm_id_t, bool>(it->id, false));
#endif
    am_responses_needed.insert(std::pair<realm_id_t, atomic<int>>(it->id, 0));
    am_mutex.insert(std::pair<realm_id_t, Mutex*>(it->id, new Mutex()));
    am_condvar.insert(std::pair<realm_id_t, Mutex::CondVar*> (it->id, new Mutex::CondVar(*(am_mutex[it->id]))));
    stat_map_measure_each_sample.insert(std::pair<realm_id_t, Stat>(it->id, Stat()));
#ifdef MEASURE_EACH_AM
    stat_map_measure_each_am.insert(std::pair<realm_id_t, Stat>(it->id, Stat()));
    update_stat_mutex_map.insert(std::pair<realm_id_t, Mutex*>(it->id, new Mutex()));
#endif
  }

  ping_payload_buff = (unsigned char *)malloc(sizeof(char) * TestConfig::payload_size);
  pong_payload_buff = (unsigned char *)malloc(sizeof(char) * TestConfig::payload_size);
  memset(ping_payload_buff, Network::my_node_id % 256, TestConfig::payload_size);
  memset(pong_payload_buff, (Network::my_node_id+1) % 256, TestConfig::payload_size);

  rt.register_task(BENCHMARK_TASK, benchmark_task);
  rt.register_task(SHARD_TASK, shard_task);
  rt.register_task(TIME_REPORT_TASK, time_report_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, BENCHMARK_TASK, nullptr, 0);

  e = rt.collective_spawn_by_kind(Processor::LOC_PROC, TIME_REPORT_TASK, nullptr, 0, true, e);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

#ifdef MEASURE_EACH_AM
  for(std::vector<Processor>::iterator it = local_cpu_procs.begin();
      it != local_cpu_procs.end(); it++) {
    delete update_stat_mutex_map[it->id];
  }
#endif

  free(ping_payload_buff);
  free(pong_payload_buff);

  for(std::vector<Processor>::iterator it = local_cpu_procs.begin();
      it != local_cpu_procs.end(); it++) {
    delete am_condvar[it->id];
    delete am_mutex[it->id];
  }

  return 0;
}