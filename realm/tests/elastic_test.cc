#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"
#include "realm/network.h"
#include <unistd.h>
#include <cstdio>

using namespace Realm;

enum
{
  TOP_TASK_ID = Processor::TASK_ID_FIRST_AVAILABLE,
  NODE_TASK,
};

struct TaskArgs {};

namespace TestConfig {
  int expected_peers{4};
  int shutdown_node_id{1};
  // int failed_node_id{2};
  int timeout_sec{16};
  std::vector<NodeID> failed_node_ids{2};
  std::vector<NodeID> active_node_ids{0};
}; // namespace TestConfig

void node_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;
}

static bool is_failed_node(NodeID n)
{
  return std::find(TestConfig::failed_node_ids.begin(), TestConfig::failed_node_ids.end(),
                   NodeID(n)) != TestConfig::failed_node_ids.end();
}

static bool wait_until(std::function<bool()> pred, int timeout_sec)
{
  auto start = std::chrono::steady_clock::now();
  while(true) {
    if(pred()) {
      return true;
    } else if(std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::steady_clock::now() - start)
                  .count() > timeout_sec) {
      return false;
    }
    sleep(1);
  }
}

void top_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
              Processor p)
{
  auto machine = Machine::get_machine();

  bool ok = wait_until(
      [&]() {
        size_t peers = Network::node_directory.size();
        Epoch_t epoch = machine.get_epoch();
        return (peers >= size_t(TestConfig::expected_peers)) &&
               (epoch >= size_t(TestConfig::expected_peers));
      },
      TestConfig::timeout_sec);
  assert(ok && "timeout waiting for initial peers/epoch");

  Event e = machine.update(uint64_t(TestConfig::expected_peers));
  e.wait();

  if(!is_failed_node(Network::my_node_id)) {
    for(NodeID fn : TestConfig::failed_node_ids) {
      for(Machine::ProcessorQuery::iterator it =
              Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC).begin();
          it; ++it) {
        if((*it).address_space() == fn && fn != Network::my_node_id) {
          TaskArgs args;
          // args.wait_on = done;
          Event e = (*it).spawn(NODE_TASK, &args, sizeof(args), Event::NO_EVENT, 0);
          bool poisoned = false;
          e.wait_faultaware(poisoned);
          assert(poisoned);
          break;
        }
      }
    }
  }

  std::map<NodeID, int> cpu_count, sysmem_visible, pma_count;
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  for(Processor pr : pq) {
    NodeID as = pr.address_space();
    cpu_count[as]++;

    std::set<Memory> vis;
    machine.get_visible_memories(pr, vis, false);
    for(Memory mem : vis) {
      if(mem.kind() == Memory::SYSTEM_MEM) {
        sysmem_visible[as]++;
        break; // one is enough
      }
    }
  }

  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  machine.get_proc_mem_affinity(pmas);
  for(const auto &a : pmas) {
    pma_count[a.p.address_space()]++;
  }

  for(NodeID n = 0; n < TestConfig::expected_peers; n++) {
    if(!is_failed_node(n)) {
      assert(cpu_count[n] > 0);
      assert(sysmem_visible[n] > 0);
      assert(pma_count[n] > 0);
    }
  }

  sleep(8);

  if(is_failed_node(Network::my_node_id)) {
    return;
  }

  if(Network::my_node_id != TestConfig::shutdown_node_id) {
    const size_t expected_after_fail =
        size_t(TestConfig::expected_peers) - size_t(TestConfig::failed_node_ids.size());
    const size_t expected_epoch_min =
        size_t(TestConfig::expected_peers) + size_t(TestConfig::failed_node_ids.size());

    bool f_ok = wait_until(
        [&]() {
          size_t peers = Network::node_directory.size();
          Epoch_t epoch = machine.get_epoch();
          return (peers <= expected_after_fail) && (epoch >= expected_epoch_min);
        },
        TestConfig::timeout_sec);
    assert(f_ok && "timeout waiting for post-failure peers/epoch");
  }

  std::cout << "Call Shutdown Me:" << Network::my_node_id << std::endl;
  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;
  if(!rt.init(&argc, &argv)) {
    return 0;
  }

  rt.register_task(TOP_TASK_ID, top_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, NODE_TASK,
                                   CodeDescriptor(node_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .local_address_space()
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = p.spawn(TOP_TASK_ID, /*args=*/nullptr, /*arglen=*/0);

  // rt.shutdown(e);
  rt.wait_for_shutdown();

  return 0;
}
