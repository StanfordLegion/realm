#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"
#include "realm/network.h"
#include <unistd.h>
#include <cstdio>

using namespace Realm;

enum
{
  TOP_TASK_ID = Processor::TASK_ID_FIRST_AVAILABLE
};

namespace TestConfig {
  int expected_peers{4};
  int shutdown_node_id{1};
}; // namespace TestConfig

void top_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
              Processor p)
{
  std::cout << "Top task:" << Network::my_node_id << std::endl;

  auto machine = Machine::get_machine();
  auto last_epoch = machine.get_epoch();

  while(true) {
    size_t peers = Network::node_directory.size();
    Epoch_t epoch = machine.get_epoch();
    if(peers >= size_t(TestConfig::expected_peers) &&
       epoch >= size_t(TestConfig::expected_peers)) {
      break;
    }

    sleep(1);

    std::cout << "Peers:" << Network::node_directory.size()
              << " on node:" << Network::my_node_id << " epoch:" << machine.get_epoch()
              << std::endl;
  }

  Event e = machine.update(uint64_t(TestConfig::expected_peers));
  e.wait();

  std::map<NodeID, int> cpu_count, sysmem_visible, pma_count;

  // Count CPUs and detect whether each CPU can reach at least one SYSTEM_MEM
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  for(Processor pr : pq) {
    NodeID as = pr.address_space();
    cpu_count[as]++;

    // Check visible memories for this processor (local_only = false)
    std::set<Memory> vis;
    machine.get_visible_memories(pr, vis, /*local_only=*/false);
    for(Memory mem : vis) {
      if(mem.kind() == Memory::SYSTEM_MEM) {
        sysmem_visible[as]++;
        break; // one is enough
      }
    }
  }
  // Count PMA records per address space
  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  machine.get_proc_mem_affinity(pmas);
  for(const auto &a : pmas)
    pma_count[a.p.address_space()]++;

  for(NodeID n = 0; n < TestConfig::expected_peers; n++) {
    // Each rank must have at least one CPU
    assert(cpu_count[n] > 0);
    // and every CPU must reach a SYSTEM_MEM
    assert(sysmem_visible[n] > 0);
    // plus at least one processor–memory affinity record
    assert(pma_count[n] > 0);
  }

  sleep(8);

  // Runtime::get_runtime().shutdown();
  // return;

  if(Network::my_node_id == TestConfig::shutdown_node_id) {
    Runtime::get_runtime().shutdown();
  } else {

    while(true) {
      size_t peers = Network::node_directory.size();
      Epoch_t epoch = machine.get_epoch();
      if(peers <= size_t(TestConfig::expected_peers - 1) &&
         epoch >= size_t(TestConfig::expected_peers + 1)) {
        break;
      }

      sleep(1);

      std::cout << "WaitForShutDown Peers:" << Network::node_directory.size()
                << " on node:" << Network::my_node_id << " epoch:" << machine.get_epoch()
                << std::endl;
    }
    Runtime::get_runtime().shutdown();
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  if(!rt.init(&argc, &argv)) {
    return 0;
  }

  rt.register_task(TOP_TASK_ID, top_task);

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
