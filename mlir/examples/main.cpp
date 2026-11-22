// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "realm.h"

#ifndef REALM_USE_LLVM
#error "This example requires Realm built with REALM_ENABLE_LLVM=ON"
#endif

#include "realm/llvmjit/llvmjit.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>

extern "C" int realm_mlir_collective_start(Realm::Event::id_t registered);

std::tuple<Realm::Processor, bool> find_local_cpu();
std::tuple<Realm::Memory, bool> find_cpu_memory(Realm::Processor cpu);
std::tuple<Realm::RegionInstance, Realm::Event>
run_pipeline(Realm::Processor cpu, Realm::Memory memory,
             const Realm::InstanceLayoutGeneric *layout,
             Realm::Processor::TaskFuncID seed, Realm::Processor::TaskFuncID double_task,
             Realm::Event registered, Realm::Event start);
bool await_pipeline(Realm::Event finished);
void release_result(Realm::RegionInstance instance, Realm::Event finished);
bool run_feature_examples(Realm::Runtime &runtime, Realm::Processor cpu,
                          Realm::Memory memory, Realm::Processor::TaskFuncID seed,
                          Realm::Processor::TaskFuncID sum,
                          Realm::Processor::TaskFuncID increment);

namespace {
  constexpr Realm::Processor::TaskFuncID seed = Realm::Processor::TASK_ID_FIRST_AVAILABLE;
  constexpr Realm::Processor::TaskFuncID double_task = seed + 1;
  constexpr Realm::Processor::TaskFuncID sum = seed + 2;
  constexpr Realm::Processor::TaskFuncID increment = seed + 3;

  void *instance_address(Realm::RegionInstance::id_t id)
  {
    return Realm::RegionInstance(id).pointer_untyped(0, sizeof(int32_t));
  }

  Realm::Event register_task(Realm::Processor::TaskFuncID task,
                             const std::string &llvm_ir, const char *entry)
  {
    Realm::CodeDescriptor code(
        Realm::TypeConv::from_cpp_type<Realm::Processor::TaskFuncPtr>());
    code.add_implementation(
        new Realm::LLVMIRImplementation(llvm_ir.data(), llvm_ir.size(), entry));
    auto address_of = &instance_address;
    return Realm::Processor::register_task_by_kind(
        Realm::Processor::LOC_PROC, false, task, code, Realm::ProfilingRequestSet(),
        &address_of, sizeof(address_of));
  }

  int run_task(Realm::Runtime &runtime, Realm::Event registered)
  {
    static_assert(sizeof(size_t) == sizeof(uint64_t) &&
                      sizeof(Realm::Processor) == sizeof(uint64_t) &&
                      sizeof(Realm::RegionInstance::id_t) == sizeof(uint64_t),
                  "The task example uses Realm's 64-bit host task ABI");

    auto [cpu, found] = find_local_cpu();
    if(!found) {
      std::cerr << "No local Realm CPU processor found\n";
      return 1;
    }

    auto [memory, found_memory] = find_cpu_memory(cpu);
    if(!found_memory) {
      std::cerr << "No local Realm system memory found\n";
      return 1;
    }

    Realm::InstanceLayoutOpaque layout(sizeof(int32_t), alignof(int32_t));
    Realm::UserEvent start = Realm::UserEvent::create_user_event();
    auto [output, done] =
        run_pipeline(cpu, memory, &layout, seed, double_task, registered, start);
    bool poisoned_before_start = false;
    bool completed_before_start = done.has_triggered_faultaware(poisoned_before_start);
    start.trigger();

    bool poisoned = await_pipeline(done);
    int32_t result = poisoned || !output.exists() ? 0 : output.read<int32_t>(0);
    release_result(output, done);
    if(completed_before_start || poisoned || result != 42) {
      std::cerr << "LLVM task pipeline failed with result " << result << '\n';
      return 1;
    }

    std::cout << "MLIR pipeline queued two allocations and two LLVM tasks before start\n"
              << "LLVM task pipeline computed " << result << '\n';
    return run_feature_examples(runtime, cpu, memory, seed, sum, increment) ? 0 : 1;
  }
} // namespace

int main(int argc, char **argv)
{
  std::ifstream input(REALM_LLVM_TASK_IR_PATH);
  if(!input) {
    std::cerr << "Cannot open " << REALM_LLVM_TASK_IR_PATH << '\n';
    return 1;
  }
  std::string llvm_ir((std::istreambuf_iterator<char>(input)),
                      std::istreambuf_iterator<char>());
  if(llvm_ir.empty() || input.bad()) {
    std::cerr << "Cannot read LLVM IR from " << REALM_LLVM_TASK_IR_PATH << '\n';
    return 1;
  }

  Realm::Runtime runtime;
  if(!runtime.init(&argc, &argv))
    return 1;
  Realm::Event seed_registered = register_task(seed, llvm_ir, "seed_value");
  Realm::Event double_registered = register_task(double_task, llvm_ir, "double_value");
  Realm::Event sum_registered = register_task(sum, llvm_ir, "sum_values");
  Realm::Event increment_registered =
      register_task(increment, llvm_ir, "increment_value");
  Realm::Event registered = Realm::Event::merge_events(
      seed_registered, double_registered, sum_registered, increment_registered);
  if(realm_mlir_collective_start(registered.id)) {
    int result = run_task(runtime, registered);
    runtime.shutdown(Realm::Event::NO_EVENT, result);
  }
  return runtime.wait_for_shutdown();
}
