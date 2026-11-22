// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "realm.h"
#include "realm/cuda/cuda_access.h"
#include "realm/cuda/cuda_module.h"
#include "realm/realm_c.h"

#include <cuda.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

extern "C" int realm_mlir_cuda_examples(realm_processor_t gpu, realm_memory_t memory,
                                        realm_region_instance_t source,
                                        realm_region_instance_t separate,
                                        realm_region_instance_t combined,
                                        long long count);
extern "C" realm_event_t realm_mlir_cuda_external(realm_processor_t gpu,
                                                  realm_region_instance_t buffer,
                                                  long long count, realm_event_t ready);
extern "C" void realm_mlir_gpu_input(realm_processor_t gpu, realm_memory_t memory,
                                     realm_region_instance_t source,
                                     realm_region_instance_t target);
extern "C" realm_event_t realm_mlir_init_cuda_tasks();
extern "C" realm_event_t realm_mlir_init_gpu_input_tasks();
extern "C" int realm_mlir_collective_start(realm_event_t registered);
extern "C" void *realm_mlir_instance_address(realm_region_instance_t instance,
                                             size_t byte_offset, size_t byte_count);

namespace {
  constexpr int count = 513;
  std::string ptx;
  std::mutex kernelMutex;
  // Only one GPU processor runs the demo. Keep the module alive until its Realm
  // context is torn down at runtime shutdown.
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;

  void checkCuda(CUresult result)
  {
    if(result == CUDA_SUCCESS)
      return;
    const char *description = nullptr;
    cuGetErrorString(result, &description);
    std::fprintf(stderr, "Realm MLIR CUDA call failed (%d): %s\n", result,
                 description ? description : "unknown error");
    std::abort();
  }

  std::unique_ptr<Realm::InstanceLayoutGeneric> makeLayout()
  {
    using namespace Realm;
    std::map<FieldID, size_t> fields{{0, sizeof(int32_t)}};
    Rect<1, long long> domain(Point<1, long long>(0), Point<1, long long>(count - 1));
    int order[] = {0};
    return std::unique_ptr<InstanceLayoutGeneric>(
        InstanceLayoutGeneric::choose_instance_layout(
            IndexSpace<1, long long>(domain), InstanceLayoutConstraints(fields, 1),
            order));
  }

  bool run(Realm::Processor gpu, Realm::Memory framebuffer, Realm::Memory system)
  {
    using namespace Realm;
    auto layout = makeLayout();
    std::array<RegionInstance, 4> instances{};
    bool ok = true;
    for(RegionInstance &instance : instances) {
      Event ready = RegionInstance::create_instance(instance, system, *layout,
                                                    ProfilingRequestSet());
      bool poisoned = false;
      ready.wait_faultaware(poisoned);
      if(poisoned || !instance.exists()) {
        ok = false;
        break;
      }
    }
    auto [source, separate, combined, lowered] = instances;
    if(ok) {
      for(int i = 0; i < count; ++i)
        source.write<int32_t>(i * sizeof(int32_t), i);
      ok = realm_mlir_cuda_examples(gpu.id, framebuffer.id, source.id, separate.id,
                                    combined.id, count) == 0;
      if(ok)
        realm_mlir_gpu_input(gpu.id, framebuffer.id, source.id, lowered.id);
      for(int i = 0; ok && i < count; ++i)
        ok = separate.read<int32_t>(i * sizeof(int32_t)) == i + 14 &&
             combined.read<int32_t>(i * sizeof(int32_t)) == i + 14 &&
             lowered.read<int32_t>(i * sizeof(int32_t)) == i + 14;
    }
    for(RegionInstance instance : instances)
      if(instance.exists())
        instance.destroy();
    return ok;
  }

  // Publish the CUDA event only when its record has been enqueued. Call from a
  // host thread with the caller's CUDA context current.
  bool exportAfterCompletion(Realm::Event done, CUstream stream, CUevent signal)
  {
    bool poisoned = false;
    done.wait_faultaware(poisoned);
    if(poisoned)
      return false;
    checkCuda(cuEventRecord(signal, stream));
    return true;
  }

  bool runInterop(Realm::Processor gpu, Realm::Cuda::CudaModule &cuda)
  {
    using namespace Realm;
    int index;
    if(!Cuda::get_cuda_device_id(gpu, &index))
      return false;
    CUdevice device;
    CUcontext context;
    checkCuda(cuDeviceGet(&device, index));
    checkCuda(cuDevicePrimaryCtxRetain(&context, device));
    checkCuda(cuCtxPushCurrent(context));

    CUstream producer, consumer;
    CUevent input, output;
    CUdeviceptr allocation;
    void *readback = nullptr;
    constexpr size_t bytes = count * sizeof(int32_t);
    checkCuda(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING));
    checkCuda(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING));
    checkCuda(cuEventCreate(&input, CU_EVENT_DISABLE_TIMING));
    checkCuda(cuEventCreate(&output, CU_EVENT_DISABLE_TIMING));
    checkCuda(cuMemAlloc(&allocation, bytes));
    checkCuda(cuMemAllocHost(&readback, bytes));
    checkCuda(cuMemsetD32Async(allocation, 21, count, producer));
    checkCuda(cuEventRecord(input, producer));
    Event imported = cuda.make_realm_event(input);

    RegionInstance instance;
    auto layout = makeLayout();
    ExternalCudaMemoryResource external(index, static_cast<uintptr_t>(allocation), bytes,
                                        false);
    Event ready = RegionInstance::create_external_instance(
        instance, external.suggested_memory(), *layout, external, ProfilingRequestSet());
    Event done = ready;
    bool ok = instance.exists();
    if(ok) {
      Event gate = Event::merge_events(imported, ready);
      done = Event(realm_mlir_cuda_external(gpu.id, instance.id, count, gate.id));
      ok = exportAfterCompletion(done, producer, output);
    }
    if(ok) {
      checkCuda(cuStreamWaitEvent(consumer, output, 0));
      checkCuda(cuMemcpyDtoHAsync(readback, allocation, bytes, consumer));
      checkCuda(cuStreamSynchronize(consumer));
      const auto *values = static_cast<const int32_t *>(readback);
      for(int i = 0; ok && i < count; ++i)
        ok = values[i] == 28;
    }

    Event uses[] = {imported, ready, done};
    Event::merge_events_ignorefaults(uses, 3).wait();
    if(instance.exists())
      instance.destroy();
    checkCuda(cuMemFree(allocation));
    checkCuda(cuMemFreeHost(readback));
    checkCuda(cuEventDestroy(input));
    checkCuda(cuEventDestroy(output));
    checkCuda(cuStreamDestroy(producer));
    checkCuda(cuStreamDestroy(consumer));
    CUcontext popped;
    checkCuda(cuCtxPopCurrent(&popped));
    checkCuda(cuDevicePrimaryCtxRelease(device));
    return ok;
  }
} // namespace

extern "C" void realm_mlir_cuda_add_dims(realm_region_instance_t instance,
                                         long long length, int amount, long long gx,
                                         long long gy, long long gz, long long bx,
                                         long long by, long long bz)
{
  CUstream stream = Realm::Cuda::get_task_cuda_stream();
  if(!stream || length != count)
    std::abort();
  for(long long dim : {gx, gy, gz, bx, by, bz})
    if(dim < 1 || dim > std::numeric_limits<unsigned>::max())
      std::abort();

  Realm::Cuda::set_task_ctxsync_required(false);
  CUfunction function;
  {
    std::lock_guard<std::mutex> lock(kernelMutex);
    if(!kernel) {
      checkCuda(cuModuleLoadData(&module, ptx.c_str()));
      checkCuda(cuModuleGetFunction(&kernel, module, "add_to_buffer"));
    }
    function = kernel;
  }
  auto address = reinterpret_cast<CUdeviceptr>(
      realm_mlir_instance_address(instance, 0, length * sizeof(int32_t)));
  void *args[] = {&address, &length, &amount};
  checkCuda(cuLaunchKernel(function, gx, gy, gz, bx, by, bz, 0, stream, args, nullptr));
}

extern "C" void realm_mlir_cuda_add(realm_region_instance_t instance, long long length,
                                    int amount)
{
  constexpr int threads = 128;
  realm_mlir_cuda_add_dims(instance, length, amount, (length + threads - 1) / threads, 1,
                           1, threads, 1, 1);
}

int main(int argc, char **argv)
{
  using namespace Realm;
  Runtime runtime;
  if(!runtime.init(&argc, &argv))
    return 1;
  Event cuda_ready(realm_mlir_init_cuda_tasks());
  Event gpu_input_ready(realm_mlir_init_gpu_input_tasks());
  if(!realm_mlir_collective_start(Event::merge_events(cuda_ready, gpu_input_ready).id))
    return runtime.wait_for_shutdown();

  std::ifstream input(REALM_MLIR_CUDA_PTX);
  ptx.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  if(ptx.empty()) {
    std::fputs("Could not read the MLIR CUDA kernel PTX\n", stderr);
    runtime.shutdown(Event::NO_EVENT, 1);
    return runtime.wait_for_shutdown();
  }

  Machine machine = Machine::get_machine();
  Processor gpu = Machine::ProcessorQuery(machine)
                      .local_address_space()
                      .only_kind(Processor::TOC_PROC)
                      .first();
  bool ok = false;
  if(gpu.exists()) {
    Memory framebuffer = Machine::MemoryQuery(machine)
                             .only_kind(Memory::GPU_FB_MEM)
                             .same_address_space_as(gpu)
                             .best_affinity_to(gpu)
                             .first();
    Memory system = Machine::MemoryQuery(machine)
                        .local_address_space()
                        .only_kind(Memory::SYSTEM_MEM)
                        .first();
    if(framebuffer.exists() && system.exists())
      ok = run(gpu, framebuffer, system);
  }
  if(ok)
    std::puts("Separate Realm tasks, one Realm stream, and GPU dialect lowering produced "
              "i + 14");
  else
    std::fputs(
        "CUDA example failed; check the local CUDA processor, framebuffer, and output\n",
        stderr);
  if(ok) {
    auto *cuda = runtime.get_module<Cuda::CudaModule>("cuda");
    ok = cuda && runInterop(gpu, *cuda);
    if(ok)
      std::puts("Borrowed CUDA pointer and event round-trip produced 28");
    else
      std::fputs("CUDA event or external instance round-trip failed\n", stderr);
  }
  runtime.shutdown(Event::NO_EVENT, ok ? 0 : 1);
  return runtime.wait_for_shutdown();
}
