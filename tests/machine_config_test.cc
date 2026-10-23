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
#include "realm/cmdline.h"
#include "realm/network.h"
#include "realm/cuda/cuda_module.h"
#include "realm/hip/hip_module.h"
#include "realm/numa/numa_module.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iomanip>

#ifdef REALM_USE_CUDA
#include <cuda.h>
#endif

#ifdef REALM_ON_WINDOWS
#include <malloc.h>
#elif !defined(REALM_ON_FREEBSD) // (FreeBSD's alloca() is in stdlib.h)
#include <alloca.h>
#endif

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

namespace TestConfig {
  // core module
  int num_cpu_procs = 4;
  int num_util_procs = 2;
  int num_io_procs = 1;
  size_t sysmem_size = 16 * 1024 * 1024;
  size_t stack_size = 4 * 1024 * 1024;
  bool pin_util_procs = true;
  bool use_ext_sysmem = false;
  size_t regmem_size = 2 * 1024 * 1024;

  // numa module
  size_t numa_mem_size = 2 * 1024 * 1024;
  size_t numa_nocpu_mem_size = 1 * 1024 * 1024;
  int num_numa_cpus = 1;
  bool numa_pin_memory = true;

  // cuda/hip module
  int num_gpus = 1;
  size_t fb_mem_size = 1024 * 1024 * 1024;
  size_t zc_mem_size = 8 * 1024 * 1024;
  size_t fb_ib_size = 16 * 1024 * 1024;
  size_t zc_ib_size = 32 * 1024 * 1024;
  size_t uvm_mem_size = 64 * 1024 * 1024;
  bool use_dynamic_fb = true;
  size_t dynfb_max_size = 128 * 1024 * 1024;
  unsigned task_streams = 2;
  unsigned d2d_streams = 2;

  // openmp module
  int num_openmp_cpus = 1;
  int num_threads_per_cpu = 2;
  bool openmp_use_numa = false;
  size_t openmp_stack_size = 4 * 1024 * 1024;

  // python module
  int num_python_cpus = 1;
  size_t py_stack_size = 4 * 1024 * 1024;
}; // namespace TestConfig

#ifdef REALM_USE_CUDA
static std::string convert_uuid(char *cu_uuid)
{
  stringbuilder ss;
  ss << "GPU-";
  for(size_t i = 0; i < Cuda::UUID_SIZE; i++) {
    switch(i) {
    case 4:
    case 6:
    case 8:
    case 10:
      ss << '-';
    }
    ss << std::hex << std::setfill('0') << std::setw(2) << (0xFF & (int)cu_uuid[i]);
  }
  return ss;
}

static bool compare_uuid(char *expected, char *actual)
{
  for(size_t i = 0; i < Cuda::UUID_SIZE; i++) {
    if(expected[i] != actual[i]) {
      log_app.error("wrong at %zu", i);
      assert(0);
      return false;
    }
  }
  return true;
}
#endif

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  Runtime rt = Runtime::get_runtime();
  int wrong_config = 0;
  RealmStatus ret_value = REALM_ERROR;

  // core module
  {
    ModuleConfig *core_config = rt.get_module_config("core");
    assert(core_config != NULL);
    int num_cpu_procs = 0;
    int num_util_procs = 0;
    int num_io_procs = 0;
    size_t sysmem_size = 0;
    size_t regmem_size = 0;
    size_t stack_size = 0;
    bool pin_util_procs = false;
    bool use_ext_sysmem = true;
    ASSERT_REALM(core_config->get_property<int>("cpu", num_cpu_procs));
    assert(num_cpu_procs == TestConfig::num_cpu_procs);
    ASSERT_REALM(core_config->get_property<int>("util", num_util_procs));
    assert(num_util_procs == TestConfig::num_util_procs);
    ASSERT_REALM(core_config->get_property<int>("io", num_io_procs));
    assert(num_io_procs == TestConfig::num_io_procs);
    ASSERT_REALM(core_config->get_property<size_t>("sysmem", sysmem_size));
    assert(sysmem_size == TestConfig::sysmem_size);
    ASSERT_REALM(core_config->get_property<size_t>("stack_size", stack_size));
    assert(stack_size == TestConfig::stack_size);
    ASSERT_REALM(core_config->get_property<bool>("pin_util_procs", pin_util_procs));
    assert(pin_util_procs == TestConfig::pin_util_procs);
    ASSERT_REALM(core_config->get_property<bool>("use_ext_sysmem", use_ext_sysmem));
    assert(use_ext_sysmem == TestConfig::use_ext_sysmem);

    ASSERT_REALM(core_config->get_property<size_t>("regmem", regmem_size));
    assert(regmem_size == TestConfig::regmem_size);
    // test wrong property
    ret_value = core_config->get_property<int>("get_error_core", wrong_config);
    assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    log_app.print("cpus %d, utils %d, ios %d, sysmem %zu, regmem %zu", num_cpu_procs,
                  num_util_procs, num_io_procs, sysmem_size, regmem_size);
    // test stack size
#ifdef REALM_ON_WINDOWS
    void *stack_ptr = _alloca(TestConfig::stack_size * 2 / 3);
#else
    void *stack_ptr = alloca(TestConfig::stack_size * 2 / 3);
#endif
    assert(stack_ptr != nullptr);
  }

  // numa module
  ModuleConfig *numa_config = rt.get_module_config("numa");
  if(numa_config) {
    std::vector<size_t> numa_mems;
    ret_value = numa_config->get_resource("numa_mems", numa_mems);
    if(ret_value == REALM_SUCCESS) {
      size_t numa_mem_size = 0;
      size_t numa_nocpu_mem_size = 0;
      int num_numa_cpus = 0;
      bool numa_pin_memory = false;
      ASSERT_REALM(numa_config->get_property("numamem", numa_mem_size));
      assert(numa_mem_size == TestConfig::numa_mem_size);
      ASSERT_REALM(numa_config->get_property("numa_nocpumem", numa_nocpu_mem_size));
      assert(numa_nocpu_mem_size == TestConfig::numa_nocpu_mem_size);
      ASSERT_REALM(numa_config->get_property("numacpus", num_numa_cpus));
      assert(num_numa_cpus == TestConfig::num_numa_cpus);
      ASSERT_REALM(numa_config->get_property("pin_memory", numa_pin_memory));
      assert(numa_pin_memory == TestConfig::numa_pin_memory);
      // test wrong property
      ret_value = numa_config->get_property("get_error_numa", wrong_config);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
      log_app.print("numa numamem %zu, numa_nocpumem %zu, numacpsus %d, numa_pin_memory "
                    "%d",
                    numa_mem_size, numa_nocpu_mem_size, num_numa_cpus, numa_pin_memory);
    } else {
      log_app.warning("numa is not available");
    }
  } else {
    log_app.print("numa is not loaded");
  }

  // cuda/hip module
  ModuleConfig *cuda_config = rt.get_module_config("cuda");
  ModuleConfig *hip_config = rt.get_module_config("hip");
  ModuleConfig *gpu_config = nullptr;
  if(cuda_config) {
    gpu_config = cuda_config;
  } else if(hip_config) {
    gpu_config = hip_config;
  }
  if(gpu_config) {
    int num_gpus = 0;
    size_t fb_mem_size = 0;
    size_t zc_mem_size = 0;
    size_t fb_ib_size = 0;
    size_t zc_ib_size = 0;
    size_t uvm_mem_size = 0;
    bool use_dynamic_fb = false;
    size_t dynfb_max_size = 0;
    unsigned task_streams = 0;
    unsigned d2d_streams = 0;
    ASSERT_REALM(gpu_config->get_property("gpu", num_gpus));
    assert(num_gpus == TestConfig::num_gpus);
    ASSERT_REALM(gpu_config->get_property("fbmem", fb_mem_size));
    assert(fb_mem_size == TestConfig::fb_mem_size);
    ASSERT_REALM(gpu_config->get_property("zcmem", zc_mem_size));
    assert(zc_mem_size == TestConfig::zc_mem_size);
    ASSERT_REALM(gpu_config->get_property("ib_fbmem", fb_ib_size));
    assert(fb_ib_size == TestConfig::fb_ib_size);
    ASSERT_REALM(gpu_config->get_property("ib_zcmem", zc_ib_size));
    assert(zc_ib_size == TestConfig::zc_ib_size);
    // uvm is only available in cuda
    if(cuda_config) {
      ASSERT_REALM(gpu_config->get_property("uvmem", uvm_mem_size));
      assert(uvm_mem_size == TestConfig::uvm_mem_size);
    }
    ASSERT_REALM(gpu_config->get_property("use_dynamic_fb", use_dynamic_fb));
    assert(use_dynamic_fb == TestConfig::use_dynamic_fb);
    ASSERT_REALM(gpu_config->get_property("dynfb_max_size", dynfb_max_size));
    assert(dynfb_max_size == TestConfig::dynfb_max_size);
    ASSERT_REALM(gpu_config->get_property("task_streams", task_streams));
    assert(task_streams == TestConfig::task_streams);
    ASSERT_REALM(gpu_config->get_property("d2d_streams", d2d_streams));
    assert(d2d_streams == TestConfig::d2d_streams);
    // test wrong property
    ret_value = gpu_config->get_property("get_error_cuda", wrong_config);
    assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    log_app.print(
        "cuda gpus %d, fbmem %zu, zcmem %zu, fb_ib %zu, zc_ib %zu, uvm %zu, "
        "use_dynamic_fb %d, dynfb_max_size %zu, task_streams %u, d2d_streams %u",
        num_gpus, fb_mem_size, zc_mem_size, fb_ib_size, zc_ib_size, uvm_mem_size,
        use_dynamic_fb, dynfb_max_size, task_streams, d2d_streams);
  } else {
    log_app.print("cuda/hip is not loaded");
  }

  // openmp module
  ModuleConfig *openmp_config = rt.get_module_config("openmp");
  if(openmp_config) {
    int num_openmp_cpus = 0;
    int num_threads_per_cpu = 0;
    bool openmp_use_numa = false;
    size_t openmp_stack_size = 0;
    ASSERT_REALM(openmp_config->get_property("ocpu", num_openmp_cpus));
    assert(num_openmp_cpus == TestConfig::num_openmp_cpus);
    ASSERT_REALM(openmp_config->get_property("othr", num_threads_per_cpu));
    assert(num_threads_per_cpu == TestConfig::num_threads_per_cpu);
    ASSERT_REALM(openmp_config->get_property("onuma", openmp_use_numa));
    assert(openmp_use_numa == TestConfig::openmp_use_numa);
    ASSERT_REALM(openmp_config->get_property("ostack", openmp_stack_size));
    assert(openmp_stack_size == TestConfig::openmp_stack_size);
    // test wrong property
    ret_value = openmp_config->get_property("get_error_openmp", wrong_config);
    assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    log_app.print("ocpus %d, othr %d, use_numa %d, stack_size %zu", num_openmp_cpus,
                  num_threads_per_cpu, openmp_use_numa, openmp_stack_size);
  } else {
    log_app.print("openmp is not loaded");
  }

  // python module
  ModuleConfig *python_config = rt.get_module_config("python");
  if(python_config) {
    int num_python_cpus = 0;
    size_t py_stack_size = 0;
    ASSERT_REALM(python_config->get_property("pyproc", num_python_cpus));
    assert(num_python_cpus == TestConfig::num_python_cpus);
    ASSERT_REALM(python_config->get_property("pystack", py_stack_size));
    assert(py_stack_size == TestConfig::openmp_stack_size);
    // test wrong property
    ret_value = python_config->get_property("get_error_python", wrong_config);
    assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    log_app.print("python procs %d, stack_size %zu", num_python_cpus, py_stack_size);
  } else {
    log_app.print("python is not loaded");
  }

  // typed access to the same configurations - every *ModuleConfig type can be
  //  named unconditionally (the headers are always installed), and the typed
  //  getter returns a null pointer for modules that are not present
  {
    CoreModuleConfig *core = rt.get_module_config<CoreModuleConfig>();
    assert(core != nullptr);
    assert(static_cast<ModuleConfig *>(core) == rt.get_module_config("core"));
    assert(core->num_cpu_procs == TestConfig::num_cpu_procs);
    assert(core->num_util_procs == TestConfig::num_util_procs);
    assert(core->num_io_procs == TestConfig::num_io_procs);
    assert(core->sysmem_size == TestConfig::sysmem_size);
    assert(core->stack_size == TestConfig::stack_size);
    assert(core->pin_util_procs == TestConfig::pin_util_procs);
    assert(core->use_ext_sysmem == TestConfig::use_ext_sysmem);
    assert(core->reg_mem_size == TestConfig::regmem_size);
    // the "pick based on node count" default must have been resolved by now
    assert(core->reg_ib_mem_size != ~size_t(0));

    Numa::NumaModuleConfig *numa = rt.get_module_config<Numa::NumaModuleConfig>();
    Cuda::CudaModuleConfig *cuda = rt.get_module_config<Cuda::CudaModuleConfig>();
    Hip::HipModuleConfig *hip = rt.get_module_config<Hip::HipModuleConfig>();
    OpenMP::OpenMPModuleConfig *openmp =
        rt.get_module_config<OpenMP::OpenMPModuleConfig>();
    Python::PythonModuleConfig *python =
        rt.get_module_config<Python::PythonModuleConfig>();
    HDF5::HDF5ModuleConfig *hdf5 = rt.get_module_config<HDF5::HDF5ModuleConfig>();
    ProcSet::ProcSetModuleConfig *procset =
        rt.get_module_config<ProcSet::ProcSetModuleConfig>();
    GASNetEXModuleConfig *gasnetex = rt.get_module_config<GASNetEXModuleConfig>();
    UCPModuleConfig *ucx = rt.get_module_config<UCPModuleConfig>();
    MPIModuleConfig *mpi = rt.get_module_config<MPIModuleConfig>();

    // typed and string-based lookups must agree
    assert(static_cast<ModuleConfig *>(numa) == rt.get_module_config("numa"));
    assert(static_cast<ModuleConfig *>(cuda) == rt.get_module_config("cuda"));
    assert(static_cast<ModuleConfig *>(hip) == rt.get_module_config("hip"));
    assert(static_cast<ModuleConfig *>(openmp) == rt.get_module_config("openmp"));
    assert(static_cast<ModuleConfig *>(python) == rt.get_module_config("python"));
    assert(static_cast<ModuleConfig *>(hdf5) == rt.get_module_config("hdf5"));
    assert(static_cast<ModuleConfig *>(procset) == rt.get_module_config("procset"));
    assert(static_cast<ModuleConfig *>(gasnetex) == rt.get_module_config("gasnetex"));
    assert(static_cast<ModuleConfig *>(ucx) == rt.get_module_config("ucx"));
    assert(static_cast<ModuleConfig *>(mpi) == rt.get_module_config("mpi"));

    // modules that are not compiled in must not have a config
    assert(procset != nullptr);
#ifndef REALM_USE_CUDA
    assert(cuda == nullptr);
#endif
#ifndef REALM_USE_HIP
    assert(hip == nullptr);
#endif
#ifndef REALM_USE_OPENMP
    assert(openmp == nullptr);
#endif
#ifndef REALM_USE_PYTHON
    assert(python == nullptr);
#endif
#ifndef REALM_USE_HDF5
    assert(hdf5 == nullptr);
#endif
#ifndef REALM_USE_GASNETEX
    assert(gasnetex == nullptr);
#endif
#ifndef REALM_USE_UCX
    assert(ucx == nullptr);
#endif
#ifndef REALM_USE_MPI
    assert(mpi == nullptr);
#endif

    std::vector<size_t> numa_mems;
    if(numa && (numa->get_resource("numa_mems", numa_mems) == REALM_SUCCESS) &&
       !numa_mems.empty()) {
      assert(numa->cfg_numa_mem_size == TestConfig::numa_mem_size);
      assert(numa->cfg_numa_nocpu_mem_size ==
             static_cast<ssize_t>(TestConfig::numa_nocpu_mem_size));
      assert(numa->cfg_num_numa_cpus == TestConfig::num_numa_cpus);
      assert(numa->cfg_pin_memory == TestConfig::numa_pin_memory);
    }
    if(cuda) {
      assert(cuda->cfg_num_gpus == TestConfig::num_gpus);
      assert(cuda->cfg_fb_mem_size == TestConfig::fb_mem_size);
      assert(cuda->cfg_zc_mem_size == TestConfig::zc_mem_size);
      assert(cuda->cfg_fb_ib_size == TestConfig::fb_ib_size);
      assert(cuda->cfg_zc_ib_size == TestConfig::zc_ib_size);
      assert(cuda->cfg_uvm_mem_size == TestConfig::uvm_mem_size);
      assert(cuda->cfg_use_dynamic_fb == TestConfig::use_dynamic_fb);
      assert(cuda->cfg_dynfb_max_size == TestConfig::dynfb_max_size);
      assert(cuda->cfg_task_streams == TestConfig::task_streams);
      assert(cuda->cfg_d2d_streams == TestConfig::d2d_streams);
    }
    if(hip) {
      assert(hip->cfg_num_gpus == TestConfig::num_gpus);
      assert(hip->cfg_fb_mem_size == TestConfig::fb_mem_size);
      assert(hip->cfg_zc_mem_size == TestConfig::zc_mem_size);
      assert(hip->cfg_fb_ib_size == TestConfig::fb_ib_size);
      assert(hip->cfg_zc_ib_size == TestConfig::zc_ib_size);
      assert(hip->cfg_use_dynamic_fb == TestConfig::use_dynamic_fb);
      assert(hip->cfg_dynfb_max_size == TestConfig::dynfb_max_size);
      assert(hip->cfg_task_streams == TestConfig::task_streams);
      assert(hip->cfg_d2d_streams == TestConfig::d2d_streams);
    }
    if(openmp) {
      assert(openmp->cfg_num_openmp_cpus == TestConfig::num_openmp_cpus);
      assert(openmp->cfg_num_threads_per_cpu == TestConfig::num_threads_per_cpu);
      assert(openmp->cfg_use_numa == TestConfig::openmp_use_numa);
      assert(openmp->cfg_stack_size == TestConfig::openmp_stack_size);
    }
    if(python) {
      assert(python->cfg_num_python_cpus == TestConfig::num_python_cpus);
      assert(python->cfg_stack_size == TestConfig::py_stack_size);
    }
    log_app.print("typed module configs: core cpus=%d, present: numa=%d cuda=%d hip=%d "
                  "openmp=%d python=%d hdf5=%d procset=%d gasnetex=%d ucx=%d mpi=%d",
                  core->num_cpu_procs, numa != nullptr, cuda != nullptr, hip != nullptr,
                  openmp != nullptr, python != nullptr, hdf5 != nullptr,
                  procset != nullptr, gasnetex != nullptr, ucx != nullptr,
                  mpi != nullptr);
  }

  if(Network::my_node_id == 0) {
    int num_cpu_procs = 0;
    int num_util_procs = 0;
    int num_io_procs = 0;
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    int num_gpus = 0;
#endif
#ifdef REALM_USE_OPENMP
    int num_openmp_cpus = 0;
#endif
#ifdef REALM_USE_PYTHON
    int num_python_cpus = 0;
#endif
    Machine machine = Machine::get_machine();
    int num_nodes = static_cast<int>(machine.get_address_space_count());
    // iterate all processors within the machine.
    for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin();
        it; ++it) {
      Processor proc = *it;
      Processor::Kind kind = proc.kind();
      Machine::ProcessInfo process_info;
      bool ret_value = machine.get_process_info(proc, &process_info);
      assert(ret_value);
      log_app.print() << "Process " << proc << ", process_id:" << process_info.processid
                      << ", host_id:" << process_info.hostid
                      << ", hostname:" << process_info.hostname;
      switch(kind) {
      case Processor::LOC_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is CPU.", p.address_space(),
                      proc.id);
        num_cpu_procs++;
        break;
      }
      case Processor::UTIL_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is utility.", p.address_space(),
                      proc.id);
        num_util_procs++;
        break;
      }
      case Processor::IO_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is I/O Proc.", p.address_space(),
                      proc.id);
        num_io_procs++;
        break;
      }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
      case Processor::TOC_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is GPU.", p.address_space(),
                      proc.id);
#ifdef REALM_USE_CUDA
        Cuda::Uuid cuda_uuid;
        ret_value = Cuda::get_cuda_device_uuid(proc, &cuda_uuid);
        if(ret_value) {
          int device_id;
          assert(Cuda::get_cuda_device_id(proc, &device_id));
          CUuuid expected_uuid;
          cuDeviceGetUuid(&expected_uuid, device_id);
          assert(compare_uuid(reinterpret_cast<char *>(&expected_uuid), &(cuda_uuid[0])));
          std::string uuid = convert_uuid(&(cuda_uuid[0]));
          log_app.print("GPU Processor %llx, devide_id %d, uuid:%s", proc.id, device_id,
                        uuid.c_str());
        }
#endif
        num_gpus++;
        break;
      }
#endif
#ifdef REALM_USE_OPENMP
      case Processor::OMP_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is OMP.", p.address_space(),
                      proc.id);
        num_openmp_cpus++;
        break;
      }
#endif
#ifdef REALM_USE_PYTHON
      case Processor::PY_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is Python.", p.address_space(),
                      proc.id);
        num_python_cpus++;
        break;
      }
#endif
      default:
      {
        break;
      }
      }
    }
    Numa::NumaModule *numa_module = rt.get_module<Numa::NumaModule>("numa");
    if(numa_module) {
      int num_numas = numa_module->numa_cpu_counts.size();
      assert(num_cpu_procs ==
             (TestConfig::num_cpu_procs + TestConfig::num_numa_cpus * num_numas) *
                 num_nodes);
    } else {
      assert(num_cpu_procs == TestConfig::num_cpu_procs * num_nodes);
    }
    assert(num_util_procs == TestConfig::num_util_procs * num_nodes);
    assert(num_io_procs == TestConfig::num_io_procs * num_nodes);
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    assert(num_gpus == TestConfig::num_gpus * num_nodes);
#endif
#ifdef REALM_USE_OPENMP
    assert(num_openmp_cpus == TestConfig::num_openmp_cpus * num_nodes);
#endif
#ifdef REALM_USE_PYTHON
    assert(num_python_cpus == TestConfig::num_python_cpus * num_nodes);
#endif

    size_t sysmem_size = 0;
    size_t regmem_size = 0;
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    size_t fb_mem_size = 0;
    size_t zc_mem_size = 0;
    size_t uvm_mem_size = 0;
    size_t dynfb_max_size = 0;
#endif
    // iterate all memories within the machine.
    for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
        ++it) {
      Memory p = *it;
      Memory::Kind kind = p.kind();
      switch(kind) {
      case Memory::SYSTEM_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is SYSTEM MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        sysmem_size += p.capacity();
        break;
      }
      case Memory::REGDMA_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is REGDMA MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        regmem_size += p.capacity();
        break;
      }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
      case Memory::GPU_FB_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is FB MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        fb_mem_size += p.capacity();
        break;
      }
      case Memory::Z_COPY_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is ZC MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        zc_mem_size += p.capacity();
        break;
      }
      case Memory::GPU_MANAGED_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is GPU MANAGED MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        uvm_mem_size += p.capacity();
        break;
      }
      case Memory::GPU_DYNAMIC_MEM:
      {
        log_app.print("Rank %u, Memory ID " IDFMT " is GPU DYNAMIC FB MEM, size %zu.",
                      p.address_space(), p.id, p.capacity());
        dynfb_max_size += p.capacity();
        break;
      }
#endif
      default:
      {
        break;
      }
      }
    }
    assert(sysmem_size == TestConfig::sysmem_size * num_nodes);
    assert(regmem_size == TestConfig::regmem_size * num_nodes);
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    assert(fb_mem_size == TestConfig::fb_mem_size * TestConfig::num_gpus * num_nodes);
    assert(zc_mem_size == TestConfig::zc_mem_size * num_nodes);
#ifdef REALM_USE_CUDA
    // uvm is only available in cuda
    assert(uvm_mem_size == TestConfig::uvm_mem_size * num_nodes);
#endif
    assert(dynfb_max_size ==
           TestConfig::dynfb_max_size * TestConfig::num_gpus * num_nodes);
#endif
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.network_init(&argc, &argv);

  rt.create_configs(argc, argv);

  std::vector<std::string> cmdline;
  cmdline.reserve(argc);
  for(int i = 1; i < argc; i++) {
    cmdline.push_back(argv[i]);
  }

  // rt.parse_command_line(cmdline, true);
  CommandLineParser cp;
  int test_args = 0;
  cp.add_option_int("-test_args", test_args);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  if(test_args == 0) {
    // core module
    ModuleConfig *core_config = rt.get_module_config("core");
    assert(core_config != NULL);
    RealmStatus ret_value = REALM_ERROR;
    {
      int ncores = 0;
      size_t sysmem = 0;
      ASSERT_REALM(core_config->get_resource("cpu", ncores));
      ASSERT_REALM(core_config->get_resource("sysmem", sysmem));
      log_app.print("Discover number of CPU cores %d, sysmem %zu", ncores, sysmem);
      ASSERT_REALM(core_config->set_property<int>("cpu", TestConfig::num_cpu_procs));
      ASSERT_REALM(core_config->set_property<int>("util", TestConfig::num_util_procs));
      ASSERT_REALM(core_config->set_property<int>("io", TestConfig::num_io_procs));
      ASSERT_REALM(core_config->set_property<size_t>("sysmem", TestConfig::sysmem_size));
      ASSERT_REALM(
          core_config->set_property<size_t>("stack_size", TestConfig::stack_size));
      ASSERT_REALM(
          core_config->set_property<bool>("pin_util_procs", TestConfig::pin_util_procs));
      ASSERT_REALM(
          core_config->set_property<bool>("use_ext_sysmem", TestConfig::use_ext_sysmem));
      ASSERT_REALM(core_config->set_property<size_t>("regmem", TestConfig::regmem_size));
      // test wrong config
      ret_value = core_config->set_property("set_error_core", TestConfig::sysmem_size);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
      // test wrong type - "cpu" is an int, so neither a size_t write nor a
      //  bool read may touch it
      ret_value = core_config->set_property<size_t>("cpu", 1);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_TYPE);
      bool wrong_type = true;
      ret_value = core_config->get_property("cpu", wrong_type);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_TYPE);
      int cpus_after = 0;
      ASSERT_REALM(core_config->get_property("cpu", cpus_after));
      assert(cpus_after == TestConfig::num_cpu_procs);
    }

    // numa module
    ModuleConfig *numa_config = rt.get_module_config("numa");
    if(numa_config) {
      std::vector<size_t> numa_mems;
      ret_value = numa_config->get_resource("numa_mems", numa_mems);
      if(ret_value == REALM_SUCCESS) {
        assert(numa_mems.size() >= 1);
        log_app.info() << "numa mems:" << PrettyVector<size_t>(numa_mems);
        ASSERT_REALM(
            numa_config->set_property<size_t>("numamem", TestConfig::numa_mem_size));
        ASSERT_REALM(numa_config->set_property<size_t>("numa_nocpumem",
                                                       TestConfig::numa_nocpu_mem_size));
        ASSERT_REALM(
            numa_config->set_property<int>("numacpus", TestConfig::num_numa_cpus));
        ASSERT_REALM(
            numa_config->set_property<bool>("pin_memory", TestConfig::numa_pin_memory));
        // test wrong config
        ret_value =
            numa_config->set_property("set_error_numa", TestConfig::numa_mem_size);
        assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
        ASSERT_REALM(
            numa_config->set_property<bool>("pin_memory", TestConfig::numa_pin_memory));
        // test wrong config
        ret_value =
            numa_config->set_property("set_error_numa", TestConfig::numa_mem_size);
        assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
      } else {
        log_app.warning("numa is not available");
      }
    } else {
      log_app.print("numa is not loaded");
    }

    // cuda/hip module
    ModuleConfig *cuda_config = rt.get_module_config("cuda");
    ModuleConfig *hip_config = rt.get_module_config("hip");
    ModuleConfig *gpu_config = nullptr;
    if(cuda_config) {
      gpu_config = cuda_config;
    } else if(hip_config) {
      gpu_config = hip_config;
    }
    if(gpu_config) {
      int ngpus = 0;
      size_t fbmem = 0;
      ASSERT_REALM(gpu_config->get_resource("gpu", ngpus));
      ASSERT_REALM(gpu_config->get_resource("fbmem", fbmem));
      log_app.print("number of cuda GPUs %d, fbmem size %zu", ngpus, fbmem);
      ASSERT_REALM(gpu_config->set_property<int>("gpu", TestConfig::num_gpus));
      ASSERT_REALM(gpu_config->set_property<size_t>("zcmem", TestConfig::zc_mem_size));
      ASSERT_REALM(gpu_config->set_property<size_t>("fbmem", TestConfig::fb_mem_size));
      ASSERT_REALM(gpu_config->set_property<size_t>("ib_fbmem", TestConfig::fb_ib_size));
      ASSERT_REALM(gpu_config->set_property<size_t>("ib_zcmem", TestConfig::zc_ib_size));
      // uvm is only available in cuda
      if(cuda_config) {
        ASSERT_REALM(gpu_config->set_property<size_t>("uvmem", TestConfig::uvm_mem_size));
      }
      ASSERT_REALM(
          gpu_config->set_property<bool>("use_dynamic_fb", TestConfig::use_dynamic_fb));
      ASSERT_REALM(
          gpu_config->set_property<size_t>("dynfb_max_size", TestConfig::dynfb_max_size));
      ASSERT_REALM(
          gpu_config->set_property<unsigned>("task_streams", TestConfig::task_streams));
      ASSERT_REALM(
          gpu_config->set_property<unsigned>("d2d_streams", TestConfig::d2d_streams));
      // test wrong config
      ret_value = gpu_config->set_property("set_error_cuda", TestConfig::fb_mem_size);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
      ASSERT_REALM(
          gpu_config->set_property<unsigned>("d2d_streams", TestConfig::d2d_streams));
      // test wrong config
      ret_value = gpu_config->set_property("set_error_cuda", TestConfig::fb_mem_size);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    } else {
      log_app.print("cuda/hip is not loaded");
    }

    // openmp module
    ModuleConfig *openmp_config = rt.get_module_config("openmp");
    if(openmp_config) {
      ASSERT_REALM(openmp_config->set_property<int>("ocpu", TestConfig::num_openmp_cpus));
      ASSERT_REALM(
          openmp_config->set_property<int>("othr", TestConfig::num_threads_per_cpu));
      ASSERT_REALM(
          openmp_config->set_property<bool>("onuma", TestConfig::openmp_use_numa));
      ASSERT_REALM(
          openmp_config->set_property<size_t>("ostack", TestConfig::openmp_stack_size));
      // test wrong config
      ret_value =
          openmp_config->set_property("set_error_openmp", TestConfig::num_openmp_cpus);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    } else {
      log_app.print("openmp is not loaded");
    }

    // python module
    ModuleConfig *python_config = rt.get_module_config("python");
    if(python_config) {
      ASSERT_REALM(
          python_config->set_property<int>("pyproc", TestConfig::num_python_cpus));
      ASSERT_REALM(
          python_config->set_property<size_t>("pystack", TestConfig::py_stack_size));
      // test wrong config
      ret_value =
          python_config->set_property("set_error_openmp", TestConfig::num_openmp_cpus);
      assert(ret_value == REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    } else {
      log_app.print("python is not loaded");
    }
  } else if(test_args == 2) {
    // configure through the typed structs instead of the string API - the
    //  config types for modules that are not compiled in can still be named
    //  here, their pointers are simply null
    CoreModuleConfig *core = rt.get_module_config<CoreModuleConfig>();
    assert(core != nullptr);
    {
      int ncores = 0;
      size_t sysmem = 0;
      ASSERT_REALM(core->get_resource("cpu", ncores));
      ASSERT_REALM(core->get_resource("sysmem", sysmem));
      log_app.print("Discover number of CPU cores %d, sysmem %zu", ncores, sysmem);

      // a configuration can be prepared by value (starting from Realm's
      //  defaults, or as a copy of the runtime's object) and then assigned to
      //  the runtime's object - that copies the settings, but leaves the
      //  discovered resources (which are not settings) alone
      CoreModuleConfig cfg;
      assert(cfg.num_cpu_procs == 1); // Realm's default
      cfg.num_cpu_procs = TestConfig::num_cpu_procs;
      cfg.num_util_procs = TestConfig::num_util_procs;
      cfg.num_io_procs = TestConfig::num_io_procs;
      cfg.sysmem_size = TestConfig::sysmem_size;
      cfg.stack_size = TestConfig::stack_size;
      cfg.pin_util_procs = TestConfig::pin_util_procs;
      cfg.use_ext_sysmem = TestConfig::use_ext_sysmem;
      cfg.reg_mem_size = TestConfig::regmem_size;
      *core = cfg;
      assert(core->num_cpu_procs == TestConfig::num_cpu_procs);
      assert(core->get_name() == CoreModuleConfig::MODULE_NAME);
      int ncores_after = 0;
      size_t sysmem_after = 0;
      ASSERT_REALM(core->get_resource("cpu", ncores_after));
      ASSERT_REALM(core->get_resource("sysmem", sysmem_after));
      assert((ncores_after == ncores) && (sysmem_after == sysmem));

      // a copy of the runtime's object agrees with it, including through the
      //  string interface, and modifying the copy does not touch the original
      CoreModuleConfig copy(*core);
      int copied_cpus = 0;
      ASSERT_REALM(copy.get_property("cpu", copied_cpus));
      assert(copied_cpus == TestConfig::num_cpu_procs);
      ASSERT_REALM(copy.get_resource("cpu", ncores_after));
      assert(ncores_after == ncores);
      copy.num_cpu_procs = TestConfig::num_cpu_procs + 1;
      assert(core->num_cpu_procs == TestConfig::num_cpu_procs);

      // the string interface works on any object of a present module's
      //  configuration class, not just the runtime's own
      CoreModuleConfig fresh;
      ASSERT_REALM(fresh.set_property<int>("cpu", 7));
      assert(fresh.num_cpu_procs == 7);
      // ... but a fresh object has not discovered any resources
      assert(fresh.get_resource("cpu", ncores_after) ==
             REALM_MODULE_CONFIG_ERROR_NO_RESOURCE);
      // the configuration class of a module that is not present can be
      //  constructed and copied, but has nothing to look names up in
      assert(sizeof(MPIModuleConfig) > 0);
#ifndef REALM_USE_MPI
      MPIModuleConfig absent;
      size_t gsize = 0;
      assert(absent.get_property("global_mem_size", gsize) ==
             REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
#endif
    }

    Numa::NumaModuleConfig *numa = rt.get_module_config<Numa::NumaModuleConfig>();
    std::vector<size_t> numa_mems;
    if(numa && (numa->get_resource("numa_mems", numa_mems) == REALM_SUCCESS) &&
       !numa_mems.empty()) {
      log_app.info() << "numa mems:" << PrettyVector<size_t>(numa_mems);
      numa->cfg_numa_mem_size = TestConfig::numa_mem_size;
      numa->cfg_numa_nocpu_mem_size = TestConfig::numa_nocpu_mem_size;
      numa->cfg_num_numa_cpus = TestConfig::num_numa_cpus;
      numa->cfg_pin_memory = TestConfig::numa_pin_memory;
    } else {
      log_app.print("numa is not available");
    }

    Cuda::CudaModuleConfig *cuda = rt.get_module_config<Cuda::CudaModuleConfig>();
    if(cuda) {
      int ngpus = 0;
      size_t fbmem = 0;
      ASSERT_REALM(cuda->get_resource("gpu", ngpus));
      ASSERT_REALM(cuda->get_resource("fbmem", fbmem));
      log_app.print("number of cuda GPUs %d, fbmem size %zu", ngpus, fbmem);
      cuda->cfg_num_gpus = TestConfig::num_gpus;
      cuda->cfg_zc_mem_size = TestConfig::zc_mem_size;
      cuda->cfg_fb_mem_size = TestConfig::fb_mem_size;
      cuda->cfg_fb_ib_size = TestConfig::fb_ib_size;
      cuda->cfg_zc_ib_size = TestConfig::zc_ib_size;
      cuda->cfg_uvm_mem_size = TestConfig::uvm_mem_size;
      cuda->cfg_use_dynamic_fb = TestConfig::use_dynamic_fb;
      cuda->cfg_dynfb_max_size = TestConfig::dynfb_max_size;
      cuda->cfg_task_streams = TestConfig::task_streams;
      cuda->cfg_d2d_streams = TestConfig::d2d_streams;
    }
    Hip::HipModuleConfig *hip = rt.get_module_config<Hip::HipModuleConfig>();
    if(hip) {
      int ngpus = 0;
      size_t fbmem = 0;
      ASSERT_REALM(hip->get_resource("gpu", ngpus));
      ASSERT_REALM(hip->get_resource("fbmem", fbmem));
      log_app.print("number of hip GPUs %d, fbmem size %zu", ngpus, fbmem);
      hip->cfg_num_gpus = TestConfig::num_gpus;
      hip->cfg_zc_mem_size = TestConfig::zc_mem_size;
      hip->cfg_fb_mem_size = TestConfig::fb_mem_size;
      hip->cfg_fb_ib_size = TestConfig::fb_ib_size;
      hip->cfg_zc_ib_size = TestConfig::zc_ib_size;
      hip->cfg_use_dynamic_fb = TestConfig::use_dynamic_fb;
      hip->cfg_dynfb_max_size = TestConfig::dynfb_max_size;
      hip->cfg_task_streams = TestConfig::task_streams;
      hip->cfg_d2d_streams = TestConfig::d2d_streams;
    }
    if(!cuda && !hip) {
      log_app.print("cuda/hip is not loaded");
    }

    OpenMP::OpenMPModuleConfig *openmp =
        rt.get_module_config<OpenMP::OpenMPModuleConfig>();
    if(openmp) {
      openmp->cfg_num_openmp_cpus = TestConfig::num_openmp_cpus;
      openmp->cfg_num_threads_per_cpu = TestConfig::num_threads_per_cpu;
      openmp->cfg_use_numa = TestConfig::openmp_use_numa;
      openmp->cfg_stack_size = TestConfig::openmp_stack_size;
    } else {
      log_app.print("openmp is not loaded");
    }

    Python::PythonModuleConfig *python =
        rt.get_module_config<Python::PythonModuleConfig>();
    if(python) {
      python->cfg_num_python_cpus = TestConfig::num_python_cpus;
      python->cfg_stack_size = TestConfig::py_stack_size;
    } else {
      log_app.print("python is not loaded");
    }
  }

  rt.parse_command_line(cmdline, true);
  rt.finish_configure();
  rt.start();

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // Collective launch a single task on each CPU processor of each node, that
  // just means this test will be done once on each node
  Event e = rt.collective_spawn_by_kind(Processor::LOC_PROC, TOP_LEVEL_TASK, NULL, 0,
                                        true /*one per node*/);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
