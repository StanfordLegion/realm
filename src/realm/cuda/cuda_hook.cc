
#include "realm/cuda/cuda_hook.h"
#include "realm/cuda/cuda_module.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>

namespace Realm {

  namespace Cuda {

    extern Logger log_gpu;

    Logger log_cuhook("cuhook");

    namespace ThreadLocal {
      extern thread_local GPUStream
          *current_gpu_stream; // declared in cuda_module.cc
      static thread_local std::unordered_map<
          CUstream, std::pair<CUpti_CallbackId, CUevent>> *cuhook_stream_status = nullptr;
      static thread_local int nb_hooked_functions_per_task = 0;
    }; // namespace ThreadLocal

    static void CUPTIAPI cuhook_callback(void *userdata, CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
                                         const CUpti_CallbackData *cbInfo)
    {
      if(ThreadLocal::current_gpu_stream == nullptr) {
        const char *function_name = cbInfo->functionName;
        log_cuhook.debug("Callback outside task, function: %s", function_name);
        return;
      }

      if(domain != CUPTI_CB_DOMAIN_DRIVER_API ||
         cbInfo->callbackSite != CUPTI_API_ENTER) {
        return;
      }

      enum class CallbackOperation
      {
        RECORD,
        CLEAR_STREAM,
        CLEAR_EVENT,
        CLEAR_CTX,
      };

      CallbackOperation operation = CallbackOperation::RECORD;
      const char *function_name = cbInfo->functionName;
      const char *symbol_name = nullptr;
      CUstream stream = nullptr;
      CUevent event = nullptr;

      switch(cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      {
        symbol_name = cbInfo->symbolName;
        stream = ((cuLaunchKernel_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      {
        stream = ((cuMemcpyHtoDAsync_v2_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      {
        stream = ((cuMemcpyDtoHAsync_v2_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
      {
        stream = ((cuMemcpyDtoDAsync_v2_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
      {
        stream = ((cuMemcpy2DAsync_v2_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
      {
        stream = ((cuMemcpy3DAsync_v2_params *)(cbInfo->functionParams))->hStream;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
      {
        stream = ((cuStreamSynchronize_params *)(cbInfo->functionParams))->hStream;
        operation = CallbackOperation::CLEAR_STREAM;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuEventRecord:
      {
        stream = ((cuEventRecord_params *)(cbInfo->functionParams))->hStream;
        event = ((cuEventRecord_params *)(cbInfo->functionParams))->hEvent;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
      {
        event = ((cuEventSynchronize_params *)(cbInfo->functionParams))->hEvent;
        operation = CallbackOperation::CLEAR_EVENT;
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
      {
        operation = CallbackOperation::CLEAR_CTX;
        break;
      }
      default:
      {
        return;
      }
      }

      ThreadLocal::nb_hooked_functions_per_task++;

      switch(operation) {
      case CallbackOperation::RECORD:
      {
        log_cuhook.debug("Record driver API: %s, %s on stream %p, event %p", symbol_name,
                         function_name, (void *)stream, (void *)event);
        ThreadLocal::cuhook_stream_status->operator[](stream) =
            std::make_pair(cbid, event);
        break;
      }
      case CallbackOperation::CLEAR_STREAM:
      {
        log_cuhook.debug("Clear Stream driver API: %s on stream %p", function_name,
                         (void *)stream);
        auto it = ThreadLocal::cuhook_stream_status->find(stream);
        if(it != ThreadLocal::cuhook_stream_status->end()) {
          ThreadLocal::cuhook_stream_status->erase(it);
        }
        break;
      }
      case CallbackOperation::CLEAR_EVENT:
      {
        log_cuhook.debug("Clear Event driver API: %s on stream %p, event %p",
                         function_name, (void *)stream, (void *)event);
        for(auto it = ThreadLocal::cuhook_stream_status->begin();
            it != ThreadLocal::cuhook_stream_status->end(); ++it) {
          if(it->second.second == event) {
            ThreadLocal::cuhook_stream_status->erase(it);
            break;
          }
        }
        break;
      }
      case CallbackOperation::CLEAR_CTX:
      {
        log_cuhook.debug("Clear CTX driver API: %s", function_name);
        ThreadLocal::cuhook_stream_status->clear();
        break;
      }
      default:
        break;
      }
    }

    static void cuhook_stream_sanity_check(CUstream current_task_stream,
                                           Processor::TaskFuncID task_id)
    {
      if(ThreadLocal::cuhook_stream_status->size() == 0) {
        log_cuhook.info("END Task %u, cuda stream sanity check: safe, nb_calls %d",
                         task_id, ThreadLocal::nb_hooked_functions_per_task);
      } else {
        // we remove streams that are realm's task streams
        std::unordered_map<CUstream, std::pair<CUpti_CallbackId, CUevent>>::iterator
            stream_it = ThreadLocal::cuhook_stream_status->find(current_task_stream);
        if(stream_it != ThreadLocal::cuhook_stream_status->end()) {
          ThreadLocal::cuhook_stream_status->erase(stream_it);
        }
        if(ThreadLocal::cuhook_stream_status->size() == 0) {
          log_cuhook.info("END Task %u, cuda stream sanity check: safe, nb calls %d",
                           task_id, ThreadLocal::nb_hooked_functions_per_task);
        } else {
          log_cuhook.warning(
              "END Task %u, cuda stream sanity check: unsafe, leak %ld, nb calls %d",
              task_id, ThreadLocal::cuhook_stream_status->size(),
              ThreadLocal::nb_hooked_functions_per_task);
        }
      }
    }

    CudaHook::CudaHook()
    {
      CHECK_CUPTI(CUPTI_FNPTR(cuptiSubscribe)(&cupti_subscriber,
                                              (CUpti_CallbackFunc)cuhook_callback, NULL));
      // init cuda driver api callbacks
      const CUpti_CallbackId enabled_callbacks[] = {
          // launch kernel
          CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
          // memcpy
          CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
          CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
          CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2,
          CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2,
          CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2,
          // stream
          CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
          // event
          CUPTI_DRIVER_TRACE_CBID_cuEventRecord,
          CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize,
          // ctx
          CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize,
      };

      for(const auto &cbid : enabled_callbacks) {
        CHECK_CUPTI(CUPTI_FNPTR(cuptiEnableCallback)(1, cupti_subscriber,
                                                     CUPTI_CB_DOMAIN_DRIVER_API, cbid));
      }
    }

    CudaHook::~CudaHook()
    {
      CHECK_CUPTI(CUPTI_FNPTR(cuptiUnsubscribe)(cupti_subscriber));
    }

    void CudaHook::start_task(GPUStream *current_stream)
    {
      assert(ThreadLocal::current_gpu_stream == current_stream);
      assert(ThreadLocal::cuhook_stream_status == nullptr);
      ThreadLocal::cuhook_stream_status =
          new std::unordered_map<CUstream, std::pair<CUpti_CallbackId, CUevent>>();
      ThreadLocal::nb_hooked_functions_per_task = 0;
    }

    void CudaHook::end_task(GPUStream *current_task_stream, Processor::TaskFuncID task_id)
    {
      assert(ThreadLocal::current_gpu_stream == current_task_stream);
      cuhook_stream_sanity_check(current_task_stream->get_stream(), task_id);
      delete ThreadLocal::cuhook_stream_status;
      ThreadLocal::cuhook_stream_status = nullptr;
    }
  }; // namespace Cuda
}; // namespace Realm