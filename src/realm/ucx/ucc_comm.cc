/*
 * Copyright 2025 NVIDIA Corporation
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

#include <algorithm>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <numeric>

#include "realm/logging.h"
#include "bootstrap/bootstrap.h"
#include "ucc_comm.h"

#if defined(REALM_UCX_DYNAMIC_LOAD)
#include <dlfcn.h>
#endif
namespace Realm {
  Logger log_ucc("ucc");

  namespace ucc {

    // clang-format off
#define UCC_APIS(__op__)                 \
  __op__(ucc_get_version);               \
  __op__(ucc_status_string);             \
  __op__(ucc_lib_config_read);           \
  __op__(ucc_lib_config_release);        \
  __op__(ucc_init_version);              \
  __op__(ucc_finalize);                  \
  __op__(ucc_context_config_read);       \
  __op__(ucc_context_config_release);    \
  __op__(ucc_context_create);            \
  __op__(ucc_context_progress);          \
  __op__(ucc_context_destroy);           \
  __op__(ucc_team_create_post);          \
  __op__(ucc_team_create_test);          \
  __op__(ucc_team_destroy);              \
  __op__(ucc_collective_init);           \
  __op__(ucc_collective_post);           \
  __op__(ucc_collective_finalize);
    // clang-format on

    static void *libucc_handle = nullptr;

#ifdef REALM_UCX_DYNAMIC_LOAD
#define DEFINE_FNPTR(name) decltype(&name) name##_fnptr = 0;
    UCC_APIS(DEFINE_FNPTR);
#undef DEFINE_FNPTR
#define UCC_FUNC(name) name##_fnptr
#else
#define UCC_FUNC(name) name
#endif

    UCCComm::UCCComm(int _rank, int _world_sz, bootstrap_handle_t *bh)
      : rank(_rank)
      , world_sz(_world_sz)
      , oob_comm(rank, world_sz, bh)
    {}

    template <typename F>
    static ucc_status_t wait(ucc_context_h context, F f)
    {
      ucc_status_t status = f();
      while(status == UCC_INPROGRESS) {
        status = UCC_FUNC(ucc_context_progress)(context);
        if(status == UCC_OK) {
          status = f();
        }
      }
      return status;
    }

    static bool resolve_fnptrs()
    {
#if defined(REALM_UCX_DYNAMIC_LOAD)
      if(libucc_handle == nullptr) {
        libucc_handle = dlopen("libucc.so.1", RTLD_NOW);
        if(libucc_handle == nullptr) {
          log_ucc.warning("Failed to load libucc.so.1: %s", dlerror());
          return false;
        }
      }

#define STRINGIFY(s) #s
#define UCC_GET_FNPTR(name)                                                              \
  do {                                                                                   \
    void *sym = dlsym(libucc_handle, STRINGIFY(name));                                   \
    if(sym == nullptr) {                                                                 \
      log_ucc.warning() << "symbol '" STRINGIFY(name) "' missing from libucc.so!";       \
    } else {                                                                             \
      name##_fnptr = reinterpret_cast<decltype(&name)>(sym);                             \
    }                                                                                    \
  } while(0)

      UCC_APIS(UCC_GET_FNPTR);

#undef UCC_GET_FNPTR
#undef STRINGIFY

      return true;
#else
      return UCC_OK;
#endif
    }

    static ucc_status_t init_lib(ucc_lib_h &lib)
    {
      ucc_lib_config_h lib_config{};
      ucc_lib_params_t lib_params{};
      ucc_status_t status = UCC_OK;

      status = UCC_FUNC(ucc_lib_config_read)(/* env_prefix */ nullptr,
                                             /* filename */ nullptr, &lib_config);

      if(UCC_OK != status) {
        log_ucc.error() << "Failed to read the library configuration\n";
        goto Done;
      }

      lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
      lib_params.thread_mode = UCC_THREAD_MULTIPLE;

      status = UCC_FUNC(ucc_init_version)(UCC_API_MAJOR, UCC_API_MINOR, &lib_params,
                                          lib_config, &lib);

      if(UCC_OK != status) {
        log_ucc.error() << "UCCLayer : Failed to initialize the ucc library\n";
        goto Done;
      }

      log_ucc.info() << "UCC library configured successfully\n";

    Done:
      UCC_FUNC(ucc_lib_config_release)(lib_config);
      return status;
    }

    static ucc_status_t create_context(ucc_lib_h lib, OOBGroupComm &oob_comm,
                                       ucc_context_h &context)
    {
      ucc_context_config_h ctx_config{};
      ucc_context_params_t ctx_params{};
      ucc_status_t status = UCC_OK;

      status = UCC_FUNC(ucc_context_config_read)(lib, NULL, &ctx_config);
      if(UCC_OK != status) {
        log_ucc.error() << "Failed to read context config\n";
        return status;
      }

      ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
      ctx_params.type = UCC_CONTEXT_SHARED;
      ctx_params.oob.allgather = ucc::OOBGroupComm::oob_allgather;
      ctx_params.oob.req_test = ucc::OOBGroupComm::oob_allgather_test;
      ctx_params.oob.req_free = ucc::OOBGroupComm::oob_allgather_free;

      ctx_params.oob.coll_info = oob_comm.get_coll_info();

      ctx_params.oob.n_oob_eps = static_cast<uint32_t>(oob_comm.get_world_size());
      ctx_params.oob.oob_ep = static_cast<uint32_t>(oob_comm.get_rank());

      status = UCC_FUNC(ucc_context_create)(lib, &ctx_params, ctx_config, &context);
      if(UCC_OK != status) {
        log_ucc.error() << "UCCComm : Failed to create ucc context\n";
        goto Done;
      }

      log_ucc.info() << "UCC Context created successfully\n";

    Done:
      UCC_FUNC(ucc_context_config_release)(ctx_config);
      return status;
    }

    static ucc_status_t create_team(ucc_context_h context, OOBGroupComm &oob_comm,
                                    ucc_team_h &team)
    {
      ucc_team_params team_params;
      ucc_status_t status;

      team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;
      team_params.ordering = UCC_COLLECTIVE_POST_ORDERED;
      team_params.oob.coll_info = oob_comm.get_coll_info();

      team_params.oob.allgather = ucc::OOBGroupComm::oob_allgather;
      team_params.oob.req_test = ucc::OOBGroupComm::oob_allgather_test;
      team_params.oob.req_free = ucc::OOBGroupComm::oob_allgather_free;

      team_params.oob.n_oob_eps = static_cast<uint32_t>(oob_comm.get_world_size());
      team_params.oob.oob_ep = static_cast<uint32_t>(oob_comm.get_rank());

      status = UCC_FUNC(ucc_team_create_post)(&context, 1, &team_params, &team);

      if(UCC_OK != status) {
        log_ucc.error() << "Failed to post team creation request\n";
        return status;
      }

      status = wait(context, [team]() { return UCC_FUNC(ucc_team_create_test)(team); });

      if(status == UCC_OK) {
        log_ucc.info() << "UCC Team created successfully.\n"
                       << "My rank is " << oob_comm.get_rank() << ", world size is "
                       << oob_comm.get_world_size() << "\n";
      } else {
        log_ucc.error() << "UCC Team creation failed.\n";
      }

      return status;
    }

    ucc_status_t UCCComm::init()
    {
      ucc_status_t status{UCC_OK};
      unsigned major = 0, minor = 0, release = 0;

      if(!resolve_fnptrs()) {
        status = UCC_ERR_NOT_FOUND;
        goto Done;
      }

      UCC_FUNC(ucc_get_version)(&major, &minor, &release);

      if(UCC_API_VERSION > UCC_VERSION(major, minor)) {
        log_ucc.error("UCC version mismatch: compiled with %d, found %d", UCC_API_VERSION,
                      UCC_VERSION(major, minor));
        goto Done;
      }

      status = init_lib(lib);
      if(UCC_OK != status) {
        goto Done;
      }

      status = create_context(lib, oob_comm, context);
      if(UCC_OK != status) {
        goto Done;
      }

      status = create_team(context, oob_comm, team);
      if(UCC_OK != status) {
        goto Done;
      }

    Done:
      if(status != UCC_OK) {
        UCC_Finalize();
      }

      return status;
    }

    ucc_status_t UCCComm::ucc_collective(ucc_coll_args_t &coll_args, ucc_coll_req_h &req)
    {
      ucc_status_t status;
      status = UCC_FUNC(ucc_collective_init)(&coll_args, &req, team);
      if(status != UCC_OK) {
        return status;
      }

      status = UCC_FUNC(ucc_collective_post)(req);
      if(status != UCC_OK) {
        goto Done;
      }

      status = wait(context, [&req]() { return req->status; });

    Done:
      UCC_FUNC(ucc_collective_finalize)(req);
      if(status != UCC_OK) {
        log_ucc.error("UCC collective (type=%d) failed: %d (%s)", coll_args.coll_type,
                      status, UCC_FUNC(ucc_status_string)(status));
      }
      return status;
    }

    ucc_status_t UCCComm::UCC_Bcast(void *buffer, int count, ucc_datatype_t datatype,
                                    int root)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_BCAST;
      coll_args.root = root;
      coll_args.src.info.buffer = buffer;
      coll_args.src.info.count = count;
      coll_args.src.info.datatype = datatype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Gather(void *sbuf, int sendcount, ucc_datatype_t sendtype,
                                     void *rbuf, int recvcount, ucc_datatype_t recvtype,
                                     int root)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_GATHER;
      coll_args.root = root;
      coll_args.src.info.buffer = sbuf;
      coll_args.src.info.count = sendcount;
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      if(rank == root) {
        coll_args.dst.info.buffer = rbuf;
        coll_args.dst.info.count = recvcount;
        coll_args.dst.info.datatype = recvtype;
        coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
      }

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allgather(void *sbuf, int sendcount,
                                        ucc_datatype_t sendtype, void *rbuf,
                                        int recvcount, ucc_datatype_t recvtype)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLGATHER;
      coll_args.src.info.buffer = sbuf;
      coll_args.src.info.count = sendcount;
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      coll_args.dst.info.buffer = rbuf;
      coll_args.dst.info.count = recvcount;
      coll_args.dst.info.datatype = recvtype;
      coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allreduce(void *sbuf, void *rbuf, int count,
                                        ucc_datatype_t datatype, ucc_reduction_op_t op)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
      coll_args.src.info.buffer = const_cast<void *>(sbuf);
      coll_args.src.info.count = count;
      coll_args.src.info.datatype = datatype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.dst.info.buffer = rbuf;
      coll_args.dst.info.count = count;
      coll_args.dst.info.datatype = datatype;
      coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.op = op;

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allgatherv(void *sbuf, int count, ucc_datatype_t sendtype,
                                         void *rbuf, const std::vector<int> &recvcounts,
                                         const std::vector<int> &displs,
                                         ucc_datatype_t recvtype)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
      coll_args.src.info.buffer = const_cast<void *>(sbuf);
      coll_args.src.info.count = recvcounts[rank];
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.dst.info_v.buffer = rbuf;
      coll_args.dst.info_v.counts = (ucc_count_t *)(recvcounts.data());
      coll_args.dst.info_v.displacements = (ucc_aint_t *)(displs.data());
      coll_args.dst.info_v.datatype = recvtype;
      coll_args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Barrier()
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_BARRIER;

      return UCCComm::ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Finalize()
    {
      ucc_status_t status = UCC_OK;
      if(team != nullptr) {
        ucc_team_h local_team = team;
        wait(context, [local_team]() { return UCC_FUNC(ucc_team_destroy)(local_team); });
        team = nullptr;
        if(status != UCC_OK) {
          log_ucc.error("ucc team destroy error: %s",
                        UCC_FUNC(ucc_status_string)(status));
        }
      }
      if(context != nullptr) {
        status = UCC_FUNC(ucc_context_destroy)(context);
        context = nullptr;
        if(status != UCC_OK) {
          log_ucc.error("Failed to tear down ucc context: %s",
                        UCC_FUNC(ucc_status_string)(status));
        }
      }
      if(lib != nullptr) {
        status = UCC_FUNC(ucc_finalize)(lib);
        lib = nullptr;
        if(status != UCC_OK) {
          log_ucc.error("Failed to finalize ucc: %s",
                        UCC_FUNC(ucc_status_string)(status));
        }
      }

      return status;
    }
  } // namespace ucc
} // namespace Realm
