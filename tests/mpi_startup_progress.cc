/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

// Startup must not leave an MPI barrier and then wait on another rank outside MPI.
//
// MPI promises only weak progress: a send that is complete at the sender may still
//  need the sender to call MPI again before the receiver gets it. This test gives
//  MPI_Barrier(MPI_COMM_WORLD) that behavior on every run. Rank 0 leaves each barrier
//  with its last message still queued, and hands it to MPI only in a later blocking or
//  test call (the calls in which MPI must make progress). Every other MPI call goes
//  straight to the library.
//
// If startup leaves a barrier and then waits on a peer by some other means without
//  polling MPI, the peer never leaves the barrier and the job hangs.

#include "realm.h"

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace Realm;

namespace {
  MPI_Comm barrier_comm = MPI_COMM_NULL;
  int world_rank = 0;
  int world_size = 1;
  std::atomic<unsigned> barrier_count(0);
  std::atomic<unsigned> barriers_left_queued(0);
  std::mutex queue_mutex;
  std::vector<std::pair<int, int>> queued_sends; // (destination, tag)
  char token;

  // what MPI does to "make progress" in this model: send what rank 0 queued
  void progress()
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for(const std::pair<int, int> &q : queued_sends) {
      MPI_Request req;
      PMPI_Isend(&token, 0, MPI_BYTE, q.first, q.second, barrier_comm, &req);
      PMPI_Request_free(&req);
    }
    queued_sends.clear();
  }

  // dissemination barrier; rank 0 queues its last-round send instead of sending it
  int weak_barrier()
  {
    const unsigned seq = barrier_count.fetch_add(1);
    int rounds = 0;
    while((1 << rounds) < world_size)
      rounds++;
    for(int k = 0; k < rounds; k++) {
      const int to = (world_rank + (1 << k)) % world_size;
      const int from = (world_rank - (1 << k) + world_size) % world_size;
      const int tag = int(seq % 1024) * 32 + k; // stays within the minimum MPI_TAG_UB
      MPI_Request recv;
      PMPI_Irecv(&token, 0, MPI_BYTE, from, tag, barrier_comm, &recv);
      if((world_rank == 0) && (k == rounds - 1)) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queued_sends.emplace_back(to, tag);
      } else {
        MPI_Request send;
        PMPI_Isend(&token, 0, MPI_BYTE, to, tag, barrier_comm, &send);
        PMPI_Wait(&send, MPI_STATUS_IGNORE);
      }
      PMPI_Wait(&recv, MPI_STATUS_IGNORE);
    }
    if(world_rank == 0)
      barriers_left_queued.fetch_add(1);
    return MPI_SUCCESS;
  }
} // namespace

extern "C" {

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
  int ret = PMPI_Init_thread(argc, argv, required, provided);
  if(ret != MPI_SUCCESS)
    return ret;
  PMPI_Comm_dup(MPI_COMM_WORLD, &barrier_comm);
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return ret;
}

int MPI_Barrier(MPI_Comm comm)
{
  progress();
  if((comm != MPI_COMM_WORLD) || (world_size == 1))
    return PMPI_Barrier(comm);
  return weak_barrier();
}

// the other blocking and test procedures the MPI module calls
int MPI_Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm)
{
  progress();
  return PMPI_Bcast(buf, count, type, root, comm);
}

int MPI_Gather(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount,
               MPI_Datatype rtype, int root, MPI_Comm comm)
{
  progress();
  return PMPI_Gather(sbuf, scount, stype, rbuf, rcount, rtype, root, comm);
}

int MPI_Allgather(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                  int rcount, MPI_Datatype rtype, MPI_Comm comm)
{
  progress();
  return PMPI_Allgather(sbuf, scount, stype, rbuf, rcount, rtype, comm);
}

int MPI_Allgatherv(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                   const int rcounts[], const int displs[], MPI_Datatype rtype,
                   MPI_Comm comm)
{
  progress();
  return PMPI_Allgatherv(sbuf, scount, stype, rbuf, rcounts, displs, rtype, comm);
}

int MPI_Allreduce(const void *sbuf, void *rbuf, int count, MPI_Datatype type, MPI_Op op,
                  MPI_Comm comm)
{
  progress();
  return PMPI_Allreduce(sbuf, rbuf, count, type, op, comm);
}

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info,
                        MPI_Comm *newcomm)
{
  progress();
  return PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{
  progress();
  return PMPI_Comm_dup(comm, newcomm);
}

int MPI_Comm_free(MPI_Comm *comm)
{
  progress();
  return PMPI_Comm_free(comm);
}

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                     void *baseptr, MPI_Win *win)
{
  progress();
  return PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
}

int MPI_Win_free(MPI_Win *win)
{
  progress();
  return PMPI_Win_free(win);
}

int MPI_Send(const void *buf, int count, MPI_Datatype type, int dest, int tag,
             MPI_Comm comm)
{
  progress();
  return PMPI_Send(buf, count, type, dest, tag, comm);
}

int MPI_Recv(void *buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm,
             MPI_Status *status)
{
  progress();
  return PMPI_Recv(buf, count, type, source, tag, comm, status);
}

int MPI_Wait(MPI_Request *req, MPI_Status *status)
{
  progress();
  return PMPI_Wait(req, status);
}

int MPI_Test(MPI_Request *req, int *flag, MPI_Status *status)
{
  progress();
  return PMPI_Test(req, flag, status);
}

int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)
{
  progress();
  return PMPI_Iprobe(source, tag, comm, flag, status);
}

int MPI_Finalize(void)
{
  progress();
  PMPI_Barrier(barrier_comm);
  PMPI_Comm_free(&barrier_comm);
  return PMPI_Finalize();
}

} // extern "C"

int main(int argc, char **argv)
{
  Runtime rt;
  if(!rt.network_init(&argc, &argv) || world_size < 2) {
    fprintf(stderr, "mpi_startup_progress: expected at least two MPI ranks\n");
    return 1;
  }
  if(!rt.create_configs(argc, argv))
    return 1;

  std::vector<std::string> cmdline(argv + 1, argv + argc);
  auto asymmetric_csize =
      std::find(cmdline.begin(), cmdline.end(), "-test:asymmetric-csize");
  if(asymmetric_csize != cmdline.end()) {
    // Rank 0 has no system memory of its own to send, but still needs its peers' maps.
    cmdline.erase(asymmetric_csize);
    cmdline.emplace_back("-ll:csize");
    cmdline.emplace_back(world_rank == 0 ? "0" : "16");
  }
  if(!rt.configure_from_command_line(cmdline))
    return 1;
  rt.start();

  // make sure the barrier above is the one startup used: otherwise this test proves
  //  nothing
  const bool interposed = (world_rank != 0) || (barriers_left_queued.load() > 0);
  if(!interposed)
    fprintf(stderr, "mpi_startup_progress: FAILED - MPI_Barrier was not interposed\n");

  rt.shutdown();
  int ret = rt.wait_for_shutdown();
  if(ret != 0)
    return ret;
  return interposed ? 0 : 1;
}
