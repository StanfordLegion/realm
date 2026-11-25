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

#include "realm/transfer/channel.h"
#include <benchmark/benchmark.h>
#include <memory>
#include <thread>

class MockChannel : public Realm::Channel {
  public:
    MockChannel(Realm::XferDesKind kind) : Realm::Channel(kind)
    {
      // create some memories
      std::vector<Realm::Memory> src_memories;
      std::vector<Realm::Memory> dst_memories;
      for (int i = 0; i < num_memories_per_node; i++) {
        Realm::Memory mem = Realm::ID::make_memory(src_node, i).convert<Realm::Memory>();
        src_memories.push_back(mem);
      }
      for (int i = 0; i < num_memories_per_node; i++) {
        Realm::Memory mem = Realm::ID::make_memory(dst_node, i).convert<Realm::Memory>();
        dst_memories.push_back(mem);
      }

      // create some paths
      add_path(src_memories, dst_memories, 10000, 10000, 200, Realm::XferDesKind::XFER_MEM_CPY).set_max_dim(3);
    }

    virtual Realm::XferDesFactory *get_factory() override {
      return nullptr;
    }

    virtual long submit(Realm::Request **requests, long nr) override {
      return 0;
    }

    virtual void pull() override {
      return;
    }

    virtual long available() override {
      return 0;
    }

    virtual void enqueue_ready_xd(Realm::XferDes *xd) override {
      return;
    }

    virtual void wakeup_xd(Realm::XferDes *xd) override {
      return;
    }

    void add_a_path(std::vector<Realm::Memory> src_mems, std::vector<Realm::Memory> dst_mems, unsigned bandwidth, unsigned latency,
                    unsigned frag_overhead, Realm::XferDesKind xd_kind, int max_dim)
    {
      add_path(src_mems, dst_mems, bandwidth, latency, frag_overhead, xd_kind).set_max_dim(max_dim);
    }
};

class RemoteChannelFixture : public benchmark::Fixture 
{
  public:
    void SetUp(const ::benchmark::State& state) override {

      const char* channels_env = std::getenv("NUM_CHANNELS");
      if (channels_env) {
        num_channels = std::atoi(channels_env);
      }

      std::vector<Realm::Memory> src_memories;
      std::vector<Realm::Memory> dst_memories;
      src_memories.reserve(num_memories);
      dst_memories.reserve(num_memories);
      for (int i = 0; i < num_memories; i++) {
        Realm::Memory mem = Realm::ID::make_memory(0, i).convert<Realm::Memory>();
        src_memories.push_back(mem);
      }

      channels.reserve(num_channels);
      // heer are the configuration values used for the channel
      constexpr int bandwidth = 10000;
      constexpr int latency = 100;
      constexpr int frag_overhead = 200;
      constexpr int max_dim = 3;
      for (int i = 0; i < num_channels; i++) {
        int dst_node = i + 1;
        MockChannel channel(Realm::XferDesKind::XFER_MEM_CPY);
        for (int j = 0; j < num_memories; j++) {
          Realm::Memory mem = Realm::ID::make_memory(dst_node, j).convert<Realm::Memory>();
          dst_memories.push_back(mem);
        }
        channel.add_a_path(src_memories, dst_memories, bandwidth, latency, frag_overhead, Realm::XferDesKind::XFER_MEM_CPY, max_dim);
        dst_memories.clear();
        std::vector<Realm::Channel::SupportedPath> paths = channel.get_paths();

        std::unique_ptr<Realm::SimpleRemoteChannelInfo> rci = std::make_unique<Realm::SimpleRemoteChannelInfo>(
                                                                dst_node, Realm::XferDesKind::XFER_MEM_CPY, 0, paths);
        Realm::RemoteChannel* rc = rci->create_remote_channel();
        rc->register_redop(0);
        rc->has_non_redop_path = true;
        channels.push_back(std::unique_ptr<Realm::RemoteChannel>(rc));
      }
    }
  
    void TearDown(const ::benchmark::State& state) override {
      channels.clear();
    }
  
    std::vector<std::unique_ptr<Realm::RemoteChannel>> channels;
    int num_channels{100};
    int num_memories{10};
};

BENCHMARK_F(RemoteChannelFixture, BenchSupportsChannel)(benchmark::State& state) {
  Realm::Memory src_mem = Realm::ID::make_memory(0, 0).convert<Realm::Memory>();
  // can not use vector here because ChannelCopyInfo has no default constructor
  Realm::ChannelCopyInfo *channel_copy_infos = reinterpret_cast<Realm::ChannelCopyInfo *>(malloc(num_channels * sizeof(Realm::ChannelCopyInfo)));
  for (int i = 0; i < num_channels; i++) {
    Realm::Memory dst_mem = Realm::ID::make_memory(i+1, 0).convert<Realm::Memory>();
    channel_copy_infos[i] = Realm::ChannelCopyInfo{src_mem, dst_mem};
  }
  size_t bytes = 10000; // we benchmark the case with 10KB transfer. this number does not matter in this benchmark
  int redop_id = 0; // we benchmark the case with no reduction
  for (auto _ : state) {
    uint64_t best_time = 0;
    Realm::XferDesKind best_kind = Realm::XferDesKind::XFER_NONE;
    for (int i = 0; i < num_channels; i++) {
      best_time = channels[i]->supports_path(channel_copy_infos[i], 0, 0, redop_id, bytes, nullptr, nullptr, &best_kind, nullptr, nullptr);
      assert(best_time > 0);
    }
  }
  free(channel_copy_infos);
}

BENCHMARK_MAIN();