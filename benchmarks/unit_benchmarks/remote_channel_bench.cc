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
      int num_memories = 10;
      for (int i = 0; i < num_memories; i++) {
        Realm::Memory mem = Realm::ID::make_memory(0, i).convert<Realm::Memory>();
        src_memories.push_back(mem);
      }
      for (int i = 0; i < num_memories; i++) {
        Realm::Memory mem = Realm::ID::make_memory(1, i).convert<Realm::Memory>();
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
};

class RemoteChannelFixture : public benchmark::Fixture 
{
  public:
    void SetUp(const ::benchmark::State& state) override {

      const char* channels_env = std::getenv("NUM_CHANNELS");
      if (channels_env) {
        num_channels = std::atoi(channels_env);
      }

      channels.reserve(num_channels);
      for (int i = 0; i < num_channels; i++) {
        MockChannel channel(Realm::XferDesKind::XFER_MEM_CPY);
        std::vector<Realm::Channel::SupportedPath> paths = channel.get_paths();

        std::unique_ptr<Realm::SimpleRemoteChannelInfo> rci = std::make_unique<Realm::SimpleRemoteChannelInfo>(
                                                                1, Realm::XferDesKind::XFER_MEM_CPY, 0, paths);
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
};

BENCHMARK_F(RemoteChannelFixture, BenchSupportsRedop)(benchmark::State& state) {
  Realm::Memory src_mem = Realm::ID::make_memory(0, 0).convert<Realm::Memory>();
  Realm::Memory dst_mem = Realm::ID::make_memory(1, 0).convert<Realm::Memory>();
  Realm::ChannelCopyInfo channel_copy_info{src_mem, dst_mem};
  for (auto _ : state) {
    uint64_t best_time = 0;
    Realm::XferDesKind best_kind = Realm::XferDesKind::XFER_NONE;
    for (auto& channel : channels) {
      best_time = channel->supports_path(channel_copy_info, 0, 0, 0, 10000, nullptr, nullptr, &best_kind, nullptr, nullptr);
      assert(best_time > 0);
    }
  }
}

BENCHMARK_MAIN();