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

class RemoteChannelFixture : public benchmark::Fixture 
{
  public:
    void SetUp(const ::benchmark::State& state) override {
      const char* channels_env = std::getenv("NUM_CHANNELS");

      if (channels_env) {
        num_channels = std::atoi(channels_env);
      }
      std::unique_ptr<Realm::SimpleRemoteChannelInfo> rci = std::make_unique<Realm::SimpleRemoteChannelInfo>(
                                                        1, Realm::XferDesKind::XFER_MEM_CPY, 0, std::vector<Realm::Channel::SupportedPath>());
      channels.reserve(num_channels);
      for (int i = 0; i < num_channels; i++) {
        Realm::RemoteChannel* rc = rci->create_remote_channel();
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
  for (auto _ : state) {
    for (auto& channel : channels) {
      channel->supports_redop(0);
    }
  }
}

BENCHMARK_MAIN();