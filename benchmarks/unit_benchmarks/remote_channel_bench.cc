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