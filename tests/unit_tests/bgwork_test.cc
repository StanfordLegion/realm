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

#include "realm/bgwork.h"
#include "realm/logging.h"
#include "realm/timers.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace Realm;

// The global bgwork logger defined in bgwork.cc.
namespace Realm {
  extern Logger log_bgwork;
}

// A LoggerOutputStream that captures warning messages for test verification.
// Allocated as a static singleton so it outlives all tests (the global logger
// holds a raw pointer to it).
class CapturingLogStream : public LoggerOutputStream {
public:
  void log_msg(Logger::LoggingLevel level, const char *name, const char *msgdata,
               size_t msglen) override
  {
    if(level == Logger::LEVEL_WARNING) {
      warning_count++;
      last_warning.assign(msgdata, msglen);
    }
  }

  void flush() override {}

  void reset()
  {
    warning_count = 0;
    last_warning.clear();
  }

  int warning_count = 0;
  std::string last_warning;
};

static CapturingLogStream &get_capture()
{
  static CapturingLogStream instance;
  return instance;
}

// Expose Logger::add_stream and configure_done for testing via a
// derived-class static_cast.
class LoggerAccessor : public Logger {
public:
  using Logger::add_stream;
  using Logger::configure_done;
};

// One-time setup: attach the capturing stream to the global bgwork logger.
static void ensure_logger_configured()
{
  static bool done = false;
  if(!done) {
    LoggerAccessor &acc = static_cast<LoggerAccessor &>(log_bgwork);
    acc.add_stream(&get_capture(), Logger::LEVEL_WARNING, false /*delete_when_done*/,
                   true /*flush_each_write*/);
    acc.configure_done();
    done = true;
  }
}

// A mock work item that busy-waits for a configurable duration to simulate
// an overrunning background work item.
class SlowWorkItem : public BackgroundWorkItem {
public:
  SlowWorkItem(const std::string &name, long long _sleep_ns)
    : BackgroundWorkItem(name)
    , sleep_ns(_sleep_ns)
  {}

  bool do_work(TimeLimit work_until) override
  {
    long long start = Clock::current_time_in_nanoseconds(true /*absolute*/);
    while((Clock::current_time_in_nanoseconds(true /*absolute*/) - start) < sleep_ns) {
    }
    return false;
  }

  // Expose protected make_active for test use.
  void activate() { make_active(); }

#ifdef DEBUG_REALM
  ~SlowWorkItem() override = default;

  void prepare_for_shutdown() { shutdown_work_item(); }
#endif

  long long sleep_ns;
};

// Helper: register item, run worker inline, clean up debug state.
static void run_single_work_item(BackgroundWorkManager &mgr, SlowWorkItem &item)
{
  item.add_to_manager(&mgr);
  item.activate();

  BackgroundWorkManager::Worker worker;
  worker.set_manager(&mgr);
  worker.do_work(-1 /*max_time*/, nullptr /*interrupt_flag*/);

#ifdef DEBUG_REALM
  item.prepare_for_shutdown();
#endif
}

class BgWorkOverrunTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    ensure_logger_configured();
    get_capture().reset();
  }
};

// Test that a work item completing within the overrun limit does NOT
// produce a warning.
TEST_F(BgWorkOverrunTest, NoWarningWithinBudget)
{
  BackgroundWorkManager mgr;
  // timeslice=1,000,000 ns (1ms), threshold=10x -> limit=10ms
  std::vector<std::string> args = {"-ll:bgslice", "1000000", "-ll:bgoverrun", "10"};
  mgr.configure_from_cmdline(args);

  // Work item that takes ~100us - well within the 10ms limit
  SlowWorkItem item("fast_item", 100000);
  run_single_work_item(mgr, item);

  EXPECT_EQ(get_capture().warning_count, 0);
}

// Test that a work item exceeding the overrun limit DOES produce a warning
// with the expected content.
TEST_F(BgWorkOverrunTest, WarningOnOverrun)
{
  BackgroundWorkManager mgr;
  // timeslice=100,000 ns (100us), threshold=2x -> limit=200,000 ns (200us)
  std::vector<std::string> args = {"-ll:bgslice", "100000", "-ll:bgoverrun", "2"};
  mgr.configure_from_cmdline(args);

  // Work item that takes ~5ms - well above the 200us limit
  SlowWorkItem item("slow_test_item", 5000000);
  run_single_work_item(mgr, item);

  EXPECT_GE(get_capture().warning_count, 1);
  // Verify the warning identifies the work item by name
  EXPECT_NE(get_capture().last_warning.find("slow_test_item"), std::string::npos);
  // Verify the warning mentions unresponsiveness
  EXPECT_NE(get_capture().last_warning.find("runtime may appear unresponsive"),
            std::string::npos);
}

// Test that overrun detection is disabled when threshold is set to 0.
TEST_F(BgWorkOverrunTest, DisabledWhenZero)
{
  BackgroundWorkManager mgr;
  // timeslice=100,000 ns (100us), threshold=0 -> disabled
  std::vector<std::string> args = {"-ll:bgslice", "100000", "-ll:bgoverrun", "0"};
  mgr.configure_from_cmdline(args);

  // Work item that takes ~5ms - would exceed any reasonable limit,
  // but detection is disabled
  SlowWorkItem item("disabled_test_item", 5000000);
  run_single_work_item(mgr, item);

  EXPECT_EQ(get_capture().warning_count, 0);
}

// Test that the default config (no command-line args) has overrun detection
// enabled by verifying that a sufficiently slow work item triggers a warning.
// Default: timeslice=100,000 ns (100us), threshold=100x -> limit=10ms.
TEST_F(BgWorkOverrunTest, DefaultConfigWarns)
{
  BackgroundWorkManager mgr;
  std::vector<std::string> args;
  mgr.configure_from_cmdline(args);

  // Work item that takes ~50ms - exceeds the default 10ms limit
  SlowWorkItem item("default_config_item", 50000000);
  run_single_work_item(mgr, item);

  EXPECT_GE(get_capture().warning_count, 1);
  EXPECT_NE(get_capture().last_warning.find("default_config_item"), std::string::npos);
}
