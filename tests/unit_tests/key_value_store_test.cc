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

#include "realm/runtime_impl.h"
#include "realm/runtime.h"
#include "test_mock.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include <cstring>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

// Mock key-value store context for testing
struct MockKVStoreContext {
  std::map<std::string, std::vector<uint8_t>> store;
  int barrier_call_count = 0;
  int put_call_count = 0;
  int get_call_count = 0;
  int cas_call_count = 0;
  bool barrier_should_fail = false;
  bool put_should_fail = false;
  bool get_should_fail = false;
  bool cas_should_fail = false;

  // For group testing
  uint64_t rank = 0;
  uint64_t ranks = 1;
  uint64_t group = 0;
};

// Mock vtable callbacks
static bool mock_put(const void *key, size_t key_size, const void *value,
                     size_t value_size, const void *vtable_data, size_t vtable_data_size)
{
  if(!key || !value || key_size == 0 || !vtable_data)
    return false;

  MockKVStoreContext *ctx = *(MockKVStoreContext **)vtable_data;
  ctx->put_call_count++;

  if(ctx->put_should_fail)
    return false;

  std::string k(static_cast<const char *>(key), key_size);
  std::vector<uint8_t> v(static_cast<const uint8_t *>(value),
                         static_cast<const uint8_t *>(value) + value_size);
  ctx->store[k] = v;
  return true;
}

static bool mock_get(const void *key, size_t key_size, void *value, size_t *value_size,
                     const void *vtable_data, size_t vtable_data_size)
{
  if(!key || !value || !value_size || key_size == 0 || !vtable_data)
    return false;

  MockKVStoreContext *ctx = *(MockKVStoreContext **)vtable_data;
  ctx->get_call_count++;

  if(ctx->get_should_fail)
    return false;

  std::string k(static_cast<const char *>(key), key_size);

  // Handle special realm keys
  if(k == "realm_rank") {
    if(*value_size < sizeof(uint32_t)) {
      *value_size = sizeof(uint32_t);
      return true;
    }
    uint32_t v = static_cast<uint32_t>(ctx->rank);
    std::memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }

  if(k == "realm_ranks") {
    if(*value_size < sizeof(uint32_t)) {
      *value_size = sizeof(uint32_t);
      return true;
    }
    uint32_t v = static_cast<uint32_t>(ctx->ranks);
    std::memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }

  if(k == "realm_group") {
    if(*value_size < sizeof(uint32_t)) {
      *value_size = sizeof(uint32_t);
      return true;
    }
    uint32_t v = static_cast<uint32_t>(ctx->group);
    std::memcpy(value, &v, sizeof(v));
    *value_size = sizeof(v);
    return true;
  }

  auto it = ctx->store.find(k);
  if(it != ctx->store.end()) {
    if(it->second.size() > *value_size) {
      *value_size = it->second.size();
      return true;
    }
    std::memcpy(value, it->second.data(), it->second.size());
    *value_size = it->second.size();
    return true;
  }

  // Key not found
  *value_size = 0;
  return true;
}

static bool mock_bar(const void *vtable_data, size_t vtable_data_size)
{
  if(!vtable_data)
    return false;

  MockKVStoreContext *ctx = *(MockKVStoreContext **)vtable_data;
  ctx->barrier_call_count++;

  return !ctx->barrier_should_fail;
}

static bool mock_cas(const void *key, size_t key_size, void *expected,
                     size_t *expected_size, const void *desired, size_t desired_size,
                     const void *vtable_data, size_t vtable_data_size)
{
  if(!key || !expected || !expected_size || !desired || key_size == 0 || !vtable_data)
    return false;

  MockKVStoreContext *ctx = *(MockKVStoreContext **)vtable_data;
  ctx->cas_call_count++;

  if(ctx->cas_should_fail)
    return false;

  std::string k(static_cast<const char *>(key), key_size);

  auto it = ctx->store.find(k);
  if(it == ctx->store.end()) {
    // Key doesn't exist, create it with desired value
    std::vector<uint8_t> v(static_cast<const uint8_t *>(desired),
                           static_cast<const uint8_t *>(desired) + desired_size);
    ctx->store[k] = v;
    return true;
  }

  // Key exists, check if current value matches expected
  if(it->second.size() == *expected_size &&
     std::memcmp(it->second.data(), expected, *expected_size) == 0) {
    // Match! Update to desired
    std::vector<uint8_t> v(static_cast<const uint8_t *>(desired),
                           static_cast<const uint8_t *>(desired) + desired_size);
    ctx->store[k] = v;
    return true;
  }

  // No match, return current value in expected
  if(it->second.size() > *expected_size) {
    *expected_size = it->second.size();
    return false;
  }
  std::memcpy(expected, it->second.data(), it->second.size());
  *expected_size = it->second.size();
  return false;
}

// Helper class to expose protected members for testing
class TestableRuntimeImpl : public MockRuntimeImpl {
public:
  using MockRuntimeImpl::key_value_store_vtable;
};

class KeyValueStoreTestBase : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;

    ctx = new MockKVStoreContext();
    ctx_ptr = new MockKVStoreContext *(ctx);

    // Create a minimal runtime for testing
    runtime_impl = new TestableRuntimeImpl();
    runtime_impl->init(1);
  }

  void TearDown() override
  {
    runtime_impl->finalize();
    delete runtime_impl;
    delete ctx;
    delete ctx_ptr;

    Realm::enable_unit_tests = false;
  }

  void InitVtable(bool with_put = true, bool with_get = true, bool with_bar = false,
                  bool with_cas = false)
  {
    vtable.vtable_data = ctx_ptr;
    vtable.vtable_data_size = sizeof(MockKVStoreContext *);
    vtable.put = with_put ? mock_put : nullptr;
    vtable.get = with_get ? mock_get : nullptr;
    vtable.bar = with_bar ? mock_bar : nullptr;
    vtable.cas = with_cas ? mock_cas : nullptr;

    // Directly set the vtable (now accessible via TestableRuntimeImpl)
    runtime_impl->key_value_store_vtable = vtable;
  }

  MockKVStoreContext *ctx;
  MockKVStoreContext **ctx_ptr;
  TestableRuntimeImpl *runtime_impl;
  Runtime::KeyValueStoreVtable vtable;
};

// Test has_key_value_store
TEST_F(KeyValueStoreTestBase, HasKeyValueStore_WithPut)
{
  InitVtable(true, true, false, false);
  EXPECT_TRUE(runtime_impl->has_key_value_store());
}

TEST_F(KeyValueStoreTestBase, HasKeyValueStore_WithoutPut)
{
  InitVtable(false, true, false, false);
  EXPECT_FALSE(runtime_impl->has_key_value_store());
}

// Test key_value_store_elastic
TEST_F(KeyValueStoreTestBase, Elastic_WithCas)
{
  InitVtable(true, true, false, true);
  EXPECT_TRUE(runtime_impl->key_value_store_elastic());
}

TEST_F(KeyValueStoreTestBase, Elastic_WithoutCas)
{
  InitVtable(true, true, false, false);
  EXPECT_FALSE(runtime_impl->key_value_store_elastic());
}

// Test key_value_store_group
TEST_F(KeyValueStoreTestBase, Group_WithBar)
{
  InitVtable(true, true, true, false);
  EXPECT_TRUE(runtime_impl->key_value_store_group());
}

TEST_F(KeyValueStoreTestBase, Group_WithoutBar)
{
  InitVtable(true, true, false, false);
  EXPECT_FALSE(runtime_impl->key_value_store_group());
}

// Test key_value_store_local_group
TEST_F(KeyValueStoreTestBase, LocalGroup_Success)
{
  ctx->group = 42;
  InitVtable(true, true, true, false);

  auto result = runtime_impl->key_value_store_local_group();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

TEST_F(KeyValueStoreTestBase, LocalGroup_DifferentSizes)
{
  InitVtable(true, true, true, false);

  // Test with uint8_t - use context value
  ctx->group = 255;
  auto result8 = runtime_impl->key_value_store_local_group();
  ASSERT_TRUE(result8.has_value());
  EXPECT_EQ(result8.value(), 255);

  // Test with uint16_t - update context to test larger value
  ctx->group = 1000;
  auto result16 = runtime_impl->key_value_store_local_group();
  ASSERT_TRUE(result16.has_value());
  EXPECT_EQ(result16.value(), 1000);

  // Test with uint32_t - update context for even larger value
  ctx->group = 100000;
  auto result32 = runtime_impl->key_value_store_local_group();
  ASSERT_TRUE(result32.has_value());
  EXPECT_EQ(result32.value(), 100000);

  // Test with uint64_t - update context for maximum value
  ctx->group = 1000000;
  auto result64 = runtime_impl->key_value_store_local_group();
  ASSERT_TRUE(result64.has_value());
  EXPECT_EQ(result64.value(), 1000000);
}

// Test key_value_store_local_rank
TEST_F(KeyValueStoreTestBase, LocalRank_Success)
{
  ctx->rank = 3;
  ctx->ranks = 8;
  InitVtable(true, true, true, false);

  auto result = runtime_impl->key_value_store_local_rank();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 3);
}

// Test key_value_store_local_ranks
TEST_F(KeyValueStoreTestBase, LocalRanks_Success)
{
  ctx->rank = 3;
  ctx->ranks = 8;
  InitVtable(true, true, true, false);

  auto result = runtime_impl->key_value_store_local_ranks();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 8);
}

// Test key_value_store_put
TEST_F(KeyValueStoreTestBase, Put_Success)
{
  InitVtable(true, true, false, false);

  const char *key = "test_key";
  const char *value = "test_value";

  bool result = runtime_impl->key_value_store_put(key, strlen(key), value, strlen(value));

  EXPECT_TRUE(result);
  EXPECT_EQ(ctx->put_call_count, 1);
  EXPECT_EQ(ctx->store.size(), 1);

  std::string stored_value(ctx->store["test_key"].begin(), ctx->store["test_key"].end());
  EXPECT_EQ(stored_value, "test_value");
}

TEST_F(KeyValueStoreTestBase, Put_BinaryData)
{
  InitVtable(true, true, false, false);

  const char *key = "binary_key";
  uint8_t value[] = {0x01, 0x02, 0x03, 0x04, 0xFF};

  bool result = runtime_impl->key_value_store_put(key, strlen(key), value, sizeof(value));

  EXPECT_TRUE(result);
  EXPECT_EQ(ctx->store["binary_key"].size(), 5);
  EXPECT_EQ(ctx->store["binary_key"][0], 0x01);
  EXPECT_EQ(ctx->store["binary_key"][4], 0xFF);
}

TEST_F(KeyValueStoreTestBase, Put_Overwrite)
{
  InitVtable(true, true, false, false);

  const char *key = "key";

  runtime_impl->key_value_store_put(key, strlen(key), "value1", 6);
  runtime_impl->key_value_store_put(key, strlen(key), "value2", 6);

  EXPECT_EQ(ctx->put_call_count, 2);
  std::string stored_value(ctx->store["key"].begin(), ctx->store["key"].end());
  EXPECT_EQ(stored_value, "value2");
}

// Test key_value_store_get
TEST_F(KeyValueStoreTestBase, Get_Success)
{
  InitVtable(true, true, false, false);

  const char *key = "test_key";
  const char *value = "test_value";

  // First put a value
  runtime_impl->key_value_store_put(key, strlen(key), value, strlen(value));

  // Now get it
  char buffer[256];
  size_t buffer_size = sizeof(buffer);
  bool result = runtime_impl->key_value_store_get(key, strlen(key), buffer, &buffer_size);

  EXPECT_TRUE(result);
  EXPECT_EQ(ctx->get_call_count, 1);
  EXPECT_EQ(buffer_size, strlen(value));
  EXPECT_EQ(std::string(buffer, buffer_size), "test_value");
}

TEST_F(KeyValueStoreTestBase, Get_KeyNotFound)
{
  InitVtable(true, true, false, false);

  const char *key = "nonexistent_key";
  char buffer[256];
  size_t buffer_size = sizeof(buffer);

  bool result = runtime_impl->key_value_store_get(key, strlen(key), buffer, &buffer_size);

  EXPECT_TRUE(result);
  EXPECT_EQ(buffer_size, 0);
}

TEST_F(KeyValueStoreTestBase, Get_BufferTooSmall)
{
  InitVtable(true, true, false, false);

  const char *key = "test_key";
  const char *value = "this_is_a_long_value";

  runtime_impl->key_value_store_put(key, strlen(key), value, strlen(value));

  char buffer[5];
  size_t buffer_size = sizeof(buffer);
  bool result = runtime_impl->key_value_store_get(key, strlen(key), buffer, &buffer_size);

  EXPECT_TRUE(result);
  EXPECT_EQ(buffer_size, strlen(value)); // Should return actual size
}

TEST_F(KeyValueStoreTestBase, Get_BinaryData)
{
  InitVtable(true, true, false, false);

  const char *key = "binary_key";
  uint8_t value[] = {0xDE, 0xAD, 0xBE, 0xEF};

  runtime_impl->key_value_store_put(key, strlen(key), value, sizeof(value));

  uint8_t buffer[10];
  size_t buffer_size = sizeof(buffer);
  bool result = runtime_impl->key_value_store_get(key, strlen(key), buffer, &buffer_size);

  EXPECT_TRUE(result);
  EXPECT_EQ(buffer_size, 4);
  EXPECT_EQ(buffer[0], 0xDE);
  EXPECT_EQ(buffer[1], 0xAD);
  EXPECT_EQ(buffer[2], 0xBE);
  EXPECT_EQ(buffer[3], 0xEF);
}

// Test key_value_store_bar
TEST_F(KeyValueStoreTestBase, Bar_Success)
{
  InitVtable(true, true, true, false);

  bool result = runtime_impl->key_value_store_bar();

  EXPECT_TRUE(result);
  EXPECT_EQ(ctx->barrier_call_count, 1);
}

TEST_F(KeyValueStoreTestBase, Bar_Failure)
{
  InitVtable(true, true, true, false);
  ctx->barrier_should_fail = true;

  bool result = runtime_impl->key_value_store_bar();

  EXPECT_FALSE(result);
  EXPECT_EQ(ctx->barrier_call_count, 1);
}

TEST_F(KeyValueStoreTestBase, Bar_MultipleCalls)
{
  InitVtable(true, true, true, false);

  runtime_impl->key_value_store_bar();
  runtime_impl->key_value_store_bar();
  runtime_impl->key_value_store_bar();

  EXPECT_EQ(ctx->barrier_call_count, 3);
}

// Test key_value_store_cas
TEST_F(KeyValueStoreTestBase, CAS_CreateNew)
{
  InitVtable(true, true, false, true);

  const char *key = "new_key";
  uint32_t expected = 0;
  size_t expected_size = sizeof(expected);
  uint32_t desired = 42;

  bool result = runtime_impl->key_value_store_cas(
      key, strlen(key), &expected, &expected_size, &desired, sizeof(desired));

  EXPECT_TRUE(result);
  EXPECT_EQ(ctx->cas_call_count, 1);

  // Verify the value was stored
  char buffer[256];
  size_t buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get(key, strlen(key), buffer, &buffer_size);
  EXPECT_EQ(buffer_size, sizeof(uint32_t));
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(buffer), 42);
}

TEST_F(KeyValueStoreTestBase, CAS_SuccessfulUpdate)
{
  InitVtable(true, true, false, true);

  const char *key = "counter";
  uint32_t initial = 10;

  // Put initial value
  runtime_impl->key_value_store_put(key, strlen(key), &initial, sizeof(initial));

  // CAS update from 10 to 11
  uint32_t expected = 10;
  size_t expected_size = sizeof(expected);
  uint32_t desired = 11;

  bool result = runtime_impl->key_value_store_cas(
      key, strlen(key), &expected, &expected_size, &desired, sizeof(desired));

  EXPECT_TRUE(result);

  // Verify the value was updated
  uint32_t buffer;
  size_t buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get(key, strlen(key), &buffer, &buffer_size);
  EXPECT_EQ(buffer, 11);
}

TEST_F(KeyValueStoreTestBase, CAS_FailedUpdate)
{
  InitVtable(true, true, false, true);

  const char *key = "counter";
  uint32_t initial = 10;

  // Put initial value
  runtime_impl->key_value_store_put(key, strlen(key), &initial, sizeof(initial));

  // Try to CAS with wrong expected value
  uint32_t expected = 5; // Wrong!
  size_t expected_size = sizeof(expected);
  uint32_t desired = 11;

  bool result = runtime_impl->key_value_store_cas(
      key, strlen(key), &expected, &expected_size, &desired, sizeof(desired));

  EXPECT_FALSE(result);
  EXPECT_EQ(expected, 10); // Should return actual value

  // Verify the value was NOT updated
  uint32_t buffer;
  size_t buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get(key, strlen(key), &buffer, &buffer_size);
  EXPECT_EQ(buffer, 10);
}

TEST_F(KeyValueStoreTestBase, CAS_AtomicCounter)
{
  InitVtable(true, true, false, true);

  const char *key = "atomic_counter";
  uint32_t value = 0;

  // Simulate atomic increment operations
  for(int i = 0; i < 5; i++) {
    uint32_t expected = value;
    size_t expected_size = sizeof(expected);
    uint32_t desired = value + 1;

    bool result = runtime_impl->key_value_store_cas(
        key, strlen(key), &expected, &expected_size, &desired, sizeof(desired));
    EXPECT_TRUE(result);
    value++;
  }

  // Final value should be 5
  uint32_t buffer;
  size_t buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get(key, strlen(key), &buffer, &buffer_size);
  EXPECT_EQ(buffer, 5);
  EXPECT_EQ(ctx->cas_call_count, 5);
}

// Integration tests combining multiple operations
TEST_F(KeyValueStoreTestBase, Integration_PutGetWorkflow)
{
  InitVtable(true, true, false, false);

  // Put multiple key-value pairs
  runtime_impl->key_value_store_put("key1", 4, "value1", 6);
  runtime_impl->key_value_store_put("key2", 4, "value2", 6);
  runtime_impl->key_value_store_put("key3", 4, "value3", 6);

  // Get them back
  char buffer[256];
  size_t buffer_size;

  buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get("key1", 4, buffer, &buffer_size);
  EXPECT_EQ(std::string(buffer, buffer_size), "value1");

  buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get("key2", 4, buffer, &buffer_size);
  EXPECT_EQ(std::string(buffer, buffer_size), "value2");

  buffer_size = sizeof(buffer);
  runtime_impl->key_value_store_get("key3", 4, buffer, &buffer_size);
  EXPECT_EQ(std::string(buffer, buffer_size), "value3");
}

TEST_F(KeyValueStoreTestBase, Integration_GroupWithBarrier)
{
  ctx->rank = 2;
  ctx->ranks = 4;
  ctx->group = 1;
  InitVtable(true, true, true, false);

  EXPECT_TRUE(runtime_impl->key_value_store_group());

  auto rank = runtime_impl->key_value_store_local_rank();
  auto ranks = runtime_impl->key_value_store_local_ranks();
  auto group = runtime_impl->key_value_store_local_group();

  ASSERT_TRUE(rank.has_value());
  ASSERT_TRUE(ranks.has_value());
  ASSERT_TRUE(group.has_value());

  EXPECT_EQ(rank.value(), 2);
  EXPECT_EQ(ranks.value(), 4);
  EXPECT_EQ(group.value(), 1);

  EXPECT_TRUE(runtime_impl->key_value_store_bar());
}

TEST_F(KeyValueStoreTestBase, Integration_ElasticWithCAS)
{
  InitVtable(true, true, false, true);

  EXPECT_TRUE(runtime_impl->key_value_store_elastic());

  // Simulate elastic node joining
  const char *counter_key = "node_counter";
  uint32_t expected = 0;
  size_t expected_size = sizeof(expected);
  uint32_t desired = 1;

  bool result =
      runtime_impl->key_value_store_cas(counter_key, strlen(counter_key), &expected,
                                        &expected_size, &desired, sizeof(desired));
  EXPECT_TRUE(result);
}
