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

// This test is only relevant when using the custom span implementation
// (i.e., when C++20 is not available). When C++20 is available, std::span is used.
#if __cplusplus < 202002L

#define REALM_NAMESPACE RealmHPP

#include "realm.hpp"
#include <gtest/gtest.h>

using namespace RealmHPP;

// Test the custom span implementation
TEST(HPPSpanTest, BasicSpanOperations)
{
  std::vector<int> data = {1, 2, 3, 4, 5};
  span<int> s(data);

  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), data.data());
  EXPECT_EQ(s[0], 1);
  EXPECT_EQ(s[4], 5);
}

TEST(HPPSpanTest, SpanFromPointerAndLength)
{
  int arr[] = {10, 20, 30};
  span<int> s(arr, 3);

  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), arr);
  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[2], 30);
}

TEST(HPPSpanTest, EmptySpan)
{
  span<int> s;
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s.data(), nullptr);
}

TEST(HPPSpanTest, IteratorTypes)
{
  span<int> s;
  // Test that iterator types are correctly defined
  EXPECT_TRUE((std::is_same<decltype(s.begin()), int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.end()), int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.cbegin()), const int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.cend()), const int *>::value));
}

TEST(HPPSpanTest, IteratorBeginEnd)
{
  int arr[] = {10, 20, 30, 40, 50};
  span<int> s(arr, 5);

  // Test begin() and end()
  auto it_begin = s.begin();
  auto it_end = s.end();

  EXPECT_EQ(it_begin, arr);
  EXPECT_EQ(it_end, arr + 5);
  EXPECT_EQ(std::distance(it_begin, it_end), 5);
}

TEST(HPPSpanTest, IteratorConstBeginEnd)
{
  int arr[] = {10, 20, 30, 40, 50};
  const span<int> s(arr, 5);

  // Test const begin() and end()
  auto it_begin = s.begin();
  auto it_end = s.end();

  EXPECT_EQ(it_begin, arr);
  EXPECT_EQ(it_end, arr + 5);
  EXPECT_EQ(std::distance(it_begin, it_end), 5);
}

TEST(HPPSpanTest, IteratorCbeginCend)
{
  int arr[] = {10, 20, 30, 40, 50};
  span<int> s(arr, 5);

  // Test cbegin() and cend()
  auto it_cbegin = s.cbegin();
  auto it_cend = s.cend();

  EXPECT_EQ(it_cbegin, arr);
  EXPECT_EQ(it_cend, arr + 5);
  EXPECT_EQ(std::distance(it_cbegin, it_cend), 5);
}

TEST(HPPSpanTest, IteratorRangeBasedFor)
{
  std::vector<int> data = {1, 2, 3, 4, 5};
  span<int> s(data);

  // Test range-based for loop
  int sum = 0;
  for(const auto &val : s) {
    sum += val;
  }
  EXPECT_EQ(sum, 15);
}

TEST(HPPSpanTest, IteratorConstRangeBasedFor)
{
  std::vector<int> data = {1, 2, 3, 4, 5};
  const span<int> s(data);

  // Test range-based for loop with const span
  int sum = 0;
  for(const auto &val : s) {
    sum += val;
  }
  EXPECT_EQ(sum, 15);
}

TEST(HPPSpanTest, IteratorModification)
{
  int arr[] = {1, 2, 3, 4, 5};
  span<int> s(arr, 5);

  // Test that iterators allow modification
  for(auto it = s.begin(); it != s.end(); ++it) {
    *it *= 2;
  }

  EXPECT_EQ(arr[0], 2);
  EXPECT_EQ(arr[1], 4);
  EXPECT_EQ(arr[2], 6);
  EXPECT_EQ(arr[3], 8);
  EXPECT_EQ(arr[4], 10);
}

TEST(HPPSpanTest, IteratorEmptySpan)
{
  span<int> s;

  // Test iterators on empty span
  EXPECT_EQ(s.begin(), s.end());
  EXPECT_EQ(s.cbegin(), s.cend());

  // Range-based for should not execute any iterations
  int count = 0;
  for(const auto &val : s) {
    count++;
  }
  EXPECT_EQ(count, 0);
}

TEST(HPPSpanTest, IteratorConstEmptySpan)
{
  const span<int> s;

  // Test const iterators on empty span
  EXPECT_EQ(s.begin(), s.end());
  EXPECT_EQ(s.cbegin(), s.cend());

  // Range-based for should not execute any iterations
  int count = 0;
  for(const auto &val : s) {
    count++;
  }
  EXPECT_EQ(count, 0);
}

TEST(HPPSpanTest, IteratorArithmetic)
{
  int arr[] = {10, 20, 30, 40, 50};
  span<int> s(arr, 5);

  auto it = s.begin();

  // Test iterator arithmetic
  EXPECT_EQ(*(it + 0), 10);
  EXPECT_EQ(*(it + 1), 20);
  EXPECT_EQ(*(it + 2), 30);
  EXPECT_EQ(*(it + 3), 40);
  EXPECT_EQ(*(it + 4), 50);

  // Test iterator increment
  EXPECT_EQ(*it, 10);
  ++it;
  EXPECT_EQ(*it, 20);
  it++;
  EXPECT_EQ(*it, 30);

  // Test iterator decrement
  --it;
  EXPECT_EQ(*it, 20);
  it--;
  EXPECT_EQ(*it, 10);
}

TEST(HPPSpanTest, IteratorComparison)
{
  int arr[] = {10, 20, 30, 40, 50};
  span<int> s(arr, 5);

  auto it1 = s.begin();
  auto it2 = s.begin() + 2;
  auto it3 = s.end();

  // Test iterator comparisons
  EXPECT_TRUE(it1 < it2);
  EXPECT_TRUE(it2 < it3);
  EXPECT_TRUE(it1 <= it2);
  EXPECT_TRUE(it2 <= it3);
  EXPECT_TRUE(it2 > it1);
  EXPECT_TRUE(it3 > it2);
  EXPECT_TRUE(it2 >= it1);
  EXPECT_TRUE(it3 >= it2);
  EXPECT_TRUE(it1 != it2);
  EXPECT_TRUE(it1 == it1);
}

#else
// When C++20 is available, this test is a no-op since std::span is used
#include <gtest/gtest.h>

TEST(HPPSpanTest, NoOpWhenCpp20Available)
{
  // This test intentionally does nothing when C++20 is available
  // because std::span is used instead of the custom implementation
  SUCCEED();
}

#endif
