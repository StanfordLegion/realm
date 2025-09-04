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

#define REALM_NAMESPACE RealmHPP

#include "realm.hpp"
#include <gtest/gtest.h>

// Test the custom span implementation
TEST(HPPSpanTest, SpanFromVector)
{
  std::vector<int> data = {1, 2, 3, 4, 5};
  REALM_NAMESPACE::span<int> s(data);

  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), data.data());
  EXPECT_EQ(s[0], 1);
  EXPECT_EQ(s[4], 5);
}

TEST(HPPSpanTest, SpanFromConstVector)
{
  const std::vector<int> data = {1, 2, 3, 4, 5};
  REALM_NAMESPACE::span<const int> s(data);

  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), data.data());
  EXPECT_EQ(s[0], 1);
  EXPECT_EQ(s[4], 5);
  EXPECT_TRUE(std::is_const<std::remove_reference_t<decltype(s[0])>>::value);
}

TEST(HPPSpanTest, SpanFromPointerAndLength)
{
  int arr[] = {10, 20, 30};
  REALM_NAMESPACE::span<int> s(arr, 3);

  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), arr);
  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[2], 30);
}

TEST(HPPSpanTest, SpanFromPointerAndLengthConst)
{
  const int arr[] = {10, 20, 30};
  REALM_NAMESPACE::span<const int> s(arr, 3);

  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s.data(), arr);
  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[2], 30);
  EXPECT_TRUE(std::is_const<std::remove_reference_t<decltype(s[0])>>::value);
}

TEST(HPPSpanTest, EmptySpan)
{
  REALM_NAMESPACE::span<int> s;
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s.data(), nullptr);
  EXPECT_TRUE(s.empty());
}

TEST(HPPSpanTest, IteratorTypes)
{
  REALM_NAMESPACE::span<int> s;
  // Test that iterator types are correctly defined
  EXPECT_TRUE((std::is_same<decltype(s.begin()), int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.end()), int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.cbegin()), const int *>::value));
  EXPECT_TRUE((std::is_same<decltype(s.cend()), const int *>::value));
}

TEST(HPPSpanTest, IteratorBeginEnd)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);

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
  const REALM_NAMESPACE::span<int> s(arr, 5);

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
  REALM_NAMESPACE::span<int> s(arr, 5);

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
  REALM_NAMESPACE::span<int> s(data);

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
  const REALM_NAMESPACE::span<int> s(data);

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
  REALM_NAMESPACE::span<int> s(arr, 5);

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
  REALM_NAMESPACE::span<int> s;

  int count = 0;
  for(const auto &val : s) {
    count++;
  }

  EXPECT_EQ(count, 0);
  EXPECT_EQ(s.begin(), s.end());
  EXPECT_EQ(s.cbegin(), s.cend());
}

TEST(HPPSpanTest, IteratorConstEmptySpan)
{
  const REALM_NAMESPACE::span<int> s;

  int count = 0;
  for(const auto &val : s) {
    count++;
  }

  EXPECT_EQ(s.begin(), s.end());
  EXPECT_EQ(s.cbegin(), s.cend());
  EXPECT_EQ(count, 0);
}

TEST(HPPSpanTest, IteratorArithmetic)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);
  auto it = s.begin();

  EXPECT_EQ(*(it + 0), 10);
  EXPECT_EQ(*(it + 1), 20);
  EXPECT_EQ(*(it + 2), 30);
  EXPECT_EQ(*(it + 3), 40);
  EXPECT_EQ(*(it + 4), 50);
}

TEST(HPPSpanTest, IteratorPrefixIncrement)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);
  auto it = s.begin();

  EXPECT_EQ(*it, 10);
  ++it;
  EXPECT_EQ(*it, 20);
  ++it;
  EXPECT_EQ(*it, 30);
}

TEST(HPPSpanTest, IteratorPostfixIncrement)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);
  auto it = s.begin();

  EXPECT_EQ(*it, 10);
  auto old_it = it++;
  EXPECT_EQ(*old_it, 10);
  EXPECT_EQ(*it, 20);
  old_it = it++;
  EXPECT_EQ(*old_it, 20);
  EXPECT_EQ(*it, 30);
}

TEST(HPPSpanTest, IteratorPrefixDecrement)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);
  auto it = s.begin() + 2;

  EXPECT_EQ(*it, 30);
  --it;
  EXPECT_EQ(*it, 20);
  --it;
  EXPECT_EQ(*it, 10);
}

TEST(HPPSpanTest, IteratorPostfixDecrement)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);
  auto it = s.begin() + 2;

  EXPECT_EQ(*it, 30);
  auto old_it = it--;
  EXPECT_EQ(*old_it, 30);
  EXPECT_EQ(*it, 20);
  old_it = it--;
  EXPECT_EQ(*old_it, 20);
  EXPECT_EQ(*it, 10);
}

TEST(HPPSpanTest, IteratorComparison)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr, 5);

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

TEST(HPPSpanTest, CopyConstructorFromDifferentExtent)
{
  int arr[] = {1, 2, 3, 4, 5};
  REALM_NAMESPACE::span<int> source_span(arr, 5);

  REALM_NAMESPACE::span<int> copied_span(source_span);

  EXPECT_EQ(copied_span.size(), 5);
  EXPECT_EQ(copied_span.data(), arr);
  EXPECT_EQ(copied_span[0], 1);
  EXPECT_EQ(copied_span[4], 5);

  copied_span[2] = 300;

  EXPECT_EQ(arr[2], 300);
  EXPECT_EQ(source_span[2], 300);
}

TEST(HPPSpanTest, SingleScalarReferenceConstructor)
{
  int value = 42;
  REALM_NAMESPACE::span<int> s(value);

  EXPECT_EQ(s.size(), 1);
  EXPECT_EQ(s.data(), &value);
  EXPECT_EQ(s[0], 42);

  s[0] = 100;

  EXPECT_EQ(value, 100);
}

TEST(HPPSpanTest, CStyleArrayConstructor)
{
  int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<int> s(arr);

  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), arr);
  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[4], 50);

  s[2] = 300;

  EXPECT_EQ(arr[2], 300);
}

TEST(HPPSpanTest, CStyleArrayConstructorConst)
{
  const int arr[] = {10, 20, 30, 40, 50};
  REALM_NAMESPACE::span<const int> s(arr);

  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.data(), arr);
  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[4], 50);

  EXPECT_TRUE(std::is_const<std::remove_reference_t<decltype(s[0])>>::value);
}
