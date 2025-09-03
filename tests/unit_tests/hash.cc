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

#include <gtest/gtest.h>
#include <functional>
#include <random>
#include "realm.h"

using namespace Realm;
using testing::Types;

template <typename T>
class HashIDTest : public testing::Test {};

typedef Types<Memory, Event, Barrier, UserEvent, RegionInstance, Processor> IDTypes;

TYPED_TEST_SUITE(HashIDTest, IDTypes);

TYPED_TEST(HashIDTest, HashEqualsIDHash)
{
  TypeParam obj1, obj2;
  std::hash<TypeParam> obj_hash;
  std::hash<realm_id_t> id_hash;
  obj1.id = 0xDEADBEEFCAFEBABEULL;
  obj2.id = 0xCAFEBABEDEADBEEFULL;
  EXPECT_EQ(obj_hash(obj1), id_hash(obj1.id));
  EXPECT_EQ(obj_hash(obj2), id_hash(obj2.id));
  EXPECT_NE(obj_hash(obj1), obj_hash(obj2));
}

template <typename T>
class HashPODTest : public testing::Test {};

typedef Types<Point<2, int>, Matrix<2, 2, int>, Rect<2, int>, Point<2, int64_t>,
              Matrix<2, 2, int64_t>, Rect<2, int64_t>>
    PODTypes;

TYPED_TEST_SUITE(HashPODTest, PODTypes);

template <typename R, int N, typename T>
void generate_random(R &rand_gen, Realm::Point<N, T> &p)
{
  for(int i = 0; i < N; i++) {
    p[i] = rand_gen();
  }
}
template <typename R, int N, int M, typename T>
void generate_random(R &rand_gen, Realm::Matrix<N, M, T> &m)
{
  for(int i = 0; i < M; i++) {
    generate_random(rand_gen, m[i]);
  }
}
template <typename R, int N, typename T>
void generate_random(R &rand_gen, Realm::Rect<N, T> &r)
{
  generate_random(rand_gen, r.lo);
  generate_random(rand_gen, r.hi);
}

TYPED_TEST(HashPODTest, HashNotEqual)
{
  TypeParam obj1, obj2;
  std::hash<TypeParam> obj_hash;
  std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
  generate_random(rng, obj1);
  generate_random(rng, obj2);

  ASSERT_NE(obj1, obj2);
  EXPECT_NE(obj_hash(obj1), obj_hash(obj2));
}

TYPED_TEST(HashPODTest, HashEqual)
{
  TypeParam obj1, obj2;
  std::hash<TypeParam> obj_hash;
  std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
  generate_random(rng, obj1);
  obj2 = obj1;

  ASSERT_EQ(obj1, obj2);
  EXPECT_EQ(obj_hash(obj1), obj_hash(obj2));
}