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

#include "realm/point.h"
#include <gtest/gtest.h>

using namespace Realm;

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class PointTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override
  {
    for(int i = 0; i < N; i++) {
      values1[i] = i + 1;
      values2[i] = i * 2;
    }
  }

  T values1[N];
  T values2[N];
};

TYPED_TEST_SUITE_P(PointTest);

TYPED_TEST_P(PointTest, Zeroes)
{
  using T = typename TestFixture::T;
  TypeParam point = TypeParam::ZEROES();
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], static_cast<T>(0));
  }
}

TYPED_TEST_P(PointTest, Ones)
{
  using T = typename TestFixture::T;
  TypeParam point = TypeParam::ONES();
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], static_cast<T>(1));
  }
}

TYPED_TEST_P(PointTest, BaseAccess)
{
  TypeParam point(this->values1);
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Equality)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values1);
  TypeParam point3(this->values2);

  EXPECT_TRUE(point1 == point2);
  EXPECT_FALSE(point1 == point3);
}

TYPED_TEST_P(PointTest, Add)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  TypeParam result = point1 + point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values1[i] + this->values2[i]);
  }

  EXPECT_EQ(result.x(), this->values1[0] + this->values2[0]);

  if constexpr(TestFixture::N > 1) {
    EXPECT_EQ(result.y(), this->values1[1] + this->values2[1]);
  }

  if constexpr(TestFixture::N > 2) {
    EXPECT_EQ(result.z(), this->values1[2] + this->values2[2]);
  }

  if constexpr(TestFixture::N > 3) {
    EXPECT_EQ(result.w(), this->values1[3] + this->values2[3]);
  }
}

TYPED_TEST_P(PointTest, AdditionAssignment)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  point1 += point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point1[i], this->values1[i] + this->values2[i]);
  }
}

TYPED_TEST_P(PointTest, Subtract)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 - point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] - this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, SubstractAssignment)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  point2 -= point1;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point2[i], this->values2[i] - this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Multiply)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 * point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] * this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, MultiplyAssignment)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  point1 *= point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point1[i], this->values2[i] * this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Divide)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 / point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] / this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, DivideAssignment)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  point1 /= point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point1[i], this->values2[i] / this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Modulo)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 % point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] % this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, ModuloAssignment)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  point1 %= point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point1[i], this->values2[i] % this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Dot)
{
  using T = typename TestFixture::T;
  T product = 0;
  for(int i = 0; i < TestFixture::N; i++) {
    product += this->values1[i] * this->values2[i];
  }

  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  T dot = point1.dot(point2);

  EXPECT_EQ(dot, product);
}

TYPED_TEST_P(PointTest, Conversion)
{
  constexpr int N = TestFixture::N;

  using PointUnsigned = Point<N, unsigned>;
  PointUnsigned point_unsigned;
  for(int i = 0; i < N; i++) {
    point_unsigned[i] = 2u;
  }

  using PointInt = Point<N, int>;
  PointInt point_int = point_unsigned;

  for(int i = 0; i < N; i++) {
    EXPECT_EQ(point_int[i], static_cast<int>(point_unsigned[i]));
  }
}

REGISTER_TYPED_TEST_SUITE_P(PointTest, BaseAccess, Equality, Dot, Zeroes, Ones, Add,
                            AdditionAssignment, Subtract, SubstractAssignment, Multiply,
                            MultiplyAssignment, Divide, DivideAssignment, Modulo,
                            ModuloAssignment, Conversion);

#define TEST_POINT_TYPES(T) GeneratePointTypesForAllDims<T>()

template <typename T, int... Ns>
auto GeneratePointTypes(std::integer_sequence<int, Ns...>)
{
  return ::testing::Types<Realm::Point<Ns + 1, T>...>{};
}

template <typename T>
auto GeneratePointTypesForAllDims()
{
  return GeneratePointTypes<T>(std::make_integer_sequence<int, REALM_MAX_DIM>{});
}

#define INSTANTIATE_TEST_TYPES(BASE_TYPE, SUFFIX, SUITE)                                 \
  using N##SUFFIX = decltype(TEST_POINT_TYPES(BASE_TYPE));                               \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, SUITE, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int, PointTest);
INSTANTIATE_TEST_TYPES(long long, LongLong, PointTest);
