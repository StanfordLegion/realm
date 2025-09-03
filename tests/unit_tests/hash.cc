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
#include "realm.h"

using namespace Realm;
using testing::Types;

template <typename T>
class HashTest : public testing::Test {
protected:
  T obj;
};

typedef Types<Memory, Event, Barrier, UserEvent, RegionInstance, Processor> IDTypes;

TYPED_TEST_SUITE(HashTest, IDTypes);

TYPED_TEST(HashTest, HashEqualsIDHash)
{
  this->obj.id = 0xDEADBEEFCAFEBABEULL;
  EXPECT_EQ(std::hash<decltype(this->obj)>()(this->obj),
            std::hash<realm_id_t>()(this->obj.id));
}