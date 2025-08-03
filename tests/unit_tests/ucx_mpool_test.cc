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

#include <cstddef>
#include "realm/ucx/mpool.h"
#include <gtest/gtest.h>

using namespace Realm;
using namespace Realm::UCP;

class UCXMPoolTest : public ::testing::Test {
protected:
  size_t obj_size = 16;
  size_t alignment = 16;
  size_t alignment_offset = 0;
};

TEST_F(UCXMPoolTest, GetMoreObjects)
{
  constexpr size_t obj_per_chunks = 1;
  constexpr size_t init_number_objects = 4;
  constexpr size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  std::vector<void *> objects;
  for(size_t i = 0; i < init_number_objects + 1; i++) {
    objects.push_back(mpool.get());
  }

  for(size_t i = 0; i < init_number_objects; i++) {
    EXPECT_NE(objects[i], nullptr);
  }

  EXPECT_EQ(objects.at(init_number_objects), nullptr);
  EXPECT_FALSE(mpool.has(false));
  EXPECT_FALSE(mpool.expand(1));
}

TEST_F(UCXMPoolTest, Expand)
{
  constexpr size_t obj_per_chunks = 4;
  constexpr size_t init_number_objects = 0;
  constexpr size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  EXPECT_TRUE(mpool.expand(max_objects));
}

TEST_F(UCXMPoolTest, GetAndPutBack)
{
  constexpr size_t obj_per_chunks = 4;
  constexpr size_t init_number_objects = 1;
  constexpr size_t max_objects = 1;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  void *object = mpool.get();
  bool has_before = mpool.has(false);
  MPool::put(object);
  bool has_after = mpool.has(false);

  EXPECT_NE(object, nullptr);
  EXPECT_FALSE(has_before);
  EXPECT_TRUE(has_after);
}

struct UCXMPoolTestCase {
  size_t obj_size;
  bool leak_check;
  size_t alignment;
  size_t alignment_offset;
  size_t obj_per_chunks;
  size_t init_number_objects;
  size_t get_objects_count;
  size_t max_objects;
  size_t max_chunk_size;
  double expand_factor;
  int obj_init_args;
  bool not_empty;
};

class UCXMPoolParamTest : public ::testing::TestWithParam<UCXMPoolTestCase> {
protected:
};

TEST_P(UCXMPoolParamTest, GetObjectsBase)
{
  auto test_case = GetParam();

  auto obj_init_func = [](void *obj, void *arg) {
    int size = *static_cast<int *>(arg);
    int *object = static_cast<int *>(obj);
    for(size_t i = 0; i < size / sizeof(*object); ++i) {
      object[i] = 7;
    }
  };

  MPool mpool("test", test_case.leak_check, test_case.obj_size, test_case.alignment,
              test_case.alignment_offset, test_case.obj_per_chunks,
              test_case.init_number_objects, test_case.max_objects,
              test_case.max_chunk_size, test_case.expand_factor, &MPool::malloc_wrapper,
              nullptr, &MPool::free_wrapper, nullptr, obj_init_func,
              static_cast<void *>(&test_case.obj_init_args));

  std::vector<void *> objects;
  for(size_t i = 0; i < test_case.get_objects_count; i++) {
    objects.push_back(mpool.get());
  }

  for(const auto *ptr : objects) {
    EXPECT_NE(ptr, nullptr);
    const int *object = static_cast<const int *>(ptr);
    for(size_t i = 0; i < test_case.obj_size / sizeof(*object); ++i) {
      EXPECT_EQ(object[i], 7);
    }
  }

  EXPECT_EQ(mpool.has(/*with_expand=*/false), test_case.not_empty);
}

INSTANTIATE_TEST_SUITE_P(
    MPool, UCXMPoolParamTest,
    // Case 1: init/get same number of objects
    testing::Values(UCXMPoolTestCase{
                        /*obj_size=*/16,
                        /*leak_check=*/true,
                        /*alignment=*/16,
                        /*alignment_offset=*/0,
                        /*obj_per_chunks=*/4,
                        /*init_number_objects=*/4,
                        /*get_objects_count=*/4,
                        /*max_objects=*/4,
                        /*max_chunk_size=*/16,
                        /*expand_factor=*/1.5,
                        /*obj_init_args=*/16,
                        /*not_empty=*/false,
                    },

                    // Case 2: init less objects
                    UCXMPoolTestCase{
                        /*obj_size=*/16,
                        /*leak_check=*/false,
                        /*alignment=*/16,
                        /*alignment_offset=*/0,
                        /*obj_per_chunks=*/4,
                        /*init_number_objects=*/4,
                        /*get_objects_count=*/5,
                        /*max_objects=*/5,
                        /*max_chunk_size=*/16,
                        /*expand_factor=*/1.5,
                        /*obj_init_args=*/16,
                        /*not_empty=*/false,
                    },

                    // Case 2: get less objects
                    UCXMPoolTestCase{
                        /*obj_size=*/16,
                        /*leak_check=*/false,
                        /*alignment=*/16,
                        /*alignment_offset=*/0,
                        /*obj_per_chunks=*/4,
                        /*init_number_objects=*/4,
                        /*get_objects_count=*/3,
                        /*max_objects=*/4,
                        /*max_chunk_size=*/16,
                        /*expand_factor=*/1.5,
                        /*obj_init_args=*/16,
                        /*not_empty=*/true,
                    },

                    // Case 3: zero init objects, dynamically expand
                    UCXMPoolTestCase{
                        /*obj_size=*/16,
                        /*leak_check=*/false,
                        /*alignment=*/16,
                        /*alignment_offset=*/0,
                        /*obj_per_chunks=*/1,
                        /*init_number_objects=*/0,
                        /*get_objects_count=*/3,
                        /*max_objects=*/4,
                        /*max_chunk_size=*/16,
                        /*expand_factor=*/1.5,
                        /*obj_init_args=*/16,
                        /*not_empty=*/false,
                    },

                    // Case 4: init/get same number of objects with leak checker
                    UCXMPoolTestCase{
                        /*obj_size=*/16,
                        /*leak_check=*/true,
                        /*alignment=*/16,
                        /*alignment_offset=*/0,
                        /*obj_per_chunks=*/4,
                        /*init_number_objects=*/4,
                        /*get_objects_count=*/4,
                        /*max_objects=*/4,
                        /*max_chunk_size=*/16,
                        /*expand_factor=*/1.5,
                        /*obj_init_args=*/16,
                        /*not_empty=*/false,
                    }));
