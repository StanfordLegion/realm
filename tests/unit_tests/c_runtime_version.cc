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

#include "realm/realm_c.h"
#include <gtest/gtest.h>

namespace Realm {
  extern const char *realm_library_version;
}

using namespace Realm;

TEST(RuntimeVersionTest, GetLibraryVersionSuccess)
{
  const char *version = nullptr;
  realm_status_t status = realm_get_library_version(&version);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_NE(version, nullptr);
  EXPECT_EQ(strcmp(version, Realm::realm_library_version), 0);
}

TEST(RuntimeVersionTest, GetLibraryVersionNullVersion)
{
  realm_status_t status = realm_get_library_version(nullptr);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}