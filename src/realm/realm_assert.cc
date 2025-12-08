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

#include "realm/realm_assert.h"
#include "realm/logging.h"

namespace Realm {

  Logger log_assert("assert");

  REALM_INTERNAL_API_EXTERNAL_LINKAGE void realm_assert_fail(const char* cond_text, const char* file, int line)
  {
    log_assert.fatal("Assertion failed: (%s) at %s:%d",
                     cond_text, file, line);
    abort();
  }

} // namespace Realm
