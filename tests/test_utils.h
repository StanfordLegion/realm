/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cassert>

#define ASSERT_REALM(expr)                                                               \
  do {                                                                                   \
    realm_status_t _status = (expr);                                                     \
    assert(_status == REALM_SUCCESS);                                                    \
  } while(0)

#endif
