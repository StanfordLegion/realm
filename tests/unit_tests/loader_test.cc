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

#include "realm/realm_config.h"
#include "realm/loader.h"
#include "dummy_library.h"
#include <gtest/gtest.h>

using namespace Realm;

#if defined(REALM_ON_WINDOWS)
#define DUMMY_LIB_NAME "./dummy.dll"
#else
#define DUMMY_LIB_NAME "./libdummy.so"
#endif

struct DummyLoader : public Loader<DummyLoader> {
  decltype(&dummy) dummy_fnptr = nullptr;
  bool load_symbols() { return get_symbol("dummy", dummy_fnptr); }
};

struct DummyBadLoader : public Loader<DummyBadLoader> {
  decltype(&dummy) dummy_fnptr = nullptr;
  bool load_symbols() { return get_symbol("dummy.does.not.exist", dummy_fnptr); }
};

TEST(Loader, CanLoadDummy)
{
  lib_handle_t dummy_lib = load_library(DUMMY_LIB_NAME);

  EXPECT_NE(dummy_lib, nullptr);

  close_library(dummy_lib);
}

TEST(Loader, LibraryNotFound)
{
  lib_handle_t dummy_lib = load_library(DUMMY_LIB_NAME ".does_not_exist");
  EXPECT_EQ(dummy_lib, nullptr);
}

TEST(Loader, CanRetrieveValidSymbol)
{
  const int TEST_VALUE = 0xDEADBEEF;
  int ret_value = 0;
  lib_handle_t dummy_lib = load_library(DUMMY_LIB_NAME);

  decltype(&dummy) dummy_fnptr =
      reinterpret_cast<decltype(&dummy)>(get_symbol(dummy_lib, "dummy"));
  ASSERT_NE(dummy_fnptr, nullptr);

  ret_value = dummy_fnptr(TEST_VALUE);
  close_library(dummy_lib);

  EXPECT_EQ(TEST_VALUE, ret_value);
}

TEST(Loader, SymbolNotFound)
{
  lib_handle_t dummy_lib = load_library(DUMMY_LIB_NAME);
  void *fnptr = get_symbol(dummy_lib, "bar_does_not_exist");
  close_library(dummy_lib);

  EXPECT_EQ(fnptr, nullptr);
}

TEST(Loader, DefaultConstructedInvalid)
{
  DummyLoader loader;
  EXPECT_FALSE(!!loader);
}

TEST(Loader, LoaderClassLoads)
{
  const int TEST_VALUE = 0xCAFEBABE;
  DummyLoader loader;
  EXPECT_TRUE(loader.load(DUMMY_LIB_NAME));
  EXPECT_TRUE(!!loader);
  EXPECT_NE(loader.dummy_fnptr, nullptr);
  EXPECT_EQ(TEST_VALUE, loader.dummy_fnptr(TEST_VALUE));
}

TEST(Loader, LoaderClassLoadMultipleNames)
{
  const int TEST_VALUE = 0xCAFEBABE;
  DummyLoader loader;
  EXPECT_TRUE(loader.load({DUMMY_LIB_NAME ".does_not_exist", DUMMY_LIB_NAME}));
  EXPECT_TRUE(!!loader);
  EXPECT_NE(loader.dummy_fnptr, nullptr);
  EXPECT_EQ(TEST_VALUE, loader.dummy_fnptr(TEST_VALUE));
}

TEST(Loader, LoaderClassSymbolFails)
{
  const int TEST_VALUE = 0xCAFEBABE;
  DummyBadLoader loader;
  EXPECT_TRUE(loader.load(DUMMY_LIB_NAME));
  EXPECT_FALSE(!!loader);
}