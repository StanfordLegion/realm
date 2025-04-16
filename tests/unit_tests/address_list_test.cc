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

#include "realm/transfer/address_list.h"
#include <gtest/gtest.h>

using namespace Realm;

namespace {

  constexpr size_t kStride = 8;
  constexpr size_t kBytes = 1024;

  TEST(AddressListTests, Basic1DEntryNoPayload)
  {
    AddressList addrlist;
    const int dim = 1;

    size_t *entry = addrlist.being_entry(dim);
    ASSERT_NE(entry, nullptr);
    entry[0] = AddressList::pack_entry_header(kBytes, dim);
    addrlist.commit_entry(dim, kBytes);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(cursor.get_dim(), 1);
    EXPECT_EQ(cursor.remaining(0), kBytes);
    EXPECT_EQ(cursor.get_offset(), 0);

    cursor.advance(0, kStride);
    EXPECT_EQ(cursor.remaining(0), kBytes - kStride);

    cursor.skip_bytes(kStride);
    EXPECT_EQ(cursor.remaining(0), kBytes - 2 * kStride);

    cursor.advance(0, kBytes - 2 * kStride);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, Basic1DEntryWithPayload)
  {
    AddressList addrlist;
    const int dim = 1;
    std::vector<size_t> payload = {10, 20, 30};

    size_t *entry = addrlist.being_entry(dim, payload.size());
    ASSERT_NE(entry, nullptr);

    entry[dim * AddressList::ADDRLIST_DIM_SLOTS] = payload.size();
    std::copy(payload.begin(), payload.end(),
              entry + dim * AddressList::ADDRLIST_DIM_SLOTS + 1);
    entry[0] = AddressList::pack_entry_header(kBytes, dim);
    addrlist.commit_entry(dim, kBytes, payload.size());

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    size_t payload_count = 0;
    const size_t *loaded_payload = cursor.get_payload(payload_count);
    ASSERT_NE(loaded_payload, nullptr);
    ASSERT_EQ(payload_count, payload.size());
    for(size_t i = 0; i < payload.size(); ++i)
      EXPECT_EQ(payload[i], loaded_payload[i]);
  }

  TEST(AddressListTests, Multiple1DEntries)
  {
    AddressList addrlist;
    const size_t entries = 10;

    for(size_t i = 0; i < entries; ++i) {
      size_t *entry = addrlist.being_entry(1);
      ASSERT_NE(entry, nullptr);
      entry[0] = AddressList::pack_entry_header(kBytes, 1);
      addrlist.commit_entry(1, kBytes);
    }

    EXPECT_EQ(addrlist.bytes_pending(), entries * kBytes);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);
    for(size_t i = 0; i < entries; ++i) {
      EXPECT_EQ(cursor.remaining(0), kBytes);
      cursor.advance(0, kBytes);
    }
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, Complex3DEntry)
  {
    AddressList addrlist;

    size_t *entry = addrlist.being_entry(3);
    entry[0] = AddressList::pack_entry_header(kBytes, 3);
    entry[1] = 0;    // base offset
    entry[2] = 8;    // dim1 count
    entry[3] = 1024; // dim1 stride
    entry[4] = 2;    // dim2 count
    entry[5] = 8192; // dim2 stride
    const size_t volume = kBytes * 8 * 2;
    addrlist.commit_entry(3, volume);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(cursor.remaining(0), kBytes);
    EXPECT_EQ(cursor.remaining(1), 8);
    EXPECT_EQ(cursor.remaining(2), 2);
    EXPECT_EQ(cursor.get_offset(), 0);

    cursor.advance(2, 2);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, WraparoundBufferSafety)
  {
    const size_t max_entries = 16;
    AddressList addrlist(max_entries);

    size_t successful = 0;
    while(true) {
      size_t *entry = addrlist.being_entry(1);
      if(!entry)
        break;
      entry[0] = AddressList::pack_entry_header(kBytes, 1);
      addrlist.commit_entry(1, kBytes);
      successful++;
    }

    EXPECT_GT(successful, 0);
    EXPECT_LE(successful, max_entries);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);
    for(size_t i = 0; i < successful; ++i) {
      cursor.advance(0, kBytes);
    }
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

} // namespace
