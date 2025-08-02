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

#include "realm.h"

using namespace Realm;

template <int N, typename T, int N2 = N, typename T2 = T>
bool test_apply_transform(std::vector<T2> transform, const Point<N2, T2> &offset,
                          const Rect<N, T> &rect, const Rect<N2, T2> &exp_rect)
{
  Matrix<N2, N, T2> transpose;
  for(int i = 0; i < N2; i++) {
    for(int j = 0; j < N; j++) {
      transpose[i][j] = transform[j * N2 + i];
    }
  }
  return rect.apply_transform(transpose, offset) == exp_rect;
}

template <typename T, typename T2>
void run_tests()
{

  assert((test_apply_transform<2, T, 2, T2>(
      /*transform=*/{-1, 0, 0, 1}, /*offset=*/Point<2, T2>({0, 0}),
      Rect<2, T>(/*lo=*/Point<2, T>({0, 0}), /*hi=*/Point<2, T>({-4, 5})),
      Rect<2, T2>(/*lo=*/Point<2, T>({4, 0}), /*hi=*/Point<2, T>({0, 5})))));

  assert((test_apply_transform<1, T, 1, T2>(
      /*transform=*/{-1}, /*offset=*/{{0}}, Rect<1, T>(/*lo=*/{{0}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{-10}}, /*hi=*/{{0}}))));

  assert((test_apply_transform<1, T, 1, T2>(
      /*transform=*/{1}, /*offset=*/{{0}}, Rect<1, T>(/*lo=*/{{0}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{0}}, /*hi=*/{{10}}))));

  assert((test_apply_transform<1, T, 1, T2>(
      /*transform=*/{1}, /*offset=*/{{2}}, Rect<1, T>(/*lo=*/{{0}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{2}}, /*hi=*/{{12}}))));

  assert(!(test_apply_transform<1, T, 1, T2>(
      /*transform=*/{1}, /*offset=*/{{2}}, Rect<1, T>(/*lo=*/{{0}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{1}}, /*hi=*/{{12}}))));

  assert((test_apply_transform<1, T, 1, T2>(
      /*transform=*/{2}, /*offset=*/{{2}}, Rect<1, T>(/*lo=*/{{4}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{10}}, /*hi=*/{{22}}))));

  assert((test_apply_transform<1, T, 1, T2>(
      /*transform=*/{-2}, /*offset=*/{{-2}}, Rect<1, T>(/*lo=*/{{4}}, /*hi=*/{{10}}),
      Rect<1, T2>(/*lo=*/{{-22}}, /*hi=*/{{-10}}))));

  assert((test_apply_transform<2, T, 2, T2>(
      /*transform=*/{1, 0, 0, 1}, /*offset=*/Point<2, T2>({0, 2}),
      Rect<2, T>(/*lo=*/Point<2, T>({0, 0}), /*hi=*/Point<2, T>({4, 5})),
      Rect<2, T2>(/*lo=*/Point<2, T>({0, 2}), /*hi=*/Point<2, T>({4, 7})))));

  assert((test_apply_transform<2, T, 2, T2>(
      /*transform=*/{0, 1, 2, 0}, /*offset=*/Point<2, T2>({0, 0}),
      Rect<2, T>(/*lo=*/Point<2, T>({0, 0}), /*hi=*/Point<2, T>({4, 5})),
      Rect<2, T2>(/*lo=*/Point<2, T>({0, 0}), /*hi=*/Point<2, T>({10, 4})))));

  std::cout << "All tests passed" << std::endl;
}

int main(int argc, char **argv)
{
  run_tests<int, int>();
  return 0;
}
