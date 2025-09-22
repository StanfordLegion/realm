/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#define BENCHMARK_USE_JSON_FORMAT

class Stat {
public:
  Stat()
    : count(0)
    , mean(0.0)
    , sum(0.0)
    , square_sum(0.0)
    , smallest(std::numeric_limits<double>::max())
    , largest(-std::numeric_limits<double>::max())
  {}
  void reset() { *this = Stat(); }
  void sample(double s)
  {
    count++;
    if(s < smallest)
      smallest = s;
    if(s > largest)
      largest = s;
    sum += s;
    double delta0 = s - mean;
    mean += delta0 / count;
    double delta1 = s - mean;
    square_sum += delta0 * delta1;
  }
  void accumulate(const Stat &stat)
  {
    if(stat.get_smallest() < smallest)
      smallest = stat.get_smallest();
    if(stat.get_largest() > largest)
      largest = stat.get_largest();
    sum += stat.get_sum();
    count += stat.get_count();
    mean = sum / count;
    double delta0 = stat.get_average() - mean;
    double delta1 = stat.get_average() - mean;
    for (unsigned int i = 0; i < stat.get_count(); i++) {
      square_sum += delta0 * delta1;
    }
  }
  unsigned get_count() const { return count; }
  double get_average() const { return mean; }
  double get_sum() const { return sum; }
  double get_stddev() const
  {
    return get_variance() > 0.0 ? std::sqrt(get_variance()) : 0.0;
  }
  double get_variance() const { return square_sum / (count < 2 ? 1 : count - 1); }
  double get_smallest() const { return smallest; }
  double get_largest() const { return largest; }

  friend std::ostream &operator<<(std::ostream &os, const Stat &s);

private:
  unsigned count;
  double mean;
  double sum;
  double square_sum;
  double smallest;
  double largest;
};

std::ostream &operator<<(std::ostream &os, const Stat &s)
{
#ifdef BENCHMARK_USE_JSON_FORMAT
  return os << std::scientific << std::setprecision(2) << "average:" << s.get_average()
            << ", min:" << s.get_smallest() << ", max:" << s.get_largest()
            << ", std:" << s.get_stddev() << "(" << s.get_stddev() / s.get_average() * 100 << "%)"
            << ", N:" << s.get_count();
#else
  return os << std::scientific << std::setprecision(2)
            << s.get_average() /*<< "(+/-" << s.get_stddev() << ')'*/
            << ", MIN=" << s.get_smallest() << ", MAX=" << s.get_largest()
            << ", N=" << s.get_count();
#endif
}

void output_machine_config(void)
{
  Machine machine = Machine::get_machine();
  size_t nodes = machine.get_address_space_count();
  Machine::ProcessorQuery cpu_query =
      Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  size_t cpus = cpu_query.count();
  Machine::ProcessorQuery gpu_query =
      Machine::ProcessorQuery(machine).only_kind(Processor::TOC_PROC);
  size_t gpus = gpu_query.count();
  Machine::ProcessorQuery util_query =
      Machine::ProcessorQuery(machine).only_kind(Processor::UTIL_PROC);
  size_t utils = util_query.count();
  printf("MACHINE_CONFIGURATION {address_spaces:%zu, cpu:%zu, gpu:%zu, util:%zu}\n",
            nodes, cpus / nodes, gpus / nodes, utils / nodes);
}

#endif // ifndef BENCHMARK_COMMON_H