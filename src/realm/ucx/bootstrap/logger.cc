/*
 * Copyright 2025 NVIDIA Corporation
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

#include <iostream>
#include <fstream>

#include "logger.h"
#include "types.h"

namespace p2p {
  std::shared_ptr<Logger::p2p_log> Logger::instance_;

  std::shared_ptr<Logger::p2p_log> Logger::getInstance(int rank)
  {
    if(instance_ == nullptr) {
      // Read from env varaible; if env is not set use default value.
      std::string logfile_name = mesh::DEF_LOG + std::string(".") + std::to_string(rank);
      instance_ = std::make_shared<Logger::p2p_log>(std::ofstream(logfile_name));
    }
    return instance_;
  }
} // namespace p2p
