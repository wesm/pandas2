// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#pragma once

#include <cstdint>

#include "pandas/visibility.h"

namespace pandas {

class Status;

class PANDAS_EXPORT MemoryPool {
 public:
  virtual ~MemoryPool();

  virtual Status Allocate(int64_t size, uint8_t** out) = 0;
  virtual void Free(uint8_t* buffer, int64_t size) = 0;

  virtual int64_t bytes_allocated() const = 0;
};

PANDAS_EXPORT MemoryPool* default_memory_pool();

}  // namespace pandas
