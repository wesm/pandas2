// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#pragma once

#include <cstdint>
#include <mutex>

#include "pandas/common.h"

namespace pandas {

class PandasMemoryPool : public MemoryPool {
 public:
  PandasMemoryPool() : bytes_allocated_(0) {}
  virtual ~PandasMemoryPool();

  Status Allocate(int64_t size, uint8_t** out) override;
  void Free(uint8_t* buffer, int64_t size) override;
  int64_t bytes_allocated() const override;

 private:
  mutable std::mutex pool_lock_;
  int64_t bytes_allocated_;
};

PANDAS_EXPORT MemoryPool* memory_pool();

}  // namespace pandas
