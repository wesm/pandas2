// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "arrow/util/bit-util.h"
#include "arrow/util/buffer.h"
#include "arrow/util/status.h"

#include "pandas/visibility.h"

namespace pandas {

// ----------------------------------------------------------------------
// Common imports from libarrow

namespace BitUtil = arrow::BitUtil;
using Buffer = arrow::Buffer;
using MutableBuffer = arrow::MutableBuffer;
using ResizableBuffer = arrow::ResizableBuffer;
using PoolBuffer = arrow::PoolBuffer;
using Status = arrow::Status;

// Bitmap utilities
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

class BitmapWriter {
 public:
  explicit BitmapWriter(uint8_t* bitmap) : bitmap_(bitmap), cycle_(0), zero_count_(0) {}

  void CheckCycle() {
    if (cycle_ == 8) {
      ++bitmap_;
      cycle_ = 0;
    }
  }

  void Set1() {
    *bitmap_ |= kBitmask[cycle_++];
    CheckCycle();
  }

  void Set0() {
    *bitmap_ &= ~kBitmask[cycle_++];
    ++zero_count_;
    CheckCycle();
  }

 private:
  uint8_t* bitmap_;
  int cycle_;
  int64_t zero_count_;
};

constexpr size_t kMemoryAlignment = 64;

}  // namespace pandas
