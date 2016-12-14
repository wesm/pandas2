// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <string>

#include "arrow/buffer.h"
#include "arrow/status.h"
#include "arrow/util/bit-util.h"

#include "pandas/visibility.h"

namespace pandas {

// ----------------------------------------------------------------------
// Common imports from libarrow

namespace BitUtil = arrow::BitUtil;
using Buffer = arrow::Buffer;
using MutableBuffer = arrow::MutableBuffer;
using ResizableBuffer = arrow::ResizableBuffer;
using PoolBuffer = arrow::PoolBuffer;

class PANDAS_EXPORT PandasException : public std::exception {
 public:
  static void NYI(const std::string& msg);
  static void Throw(const std::string& msg);

  explicit PandasException(const char* msg);
  explicit PandasException(const std::string& msg);
  explicit PandasException(const char* msg, exception& e);

  virtual ~PandasException() throw();
  virtual const char* what() const throw();

 private:
  std::string msg_;
};

class PANDAS_EXPORT NotImplementedError : public PandasException {
 public:
  using PandasException::PandasException;
};
class PANDAS_EXPORT OutOfMemory : public PandasException {
 public:
  using PandasException::PandasException;
};
class PANDAS_EXPORT IOError : public PandasException {
 public:
  using PandasException::PandasException;
};
class PANDAS_EXPORT TypeError : public PandasException {
 public:
  using PandasException::PandasException;
};
class PANDAS_EXPORT ValueError : public PandasException {
 public:
  using PandasException::PandasException;
};

#define PANDAS_THROW_NOT_OK(s)                               \
  do {                                                       \
    ::arrow::Status _s = (s);                                \
    if (!_s.ok()) { PandasException::Throw(_s.ToString()); } \
  } while (0);

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
