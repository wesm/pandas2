// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#ifndef PANDAS_UTIL_BITARRAY_H
#define PANDAS_UTIL_BITARRAY_H

#include <cstdint>
#include <cstdlib>

#include "pandas/common.h"

namespace pandas {

class BitArray {
 public:
  BitArray() : length_(0), bits_(nullptr), count_(0) {}
  ~BitArray();

  void Init(int64_t length);

  bool IsSet(int64_t i) { return bits_[i / 8] & (1 << (i % 8)); }

  void Set(int64_t i) {
    if (!IsSet(i)) ++count_;
    bits_[i / 8] |= (1 << (i % 8));
  }

  void Unset(int64_t i) {
    if (IsSet(i)) --count_;
    // clear bit
    bits_[i / 8] &= ~(1 << (i % 8));
  }

  // Set a range from start (inclusive) to end (not inclusive)
  // Bounds are not checked
  // void SetRange(int64_t start, int64_t end) {
  //   for (int64_t i = start; i < end; ++i) {
  //     Set(i);
  //   }
  // }

  // Unset a range from start (inclusive) to end (not inclusive)
  // Bounds are not checked
  // void UnsetRange(int64_t start, int64_t end) {
  //   for (int64_t i = start; i < end; ++i) {
  //     Unset(i);
  //   }
  // }

  int64_t set_count() { return count_; }

  int64_t length() { return length_; }

 private:
  int64_t length_;
  uint8_t* bits_;
  int64_t count_;
};

}  // namespace pandas

#endif  // PANDAS_UTIL_BITARRAY_H
