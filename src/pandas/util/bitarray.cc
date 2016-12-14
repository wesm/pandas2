// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/util/bitarray.h"

#include <new>

#include "pandas/common.h"

namespace pandas {

BitArray::~BitArray() {
  delete[] bits_;
}

void BitArray::Init(int64_t length) {
  int64_t bufsize = BitUtil::CeilByte(length / 8);
  try {
    bits_ = new uint8_t[bufsize];
    memset(bits_, 0, bufsize);
  } catch (const std::bad_alloc& e) { throw OutOfMemory("BitArray allocation failed"); }
  length_ = length;
  count_ = 0;
}

}  // namespace pandas
