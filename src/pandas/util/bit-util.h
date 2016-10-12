// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "pandas/visibility.h"

namespace pandas {

class Buffer;
class Status;

namespace BitUtil {

static inline int64_t CeilByte(int64_t size) {
  return (size + 7) & ~7;
}

static inline int64_t BytesForBits(int64_t size) {
  return CeilByte(size) / 8;
}

static inline int64_t Ceil2Bytes(int64_t size) {
  return (size + 15) & ~15;
}

static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline bool GetBit(const uint8_t* bits, int i) {
  return static_cast<bool>(bits[i / 8] & kBitmask[i % 8]);
}

static inline bool BitNotSet(const uint8_t* bits, int i) {
  return (bits[i / 8] & kBitmask[i % 8]) == 0;
}

static inline void ClearBit(uint8_t* bits, int i) {
  bits[i / 8] &= ~kBitmask[i % 8];
}

static inline void SetBit(uint8_t* bits, int i) {
  bits[i / 8] |= kBitmask[i % 8];
}

static inline int64_t NextPower2(int64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}

static inline bool IsMultipleOf64(int64_t n) {
  return (n & 63) == 0;
}

inline int64_t RoundUpToMultipleOf64(int64_t num) {
  // TODO(wesm): is this definitely needed?
  // DCHECK_GE(num, 0);
  constexpr int64_t round_to = 64;
  constexpr int64_t force_carry_addend = round_to - 1;
  constexpr int64_t truncate_bitmask = ~(round_to - 1);
  constexpr int64_t max_roundable_num = std::numeric_limits<int64_t>::max() - round_to;
  if (num <= max_roundable_num) { return (num + force_carry_addend) & truncate_bitmask; }
  // handle overflow case.  This should result in a malloc error upstream
  return num;
}

void BytesToBits(const std::vector<uint8_t>& bytes, uint8_t* bits);
PANDAS_EXPORT Status BytesToBits(const std::vector<uint8_t>&, std::shared_ptr<Buffer>*);

}  // namespace BitUtil
}  // namespace pandas
