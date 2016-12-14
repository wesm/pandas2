// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Take accepts an array of values and an array of integer indices, and
// produces an array of values selected according to the indices
//
// take(['a', 'b', 'c', 'd', 'e'], [3, 0, 1, 4, 3]) -> ['d', 'a', 'b', 'e', 'd']

#pragma once

#include <cstdint>

#include "pandas/common.h"
#include "pandas/expr.h"
#include "pandas/kernels/util.h"

namespace pandas {

class Array;
class DataType;

std::shared_ptr<Array> Take(
    std::shared_ptr<Array> values, std::shared_ptr<Array> indices);

// Compute take into pre-allocated memory
void TakeInto(std::shared_ptr<Array> values, std::shared_ptr<Array> indices,
    std::shared_ptr<Array> out);

// A Take operation on array expressions
class TakeOperation : public Operation {
 public:
  using Operation::Operation;
  std::shared_ptr<DataType> GetReturnType() const override;
};

template <typename VALUE_TYPE, typename INDEX_TYPE>
inline void take(
    const VALUE_TYPE& values, const INDEX_TYPE& indices, const VALUE_TYPE& out) {
  const auto values_data = values.data();
  const auto indices_data = indices.data();
  auto out_data = out.mutable_data();

  int64_t nvalues = values.length();
  int64_t nindices = indices.length();

  // Propagate bitmap
  BitmapWriter bit_writer(out.valid);

  if (indices.valid != nullptr) {
    for (int64_t i = 0; i < nindices; ++i) {
      auto index = indices.data[i];
      while (index < 0) {
        index += nvalues;
      }
      if (BitUtil::GetBit(indices.valid, i)) { out_data[i] = values_data[index]; }
    }
  } else if (values.valid != nullptr) {
    for (int64_t i = 0; i < nindices; ++i) {
      auto index = indices_data[i];
      while (index < 0) {
        index += nvalues;
      }
      if (BitUtil::GetBit(values.valid, index)) {
        out_data[i] = values_data[index];
        bit_writer.Set1();
      } else {
        bit_writer.Set0();
      }
    }
  }
}

}  // namespace pandas
