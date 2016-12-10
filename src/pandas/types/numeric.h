// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

template <typename TYPE>
class PANDAS_EXPORT NumericScalar : public Scalar {
 public:
  using TypeClass = TYPE;
  using T = typename TYPE::c_type;
  using Scalar::Scalar;

  NumericScalar(T value, bool is_null)
      : Scalar(TYPE::SINGLETON, is_null), value_(value) {}

  T value() const { return value_; }

 protected:
  T value_;
};

template <typename TYPE>
class PANDAS_EXPORT NumericArray : public Array {
 public:
  using T = typename TYPE::c_type;
  using DataTypePtr = std::shared_ptr<TYPE>;
  using Array::Array;

  NumericArray(const DataTypePtr& type, int64_t length,
      const std::shared_ptr<Buffer>& data, const std::shared_ptr<Buffer>& valid_bits);

  auto data() const -> const T*;
  auto mutable_data() const -> T*;

  std::shared_ptr<Buffer> data_buffer() const;

  // Despite being virtual, compiler could inline this if
  // the call is performed with a NumericArray reference
  const TYPE& type_reference() const override;

  std::shared_ptr<Buffer> valid_bits() const;

 protected:
  std::shared_ptr<TYPE> type_;
  std::shared_ptr<Buffer> data_;
  std::shared_ptr<Buffer> valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT IntegerArray : public NumericArray<TYPE> {
 public:
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data);
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits);

  int64_t GetNullCount() override;

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  bool owns_data() const override;

 protected:
  using NumericArray<TYPE>::valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArray : public NumericArray<TYPE> {
 public:
  FloatingArray(int64_t length, const std::shared_ptr<Buffer>& data);

  int64_t GetNullCount() override;
  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  bool owns_data() const override;
};

using FloatScalar = NumericScalar<FloatType>;
using DoubleScalar = NumericScalar<DoubleType>;
using Int8Scalar = NumericScalar<Int8Type>;
using UInt8Scalar = NumericScalar<UInt8Type>;
using Int16Scalar = NumericScalar<Int16Type>;
using UInt16Scalar = NumericScalar<UInt16Type>;
using Int32Scalar = NumericScalar<Int32Type>;
using UInt32Scalar = NumericScalar<UInt32Type>;
using Int64Scalar = NumericScalar<Int64Type>;
using UInt64Scalar = NumericScalar<UInt64Type>;

using FloatArray = FloatingArray<FloatType>;
using DoubleArray = FloatingArray<DoubleType>;

using Int8Array = IntegerArray<Int8Type>;
using UInt8Array = IntegerArray<UInt8Type>;

using Int16Array = IntegerArray<Int16Type>;
using UInt16Array = IntegerArray<UInt16Type>;

using Int32Array = IntegerArray<Int32Type>;
using UInt32Array = IntegerArray<UInt32Type>;

using Int64Array = IntegerArray<Int64Type>;
using UInt64Array = IntegerArray<UInt64Type>;

// Only instantiate these templates once
extern template class PANDAS_EXPORT IntegerArray<Int8Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt8Type>;
extern template class PANDAS_EXPORT IntegerArray<Int16Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt16Type>;
extern template class PANDAS_EXPORT IntegerArray<Int32Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt32Type>;
extern template class PANDAS_EXPORT IntegerArray<Int64Type>;
extern template class PANDAS_EXPORT IntegerArray<UInt64Type>;
extern template class PANDAS_EXPORT FloatingArray<FloatType>;
extern template class PANDAS_EXPORT FloatingArray<DoubleType>;

class PANDAS_EXPORT BooleanArray : public IntegerArray<BooleanType> {
 public:
  BooleanArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits = nullptr);

  // PyObject* GetItem(int64_t i) override;
  // Status SetItem(int64_t i, PyObject* val) override;
};

}  // namespace pandas
