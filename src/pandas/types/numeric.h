// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

template <typename TYPE>
class PANDAS_EXPORT NumericArray : public Array {
 public:
  using T = typename TYPE::c_type;
  using Array::Array;

  NumericArray(const std::shared_ptr<DataType>& type, int64_t length,
      const std::shared_ptr<Buffer>& data);

  auto data() const -> const T*;
  auto mutable_data() const -> T*;

  std::shared_ptr<Buffer> data_buffer() const;

 protected:
  std::shared_ptr<Buffer> data_;
};

template <typename TYPE>
class PANDAS_EXPORT IntegerArray : public NumericArray<TYPE> {
 public:
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data);
  IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits);

  int64_t GetNullCount() override;

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  bool owns_data() const override;

  std::shared_ptr<Buffer> valid_buffer() const;

 protected:
  std::shared_ptr<Buffer> valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArray : public NumericArray<TYPE> {
 public:
  FloatingArray(int64_t length, const std::shared_ptr<Buffer>& data);

  int64_t GetNullCount() override;
  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  bool owns_data() const override;
};

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

class PANDAS_EXPORT BooleanArray : public UInt8Array {
 public:
  BooleanArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits = nullptr);

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;
};

}  // namespace pandas
