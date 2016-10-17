// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include "pandas/array.h"
#include "pandas/status.h"
#include "pandas/type.h"

namespace pandas {

class Buffer;

class PANDAS_EXPORT NumericArray : public Array {
 public:
  using Array::Array;
};

class PANDAS_EXPORT IntegerArray : public NumericArray {
 public:
  int64_t GetNullCount() override;

 protected:
  IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data);
  IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits);

  std::shared_ptr<Buffer> data_;
  std::shared_ptr<Buffer> valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT IntegerArrayImpl : public IntegerArray {
 public:
  using T = typename TYPE::c_type;

  IntegerArrayImpl(int64_t length, const std::shared_ptr<Buffer>& data);

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  bool owns_data() const override;

  const T* data() const;
  T* mutable_data() const;
};

class PANDAS_EXPORT FloatingArray : public NumericArray {
 protected:
  FloatingArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data);
  std::shared_ptr<Buffer> data_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArrayImpl : public FloatingArray {
 public:
  using T = typename TYPE::c_type;

  FloatingArrayImpl(int64_t length, const std::shared_ptr<Buffer>& data);

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;
  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  int64_t GetNullCount() override;

  bool owns_data() const override;

  const T* data() const;
  T* mutable_data() const;
};

using FloatArray = FloatingArrayImpl<FloatType>;
using DoubleArray = FloatingArrayImpl<DoubleType>;

using Int8Array = IntegerArrayImpl<Int8Type>;
using UInt8Array = IntegerArrayImpl<UInt8Type>;

using Int16Array = IntegerArrayImpl<Int16Type>;
using UInt16Array = IntegerArrayImpl<UInt16Type>;

using Int32Array = IntegerArrayImpl<Int32Type>;
using UInt32Array = IntegerArrayImpl<UInt32Type>;

using Int64Array = IntegerArrayImpl<Int64Type>;
using UInt64Array = IntegerArrayImpl<UInt64Type>;

// Only instantiate these templates once
extern template class PANDAS_EXPORT IntegerArrayImpl<Int8Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt8Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int16Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt16Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int32Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt32Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int64Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt64Type>;
extern template class PANDAS_EXPORT FloatingArrayImpl<FloatType>;
extern template class PANDAS_EXPORT FloatingArrayImpl<DoubleType>;

}  // namespace pandas
