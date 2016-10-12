// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include "pandas/array.h"
#include "pandas/numpy_interop.h"
#include "pandas/types.h"

#include "pandas/status.h"

namespace pandas {

class Buffer;

class PANDAS_EXPORT FloatingArray : public Array {
 public:
  int64_t GetNullCount() override;

 protected:
  FloatingArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data);

  Status EnsureMutable() override;

  std::shared_ptr<Buffer> data_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArrayImpl : public FloatingArray {
 public:
  using T = typename TYPE::c_type;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  const T* data() const;
  T* mutable_data() const;
};

typedef FloatingArrayImpl<FloatType> FloatArray;
typedef FloatingArrayImpl<DoubleType> DoubleArray;

extern template class PANDAS_EXPORT FloatingArrayImpl<FloatType>;
extern template class PANDAS_EXPORT FloatingArrayImpl<DoubleType>;

} // namespace pandas
