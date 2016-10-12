// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#ifndef PANDAS_TYPES_INTEGER_H
#define PANDAS_TYPES_INTEGER_H

#include "pandas/config.h"

#include "pandas/array.h"
#include "pandas/types.h"
#include "pandas/status.h"

namespace pandas {

class Buffer;

class PANDAS_EXPORT IntegerArray : public Array {
 public:
  int64_t GetNullCount() override;

 protected:
  IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data);
  IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits);

  Status EnsureMutable() override;

  std::shared_ptr<Buffer> data_;
  std::shared_ptr<Buffer> valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT IntegerArrayImpl : public IntegerArray {
 public:
  using T = typename TYPE::c_type;

  PyObject* GetItem(int64_t i) override;
  Status SetItem(int64_t i, PyObject* val) override;

  const T* data() const;
  T* mutable_data() const;
};

using Int8Array = IntegerArrayImpl<Int8Type>;
using UInt8Array = IntegerArrayImpl<UInt8Type>;

using Int16Array = IntegerArrayImpl<Int16Type>;
using UInt16Array = IntegerArrayImpl<UInt16Type>;

using Int32Array = IntegerArrayImpl<Int32Type>;
using UInt32Array = IntegerArrayImpl<UInt32Type>;

using Int64Array = IntegerArrayImpl<Int64Type>;
using UInt64Array = IntegerArrayImpl<UInt64Type>;

extern template class PANDAS_EXPORT IntegerArrayImpl<Int8Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt8Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int16Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt16Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int32Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt32Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<Int64Type>;
extern template class PANDAS_EXPORT IntegerArrayImpl<UInt64Type>;

} // namespace pandas

#endif // PANDAS_TYPES_INTEGER_H
