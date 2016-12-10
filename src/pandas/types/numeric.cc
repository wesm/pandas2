// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/numeric.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/pytypes.h"
#include "pandas/type.h"
#include "pandas/types/common.h"

namespace pandas {

// ----------------------------------------------------------------------
// Generic numeric class

template <typename TYPE>
NumericArray<TYPE>::NumericArray(const DataTypePtr& type, int64_t length,
    const std::shared_ptr<Buffer>& data, const std::shared_ptr<Buffer>& valid_bits)
    : Array(type, length, 0), data_(data), valid_bits_(valid_bits) {}

template <typename TYPE>
auto NumericArray<TYPE>::data() const -> const T* {
  return reinterpret_cast<const T*>(data_->data());
}

template <typename TYPE>
auto NumericArray<TYPE>::mutable_data() const -> T* {
  auto mutable_buf = static_cast<MutableBuffer*>(data_.get());
  return reinterpret_cast<T*>(mutable_buf->mutable_data());
}

template <typename TYPE>
std::shared_ptr<Buffer> NumericArray<TYPE>::data_buffer() const {
  return data_;
}

template <typename TYPE>
const TYPE& NumericArray<TYPE>::type_reference() const {
  return dynamic_cast<const TYPE&>(*type_);
}

// ----------------------------------------------------------------------
// Floating point class

template <typename TYPE>
FloatingArray<TYPE>::FloatingArray(int64_t length, const std::shared_ptr<Buffer>& data)
    : NumericArray<TYPE>(TYPE::SINGLETON, length, data, nullptr) {}

template <typename TYPE>
int64_t FloatingArray<TYPE>::GetNullCount() {
  // TODO(wesm)
  return 0;
}

template <typename TYPE>
Status FloatingArray<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;

  RETURN_NOT_OK(this->data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  *out = std::make_shared<FloatingArray<TYPE>>(length, copied_data);
  return Status::OK();
}

template <typename TYPE>
bool FloatingArray<TYPE>::owns_data() const {
  return this->data_.use_count() == 1;
}

// ----------------------------------------------------------------------
// Typed integers

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data)
    : IntegerArray(length, data, nullptr) {}

template <typename TYPE>
IntegerArray<TYPE>::IntegerArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : NumericArray<TYPE>(TYPE::SINGLETON, length, data, valid_bits) {}

template <typename TYPE>
int64_t IntegerArray<TYPE>::GetNullCount() {
  // TODO(wesm)
  // return nulls_.set_count();
  return 0;
}

template <typename TYPE>
bool IntegerArray<TYPE>::owns_data() const {
  bool owns_data = this->data_.use_count() == 1;
  if (valid_bits_) { owns_data &= valid_bits_.use_count() == 1; }
  return owns_data;
}

template <typename TYPE>
Status IntegerArray<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  std::shared_ptr<Buffer> copied_valid_bits;

  RETURN_NOT_OK(this->data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  if (valid_bits_) {
    RETURN_NOT_OK(CopyBitmap(this->data_, offset, length, &copied_valid_bits));
  }
  *out = std::make_shared<IntegerArray<TYPE>>(length, copied_data);
  return Status::OK();
}

// Instantiate templates
template class IntegerArray<UInt8Type>;
template class IntegerArray<Int8Type>;
template class IntegerArray<UInt16Type>;
template class IntegerArray<Int16Type>;
template class IntegerArray<UInt32Type>;
template class IntegerArray<Int32Type>;
template class IntegerArray<UInt64Type>;
template class IntegerArray<Int64Type>;
template class FloatingArray<FloatType>;
template class FloatingArray<DoubleType>;

template class NumericArray<BooleanType>;
template class NumericArray<UInt8Type>;
template class NumericArray<Int8Type>;
template class NumericArray<UInt16Type>;
template class NumericArray<Int16Type>;
template class NumericArray<UInt32Type>;
template class NumericArray<Int32Type>;
template class NumericArray<UInt64Type>;
template class NumericArray<Int64Type>;
template class NumericArray<FloatType>;
template class NumericArray<DoubleType>;

// ----------------------------------------------------------------------
// Implement Boolean as subclass of UInt8

BooleanArray::BooleanArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : IntegerArray<BooleanType>(length, data, valid_bits) {
  type_ = BooleanType::SINGLETON;
}

}  // namespace pandas
