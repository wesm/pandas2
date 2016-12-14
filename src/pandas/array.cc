// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/array.h"

#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/type.h"
#include "pandas/util/logging.h"

namespace pandas {

// ----------------------------------------------------------------------
// Array

Array::Array(const std::shared_ptr<DataType>& type, int64_t length, int64_t offset)
    : Value(Kind::ARRAY, type), length_(length), offset_(offset) {}

std::shared_ptr<Array> Array::Copy() const {
  return Copy(0, length());
}

void CopyBitmap(const std::shared_ptr<Buffer>& bitmap, int64_t bit_offset, int64_t length,
    std::shared_ptr<Buffer>* out) {
  // TODO(wesm): Optimize this bitmap copy for each bit_offset mod 8
  int64_t nbytes = BitUtil::BytesForBits(length);
  auto buf = std::make_shared<PoolBuffer>(memory_pool());
  PANDAS_THROW_NOT_OK(buf->Resize(nbytes));

  // Set to all 1s, since all valid
  memset(buf->mutable_data(), 0xFF, nbytes);

  *out = buf;
}

void AllocateValidityBitmap(int64_t length, std::shared_ptr<Buffer>* out) {
  int64_t nbytes = BitUtil::BytesForBits(length);
  auto buf = std::make_shared<PoolBuffer>(memory_pool());
  PANDAS_THROW_NOT_OK(buf->Resize(nbytes));

  // Set to all 1s, since all valid
  memset(buf->mutable_data(), 0xFF, nbytes);

  *out = buf;
}

// ----------------------------------------------------------------------
// ArrayView

ArrayView::ArrayView(const std::shared_ptr<Array>& data)
    : data_(data), offset_(0), length_(data->length()) {}

ArrayView::ArrayView(const std::shared_ptr<Array>& data, int64_t offset)
    : data_(data), offset_(offset), length_(data->length() - offset) {
  // Debugging sanity checks
  PANDAS_DCHECK_GE(offset, 0);
  PANDAS_DCHECK_LT(offset, data->length());
}

ArrayView::ArrayView(const std::shared_ptr<Array>& data, int64_t offset, int64_t length)
    : data_(data), offset_(offset), length_(length) {
  // Debugging sanity checks
  PANDAS_DCHECK_GE(offset, 0);
  PANDAS_DCHECK_LT(offset, data->length());
  PANDAS_DCHECK_GE(length, 0);
  PANDAS_DCHECK_LE(length, data->length() - offset);
}

// Copy ctor
ArrayView::ArrayView(const ArrayView& other)
    : data_(other.data_), offset_(other.offset_), length_(other.length_) {}

// Move ctor
ArrayView::ArrayView(ArrayView&& other)
    : data_(std::move(other.data_)), offset_(other.offset_), length_(other.length_) {}

// Copy assignment
ArrayView& ArrayView::operator=(const ArrayView& other) {
  data_ = other.data_;
  offset_ = other.offset_;
  length_ = other.length_;
  return *this;
}

// Move assignment
ArrayView& ArrayView::operator=(ArrayView&& other) {
  data_ = std::move(other.data_);
  offset_ = other.offset_;
  length_ = other.length_;
  return *this;
}

void ArrayView::EnsureMutable() {
  if (ref_count() > 1) { data_ = data_->Copy(); }
}

ArrayView ArrayView::Slice(int64_t offset) {
  return ArrayView(data_, offset_ + offset, length_ - offset);
}

ArrayView ArrayView::Slice(int64_t offset, int64_t length) {
  return ArrayView(data_, offset_ + offset, length);
}

// Return the reference count for the underlying array
int64_t ArrayView::ref_count() const {
  return data_.use_count();
}

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
std::shared_ptr<Array> FloatingArray<TYPE>::Copy(int64_t offset, int64_t length) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  PANDAS_THROW_NOT_OK(
      this->data_->Copy(offset * itemsize, length * itemsize, &copied_data));
  return std::make_shared<FloatingArray<TYPE>>(length, copied_data);
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
std::shared_ptr<Array> IntegerArray<TYPE>::Copy(int64_t offset, int64_t length) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  std::shared_ptr<Buffer> copied_valid_bits;

  PANDAS_THROW_NOT_OK(
      this->data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  if (valid_bits_) { CopyBitmap(this->data_, offset, length, &copied_valid_bits); }
  return std::make_shared<IntegerArray<TYPE>>(length, copied_data);
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
// CategoryArray

CategoryArray::CategoryArray(
    const std::shared_ptr<CategoryType>& type, const std::shared_ptr<Array>& codes)
    : Array(type, codes->length(), 0), codes_(codes), type_(type) {}

// ----------------------------------------------------------------------
// Implement Boolean as subclass of UInt8

BooleanArray::BooleanArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : IntegerArray<BooleanType>(length, data, valid_bits) {
  type_ = BooleanType::SINGLETON;
}

}  // namespace pandas
