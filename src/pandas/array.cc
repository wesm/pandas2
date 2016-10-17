// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/array.h"
#include "pandas/status.h"
#include "pandas/type.h"
#include "pandas/util/logging.h"

namespace pandas {

// ----------------------------------------------------------------------
// Array

Array::Array(const std::shared_ptr<DataType>& type, int64_t length)
    : type_(type), length_(length) {}

Status Array::Copy(std::shared_ptr<Array>* out) const {
  return Copy(0, length_, out);
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

Status ArrayView::EnsureMutable() {
  if (ref_count() > 1) {
    std::shared_ptr<Array> copied_data;
    RETURN_NOT_OK(data_->Copy(&copied_data));
    data_ = copied_data;
  }
  return Status::OK();
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

}  // namespace pandas
