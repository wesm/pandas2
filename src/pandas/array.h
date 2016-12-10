// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pandas/common.h"
#include "pandas/type.h"
#include "pandas/util.h"

namespace pandas {

struct ColumnStatistics {
  // If the array has been mutated, this will be set to true to indicate that
  // the statistics need to be recomputed.
  bool dirty;

  bool is_monotonic;
  int64_t null_count;
  int64_t unique_count;
};

// A typed value, either scalar or array
class Value {
 public:
  enum class Kind : char { SCALAR = 0, ARRAY = 1 };

  Value(Kind kind, const std::shared_ptr<DataType>& type) : kind_(kind), type_(type) {}

  Kind kind() const { return kind_; }
  std::shared_ptr<DataType> type() const { return type_; }
  TypeId type_id() const { return type_->type(); }

 protected:
  Kind kind_;
  std::shared_ptr<DataType> type_;
};

// A value consisting of a single scalar element
class Scalar : public Value {
 public:
  Scalar(const std::shared_ptr<DataType>& type, bool is_null)
      : Value(Kind::SCALAR, type), is_null_(is_null) {}

  bool is_null() const { return is_null_; }

 protected:
  bool is_null_;
};

// A value as a sequence of multiple homogeneously-typed elements
class Array : public Value {
 public:
  virtual ~Array() {}

  int64_t length() const { return length_; }

  // There are two methods to obtain the data type.
  // The signature without a shared_ptr allows sub-classes
  // to have a covariant return type, which eliminates the
  // need/danger of doing a static_cast when dealing with
  // a concrete sub-class. Ideally, the shared_ptr signature
  // would suffice, but the compiler cannot treat a shared_ptr
  // to a base class and a shared_ptr to a subclass as a
  // covariant return type.
  std::shared_ptr<DataType> type() const { return type_; }
  virtual const DataType& type_reference() const = 0;

  // Copy a section of the array into a new output array
  virtual Status Copy(
      int64_t offset, int64_t length, std::shared_ptr<Array>* out) const = 0;

  // Copy the entire array (using the virtual Copy function)
  Status Copy(std::shared_ptr<Array>* out) const;

  virtual int64_t GetNullCount() = 0;

  // For each array type, determine if all of its memory buffers belong to it
  // (for determining if they can be safely mutated). Otherwise, they may need
  // to be copied (for copy-on-write operations)
  virtual bool owns_data() const = 0;

 protected:
  int64_t length_;
  int64_t offset_;

  Array(const std::shared_ptr<DataType>& type, int64_t length, int64_t offset);
  Array(const std::shared_ptr<DataType>& type, int64_t length);

 private:
  DISALLOW_COPY_AND_ASSIGN(Array);
};

// An object that is a view on a section of another array (possibly the whole
// array). This is used to implement slicing and copy-on-write.
class ArrayView {
 public:
  ArrayView() {}

  explicit ArrayView(const std::shared_ptr<Array>& data);
  ArrayView(const std::shared_ptr<Array>& data, int64_t offset);
  ArrayView(const std::shared_ptr<Array>& data, int64_t offset, int64_t length);

  // Copy / move constructor
  ArrayView(const ArrayView& other);
  ArrayView(ArrayView&& other);

  // Copy / move assignment
  ArrayView& operator=(const ArrayView& other);
  ArrayView& operator=(ArrayView&& other);

  // If the contained array is not the sole reference to that array, then
  // mutation operations must produce a copy of the referenced
  Status EnsureMutable();

  // Construct view from start offset to the end of the array
  ArrayView Slice(int64_t offset);

  // Construct view from start offset of the indicated length
  ArrayView Slice(int64_t offset, int64_t length);

  std::shared_ptr<Array> data() const { return data_; }
  int64_t offset() const { return offset_; }
  int64_t length() const { return length_; }

  // Return the reference count for the underlying array
  int64_t ref_count() const;

 private:
  std::shared_ptr<Array> data_;
  int64_t offset_;
  int64_t length_;
};

using ArrayPtr = std::shared_ptr<Array>;

}  // namespace pandas
