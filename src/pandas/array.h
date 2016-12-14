// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pandas/array_fwd.h"
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
  // @throws : exception if copy fails
  virtual std::shared_ptr<Array> Copy(int64_t offset, int64_t length) const = 0;

  // Copy the entire array (using the virtual Copy function)
  std::shared_ptr<Array> Copy() const;

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
  void EnsureMutable();

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

  std::shared_ptr<Array> Copy(int64_t offset, int64_t length) const override;

  bool owns_data() const override;

 protected:
  using NumericArray<TYPE>::valid_bits_;
};

template <typename TYPE>
class PANDAS_EXPORT FloatingArray : public NumericArray<TYPE> {
 public:
  FloatingArray(int64_t length, const std::shared_ptr<Buffer>& data);

  int64_t GetNullCount() override;
  std::shared_ptr<Array> Copy(int64_t offset, int64_t length) const override;

  bool owns_data() const override;
};

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

class CategoryArray : public Array {
 public:
  CategoryArray(
      const std::shared_ptr<CategoryType>& type, const std::shared_ptr<Array>& codes);

  std::shared_ptr<Array> codes() const { return codes_; }
  std::shared_ptr<Array> categories() const { return type_->categories(); }

 private:
  std::shared_ptr<Array> codes_;
  std::shared_ptr<CategoryType> type_;
};

void CopyBitmap(const std::shared_ptr<Buffer>& bitmap, int64_t bit_offset, int64_t length,
    std::shared_ptr<Buffer>* out);

void AllocateValidityBitmap(int64_t length, std::shared_ptr<Buffer>* out);

}  // namespace pandas
