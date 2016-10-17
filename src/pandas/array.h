// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <memory>
#include <string>
#include <vector>

#include "pandas/type.h"
#include "pandas/util.h"

namespace pandas {

// Forward declarations
class Status;

// Base class for physical array data structures.
class Array {
 public:
  virtual ~Array() {}

  int64_t length() const { return length_;}
  std::shared_ptr<DataType> type() const { return type_;}
  DataType::TypeId type_id() const { return type_->type();}

  // Copy a section of the array into a new output array
  virtual Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const = 0;

  // Copy the entire array (using the virtual Copy function)
  Status Copy(std::shared_ptr<Array>* out) const;

  virtual int64_t GetNullCount() = 0;

  virtual PyObject* GetItem(int64_t i) = 0;
  virtual Status SetItem(int64_t i, PyObject* val) = 0;

  // For each array type, determine if all of its memory buffers belong to it
  // (for determining if they can be safely mutated). Otherwise, they may need
  // to be copied (for copy-on-write operations)
  virtual bool owns_data() const = 0;

 protected:
  std::shared_ptr<DataType> type_;
  int64_t length_;

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

} // namespace pandas
