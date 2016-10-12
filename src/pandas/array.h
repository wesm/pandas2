// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <memory>
#include <string>
#include <vector>

#include "pandas/types.h"
#include "pandas/util.h"

namespace pandas {

// Forward declarations
class Status;

class Array {
 public:
  virtual ~Array() {}

  int64_t length() const { return length_;}
  std::shared_ptr<DataType> type() const { return type_;}
  DataType::TypeId type_id() const { return type_->type();}

  virtual int64_t GetNullCount() = 0;

  virtual PyObject* GetItem(int64_t i) = 0;
  virtual Status SetItem(int64_t i, PyObject* val) = 0;

 protected:
  std::shared_ptr<DataType> type_;
  int64_t length_;

  Array(const std::shared_ptr<DataType>& type, int64_t length)
      : type_(type), length_(length) {}

  virtual Status EnsureMutable() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(Array);
};


typedef std::shared_ptr<Array> ArrayPtr;

// TODO: define an operator model

// Shallow copy
// virtual Status Copy(Array** out);

// Type casting
// virtual Status Cast(const TypePtr& new_type);

// Python get scalar
// virtual PyObject* GetValue(int64_t i) = 0;
// virtual void SetValue(int64_t i, PyObject* val) = 0;

// ----------------------------------------------------------------------
// Indexing

// Slice the array. The result is not a copy but will have copy-on-write
// semantics
// virtual Status Slice(int64_t start, int64_t end, Array** out);

// virtual Status Filter(Array* mask, Array** out);

// virtual Status Put(Array* indices, Array* values, Array** out);
// virtual Status Take(Array* indices, Array** out);

// ----------------------------------------------------------------------
// Array-Array binary operations

// Array-Scalar binary operations

// ----------------------------------------------------------------------
// Array equality
// virtual bool Equals(Array* other);
// virtual bool AlmostEquals(Array* other);

// virtual Status IsNull(Array** out);
// virtual Status NotNull(Array** out);

// ----------------------------------------------------------------------
// Array APIs from NumPy / APL-variants
// virtual Status Repeat(int64_t repeats, Array** out);
// virtual Status Repeat(Array* repeats, Array** out);

// virtual Status Tile(int64_t tiles, Array** out);

// ----------------------------------------------------------------------
// Missing data methods

// virtual Status FillNull(const Scalar& fill_value, Array** out);

// ----------------------------------------------------------------------
// Hash table-based functions

// virtual Status Isin(Array* other, Array** out);
// virtual Status Match(Array* other, Array** out);
// virtual Status Unique(Array** out);
// virtual Status ValueCounts(Array** out);

// Requires a scalar box
// virtual Status Mode(Scalar* out);

} // namespace pandas
