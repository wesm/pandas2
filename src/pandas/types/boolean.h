// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#ifndef PANDAS_TYPES_BOOLEAN_H
#define PANDAS_TYPES_BOOLEAN_H

#include <Python.h>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/numpy_interop.h"
#include "pandas/type.h"
#include "pandas/util/bitarray.h"

namespace pandas {

class BooleanArray : public Array {
 public:
  virtual PyObject* GetValue(size_t i);
  virtual void SetValue(size_t i, PyObject* val);

 protected:
  NumPyBuffer numpy_array_;
  BitArray nulls_;
};

}  // namespace pandas

#endif  // PANDAS_TYPES_BOOLEAN_H
