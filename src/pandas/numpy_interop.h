// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <Python.h>

#include <numpy/numpyconfig.h>

#define NPY_1_7_API_VERSION 1

// Don't use the deprecated Numpy functions
#ifdef NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
#define NPY_ARRAY_NOTSWAPPED NPY_NOTSWAPPED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#define NPY_ARRAY_UPDATEIFCOPY NPY_UPDATEIFCOPY
#endif

// This is required to be able to access the NumPy C API properly in C++ files
// other than this main one
#define PY_ARRAY_UNIQUE_SYMBOL pandas_ARRAY_API
#ifndef NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/pytypes.h"
#include "pandas/type.h"
#include "pandas/visibility.h"

namespace pandas {

inline int import_numpy() {
#ifdef NUMPY_IMPORT_ARRAY
  import_array1(-1);
  import_umath1(-1);
#endif

  return 0;
}

std::shared_ptr<DataType> PANDAS_EXPORT PandasTypeFromNumPy(PyArray_Descr* dtype);

// These are zero-copy if the data is contiguous (not strided)
std::shared_ptr<Array> PANDAS_EXPORT CreateArrayFromNumPy(PyArrayObject* arr);
std::shared_ptr<Array> PANDAS_EXPORT CreateArrayFromMaskedNumPy(
    PyArrayObject* arr, PyArrayObject* mask);

// Container for strided (but contiguous) data contained in a NumPy array
class NumPyBuffer : public MutableBuffer {
 public:
  NumPyBuffer(PyArrayObject* arr);
  virtual ~NumPyBuffer();

  PyArrayObject* array() const { return reinterpret_cast<PyArrayObject*>(arr_); }
  PyArray_Descr* dtype() { return PyArray_DESCR(array()); }

 protected:
  PyArrayObject* arr_;
};

template <int NPY_TYPE>
struct NumPyTraits {};

#define NUMPY_TRAITS_DECL(NPY_TYPE, PandasArrayType) \
  template <>                                        \
  struct NumPyTraits<NPY_TYPE> {                     \
    using ArrayType = PandasArrayType;               \
    using T = typename PandasArrayType::T;           \
  }

NUMPY_TRAITS_DECL(NPY_INT8, Int8Array);
NUMPY_TRAITS_DECL(NPY_INT16, Int16Array);
NUMPY_TRAITS_DECL(NPY_INT32, Int32Array);
NUMPY_TRAITS_DECL(NPY_INT64, Int64Array);
NUMPY_TRAITS_DECL(NPY_UINT8, UInt8Array);
NUMPY_TRAITS_DECL(NPY_UINT16, UInt16Array);
NUMPY_TRAITS_DECL(NPY_UINT32, UInt32Array);
NUMPY_TRAITS_DECL(NPY_UINT64, UInt64Array);
NUMPY_TRAITS_DECL(NPY_FLOAT32, FloatArray);
NUMPY_TRAITS_DECL(NPY_FLOAT64, DoubleArray);
NUMPY_TRAITS_DECL(NPY_OBJECT, PyObjectArray);
NUMPY_TRAITS_DECL(NPY_BOOL, BooleanArray);

}  // namespace pandas
