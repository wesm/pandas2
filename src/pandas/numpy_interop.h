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
#include "pandas/type.h"

namespace pandas {

inline int import_numpy() {
#ifdef NUMPY_IMPORT_ARRAY
  import_array1(-1);
  import_umath1(-1);
#endif

  return 0;
}

Status PandasTypeFromNumPy(PyArray_Descr* dtype, std::shared_ptr<DataType>* type);

// These are zero-copy if the data is contiguous (not strided)
Status CreateArrayFromNumPy(PyArrayObject* arr, std::shared_ptr<Array>* out);
Status CreateArrayFromMaskedNumPy(
    PyArrayObject* arr, PyArrayObject* mask, std::shared_ptr<Array>* out);

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

}  // namespace pandas
