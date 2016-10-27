// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/numpy_interop.h"

#include <numpy/arrayobject.h>

#include <memory>

#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/type.h"
#include "pandas/type_traits.h"
#include "pandas/types/numeric.h"
#include "pandas/types/pyobject.h"

namespace pandas {

#define TYPE_MAP_CASE(NP_NAME, PD_CAP_TYPE) \
  case NPY_##NP_NAME:                       \
    *type = PD_CAP_TYPE##Type::SINGLETON;   \
    break;

Status PandasTypeFromNumPy(PyArray_Descr* npy_type, std::shared_ptr<DataType>* type) {
  switch (npy_type->type_num) {
    TYPE_MAP_CASE(INT8, Int8);
    TYPE_MAP_CASE(INT16, Int16);
    TYPE_MAP_CASE(INT32, Int32);
    TYPE_MAP_CASE(INT64, Int64);
    TYPE_MAP_CASE(UINT8, UInt8);
    TYPE_MAP_CASE(UINT16, UInt16);
    TYPE_MAP_CASE(UINT32, UInt32);
    TYPE_MAP_CASE(UINT64, UInt64);
    TYPE_MAP_CASE(FLOAT32, Float);
    TYPE_MAP_CASE(FLOAT64, Double);
    TYPE_MAP_CASE(BOOL, Boolean);
    // TYPE_MAP_CASE(OBJECT, PythonObject);
    default:
      return Status::NotImplemented("unsupported numpy type");
  }
  return Status::OK();
}

template <typename T>
void CopyStrided(T* input_data, int64_t length, int64_t stride, T* output_data) {
  // Passing input_data as non-const is a concession to PyObject*
  int64_t j = 0;
  for (int64_t i = 0; i < length; ++i) {
    output_data[i] = input_data[j];
    j += stride;
  }
}

template <>
void CopyStrided<PyObject*>(
    PyObject** input_data, int64_t length, int64_t stride, PyObject** output_data) {
  int64_t j = 0;
  for (int64_t i = 0; i < length; ++i) {
    output_data[i] = input_data[j];
    if (output_data[i] != nullptr) { Py_INCREF(output_data[i]); }
    j += stride;
  }
}

// ----------------------------------------------------------------------
// NumPy conversion (zero-copy when possible)

template <int NPY_TYPE>
inline Status WrapNumPyArray(PyArrayObject* arr, std::shared_ptr<Buffer>* out) {
  using T = typename NumPyTraits<NPY_TYPE>::T;

  if (PyArray_NDIM(arr) != 1) { return Status::Invalid("Only support 1D NumPy arrays "); }

  const int64_t stride = PyArray_STRIDES(arr)[0];
  const int64_t length = PyArray_SIZE(arr);

  const int64_t stride_elements = stride / sizeof(T);

  if (PyArray_DTYPE(arr)->elsize == stride) {
    // It is contiguous, zero copy possible
    *out = std::make_shared<NumPyBuffer>(arr);
  } else {
    // Strided, must copy into new contiguous memory
    auto new_buffer = std::make_shared<PoolBuffer>(memory_pool());
    RETURN_NOT_OK(new_buffer->Resize(sizeof(T) * length));
    CopyStrided(reinterpret_cast<T*>(PyArray_DATA(arr)), length, stride_elements,
        reinterpret_cast<T*>(new_buffer->mutable_data()));
    *out = new_buffer;
  }
  return Status::OK();
}

// TODO(wesm): Do we want to implement BooleanArray as bits?

// template <>
// inline Status WrapNumPyArray<NPY_BOOL>(PyArrayObject* arr, std::shared_ptr<Buffer>*
// out) {
//   if (PyArray_NDIM(arr) != 1) {
//     return Status::Invalid("Only support 1D NumPy arrays ");
//   }

//   const int64_t stride = PyArray_STRIDES(arr)[0];
//   const int64_t length = PyArray_SIZE(arr);
//   const int64_t bitmap_bytes = BitUtil::BytesForBits(length);

//   const uint8_t* data = reinterpret_cast<const uint8_t*>(PyArray_DATA(arr));

//   auto new_buffer = std::make_shared<PoolBuffer>(memory_pool());
//   uint8_t* out_data = new_buffer->mutable_data();
//   RETURN_NOT_OK(new_buffer->Resize(bitmap_bytes));
//   memset(out_data, 0x00, bitmap_bytes);

//   if (PyArray_DTYPE(arr)->elsize == stride) {
//     for (int64_t i = 0; i < length; ++i) {
//       if (data[i] > 0) {
//         BitUtil::SetBit(out_data, i);
//       }
//     }
//   } else {
//     for (int64_t i = 0; i < length; ++i) {
//       if (data[i * stride] > 0) {
//         BitUtil::SetBit(out_data, i);
//       }
//     }
//   }
// }

template <int NPY_TYPE>
inline Status ConvertNumPyArray(PyArrayObject* arr, std::shared_ptr<Array>* out) {
  using ArrayType = typename NumPyTraits<NPY_TYPE>::ArrayType;

  // Check if contiguous
  std::shared_ptr<Buffer> data;
  RETURN_NOT_OK(WrapNumPyArray<NPY_TYPE>(arr, &data));

  *out = std::make_shared<ArrayType>(PyArray_SIZE(arr), data);
  return Status::OK();
}

#define NUMPY_CONVERTER_CASE(NP_NAME)                          \
  case NPY_##NP_NAME:                                          \
    RETURN_NOT_OK(ConvertNumPyArray<NPY_##NP_NAME>(arr, out)); \
    break;

Status CreateArrayFromNumPy(PyArrayObject* arr, std::shared_ptr<Array>* out) {
  PyArray_Descr* dtype = PyArray_DTYPE(arr);
  switch (dtype->type_num) {
    NUMPY_CONVERTER_CASE(INT8);
    NUMPY_CONVERTER_CASE(INT16);
    NUMPY_CONVERTER_CASE(INT32);
    NUMPY_CONVERTER_CASE(INT64);
    NUMPY_CONVERTER_CASE(UINT8);
    NUMPY_CONVERTER_CASE(UINT16);
    NUMPY_CONVERTER_CASE(UINT32);
    NUMPY_CONVERTER_CASE(UINT64);
    NUMPY_CONVERTER_CASE(FLOAT32);
    NUMPY_CONVERTER_CASE(FLOAT64);
    NUMPY_CONVERTER_CASE(BOOL);
    NUMPY_CONVERTER_CASE(OBJECT);
    default:
      return Status::NotImplemented("unsupported numpy type");
  }
  return Status::OK();
}

// Convert a NumPy array to a pandas::Array with appropriate missing values set
// according to the passed uint8 dtype mask array
Status CreateArrayFromMaskedNumPy(
    PyArrayObject* arr, PyArrayObject* mask, std::shared_ptr<Array>* out) {
  return Status::NotImplemented("NYI");
}

// ----------------------------------------------------------------------
// NumPy array container

NumPyBuffer::NumPyBuffer(PyArrayObject* arr)
    : MutableBuffer(reinterpret_cast<uint8_t*>(PyArray_DATA(arr)), PyArray_NBYTES(arr)),
      arr_(arr) {
  Py_INCREF(arr);
}

NumPyBuffer::~NumPyBuffer() {
  Py_XDECREF(arr_);
}

}  // namespace pandas
