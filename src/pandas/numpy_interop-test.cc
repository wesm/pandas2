
// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/numpy_interop.h"
#include "pandas/test-util.h"
#include "pandas/type.h"
#include "pandas/type_traits.h"
#include "pandas/types/numeric.h"

namespace pandas {

Status AllocateNDArray1D(int type, int64_t length, PyArrayObject** out) {
  npy_intp dims[1] = {length};
  *out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, type));
  RETURN_IF_PYERROR();
  return Status::OK();
}

Status ContiguousFromNumPy(int type, int64_t length, std::shared_ptr<Array>* out) {
  PyArrayObject* np_arr;
  RETURN_NOT_OK(AllocateNDArray1D(type, length, &np_arr));
  OwnedRef ref(PyObject * (np_arr));
  return CreateArrayFromNumPy(np_arr, out);
}

void CheckNDArray(int npy_type_num, DataType::TypeId ex_type) {}

static constexpr int kLength = 10;
static constexpr int kNpyTypes[10] = {NPY_UINT8, NPY_INT8, NPY_UINT16, NPY_INT16,
    NPY_UINT32, NPY_INT32, NPY_UINT64, NPY_INT64, NPY_FLOAT, NPY_DOUBLE};

static constexpr DataType::TypeId kPandasTypes[10] = {DataType::UINT8, DataType::INT8,
    DataType::UINT16, DataType::INT16, DataType::UINT32, DataType::INT32,
    DataType::UINT64, DataType::INT64, DataType::FLOAT32, DataType::FLOAT64};

TEST(TestNumPyInterop, TestMetadataBasics) {
  const int length = 50;
  std::shared_ptr<Array> out;
  for (size_t i = 0; i < kLength; ++i) {
    ASSERT_OK(ContiguousFromNumPy(kNpyTypes[i], length, &out));
    ASSERT_EQ(kPandasTypes[i], out->type_id());
  }
}

void AssertNoAllocatedMemory() {
  // Make sure we are starting off at zero
  ASSERT_EQ(0, memory_pool()->bytes_allocated());
}

// Non-strided (zero copy) tests
TEST(TestNumPyInterop, ZeroCopyInit) {
  AssertNoAllocatedMemory();

  const int length = 50;
  std::shared_ptr<Array> out;

  // These are all zero copy
  for (size_t i = 0; i < kLength; ++i) {
    int npy_num = kNpyTypes[i];
    ASSERT_OK(ContiguousFromNumPy(npy_num, length, &out));
    ASSERT_EQ(0, memory_pool()->bytes_allocated());
  }
}

// Strided tests

void AssertNoPythonError() {
  if (PyErr_Occurred()) { ASSERT_OK(GetPythonError()); }
}

void AssertMemorySize(int64_t expected, int64_t actual) {
  ASSERT_EQ(BitUtil::RoundUpToMultipleOf64(expected), memory_pool()->bytes_allocated());
}

template <int NPY_TYPE>
void CheckStridedNumbers() {
  using ArrayType = typename NumPyTraits<NPY_TYPE>::ArrayType;
  using T = typename ArrayType::T;
  std::vector<T> values = {0, 1, 2, 3, 0, 5, 6, 0};

  const int length = 4;
  const int stride = 2;

  npy_intp dims[1] = {length};
  npy_intp strides[1] = {stride * sizeof(T)};

  PyArray_Descr* dtype = PyArray_DescrFromType(NPY_TYPE);
  OwnedRef ref(PyArray_NewFromDescr(&PyArray_Type, dtype, 1, dims, strides,
      reinterpret_cast<void*>(values.data()), 0, NULL));
  AssertNoPythonError();

  PyArrayObject* np_arr = reinterpret_cast<PyArrayObject*>(ref.get());

  std::shared_ptr<Array> out;
  ASSERT_OK(CreateArrayFromNumPy(np_arr, &out));

  // Check that the values were copied over
  const auto ap = static_cast<const ArrayType*>(out.get());
  for (int i = 0; i < length; ++i) {
    ASSERT_EQ(values[i * stride], ap->data()[i]);
  }

  AssertMemorySize(length * sizeof(T), memory_pool()->bytes_allocated());
}

TEST(TestNumPyInterop, StridedCopy) {
  AssertNoAllocatedMemory();

  CheckStridedNumbers<NPY_BOOL>();

  CheckStridedNumbers<NPY_INT8>();
  CheckStridedNumbers<NPY_UINT8>();
  CheckStridedNumbers<NPY_INT64>();
  CheckStridedNumbers<NPY_UINT64>();

  CheckStridedNumbers<NPY_FLOAT>();
  CheckStridedNumbers<NPY_DOUBLE>();
}

TEST(TestNumPyInterop, StridedPyObject) {}

// Initialize homogenous collection of arrays with zero copy

TEST(TestNumPyInterop, NDArray2DFortranOrderZeroCopy) {}

}  // namespace pandas
