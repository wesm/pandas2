// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/numeric.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include "pandas/buffer.h"
#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/pytypes.h"
#include "pandas/status.h"
#include "pandas/type.h"
#include "pandas/types/common.h"
#include "pandas/util/bit-util.h"

namespace pandas {

template <typename T>
static inline std::shared_ptr<DataType> get_type_singleton() {
  return nullptr;
}

#define MAKE_TYPE_SINGLETON(NAME)                                       \
  static const auto k##NAME = std::make_shared<NAME##Type>();           \
  template <>                                                           \
  inline std::shared_ptr<DataType> get_type_singleton<NAME##Type>() {   \
    return k##NAME;                                                     \
  }

MAKE_TYPE_SINGLETON(Int8);
MAKE_TYPE_SINGLETON(UInt8);
MAKE_TYPE_SINGLETON(Int16);
MAKE_TYPE_SINGLETON(UInt16);
MAKE_TYPE_SINGLETON(Int32);
MAKE_TYPE_SINGLETON(UInt32);
MAKE_TYPE_SINGLETON(Int64);
MAKE_TYPE_SINGLETON(UInt64);
MAKE_TYPE_SINGLETON(Float);
MAKE_TYPE_SINGLETON(Double);

// ----------------------------------------------------------------------
// Floating point base class

FloatingArray::FloatingArray(
    const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data)
    : NumericArray(type, length),
      data_(data) {}

// ----------------------------------------------------------------------
// Specific implementations

template <typename TYPE>
FloatingArrayImpl<TYPE>::FloatingArrayImpl(
    int64_t length, const std::shared_ptr<Buffer>& data)
    : FloatingArray(get_type_singleton<TYPE>(), length, data) {}

template <typename TYPE>
int64_t FloatingArrayImpl<TYPE>::GetNullCount() {
  // TODO(wesm)
  return 0;
}

template <typename TYPE>
PyObject* FloatingArrayImpl<TYPE>::GetItem(int64_t i) {
  return NULL;
}

template <typename TYPE>
Status FloatingArrayImpl<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;

  RETURN_NOT_OK(data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  *out = std::make_shared<FloatingArrayImpl<TYPE>>(length, copied_data);
  return Status::OK();
}

template <typename TYPE>
Status FloatingArrayImpl<TYPE>::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

template <typename TYPE>
bool FloatingArrayImpl<TYPE>::owns_data() const {
  return data_.use_count() == 1;
}

// Instantiate templates
template class FloatingArrayImpl<FloatType>;
template class FloatingArrayImpl<DoubleType>;

// ----------------------------------------------------------------------
// Any integer

IntegerArray::IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data)
    : NumericArray(type, length),
      data_(data),
      valid_bits_(nullptr) {}

IntegerArray::IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : NumericArray(type, length),
      data_(data),
      valid_bits_(valid_bits) {}

int64_t IntegerArray::GetNullCount() {
  // TODO(wesm)
  // return nulls_.set_count();
  return 0;
}

// ----------------------------------------------------------------------
// Typed integers

template <typename TYPE>
IntegerArrayImpl<TYPE>::IntegerArrayImpl(
    int64_t length, const std::shared_ptr<Buffer>& data)
    : IntegerArray(get_type_singleton<TYPE>(), length, data) {}

template <typename TYPE>
const typename TYPE::c_type* IntegerArrayImpl<TYPE>::data() const {
  return reinterpret_cast<const T*>(data_->data());
}

template <typename TYPE>
typename TYPE::c_type* IntegerArrayImpl<TYPE>::mutable_data() const {
  return reinterpret_cast<T*>(data_->mutable_data());
}

template <typename TYPE>
PyObject* IntegerArrayImpl<TYPE>::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  return PyLong_FromLongLong(data()[i]);
}

template <typename TYPE>
bool IntegerArrayImpl<TYPE>::owns_data() const {
  bool owns_data = data_.use_count() == 1;
  if (valid_bits_) {
    owns_data &= valid_bits_.use_count() == 1;
  }
  return owns_data;
}

template <typename TYPE>
Status IntegerArrayImpl<TYPE>::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  size_t itemsize = sizeof(typename TYPE::c_type);

  std::shared_ptr<Buffer> copied_data;
  std::shared_ptr<Buffer> copied_valid_bits;

  RETURN_NOT_OK(data_->Copy(offset * itemsize, length * itemsize, &copied_data));

  if (valid_bits_) {
    RETURN_NOT_OK(CopyBitmap(data_, offset, length, &copied_valid_bits));
  }
  *out = std::make_shared<FloatingArrayImpl<TYPE>>(length, copied_data);
  return Status::OK();
}

static Status PyObjectToInt64(PyObject *obj, int64_t* out) {
  PyObject *num = PyNumber_Long(obj);

  RETURN_IF_PYERROR();
  *out = PyLong_AsLongLong(num);
  Py_DECREF(num);
  return Status::OK();
}

template <typename TYPE>
Status IntegerArrayImpl<TYPE>::SetItem(int64_t i, PyObject* val) {
  if (py::is_na(val)) {
    if (!valid_bits_) {
      // TODO: raise Python exception on error status
      RETURN_NOT_OK(AllocateValidityBitmap(length_, &valid_bits_));
    }
    BitUtil::ClearBit(valid_bits_->mutable_data(), i);
  } else {
    if (valid_bits_) {
      BitUtil::SetBit(valid_bits_->mutable_data(), i);
    }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}


// Instantiate templates
template class IntegerArrayImpl<UInt8Type>;
template class IntegerArrayImpl<Int8Type>;
template class IntegerArrayImpl<UInt16Type>;
template class IntegerArrayImpl<Int16Type>;
template class IntegerArrayImpl<UInt32Type>;
template class IntegerArrayImpl<Int32Type>;
template class IntegerArrayImpl<UInt64Type>;
template class IntegerArrayImpl<Int64Type>;

} // namespace pandas
