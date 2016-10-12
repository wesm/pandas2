// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/integer.h"

#include <cstdint>
#include <cstring>

#include "pandas/buffer.h"
#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/numpy_interop.h"
#include "pandas/pytypes.h"
#include "pandas/util/bit-util.h"

namespace pandas {

// ----------------------------------------------------------------------
// Any integer

IntegerArray::IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data)
    : Array(type, length),
      data_(data),
      valid_bits_(nullptr) {}

IntegerArray::IntegerArray(const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : Array(type, length),
      data_(data),
      valid_bits_(valid_bits) {}

int64_t IntegerArray::GetNullCount() {
  // TODO(wesm)
  // return nulls_.set_count();
  return 0;
}

Status IntegerArray::EnsureMutable() {
  // TODO(wesm):
  return Status::OK();
}

// ----------------------------------------------------------------------
// Typed integers

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

Status AllocateValidityBitmap(int64_t length, std::shared_ptr<Buffer>* out) {
  int64_t nbytes = BitUtil::BytesForBits(length);
  auto buf = std::make_shared<PoolBuffer>();
  RETURN_NOT_OK(buf->Resize(nbytes));

  // Set to all 1s, since all valid
  memset(buf->mutable_data(), 0xFF, nbytes);

  *out = buf;
  return Status::OK();
}

static Status PyObjectToInt64(PyObject *obj, int64_t* out)
{
  PyObject *num = PyNumber_Long(obj);

  RETURN_IF_PYERROR();
  *out = PyLong_AsLongLong(num);
  Py_DECREF(num);
  return Status::OK();
}

template <typename TYPE>
Status IntegerArrayImpl<TYPE>::SetItem(int64_t i, PyObject* val) {
  RETURN_NOT_OK(EnsureMutable());

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
