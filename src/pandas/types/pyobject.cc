// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/pyobject.h"

#include <cstdint>

#include "pandas/type.h"
#include "pandas/type_traits.h"

namespace pandas {

PyObjectArray::PyObjectArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : Array(get_type_singleton<PyObjectType>(), length),
      data_(data),
      valid_bits_(valid_bits) {}

int64_t PyObjectArray::GetNullCount() {
  // TODO(wesm)
  return 0;
}

PyObject* PyObjectArray::GetItem(int64_t i) {
  return nullptr;
}

Status PyObjectArray::Copy(
    int64_t offset, int64_t length, std::shared_ptr<Array>* out) const {
  std::shared_ptr<Buffer> copied_data;
  RETURN_NOT_OK(
      data_->Copy(offset * sizeof(PyObject*), length * sizeof(PyObject*), &copied_data));

  PyObject** values =
      reinterpret_cast<PyObject**>(const_cast<uint8_t*>(copied_data->data()));
  for (int64_t i = 0; i < length; ++i) {
    if (values[i] != nullptr) { Py_INCREF(values[i]); }
  }

  *out = std::make_shared<PyObjectArray>(length, copied_data);
  return Status::OK();
}

Status PyObjectArray::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

bool PyObjectArray::owns_data() const {
  // TODO(wesm): Address Buffer-level data ownership
  return true;
}

PyObject** PyObjectArray::data() const {
  return mutable_data();
}

PyObject** PyObjectArray::mutable_data() const {
  // const PyObject** is unpleasant to work with
  return reinterpret_cast<PyObject**>(const_cast<uint8_t*>(data_->data()));
}

}  // namespace pandas
