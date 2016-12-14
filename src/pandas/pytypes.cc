// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include <Python.h>

#include "pandas/pytypes.h"

#include <cstdint>

#include "pandas/type.h"

namespace pandas {

namespace py {

PyObject* NAType;
PyObject* NA;

void init_natype(PyObject* na_type, PyObject* na_singleton) {
  Py_INCREF(na_type);
  Py_INCREF(na_singleton);
  NAType = na_type;
  NA = na_singleton;
}

}  // namespace py

std::string GetPythonError() {
  PyObject *exc_type, *exc_value, *traceback;
  PyErr_Fetch(&exc_type, &exc_value, &traceback);
  PyObjectStringify stringified(exc_value);
  std::string message(stringified.bytes);
  Py_XDECREF(exc_type);
  Py_XDECREF(exc_value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return message;
}

PyObjectArray::PyObjectArray(int64_t length, const std::shared_ptr<Buffer>& data,
    const std::shared_ptr<Buffer>& valid_bits)
    : Array(PyObjectType::SINGLETON, length, 0), data_(data), valid_bits_(valid_bits) {}

int64_t PyObjectArray::GetNullCount() {
  // TODO(wesm)
  return 0;
}

std::shared_ptr<Array> PyObjectArray::Copy(int64_t offset, int64_t length) const {
  std::shared_ptr<Buffer> copied_data;
  PANDAS_THROW_NOT_OK(
      data_->Copy(offset * sizeof(PyObject*), length * sizeof(PyObject*), &copied_data));

  PyObject** values =
      reinterpret_cast<PyObject**>(const_cast<uint8_t*>(copied_data->data()));
  for (int64_t i = 0; i < length; ++i) {
    if (values[i] != nullptr) { Py_INCREF(values[i]); }
  }

  return std::make_shared<PyObjectArray>(length, copied_data);
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

const PyObjectType& PyObjectArray::type_reference() const {
  return dynamic_cast<const PyObjectType&>(*type_);
}

}  // namespace pandas
