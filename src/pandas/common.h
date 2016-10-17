// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <string>

#include "pandas/config.h"

namespace pandas {

class OwnedRef {
 public:
  OwnedRef() : obj_(nullptr) {}

  explicit OwnedRef(PyObject* obj) : obj_(obj) {}

  ~OwnedRef() { Py_XDECREF(obj_); }

  void reset(PyObject* obj) {
    if (obj_ != nullptr) { Py_XDECREF(obj_); }
    obj_ = obj;
  }

  void release() { obj_ = nullptr; }

  PyObject* obj() const { return obj_; }

 private:
  PyObject* obj_;
};

struct PyObjectStringify {
  OwnedRef tmp_obj;
  const char* bytes;

  explicit PyObjectStringify(PyObject* obj) {
    PyObject* bytes_obj;
    if (PyUnicode_Check(obj)) {
      bytes_obj = PyUnicode_AsUTF8String(obj);
      tmp_obj.reset(bytes_obj);
    } else {
      bytes_obj = obj;
    }
    bytes = PyBytes_AsString(bytes_obj);
  }
};

class PyAcquireGIL {
 public:
  PyAcquireGIL() { state_ = PyGILState_Ensure(); }

  ~PyAcquireGIL() { PyGILState_Release(state_); }

 private:
  PyGILState_STATE state_;
  DISALLOW_COPY_AND_ASSIGN(PyAcquireGIL);
};

// TODO(wesm): We can just let errors pass through. To be explored later
#define RETURN_IF_PYERROR()                         \
  if (PyErr_Occurred()) {                           \
    PyObject *exc_type, *exc_value, *traceback;     \
    PyErr_Fetch(&exc_type, &exc_value, &traceback); \
    PyObjectStringify stringified(exc_value);       \
    std::string message(stringified.bytes);         \
    Py_XDECREF(exc_type);                           \
    Py_XDECREF(exc_value);                          \
    Py_XDECREF(traceback);                          \
    PyErr_Clear();                                  \
    return Status::UnknownError(message);           \
  }

}  // namespace pandas
