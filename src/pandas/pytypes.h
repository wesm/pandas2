// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#ifndef PANDAS_PYTYPES_H
#define PANDAS_PYTYPES_H

#include <Python.h>

#include "pandas/common.h"

namespace pandas {
namespace py {

void init_natype(PyObject* na_type, PyObject* na_singleton);

extern PyObject* NAType;
extern PyObject* NA;

static inline bool is_na(PyObject* obj) {
  return (obj == NA) || PyObject_IsInstance(obj, NAType);
}

}  // namespace py

class OwnedRef {
 public:
  OwnedRef() : obj_(nullptr) {}

  explicit OwnedRef(PyObject* obj) : obj_(obj) {}

  template <typename T>
  explicit OwnedRef(T* obj) : OwnedRef(reinterpret_cast<PyObject*>(obj)) {}

  ~OwnedRef() { Py_XDECREF(obj_); }

  void reset(PyObject* obj) {
    if (obj_ != nullptr) { Py_XDECREF(obj_); }
    obj_ = obj;
  }

  PyObject* release() {
    PyObject* ret = obj_;
    obj_ = nullptr;
    return ret;
  }

  PyObject* get() const { return obj_; }

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
Status GetPythonError();

#define RETURN_IF_PYERROR() \
  if (PyErr_Occurred()) { return GetPythonError(); }

}  // namespace pandas

#endif  // PANDAS_PYTYPES_H
