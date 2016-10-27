// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <cstdint>
#include <limits>
#include <string>

#include "arrow/util/bit-util.h"
#include "arrow/util/buffer.h"
#include "arrow/util/memory-pool.h"
#include "arrow/util/status.h"

#include "pandas/visibility.h"

namespace pandas {

// ----------------------------------------------------------------------
// Common imports from libarrow

namespace BitUtil = arrow::BitUtil;
using Buffer = arrow::Buffer;
using MutableBuffer = arrow::MutableBuffer;
using MemoryPool = arrow::MemoryPool;
using ResizableBuffer = arrow::ResizableBuffer;
using PoolBuffer = arrow::PoolBuffer;
using Status = arrow::Status;

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

constexpr size_t kMemoryAlignment = 64;

}  // namespace pandas
