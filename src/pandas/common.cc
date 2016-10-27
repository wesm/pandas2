// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/common.h"

namespace pandas {

Status GetPythonError() {
  PyObject *exc_type, *exc_value, *traceback;
  PyErr_Fetch(&exc_type, &exc_value, &traceback);
  PyObjectStringify stringified(exc_value);
  std::string message(stringified.bytes);
  Py_XDECREF(exc_type);
  Py_XDECREF(exc_value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return Status::UnknownError(message);
}

}  // namespace pandas
