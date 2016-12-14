// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <Python.h>

#include "pandas/common.h"

namespace pandas {

static inline void catch_exception() {
  try {
    // Pass through Python exceptions
    if (!PyErr_Occurred()) {
      throw;
    }
  } catch (const NotImplementedError& e) {
    PyErr_SetString(PyExc_NotImplementedError, e.what());
  } catch (const OutOfMemory& e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
  } catch (const IOError& e) {
    PyErr_SetString(PyExc_IOError, e.what());
  } catch (const TypeError& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  } catch (const ValueError& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const PandasException& e) {
    PyErr_SetString(PyExc_Exception, e.what());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
}

} // namespace pandas
