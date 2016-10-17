// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <Python.h>

#include "pandas/visibility.h"

#if PY_MAJOR_VERSION >= 3
#define PyString_Check PyUnicode_Check
#endif

namespace pandas {

PANDAS_EXPORT
void libpandas_init();

}  // namespace pandas
