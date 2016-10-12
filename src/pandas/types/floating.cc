// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/floating.h"

#include <cstdint>

#include "pandas/buffer.h"
#include "pandas/status.h"

namespace pandas {

// ----------------------------------------------------------------------
// Floating point base class

FloatingArray::FloatingArray(
    const TypePtr type, int64_t length, const std::shared_ptr<Buffer>& data)
    : Array(type, length),
      data_(data) {}

Status FloatingArray::EnsureMutable() {
  // TODO(wesm)
  return Status::OK();
}

int64_t FloatingArray::GetNullCount() {
  // TODO(wesm)
  return 0;
}

// ----------------------------------------------------------------------
// Specific implementations

template <typename T>
PyObject* FloatingArrayImpl<T>::GetItem(int64_t i) {
  return NULL;
}

template <typename T>
Status FloatingArrayImpl<T>::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

// Instantiate templates
template class FloatingArrayImpl<FloatType>;
template class FloatingArrayImpl<DoubleType>;

} // namespace pandas
