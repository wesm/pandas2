// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/dispatch.h"

#include "pandas/common.h"

namespace pandas {

#define MAKE_TYPE_CASE(NAME, CapName) \
  case NAME:                          \
    *out = new CapName##Type();       \
    break;

Status primitive_type_from_enum(TypeId tp_enum, DataType** out) {
  switch (tp_enum) {
    MAKE_TYPE_CASE(TypeId::INT8, Int8);
    MAKE_TYPE_CASE(TypeId::INT16, Int16);
    MAKE_TYPE_CASE(TypeId::INT32, Int32);
    MAKE_TYPE_CASE(TypeId::INT64, Int64);
    MAKE_TYPE_CASE(TypeId::UINT8, UInt8);
    MAKE_TYPE_CASE(TypeId::UINT16, UInt16);
    MAKE_TYPE_CASE(TypeId::UINT32, UInt32);
    MAKE_TYPE_CASE(TypeId::UINT64, UInt64);
    MAKE_TYPE_CASE(TypeId::FLOAT32, Float);
    MAKE_TYPE_CASE(TypeId::FLOAT64, Double);
    MAKE_TYPE_CASE(TypeId::BOOL, Boolean);
    MAKE_TYPE_CASE(TypeId::PYOBJECT, PyObject);
    default:
      return Status::NotImplemented("Not a primitive type");
  }
  return Status::OK();
}

}  // namespace pandas
