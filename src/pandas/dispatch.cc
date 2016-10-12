// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/dispatch.h"

#include "pandas/status.h"

namespace pandas {

#define MAKE_TYPE_CASE(NAME, CapName)           \
  case NAME:                                    \
    *out = new CapName##Type();                 \
    break;

Status primitive_type_from_enum(DataType::TypeId tp_enum, DataType** out) {
  switch (tp_enum) {
    MAKE_TYPE_CASE(DataType::INT8, Int8);
    MAKE_TYPE_CASE(DataType::INT16, Int16);
    MAKE_TYPE_CASE(DataType::INT32, Int32);
    MAKE_TYPE_CASE(DataType::INT64, Int64);
    MAKE_TYPE_CASE(DataType::UINT8, UInt8);
    MAKE_TYPE_CASE(DataType::UINT16, UInt16);
    MAKE_TYPE_CASE(DataType::UINT32, UInt32);
    MAKE_TYPE_CASE(DataType::UINT64, UInt64);
    MAKE_TYPE_CASE(DataType::FLOAT, Float);
    MAKE_TYPE_CASE(DataType::DOUBLE, Double);
    MAKE_TYPE_CASE(DataType::BOOL, Boolean);
    MAKE_TYPE_CASE(DataType::PYOBJECT, PyObject);
    default:
      return Status::NotImplemented("Not a primitive type");
  }
  return Status::OK();
}

} // namespace pandas
