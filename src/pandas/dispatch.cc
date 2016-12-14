// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/dispatch.h"

#include <memory>

#include "pandas/common.h"

namespace pandas {

#define MAKE_TYPE_CASE(NAME, FACTORY) \
  case NAME:                          \
    return FACTORY();

std::shared_ptr<DataType> primitive_type_from_enum(TypeId tp_enum) {
  switch (tp_enum) {
    MAKE_TYPE_CASE(TypeId::INT8, int8);
    MAKE_TYPE_CASE(TypeId::INT16, int16);
    MAKE_TYPE_CASE(TypeId::INT32, int32);
    MAKE_TYPE_CASE(TypeId::INT64, int64);
    MAKE_TYPE_CASE(TypeId::UINT8, uint8);
    MAKE_TYPE_CASE(TypeId::UINT16, uint16);
    MAKE_TYPE_CASE(TypeId::UINT32, uint32);
    MAKE_TYPE_CASE(TypeId::UINT64, uint64);
    MAKE_TYPE_CASE(TypeId::FLOAT32, float32);
    MAKE_TYPE_CASE(TypeId::FLOAT64, float64);
    MAKE_TYPE_CASE(TypeId::BOOL, boolean);
    MAKE_TYPE_CASE(TypeId::PYOBJECT, pyobject);
    default:
      break;
  }
  throw NotImplementedError("Not implemented");
}

}  // namespace pandas
