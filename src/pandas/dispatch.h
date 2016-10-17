// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

Status primitive_type_from_enum(DataType::TypeId tp_enum, DataType** out);

}  // namespace pandas
