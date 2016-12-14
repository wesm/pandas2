// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <memory>

#include "pandas/type.h"

namespace pandas {

std::shared_ptr<DataType> primitive_type_from_enum(TypeId tp_enum);

}  // namespace pandas
