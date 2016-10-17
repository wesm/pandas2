// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/type.h"

namespace pandas {

class Status;

Status primitive_type_from_enum(DataType::TypeId tp_enum, DataType** out);
}
