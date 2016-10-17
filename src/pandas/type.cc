// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/type.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace pandas {

// ----------------------------------------------------------------------
// PyObject

std::string PyObjectType::ToString() const {
  return name();
}

// ----------------------------------------------------------------------
// Timestamp

std::string TimestampType::ToString() const {
  std::stringstream ss;

  // TODO(wesm): Add unit string
  ss << name() << "[" << "]";
  return ss.str();
}


}  // namespace pandas
