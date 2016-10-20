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
  ss << name() << "["
     << "]";
  return ss.str();
}

// Constexpr numeric type names
constexpr const char* UInt8Type::NAME;
constexpr const char* Int8Type::NAME;
constexpr const char* UInt16Type::NAME;
constexpr const char* Int16Type::NAME;
constexpr const char* UInt32Type::NAME;
constexpr const char* Int32Type::NAME;
constexpr const char* UInt64Type::NAME;
constexpr const char* Int64Type::NAME;
constexpr const char* FloatType::NAME;
constexpr const char* DoubleType::NAME;
constexpr const char* BooleanType::NAME;

std::shared_ptr<PyObjectType> PyObjectType::SINGLETON =
        std::move(std::make_shared<PyObjectType>());

}  // namespace pandas
