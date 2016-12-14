// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/type.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "pandas/array.h"
#include "pandas/type.h"

namespace pandas {

// ----------------------------------------------------------------------
// PyObject

std::string PyObjectType::ToString() const {
  return name();
}

std::shared_ptr<PyObjectType> PyObjectType::SINGLETON =
    std::move(std::make_shared<PyObjectType>());

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

std::string CategoryType::ToString() const {
  std::stringstream s;
  s << "category<" << category_type()->ToString() << ">";
  return s.str();
}

std::shared_ptr<const DataType> CategoryType::category_type() const {
  return categories_->type();
}

#define TYPE_FACTORY(NAME, KLASS)                                        \
  std::shared_ptr<DataType> NAME() {                                     \
    static std::shared_ptr<DataType> result = std::make_shared<KLASS>(); \
    return result;                                                       \
  }

TYPE_FACTORY(null, NullType);
TYPE_FACTORY(boolean, BooleanType);
TYPE_FACTORY(int8, Int8Type);
TYPE_FACTORY(uint8, UInt8Type);
TYPE_FACTORY(int16, Int16Type);
TYPE_FACTORY(uint16, UInt16Type);
TYPE_FACTORY(int32, Int32Type);
TYPE_FACTORY(uint32, UInt32Type);
TYPE_FACTORY(int64, Int64Type);
TYPE_FACTORY(uint64, UInt64Type);
TYPE_FACTORY(float32, FloatType);
TYPE_FACTORY(float64, DoubleType);
TYPE_FACTORY(pyobject, PyObjectType);

}  // namespace pandas
