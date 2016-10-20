// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pandas/visibility.h"

namespace pandas {

class DataType {
 public:
  enum TypeId {
    // A degerate NULL type
    NA = 0,

    // Little-endian integer types
    UINT8 = 1,
    INT8 = 2,
    UINT16 = 3,
    INT16 = 4,
    UINT32 = 5,
    INT32 = 6,
    UINT64 = 7,
    INT64 = 8,

    // A boolean value represented as 1 byte
    BOOL = 9,

    // 4-byte floating point value
    FLOAT32 = 10,

    // 8-byte floating point value
    FLOAT64 = 11,

    // PyObject*
    PYOBJECT = 12,

    // Timestamp types
    TIMESTAMP = 13,
    TIMESTAMP_TZ = 14,

    // UTF8 variable-length string
    STRING = 15,

    // Categorical
    CATEGORY = 16
  };

  explicit DataType(TypeId type) : type_(type) {}

  virtual ~DataType() {}

  virtual std::string ToString() const = 0;

  virtual bool Equals(const DataType& other) { return type_ == other.type_; }

  TypeId type() const { return type_; }

 private:
  TypeId type_;
};

typedef std::shared_ptr<DataType> TypePtr;

class PANDAS_EXPORT TimestampType : public DataType {
 public:
  enum class Unit : char { SECOND = 0, MILLISECOND = 1, MICROSECOND = 2, NANOSECOND = 3 };

  Unit unit;

  explicit TimestampType(Unit unit = Unit::MICROSECOND)
      : DataType(DataType::TIMESTAMP), unit(unit) {}

  TimestampType(const TimestampType& other) : TimestampType(other.unit) {}

  static char const* name() { return "timestamp"; }

  std::string ToString() const override;
};

class PANDAS_EXPORT PyObjectType : public DataType {
 public:
  PyObjectType() : DataType(DataType::PYOBJECT) {}

  PyObjectType(const PyObjectType& other) : PyObjectType() {}

  static char const* name() { return "object"; }

  std::string ToString() const override;
};

template <typename DERIVED, typename C_TYPE, DataType::TypeId TYPE_ID,
    std::size_t SIZE = sizeof(C_TYPE)>
class PANDAS_EXPORT NumericType : public DataType {
 public:
  using c_type = C_TYPE;
  static constexpr DataType::TypeId type_id = TYPE_ID;
  static constexpr size_t size = SIZE;

  NumericType() : DataType(type_id) {}

  std::string ToString() const override { return std::string(DERIVED::NAME); }

  static std::shared_ptr<DERIVED> SINGLETON;
};

template <typename DERIVED, typename C_TYPE, DataType::TypeId TYPE_ID, std::size_t SIZE>
std::shared_ptr<DERIVED> NumericType<DERIVED, C_TYPE, TYPE_ID, SIZE>::SINGLETON(
    std::move(std::make_shared<DERIVED>()));

class PANDAS_EXPORT NullType : public DataType {
 public:
  NullType() : DataType(DataType::TypeId::NA) {}

  std::string ToString() const override { return std::string("null"); }
};

class PANDAS_EXPORT UInt8Type
    : public NumericType<UInt8Type, std::uint8_t, DataType::TypeId::UINT8> {
 public:
  constexpr static const char* NAME = "uint8";
};

class PANDAS_EXPORT Int8Type
    : public NumericType<Int8Type, std::int8_t, DataType::TypeId::INT8> {
 public:
  constexpr static const char* NAME = "int8";
};

class PANDAS_EXPORT UInt16Type
    : public NumericType<UInt16Type, std::uint16_t, DataType::TypeId::UINT16> {
 public:
  constexpr static const char* NAME = "uint16";
};

class PANDAS_EXPORT Int16Type
    : public NumericType<Int16Type, std::int16_t, DataType::TypeId::INT16> {
 public:
  constexpr static const char* NAME = "int16";
};

class PANDAS_EXPORT UInt32Type
    : public NumericType<UInt32Type, std::uint32_t, DataType::TypeId::UINT32> {
 public:
  constexpr static const char* NAME = "uint32";
};

class PANDAS_EXPORT Int32Type
    : public NumericType<Int32Type, std::int32_t, DataType::TypeId::INT32> {
 public:
  constexpr static const char* NAME = "int32";
};

class PANDAS_EXPORT UInt64Type
    : public NumericType<UInt64Type, std::uint64_t, DataType::TypeId::UINT64> {
 public:
  constexpr static const char* NAME = "uint64";
};

class PANDAS_EXPORT Int64Type
    : public NumericType<Int64Type, std::int64_t, DataType::TypeId::INT64> {
 public:
  constexpr static const char* NAME = "int64";
};

class PANDAS_EXPORT FloatType
    : public NumericType<FloatType, float, DataType::TypeId::FLOAT32> {
 public:
  constexpr static const char* NAME = "float32";
};

class PANDAS_EXPORT DoubleType
    : public NumericType<DoubleType, double, DataType::TypeId::FLOAT64> {
 public:
  constexpr static const char* NAME = "float64";
};

class PANDAS_EXPORT BooleanType
    : public NumericType<BooleanType, std::uint8_t, DataType::TypeId::BOOL> {
 public:
  constexpr static const char* NAME = "bool";
};

}  // namespace pandas
