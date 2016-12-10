// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pandas/visibility.h"

namespace pandas {

class Array;

enum class TypeId : char {
  // A degenerate NULL type
  NA = 0,

  // Little-endian integer types
  INT8 = 1,
  INT16 = 2,
  INT32 = 3,
  INT64 = 4,

  UINT8 = 5,
  UINT16 = 6,
  UINT32 = 7,
  UINT64 = 8,

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

class DataType {
 public:

  explicit DataType(TypeId type) : type_(type) {}

  virtual ~DataType() {}

  virtual std::string ToString() const = 0;

  virtual bool Equals(const DataType& other) const { return type_ == other.type_; }

  TypeId type() const { return type_; }

 private:
  TypeId type_;
};

using TypePtr = std::shared_ptr<const DataType>;

class PANDAS_EXPORT TimestampType : public DataType {
 public:
  enum class Unit : char { SECOND = 0, MILLISECOND = 1, MICROSECOND = 2, NANOSECOND = 3 };

  Unit unit;

  explicit TimestampType(Unit unit = Unit::MICROSECOND)
      : DataType(TypeId::TIMESTAMP), unit(unit) {}

  TimestampType(const TimestampType& other) : TimestampType(other.unit) {}

  static char const* name() { return "timestamp"; }

  std::string ToString() const override;
};

class PANDAS_EXPORT PyObjectType : public DataType {
 public:
  PyObjectType() : DataType(TypeId::PYOBJECT) {}

  PyObjectType(const PyObjectType& other) : PyObjectType() {}

  static char const* name() { return "object"; }

  std::string ToString() const override;

  static std::shared_ptr<PyObjectType> SINGLETON;
};

template <typename DERIVED, typename C_TYPE, TypeId TYPE_ID,
    std::size_t SIZE = sizeof(C_TYPE)>
class PANDAS_EXPORT NumericType : public DataType {
 public:
  using c_type = C_TYPE;
  static constexpr TypeId type_id = TYPE_ID;
  static constexpr size_t size = SIZE;
  static constexpr bool is_primitive = true;

  NumericType() : DataType(type_id) {}

  std::string ToString() const override { return std::string(DERIVED::NAME); }

  static std::shared_ptr<DERIVED> SINGLETON;
};

template <typename DERIVED, typename C_TYPE, TypeId TYPE_ID, std::size_t SIZE>
std::shared_ptr<DERIVED> NumericType<DERIVED, C_TYPE, TYPE_ID, SIZE>::SINGLETON(
    std::move(std::make_shared<DERIVED>()));

class PANDAS_EXPORT NullType : public DataType {
 public:
  NullType() : DataType(TypeId::NA) {}

  std::string ToString() const override { return std::string("null"); }
};

class PANDAS_EXPORT UInt8Type
    : public NumericType<UInt8Type, std::uint8_t, TypeId::UINT8> {
 public:
  constexpr static const char* NAME = "uint8";
};

class PANDAS_EXPORT Int8Type
    : public NumericType<Int8Type, std::int8_t, TypeId::INT8> {
 public:
  constexpr static const char* NAME = "int8";
};

class PANDAS_EXPORT UInt16Type
    : public NumericType<UInt16Type, std::uint16_t, TypeId::UINT16> {
 public:
  constexpr static const char* NAME = "uint16";
};

class PANDAS_EXPORT Int16Type
    : public NumericType<Int16Type, std::int16_t, TypeId::INT16> {
 public:
  constexpr static const char* NAME = "int16";
};

class PANDAS_EXPORT UInt32Type
    : public NumericType<UInt32Type, std::uint32_t, TypeId::UINT32> {
 public:
  constexpr static const char* NAME = "uint32";
};

class PANDAS_EXPORT Int32Type
    : public NumericType<Int32Type, std::int32_t, TypeId::INT32> {
 public:
  constexpr static const char* NAME = "int32";
};

class PANDAS_EXPORT UInt64Type
    : public NumericType<UInt64Type, std::uint64_t, TypeId::UINT64> {
 public:
  constexpr static const char* NAME = "uint64";
};

class PANDAS_EXPORT Int64Type
    : public NumericType<Int64Type, std::int64_t, TypeId::INT64> {
 public:
  constexpr static const char* NAME = "int64";
};

class PANDAS_EXPORT FloatType
    : public NumericType<FloatType, float, TypeId::FLOAT32> {
 public:
  constexpr static const char* NAME = "float32";
};

class PANDAS_EXPORT DoubleType
    : public NumericType<DoubleType, double, TypeId::FLOAT64> {
 public:
  constexpr static const char* NAME = "float64";
};

class PANDAS_EXPORT BooleanType
    : public NumericType<BooleanType, std::uint8_t, TypeId::BOOL> {
 public:
  constexpr static const char* NAME = "bool";
};

class PANDAS_EXPORT CategoryType : public DataType {
 public:
  explicit CategoryType(const std::shared_ptr<Array>& categories)
      : DataType(TypeId::CATEGORY), categories_(categories) {}

  std::string ToString() const override;
  std::shared_ptr<const DataType> category_type() const;
  std::shared_ptr<Array> categories() const { return categories_; }

 protected:
  std::shared_ptr<Array> categories_;
};

inline bool is_integer(TypeId type_id) {
  return type_id >= TypeId::INT8 && type_id <= TypeId::UINT64;
}

inline bool is_signed_integer(TypeId type_id) {
  return type_id >= TypeId::INT8 && type_id <= TypeId::INT64;
}

inline bool is_unsigned_integer(TypeId type_id) {
  return type_id >= TypeId::UINT8 && type_id <= TypeId::UINT64;
}

}  // namespace pandas
