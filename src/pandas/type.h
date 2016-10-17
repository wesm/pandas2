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
    FLOAT = 10,

    // 8-byte floating point value
    DOUBLE = 11,

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

  explicit DataType(TypeId type)
      : type_(type) {}

  virtual ~DataType() {}

  virtual std::string ToString() const = 0;

  virtual bool Equals(const DataType& other) {
    return type_ == other.type_;
  }

  TypeId type() const { return type_; }

 private:
  TypeId type_;
};


typedef std::shared_ptr<DataType> TypePtr;


class PANDAS_EXPORT TimestampType : public DataType {
 public:
  enum class Unit: char {
    SECOND = 0,
    MILLISECOND = 1,
    MICROSECOND = 2,
    NANOSECOND = 3
  };

  Unit unit;

  explicit TimestampType(Unit unit = Unit::MICROSECOND)
      : DataType(DataType::TIMESTAMP),
        unit(unit) {}

  TimestampType(const TimestampType& other)
      : TimestampType(other.unit) {}

  static char const *name() {
    return "timestamp";
  }

  std::string ToString() const override;
};


class PANDAS_EXPORT PyObjectType : public DataType {
 public:

  PyObjectType() : DataType(DataType::PYOBJECT) {}

  PyObjectType(const PyObjectType& other)
      : PyObjectType() {}

  static char const *name() {
    return "object";
  }

  std::string ToString() const override;
};


template <typename Derived>
class PANDAS_EXPORT PrimitiveType : public DataType {
 public:
  PrimitiveType()
      : DataType(Derived::type_enum) {}

  std::string ToString() const override {
    return std::string(static_cast<const Derived*>(this)->name());
  }
};


#define PRIMITIVE_DECL(TYPENAME, C_TYPE, ENUM, SIZE, NAME)          \
  public:                                                           \
   typedef C_TYPE c_type;                                           \
   static constexpr DataType::TypeId type_enum = DataType::ENUM;    \
   static constexpr size_t size = SIZE;                             \
                                                                    \
   explicit TYPENAME()                                              \
       : PrimitiveType<TYPENAME>() {}                               \
                                                                    \
   static const char* name() {                                      \
     return NAME;                                                   \
   }


class PANDAS_EXPORT NullType : public PrimitiveType<NullType> {
  PRIMITIVE_DECL(NullType, void, NA, 0, "null");
};

class PANDAS_EXPORT UInt8Type : public PrimitiveType<UInt8Type> {
  PRIMITIVE_DECL(UInt8Type, uint8_t, UINT8, 1, "uint8");
};

class PANDAS_EXPORT Int8Type : public PrimitiveType<Int8Type> {
  PRIMITIVE_DECL(Int8Type, int8_t, INT8, 1, "int8");
};

class PANDAS_EXPORT UInt16Type : public PrimitiveType<UInt16Type> {
  PRIMITIVE_DECL(UInt16Type, uint16_t, UINT16, 2, "uint16");
};

class PANDAS_EXPORT Int16Type : public PrimitiveType<Int16Type> {
  PRIMITIVE_DECL(Int16Type, int16_t, INT16, 2, "int16");
};

class PANDAS_EXPORT UInt32Type : public PrimitiveType<UInt32Type> {
  PRIMITIVE_DECL(UInt32Type, uint32_t, UINT32, 4, "uint32");
};

class PANDAS_EXPORT Int32Type : public PrimitiveType<Int32Type> {
  PRIMITIVE_DECL(Int32Type, int32_t, INT32, 4, "int32");
};

class PANDAS_EXPORT UInt64Type : public PrimitiveType<UInt64Type> {
  PRIMITIVE_DECL(UInt64Type, uint64_t, UINT64, 8, "uint64");
};

class PANDAS_EXPORT Int64Type : public PrimitiveType<Int64Type> {
  PRIMITIVE_DECL(Int64Type, int64_t, INT64, 8, "int64");
};

class PANDAS_EXPORT FloatType : public PrimitiveType<FloatType> {
  PRIMITIVE_DECL(FloatType, float, FLOAT, 4, "float");
};

class PANDAS_EXPORT DoubleType : public PrimitiveType<DoubleType> {
  PRIMITIVE_DECL(DoubleType, double, DOUBLE, 8, "double");
};

class PANDAS_EXPORT BooleanType : public PrimitiveType<BooleanType> {
  PRIMITIVE_DECL(BooleanType, uint8_t, BOOL, 1, "bool");
};

} // namespace pandas
