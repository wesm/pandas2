// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <type_traits>

#include "pandas/array_fwd.h"

namespace pandas {

template <typename T>
struct TypeTraits {};

template <>
struct TypeTraits<UInt8Type> {
  using ArrayType = UInt8Array;
  static inline int bytes_required(int elements) { return elements; }
};

template <>
struct TypeTraits<Int8Type> {
  using ArrayType = Int8Array;
  static inline int bytes_required(int elements) { return elements; }
};

template <>
struct TypeTraits<UInt16Type> {
  using ArrayType = UInt16Array;
  static inline int bytes_required(int elements) { return elements * sizeof(uint16_t); }
};

template <>
struct TypeTraits<Int16Type> {
  using ArrayType = Int16Array;
  static inline int bytes_required(int elements) { return elements * sizeof(int16_t); }
};

template <>
struct TypeTraits<UInt32Type> {
  using ArrayType = UInt32Array;
  static inline int bytes_required(int elements) { return elements * sizeof(uint32_t); }
};

template <>
struct TypeTraits<Int32Type> {
  using ArrayType = Int32Array;
  static inline int bytes_required(int elements) { return elements * sizeof(int32_t); }
};

template <>
struct TypeTraits<UInt64Type> {
  using ArrayType = UInt64Array;
  static inline int bytes_required(int elements) { return elements * sizeof(uint64_t); }
};

template <>
struct TypeTraits<Int64Type> {
  using ArrayType = Int64Array;
  static inline int bytes_required(int elements) { return elements * sizeof(int64_t); }
};

template <>
struct TypeTraits<FloatType> {
  using ArrayType = FloatArray;
  static inline int bytes_required(int elements) { return elements * sizeof(float); }
};

template <>
struct TypeTraits<DoubleType> {
  using ArrayType = DoubleArray;
  static inline int bytes_required(int elements) { return elements * sizeof(double); }
};

template <>
struct TypeTraits<BooleanType> {
  using ArrayType = BooleanArray;
  static inline int bytes_required(int elements) { return elements; }
};

// template <>
// struct TypeTraits<StringType> {
//   using ArrayType = StringArray;
// };

// template <>
// struct TypeTraits<BinaryType> {
//   using ArrayType = BinaryArray;
// };

// Not all type classes have a c_type
template <typename T>
struct as_void {
  using type = void;
};

// The partial specialization will match if T has the ATTR_NAME member
#define GET_ATTR(ATTR_NAME, DEFAULT)                                             \
  template <typename T, typename Enable = void>                                  \
  struct GetAttr_##ATTR_NAME {                                                   \
    using type = DEFAULT;                                                        \
  };                                                                             \
                                                                                 \
  template <typename T>                                                          \
  struct GetAttr_##ATTR_NAME<T, typename as_void<typename T::ATTR_NAME>::type> { \
    using type = typename T::ATTR_NAME;                                          \
  };

GET_ATTR(c_type, void);
GET_ATTR(TypeClass, void);

#undef GET_ATTR

#define PRIMITIVE_TRAITS(T)                                                           \
  using TypeClass = typename std::conditional<std::is_base_of<DataType, T>::value, T, \
      typename GetAttr_TypeClass<T>::type>::type;                                     \
  using c_type = typename GetAttr_c_type<TypeClass>::type;

template <typename T>
struct IsUnsignedInt {
  PRIMITIVE_TRAITS(T);
  static constexpr bool value =
      std::is_integral<c_type>::value && std::is_unsigned<c_type>::value;
};

template <typename T>
struct IsSignedInt {
  PRIMITIVE_TRAITS(T);
  static constexpr bool value =
      std::is_integral<c_type>::value && std::is_signed<c_type>::value;
};

template <typename T>
struct IsInteger {
  PRIMITIVE_TRAITS(T);
  static constexpr bool value = std::is_integral<c_type>::value;
};

template <typename T>
struct IsFloatingPoint {
  PRIMITIVE_TRAITS(T);
  static constexpr bool value = std::is_floating_point<c_type>::value;
};

template <typename T>
struct IsNumeric {
  PRIMITIVE_TRAITS(T);
  static constexpr bool value = std::is_arithmetic<c_type>::value;
};

}  // namespace pandas
