// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

namespace pandas {

// Forward declarations for numeric types
// and numeric arrays

class FloatType;
class DoubleType;
class Int8Type;
class UInt8Type;
class Int16Type;
class UInt16Type;
class Int32Type;
class UInt32Type;
class Int64Type;
class UInt64Type;

template <typename TYPE>
class IntegerArray;

template <typename TYPE>
class FloatingArray;

using FloatArray = FloatingArray<FloatType>;
using DoubleArray = FloatingArray<DoubleType>;

using Int8Array = IntegerArray<Int8Type>;
using UInt8Array = IntegerArray<UInt8Type>;

using Int16Array = IntegerArray<Int16Type>;
using UInt16Array = IntegerArray<UInt16Type>;

using Int32Array = IntegerArray<Int32Type>;
using UInt32Array = IntegerArray<UInt32Type>;

using Int64Array = IntegerArray<Int64Type>;
using UInt64Array = IntegerArray<UInt64Type>;

class Scalar;

template <typename TYPE>
class NumericScalar;

using FloatScalar = NumericScalar<FloatType>;
using DoubleScalar = NumericScalar<DoubleType>;
using Int8Scalar = NumericScalar<Int8Type>;
using UInt8Scalar = NumericScalar<UInt8Type>;
using Int16Scalar = NumericScalar<Int16Type>;
using UInt16Scalar = NumericScalar<UInt16Type>;
using Int32Scalar = NumericScalar<Int32Type>;
using UInt32Scalar = NumericScalar<UInt32Type>;
using Int64Scalar = NumericScalar<Int64Type>;
using UInt64Scalar = NumericScalar<UInt64Type>;
}
