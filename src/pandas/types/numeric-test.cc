// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Test non-type specific array functionality

#include <cstdint>

#include "gtest/gtest.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/test-util.h"
#include "pandas/type.h"
#include "pandas/types/numeric.h"

namespace pandas {

template <typename TYPE>
void CheckType() {
  using T = typename TYPE::TypeClass::c_type;

  const T v1 = 10;

  TYPE value1(10, true);
  ASSERT_TRUE(value1.is_null());

  TYPE value2(10, false);
  ASSERT_FALSE(value2.is_null());
  ASSERT_EQ(v1, value2.value());
}

TEST(TestNumericScalar, Integers) {
  CheckType<UInt8Scalar>();
  CheckType<UInt16Scalar>();
  CheckType<UInt32Scalar>();
  CheckType<UInt64Scalar>();
  CheckType<Int8Scalar>();
  CheckType<Int16Scalar>();
  CheckType<Int32Scalar>();
  CheckType<Int64Scalar>();
}

}  // namespace pandas
