// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Test non-type specific array functionality

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/memory.h"
#include "pandas/meta/typelist.h"
#include "pandas/test-util.h"
#include "pandas/type.h"

using std::string;

namespace pandas {

class TestArray : public ::testing::Test {
 public:
  void SetUp() {
    values_ = {0, 1, 2, 3, 4, 5, 6, 7};

    auto buffer =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(values_.data()),
            values_.size() * sizeof(double));

    array_ = std::make_shared<DoubleArray>(values_.size(), buffer);
  }

 protected:
  std::shared_ptr<Array> array_;
  std::vector<double> values_;
};

TEST_F(TestArray, Attrs) {
  DoubleType ex_type;
  ASSERT_TRUE(array_->type()->Equals(ex_type));
  ASSERT_EQ(TypeId::FLOAT64, array_->type_id());

  ASSERT_EQ(values_.size(), array_->length());
}

// ----------------------------------------------------------------------
// Array view object

class TestArrayView : public ::testing::Test {
 public:
  using value_t = double;

  void SetUp() {
    values_ = {0, 1, 2, 3, 4, 5, 6, 7};

    auto buffer =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(values_.data()),
            values_.size() * sizeof(value_t));

    auto arr = std::make_shared<DoubleArray>(values_.size(), buffer);
    view_ = ArrayView(arr);
  }

 protected:
  ArrayView view_;
  std::vector<value_t> values_;
};

TEST_F(TestArrayView, Ctors) {
  ASSERT_EQ(1, view_.ref_count());
  ASSERT_EQ(0, view_.offset());
  ASSERT_EQ(values_.size(), view_.length());

  // Copy ctor
  ArrayView view2(view_);
  ASSERT_EQ(2, view2.ref_count());
  ASSERT_EQ(0, view_.offset());
  ASSERT_EQ(values_.size(), view_.length());

  // move ctor
  ArrayView view3(view_.data(), 3);
  ArrayView view4(std::move(view3));
  ASSERT_EQ(3, view4.ref_count());
  ASSERT_EQ(3, view3.offset());
  ASSERT_EQ(values_.size() - 3, view3.length());

  // With offset and length
  ArrayView view5(view4.data(), 2, 4);
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());

  // Copy assignment
  ArrayView view6 = view5;
  ASSERT_EQ(5, view4.ref_count());
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());

  // Move assignment
  ArrayView view7 = std::move(view6);
  ASSERT_EQ(5, view4.ref_count());
  ASSERT_EQ(2, view5.offset());
  ASSERT_EQ(4, view5.length());
}

TEST_F(TestArrayView, EnsureMutable) {
  // This only tests for one data type -- we will need to test more rigorously
  // across all data types elsewhere

  const Array* ap = view_.data().get();

  ASSERT_NO_THROW(view_.EnsureMutable());
  ASSERT_EQ(ap, view_.data().get());

  ArrayView view2 = view_;

  ASSERT_NO_THROW(view_.EnsureMutable());

  // The views now have their own distinct copies of the array
  ASSERT_NE(ap, view_.data().get());
  ASSERT_EQ(ap, view2.data().get());

  ASSERT_EQ(1, view_.ref_count());
  ASSERT_EQ(1, view2.ref_count());
}

TEST_F(TestArrayView, Slice) {
  ArrayView s1 = view_.Slice(3);
  ASSERT_EQ(2, s1.ref_count());
  ASSERT_EQ(3, s1.offset());
  ASSERT_EQ(view_.length() - 3, s1.length());

  ArrayView s2 = view_.Slice(2, 4);
  ASSERT_EQ(3, s2.ref_count());
  ASSERT_EQ(2, s2.offset());
  ASSERT_EQ(4, s2.length());

  // Slice of a slice
  ArrayView s3 = s1.Slice(2);
  ASSERT_EQ(4, s3.ref_count());
  ASSERT_EQ(5, s3.offset());
  ASSERT_EQ(view_.length() - 5, s3.length());

  ArrayView s4 = s1.Slice(1, 2);
  ASSERT_EQ(5, s4.ref_count());
  ASSERT_EQ(4, s4.offset());
  ASSERT_EQ(2, s4.length());
}

}  // namespace pandas
