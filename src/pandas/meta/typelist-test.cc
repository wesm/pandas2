// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include <cstdint>

#include "gtest/gtest.h"

#include "pandas/meta/typelist.h"

namespace pandas {

class TestTypeListFunctions : public ::testing::Test {
 public:
  void SetUp() {
    saw_int32_ = saw_float_ = false;
    int32_index_ = float_index_ = -1;
    count_ = 0;
  }

  template <typename T>
  void operator()() {
    if (std::is_same<T, std::int32_t>::value) {
      saw_int32_ = true;
      int32_index_ = count_++;
    } else if (std::is_same<T, float>::value) {
      saw_float_ = true;
      float_index_ = count_++;
    }
  }

 protected:
  bool saw_int32_;
  bool saw_float_;
  int64_t int32_index_;
  int64_t float_index_;
  int64_t count_;
};

TEST_F(TestTypeListFunctions, Iterate) {
  TypeList<int32_t, float>().Iterate(*this);
  ASSERT_TRUE(saw_int32_);
  ASSERT_TRUE(saw_float_);
  ASSERT_EQ(int32_index_, 0);
  ASSERT_EQ(float_index_, 1);
}

TEST_F(TestTypeListFunctions, ReverseIterate) {
  TypeList<int32_t, float>().ReverseIterate(*this);
  ASSERT_TRUE(saw_int32_);
  ASSERT_TRUE(saw_float_);
  ASSERT_EQ(int32_index_, 1);
  ASSERT_EQ(float_index_, 0);
}

TEST_F(TestTypeListFunctions, Addition) {
  using AddedList = decltype(TypeList<int32_t>() + TypeList<float>());
  AddedList().Iterate(*this);
  ASSERT_TRUE(saw_int32_);
  ASSERT_TRUE(saw_float_);
  ASSERT_EQ(int32_index_, 0);
  ASSERT_EQ(float_index_, 1);
}

TEST(TestTypeList, Length) {
  static constexpr auto length2 = TypeList<int32_t, float>::length;
  static constexpr auto length1 = TypeList<int32_t>::length;
  ASSERT_EQ(length2, 2);
  ASSERT_EQ(length1, 1);
}

TEST(TestTypeList, CurrentAndNext) {
  static constexpr auto current1 =
      std::is_same<typename TypeList<int32_t, float>::current, int32_t>::value;
  static constexpr auto current2 =
      std::is_same<typename TypeList<int32_t, float>::next::current, float>::value;
  ASSERT_TRUE(current1);
  ASSERT_TRUE(current2);
}

TEST(TestTypeList, Last) {
  static constexpr auto last1 = TypeList<int32_t, float>::last;
  static constexpr auto last2 = TypeList<int32_t, float>::next::last;
  ASSERT_EQ(last1, false);
  ASSERT_EQ(last2, true);
}

TEST(TestTypeList, IndexOf) {
  static constexpr auto index1 = TypeList<int32_t, float>().IndexOf<int32_t>();
  static constexpr auto index2 = TypeList<int32_t, float>().IndexOf<float>();
  ASSERT_EQ(index1, 0);
  ASSERT_EQ(index2, 1);
}

TEST(TestTypeList, At) {
  using List = TypeList<int32_t, float>;
  using at1type = typename List::At<0>::type;
  using at2type = typename List::At<1>::type;
  std::cout << typeid(at2type).name() << std::endl;
  static constexpr auto at1correct = std::is_same<at1type, int32_t>::value;
  static constexpr auto at2correct = std::is_same<at2type, float>::value;
  ASSERT_TRUE(at1correct);
  ASSERT_TRUE(at2correct);
}

class TestCartesianProduct : public ::testing::Test {
 public:
  void SetUp() {
    saw_i_f_ = saw_i_i_ = saw_f_i_ = saw_f_f_ = false;
    i_f_index_ = i_i_index_ = f_i_index_ = f_f_index_ = -1;
    count_ = 0;
  }

  template <typename T>
  void operator()() {
    if (std::is_same<T, std::tuple<int32_t, float>>::value) {
      saw_i_f_ = true;
      i_f_index_ = count_++;
    } else if (std::is_same<T, std::tuple<int32_t, int32_t>>::value) {
      saw_i_i_ = true;
      i_i_index_ = count_++;
    } else if (std::is_same<T, std::tuple<float, int32_t>>::value) {
      saw_f_i_ = true;
      f_i_index_ = count_++;
    } else if (std::is_same<T, std::tuple<float, float>>::value) {
      saw_f_f_ = true;
      f_f_index_ = count_++;
    }
  }

 protected:
  bool saw_i_f_;
  bool saw_i_i_;
  bool saw_f_i_;
  bool saw_f_f_;

  int64_t i_f_index_;
  int64_t i_i_index_;
  int64_t f_i_index_;
  int64_t f_f_index_;

  int64_t count_;
};

TEST_F(TestCartesianProduct, SingleProductSingle) {
  using List = TypeList<int32_t>;
  static constexpr auto product = List().CartesianProduct(List());
  product.Iterate(*this);
  ASSERT_TRUE(saw_i_i_);
  ASSERT_EQ(i_i_index_, 0);
}

TEST_F(TestCartesianProduct, SingleProductDouble) {
  static constexpr auto product =
      TypeList<int32_t, float>().CartesianProduct(TypeList<float>());
  product.Iterate(*this);
  ASSERT_TRUE(saw_i_f_);
  ASSERT_TRUE(saw_f_f_);
  ASSERT_FALSE(saw_i_i_);
  ASSERT_FALSE(saw_f_i_);

  ASSERT_EQ(i_f_index_, 0);
  ASSERT_EQ(f_f_index_, 1);
  ASSERT_EQ(i_i_index_, -1);
  ASSERT_EQ(f_i_index_, -1);
}

TEST_F(TestCartesianProduct, DoubleProductDouble) {
  using List = TypeList<int32_t, float>;
  static constexpr auto product = List().CartesianProduct(List());
  product.Iterate(*this);
  ASSERT_TRUE(saw_i_f_);
  ASSERT_TRUE(saw_f_f_);
  ASSERT_TRUE(saw_i_i_);
  ASSERT_TRUE(saw_f_i_);

  ASSERT_EQ(i_i_index_, 0);
  ASSERT_EQ(i_f_index_, 1);
  ASSERT_EQ(f_i_index_, 2);
  ASSERT_EQ(f_f_index_, 3);
}
}
