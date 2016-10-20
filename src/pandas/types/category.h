// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <sstream>
#include <string>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

struct CategoryType : public DataType {
  explicit CategoryType(const std::shared_ptr<Array>& categories)
      : DataType(DataType::CATEGORY), categories_(categories) {}

  std::string ToString() const override;

  std::shared_ptr<const DataType> category_type() const {
    return categories_->type();
  }

  std::shared_ptr<Array> categories() const { return categories_; }

 protected:
  std::shared_ptr<Array> categories_;
};

class CategoryArray : public Array {
 public:
  CategoryArray(const std::shared_ptr<CategoryType>& type, const std::shared_ptr<Array>& codes);

  std::shared_ptr<Array> codes() const { return codes_; }
  std::shared_ptr<Array> categories() const { return type_->categories(); }

 private:
  std::shared_ptr<Array> codes_;
  std::shared_ptr<CategoryType> type_;
};

}  // namespace pandas
