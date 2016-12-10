// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <sstream>
#include <string>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

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
