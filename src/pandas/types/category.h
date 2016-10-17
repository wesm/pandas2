// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/config.h"

#include <sstream>
#include <string>

#include "pandas/array.h"
#include "pandas/type.h"

#include "pandas/status.h"

namespace pandas {

struct CategoryType : public DataType {
  explicit CategoryType(const ArrayView& categories)
      : DataType(DataType::CATEGORY), categories_(categories) {}

  std::string ToString() const override;

  std::shared_ptr<DataType> category_type() const { return categories_.data()->type(); }

  const ArrayView& categories() const { return categories_; }

 protected:
  ArrayView categories_;
};

class CategoryArray : public Array {
 public:
  const ArrayView& codes() const { return codes_; }

  const ArrayView& categories() const {
    return static_cast<CategoryType*>(type_.get())->categories();
  }

 private:
  ArrayView codes_;
};

}  // namespace pandas
