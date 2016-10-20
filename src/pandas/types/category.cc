// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/category.h"

namespace pandas {

CategoryArray::CategoryArray(const std::shared_ptr<CategoryType>& type,
    const std::shared_ptr<Array>& codes)
    : Array(codes->length(), 0), codes_(codes), type_(type) {}

std::string CategoryType::ToString() const {
  std::stringstream s;
  s << "category<" << category_type()->ToString() << ">";
  return s.str();
}

}  // namespace pandas
