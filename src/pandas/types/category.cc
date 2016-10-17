// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/category.h"

namespace pandas {

std::string CategoryType::ToString() const {
  std::stringstream s;
  s << "category<" << category_type()->ToString() << ">";
  return s.str();
}

} // namespace pandas
