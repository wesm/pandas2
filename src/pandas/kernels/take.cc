// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/kernels/take.h"

#include "pandas/type.h"

namespace pandas {

std::shared_ptr<DataType> TakeOperation::GetReturnType() const {
  auto arg0 = static_cast<const ValueExpr*>(args_[0].get());
  return arg0->type();
}

}  // namespace pandas
