// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/expr.h"

namespace pandas {

Expr::Expr(Kind kind, const std::shared_ptr<Operation>& operation)
    : kind_(kind), operation_(operation) {}

ScalarExpr::ScalarExpr(const std::shared_ptr<Operation>& operation)
    : ValueExpr(Kind::SCALAR, operation) {}

ArrayExpr::ArrayExpr(const std::shared_ptr<Operation>& operation)
    : ValueExpr(Kind::ARRAY, operation) {}

}  // namespace pandas
