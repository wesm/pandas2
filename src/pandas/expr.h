// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "pandas/visibility.h"

namespace pandas {

class Array;
class DataType;
class Operation;
class Scalar;

// An Expr / expression represents a typed output of an operation
class PANDAS_EXPORT Expr {
 public:
  enum class Kind : char { SCALAR = 0, ARRAY = 1, TABLE = 2 };

  std::shared_ptr<Operation> operation() const;

 protected:
  Expr(Kind kind, const std::shared_ptr<Operation>& operation);

  Kind kind_;
  std::shared_ptr<Operation> operation_;
};

class PANDAS_EXPORT ValueExpr : public Expr {
 public:
  using Expr::Expr;
  std::shared_ptr<DataType> type() const { return type_; }

 protected:
  std::shared_ptr<DataType> type_;
};

class PANDAS_EXPORT ArrayExpr : public ValueExpr {
 public:
  explicit ArrayExpr(const std::shared_ptr<Operation>& operation);
};

class PANDAS_EXPORT ScalarExpr : public ValueExpr {
 public:
  explicit ScalarExpr(const std::shared_ptr<Operation>& operation);
};

class PANDAS_EXPORT TableExpr : public Expr {};

// Operation represents a computational operation that yields a typed expression
class PANDAS_EXPORT Operation {
 public:
  virtual std::shared_ptr<Expr> ToExpr() const = 0;
  virtual void Print(std::ostream* out) const = 0;
  virtual std::shared_ptr<DataType> GetReturnType() const = 0;

 protected:
  explicit Operation(const std::shared_ptr<Expr>& arg);
  explicit Operation(const std::vector<std::shared_ptr<Expr>>& args);
  std::vector<std::shared_ptr<Expr>> args_;
};

class PANDAS_EXPORT ArrayIdentity : public Operation {
 public:
  explicit ArrayIdentity(const std::shared_ptr<Array>& value);

  std::shared_ptr<Expr> ToExpr() const override;
  std::shared_ptr<Array> value() const;

 protected:
  std::shared_ptr<Array> value_;
};

class PANDAS_EXPORT ScalarIdentity : public Operation {
 public:
  explicit ScalarIdentity(const std::shared_ptr<Scalar>& value);
  std::shared_ptr<Expr> ToExpr() const override;
  std::shared_ptr<Scalar> value() const;

 protected:
  std::shared_ptr<Scalar> value_;
};

}  // namespace pandas
