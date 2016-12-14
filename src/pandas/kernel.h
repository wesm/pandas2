// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "pandas/array.h"
#include "pandas/visibility.h"

namespace pandas {

using ValueList = std::vector<std::shared_ptr<Value>>;

class OutputSink {
 public:
  enum Kind {
    PRIMITIVE,
    VARBYTES,
    LIST
  };

  OutputSink(Kind kind) : kind_(kind) {}

  Kind kind() const { return kind_; }

 protected:
  Kind kind_;

 private:
  OutputSink() {}
};

// An output sink for writing into pre-allocated array memory
class PrimitiveOutputSink : public OutputSink {
 public:
  PrimitiveOutputSink(Array* out)
      : OutputSink(OutputSink::PRIMITIVE),
        out_(out) {}

  Array* out() const { return out_; }

 protected:
  Array* out_;
};

// An output sink for writing binary or string-like data with as yet unknown
// size
class VarbytesOutputSink : public OutputSink {};

// An output sink for list values (similar to varbytes)
class ListOutputSink : public OutputSink {};

class Kernel {
 public:
  enum Arity {
    NONE,     // No value arguments
    MONADIC,  // 1 value argument
    DYADIC,   // 2 value arguments
    TRYADIC,  // 3 value arguments
    NARY      // N value arguments
  };

  Kernel(Arity arity) : arity_(arity) {}

  Arity arity() const { return arity_; }

  virtual void Eval(const ValueList& inputs, OutputSink* out) const = 0;

 protected:
  Arity arity_;
};

class ArrayKernel : public Kernel {
 public:
  ArrayKernel(Arity arity, bool is_splittable)
      : Kernel(arity), is_splittable_(is_splittable) {}

  bool is_splittable() const { return is_splittable_; }

 private:
  bool is_splittable_;
};

class UnaryArrayKernel : public ArrayKernel {
 public:
  UnaryArrayKernel(bool is_splittable)
      : ArrayKernel(Kernel::MONADIC, is_splittable) {}
};

class ElwiseUnaryKernel : public UnaryArrayKernel {
 public:
  ElwiseUnaryKernel() : UnaryArrayKernel(true) {}
};

}  // namespace pandas
