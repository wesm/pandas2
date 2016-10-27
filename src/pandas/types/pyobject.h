// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/type.h"

namespace pandas {

class PANDAS_EXPORT PyObjectArray : public Array {
 public:
  using T = PyObject*;
  PyObjectArray(int64_t length, const std::shared_ptr<Buffer>& data,
      const std::shared_ptr<Buffer>& valid_bits = nullptr);

  Status Copy(int64_t offset, int64_t length, std::shared_ptr<Array>* out) const override;

  // Returns a NEW reference
  PyObject* GetItem(int64_t i) override;

  // Does not steal a reference
  Status SetItem(int64_t i, PyObject* val) override;

  int64_t GetNullCount() override;

  bool owns_data() const override;

  PyObject** data() const;
  PyObject** mutable_data() const;

 protected:
  std::shared_ptr<Buffer> data_;
  std::shared_ptr<Buffer> valid_bits_;
};

}  // namespace pandas
