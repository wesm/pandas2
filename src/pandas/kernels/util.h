// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

namespace pandas {

template <typename VALUE_TYPE, typename ENABLED = void>
struct ArrayData {
  typename VALUE_TYPE::c_type* data;
  uint8_t* valid;
  int64_t length;
  int64_t offset;
};

// template <typename VALUE_TYPE>
// struct ArrayData<VALUE_TYPE,
//                  typename std::enable_if<VALUE_TYPE::is_primitive, VALUE_TYPE>::type> {
//   typename VALUE_TYPE::c_type* data;
//   uint8_t* valid;
//   int64_t length;
//   int64_t offset;
// };

}  // namespace pandas
