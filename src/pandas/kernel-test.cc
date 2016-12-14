// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include <gtest/gtest.h>

#include "pandas/array.h"
#include "pandas/common.h"
#include "pandas/kernel.h"
#include "pandas/test-util.h"
#include "pandas/type_traits.h"
#include "pandas/util/logging.h"

namespace pandas {

template <typename ArrayType0>
std::enable_if_t<IsNumeric<ArrayType0>::value, void>
AddOne(const ArrayType0& input, ArrayType0* output) {
  PANDAS_DCHECK_EQ(input.length(), output->length());
  auto input_data = input.data();
  auto output_data = output->mutable_data();
  for (int64_t i = 0; i < input.length(); ++i) {
    output_data[i] = input_data[i] + 1;
  }
  // Propagate null bitmap
}

// This will be code generated in general
class AddOneKernel_Double : public ElwiseUnaryKernel {
 public:
  void Eval(const ValueList& inputs, OutputSink* out) const override {
    PANDAS_DCHECK_EQ(inputs[0]->kind(), ValueKind::ARRAY) << __FILE__ << __LINE__;
    PANDAS_DCHECK_EQ(inputs[0]->type()->type(), TypeId::FLOAT64) << __FILE__ << __LINE__;
    PANDAS_DCHECK_EQ(inputs[0]->type()->type(), TypeId::FLOAT64) << __FILE__ << __LINE__;
    const auto& in1 = static_cast<const DoubleArray&>(*inputs[0].get());

    PANDAS_DCHECK_EQ(out->kind(), OutputSink::PRIMITIVE);
    auto prim_sink = static_cast<PrimitiveOutputSink*>(out);
    auto out2 = static_cast<DoubleArray*>(prim_sink->out());
    AddOne<DoubleArray>(in1, out2);
  }
};

// std::shared_ptr<Kernel> AddOne(const ValueExpr& in1) {
//   switch (in1->type_id()) {
//     case TypeId::FLOAT64:
//       return std::make_shared<AddOneKernel_Double>();
//     default:
//       throw NotImplementedError(in1->type()->ToString());
//   }
// }

TEST(AddOneKernel, Properties) {
  AddOneKernel_Double kernel;

  ASSERT_TRUE(kernel.is_splittable());
  ASSERT_EQ(Kernel::MONADIC, kernel.arity());
}

}  // namespace pandas
