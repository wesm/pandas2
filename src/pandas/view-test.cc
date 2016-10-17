// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Test the mechanics of table and array views

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "pandas/array.h"
#include "pandas/buffer.h"
#include "pandas/memory.h"
#include "pandas/status.h"
#include "pandas/test-util.h"

using std::string;

namespace pandas {

class TestArrayViews : public ::testing::Test {};

}  // namespace pandas
