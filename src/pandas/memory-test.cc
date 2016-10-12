// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

#include "pandas/test-util.h"
#include "pandas/memory.h"
#include "pandas/status.h"

namespace pandas {

TEST(DefaultMemoryPool, MemoryTracking) {
  MemoryPool* pool = default_memory_pool();

  uint8_t* data;
  ASSERT_OK(pool->Allocate(100, &data));
  EXPECT_EQ(static_cast<uint64_t>(0), reinterpret_cast<uint64_t>(data) % 64);
  ASSERT_EQ(100, pool->bytes_allocated());

  pool->Free(data, 100);
  ASSERT_EQ(0, pool->bytes_allocated());
}

TEST(DefaultMemoryPool, OOM) {
  MemoryPool* pool = default_memory_pool();

  uint8_t* data;
  int64_t to_alloc = std::numeric_limits<int64_t>::max();
  ASSERT_RAISES(OutOfMemory, pool->Allocate(to_alloc, &data));
}

}  // namespace pandas
