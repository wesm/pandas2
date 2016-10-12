// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "pandas/test-util.h"
#include "pandas/buffer.h"
#include "pandas/memory.h"
#include "pandas/status.h"

using std::string;

namespace pandas {

class TestBuffer : public ::testing::Test {};

TEST_F(TestBuffer, Resize) {
  PoolBuffer buf;

  ASSERT_EQ(0, buf.size());
  ASSERT_OK(buf.Resize(100));
  ASSERT_EQ(100, buf.size());
  ASSERT_OK(buf.Resize(200));
  ASSERT_EQ(200, buf.size());

  // Make it smaller, too
  ASSERT_OK(buf.Resize(50));
  ASSERT_EQ(50, buf.size());
}

TEST_F(TestBuffer, ResizeOOM) {
  // realloc fails, even though there may be no explicit limit
  PoolBuffer buf;
  ASSERT_OK(buf.Resize(100));
  int64_t to_alloc = std::numeric_limits<int64_t>::max();
  ASSERT_RAISES(OutOfMemory, buf.Resize(to_alloc));
}

TEST_F(TestBuffer, EqualsWithSameContent) {
  MemoryPool* pool = default_memory_pool();
  const int32_t bufferSize = 128 * 1024;
  uint8_t* rawBuffer1;
  ASSERT_OK(pool->Allocate(bufferSize, &rawBuffer1));
  memset(rawBuffer1, 12, bufferSize);
  uint8_t* rawBuffer2;
  ASSERT_OK(pool->Allocate(bufferSize, &rawBuffer2));
  memset(rawBuffer2, 12, bufferSize);
  uint8_t* rawBuffer3;
  ASSERT_OK(pool->Allocate(bufferSize, &rawBuffer3));
  memset(rawBuffer3, 3, bufferSize);

  Buffer buffer1(rawBuffer1, bufferSize);
  Buffer buffer2(rawBuffer2, bufferSize);
  Buffer buffer3(rawBuffer3, bufferSize);
  ASSERT_TRUE(buffer1.Equals(buffer2));
  ASSERT_FALSE(buffer1.Equals(buffer3));

  pool->Free(rawBuffer1, bufferSize);
  pool->Free(rawBuffer2, bufferSize);
  pool->Free(rawBuffer3, bufferSize);
}

TEST_F(TestBuffer, EqualsWithSameBuffer) {
  MemoryPool* pool = default_memory_pool();
  const int32_t bufferSize = 128 * 1024;
  uint8_t* rawBuffer;
  ASSERT_OK(pool->Allocate(bufferSize, &rawBuffer));
  memset(rawBuffer, 111, bufferSize);

  Buffer buffer1(rawBuffer, bufferSize);
  Buffer buffer2(rawBuffer, bufferSize);
  ASSERT_TRUE(buffer1.Equals(buffer2));

  const int64_t nbytes = bufferSize / 2;
  Buffer buffer3(rawBuffer, nbytes);
  ASSERT_TRUE(buffer1.Equals(buffer3, nbytes));
  ASSERT_FALSE(buffer1.Equals(buffer3, nbytes + 1));

  pool->Free(rawBuffer, bufferSize);
}

}  // namespace pandas
