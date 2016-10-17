// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include "pandas/buffer.h"

#include <cstdint>
#include <limits>

#include "pandas/memory.h"
#include "pandas/status.h"
#include "pandas/util/bit-util.h"
#include "pandas/util/logging.h"

namespace pandas {

Buffer::Buffer(const std::shared_ptr<Buffer>& parent, int64_t offset, int64_t size) {
  data_ = parent->data() + offset;
  size_ = size;
  parent_ = parent;
  capacity_ = size;
}

Buffer::~Buffer() {}

Status Buffer::Copy(int64_t start, int64_t nbytes, std::shared_ptr<Buffer>* out) const {
  // Sanity checks
  PANDAS_DCHECK_LT(start, size_);
  PANDAS_DCHECK_LE(nbytes, size_ - start);

  auto new_buffer = std::make_shared<PoolBuffer>();
  RETURN_NOT_OK(new_buffer->Resize(nbytes));

  std::memcpy(new_buffer->mutable_data(), data() + start, nbytes);

  *out = new_buffer;
  return Status::OK();
}

std::shared_ptr<Buffer> SliceBuffer(const std::shared_ptr<Buffer>& buffer, int64_t offset,
    int64_t length) {
  PANDAS_DCHECK_LT(offset, buffer->size());
  PANDAS_DCHECK_LE(length, buffer->size() - offset);
  return std::make_shared<Buffer>(buffer, offset, length);
}

// ----------------------------------------------------------------------
// Buffer that allocated memory from the central memory pool

PoolBuffer::PoolBuffer(MemoryPool* pool) : ResizableBuffer(nullptr, 0) {
  if (pool == nullptr) { pool = default_memory_pool(); }
  pool_ = pool;
}

PoolBuffer::~PoolBuffer() {
  if (mutable_data_ != nullptr) { pool_->Free(mutable_data_, capacity_); }
}

Status PoolBuffer::Reserve(int64_t new_capacity) {
  if (!mutable_data_ || new_capacity > capacity_) {
    uint8_t* new_data;
    new_capacity = BitUtil::RoundUpToMultipleOf64(new_capacity);
    if (mutable_data_) {
      RETURN_NOT_OK(pool_->Allocate(new_capacity, &new_data));
      memcpy(new_data, mutable_data_, size_);
      pool_->Free(mutable_data_, capacity_);
    } else {
      RETURN_NOT_OK(pool_->Allocate(new_capacity, &new_data));
    }
    mutable_data_ = new_data;
    data_ = mutable_data_;
    capacity_ = new_capacity;
  }
  return Status::OK();
}

Status PoolBuffer::Resize(int64_t new_size) {
  RETURN_NOT_OK(Reserve(new_size));
  size_ = new_size;
  return Status::OK();
}

}  // namespace pandas
