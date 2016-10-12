// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include "pandas/memory.h"

#include <cstdlib>
#include <mutex>
#include <sstream>

#include "pandas/util/logging.h"
#include "pandas/status.h"

namespace pandas {

namespace {
// Allocate memory according to the alignment requirements for Pandas
// (as of May 2016 64 bytes)
Status AllocateAligned(int64_t size, uint8_t** out) {
  // TODO(emkornfield) find something compatible with windows
  constexpr size_t kAlignment = 64;
  const int result = posix_memalign(reinterpret_cast<void**>(out), kAlignment, size);
  if (result == ENOMEM) {
    std::stringstream ss;
    ss << "malloc of size " << size << " failed";
    return Status::OutOfMemory(ss.str());
  }

  if (result == EINVAL) {
    std::stringstream ss;
    ss << "invalid alignment parameter: " << kAlignment;
    return Status::Invalid(ss.str());
  }
  return Status::OK();
}
}  // namespace

MemoryPool::~MemoryPool() {}

class InternalMemoryPool : public MemoryPool {
 public:
  InternalMemoryPool() : bytes_allocated_(0) {}
  virtual ~InternalMemoryPool();

  Status Allocate(int64_t size, uint8_t** out) override;

  void Free(uint8_t* buffer, int64_t size) override;

  int64_t bytes_allocated() const override;

 private:
  mutable std::mutex pool_lock_;
  int64_t bytes_allocated_;
};

Status InternalMemoryPool::Allocate(int64_t size, uint8_t** out) {
  std::lock_guard<std::mutex> guard(pool_lock_);
  RETURN_NOT_OK(AllocateAligned(size, out));
  bytes_allocated_ += size;

  return Status::OK();
}

int64_t InternalMemoryPool::bytes_allocated() const {
  std::lock_guard<std::mutex> guard(pool_lock_);
  return bytes_allocated_;
}

void InternalMemoryPool::Free(uint8_t* buffer, int64_t size) {
  std::lock_guard<std::mutex> guard(pool_lock_);
  PANDAS_DCHECK_GE(bytes_allocated_, size);
  std::free(buffer);
  bytes_allocated_ -= size;
}

InternalMemoryPool::~InternalMemoryPool() {}

MemoryPool* default_memory_pool() {
  static InternalMemoryPool default_memory_pool_;
  return &default_memory_pool_;
}

}  // namespace pandas
