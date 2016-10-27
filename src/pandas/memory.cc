// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include "pandas/memory.h"

#include <cstdlib>
#include <mutex>
#include <sstream>

#include "pandas/common.h"
#include "pandas/util/logging.h"

namespace pandas {

namespace {

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

Status PandasMemoryPool::Allocate(int64_t size, uint8_t** out) {
  std::lock_guard<std::mutex> guard(pool_lock_);
  RETURN_NOT_OK(AllocateAligned(size, out));
  bytes_allocated_ += size;

  return Status::OK();
}

int64_t PandasMemoryPool::bytes_allocated() const {
  std::lock_guard<std::mutex> guard(pool_lock_);
  return bytes_allocated_;
}

void PandasMemoryPool::Free(uint8_t* buffer, int64_t size) {
  std::lock_guard<std::mutex> guard(pool_lock_);
  PANDAS_DCHECK_GE(bytes_allocated_, size);
  std::free(buffer);
  bytes_allocated_ -= size;
}

PandasMemoryPool::~PandasMemoryPool() {}

MemoryPool* memory_pool() {
  static PandasMemoryPool memory_pool_;
  return &memory_pool_;
}

}  // namespace pandas
