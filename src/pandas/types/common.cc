// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/types/common.h"

#include <cstdint>
#include <cstring>

#include "pandas/common.h"
#include "pandas/memory.h"

namespace pandas {

Status CopyBitmap(const std::shared_ptr<Buffer>& bitmap, int64_t bit_offset,
    int64_t length, std::shared_ptr<Buffer>* out) {
  // TODO(wesm): Optimize this bitmap copy for each bit_offset mod 8
  int64_t nbytes = BitUtil::BytesForBits(length);
  auto buf = std::make_shared<PoolBuffer>(memory_pool());
  RETURN_NOT_OK(buf->Resize(nbytes));

  // Set to all 1s, since all valid
  memset(buf->mutable_data(), 0xFF, nbytes);

  *out = buf;
  return Status::OK();
}

Status AllocateValidityBitmap(int64_t length, std::shared_ptr<Buffer>* out) {
  int64_t nbytes = BitUtil::BytesForBits(length);
  auto buf = std::make_shared<PoolBuffer>(memory_pool());
  RETURN_NOT_OK(buf->Resize(nbytes));

  // Set to all 1s, since all valid
  memset(buf->mutable_data(), 0xFF, nbytes);

  *out = buf;
  return Status::OK();
}

}  // namespace pandas
