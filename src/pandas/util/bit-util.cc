// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#include "pandas/util/bit-util.h"

#include <cstring>
#include <vector>

#include "pandas/buffer.h"
#include "pandas/status.h"

namespace pandas {
namespace BitUtil {

void BytesToBits(const std::vector<uint8_t>& bytes, uint8_t* bits) {
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (bytes[i] > 0) { SetBit(bits, i); }
  }
}

Status BytesToBits(
    const std::vector<uint8_t>& bytes, std::shared_ptr<Buffer>* out) {
  int bit_length = BytesForBits(bytes.size());

  auto buffer = std::make_shared<PoolBuffer>();
  RETURN_NOT_OK(buffer->Resize(bit_length));
  memset(buffer->mutable_data(), 0, bit_length);
  BytesToBits(bytes, buffer->mutable_data());

  *out = buffer;
  return Status::OK();
}

}  // namespace BitUtil
}  // namespace pandas
