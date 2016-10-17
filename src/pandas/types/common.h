// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

#include <cstdint>
#include <memory>

namespace pandas {

class Status;
class Buffer;

Status CopyBitmap(const std::shared_ptr<Buffer>& bitmap, int64_t bit_offset,
    int64_t length, std::shared_ptr<Buffer>* out);

Status AllocateValidityBitmap(int64_t length, std::shared_ptr<Buffer>* out);

}  // namespace pandas
