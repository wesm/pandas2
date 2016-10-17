// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#include "pandas/status.h"
#include "pandas/util/macros.h"
#include "pandas/visibility.h"

namespace pandas {

class MemoryPool;
class Status;

// ----------------------------------------------------------------------
// Buffer classes

// Immutable API for a chunk of bytes which may or may not be owned by the
// class instance.  Buffers have two related notions of length: size and
// capacity.  Size is the number of bytes that might have valid data.
// Capacity is the number of bytes that where allocated for the buffer in
// total.
// The following invariant is always true: Size < Capacity
class PANDAS_EXPORT Buffer {
 public:
  Buffer(uint8_t* data, int64_t size)
      : is_mutable_(true),
        mutable_data_(data),
        data_(data),
        size_(size),
        capacity_(size) {}

  Buffer(const uint8_t* data, int64_t size)
      : is_mutable_(false),
        mutable_data_(nullptr),
        data_(data),
        size_(size),
        capacity_(size) {}

  virtual ~Buffer();

  bool is_mutable() const { return is_mutable_; }

  // An offset into data that is owned by another buffer, but we want to be
  // able to retain a valid pointer to it even after other shared_ptr's to the
  // parent buffer have been destroyed
  //
  // This method makes no assertions about alignment or padding of the buffer but
  // in general we expected buffers to be aligned and padded to 64 bytes.  In the future
  // we might add utility methods to help determine if a buffer satisfies this contract.
  Buffer(const std::shared_ptr<Buffer>& parent, int64_t offset, int64_t size);

  // Return true if both buffers are the same size and contain the same bytes
  // up to the number of compared bytes
  bool Equals(const Buffer& other, int64_t nbytes) const {
    return this == &other ||
           (size_ >= nbytes && other.size_ >= nbytes &&
               (data_ == other.data_ || !memcmp(data_, other.data_, nbytes)));
  }

  bool Equals(const Buffer& other) const {
    return this == &other ||
           (size_ == other.size_ &&
               (data_ == other.data_ || !memcmp(data_, other.data_, size_)));
  }

  int64_t capacity() const { return capacity_; }

  uint8_t* mutable_data() { return mutable_data_; }
  const uint8_t* data() const { return data_; }

  int64_t size() const { return size_; }

  // Returns true if this Buffer is referencing memory (possibly) owned by some
  // other buffer
  bool is_shared() const { return static_cast<bool>(parent_); }

  std::shared_ptr<Buffer> parent() const { return parent_; }

  // Copy the indicated byte range into a newly-created buffer
  Status Copy(int64_t start, int64_t nbytes, std::shared_ptr<Buffer>* out) const;

 protected:
  bool is_mutable_;
  uint8_t* mutable_data_;
  const uint8_t* data_;
  int64_t size_;
  int64_t capacity_;

  // nullptr by default, but may be set
  std::shared_ptr<Buffer> parent_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Buffer);
};

// Construct a view on passed buffer at the indicated offset and length. This
// function cannot fail and does not error checking (except in debug builds)
std::shared_ptr<Buffer> SliceBuffer(
    const std::shared_ptr<Buffer>& buffer, int64_t offset, int64_t length);

class PANDAS_EXPORT ResizableBuffer : public Buffer {
 public:
  // Change buffer reported size to indicated size, allocating memory if
  // necessary.  This will ensure that the capacity of the buffer is a multiple
  // of 64 bytes as defined in Layout.md.
  virtual Status Resize(int64_t new_size) = 0;

  // Ensure that buffer has enough memory allocated to fit the indicated
  // capacity (and meets the 64 byte padding requirement in Layout.md).
  // It does not change buffer's reported size.
  virtual Status Reserve(int64_t new_capacity) = 0;

 protected:
  ResizableBuffer(uint8_t* data, int64_t size) : Buffer(data, size) {}
};

// A Buffer whose lifetime is tied to a particular MemoryPool
class PANDAS_EXPORT PoolBuffer : public ResizableBuffer {
 public:
  explicit PoolBuffer(MemoryPool* pool = nullptr);
  virtual ~PoolBuffer();

  Status Resize(int64_t new_size) override;
  Status Reserve(int64_t new_capacity) override;

 private:
  MemoryPool* pool_;
};

class BufferBuilder {
 public:
  explicit BufferBuilder(MemoryPool* pool)
      : pool_(pool), data_(nullptr), capacity_(0), size_(0) {}

  // Resizes the buffer to the nearest multiple of 64 bytes per Layout.md
  Status Resize(int32_t elements) {
    if (capacity_ == 0) { buffer_ = std::make_shared<PoolBuffer>(pool_); }
    RETURN_NOT_OK(buffer_->Resize(elements));
    capacity_ = buffer_->capacity();
    data_ = buffer_->mutable_data();
    return Status::OK();
  }

  Status Append(const uint8_t* data, int length) {
    if (capacity_ < length + size_) { RETURN_NOT_OK(Resize(length + size_)); }
    UnsafeAppend(data, length);
    return Status::OK();
  }

  template <typename T>
  Status Append(T arithmetic_value) {
    static_assert(std::is_arithmetic<T>::value,
        "Convenience buffer append only supports arithmetic types");
    return Append(reinterpret_cast<uint8_t*>(&arithmetic_value), sizeof(T));
  }

  template <typename T>
  Status Append(const T* arithmetic_values, int num_elements) {
    static_assert(std::is_arithmetic<T>::value,
        "Convenience buffer append only supports arithmetic types");
    return Append(
        reinterpret_cast<const uint8_t*>(arithmetic_values), num_elements * sizeof(T));
  }

  // Unsafe methods don't check existing size
  void UnsafeAppend(const uint8_t* data, int length) {
    memcpy(data_ + size_, data, length);
    size_ += length;
  }

  template <typename T>
  void UnsafeAppend(T arithmetic_value) {
    static_assert(std::is_arithmetic<T>::value,
        "Convenience buffer append only supports arithmetic types");
    UnsafeAppend(reinterpret_cast<uint8_t*>(&arithmetic_value), sizeof(T));
  }

  template <typename T>
  void UnsafeAppend(const T* arithmetic_values, int num_elements) {
    static_assert(std::is_arithmetic<T>::value,
        "Convenience buffer append only supports arithmetic types");
    UnsafeAppend(
        reinterpret_cast<const uint8_t*>(arithmetic_values), num_elements * sizeof(T));
  }

  std::shared_ptr<Buffer> Finish() {
    auto result = buffer_;
    buffer_ = nullptr;
    capacity_ = size_ = 0;
    return result;
  }
  int capacity() { return capacity_; }
  int length() { return size_; }

 private:
  std::shared_ptr<PoolBuffer> buffer_;
  MemoryPool* pool_;
  uint8_t* data_;
  int64_t capacity_;
  int64_t size_;
};

}  // namespace pandas
