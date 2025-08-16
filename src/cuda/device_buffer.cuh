// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CUDA_DEVICE_BUFFER_CUH_
#define ACCEL_CUDA_DEVICE_BUFFER_CUH_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/cuda_check.cuh"

namespace accel {

// Move-only RAII owner of a cudaMalloc allocation: exactly one owner, released
// on scope exit however that exit is reached. Every early return between
// cudaMalloc and cudaFree is otherwise a leak the compiler cannot warn about.
//
// Copying is deleted; an implicit one would double-free. Transfer with
// std::move. Not internally synchronised -- concurrent CopyFromHost calls race
// exactly as raw cudaMemcpy would.
//
// T must be trivially copyable: the transfer helpers are byte copies and run
// no constructors on either side.
template <typename T>
class DeviceBuffer {
 public:
  // Constructs an empty buffer that owns nothing.
  DeviceBuffer() noexcept = default;

  // Allocates room for `count` elements.
  //
  // Throws CudaError if `cudaMalloc` fails.
  //
  //   count  Element count; Zero yields an empty buffer without calling into
  //          the driver.
  explicit DeviceBuffer(std::size_t count) {
    if (count == 0) {
      return;
    }
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
    Data_ = static_cast<T*>(raw);
    size_ = count;
  }

  // Swallows any `cudaFree` error: destructors must not throw, and a
  //       free failing during context teardown is not actionable.
  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : Data_(std::exchange(other.Data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      Data_ = std::exchange(other.Data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  T* get() noexcept { return Data_; }
  const T* get() const noexcept { return Data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t SizeBytes() const noexcept { return size_ * sizeof(T); }
  bool empty() const noexcept { return size_ == 0; }
  explicit operator bool() const noexcept { return Data_ != nullptr; }

  // Releases the allocation and returns the buffer to the empty state.
  void Reset() noexcept {
    if (Data_ != nullptr) {
      static_cast<void>(cudaFree(Data_));
      Data_ = nullptr;
      size_ = 0;
    }
  }

  // Uploads `count` elements from host memory.
  //
  // Stream-ordered and asynchronous. A subsequent kernel launched on the same
  // stream observes the data, but work on any *other* stream does not until
  // the caller synchronises.
  //
  // Safe to let `host_src` die on return even so: for pageable source memory
  // the driver stages through its own pinned buffer before the call returns.
  //
  // Throws CudaError on transfer failure, or std::out_of_range if `count`
  // exceeds the allocation — catching the overflow before the driver does
  // turns a silent heap corruption into a diagnosable error.
  void CopyFromHost(const T* host_src, std::size_t count,
                    cudaStream_t stream = nullptr) {
    RequireCapacity(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpyAsync(Data_, host_src, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  // Convenience overload uploading an entire host vector.
  void CopyFromHost(const std::vector<T>& host_src,
                    cudaStream_t stream = nullptr) {
    CopyFromHost(host_src.data(), host_src.size(), stream);
  }

  // Downloads `count` elements into host memory.
  //
  // Stream-ordered and asynchronous: `host_dst` is **not** readable when this
  // returns. Synchronise the stream first, or use ToHost(), which does it for
  // you. This is the one asymmetry with CopyFromHost() — there the driver's
  // staging buffer hides the asynchrony, here nothing does.
  //
  // Throws CudaError on transfer failure, std::out_of_range on overrun.
  void CopyToHost(T* host_dst, std::size_t count,
                  cudaStream_t stream = nullptr) const {
    RequireCapacity(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpyAsync(host_dst, Data_, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  // Downloads the whole buffer and synchronises, so the result is readable on
  // return. Prefer this over CopyToHost() unless the extra sync matters.
  std::vector<T> ToHost(cudaStream_t stream = nullptr) const {
    std::vector<T> host(size_);
    CopyToHost(host.data(), size_, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
  }

  // Zero-fills the entire allocation.
  // Throws CudaError if `cudaMemsetAsync` fails.
  void Zero(cudaStream_t stream = nullptr) {
    if (size_ == 0) {
      return;
    }
    CUDA_CHECK(cudaMemsetAsync(Data_, 0, SizeBytes(), stream));
  }

 private:
  void RequireCapacity(std::size_t count) const {
    if (count > size_) {
      throw std::out_of_range(
          "DeviceBuffer transfer of " + std::to_string(count) +
          " elements exceeds capacity " + std::to_string(size_));
    }
  }

  T* Data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace accel

#endif  // ACCEL_CUDA_DEVICE_BUFFER_CUH_
