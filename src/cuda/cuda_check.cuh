// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CUDA_CUDA_CHECK_CUH_
#define ACCEL_CUDA_CUDA_CHECK_CUH_

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace accel {

// Exception carrying a failed CUDA runtime status.
//
// Carrying the failure as an exception rather than the usual `printf` plus
// `exit(1)` is what makes the RAII owners in this project work: an `exit` from
// inside a check macro bypasses every destructor on the stack and leaks the
// device allocations they hold. Throwing lets them release first.
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const std::string& what)
      : std::runtime_error(what), code_(code) {}

  // The originating `cudaError_t`, for callers that branch on
  // `cudaErrorMemoryAllocation` and retry with a smaller working set.
  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

namespace detail {

inline std::string FormatCudaError(cudaError_t code, const char* expression,
                                   const char* file, int line) {
  return std::string(file) + ':' + std::to_string(line) + ": " + expression +
         " failed with " + cudaGetErrorName(code) + " (" +
         cudaGetErrorString(code) + ')';
}

// Out-of-line-ish helper so the macro expands to a single expression and stays
// usable in any statement position.
inline void CheckCudaStatus(cudaError_t code, const char* expression,
                            const char* file, int line) {
  if (code != cudaSuccess) {
    throw CudaError(code, FormatCudaError(code, expression, file, line));
  }
}

}  // namespace detail
}  // namespace accel

// Evaluates a CUDA runtime call and throws accel::CudaError on failure.
//
// Never use inside a destructor or a `noexcept` function.
#define CUDA_CHECK(expression)                                          \
  ::accel::detail::CheckCudaStatus((expression), #expression, __FILE__, \
                                   __LINE__)

// Validates a kernel launch and waits for it to retire.
//
// Two distinct failures need catching after a launch: `cudaGetLastError()`
// reports configuration errors raised synchronously at launch time, while
// `cudaDeviceSynchronize()` surfaces faults the kernel hit while executing.
// Checking only the former lets an out-of-bounds store go unreported until
// some unrelated later call inherits the sticky error.
//
// Synchronising is a benchmarking and debugging tool; do not place this inside
// a timed loop that is meant to overlap work.
#define CUDA_CHECK_LAUNCH()              \
  do {                                   \
    CUDA_CHECK(cudaGetLastError());      \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

#endif  // ACCEL_CUDA_CUDA_CHECK_CUH_
