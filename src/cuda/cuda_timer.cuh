// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CUDA_CUDA_TIMER_CUH_
#define ACCEL_CUDA_CUDA_TIMER_CUH_

#include <utility>

#include <cuda_runtime.h>

#include "cuda/cuda_check.cuh"

namespace accel {

// Move-only RAII owner of a cudaEvent_t. Events leak like device memory, and a
// timed region holds two -- created together, easily destroyed neither.
class CudaEvent {
 public:
  // Throws CudaError if the event cannot be created.
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&Event_)); }

  ~CudaEvent() {
    if (Event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(Event_));
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept
      : Event_(std::exchange(other.Event_, nullptr)) {}

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (Event_ != nullptr) {
        static_cast<void>(cudaEventDestroy(Event_));
      }
      Event_ = std::exchange(other.Event_, nullptr);
    }
    return *this;
  }

  cudaEvent_t get() const noexcept { return Event_; }

  // Enqueues this event into `stream`.
  // Throws CudaError if the record fails.
  void Record(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(Event_, stream));
  }

  // Blocks the calling thread until the event has been reached.
  // Throws CudaError if the wait fails.
  void Synchronize() { CUDA_CHECK(cudaEventSynchronize(Event_)); }

 private:
  cudaEvent_t Event_ = nullptr;
};

// Elapsed device time between two stream-ordered events. Timestamps are taken
// by the GPU on the recording stream, so this measures the work rather than
// launch latency or host scheduling jitter; resolution is about 0.5 us.
//
// Bracket a repetition loop and divide, to amortise the two records over all
// iterations. ElapsedMillis must follow a Start/Stop pair on the same stream.
class CudaTimer {
 public:
  void Start(cudaStream_t stream = nullptr) { start_.Record(stream); }

  void Stop(cudaStream_t stream = nullptr) { stop_.Record(stream); }

  // Synchronises on the stop event; returns milliseconds. Throws CudaError if
  // that event never completes, which is how an async kernel fault surfaces.
  float ElapsedMillis() {
    stop_.Synchronize();
    float millis = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&millis, start_.get(), stop_.get()));
    return millis;
  }

 private:
  CudaEvent start_;
  CudaEvent stop_;
};

}  // namespace accel

#endif  // ACCEL_CUDA_CUDA_TIMER_CUH_
