// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "cuda/sum_reduction.cuh"

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/device_buffer.cuh"

namespace accel {
namespace {

constexpr int kMaxThreadsPerBlock = 1024;

constexpr bool IsPowerOfTwo(int value) noexcept {
  return value > 0 && (value & (value - 1)) == 0;
}

// Single source of truth for the launch geometry contract, so that
// SumOnDevice() rejects a bad grid size *before* sizing an allocation from
// it. Validating only at launch time meant a negative grid_size first became
// a huge size_t and surfaced as a confusing cudaErrorMemoryAllocation.
void ValidateReductionGeometry(int grid_size, int block_size) {
  if (grid_size <= 0) {
    throw std::invalid_argument("reduction grid size must be positive, got " +
                                std::to_string(grid_size));
  }
  if (!IsPowerOfTwo(block_size) || block_size > kMaxThreadsPerBlock) {
    throw std::invalid_argument(
        "reduction block size must be a power of two in (0, 1024], got " +
        std::to_string(block_size));
  }
}

}  // namespace

__global__ void BlockSumReduceKernel(const float* __restrict__ input,
                                     float* __restrict__ partials,
                                     std::size_t count) {
  extern __shared__ float scratch[];

  const unsigned int tid = threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  // Grid-stride accumulation. Consecutive threads read consecutive addresses
  // on every pass, so each pass is fully coalesced regardless of how many
  // passes the input requires.
  float thread_sum = 0.0f;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
       i < count; i += stride) {
    thread_sum += input[i];
  }
  scratch[tid] = thread_sum;
  __syncthreads();

  // Shared-memory tree reduction. Correct only for power-of-two blockDim.x,
  // which LaunchBlockSumReduce() enforces. The barrier sits outside the
  // `if` because __syncthreads() must be reached by every thread in the block;
  // placing it inside is undefined behaviour, not merely slow.
  for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (tid < step) {
      scratch[tid] += scratch[tid + step];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = scratch[0];
  }
}

void LaunchBlockSumReduce(const float* input, float* partials,
                          std::size_t count, int grid_size, int block_size,
                          cudaStream_t stream) {
  ValidateReductionGeometry(grid_size, block_size);

  const unsigned int shared_bytes =
      static_cast<unsigned int>(block_size) * sizeof(float);
  BlockSumReduceKernel<<<grid_size, block_size, shared_bytes, stream>>>(
      input, partials, count);
  CUDA_CHECK(cudaGetLastError());
}

double SumOnDevice(const float* host_input, std::size_t count, int grid_size,
                   int block_size) {
  // Checked ahead of the early return as well as ahead of the allocation:
  // an invalid geometry is a caller bug whether or not the input is empty, and
  // reporting it only for non-empty inputs would make it easy to miss.
  ValidateReductionGeometry(grid_size, block_size);
  if (count == 0) {
    return 0.0;
  }

  // Both buffers release themselves on any throw below, including one raised
  // from inside LaunchBlockSumReduce().
  DeviceBuffer<float> device_input(count);
  DeviceBuffer<float> device_partials(static_cast<std::size_t>(grid_size));

  device_input.CopyFromHost(host_input, count);
  LaunchBlockSumReduce(device_input.get(), device_partials.get(), count,
                       grid_size, block_size);
  CUDA_CHECK_LAUNCH();

  const std::vector<float> partials = device_partials.ToHost();

  // Widen to double for the host tail: the partials are the largest values in
  // the whole reduction, and adding them in float would discard the precision
  // the tree reduction just preserved.
  double total = 0.0;
  for (const float value : partials) {
    total += static_cast<double>(value);
  }
  return total;
}

}  // namespace accel
