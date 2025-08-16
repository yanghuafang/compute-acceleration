// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CUDA_SUM_REDUCTION_CUH_
#define ACCEL_CUDA_SUM_REDUCTION_CUH_

#include <cstddef>

#include <cuda_runtime.h>

namespace accel {

// Must stay a power of two: the halving loop in BlockSumReduceKernel drops the
// odd element at every other level, silently under-counting.
inline constexpr int kReductionBlockSize = 512;

// Just how many partial sums the host is asked to finish -- the grid-stride
// load decouples this from the input length.
inline constexpr int kReductionGridSize = 8;

// Reduces `input` to one partial sum per block: a grid-stride accumulation
// folding an arbitrary length into one value per thread, then a shared-memory
// tree. The grid-stride loop is what frees grid size from input size; the
// obvious alternative, blocks * threads == count, reads out of bounds for
// every other pairing.
//
// The host finishes the handful of partials. A second launch would cost more
// than the host add, and keeping the last step on the CPU makes the numerics
// inspectable.
//
// Addition is not associative, so the result depends on the launch geometry;
// compare against a host sum by relative tolerance, never equality.
//
// Requires blockDim.x a power of two and blockDim.x * sizeof(float) bytes of
// dynamic shared memory. `partials` holds at least gridDim.x floats; `count`
// may be zero.
__global__ void BlockSumReduceKernel(const float* __restrict__ input,
                                     float* __restrict__ partials,
                                     std::size_t count);

// Validates geometry and enqueues BlockSumReduceKernel. Asynchronous; the
// caller synchronises. Shared-memory size is derived from block_size here so
// call sites cannot desynchronise the two.
//
// block_size must be a power of two in (0, 1024]. Throws
// std::invalid_argument on bad geometry, CudaError if the launch is rejected.
void LaunchBlockSumReduce(const float* input, float* partials,
                          std::size_t count, int grid_size = kReductionGridSize,
                          int block_size = kReductionBlockSize,
                          cudaStream_t stream = nullptr);

// Allocates, uploads, reduces, downloads and finishes on the host. For tests;
// the benchmark driver keeps the stages separate so it can time the kernel
// alone. Accumulates the partials in double so the tail does not lose what the
// kernel preserved.
//
// Throws std::invalid_argument on bad geometry, CudaError on runtime failure.
// Device allocations are released either way.
double SumOnDevice(const float* host_input, std::size_t count,
                   int grid_size = kReductionGridSize,
                   int block_size = kReductionBlockSize);

}  // namespace accel

#endif  // ACCEL_CUDA_SUM_REDUCTION_CUH_
