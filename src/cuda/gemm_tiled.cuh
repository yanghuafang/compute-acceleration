// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CUDA_GEMM_TILED_CUH_
#define ACCEL_CUDA_GEMM_TILED_CUH_

#include <cuda_runtime.h>

#include "core/gemm_shape.h"

namespace accel {

// A 32x32 block is exactly one warp per row, so the cooperative loads are
// warp-aligned, and each staged tile costs 4 KiB of shared memory. Changing
// this means re-checking both properties.
inline constexpr int kGemmTileDim = 32;

// Shared-memory tiled GEMM, C = A * B, row-major throughout and overwriting C
// rather than accumulating.
//
// Each block computes one tile of C by marching along K, cooperatively staging
// one tile of A and one of B per step. Every staged element is then read
// kGemmTileDim times by different threads, which is the whole
// arithmetic-intensity gain over direct global loads.
//
// Out-of-range loads yield zero rather than being skipped, keeping the inner
// product fixed-trip and uniform across the block: a divergent tail costs more
// than the wasted multiply-adds.
//
// Must be launched with blockDim == (kGemmTileDim, kGemmTileDim) -- shared
// memory is indexed by threadIdx, so any other shape reads out of bounds.
// Extents need not be multiples of the tile. `c` must not alias `a` or `b`.
__global__ void GemmTiledKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int m, int n, int k);

// Validates `shape` and launches on `stream`. Asynchronous: check completion
// with CUDA_CHECK_LAUNCH or by synchronising, which the benchmark driver
// defers until after its timing loop.
//
// Throws std::invalid_argument if the shape is degenerate, CudaError if the
// launch is rejected.
void LaunchGemmTiled(const float* a, const float* b, float* c,
                     const GemmShape& shape, cudaStream_t stream = nullptr);

}  // namespace accel

#endif  // ACCEL_CUDA_GEMM_TILED_CUH_
