// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "cuda/gemm_tiled.cuh"

#include <stdexcept>

#include "cuda/cuda_check.cuh"

namespace accel {

__global__ void GemmTiledKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int m, int n, int k) {
  __shared__ float a_tile[kGemmTileDim][kGemmTileDim];
  __shared__ float b_tile[kGemmTileDim][kGemmTileDim];

  const int row = blockIdx.y * kGemmTileDim + threadIdx.y;
  const int col = blockIdx.x * kGemmTileDim + threadIdx.x;

  float sum = 0.0f;
  const int tile_count = (k + kGemmTileDim - 1) / kGemmTileDim;

  for (int t = 0; t < tile_count; ++t) {
    const int a_col = t * kGemmTileDim + threadIdx.x;
    const int b_row = t * kGemmTileDim + threadIdx.y;

    // Threads of a warp share threadIdx.y and vary threadIdx.x, so both loads
    // are coalesced: consecutive lanes touch consecutive addresses. Offsets are
    // widened to size_t because m*k overflows int well inside the benchmark
    // range (4096 * 2048 already exceeds 8 million, and the product of the
    // largest supported extents does not fit).
    const bool a_in_range = (row < m) && (a_col < k);
    const bool b_in_range = (b_row < k) && (col < n);
    a_tile[threadIdx.y][threadIdx.x] =
        a_in_range ? a[static_cast<size_t>(row) * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        b_in_range ? b[static_cast<size_t>(b_row) * n + col] : 0.0f;

    // Barrier 1: the tiles are not readable until every thread has stored.
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kGemmTileDim; ++i) {
      sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
    }

    // Barrier 2: no thread may overwrite the tiles for step t+1 while a slower
    // warp is still consuming step t. Dropping this is a classic silent
    // shared-memory race — one `compute-sanitizer --tool racecheck` away.
    __syncthreads();
  }

  if (row < m && col < n) {
    c[static_cast<size_t>(row) * n + col] = sum;
  }
}

void LaunchGemmTiled(const float* a, const float* b, float* c,
                     const GemmShape& shape, cudaStream_t stream) {
  if (!shape.is_valid()) {
    throw std::invalid_argument("LaunchGemmTiled requires positive extents");
  }

  const dim3 block(kGemmTileDim, kGemmTileDim);
  const dim3 grid((shape.n + kGemmTileDim - 1) / kGemmTileDim,
                  (shape.m + kGemmTileDim - 1) / kGemmTileDim);

  GemmTiledKernel<<<grid, block, 0, stream>>>(a, b, c, shape.m, shape.n,
                                              shape.k);
  // Surfaces a rejected launch configuration immediately; execution faults
  // still require a later synchronisation to observe.
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace accel
