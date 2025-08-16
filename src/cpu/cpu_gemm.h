// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CPU_CPU_GEMM_H_
#define ACCEL_CPU_CPU_GEMM_H_

#include "core/gemm_shape.h"
#include "core/span.h"

namespace accel {

// 64 floats per tile edge is a 16 KiB block, so a blocked update keeps a
// ~48 KiB working set live. Whether that fits is microarchitecture-dependent:
// blocking buys 2.6x on an Apple M5 and 1.4x on a 7950X. Tune with --tile=N.
inline constexpr int kDefaultTileSize = 64;

// All five compute the accumulating update C += A * B (BLAS beta = 1), so a
// caller wanting a plain product must zero C first. Index letters give the
// loop nesting order: i walks M, j walks N, k the contraction.
//
// Extents are validated once before the loop nest, which is what lets the
// inner loops index unchecked; a violation throws std::invalid_argument.
// Requires shape.is_valid(), spans sized to match, and c not aliasing a or b.
//
// Single-threaded by design: these isolate memory access order. Parallelism is
// a separate axis, in omp/gemm_omp.h.

// Baseline. The inner loop strides B by n floats, so all but one value of each
// fetched cache line is evicted before use.
void GemmIjk(Span<const float> a, Span<const float> b, Span<float> c,
             const GemmShape& shape);

// Hoisting the contraction out of the N loop makes the inner statement a
// scalar-times-row AXPY over unit-stride B and C, which auto-vectorises. C is
// then read and written every iteration rather than held in a register -- a
// trade that pays for itself many times over.
void GemmIkj(Span<const float> a, Span<const float> b, Span<float> c,
             const GemmShape& shape);

// Cache-blocked. Tail blocks are clamped, so extents need not divide
// tile_size; a non-positive tile_size throws std::invalid_argument.
void GemmTiled(Span<const float> a, Span<const float> b, Span<float> c,
               const GemmShape& shape, int tile_size = kDefaultTileSize);

// Expects B transposed, b_col[j * k + l] == B(l, j), via RowToColumnMajor().
// Both inner-loop operands are then unit-stride: a pair of dot products.
void GemmIjkBColMajor(Span<const float> a, Span<const float> b_col,
                      Span<float> c, const GemmShape& shape);

// Blocking plus the transposed layout. The two target the same bottleneck
// instead of compounding: faster than the unblocked column-major kernel on an
// M5, slower on a 7950X, and beaten by GemmIkj everywhere. See
// docs/Benchmarks.md.
void GemmTiledBColMajor(Span<const float> a, Span<const float> b_col,
                        Span<float> c, const GemmShape& shape,
                        int tile_size = kDefaultTileSize);

}  // namespace accel

#endif  // ACCEL_CPU_CPU_GEMM_H_
