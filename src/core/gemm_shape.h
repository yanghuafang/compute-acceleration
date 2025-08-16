// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CORE_GEMM_SHAPE_H_
#define ACCEL_CORE_GEMM_SHAPE_H_

#include <cstddef>

#include "core/span.h"

namespace accel {

// Extents of the product `C(M x N) = A(M x K) * B(K x N)`.
//
// `k` is the contraction extent — the dimension summed over — and is the axis
// every kernel in this project reorders, blocks, or stages through shared
// memory. Element counts are returned as `std::size_t` because `m * k` for the
// default 4096x2048 benchmark shape already exceeds 8 million; computing them
// in `int` is a signed-overflow trap on larger inputs.
struct GemmShape {
  int m = 0;  // Rows of A and of C.
  int k = 0;  // Columns of A and rows of B; the contraction extent.
  int n = 0;  // Columns of B and of C.

  // All extents strictly positive. Empty matrices are rejected rather than
  // silently treated as no-ops, since they almost always signal a caller bug.
  constexpr bool is_valid() const noexcept { return m > 0 && k > 0 && n > 0; }

  constexpr std::size_t a_elements() const noexcept {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
  }
  constexpr std::size_t b_elements() const noexcept {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  }
  constexpr std::size_t c_elements() const noexcept {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  }

  // Multiply-accumulate count, the denominator for GFLOP/s (2 flops per MAC).
  constexpr std::size_t mac_count() const noexcept {
    return c_elements() * static_cast<std::size_t>(k);
  }
};

// Validates operand extents before any kernel touches memory.
//
// Both storage orders of B occupy `k * n` elements, so a single check covers
// row-major and column-major operands alike.
//
// Throws std::invalid_argument if `shape` is degenerate or any span's length
// disagrees with it. Throwing here — once, outside the loop nest — is what
// lets the kernels index without per-element bounds checks.
void ValidateGemmOperands(Span<const float> a, Span<const float> b,
                          Span<const float> c, const GemmShape& shape);

// Achieved throughput in GFLOP/s, counting one multiply plus one add per MAC.
// Returns 0.0 when `seconds` is not strictly positive (timer underflow).
double GemmGflops(const GemmShape& shape, double seconds) noexcept;

}  // namespace accel

#endif  // ACCEL_CORE_GEMM_SHAPE_H_
