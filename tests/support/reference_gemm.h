// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_TESTS_SUPPORT_REFERENCE_GEMM_H_
#define ACCEL_TESTS_SUPPORT_REFERENCE_GEMM_H_

#include <cstddef>
#include <vector>

#include "core/gemm_shape.h"

namespace accel::test {

// Independent double-precision GEMM used as ground truth.
//
// Kept intentionally naive and separate from every optimised path, so a bug in
// the shared blocking or indexing logic cannot hide by being present in both
// the kernel and its reference. Accumulating in `double` puts the reference's
// rounding error far below the `float` tolerance the tests apply.
//
// Returns c in row-major order, `m * n` elements.
//
// Precondition: Extents are valid and the operands match them; the caller
// controls both.
inline std::vector<float> ReferenceGemm(const std::vector<float>& a,
                                        const std::vector<float>& b,
                                        const GemmShape& shape) {
  const auto m = static_cast<std::size_t>(shape.m);
  const auto k = static_cast<std::size_t>(shape.k);
  const auto n = static_cast<std::size_t>(shape.n);

  std::vector<float> c(shape.c_elements(), 0.0f);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t l = 0; l < k; ++l) {
        sum += static_cast<double>(a[i * k + l]) *
               static_cast<double>(b[l * n + j]);
      }
      c[i * n + j] = static_cast<float>(sum);
    }
  }
  return c;
}

// Deterministic pseudo-random operand fill in [-1, 1).
//
// A Ramp makes every row of A a scalar multiple of every other, which masks
// transposition and row/column mix-ups. Mixed signs and magnitudes do not. The
// generator is a fixed LCG rather than `std::mt19937` so results are identical
// across standard-library implementations.
inline std::vector<float> PseudoRandomVector(std::size_t count,
                                             unsigned int seed) {
  std::vector<float> values(count);
  unsigned int state = seed | 1u;
  for (std::size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>(state >> 8) /
                       static_cast<float>(1u << 24);  // [0, 1)
    values[i] = unit * 2.0f - 1.0f;
  }
  return values;
}

}  // namespace accel::test

#endif  // ACCEL_TESTS_SUPPORT_REFERENCE_GEMM_H_
