// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "cpu/cpu_gemm.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace accel {
namespace {

void RequirePositiveTile(int tile_size) {
  if (tile_size <= 0) {
    throw std::invalid_argument("tile size must be positive, got " +
                                std::to_string(tile_size));
  }
}

// Extents widened once, after validation has established they are positive.
// Every kernel below drives its loops with `std::size_t` so that no index
// expression can overflow `int` and no signed/unsigned conversion is left
// implicit — the benchmark shape already needs 24 bits for a single offset.
struct Extents {
  std::size_t m;
  std::size_t k;
  std::size_t n;
};

Extents Widen(const GemmShape& shape) noexcept {
  return Extents{static_cast<std::size_t>(shape.m),
                 static_cast<std::size_t>(shape.k),
                 static_cast<std::size_t>(shape.n)};
}

}  // namespace

void GemmIjk(Span<const float> a, Span<const float> b, Span<float> c,
             const GemmShape& shape) {
  ValidateGemmOperands(a, b, c, shape);
  const Extents e = Widen(shape);

  for (std::size_t i = 0; i < e.m; ++i) {
    const std::size_t a_row = i * e.k;
    const std::size_t c_row = i * e.n;
    for (std::size_t j = 0; j < e.n; ++j) {
      // Accumulate in a local so the compiler keeps the running sum in a
      // register; C is touched exactly twice per output element.
      float sum = c[c_row + j];
      for (std::size_t l = 0; l < e.k; ++l) {
        sum += a[a_row + l] * b[l * e.n + j];
      }
      c[c_row + j] = sum;
    }
  }
}

void GemmIkj(Span<const float> a, Span<const float> b, Span<float> c,
             const GemmShape& shape) {
  ValidateGemmOperands(a, b, c, shape);
  const Extents e = Widen(shape);

  for (std::size_t i = 0; i < e.m; ++i) {
    const std::size_t a_row = i * e.k;
    const std::size_t c_row = i * e.n;
    for (std::size_t l = 0; l < e.k; ++l) {
      // Loop-invariant scalar: one A load feeds an entire row update.
      const float a_value = a[a_row + l];
      const std::size_t b_row = l * e.n;
      // Unit-stride over both B and C with no loop-carried dependence between
      // iterations, which is exactly the shape auto-vectorisers accept.
      for (std::size_t j = 0; j < e.n; ++j) {
        c[c_row + j] += a_value * b[b_row + j];
      }
    }
  }
}

void GemmTiled(Span<const float> a, Span<const float> b, Span<float> c,
               const GemmShape& shape, int tile_size) {
  ValidateGemmOperands(a, b, c, shape);
  RequirePositiveTile(tile_size);

  const Extents e = Widen(shape);
  const auto tile = static_cast<std::size_t>(tile_size);

  for (std::size_t ii = 0; ii < e.m; ii += tile) {
    const std::size_t i_end = std::min(ii + tile, e.m);
    for (std::size_t jj = 0; jj < e.n; jj += tile) {
      const std::size_t j_end = std::min(jj + tile, e.n);
      for (std::size_t kk = 0; kk < e.k; kk += tile) {
        const std::size_t k_end = std::min(kk + tile, e.k);

        // Each (ii, jj) block of C is revisited once per kk block, so partial
        // sums must round-trip through memory; only the operand tiles stay
        // resident in cache across the innermost nest.
        for (std::size_t i = ii; i < i_end; ++i) {
          const std::size_t a_row = i * e.k;
          const std::size_t c_row = i * e.n;
          for (std::size_t j = jj; j < j_end; ++j) {
            float sum = c[c_row + j];
            for (std::size_t l = kk; l < k_end; ++l) {
              sum += a[a_row + l] * b[l * e.n + j];
            }
            c[c_row + j] = sum;
          }
        }
      }
    }
  }
}

void GemmIjkBColMajor(Span<const float> a, Span<const float> b_col,
                      Span<float> c, const GemmShape& shape) {
  ValidateGemmOperands(a, b_col, c, shape);
  const Extents e = Widen(shape);

  for (std::size_t i = 0; i < e.m; ++i) {
    const std::size_t c_row = i * e.n;
    const float* const a_row = a.data() + i * e.k;
    for (std::size_t j = 0; j < e.n; ++j) {
      // Column j of B is contiguous under this layout, so the inner loop is a
      // dot product of two unit-stride vectors.
      const float* const b_column = b_col.data() + j * e.k;
      float sum = c[c_row + j];
      for (std::size_t l = 0; l < e.k; ++l) {
        sum += a_row[l] * b_column[l];
      }
      c[c_row + j] = sum;
    }
  }
}

void GemmTiledBColMajor(Span<const float> a, Span<const float> b_col,
                        Span<float> c, const GemmShape& shape, int tile_size) {
  ValidateGemmOperands(a, b_col, c, shape);
  RequirePositiveTile(tile_size);

  const Extents e = Widen(shape);
  const auto tile = static_cast<std::size_t>(tile_size);

  for (std::size_t ii = 0; ii < e.m; ii += tile) {
    const std::size_t i_end = std::min(ii + tile, e.m);
    for (std::size_t jj = 0; jj < e.n; jj += tile) {
      const std::size_t j_end = std::min(jj + tile, e.n);
      for (std::size_t kk = 0; kk < e.k; kk += tile) {
        const std::size_t k_end = std::min(kk + tile, e.k);

        for (std::size_t i = ii; i < i_end; ++i) {
          const std::size_t a_row = i * e.k;
          const std::size_t c_row = i * e.n;
          for (std::size_t j = jj; j < j_end; ++j) {
            const std::size_t b_column = j * e.k;
            float sum = c[c_row + j];
            for (std::size_t l = kk; l < k_end; ++l) {
              sum += a[a_row + l] * b_col[b_column + l];
            }
            c[c_row + j] = sum;
          }
        }
      }
    }
  }
}

}  // namespace accel
