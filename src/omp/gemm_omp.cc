// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "omp/gemm_omp.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef _OPENMP
#error "src/omp requires OpenMP; link the target against OpenMP::OpenMP_CXX"
#endif

namespace accel {
namespace {

void RequirePositiveTile(int tile_size) {
  if (tile_size <= 0) {
    throw std::invalid_argument("tile size must be positive, got " +
                                std::to_string(tile_size));
  }
}

// Extents widened once, after validation has established they are positive.
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

void GemmIkjOmp(Span<const float> a, Span<const float> b, Span<float> c,
                const GemmShape& shape) {
  ValidateGemmOperands(a, b, c, shape);
  const Extents e = Widen(shape);

  // A signed induction variable because the canonical loop form OpenMP
  // requires predates the unsigned relaxation in OpenMP 3.0, and MSVC still
  // implements only 2.0.
  const auto rows = static_cast<std::ptrdiff_t>(e.m);

  // schedule(static) both because every row costs the same - making dynamic
  // scheduling pure overhead - and because it fixes which thread owns which
  // row, so results do not shift with load.
#pragma omp parallel for schedule(static) default(none) shared(a, b, c, e, rows)
  for (std::ptrdiff_t row = 0; row < rows; ++row) {
    const auto i = static_cast<std::size_t>(row);
    const std::size_t a_row = i * e.k;
    const std::size_t c_row = i * e.n;
    for (std::size_t l = 0; l < e.k; ++l) {
      const float a_value = a[a_row + l];
      const std::size_t b_row = l * e.n;
      for (std::size_t j = 0; j < e.n; ++j) {
        c[c_row + j] += a_value * b[b_row + j];
      }
    }
  }
}

void GemmTiledBColMajorOmp(Span<const float> a, Span<const float> b_col,
                           Span<float> c, const GemmShape& shape,
                           int tile_size) {
  ValidateGemmOperands(a, b_col, c, shape);
  RequirePositiveTile(tile_size);

  const Extents e = Widen(shape);
  const auto tile = static_cast<std::size_t>(tile_size);

  const float* const a_data = a.data();
  const float* const b_data = b_col.data();
  float* const c_data = c.data();

  // Distributing row *blocks* rather than rows keeps each thread's blocked
  // working set intact; splitting inside a block would have several threads
  // streaming the same tiles.
  const std::size_t block_count = (e.m + tile - 1) / tile;
  const auto blocks = static_cast<std::ptrdiff_t>(block_count);

#pragma omp parallel for schedule(static) default(none) \
    shared(a_data, b_data, c_data, e, tile, blocks)
  for (std::ptrdiff_t block = 0; block < blocks; ++block) {
    const std::size_t ii = static_cast<std::size_t>(block) * tile;
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
            float sum = c_data[c_row + j];
            for (std::size_t l = kk; l < k_end; ++l) {
              sum += a_data[a_row + l] * b_data[b_column + l];
            }
            c_data[c_row + j] = sum;
          }
        }
      }
    }
  }
}

}  // namespace accel
