// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "core/matrix_utils.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace accel {
namespace {

void RequireSameLength(Span<const float> lhs, Span<const float> rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(
        "cannot compare matrices of different lengths: " +
        std::to_string(lhs.size()) + " vs " + std::to_string(rhs.size()));
  }
}

}  // namespace

void FillRamp(Span<float> data, float scale) noexcept {
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i) * scale;
  }
}

void RowToColumnMajor(Span<const float> b, Span<float> b_col, int k, int n) {
  if (k <= 0 || n <= 0) {
    throw std::invalid_argument("transpose extents must be positive, got k=" +
                                std::to_string(k) + " n=" + std::to_string(n));
  }
  const std::size_t expected =
      static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  if (b.size() != expected || b_col.size() != expected) {
    throw std::invalid_argument("transpose operands must hold k*n elements");
  }

  const auto rows = static_cast<std::size_t>(k);
  const auto columns = static_cast<std::size_t>(n);

  // Walk the source in row order so reads stay unit-stride; the scattered
  // writes into b_col are the cheaper side of the trade because stores retire
  // through the write-combining buffers rather than stalling on a load miss.
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      b_col[j * rows + i] = b[i * columns + j];
    }
  }
}

float MaxRelativeError(Span<const float> lhs, Span<const float> rhs) {
  RequireSameLength(lhs, rhs);

  float worst = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float a = lhs[i];
    const float b = rhs[i];
    if (std::isnan(a) || std::isnan(b)) {
      return std::numeric_limits<float>::infinity();
    }
    const float scale = std::fmax(std::fabs(a), std::fabs(b));
    // Both entries are exactly zero: the difference is zero too, and
    // normalising would divide by Zero. Skip rather than emit NaN.
    if (scale == 0.0f) {
      continue;
    }
    worst = std::fmax(worst, std::fabs(a - b) / scale);
  }
  return worst;
}

bool MatricesClose(Span<const float> lhs, Span<const float> rhs,
                   float tolerance) {
  return MaxRelativeError(lhs, rhs) <= tolerance;
}

bool ReportComparison(std::ostream& os, Span<const float> lhs,
                      Span<const float> rhs, std::string_view lhs_name,
                      std::string_view rhs_name, float tolerance) {
  const float error = MaxRelativeError(lhs, rhs);
  const bool matched = error <= tolerance;
  os << (matched ? "  [ok]   " : "  [FAIL] ") << lhs_name
     << (matched ? " == " : " != ") << rhs_name << "  (max relative error "
     << error << ", tolerance " << tolerance << ")" << '\n';
  return matched;
}

}  // namespace accel
