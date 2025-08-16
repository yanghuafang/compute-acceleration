// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "core/gemm_shape.h"

#include <stdexcept>
#include <string>

namespace accel {
namespace {

// Builds a diagnostic naming the operand, so a caller that mis-sized one of
// three look-alike float buffers is told which.
[[noreturn]] void ThrowExtentMismatch(const char* operand, std::size_t actual,
                                      std::size_t expected) {
  throw std::invalid_argument(std::string("GEMM operand ") + operand + " has " +
                              std::to_string(actual) + " elements, expected " +
                              std::to_string(expected));
}

}  // namespace

void ValidateGemmOperands(Span<const float> a, Span<const float> b,
                          Span<const float> c, const GemmShape& shape) {
  if (!shape.is_valid()) {
    throw std::invalid_argument(
        "GEMM shape must have strictly positive extents, got m=" +
        std::to_string(shape.m) + " k=" + std::to_string(shape.k) +
        " n=" + std::to_string(shape.n));
  }
  if (a.size() != shape.a_elements()) {
    ThrowExtentMismatch("A", a.size(), shape.a_elements());
  }
  // Row-major and column-major B occupy the same k*n elements, so one check
  // serves both storage orders.
  if (b.size() != shape.b_elements()) {
    ThrowExtentMismatch("B", b.size(), shape.b_elements());
  }
  if (c.size() != shape.c_elements()) {
    ThrowExtentMismatch("C", c.size(), shape.c_elements());
  }
}

double GemmGflops(const GemmShape& shape, double seconds) noexcept {
  if (!(seconds > 0.0)) {
    return 0.0;  // Also rejects NaN, which would poison downstream reports.
  }
  const double flops = 2.0 * static_cast<double>(shape.mac_count());
  return flops / seconds / 1e9;
}

}  // namespace accel
