// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "omp/reduction_omp.h"

#include <cstddef>

#ifndef _OPENMP
#error "src/omp requires OpenMP; link the target against OpenMP::OpenMP_CXX"
#endif

namespace accel {

double SumOmp(Span<const float> input) noexcept {
  const float* const data = input.data();
  const auto count = static_cast<std::ptrdiff_t>(input.size());

  double total = 0.0;
  // The reduction clause gives each thread a private accumulator and combines
  // them at the end of the region. That is what keeps this from degenerating
  // into a contended atomic on one cache line, which is slower than the serial
  // loop it replaces.
#pragma omp parallel for schedule(static) default(none) shared(data, count) \
    reduction(+ : total)
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    total += static_cast<double>(data[static_cast<std::size_t>(i)]);
  }
  return total;
}

}  // namespace accel
