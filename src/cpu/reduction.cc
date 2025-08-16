// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "cpu/reduction.h"

#include <array>
#include <cstddef>

namespace accel {

double SumSequential(Span<const float> input) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < input.size(); ++i) {
    total += static_cast<double>(input[i]);
  }
  return total;
}

double SumBlocked(Span<const float> input) noexcept {
  constexpr auto kLanes = static_cast<std::size_t>(kSumLanes);
  std::array<double, kLanes> partial{};

  const std::size_t count = input.size();
  // Trailing elements are handled separately so the main loop has a fixed trip
  // count of `kLanes`, which is what lets the compiler unroll it into vector
  // registers instead of emitting a runtime-bounded inner loop.
  const std::size_t body = count - (count % kLanes);

  for (std::size_t i = 0; i < body; i += kLanes) {
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      partial[lane] += static_cast<double>(input[i + lane]);
    }
  }

  double total = 0.0;
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    total += partial[lane];
  }
  for (std::size_t i = body; i < count; ++i) {
    total += static_cast<double>(input[i]);
  }
  return total;
}

}  // namespace accel
