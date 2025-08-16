// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_BENCHMARKS_SUPPORT_REPORTING_H_
#define ACCEL_BENCHMARKS_SUPPORT_REPORTING_H_

#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "core/gemm_shape.h"
#include "core/stopwatch.h"

namespace accel::bench {

// Fastest of `iterations` runs of `body`.
//
// The minimum rather than the mean: every source of noise on a shared machine
// adds time, none subtracts it, so the fastest sample is the closest estimate
// of the uncontended cost. A single sample is not enough for the sub-second
// kernels here — it makes a one-thread parallel run look faster than the
// serial kernel it is supposed to match.
//
// Precondition: `iterations` > 0.
template <typename Callable>
double BestSeconds(int iterations, Callable&& body) {
  double best = 0.0;
  for (int i = 0; i < iterations; ++i) {
    const double seconds = TimeSeconds(body);
    if (i == 0 || seconds < best) {
      best = seconds;
    }
  }
  return best;
}

// Thread counts to sweep: powers of two up to `limit`, plus `limit` itself
// when it is not already one. A linear sweep would spend most of its samples
// in the region where the curve has already flattened, and stopping at the
// last power of two would miss the machine's actual core count.
//
// Precondition: `limit` > 0.
inline std::vector<int> ThreadLadder(int limit) {
  std::vector<int> steps;
  for (int threads = 1; threads <= limit; threads *= 2) {
    steps.push_back(threads);
  }
  if (!steps.empty() && steps.back() != limit) {
    steps.push_back(limit);
  }
  return steps;
}

// Column width chosen so the longest kernel label in this project still leaves
// the timing columns aligned.
inline constexpr int kLabelWidth = 26;

// Prints a section banner so interleaved benchmark output stays readable when
// several drivers run back to back from a script.
inline void PrintHeader(std::string_view title, const GemmShape& shape) {
  std::cout << "== " << title << " ==\n"
            << "   shape: A(" << shape.m << 'x' << shape.k << ") * B("
            << shape.k << 'x' << shape.n << ") = C(" << shape.m << 'x'
            << shape.n << ")\n\n";
}

// One timing row: label, seconds, and derived throughput.
inline void PrintTiming(std::string_view label, double seconds,
                        const GemmShape& shape) {
  const std::ios_base::fmtflags saved = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();

  std::cout << "  " << std::left << std::setw(kLabelWidth) << label
            << std::right << std::fixed << std::setprecision(3) << std::setw(10)
            << seconds << " s" << std::setw(10) << GemmGflops(shape, seconds)
            << " GFLOP/s\n";

  // Restore rather than leak formatting state onto the shared stream; the
  // verification output that follows expects default float formatting.
  std::cout.flags(saved);
  std::cout.precision(saved_precision);
}

}  // namespace accel::bench

#endif  // ACCEL_BENCHMARKS_SUPPORT_REPORTING_H_
