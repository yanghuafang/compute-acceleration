// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Compares the three host sum reductions against each other.
//
// A reduction reads N floats and writes one, so it is memory-bound long before
// it is arithmetic-bound. That makes it the cleaner of the two tasks for
// separating the three sources of speed this project measures: instruction-
// level parallelism (SumBlocked), thread-level parallelism (SumOmp), and the
// device (bench_cuda_sum_reduction, reported in the same GiB/s units).
//
// Effective read bandwidth rather than GFLOP/s for the same reason. Be careful
// at small counts: a few MiB fits in cache and the figure then describes the
// cache, not the memory system.
//
// docs/Benchmarks.md for published numbers.
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "benchmarks/support/reporting.h"
#include "core/cli_options.h"
#include "core/stopwatch.h"
#include "cpu/reduction.h"
#include "omp/omp_control.h"
#include "omp/reduction_omp.h"

namespace {

constexpr int kLabelWidth = 22;

void PrintUsage() {
  std::cout
      << "Usage: bench_omp_sum_reduction [options]\n"
         "  --count=INT    input elements        (default 67108864)\n"
         "  --threads=INT  highest thread count  (default: all cores)\n"
         "  --iters=INT    timed repetitions     (default 5)\n"
         "  --help         show this message\n"
         "\n"
         "The default is 256 MiB of floats, chosen to exceed last-level cache\n"
         "on the machines this project targets. Smaller counts measure "
         "cache.\n";
}

// One row: label, seconds, and effective read bandwidth over `bytes`.
void PrintRow(const std::string& label, double seconds, double bytes) {
  const std::ios_base::fmtflags saved = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();

  const double gib_per_second =
      seconds > 0.0 ? bytes / seconds / (1024.0 * 1024.0 * 1024.0) : 0.0;
  std::cout << "  " << std::left << std::setw(kLabelWidth) << label
            << std::right << std::fixed << std::setprecision(4) << std::setw(10)
            << seconds << " s" << std::setprecision(2) << std::setw(10)
            << gib_per_second << " GiB/s\n";

  std::cout.flags(saved);
  std::cout.precision(saved_precision);
}

}  // namespace

int main(int argc, char** argv) {
  namespace bench = accel::bench;
  using accel::CliOptions;
  using accel::OpenmpMaxThreads;
  using accel::OpenmpSetThreads;
  using accel::OpenmpThreadsUsed;
  using accel::SumBlocked;
  using accel::SumOmp;
  using accel::SumSequential;

  try {
    const CliOptions options(argc, argv, {"count", "threads", "iters"});
    if (options.has("help")) {
      PrintUsage();
      return EXIT_SUCCESS;
    }

    const std::size_t count = options.PositiveSize("count", 67108864);
    const int max_threads = options.PositiveInt("threads", OpenmpMaxThreads());
    const int iterations = options.PositiveInt("iters", 5);

    const double bytes =
        static_cast<double>(count) * static_cast<double>(sizeof(float));
    std::cout << "== Host sum reduction ==\n"
              << "   " << count << " floats (" << bytes / (1024.0 * 1024.0)
              << " MiB), best of " << iterations << "\n\n";

    // All ones: the exact total is `count`, so the check is arithmetic rather
    // than a second implementation of the same loop. Above 2^24 a float
    // accumulator would stall at 16777216; these variants accumulate in double
    // and must not.
    const std::vector<float> input(count, 1.0f);
    const auto expected = static_cast<double>(count);

    double sequential = 0.0;
    const double sequential_seconds = bench::BestSeconds(
        iterations, [&] { sequential = SumSequential(input); });
    PrintRow("SumSequential", sequential_seconds, bytes);

    double blocked = 0.0;
    const double blocked_seconds =
        bench::BestSeconds(iterations, [&] { blocked = SumBlocked(input); });
    PrintRow("SumBlocked", blocked_seconds, bytes);

    std::cout << '\n';
    bool matched = true;
    for (const int threads : bench::ThreadLadder(max_threads)) {
      OpenmpSetThreads(threads);
      const int used = OpenmpThreadsUsed();

      double total = 0.0;
      const double seconds =
          bench::BestSeconds(iterations, [&] { total = SumOmp(input); });
      PrintRow("SumOmp x" + std::to_string(used), seconds, bytes);

      matched = matched && std::fabs(total - expected) <= 1e-6 * expected;
    }

    matched = matched && std::fabs(sequential - expected) <= 1e-6 * expected &&
              std::fabs(blocked - expected) <= 1e-6 * expected;

    std::cout << '\n'
              << std::fixed << std::setprecision(1) << "  expected    "
              << expected << '\n'
              << "  sequential  " << sequential << '\n'
              << "  blocked     " << blocked << '\n'
              << (matched ? "  [ok]   every variant matches\n"
                          : "  [FAIL] a variant disagrees\n");
    return matched ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
