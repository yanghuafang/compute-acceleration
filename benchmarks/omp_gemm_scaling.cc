// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Measures how the fastest CPU GEMM scales with OpenMP thread count.
//
// The single-thread row of this table is the serial number from
// bench_cpu_gemm_layouts, so the speedup column separates two things the
// CPU-versus-GPU comparison otherwise conflates: what parallelism buys, and
// what the device buys on top of it.
//
// Expect the curve to bend well before the core count. The kernel streams all
// of B for every block of rows, so it runs out of memory bandwidth before it
// runs out of cores, and on a hybrid CPU the efficiency cores flatten it
// further.
//
// docs/Benchmarks.md for published numbers.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "benchmarks/support/reporting.h"
#include "core/cli_options.h"
#include "core/gemm_shape.h"
#include "core/matrix_utils.h"
#include "core/stopwatch.h"
#include "cpu/cpu_gemm.h"
#include "omp/gemm_omp.h"
#include "omp/omp_control.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: bench_omp_gemm_scaling [options]\n"
         "  --m=INT        rows of A and C          (default 1024)\n"
         "  --k=INT        contraction extent       (default 512)\n"
         "  --n=INT        columns of B and C       (default 1024)\n"
         "  --tile=INT     blocking factor          (default 64)\n"
         "  --threads=INT  highest thread count     (default: all cores)\n"
         "  --iters=INT    timed repetitions        (default 3)\n"
         "  --help         show this message\n"
         "\n"
         "Sweeps 1, 2, 4, ... up to --threads, reporting the best of --iters\n"
         "runs at each step. The default shape is smaller than the other\n"
         "drivers use because this one runs it once per step per repetition.\n";
}

// One ladder row: the timing line, then the two ratios that matter.
//
// Two denominators, because they answer different questions and folding them
// into one number is how a scaling table comes to overstate its case. The
// OpenMP kernel at one thread does not match the serial kernel's time - the
// same loop nest compiled into an outlined OpenMP region optimises
// differently - so "vs serial" carries that offset and "vs 1 thread" is the
// clean parallel speedup.
void PrintLadderRow(const std::string& label, double seconds,
                    const accel::GemmShape& shape, double serial,
                    double single_thread) {
  accel::bench::PrintTiming(label, seconds, shape);

  const std::ios_base::fmtflags saved = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2) << "  "
            << std::setw(accel::bench::kLabelWidth) << ' ' << "  vs serial "
            << (seconds > 0.0 ? serial / seconds : 0.0) << "x,  vs 1 thread "
            << (seconds > 0.0 ? single_thread / seconds : 0.0) << "x\n";
  std::cout.flags(saved);
  std::cout.precision(saved_precision);
}

}  // namespace

int main(int argc, char** argv) {
  namespace bench = accel::bench;
  using accel::CliOptions;
  using accel::FillRamp;
  using accel::GemmIkj;
  using accel::GemmIkjOmp;
  using accel::GemmShape;
  using accel::GemmTiledBColMajorOmp;
  using accel::kDefaultTileSize;
  using accel::MatricesClose;
  using accel::OpenmpMaxThreads;
  using accel::OpenmpSetThreads;
  using accel::OpenmpThreadsUsed;
  using accel::ReportComparison;
  using accel::RowToColumnMajor;

  try {
    const CliOptions options(argc, argv,
                             {"m", "k", "n", "tile", "threads", "iters"});
    if (options.has("help")) {
      PrintUsage();
      return EXIT_SUCCESS;
    }

    const GemmShape shape{options.PositiveInt("m", 1024),
                          options.PositiveInt("k", 512),
                          options.PositiveInt("n", 1024)};
    const int tile_size = options.PositiveInt("tile", kDefaultTileSize);
    const int max_threads = options.PositiveInt("threads", OpenmpMaxThreads());
    const int iterations = options.PositiveInt("iters", 3);

    bench::PrintHeader("OpenMP GEMM scaling - GemmIkjOmp", shape);

    std::vector<float> a(shape.a_elements());
    std::vector<float> b(shape.b_elements());
    FillRamp(a);
    FillRamp(b);

    std::vector<float> b_col(shape.b_elements());
    RowToColumnMajor(b, b_col, shape.k, shape.n);

    // The serial kernel is the reference for both timing and correctness: a
    // speedup against a threaded one-thread run would quietly absorb whatever
    // the OpenMP entry and exit cost.
    std::vector<float> c_serial(shape.c_elements(), 0.0f);
    const double serial_seconds = bench::BestSeconds(iterations, [&] {
      std::fill(c_serial.begin(), c_serial.end(), 0.0f);
      GemmIkj(a, b, c_serial, shape);
    });
    bench::PrintTiming("GemmIkj (serial)", serial_seconds, shape);
    std::cout << '\n';

    bool all_matched = true;
    double single_thread_seconds = 0.0;
    for (const int threads : bench::ThreadLadder(max_threads)) {
      OpenmpSetThreads(threads);
      const int used = OpenmpThreadsUsed();

      std::vector<float> c(shape.c_elements(), 0.0f);
      // Re-zeroed per repetition because the kernels accumulate; without it
      // the second run would double the result and fail verification.
      const double seconds = bench::BestSeconds(iterations, [&] {
        std::fill(c.begin(), c.end(), 0.0f);
        GemmIkjOmp(a, b, c, shape);
      });
      if (single_thread_seconds == 0.0) {
        single_thread_seconds = seconds;
      }

      const std::string label =
          std::to_string(used) + (used == 1 ? " thread" : " threads");
      PrintLadderRow(label, seconds, shape, serial_seconds,
                     single_thread_seconds);

      all_matched = MatricesClose(c, c_serial) && all_matched;
    }

    // Whether blocking still pays once several threads share a cache is a
    // different question from how far the AXPY kernel scales, so it gets its
    // own row at full width rather than a column in the ladder.
    std::cout << '\n';
    OpenmpSetThreads(max_threads);
    std::vector<float> c_tiled(shape.c_elements(), 0.0f);
    const double tiled_seconds = bench::BestSeconds(iterations, [&] {
      std::fill(c_tiled.begin(), c_tiled.end(), 0.0f);
      GemmTiledBColMajorOmp(a, b_col, c_tiled, shape, tile_size);
    });
    bench::PrintTiming("GemmTiledBColMajorOmp", tiled_seconds, shape);

    std::cout << '\n';
    const bool tiled_matched = ReportComparison(
        std::cout, c_serial, c_tiled, "GemmIkj", "GemmTiledBColMajorOmp");

    if (!all_matched) {
      std::cout << "  [FAIL] a threaded result differed from the serial one\n";
    }
    return (all_matched && tiled_matched) ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
