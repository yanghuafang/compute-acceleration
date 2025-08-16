// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Isolates the effect of cache blocking on a row-major GEMM.
//
// Compares the textbook `i -> j -> k` kernel against its cache-blocked
// equivalent, with both operands row-major. Both kernels stride B by `n` floats
// in their innermost loop, so the only variable is whether the working set is
// blocked to fit in cache.
//
// Operands are seeded with a scaled Ramp and verified against a relative
// tolerance. Both matter: an unscaled `A[i] = i` pushes partial sums past 1e17,
// where `float` spacing exceeds 1e10, and an absolute tolerance of 1e-3 there
// is a check that cannot Fail.
//
// docs/Benchmarks.md for published numbers.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "benchmarks/support/reporting.h"
#include "core/cli_options.h"
#include "core/gemm_shape.h"
#include "core/matrix_utils.h"
#include "core/stopwatch.h"
#include "cpu/cpu_gemm.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: bench_cpu_gemm_row_major [options]\n"
         "  --m=INT     rows of A and C            (default 4096)\n"
         "  --k=INT     contraction extent         (default 2048)\n"
         "  --n=INT     columns of B and C         (default 4096)\n"
         "  --tile=INT  blocking factor            (default 64)\n"
         "  --help      show this message\n"
         "\n"
         "The default shape takes minutes on the naive kernel; pass small\n"
         "extents when running under a sanitizer.\n";
}

}  // namespace

int main(int argc, char** argv) {
  namespace bench = accel::bench;
  using accel::CliOptions;
  using accel::FillRamp;
  using accel::GemmIjk;
  using accel::GemmShape;
  using accel::GemmTiled;
  using accel::kDefaultTileSize;
  using accel::ReportComparison;
  using accel::TimeSeconds;

  try {
    const CliOptions options(argc, argv, {"m", "k", "n", "tile"});
    if (options.has("help")) {
      PrintUsage();
      return EXIT_SUCCESS;
    }

    const GemmShape shape{options.PositiveInt("m", 4096),
                          options.PositiveInt("k", 2048),
                          options.PositiveInt("n", 4096)};
    const int tile_size = options.PositiveInt("tile", kDefaultTileSize);

    bench::PrintHeader("CPU GEMM - row-major B", shape);

    std::vector<float> a(shape.a_elements());
    std::vector<float> b(shape.b_elements());
    FillRamp(a);
    FillRamp(b);

    // Separate outputs so both kernels start from zero and can be compared
    // afterwards; the kernels accumulate, so reusing one buffer would double
    // the second result.
    std::vector<float> c_naive(shape.c_elements(), 0.0f);
    std::vector<float> c_tiled(shape.c_elements(), 0.0f);

    const double naive_seconds =
        TimeSeconds([&] { GemmIjk(a, b, c_naive, shape); });
    bench::PrintTiming("GemmIjk", naive_seconds, shape);

    const double tiled_seconds =
        TimeSeconds([&] { GemmTiled(a, b, c_tiled, shape, tile_size); });
    bench::PrintTiming("GemmTiled", tiled_seconds, shape);

    std::cout << '\n';
    const bool matched =
        ReportComparison(std::cout, c_naive, c_tiled, "GemmIjk", "GemmTiled");
    return matched ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
