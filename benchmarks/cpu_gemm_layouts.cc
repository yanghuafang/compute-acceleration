// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Compares memory access strategies for the same CPU GEMM.
//
// Four kernels, one product. All differ only in the order they walk memory:
//   - `GemmIjkBColMajor`   dot products over a transposed B (baseline here)
//   - `GemmIkj`               loop reordering, B left row-major
//   - `GemmTiledBColMajor` cache blocking on top of the transposed B
//
// The transpose cost itself is timed and reported separately, because on the
// benchmark shape it is a real fraction of the fastest kernel's runtime and
// ignoring it overstates the layout win.
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
  std::cout << "Usage: bench_cpu_gemm_layouts [options]\n"
               "  --m=INT     rows of A and C            (default 4096)\n"
               "  --k=INT     contraction extent         (default 2048)\n"
               "  --n=INT     columns of B and C         (default 4096)\n"
               "  --tile=INT  blocking factor            (default 64)\n"
               "  --help      show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
  namespace bench = accel::bench;
  using accel::CliOptions;
  using accel::FillRamp;
  using accel::GemmIjkBColMajor;
  using accel::GemmIkj;
  using accel::GemmShape;
  using accel::GemmTiledBColMajor;
  using accel::kDefaultTileSize;
  using accel::ReportComparison;
  using accel::RowToColumnMajor;
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

    bench::PrintHeader("CPU GEMM - access-order comparison", shape);

    std::vector<float> a(shape.a_elements());
    std::vector<float> b(shape.b_elements());
    FillRamp(a);
    FillRamp(b);

    std::vector<float> b_col(shape.b_elements());
    const double transpose_seconds =
        TimeSeconds([&] { RowToColumnMajor(b, b_col, shape.k, shape.n); });
    std::cout << "  (one-off transpose of B: " << transpose_seconds
              << " s)\n\n";

    std::vector<float> c_col_major(shape.c_elements(), 0.0f);
    std::vector<float> c_reordered(shape.c_elements(), 0.0f);
    std::vector<float> c_tiled(shape.c_elements(), 0.0f);

    const double col_major_seconds =
        TimeSeconds([&] { GemmIjkBColMajor(a, b_col, c_col_major, shape); });
    bench::PrintTiming("GemmIjkBColMajor", col_major_seconds, shape);

    const double reordered_seconds =
        TimeSeconds([&] { GemmIkj(a, b, c_reordered, shape); });
    bench::PrintTiming("GemmIkj", reordered_seconds, shape);

    const double tiled_seconds = TimeSeconds(
        [&] { GemmTiledBColMajor(a, b_col, c_tiled, shape, tile_size); });
    bench::PrintTiming("GemmTiledBColMajor", tiled_seconds, shape);

    std::cout << '\n';
    // Every kernel sums the same k terms in a different order, so results agree
    // to within accumulated rounding, not bit-for-bit.
    bool matched = ReportComparison(std::cout, c_col_major, c_reordered,
                                    "GemmIjkBColMajor", "GemmIkj");
    matched &= ReportComparison(std::cout, c_col_major, c_tiled,
                                "GemmIjkBColMajor", "GemmTiledBColMajor");
    return matched ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
