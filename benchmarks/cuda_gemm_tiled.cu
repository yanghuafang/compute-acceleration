// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Times the shared-memory tiled GEMM kernel on the device.
//
// Reports the average over a repetition loop bracketed by CUDA events, after a
// discarded warm-up launch. The warm-up matters: the first launch pays JIT or
// module-load cost and context setup, which on this shape is comparable to a
// whole timed iteration.
//
// Device memory and both events are RAII-owned, so a `CudaError` thrown from
// the check macro unwinds them rather than leaking them; the product is then
// verified on the host, since a fast kernel and a fast wrong kernel are
// otherwise indistinguishable.
//
// docs/Benchmarks.md for published numbers.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "benchmarks/support/reporting.h"
#include "core/cli_options.h"
#include "core/gemm_shape.h"
#include "cuda/cuda_check.cuh"
#include "cuda/cuda_timer.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/gemm_tiled.cuh"

namespace {

void PrintUsage() {
  std::cout << "Usage: bench_cuda_gemm_tiled [options]\n"
               "  --m=INT      rows of A and C        (default 4096)\n"
               "  --k=INT      contraction extent     (default 2048)\n"
               "  --n=INT      columns of B and C     (default 4096)\n"
               "  --iters=INT  timed repetitions      (default 100)\n"
               "  --help       show this message\n";
}

// Every input element is 1.0f, so each output must be exactly `k`. Summing k
// ones in `float` is exact for k below 2^24, which makes this an equality
// check rather than a tolerance check — any deviation is a real indexing or
// synchronisation bug, not rounding.
bool VerifyAllOnesProduct(const std::vector<float>& c,
                          const accel::GemmShape& shape) {
  const float expected = static_cast<float>(shape.k);
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i] != expected) {
      std::cout << "  [FAIL] C[" << i << "] = " << c[i] << ", expected "
                << expected << '\n';
      return false;
    }
  }
  std::cout << "  [ok]   all " << c.size() << " outputs equal " << expected
            << '\n';
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  namespace bench = accel::bench;
  using accel::CliOptions;
  using accel::CudaError;
  using accel::CudaTimer;
  using accel::DeviceBuffer;
  using accel::GemmShape;
  using accel::LaunchGemmTiled;

  try {
    const CliOptions options(argc, argv, {"m", "k", "n", "iters"});
    if (options.has("help")) {
      PrintUsage();
      return EXIT_SUCCESS;
    }

    const GemmShape shape{options.PositiveInt("m", 4096),
                          options.PositiveInt("k", 2048),
                          options.PositiveInt("n", 4096)};
    const int iterations = options.PositiveInt("iters", 100);

    bench::PrintHeader("CUDA GEMM - shared-memory tiling", shape);

    const std::vector<float> host_a(shape.a_elements(), 1.0f);
    const std::vector<float> host_b(shape.b_elements(), 1.0f);

    DeviceBuffer<float> device_a(shape.a_elements());
    DeviceBuffer<float> device_b(shape.b_elements());
    DeviceBuffer<float> device_c(shape.c_elements());

    device_a.CopyFromHost(host_a);
    device_b.CopyFromHost(host_b);

    LaunchGemmTiled(device_a.get(), device_b.get(), device_c.get(), shape);
    CUDA_CHECK_LAUNCH();  // Warm-up; also the first point a fault can surface.

    CudaTimer timer;
    timer.Start();
    for (int i = 0; i < iterations; ++i) {
      LaunchGemmTiled(device_a.get(), device_b.get(), device_c.get(), shape);
    }
    timer.Stop();

    // ElapsedMillis() synchronises on the stop event, so this is also the
    // barrier that makes the result below safe to read.
    const double average_seconds = timer.ElapsedMillis() / iterations / 1000.0;
    bench::PrintTiming("GemmTiledKernel", average_seconds, shape);
    std::cout << "  (" << iterations << " timed iterations, "
              << average_seconds * 1000.0 << " ms each)\n\n";

    const std::vector<float> host_c = device_c.ToHost();
    return VerifyAllOnesProduct(host_c, shape) ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
