// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Times the block-wise sum reduction and checks its result.
//
// The device produces one partial sum per block; the host adds those few values
// to finish. Only the kernel is timed — the upload and the host tail are
// reported separately so the split between them is visible.
//
// The grid-stride load means `--blocks` and `--threads` are free of `--count`,
// so the launch geometry can be swept independently of the input size.
//
// Effective read bandwidth is reported rather than GFLOP/s, because a reduction
// reads N floats and writes a handful and is therefore memory-bound. Be careful
// reading it at the default size: 4096 floats is 16 KiB, which the device
// consumes in well under a microsecond, so the figure describes launch overhead
// and not the memory system. Pass a --count in the hundreds of millions for a
// bandwidth number that means anything.
//
// docs/Benchmarks.md for published numbers.
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "core/cli_options.h"
#include "cuda/cuda_check.cuh"
#include "cuda/cuda_timer.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/sum_reduction.cuh"

namespace {

void PrintUsage() {
  std::cout << "Usage: bench_cuda_sum_reduction [options]\n"
               "  --count=INT    input elements            (default 4096)\n"
               "  --blocks=INT   grid size / partial sums  (default 8)\n"
               "  --threads=INT  block size, power of two  (default 512)\n"
               "  --iters=INT    timed repetitions         (default 100)\n"
               "  --help         show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
  using accel::CliOptions;
  using accel::CudaTimer;
  using accel::DeviceBuffer;
  using accel::kReductionBlockSize;
  using accel::kReductionGridSize;
  using accel::LaunchBlockSumReduce;

  try {
    const CliOptions options(argc, argv,
                             {"count", "blocks", "threads", "iters"});
    if (options.has("help")) {
      PrintUsage();
      return EXIT_SUCCESS;
    }

    const std::size_t count = options.PositiveSize("count", 4096);
    const int grid_size = options.PositiveInt("blocks", kReductionGridSize);
    const int block_size = options.PositiveInt("threads", kReductionBlockSize);
    const int iterations = options.PositiveInt("iters", 100);

    std::cout << "== CUDA sum reduction ==\n"
              << "   " << count << " floats, " << grid_size << " blocks x "
              << block_size << " threads\n\n";

    // All ones, so the exact sum is `count`. Below 2^24 every intermediate is
    // integral and representable in float, making the check effectively exact;
    // past that the shared-memory tree starts rounding, hence the relative
    // tolerance rather than equality.
    const std::vector<float> host_input(count, 1.0f);

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_partials(static_cast<std::size_t>(grid_size));
    device_input.CopyFromHost(host_input);

    LaunchBlockSumReduce(device_input.get(), device_partials.get(), count,
                         grid_size, block_size);
    CUDA_CHECK_LAUNCH();  // Warm-up.

    CudaTimer timer;
    timer.Start();
    for (int i = 0; i < iterations; ++i) {
      LaunchBlockSumReduce(device_input.get(), device_partials.get(), count,
                           grid_size, block_size);
    }
    timer.Stop();
    const double kernel_millis =
        timer.ElapsedMillis() / static_cast<double>(iterations);

    const std::vector<float> partials = device_partials.ToHost();
    double total = 0.0;
    for (const float value : partials) {
      total += static_cast<double>(value);
    }

    const double bytes_read =
        static_cast<double>(count) * static_cast<double>(sizeof(float));
    const double gib_per_second =
        bytes_read / (kernel_millis * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    const double expected = static_cast<double>(count);
    const double relative_error = std::fabs(total - expected) / expected;

    std::cout << std::fixed << std::setprecision(6) << "  kernel      "
              << kernel_millis << " ms per launch\n"
              << std::setprecision(2) << "  bandwidth   " << gib_per_second
              << " GiB/s effective read\n\n"
              << std::setprecision(1) << "  total       " << total << '\n'
              << "  expected    " << expected << '\n';

    const bool matched = relative_error <= 1e-6;
    std::cout << (matched ? "  [ok]   sum matches" : "  [FAIL] sum differs")
              << " (relative error " << std::scientific << relative_error
              << ")\n";
    return matched ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
