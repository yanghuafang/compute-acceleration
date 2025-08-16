// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Device-side correctness tests for the CUDA kernels and RAII wrappers.
//
// Requires a working CUDA device. When none is visible the suite prints a
// notice and exits successfully: a developer building on a laptop without an
// NVIDIA GPU should not see a red test run for hardware they do not have, and
// CI enforces the difference by checking that a device is present.
//
// These are the cases to run under `compute-sanitizer` — they are small enough
// that memcheck and racecheck finish in seconds. See docs/Sanitizers.md.
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/gemm_shape.h"
#include "core/matrix_utils.h"
#include "cuda/cuda_check.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/gemm_tiled.cuh"
#include "cuda/sum_reduction.cuh"
#include "tests/support/reference_gemm.h"
#include "tests/support/test_harness.h"

namespace {

using accel::DeviceBuffer;
using accel::GemmShape;

// Tolerance for a float GEMM against a double reference over short
// contractions.
constexpr float kTolerance = 1e-4f;

// Runs the tiled kernel end to end and returns C on the host.
std::vector<float> RunDeviceGemm(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 const GemmShape& shape) {
  DeviceBuffer<float> device_a(shape.a_elements());
  DeviceBuffer<float> device_b(shape.b_elements());
  DeviceBuffer<float> device_c(shape.c_elements());

  device_a.CopyFromHost(a);
  device_b.CopyFromHost(b);
  accel::LaunchGemmTiled(device_a.get(), device_b.get(), device_c.get(), shape);
  CUDA_CHECK_LAUNCH();
  return device_c.ToHost();
}

}  // namespace

// Extents that are exact multiples of the 32-wide tile: no masked lanes, so a
// failure here is a core indexing bug rather than a boundary one.
TEST_CASE(GemmTiledMatchesReferenceOnAlignedShape) {
  const GemmShape shape{64, 32, 96};
  const auto a = accel::test::PseudoRandomVector(shape.a_elements(), 7u);
  const auto b = accel::test::PseudoRandomVector(shape.b_elements(), 13u);
  const auto expected = accel::test::ReferenceGemm(a, b, shape);

  CHECK_TRUE(
      accel::MatricesClose(RunDeviceGemm(a, b, shape), expected, kTolerance));
}

// Extents no multiple of 32, so every dimension has a partial tile and the
// Zero-padding path is exercised on all three axes at once.
TEST_CASE(GemmTiledMatchesReferenceOnRaggedShape) {
  const GemmShape shape{37, 53, 41};
  const auto a = accel::test::PseudoRandomVector(shape.a_elements(), 7u);
  const auto b = accel::test::PseudoRandomVector(shape.b_elements(), 13u);
  const auto expected = accel::test::ReferenceGemm(a, b, shape);

  CHECK_TRUE(
      accel::MatricesClose(RunDeviceGemm(a, b, shape), expected, kTolerance));
}

// A single-element product: grid and block are almost entirely masked off.
TEST_CASE(GemmTiledHandlesMinimalShape) {
  const GemmShape shape{1, 1, 1};
  const std::vector<float> a{2.0f};
  const std::vector<float> b{3.0f};
  const std::vector<float> c = RunDeviceGemm(a, b, shape);
  CHECK_EQ(c.size(), std::size_t{1});
  CHECK_NEAR(c[0], 6.0f, 1e-6);
}

TEST_CASE(GemmTiledOverwritesRatherThanAccumulates) {
  const GemmShape shape{8, 8, 8};
  const std::vector<float> a(shape.a_elements(), 1.0f);
  const std::vector<float> b(shape.b_elements(), 1.0f);

  DeviceBuffer<float> device_a(shape.a_elements());
  DeviceBuffer<float> device_b(shape.b_elements());
  DeviceBuffer<float> device_c(shape.c_elements());
  device_a.CopyFromHost(a);
  device_b.CopyFromHost(b);

  // Two launches into the same output. Unlike the CPU variants, the device
  // kernel assigns, so the second result must equal the first.
  accel::LaunchGemmTiled(device_a.get(), device_b.get(), device_c.get(), shape);
  accel::LaunchGemmTiled(device_a.get(), device_b.get(), device_c.get(), shape);
  CUDA_CHECK_LAUNCH();

  const std::vector<float> c = device_c.ToHost();
  for (const float value : c) {
    CHECK_NEAR(value, static_cast<float>(shape.k), 1e-6);
  }
}

TEST_CASE(LaunchGemmTiledRejectsDegenerateShape) {
  DeviceBuffer<float> scratch(1);
  CHECK_THROWS_AS(accel::LaunchGemmTiled(scratch.get(), scratch.get(),
                                         scratch.get(), GemmShape{0, 1, 1}),
                  std::invalid_argument);
}

TEST_CASE(SumOnDeviceMatchesHostSum) {
  // Not a multiple of blocks * threads, so the grid-stride loop must handle a
  // ragged final pass — the case a fixed blocks * threads == count contract
  // would read out of bounds on.
  const std::size_t count = 100000;
  const std::vector<float> input(count, 1.0f);
  CHECK_NEAR(accel::SumOnDevice(input.data(), count),
             static_cast<double>(count), 1e-6);
}

TEST_CASE(SumOnDeviceHandlesGridSmallerThanInput) {
  const std::size_t count = 4096;
  const std::vector<float> input(count, 0.5f);
  // One block of 32 threads must still fold all 4096 elements.
  CHECK_NEAR(accel::SumOnDevice(input.data(), count, 1, 32),
             0.5 * static_cast<double>(count), 1e-6);
}

TEST_CASE(SumOnDeviceHandlesGridLargerThanInput) {
  // More threads than elements: most threads contribute nothing and must not
  // read past the end.
  const std::vector<float> input{1.0f, 2.0f, 3.0f};
  CHECK_NEAR(accel::SumOnDevice(input.data(), input.size(), 8, 128), 6.0, 1e-6);
}

TEST_CASE(SumOnDeviceReturnsZeroForEmptyInput) {
  CHECK_EQ(accel::SumOnDevice(nullptr, 0), 0.0);
}

// The geometry check must run before the partials buffer is sized from
// grid_size, and before the empty-input shortcut. A negative grid_size once
// became a huge size_t and surfaced as cudaErrorMemoryAllocation, which points
// the reader at the GPU instead of at their own argument.
TEST_CASE(SumOnDeviceValidatesGeometryBeforeAllocating) {
  const std::vector<float> input(16, 1.0f);
  CHECK_THROWS_AS(accel::SumOnDevice(input.data(), input.size(), -1, 128),
                  std::invalid_argument);
  CHECK_THROWS_AS(accel::SumOnDevice(input.data(), input.size(), 4, 500),
                  std::invalid_argument);
  CHECK_THROWS_AS(accel::SumOnDevice(nullptr, 0, 0, 128),
                  std::invalid_argument);
}

TEST_CASE(LaunchBlockSumReduceRejectsInvalidGeometry) {
  DeviceBuffer<float> input(16);
  DeviceBuffer<float> partials(4);

  // 500 is not a power of two, which the halving loop silently mis-reduces.
  CHECK_THROWS_AS(
      accel::LaunchBlockSumReduce(input.get(), partials.get(), 16, 4, 500),
      std::invalid_argument);
  // Above the hardware maximum threads per block.
  CHECK_THROWS_AS(
      accel::LaunchBlockSumReduce(input.get(), partials.get(), 16, 4, 2048),
      std::invalid_argument);
  CHECK_THROWS_AS(
      accel::LaunchBlockSumReduce(input.get(), partials.get(), 16, 0, 128),
      std::invalid_argument);
}

TEST_CASE(DeviceBufferMovesOwnershipWithoutDoubleFree) {
  DeviceBuffer<float> source(16);
  const float* const original = source.get();
  CHECK_TRUE(static_cast<bool>(source));

  DeviceBuffer<float> moved(std::move(source));
  CHECK_EQ(moved.get(), original);
  CHECK_EQ(moved.size(), std::size_t{16});
  // The moved-from buffer must be empty, or both destructors would free it.
  CHECK_FALSE(static_cast<bool>(source));
  CHECK_EQ(source.size(), std::size_t{0});

  DeviceBuffer<float> assigned;
  assigned = std::move(moved);
  CHECK_EQ(assigned.get(), original);
  CHECK_FALSE(static_cast<bool>(moved));
}

TEST_CASE(DeviceBufferRoundTripsData) {
  const std::vector<float> host{1.0f, 2.0f, 3.0f, 4.0f};
  DeviceBuffer<float> buffer(host.size());
  buffer.CopyFromHost(host);

  const std::vector<float> returned = buffer.ToHost();
  CHECK_EQ(returned.size(), host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    CHECK_EQ(returned[i], host[i]);
  }

  buffer.Zero();
  CUDA_CHECK(cudaDeviceSynchronize());
  for (const float value : buffer.ToHost()) {
    CHECK_EQ(value, 0.0f);
  }
}

TEST_CASE(DeviceBufferRejectsOversizedTransfers) {
  DeviceBuffer<float> buffer(4);
  const std::vector<float> too_large(8, 1.0f);
  CHECK_THROWS_AS(buffer.CopyFromHost(too_large), std::out_of_range);

  std::vector<float> destination(8);
  CHECK_THROWS_AS(buffer.CopyToHost(destination.data(), 8), std::out_of_range);
}

TEST_CASE(DeviceBufferZeroSizeAllocatesNothing) {
  DeviceBuffer<float> buffer(0);
  CHECK_TRUE(buffer.empty());
  CHECK_EQ(buffer.get(), static_cast<float*>(nullptr));
}

TEST_CASE(cudaCheckThrowsTypedErrorOnFailure) {
  bool caught = false;
  try {
    // Freeing a host pointer is a reliably invalid argument to cudaFree.
    int stack_value = 0;
    CUDA_CHECK(cudaFree(&stack_value));
  } catch (const accel::CudaError& error) {
    caught = true;
    CHECK_TRUE(error.code() != cudaSuccess);
    CHECK_TRUE(std::string(error.what()).find("cudaFree") != std::string::npos);
  }
  CHECK_TRUE(caught);
  // The failure above leaves a sticky error only if it was asynchronous; clear
  // any residue so later cases Start from a clean status.
  static_cast<void>(cudaGetLastError());
}

int main() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "no CUDA device available (" << cudaGetErrorString(status)
              << ") - skipping device tests\n";
    return 0;
  }
  return accel::test::RunAll();
}
