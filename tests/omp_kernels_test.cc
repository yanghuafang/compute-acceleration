// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Correctness tests for the OpenMP GEMM and reduction kernels.
//
// Every case runs at more than one thread count, including one. A parallel
// kernel that is correct on a single thread and wrong on eight is the common
// failure, and a suite that never varies the count cannot see it. Shapes stay
// prime-ish and small so the suite is still cheap under TSan, where it is the
// only suite with anything to find.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/gemm_shape.h"
#include "core/matrix_utils.h"
#include "cpu/cpu_gemm.h"
#include "cpu/reduction.h"
#include "omp/gemm_omp.h"
#include "omp/omp_control.h"
#include "omp/reduction_omp.h"
#include "tests/support/reference_gemm.h"
#include "tests/support/test_harness.h"

namespace {

using accel::GemmShape;
using accel::OpenmpSetThreads;

// Not a multiple of the 64-element tile, and none of the three extents equal,
// so a transposed index or a mishandled tail block cannot pass.
constexpr GemmShape kShape{37, 53, 41};

constexpr float kTolerance = 1e-4f;

// More threads than the work can use, so the runtime's own clamping and the
// empty-slice case are both exercised.
const int kThreadCounts[] = {1, 2, 3, 8};

struct Operands {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> b_col;
  std::vector<float> expected;
};

Operands MakeOperands(const GemmShape& shape) {
  Operands operands;
  operands.a = accel::test::PseudoRandomVector(shape.a_elements(), 12345u);
  operands.b = accel::test::PseudoRandomVector(shape.b_elements(), 67890u);
  operands.b_col.resize(shape.b_elements());
  accel::RowToColumnMajor(operands.b, operands.b_col, shape.k, shape.n);
  operands.expected = accel::test::ReferenceGemm(operands.a, operands.b, shape);
  return operands;
}

}  // namespace

TEST_CASE(GemmIkjOmpMatchesTheReferenceAtEveryThreadCount) {
  const Operands operands = MakeOperands(kShape);
  for (const int threads : kThreadCounts) {
    OpenmpSetThreads(threads);
    std::vector<float> c(kShape.c_elements(), 0.0f);
    accel::GemmIkjOmp(operands.a, operands.b, c, kShape);
    CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
  }
}

TEST_CASE(GemmTiledBColMajorOmpMatchesTheReference) {
  const Operands operands = MakeOperands(kShape);
  for (const int threads : kThreadCounts) {
    OpenmpSetThreads(threads);
    std::vector<float> c(kShape.c_elements(), 0.0f);
    accel::GemmTiledBColMajorOmp(operands.a, operands.b_col, c, kShape);
    CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
  }
}

TEST_CASE(OmpGemmIsBitIdenticalAcrossThreadCounts) {
  // Each output element is accumulated by one thread in a fixed order, so the
  // thread count must not perturb the result at all. Any drift here means the
  // row partition is not the disjoint one the header promises.
  const Operands operands = MakeOperands(kShape);

  OpenmpSetThreads(1);
  std::vector<float> serial_threaded(kShape.c_elements(), 0.0f);
  accel::GemmIkjOmp(operands.a, operands.b, serial_threaded, kShape);

  OpenmpSetThreads(4);
  std::vector<float> parallel(kShape.c_elements(), 0.0f);
  accel::GemmIkjOmp(operands.a, operands.b, parallel, kShape);

  CHECK_EQ(accel::MaxRelativeError(serial_threaded, parallel), 0.0f);
}

TEST_CASE(OmpGemmMatchesItsSerialCounterpartExactly) {
  const Operands operands = MakeOperands(kShape);

  std::vector<float> serial(kShape.c_elements(), 0.0f);
  accel::GemmIkj(operands.a, operands.b, serial, kShape);

  OpenmpSetThreads(4);
  std::vector<float> parallel(kShape.c_elements(), 0.0f);
  accel::GemmIkjOmp(operands.a, operands.b, parallel, kShape);

  // Same loop nest, same association per element - only the row owner differs.
  CHECK_EQ(accel::MaxRelativeError(serial, parallel), 0.0f);
}

TEST_CASE(OmpGemmAccumulatesRatherThanAssigns) {
  const Operands operands = MakeOperands(kShape);
  OpenmpSetThreads(4);

  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmIkjOmp(operands.a, operands.b, c, kShape);
  accel::GemmIkjOmp(operands.a, operands.b, c, kShape);

  std::vector<float> doubled = operands.expected;
  for (float& value : doubled) {
    value *= 2.0f;
  }
  CHECK_TRUE(accel::MatricesClose(c, doubled, kTolerance));
}

TEST_CASE(OmpGemmRejectsDegenerateShapesBeforeSpawningThreads) {
  const std::vector<float> tiny(1, 1.0f);
  std::vector<float> out(1, 0.0f);
  CHECK_THROWS_AS(accel::GemmIkjOmp(tiny, tiny, out, GemmShape{0, 1, 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      accel::GemmTiledBColMajorOmp(tiny, tiny, out, GemmShape{1, 1, 1}, 0),
      std::invalid_argument);
}

TEST_CASE(SumOmpMatchesTheSerialSumAtEveryThreadCount) {
  const std::vector<float> values =
      accel::test::PseudoRandomVector(65537, 999u);
  const double expected = accel::SumSequential(values);

  for (const int threads : kThreadCounts) {
    OpenmpSetThreads(threads);
    const double actual = accel::SumOmp(values);
    // Relative, because the runtime chooses the combine order and the thread
    // count changes it. Equality here would be testing the scheduler.
    CHECK_TRUE(std::fabs(actual - expected) <=
               1e-9 * std::fabs(expected) + 1e-9);
  }
}

TEST_CASE(SumOmpHandlesCountsSmallerThanTheThreadCount) {
  // Threads that receive an empty slice must still contribute their identity.
  const std::vector<float> three{1.0f, 2.0f, 4.0f};
  OpenmpSetThreads(8);
  CHECK_NEAR(accel::SumOmp(three), 7.0, 1e-12);
}

TEST_CASE(SumOmpSumsAnEmptySpanToZero) {
  const std::vector<float> empty;
  OpenmpSetThreads(4);
  CHECK_EQ(accel::SumOmp(empty), 0.0);
}

TEST_CASE(OpenmpReportsTheThreadCountItActuallyUsed) {
  OpenmpSetThreads(1);
  CHECK_EQ(accel::OpenmpThreadsUsed(), 1);

  OpenmpSetThreads(2);
  CHECK_TRUE(accel::OpenmpMaxThreads() >= 1);
  // Never more than requested; the runtime may legitimately supply fewer.
  CHECK_TRUE(accel::OpenmpThreadsUsed() <= 2);
}

TEST_CASE(OpenmpSetThreadsRejectsANonPositiveCount) {
  CHECK_THROWS_AS(OpenmpSetThreads(0), std::invalid_argument);
  CHECK_THROWS_AS(OpenmpSetThreads(-4), std::invalid_argument);
}

int main() { return accel::test::RunAll(); }
