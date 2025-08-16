// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Tests for the shared helpers, with emphasis on the comparison edge cases that
// silently defeat result verification.
//
// The Zero/Zero comparison and the NaN handling are the reason this file
// exists: a relative comparison that divides by `max(|a|, |b|)` unguarded
// yields NaN for an all-Zero result, and `NaN > tolerance` is false — so a
// benchmark would print "equals" for output it had never actually verified.
#include "core/matrix_utils.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/gemm_shape.h"
#include "core/span.h"
#include "core/stopwatch.h"
#include "tests/support/test_harness.h"

TEST_CASE(FillRampScalesIndices) {
  std::vector<float> values(4);
  accel::FillRamp(values, 0.5f);
  CHECK_EQ(values[0], 0.0f);
  CHECK_EQ(values[1], 0.5f);
  CHECK_EQ(values[3], 1.5f);
}

TEST_CASE(RowToColumnMajorTransposes) {
  // 2x3 row-major: [1 2 3; 4 5 6]
  const std::vector<float> b{1, 2, 3, 4, 5, 6};
  std::vector<float> b_col(6);
  accel::RowToColumnMajor(b, b_col, 2, 3);

  // Column-major: column j is contiguous, so [1 4 | 2 5 | 3 6].
  const std::vector<float> expected{1, 4, 2, 5, 3, 6};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK_EQ(b_col[i], expected[i]);
  }
}

TEST_CASE(RowToColumnMajorRejectsBadExtents) {
  const std::vector<float> b(6, 1.0f);
  std::vector<float> b_col(6);
  CHECK_THROWS_AS(accel::RowToColumnMajor(b, b_col, 0, 3),
                  std::invalid_argument);
  CHECK_THROWS_AS(accel::RowToColumnMajor(b, b_col, 2, 4),
                  std::invalid_argument);
}

TEST_CASE(MaxRelativeErrorTreatsZeroPairsAsEqual) {
  const std::vector<float> zeros(8, 0.0f);
  // An unguarded relative comparison returns NaN here, which no tolerance
  // test can catch, because NaN > tolerance is false.
  CHECK_EQ(accel::MaxRelativeError(zeros, zeros), 0.0f);
  CHECK_TRUE(accel::MatricesClose(zeros, zeros));
}

TEST_CASE(MaxRelativeErrorNormalisesByMagnitude) {
  const std::vector<float> lhs{100.0f};
  const std::vector<float> rhs{101.0f};
  CHECK_NEAR(accel::MaxRelativeError(lhs, rhs), 1.0 / 101.0, 1e-6);
}

TEST_CASE(MaxRelativeErrorRejectsNan) {
  const std::vector<float> clean{1.0f, 2.0f};
  const std::vector<float> tainted{1.0f,
                                   std::numeric_limits<float>::quiet_NaN()};
  CHECK_TRUE(std::isinf(accel::MaxRelativeError(clean, tainted)));
  CHECK_FALSE(accel::MatricesClose(clean, tainted, 1e30f));
}

TEST_CASE(MaxRelativeErrorRejectsLengthMismatch) {
  const std::vector<float> lhs(4, 1.0f);
  const std::vector<float> rhs(5, 1.0f);
  CHECK_THROWS_AS(accel::MaxRelativeError(lhs, rhs), std::invalid_argument);
}

TEST_CASE(ReportComparisonWritesVerdictAndReturnsMatch) {
  const std::vector<float> lhs{1.0f, 2.0f};
  const std::vector<float> rhs{1.0f, 2.0f};
  std::ostringstream out;
  CHECK_TRUE(accel::ReportComparison(out, lhs, rhs, "lhs", "rhs"));
  CHECK_TRUE(out.str().find("[ok]") != std::string::npos);

  const std::vector<float> different{1.0f, 3.0f};
  std::ostringstream fail_out;
  CHECK_FALSE(accel::ReportComparison(fail_out, lhs, different, "lhs", "rhs"));
  CHECK_TRUE(fail_out.str().find("[FAIL]") != std::string::npos);
}

// Element counts must be computed in size_t: `4096 * 2048 * 2048` overflows a
// 32-bit int, and the MAC count is the term that grows fastest.
TEST_CASE(GemmShapeElementCountsAvoidIntOverflow) {
  const accel::GemmShape shape{4096, 2048, 4096};
  CHECK_EQ(shape.a_elements(), std::size_t{8388608});
  CHECK_EQ(shape.c_elements(), std::size_t{16777216});
  CHECK_EQ(shape.mac_count(), std::size_t{16777216} * 2048);
  CHECK_TRUE(shape.is_valid());
}

TEST_CASE(GemmGflopsGuardsAgainstNonPositiveTime) {
  const accel::GemmShape shape{2, 2, 2};
  CHECK_EQ(accel::GemmGflops(shape, 0.0), 0.0);
  CHECK_EQ(accel::GemmGflops(shape, -1.0), 0.0);
  CHECK_EQ(accel::GemmGflops(shape, std::numeric_limits<double>::quiet_NaN()),
           0.0);
  // 2*2*2 MACs = 8, 16 flops; over 16 ns that is exactly 1 GFLOP/s.
  CHECK_NEAR(accel::GemmGflops(shape, 16e-9), 1.0, 1e-9);
}

TEST_CASE(SpanConvertsToConstViewAndTracksExtent) {
  std::vector<float> values{1.0f, 2.0f, 3.0f};
  const accel::Span<float> mutable_view(values);
  const accel::Span<const float> const_view = mutable_view;

  CHECK_EQ(const_view.size(), std::size_t{3});
  CHECK_EQ(const_view.SizeBytes(), std::size_t{3} * sizeof(float));
  CHECK_FALSE(const_view.empty());
  CHECK_EQ(const_view[2], 3.0f);

  mutable_view[0] = 9.0f;
  CHECK_EQ(values[0], 9.0f);  // The view aliases, it does not copy.

  const accel::Span<float> empty_view;
  CHECK_TRUE(empty_view.empty());
  CHECK_EQ(empty_view.data(), static_cast<float*>(nullptr));
}

TEST_CASE(StopwatchMeasuresMonotonicNonNegativeIntervals) {
  const accel::Stopwatch watch;
  volatile double accumulator = 0.0;
  for (int i = 0; i < 100000; ++i) {
    accumulator = accumulator + i;
  }
  const double elapsed = watch.ElapsedSeconds();
  CHECK_TRUE(elapsed >= 0.0);
  // A steady clock must never run backwards between two reads.
  CHECK_TRUE(watch.ElapsedSeconds() >= elapsed);
}

int main() { return accel::test::RunAll(); }
