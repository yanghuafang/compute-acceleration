// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Correctness tests for the serial host sum reductions.
//
// The interesting cases are the ones where lane blocking and a plain loop can
// disagree: counts that are not multiples of kSumLanes, counts below one full
// lane group, and inputs whose magnitudes make the summation order visible.
#include "cpu/reduction.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "tests/support/reference_gemm.h"
#include "tests/support/test_harness.h"

namespace {

using accel::SumBlocked;
using accel::SumSequential;

// Ascending Ramp with an exactly representable closed form, so the expected
// value is arithmetic rather than another implementation of the same loop.
std::vector<float> Ramp(std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>(i + 1);
  }
  return values;
}

double RampTotal(std::size_t count) {
  const auto n = static_cast<double>(count);
  return n * (n + 1.0) / 2.0;
}

}  // namespace

TEST_CASE(SequentialSumsARamp) {
  CHECK_NEAR(SumSequential(Ramp(1000)), RampTotal(1000), 1e-9);
}

TEST_CASE(BlockedSumsARamp) {
  CHECK_NEAR(SumBlocked(Ramp(1000)), RampTotal(1000), 1e-9);
}

TEST_CASE(BlockedHandlesCountsThatAreNotLaneMultiples) {
  // 8 lanes, so 0 and 7 trailing elements bracket the tail loop, and 5 is a
  // count with no full lane group at all.
  for (const std::size_t count :
       {std::size_t{5}, std::size_t{8}, std::size_t{15}, std::size_t{16},
        std::size_t{4097}}) {
    CHECK_NEAR(SumBlocked(Ramp(count)), RampTotal(count), 1e-9);
  }
}

TEST_CASE(BothVariantsAgreeOnAnEmptySpan) {
  const std::vector<float> empty;
  CHECK_EQ(SumSequential(empty), 0.0);
  CHECK_EQ(SumBlocked(empty), 0.0);
}

TEST_CASE(BothVariantsAgreeOnASingleElement) {
  const std::vector<float> one{3.5f};
  CHECK_EQ(SumSequential(one), 3.5);
  CHECK_EQ(SumBlocked(one), 3.5);
}

TEST_CASE(BlockedIsReproducibleAcrossCalls) {
  // Lane assignment is fixed, so repeated calls must agree bit for bit. This
  // is the property SumOmp() deliberately does not have.
  const std::vector<float> values =
      accel::test::PseudoRandomVector(9973, 4242u);
  CHECK_EQ(SumBlocked(values), SumBlocked(values));
}

TEST_CASE(VariantsAgreeWithinToleranceOnRandomInput) {
  const std::vector<float> values =
      accel::test::PseudoRandomVector(65537, 777u);
  const double sequential = SumSequential(values);
  const double blocked = SumBlocked(values);
  // Same terms, different association: the gap is rounding, not disagreement.
  CHECK_TRUE(std::fabs(sequential - blocked) <=
             1e-9 * std::fabs(sequential) + 1e-9);
}

int main() { return accel::test::RunAll(); }
