// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

// Correctness and contract tests for the five CPU GEMM variants.
//
// Shapes are deliberately small, non-square and not multiples of the tile size,
// so tail handling and any m/n/k transposition is exercised. Small also means
// fast under ASan and UBSan, which is the point of having these at all.
#include "cpu/cpu_gemm.h"

#include <stdexcept>
#include <vector>

#include "core/gemm_shape.h"
#include "core/matrix_utils.h"
#include "tests/support/reference_gemm.h"
#include "tests/support/test_harness.h"

namespace {

using accel::GemmShape;

// Prime-ish extents, none a multiple of the default tile size of 64.
constexpr GemmShape kShape{37, 53, 41};

// Loose enough for float accumulation over k terms in three different orders,
// tight enough that a genuinely wrong kernel cannot slip through.
constexpr float kTolerance = 1e-4f;

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

TEST_CASE(GemmIjkMatchesReference) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmIjk(operands.a, operands.b, c, kShape);
  CHECK_NEAR(accel::MaxRelativeError(c, operands.expected), 0.0, kTolerance);
}

TEST_CASE(GemmIkjMatchesReference) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmIkj(operands.a, operands.b, c, kShape);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

TEST_CASE(GemmTiledMatchesReference) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmTiled(operands.a, operands.b, c, kShape);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

TEST_CASE(GemmIjkBColMajorMatchesReference) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmIjkBColMajor(operands.a, operands.b_col, c, kShape);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

TEST_CASE(GemmTiledBColMajorMatchesReference) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmTiledBColMajor(operands.a, operands.b_col, c, kShape);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

// A tile wider than every extent collapses the blocked kernel to a single
// clamped block — the boundary case where an off-by-one in the tail bounds
// shows up immediately.
TEST_CASE(GemmTiledHandlesTileLargerThanExtents) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmTiled(operands.a, operands.b, c, kShape, 1024);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

// A tile of one degenerates to the naive nest and must still be correct.
TEST_CASE(GemmTiledHandlesUnitTile) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmTiled(operands.a, operands.b, c, kShape, 1);
  CHECK_TRUE(accel::MatricesClose(c, operands.expected, kTolerance));
}

// The documented `C += A * B` semantics: running twice must double the result.
// Benchmark drivers depend on this, so it is part of the contract, not an
// implementation detail.
TEST_CASE(GemmAccumulatesIntoC) {
  const Operands operands = MakeOperands(kShape);
  std::vector<float> c(kShape.c_elements(), 0.0f);
  accel::GemmIjk(operands.a, operands.b, c, kShape);
  accel::GemmIjk(operands.a, operands.b, c, kShape);

  std::vector<float> doubled = operands.expected;
  for (float& value : doubled) {
    value *= 2.0f;
  }
  CHECK_TRUE(accel::MatricesClose(c, doubled, kTolerance));
}

TEST_CASE(GemmRejectsDegenerateShape) {
  const std::vector<float> operand;
  std::vector<float> output;
  CHECK_THROWS_AS(accel::GemmIjk(operand, operand, output, GemmShape{0, 1, 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(accel::GemmIjk(operand, operand, output, GemmShape{1, -1, 1}),
                  std::invalid_argument);
}

TEST_CASE(GemmRejectsMismatchedOperandSizes) {
  const GemmShape shape{2, 3, 4};
  std::vector<float> a(shape.a_elements(), 1.0f);
  std::vector<float> b(shape.b_elements(), 1.0f);
  std::vector<float> c(shape.c_elements(), 0.0f);

  const std::vector<float> short_a(shape.a_elements() - 1, 1.0f);
  CHECK_THROWS_AS(accel::GemmIjk(short_a, b, c, shape), std::invalid_argument);

  const std::vector<float> short_b(shape.b_elements() - 1, 1.0f);
  CHECK_THROWS_AS(accel::GemmIjk(a, short_b, c, shape), std::invalid_argument);

  std::vector<float> short_c(shape.c_elements() - 1, 0.0f);
  CHECK_THROWS_AS(accel::GemmIjk(a, b, short_c, shape), std::invalid_argument);
}

TEST_CASE(GemmTiledRejectsNonPositiveTile) {
  const GemmShape shape{2, 3, 4};
  const std::vector<float> a(shape.a_elements(), 1.0f);
  const std::vector<float> b(shape.b_elements(), 1.0f);
  std::vector<float> c(shape.c_elements(), 0.0f);

  CHECK_THROWS_AS(accel::GemmTiled(a, b, c, shape, 0), std::invalid_argument);
  CHECK_THROWS_AS(accel::GemmTiledBColMajor(a, b, c, shape, -8),
                  std::invalid_argument);
}

// Rectangular shapes catch the classic m/n swap, which a square test cannot.
TEST_CASE(GemmHandlesExtremeAspectRatios) {
  const GemmShape wide{1, 7, 96};
  const Operands wide_operands = MakeOperands(wide);
  std::vector<float> c_wide(wide.c_elements(), 0.0f);
  accel::GemmIkj(wide_operands.a, wide_operands.b, c_wide, wide);
  CHECK_TRUE(accel::MatricesClose(c_wide, wide_operands.expected, kTolerance));

  const GemmShape tall{96, 7, 1};
  const Operands tall_operands = MakeOperands(tall);
  std::vector<float> c_tall(tall.c_elements(), 0.0f);
  accel::GemmTiled(tall_operands.a, tall_operands.b, c_tall, tall);
  CHECK_TRUE(accel::MatricesClose(c_tall, tall_operands.expected, kTolerance));
}

int main() { return accel::test::RunAll(); }
