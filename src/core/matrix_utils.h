// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CORE_MATRIX_UTILS_H_
#define ACCEL_CORE_MATRIX_UTILS_H_

#include <iosfwd>
#include <string_view>

#include "core/span.h"

namespace accel {

// A[i] = i over an 8M-element operand drives partial sums past 1e17, where
// float spacing exceeds 1e10 and every comparison degenerates into noise.
// Scaling keeps accumulated magnitudes near unity so relative error means
// something.
inline constexpr float kDefaultRampScale = 1e-6f;

// For two float results that summed the same terms in a different order over a
// 2048-long contraction.
inline constexpr float kDefaultTolerance = 1e-5f;

// Fills with data[i] = i * scale. Deterministic so that a mismatch between
// runs indicates a kernel bug, not a new seed.
void FillRamp(Span<float> data, float scale = kDefaultRampScale) noexcept;

// Transposes a row-major K x N matrix, b_col[j * k + i] == b[i * n + j],
// turning the naive kernel's strided column walk into a unit-stride read --
// worth 4.9x to 8.5x depending on the machine, more than blocking buys
// anywhere. The drivers time the transpose separately so it is not hidden.
//
// Ranges must be distinct and non-overlapping. Throws std::invalid_argument on
// non-positive extents or a span whose length is not k * n.
void RowToColumnMajor(Span<const float> b, Span<float> b_col, int k, int n);

// Largest element-wise relative difference, normalised by max(|lhs|, |rhs|),
// or infinity if either input contains a NaN.
//
// An exactly-zero pair counts as zero error rather than dividing by zero. That
// guard is load-bearing: the unguarded quotient is NaN, NaN > tolerance is
// false, and the comparison would report a match it never made.
//
// Throws std::invalid_argument if the spans differ in length.
float MaxRelativeError(Span<const float> lhs, Span<const float> rhs);

bool MatricesClose(Span<const float> lhs, Span<const float> rhs,
                   float tolerance = kDefaultTolerance);

// Writes a one-line verdict to `os` and returns whether it matched, for
// drivers that report rather than abort: a mismatch is logged with its error
// so the run still produces timings. Propagate the result to the exit code.
bool ReportComparison(std::ostream& os, Span<const float> lhs,
                      Span<const float> rhs, std::string_view lhs_name,
                      std::string_view rhs_name,
                      float tolerance = kDefaultTolerance);

}  // namespace accel

#endif  // ACCEL_CORE_MATRIX_UTILS_H_
