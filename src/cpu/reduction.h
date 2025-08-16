// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CPU_REDUCTION_H_
#define ACCEL_CPU_REDUCTION_H_

#include "core/span.h"

namespace accel {

// Eight double lanes fill two 256-bit vector registers, the widest shape every
// target here sustains without spilling.
inline constexpr int kSumLanes = 8;

// Both return double, matching SumOnDevice so the four implementations of this
// task compare directly. A float host accumulator would be less accurate than
// the device kernel it is meant to check.
//
// Addition is not associative, so these agree to a relative tolerance, not
// exactly. An empty span sums to zero rather than being an error.

// Baseline: one accumulator, so one dependency chain of length N. Throughput
// is bounded by add latency, not by memory or issue width.
double SumSequential(Span<const float> input) noexcept;

// Same operations and memory traffic, no threads -- only the single dependency
// chain becomes kSumLanes independent ones, which the pipeline overlaps and the
// vectoriser widens. Isolates instruction-level parallelism from the
// thread-level parallelism SumOmp adds. Lane assignment is fixed, so unlike
// SumOmp this is bit-for-bit reproducible.
double SumBlocked(Span<const float> input) noexcept;

}  // namespace accel

#endif  // ACCEL_CPU_REDUCTION_H_
