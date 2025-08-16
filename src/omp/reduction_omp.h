// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_OMP_REDUCTION_OMP_H_
#define ACCEL_OMP_REDUCTION_OMP_H_

#include "core/span.h"

namespace accel {

// Thread-parallel sum via an OpenMP `reduction(+ : ...)` clause.
//
// Each thread accumulates a private partial over a static slice, and the
// runtime combines them at the end of the region. That is the same two-stage
// shape the CUDA kernel uses — private accumulation, then a tree combine —
// with the runtime supplying the tree.
//
// Thread-safety: safe to call concurrently, but see OpenmpSetThreads()
// regarding nested regions; this function is intended to be entered from the
// serial region.
//
// The combine order is chosen by the runtime and is **not** fixed across runs
// or thread counts. Two calls on identical input may differ in the last bits.
// Compare against a relative tolerance; SumBlocked() is the variant to use
// when reproducibility matters.
//
// Precondition: A non-null pointer whenever the span is non-empty.
//
// Postcondition: Returns the total; the operand is not modified.
double SumOmp(Span<const float> input) noexcept;

}  // namespace accel

#endif  // ACCEL_OMP_REDUCTION_OMP_H_
