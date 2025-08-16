// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_OMP_GEMM_OMP_H_
#define ACCEL_OMP_GEMM_OMP_H_

#include "core/gemm_shape.h"
#include "core/span.h"
#include "cpu/cpu_gemm.h"

namespace accel {

// Parallel counterparts of the two fastest serial kernels. Both split the M
// axis, so each thread owns a disjoint set of C rows: the accumulating
// contract survives with no atomics and no reduction over C, and results stay
// bit-identical to the serial kernel at any thread count.
//
// Only the fastest serial kernels are parallelised. Threading GemmIjk would
// report a speedup that is mostly the loop reorder it lacks.
//
// Contract as in cpu_gemm.h. Validation runs in the serial region, before any
// thread is spawned, because an exception cannot cross a parallel boundary.
// Thread count is whatever the runtime supplies; pin it with OpenmpSetThreads.

// GemmIkj with M distributed. Each thread still walks unit-stride B and C, so
// per-thread vectorisation is unaffected. Expect the curve to bend well before
// the core count: the kernel streams all of B per row block, so it saturates
// bandwidth rather than issue width.
void GemmIkjOmp(Span<const float> a, Span<const float> b, Span<float> c,
                const GemmShape& shape);

// GemmTiledBColMajor distributed over row *blocks*, so each thread keeps a
// whole blocked working set resident. Answers whether blocking still pays when
// threads contend for a shared cache -- the serial answer need not survive.
// A non-positive tile_size throws std::invalid_argument.
void GemmTiledBColMajorOmp(Span<const float> a, Span<const float> b_col,
                           Span<float> c, const GemmShape& shape,
                           int tile_size = kDefaultTileSize);

}  // namespace accel

#endif  // ACCEL_OMP_GEMM_OMP_H_
