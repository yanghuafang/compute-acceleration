// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#include "omp/omp_control.h"

#include <stdexcept>
#include <string>

#ifndef _OPENMP
#error "src/omp requires OpenMP; link the target against OpenMP::OpenMP_CXX"
#endif

#include <omp.h>

namespace accel {

int OpenmpMaxThreads() noexcept { return omp_get_max_threads(); }

void OpenmpSetThreads(int count) {
  if (count <= 0) {
    throw std::invalid_argument("thread count must be positive, got " +
                                std::to_string(count));
  }
  omp_set_num_threads(count);
}

int OpenmpThreadsUsed() noexcept {
  int observed = 0;
  // Written from a single thread rather than reduced: every thread in the
  // region would store the same value, but only thread 0's store is ordered
  // with respect to the read after the region.
#pragma omp parallel default(none) shared(observed)
  {
#pragma omp master
    observed = omp_get_num_threads();
  }
  return observed;
}

}  // namespace accel
