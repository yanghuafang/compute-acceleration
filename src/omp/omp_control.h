// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_OMP_OMP_CONTROL_H_
#define ACCEL_OMP_OMP_CONTROL_H_

namespace accel {

// Lets drivers and tests pin and report the thread count without including
// <omp.h>, which would push the OpenMP compile flags onto every consumer.
//
// Call from the serial region only: these set a process-wide control variable,
// and changing it inside a parallel region is undefined.

// Threads the runtime would use next, honouring OMP_NUM_THREADS and any prior
// OpenmpSetThreads call.
int OpenmpMaxThreads() noexcept;

// Pins subsequent regions to `count` threads; throws std::invalid_argument if
// it is not positive. Benchmarks must pin: an unpinned run is not
// reproducible, and on a hybrid CPU the default spans performance and
// efficiency cores.
void OpenmpSetThreads(int count);

// Threads a region actually receives, measured by entering one. The runtime
// may supply fewer than OpenmpMaxThreads promises, and reporting the requested
// count is how a scaling table claims speedups on threads that never ran.
int OpenmpThreadsUsed() noexcept;

}  // namespace accel

#endif  // ACCEL_OMP_OMP_CONTROL_H_
