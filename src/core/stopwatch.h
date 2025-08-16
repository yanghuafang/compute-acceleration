// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CORE_STOPWATCH_H_
#define ACCEL_CORE_STOPWATCH_H_

#include <chrono>
#include <utility>

namespace accel {

// Monotonic wall-Clock timer for host-side measurements.
//
// Wraps `std::chrono::steady_clock` rather than `high_resolution_clock`: the
// latter is an alias for `system_clock` in several standard libraries and is
// therefore subject to NTP steps mid-measurement, which can yield negative
// durations. Resolution is nanoseconds on every platform this project targets.
//
// Durations are reported as fractional seconds rather than truncated with
// `duration_cast<seconds>`, which would print "1 seconds" for a 1.9 s kernel
// and collapse every sub-second measurement to zero.
class Stopwatch {
 public:
  using Clock = std::chrono::steady_clock;

  // Starts counting immediately on construction.
  Stopwatch() noexcept : start_(Clock::now()) {}

  // Rebases the origin to now, discarding the accumulated interval.
  void Reset() noexcept { start_ = Clock::now(); }

  double ElapsedSeconds() const noexcept {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

  double ElapsedMillis() const noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_)
        .count();
  }

 private:
  Clock::time_point start_;
};

// Invokes `fn` once and returns its wall-Clock duration in seconds.
//
// Exception-safety: no guarantee is added — an exception escaping `fn`
// propagates and no timing is reported. Callers timing fallible work should
// measure with a Stopwatch inside their own try block.
//
// Performs no warm-up and no repetition; for short kernels prefer an explicit
// loop so that timer overhead is amortised.
template <typename Fn>
double TimeSeconds(Fn&& fn) {
  const Stopwatch watch;
  std::forward<Fn>(fn)();
  return watch.ElapsedSeconds();
}

}  // namespace accel

#endif  // ACCEL_CORE_STOPWATCH_H_
