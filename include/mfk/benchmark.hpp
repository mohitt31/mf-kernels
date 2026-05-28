// SPDX-License-Identifier: MIT
// mfk/benchmark.hpp — small timing utility.
//
// std::chrono::steady_clock is monotonic and high-resolution on Linux (clock
// source CLOCK_MONOTONIC). For benchmark resolution at the microsecond level
// it is fine; for tighter measurements we repeat the kernel N times and
// divide.

#pragma once

#include <chrono>
#include <cstdint>
#include <utility>

namespace mfk {

class Timer {
 public:
  using clock      = std::chrono::steady_clock;
  using time_point = clock::time_point;

  void start() { t0_ = clock::now(); }
  void stop()  { t1_ = clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(t1_ - t0_).count();
  }

 private:
  time_point t0_, t1_;
};

// Measure a callable F's wall time over n_iters iterations. Returns
// {seconds_per_iter, total_seconds}. F should take no arguments and not
// be optimized away; ensure your kernel touches a result the compiler
// can't prove is unused.
template <typename F>
inline std::pair<double, double> time_iters(F&& f, int n_iters) {
  Timer t;
  t.start();
  for (int i = 0; i < n_iters; ++i) f();
  t.stop();
  return {t.seconds() / n_iters, t.seconds()};
}

// Number of iterations for a target measurement duration. Probes with one
// iter, scales up. Useful so cheap kernels are timed for long enough to
// dwarf clock resolution noise.
template <typename F>
inline int auto_iters(F&& f, double target_seconds = 0.5) {
  Timer t;
  t.start();
  f();
  t.stop();
  const double one = t.seconds();
  if (one >= target_seconds) return 1;
  return static_cast<int>(target_seconds / one) + 1;
}

} // namespace mfk
