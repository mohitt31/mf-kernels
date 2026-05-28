// SPDX-License-Identifier: MIT
// bench/bench_sumfact.cpp — measure throughput of the kernel variants across
// polynomial degrees and emit a CSV.
//
// Usage:
//   ./bench_sumfact [--csv out.csv] [--target-seconds 0.5]
//
// Methodology
//   For each polynomial degree p ∈ {2..9} we benchmark each kernel by running
//   it auto_iters() times, where auto_iters picks a count so the total
//   measured interval is ≥ target_seconds. We repeat the entire measurement
//   five times and report the median GFLOPs/s — robust to occasional clock
//   noise. The reported FLOP count is the algorithm-independent one for the
//   *standard* algorithm; even-odd's actual FLOP savings show up as wall-time
//   speedup, not as inflated GFLOPs.
//
// What to inspect afterwards
//   - AVX2 vs naive ratio. With -O3 -march=native a good compiler will get
//     half to two-thirds of the gain by autovectorizing apply_1d_pitfall
//     anyway. Manual AVX2 should still beat it by 1.2–1.5× because of better
//     register scheduling.
//   - blocked vs avx2: 5–15% on degrees ≥ 4 if load pressure is the limit.
//   - evenodd vs avx2: 1.5–1.9× because the FLOP count itself halves.

#include <mfk/apply_1d.hpp>
#include <mfk/tensor_product.hpp>
#include <mfk/aligned_buffer.hpp>
#include <mfk/benchmark.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace mfk;

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

// Symmetric shape matrix builder, copied from the test code so this binary is
// self-contained.
void make_symmetric_S(double* S, int n_q, int n_d, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int q = 0; q < n_q; ++q)
    for (int i = 0; i < n_d; ++i) S[q * n_d + i] = dist(rng);
  for (int q = 0; q < n_q; ++q) {
    for (int i = 0; i < n_d; ++i) {
      const double s = 0.5 * (S[q * n_d + i] + S[(n_q - 1 - q) * n_d + (n_d - 1 - i)]);
      S[q * n_d + i] = s;
      S[(n_q - 1 - q) * n_d + (n_d - 1 - i)] = s;
    }
  }
}

#if defined(__AVX2__)
void build_evenodd_blocks(const double* S, double* Sp, double* Sm, int n_q, int n_d) {
  const int half = n_d / 2;
  for (int q = 0; q < n_q / 2; ++q) {
    for (int i = 0; i < half; ++i) {
      Sp[q * half + i] = 0.5 * (S[q * n_d + i] + S[q * n_d + (n_d - 1 - i)]);
      Sm[q * half + i] = 0.5 * (S[q * n_d + i] - S[q * n_d + (n_d - 1 - i)]);
    }
  }
}
#endif

struct Row {
  int    p;
  std::string variant;
  double seconds_per_call;
  double gflops_std;     // GFLOP/s by the standard FLOP count (= apples-to-apples wall-time view)
  double seconds_relative_to_naive;
};

// Bench the 1D kernel suite for one polynomial degree. n_spec is chosen large
// enough that the kernel runs for a measurable fraction of target_seconds in
// a single iter — this avoids the auto_iters probe being noisy.
std::vector<Row> bench_1d(int p, double target_seconds, std::mt19937_64& rng) {
  const int n_d = p + 1, n_q = p + 1;
  // Make total work substantial. n_spec controls how many spectator columns
  // each call processes. n_spec * n_d * lanes doubles must fit in L2/L3 for
  // the comparison to reflect compute, not DRAM bandwidth.
  const int n_spec = std::max(256, 2048 / n_d);   // ~16-32 KB working set

  auto S      = make_aligned<double>(static_cast<std::size_t>(n_q) * n_d);
  auto u      = make_aligned<double>(static_cast<std::size_t>(n_spec) * n_d * lanes);
  auto v      = make_aligned<double>(static_cast<std::size_t>(n_spec) * n_q * lanes);

  make_symmetric_S(S.get(), n_q, n_d, rng);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_spec) * n_d * lanes; ++i)
    u[i] = dist(rng);

  const double flops_per_call = standard_flops(n_q, n_d, n_spec);

  auto bench = [&](const char* name, auto&& call) {
    auto run = [&] { call(); };
    const int iters = auto_iters(run, target_seconds);
    std::vector<double> samples;
    for (int rep = 0; rep < 5; ++rep) {
      auto [s_per_iter, total] = time_iters(run, iters);
      samples.push_back(s_per_iter);
    }
    const double s = median(samples);
    return Row{p, name, s, flops_per_call / s * 1e-9, s};
  };

  std::vector<Row> rows;
  rows.push_back(bench("naive",
    [&]{ apply_1d_naive(S.get(), u.get(), v.get(), n_q, n_d, n_spec); }));
  rows.push_back(bench("pitfall",
    [&]{ apply_1d_pitfall(S.get(), u.get(), v.get(), n_q, n_d, n_spec); }));
#if defined(__AVX2__)
  rows.push_back(bench("avx2",
    [&]{ apply_1d_avx2(S.get(), u.get(), v.get(), n_q, n_d, n_spec); }));
  rows.push_back(bench("avx2_blocked",
    [&]{ apply_1d_avx2_blocked(S.get(), u.get(), v.get(), n_q, n_d, n_spec); }));
  if (n_d == n_q && n_d % 2 == 0) {
    const int half = n_d / 2;
    auto Sp = make_aligned<double>(static_cast<std::size_t>(half) * half);
    auto Sm = make_aligned<double>(static_cast<std::size_t>(half) * half);
    build_evenodd_blocks(S.get(), Sp.get(), Sm.get(), n_q, n_d);
    rows.push_back(bench("evenodd_avx2",
      [&]{ apply_1d_evenodd_avx2(Sp.get(), Sm.get(), u.get(), v.get(),
                                 n_q, n_d, n_spec); }));
  }
#endif

  const double t_naive = rows.front().seconds_per_call;
  for (auto& r : rows) r.seconds_relative_to_naive = r.seconds_per_call / t_naive;
  return rows;
}

} // namespace

int main(int argc, char** argv) {
  const char* csv_path        = nullptr;
  double      target_seconds  = 0.4;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
    else if (std::strcmp(argv[i], "--target-seconds") == 0 && i + 1 < argc)
      target_seconds = std::atof(argv[++i]);
  }

  std::mt19937_64 rng(0xc0ffeeULL);
  std::vector<Row> all_rows;

  std::printf("%-3s %-15s %12s %12s %10s\n", "p", "variant",
              "s/call", "GFLOP/s", "rel_naive");
  std::printf("%-3s %-15s %12s %12s %10s\n", "---", "---------------",
              "------------", "------------", "----------");

  for (int p : {2, 3, 4, 5, 6, 7, 8, 9}) {
    auto rows = bench_1d(p, target_seconds, rng);
    for (const auto& r : rows) {
      std::printf("%-3d %-15s %12.4e %12.3f %10.2fx\n",
                  r.p, r.variant.c_str(),
                  r.seconds_per_call, r.gflops_std, 1.0 / r.seconds_relative_to_naive);
      all_rows.push_back(r);
    }
    std::printf("\n");
  }

  if (csv_path) {
    std::ofstream f(csv_path);
    f << "p,variant,seconds_per_call,gflops,speedup_vs_naive\n";
    for (const auto& r : all_rows) {
      f << r.p << "," << r.variant << ","
        << r.seconds_per_call << "," << r.gflops_std << ","
        << (1.0 / r.seconds_relative_to_naive) << "\n";
    }
    std::printf("Wrote %s\n", csv_path);
  }
  return 0;
}
