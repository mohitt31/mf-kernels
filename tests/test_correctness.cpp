// SPDX-License-Identifier: MIT
// tests/test_correctness.cpp — verify that all kernel variants agree
// with the naive scalar reference up to floating-point round-off.
//
// We sweep polynomial degrees and seed inputs with deterministic PRNG so the
// test is reproducible. Tolerance is set to ~10x machine epsilon scaled by
// the problem size; sum factorization accumulates O(n_d) products per output,
// and the floating-point order differs between scalar and SIMD, so bit-exact
// equality is not expected.

#include <mfk/apply_1d.hpp>
#include <mfk/tensor_product.hpp>
#include <mfk/aligned_buffer.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace mfk;

namespace {

void fill_random(double* p, std::size_t n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// Build a 1D shape matrix that is symmetric in the sense required by the
// even-odd kernel: S(q, i) == S(n_q-1-q, n_d-1-i). One easy way to construct
// such a matrix is to start from any matrix A and symmetrize:
//   S(q, i) = 0.5 * (A(q, i) + A(n_q-1-q, n_d-1-i)).
void make_symmetric_S(double* S, int n_q, int n_d, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int q = 0; q < n_q; ++q)
    for (int i = 0; i < n_d; ++i)
      S[q * n_d + i] = dist(rng);
  // symmetrize in-place
  for (int q = 0; q < n_q; ++q) {
    for (int i = 0; i < n_d; ++i) {
      const double a = S[q * n_d + i];
      const double b = S[(n_q - 1 - q) * n_d + (n_d - 1 - i)];
      const double s = 0.5 * (a + b);
      S[q * n_d + i] = s;
      S[(n_q - 1 - q) * n_d + (n_d - 1 - i)] = s;
    }
  }
}

#if defined(__AVX2__)
void build_evenodd_blocks(const double* S,
                          double* S_plus, double* S_minus,
                          int n_q, int n_d) {
  const int half_q = n_q / 2;
  const int half_d = n_d / 2;
  for (int q = 0; q < half_q; ++q) {
    for (int i = 0; i < half_d; ++i) {
      S_plus [q * half_d + i] = 0.5 * (S[q * n_d + i] + S[q * n_d + (n_d - 1 - i)]);
      S_minus[q * half_d + i] = 0.5 * (S[q * n_d + i] - S[q * n_d + (n_d - 1 - i)]);
    }
  }
}
#endif

double max_abs_diff(const double* a, const double* b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

bool test_one_degree(int p, std::mt19937_64& rng) {
  const int n_d   = p + 1;
  const int n_q   = p + 1;
  const int n_spec = 7;        // arbitrary > 1 to exercise the outer loop
  const std::size_t u_size = static_cast<std::size_t>(n_spec) * n_d * lanes;
  const std::size_t v_size = static_cast<std::size_t>(n_spec) * n_q * lanes;
  const std::size_t S_size = static_cast<std::size_t>(n_q) * n_d;

  auto S      = make_aligned<double>(S_size);
  auto u      = make_aligned<double>(u_size);
  auto v_ref  = make_aligned<double>(v_size);
  auto v_auto = make_aligned<double>(v_size);
#if defined(__AVX2__)
  auto v_avx  = make_aligned<double>(v_size);
  auto v_blk  = make_aligned<double>(v_size);
  auto v_eo   = make_aligned<double>(v_size);
#endif

  make_symmetric_S(S.get(), n_q, n_d, rng);
  fill_random(u.get(), u_size, rng);

  apply_1d_naive  (S.get(), u.get(), v_ref.get(),  n_q, n_d, n_spec);
  apply_1d_pitfall(S.get(), u.get(), v_auto.get(), n_q, n_d, n_spec);

  const double tol = 1e-10 * n_d;
  const double d_auto = max_abs_diff(v_ref.get(), v_auto.get(), v_size);
  if (d_auto > tol) {
    std::printf("  p=%d autovec FAIL  max|Δ|=%.3e\n", p, d_auto);
    return false;
  }

#if defined(__AVX2__)
  apply_1d_avx2        (S.get(), u.get(), v_avx.get(), n_q, n_d, n_spec);
  apply_1d_avx2_blocked(S.get(), u.get(), v_blk.get(), n_q, n_d, n_spec);

  const double d_avx = max_abs_diff(v_ref.get(), v_avx.get(), v_size);
  const double d_blk = max_abs_diff(v_ref.get(), v_blk.get(), v_size);
  if (d_avx > tol) { std::printf("  p=%d avx2     FAIL  max|Δ|=%.3e\n", p, d_avx); return false; }
  if (d_blk > tol) { std::printf("  p=%d blocked  FAIL  max|Δ|=%.3e\n", p, d_blk); return false; }

  // Even-odd is only defined for even (n_d == n_q) here.
  if (n_d == n_q && n_d % 2 == 0) {
    const int half = n_d / 2;
    auto S_plus  = make_aligned<double>(static_cast<std::size_t>(half) * half);
    auto S_minus = make_aligned<double>(static_cast<std::size_t>(half) * half);
    build_evenodd_blocks(S.get(), S_plus.get(), S_minus.get(), n_q, n_d);
    apply_1d_evenodd_avx2(S_plus.get(), S_minus.get(), u.get(), v_eo.get(),
                          n_q, n_d, n_spec);
    const double d_eo = max_abs_diff(v_ref.get(), v_eo.get(), v_size);
    if (d_eo > tol) { std::printf("  p=%d evenodd  FAIL  max|Δ|=%.3e\n", p, d_eo); return false; }
    std::printf("  p=%d  OK  (autovec %.1e | avx %.1e | blocked %.1e | evenodd %.1e)\n",
                p, d_auto, d_avx, d_blk, d_eo);
  } else {
    std::printf("  p=%d  OK  (autovec %.1e | avx %.1e | blocked %.1e | evenodd N/A)\n",
                p, d_auto, d_avx, d_blk);
  }
#else
  std::printf("  p=%d  OK  (autovec %.1e | AVX2 not compiled in)\n", p, d_auto);
#endif

  return true;
}

bool test_3d(int p, std::mt19937_64& rng) {
  const int n_d = p + 1, n_q = p + 1;
  const std::size_t u_size  = static_cast<std::size_t>(n_d) * n_d * n_d * lanes;
  const std::size_t v_size  = static_cast<std::size_t>(n_q) * n_q * n_q * lanes;
  const std::size_t tmp_max = static_cast<std::size_t>(n_q) * n_q * n_d * lanes;

  auto S        = make_aligned<double>(static_cast<std::size_t>(n_q) * n_d);
  auto u        = make_aligned<double>(u_size);
  auto v_ref    = make_aligned<double>(v_size);
  auto tmp_a    = make_aligned<double>(tmp_max);
  auto tmp_b    = make_aligned<double>(tmp_max);
#if defined(__AVX2__)
  auto v_avx    = make_aligned<double>(v_size);
  auto tmp_a2   = make_aligned<double>(tmp_max);
  auto tmp_b2   = make_aligned<double>(tmp_max);
#endif

  make_symmetric_S(S.get(), n_q, n_d, rng);
  fill_random(u.get(), u_size, rng);

  apply_3d(&apply_1d_naive, S.get(), u.get(), v_ref.get(),
           tmp_a.get(), tmp_b.get(), n_q, n_d);
#if defined(__AVX2__)
  apply_3d(&apply_1d_avx2,  S.get(), u.get(), v_avx.get(),
           tmp_a2.get(), tmp_b2.get(), n_q, n_d);
  const double d = max_abs_diff(v_ref.get(), v_avx.get(), v_size);
  // Three passes of accumulation, so tolerance grows with n_d^3-ish.
  const double tol = 1e-9 * n_d * n_d * n_d;
  if (d > tol) { std::printf("  3D p=%d FAIL  max|Δ|=%.3e (tol %.1e)\n", p, d, tol); return false; }
  std::printf("  3D p=%d  OK  (max|Δ|=%.1e, tol %.1e)\n", p, d, tol);
#else
  std::printf("  3D p=%d skipped (AVX2 disabled)\n", p);
#endif
  return true;
}

} // namespace

int main() {
  std::printf("=== mf-kernels correctness ===\n");

  std::mt19937_64 rng(0xc0ffeeULL);

  std::printf("\n[1D kernels]\n");
  bool ok = true;
  for (int p : {1, 2, 3, 4, 5, 7, 9}) ok &= test_one_degree(p, rng);

  std::printf("\n[3D tensor product]\n");
  for (int p : {1, 2, 3, 5}) ok &= test_3d(p, rng);

  std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "*** FAILURES ***");
  return ok ? 0 : 1;
}
