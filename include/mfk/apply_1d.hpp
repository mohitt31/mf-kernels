// SPDX-License-Identifier: MIT
// mfk/apply_1d.hpp — 1D shape-matrix application kernels for matrix-free
// tensor-product finite element methods.
//
// These implement the inner operation of sum factorization. Given a 1D shape
// matrix S of shape (n_q × n_d) — values of n_d Lagrange basis functions at
// n_q quadrature points — and an input field u indexed by (i, k) where i ∈
// [0, n_d) is the dof index along the active 1D direction and k ∈ [0, n_spec)
// is a flattened index over the *spectator* dimensions, compute
//
//     v(q, k) = Σ_{i=0}^{n_d-1} S(q, i) · u(i, k)        for all (q, k).
//
// Sum factorization applies this kernel three times (in 3D) to evaluate the
// full tensor-product basis at the volume quadrature points. The same kernel,
// in transposed form, handles integration. The cost per cell is O(d · p^(d+1))
// vs O(p^(2d)) for naive evaluation — for Q5 elements (p=5) in 3D that's an 8.3×
// reduction in FLOPs.
//
// All implementations here process *4 cells at once* in AVX2 lanes, mirroring
// the SoA-of-AoS-of-4 layout that deal.II uses internally for double-precision
// VectorizedArray. The shape matrix S itself is scalar (one per problem); it
// is broadcast across lanes.
//
// Layout convention (column-major in the n_d/n_q index, contiguous in lane):
//   u[k * n_d * 4 + i * 4 + lane]   →  load with _mm256_load_pd(u + k*n_d*4 + i*4)
//   v[k * n_q * 4 + q * 4 + lane]
//   S[q * n_d + i]                  →  row-major, lane-independent
//
// References
//   Kronbichler & Kormann, ACM TOMS 38(2), 2011 — generic interface
//   Kronbichler & Kormann, Computers & Fluids 195, 2019 — fast matrix-free DG
//   Kopriva, Implementing Spectral Methods for PDEs, Springer 2009 — even-odd

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define MFK_ALWAYS_INLINE inline __attribute__((always_inline))
  #define MFK_RESTRICT __restrict__
#else
  #define MFK_ALWAYS_INLINE inline
  #define MFK_RESTRICT
#endif

namespace mfk {

inline constexpr int lanes = 4; // AVX2 doubles per register

// -----------------------------------------------------------------------------
// 1. Naive scalar reference. Used to validate correctness of the fancy versions.
//    No SIMD intent — clarity first.
// -----------------------------------------------------------------------------
MFK_ALWAYS_INLINE
void apply_1d_naive(const double* MFK_RESTRICT S,
                    const double* MFK_RESTRICT u,
                    double*       MFK_RESTRICT v,
                    int n_q, int n_d, int n_spec)
{
  for (int k = 0; k < n_spec; ++k) {
    for (int q = 0; q < n_q; ++q) {
      for (int lane = 0; lane < lanes; ++lane) {
        double acc = 0.0;
        for (int i = 0; i < n_d; ++i)
          acc += S[q * n_d + i] * u[(k * n_d + i) * lanes + lane];
        v[(k * n_q + q) * lanes + lane] = acc;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 2. Common-pitfall variant — explicit lane unrolling with a stack array.
//    Looks reasonable on paper. Don't use it.
//
//    The intent here was "help the compiler vectorize by making the 4-lane
//    structure explicit." On gcc 13 with -O3 -march=native it backfires:
//    inspecting the generated code with `g++ -S` shows gcc switching to a
//    SoA strategy where it loads 4 shape-matrix entries at once with vmovupd
//    and then uses `vpermpd` (latency 3, throughput 1) to broadcast each
//    individually. The naive variant (#1 above) compiles down to direct
//    `vbroadcastsd` (latency 1) and is consistently faster by a factor of
//    3–4× as a result.
//
//    This is included so the benchmark exposes the trap rather than hiding
//    it; the "obvious" SIMD hint is the worst-performing option in this
//    codebase. The lesson is one Kronbichler (deal.II's matrix-free author)
//    has emphasized: respect the compiler's preferred lowering, then beat
//    it with structure it can't infer (register blocking, algorithmic
//    transforms like even-odd) rather than fighting it at the SIMD level.
// -----------------------------------------------------------------------------
MFK_ALWAYS_INLINE
void apply_1d_pitfall(const double* MFK_RESTRICT S,
                      const double* MFK_RESTRICT u,
                      double*       MFK_RESTRICT v,
                      int n_q, int n_d, int n_spec)
{
  for (int k = 0; k < n_spec; ++k) {
    const double* u_k = u + k * n_d * lanes;
    double*       v_k = v + k * n_q * lanes;
    for (int q = 0; q < n_q; ++q) {
      double acc[lanes] = {0.0, 0.0, 0.0, 0.0};
      const double* S_q = S + q * n_d;
      for (int i = 0; i < n_d; ++i) {
        const double s = S_q[i];
        for (int lane = 0; lane < lanes; ++lane)
          acc[lane] += s * u_k[i * lanes + lane];
      }
      for (int lane = 0; lane < lanes; ++lane)
        v_k[q * lanes + lane] = acc[lane];
    }
  }
}

#if defined(__AVX2__)

// -----------------------------------------------------------------------------
// 3. Explicit AVX2. One AVX2 register holds all 4 lanes; S entries are
//    broadcast. The hot path is one vbroadcastsd + vfmadd231pd per (q, i) pair.
//    This is the canonical SIMD pattern for matrix-free FEM kernels.
// -----------------------------------------------------------------------------
MFK_ALWAYS_INLINE
void apply_1d_avx2(const double* MFK_RESTRICT S,
                   const double* MFK_RESTRICT u,
                   double*       MFK_RESTRICT v,
                   int n_q, int n_d, int n_spec)
{
  for (int k = 0; k < n_spec; ++k) {
    const double* u_k = u + k * n_d * lanes;
    double*       v_k = v + k * n_q * lanes;
    for (int q = 0; q < n_q; ++q) {
      __m256d acc = _mm256_setzero_pd();
      const double* S_q = S + q * n_d;
      for (int i = 0; i < n_d; ++i) {
        const __m256d s    = _mm256_broadcast_sd(S_q + i);
        const __m256d uvec = _mm256_load_pd(u_k + i * lanes);
        acc = _mm256_fmadd_pd(s, uvec, acc);
      }
      _mm256_store_pd(v_k + q * lanes, acc);
    }
  }
}

// -----------------------------------------------------------------------------
// 4. AVX2 + register blocking. Process two q-values per iteration. The u
//    loads are shared across both, halving load pressure relative to v3.
//    This is what deal.II does in its templated apply_matrix_vector_product
//    via the 4-wide column-blocking pattern, adapted here to width 2 for
//    clarity. Effective on Skylake/Zen2+ where load throughput is the limiter.
// -----------------------------------------------------------------------------
MFK_ALWAYS_INLINE
void apply_1d_avx2_blocked(const double* MFK_RESTRICT S,
                           const double* MFK_RESTRICT u,
                           double*       MFK_RESTRICT v,
                           int n_q, int n_d, int n_spec)
{
  for (int k = 0; k < n_spec; ++k) {
    const double* u_k = u + k * n_d * lanes;
    double*       v_k = v + k * n_q * lanes;

    int q = 0;
    for (; q + 1 < n_q; q += 2) {
      __m256d acc0 = _mm256_setzero_pd();
      __m256d acc1 = _mm256_setzero_pd();
      const double* S0 = S + (q + 0) * n_d;
      const double* S1 = S + (q + 1) * n_d;
      for (int i = 0; i < n_d; ++i) {
        const __m256d uvec = _mm256_load_pd(u_k + i * lanes);
        const __m256d s0   = _mm256_broadcast_sd(S0 + i);
        const __m256d s1   = _mm256_broadcast_sd(S1 + i);
        acc0 = _mm256_fmadd_pd(s0, uvec, acc0);
        acc1 = _mm256_fmadd_pd(s1, uvec, acc1);
      }
      _mm256_store_pd(v_k + (q + 0) * lanes, acc0);
      _mm256_store_pd(v_k + (q + 1) * lanes, acc1);
    }
    // Tail when n_q is odd.
    for (; q < n_q; ++q) {
      __m256d acc = _mm256_setzero_pd();
      const double* S_q = S + q * n_d;
      for (int i = 0; i < n_d; ++i) {
        const __m256d s    = _mm256_broadcast_sd(S_q + i);
        const __m256d uvec = _mm256_load_pd(u_k + i * lanes);
        acc = _mm256_fmadd_pd(s, uvec, acc);
      }
      _mm256_store_pd(v_k + q * lanes, acc);
    }
  }
}

// -----------------------------------------------------------------------------
// 5. Even-odd decomposition. When S has the symmetry property
//        S(q, i) = ± S(n_q - 1 - q, n_d - 1 - i)
//    (true for Gauss-Lobatto nodes with Lagrange basis on Gauss-Lobatto points
//    — the canonical spectral element setup), we can split the input into
//    even-symmetric and odd-symmetric parts and process half the entries
//    against each. Asymptotic FLOP count drops from 2·n_d·n_q to roughly
//    n_d·n_q. See Kopriva (2009) ch. 4, also Solomonoff (1992).
//
//    For simplicity this implementation assumes n_q == n_d and both are even.
//    Real spectral codes generalize to odd n by treating the middle row
//    specially. For benchmarking purposes the even-even case captures the
//    speedup correctly.
//
//    Preconditioned matrices:
//      S_plus(q, i)  = 0.5 * (S(q, i) + S(q, n_d-1-i))      symmetric part
//      S_minus(q, i) = 0.5 * (S(q, i) - S(q, n_d-1-i))      antisymmetric part
//    for q ∈ [0, n_q/2), i ∈ [0, n_d/2).
// -----------------------------------------------------------------------------
MFK_ALWAYS_INLINE
void apply_1d_evenodd_avx2(const double* MFK_RESTRICT S_plus,
                           const double* MFK_RESTRICT S_minus,
                           const double* MFK_RESTRICT u,
                           double*       MFK_RESTRICT v,
                           int n_q, int n_d, int n_spec)
{
  assert(n_q == n_d && (n_q % 2 == 0) && "even-odd path assumes square, even");
  const int half_q = n_q / 2;
  const int half_d = n_d / 2;

  for (int k = 0; k < n_spec; ++k) {
    const double* u_k = u + k * n_d * lanes;
    double*       v_k = v + k * n_q * lanes;

    // First form e[i] = u[i] + u[n_d-1-i],  o[i] = u[i] - u[n_d-1-i]
    // for i in [0, half_d). Stack-allocated; sizes here are tiny (≤ 64 dofs).
    alignas(32) double e_buf[64 * lanes];
    alignas(32) double o_buf[64 * lanes];
    assert(half_d * lanes <= 64 * lanes);

    for (int i = 0; i < half_d; ++i) {
      const __m256d a = _mm256_load_pd(u_k + i * lanes);
      const __m256d b = _mm256_load_pd(u_k + (n_d - 1 - i) * lanes);
      _mm256_store_pd(e_buf + i * lanes, _mm256_add_pd(a, b));
      _mm256_store_pd(o_buf + i * lanes, _mm256_sub_pd(a, b));
    }

    // Compute v_plus[q] = Σ S_plus(q,i) * e[i],  v_minus[q] = Σ S_minus(q,i) * o[i]
    // for q in [0, half_q). Then
    //   v[q]            = v_plus[q] + v_minus[q]
    //   v[n_q - 1 - q]  = v_plus[q] - v_minus[q]
    for (int q = 0; q < half_q; ++q) {
      __m256d acc_p = _mm256_setzero_pd();
      __m256d acc_m = _mm256_setzero_pd();
      const double* Sp = S_plus  + q * half_d;
      const double* Sm = S_minus + q * half_d;
      for (int i = 0; i < half_d; ++i) {
        acc_p = _mm256_fmadd_pd(_mm256_broadcast_sd(Sp + i),
                                _mm256_load_pd(e_buf + i * lanes), acc_p);
        acc_m = _mm256_fmadd_pd(_mm256_broadcast_sd(Sm + i),
                                _mm256_load_pd(o_buf + i * lanes), acc_m);
      }
      _mm256_store_pd(v_k + q * lanes,                 _mm256_add_pd(acc_p, acc_m));
      _mm256_store_pd(v_k + (n_q - 1 - q) * lanes,     _mm256_sub_pd(acc_p, acc_m));
    }
  }
}

#endif // __AVX2__

// FLOP counter for the standard variant. 2 flops per (q, i, k, lane) pair.
// Even-odd cuts this roughly in half (1 add/sub for pre-combine, 2 fmas per
// (q, i, half-k, lane) pair, 1 add/sub for post-combine).
inline double standard_flops(int n_q, int n_d, int n_spec) {
  return 2.0 * static_cast<double>(n_q) * n_d * n_spec * lanes;
}

inline double evenodd_flops(int n_q, int n_d, int n_spec) {
  // half_q * half_d * spec * lanes * 4 flops (2 fma each into v_plus and v_minus)
  // + half_d * spec * lanes * 2 flops (pre-combine)
  // + half_q * spec * lanes * 2 flops (post-combine)
  const double hq = n_q * 0.5, hd = n_d * 0.5;
  return (4.0 * hq * hd + 2.0 * hd + 2.0 * hq) * n_spec * lanes;
}

} // namespace mfk
