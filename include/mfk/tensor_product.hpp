// SPDX-License-Identifier: MIT
// mfk/tensor_product.hpp — 3D sum-factorized tensor-product evaluation.
//
// Given a 3D coefficient field u of shape (n_d, n_d, n_d) per cell (× lanes),
// applying the full tensor-product evaluation operator
//
//     V(q1,q2,q3) = Σ_{i1,i2,i3} S(q1,i1) S(q2,i2) S(q3,i3) U(i1,i2,i3)
//
// naively costs O(n_q^3 · n_d^3). Sum factorization rearranges it as three
// sequential 1D applies (with reshape/transpose between them) for a total
// cost of O(3 · n_q · n_d · (n_q · n_d)^2 / (n_q · n_d)) = O(n_q^3 · n_d) per
// cell — a factor of n_d^2 better. For Q5 elements (n_d=6) that's 36×.
//
// The three passes (index-i₁ first, then i₂, then i₃) each call apply_1d
// with a different value of n_spec (the spectator dimension product):
//   Pass 1: contract i1.  spec = n_d  * n_d  (i2, i3 untouched)
//   Pass 2: contract i2.  spec = n_q  * n_d  (q1 result, i3 untouched)
//   Pass 3: contract i3.  spec = n_q  * n_q  (q1, q2 results)

#pragma once

#include "apply_1d.hpp"
#include "aligned_buffer.hpp"

namespace mfk {

// kernel_fn is a pointer-to-function compatible with the apply_1d_* signatures
// above. The 3D driver dispatches to whichever you pass in.
using Apply1DFn = void (*)(const double* /*S*/,
                           const double* /*u*/,
                           double*       /*v*/,
                           int /*n_q*/, int /*n_d*/, int /*n_spec*/);

// Apply the 3D tensor product operator using the given 1D kernel. The temp
// buffers tmp1 and tmp2 must be sized to hold n_q * n_d * n_d * lanes doubles
// (worst case across the three passes).
inline void apply_3d(Apply1DFn kernel,
                     const double* MFK_RESTRICT S,
                     const double* MFK_RESTRICT u,
                     double*       MFK_RESTRICT v,
                     double*       MFK_RESTRICT tmp1,
                     double*       MFK_RESTRICT tmp2,
                     int n_q, int n_d)
{
  // Pass 1: contract i1.
  //   in : u[i3, i2, i1, lane]   -> n_spec = n_d * n_d
  //   out: tmp1[i3, i2, q1, lane]
  kernel(S, u, tmp1, n_q, n_d, n_d * n_d);

  // Pass 2: contract i2. We need the contracted index to be innermost-but-
  // -for-lane, so reshape: view tmp1 as [i3, q1, i2, lane] for the call —
  // this requires a logical transpose between passes. To keep the kernel
  // generic we perform an explicit transpose pass1->pass2 via tmp2.
  //
  // Source layout from pass 1: tmp1[i3 * n_d * n_q + i2 * n_q + q1] (per lane).
  // Target layout for pass 2:  tmp2[i3 * n_q * n_d + q1 * n_d + i2] (per lane).
  //
  // The transpose is between the inner (q1) and the spectator (i2) dimensions
  // for each fixed i3. It's O(n_q * n_d * n_d) reads/writes — small relative
  // to the FLOP cost but cache-friendly only if we tile.
  for (int i3 = 0; i3 < n_d; ++i3) {
    const double* src = tmp1 + i3 * n_d  * n_q * lanes;
    double*       dst = tmp2 + i3 * n_q  * n_d * lanes;
    for (int i2 = 0; i2 < n_d; ++i2) {
      for (int q1 = 0; q1 < n_q; ++q1) {
        for (int l = 0; l < lanes; ++l)
          dst[(q1 * n_d + i2) * lanes + l] =
            src[(i2 * n_q + q1) * lanes + l];
      }
    }
  }
  kernel(S, tmp2, tmp1, n_q, n_d, n_d * n_q);

  // Pass 3: contract i3. Transpose again so i3 is innermost.
  // Source: tmp1[i3, q1, q2, lane]  with layout [i3 * n_q * n_q + q1 * n_q + q2]
  // Target: tmp2[q1, q2, i3, lane]
  for (int q1 = 0; q1 < n_q; ++q1) {
    for (int q2 = 0; q2 < n_q; ++q2) {
      for (int i3 = 0; i3 < n_d; ++i3) {
        for (int l = 0; l < lanes; ++l)
          tmp2[((q1 * n_q + q2) * n_d + i3) * lanes + l] =
            tmp1[((i3 * n_q + q1) * n_q + q2) * lanes + l];
      }
    }
  }
  kernel(S, tmp2, v, n_q, n_d, n_q * n_q);
}

// FLOP count for the full 3D apply. Three 1D kernel calls plus two transposes
// (which are pure memory ops, no flops). The 1D kernels' flop counts depend
// on which variant; we use the standard count and let the caller add the
// even-odd correction if needed.
inline double apply_3d_standard_flops(int n_q, int n_d) {
  return standard_flops(n_q, n_d, n_d * n_d)
       + standard_flops(n_q, n_d, n_d * n_q)
       + standard_flops(n_q, n_d, n_q * n_q);
}

} // namespace mfk
