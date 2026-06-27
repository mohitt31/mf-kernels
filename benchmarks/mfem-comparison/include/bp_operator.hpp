// bp_operator.hpp - Full CEED BP1 (mass) / BP3 (Poisson) matrix-free operator
// apply on 3D hexahedral elements, built on mf-kernels' apply_1d_* contraction
// kernels.
//
//   v = G^T B^T D B G u      (CEED operator decomposition)
//
//     G    element gather/scatter (global L-vector <-> element E-vector)
//     B    sum-factorized basis evaluation, node -> quad (values; +grad for BP3)
//     D    pointwise quadrature operation (mass weight; 3x3 metric for Poisson)
//     B^T  sum-factorized integration, quad -> node
//     G^T  scatter/assemble
//
// This header implements the *element-local* part (B, D, B^T) for a batch of
// `lanes` cells. The gather/scatter G is handled by the caller (it is just an
// index map; on the benchmark machine it is MFEM's element restriction).
//
// All numeric inputs (B, dB, the quadrature data D, and the dof index map) are
// SUPPLIED by the caller. On the benchmark machine they are exported from MFEM
// so the apply reproduces MFEM's element operator to round-off; that is what
// makes the 1e-13 full-vector correctness check meaningful. In the standalone
// validation/dry-run path they come from numerics.hpp.
//
// Kernel routing. Every 1D contraction is passed two function pointers:
//     Kv  applied when the 1D matrix is a VALUE matrix (B, B^T): symmetric.
//     Kd  applied when the 1D matrix is a DERIVATIVE matrix (dB, dB^T): antisym.
// For the standard kernels (naive / avx2 / avx2_blocked) Kv == Kd == the generic
// kernel, because a generic GEMM does not care about the matrix' symmetry. For
// the even-odd kernels Kv = apply_1d_evenodd_rect (uses symmetry) and
// Kd = apply_1d_evenodd_rect_deriv (uses antisymmetry); routing the wrong one
// gives wrong answers, hence the explicit split.
#pragma once
#include "apply_1d.hpp"
#include "evenodd_rect.hpp"
#include <cstddef>

namespace mfk {

using Apply1DFn = void (*)(const double*, const double*, double*, int, int, int);

// ------------------------------------------------------------------ transposes
// batched over `lanes`. [outer][mid][inner] -> [outer][inner][mid]
inline void bp_tr_swap(const double* s, double* d, int no, int nm, int ni) {
  for (int o = 0; o < no; ++o) {
    const double* sp = s + (size_t)o * nm * ni * lanes;
    double*       dp = d + (size_t)o * ni * nm * lanes;
    for (int m = 0; m < nm; ++m)
      for (int i = 0; i < ni; ++i)
        for (int l = 0; l < lanes; ++l)
          dp[((size_t)i * nm + m) * lanes + l] = sp[((size_t)m * ni + i) * lanes + l];
  }
}
// [a][b][c] -> [b][c][a]  (rotate outer index to innermost)
inline void bp_tr_rot(const double* s, double* d, int na, int nb, int nc) {
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < nb; ++b)
      for (int c = 0; c < nc; ++c)
        for (int l = 0; l < lanes; ++l)
          d[(((size_t)b * nc + c) * na + a) * lanes + l] =
              s[(((size_t)a * nb + b) * nc + c) * lanes + l];
}

// --------------------------------------------------------------- BP1 (mass)
// nodal U[i3][i2][i1] (i1 fastest, n_d each) -> quad Q[q1][q2][q3] (q3 fastest).
inline void interp3d(Apply1DFn Kv, const double* B, const double* U, double* Q,
                     double* t1, double* t2, int nq, int nd) {
  Kv(B, U,  t1, nq, nd, nd * nd);     // [i3][i2][q1]
  bp_tr_swap(t1, t2, nd, nd, nq);     // [i3][q1][i2]
  Kv(B, t2, t1, nq, nd, nd * nq);     // [i3][q1][q2]
  bp_tr_rot(t1, t2, nd, nq, nq);      // [q1][q2][i3]
  Kv(B, t2, Q,  nq, nd, nq * nq);     // [q1][q2][q3]
}
// transpose of interp3d, using Bt (n_d x n_q). Returns node ordering to
// i1-fastest so the element operator is symmetric.
inline void integ3d(Apply1DFn Kv, const double* Bt, const double* Q, double* U,
                    double* t1, double* t2, int nq, int nd) {
  Kv(Bt, Q,  t1, nd, nq, nq * nq);    // [q1][q2][a3]
  bp_tr_swap(t1, t2, nq, nq, nd);     // [q1][a3][q2]
  Kv(Bt, t2, t1, nd, nq, nq * nd);    // [q1][a3][a2]
  bp_tr_rot(t1, t2, nq, nd, nd);      // [a3][a2][q1]
  Kv(Bt, t2, U,  nd, nq, nd * nd);    // [a3][a2][a1] == [i3][i2][i1]
}

// BP1 element apply. qd: nq^3 mass weights (w*detJ) per cell-lane, quad order Q.
inline void bp1_elem(Apply1DFn Kv, const double* B, const double* Bt,
                     const double* ue, double* ve, const double* qd,
                     double* t1, double* t2, double* Q, int nq, int nd) {
  interp3d(Kv, B, ue, Q, t1, t2, nq, nd);
  const int nq3 = nq * nq * nq;
  for (int q = 0; q < nq3; ++q)
    for (int l = 0; l < lanes; ++l)
      Q[(size_t)q * lanes + l] *= qd[(size_t)q * lanes + l];
  integ3d(Kv, Bt, Q, ve, t1, t2, nq, nd);
}

// --------------------------------------------------------------- BP3 (Poisson)
// gradient at quad points. Gx/Gy/Gz are d/dx1, d/dx2, d/dx3 in reference coords.
// Kv applied on the two value axes, Kd on the single derivative axis.
inline void grad3d(Apply1DFn Kv, Apply1DFn Kd, const double* B, const double* dB,
                   const double* U, double* Gx, double* Gy, double* Gz,
                   double* t1, double* t2, int nq, int nd) {
  // Gx: derivative along i1
  Kd(dB, U,  t1, nq, nd, nd * nd); bp_tr_swap(t1, t2, nd, nd, nq);
  Kv(B,  t2, t1, nq, nd, nd * nq); bp_tr_rot(t1, t2, nd, nq, nq);
  Kv(B,  t2, Gx, nq, nd, nq * nq);
  // Gy: derivative along i2
  Kv(B,  U,  t1, nq, nd, nd * nd); bp_tr_swap(t1, t2, nd, nd, nq);
  Kd(dB, t2, t1, nq, nd, nd * nq); bp_tr_rot(t1, t2, nd, nq, nq);
  Kv(B,  t2, Gy, nq, nd, nq * nq);
  // Gz: derivative along i3
  Kv(B,  U,  t1, nq, nd, nd * nd); bp_tr_swap(t1, t2, nd, nd, nq);
  Kv(B,  t2, t1, nq, nd, nd * nq); bp_tr_rot(t1, t2, nd, nq, nq);
  Kd(dB, t2, Gz, nq, nd, nq * nq);
}
// integrate one gradient component: dBt on `axis`, Bt on the other two.
inline void integ3d_axis(Apply1DFn Kv, Apply1DFn Kd, const double* Bt,
                         const double* dBt, const double* Q, double* U,
                         double* t1, double* t2, int nq, int nd, int axis) {
  // pass order contracts q3 -> a3, q2 -> a2, q1 -> a1
  const double* M0 = (axis == 2) ? dBt : Bt;  Apply1DFn K0 = (axis == 2) ? Kd : Kv;
  const double* M1 = (axis == 1) ? dBt : Bt;  Apply1DFn K1 = (axis == 1) ? Kd : Kv;
  const double* M2 = (axis == 0) ? dBt : Bt;  Apply1DFn K2 = (axis == 0) ? Kd : Kv;
  K0(M0, Q,  t1, nd, nq, nq * nq); bp_tr_swap(t1, t2, nq, nq, nd);
  K1(M1, t2, t1, nd, nq, nq * nd); bp_tr_rot(t1, t2, nq, nd, nd);
  K2(M2, t2, U,  nd, nq, nd * nd);
}
inline void grad3d_T(Apply1DFn Kv, Apply1DFn Kd, const double* Bt, const double* dBt,
                     const double* Gx, const double* Gy, const double* Gz,
                     double* U, double* acc, double* t1, double* t2, int nq, int nd) {
  const int nd3 = nd * nd * nd;
  for (int i = 0; i < nd3 * lanes; ++i) U[i] = 0.0;
  integ3d_axis(Kv, Kd, Bt, dBt, Gx, acc, t1, t2, nq, nd, 0);
  for (int i = 0; i < nd3 * lanes; ++i) U[i] += acc[i];
  integ3d_axis(Kv, Kd, Bt, dBt, Gy, acc, t1, t2, nq, nd, 1);
  for (int i = 0; i < nd3 * lanes; ++i) U[i] += acc[i];
  integ3d_axis(Kv, Kd, Bt, dBt, Gz, acc, t1, t2, nq, nd, 2);
  for (int i = 0; i < nd3 * lanes; ++i) U[i] += acc[i];
}

// BP3 element apply. qd6 layout [q][6][lane]: components (xx,xy,xz,yy,yz,zz) of
// the symmetric metric w*detJ*J^{-1}J^{-T} at each quad point.
inline void bp3_elem(Apply1DFn Kv, Apply1DFn Kd, const double* B, const double* dB,
                     const double* Bt, const double* dBt, const double* ue,
                     double* ve, const double* qd6, double* t1, double* t2,
                     double* Gx, double* Gy, double* Gz, double* acc,
                     int nq, int nd) {
  grad3d(Kv, Kd, B, dB, ue, Gx, Gy, Gz, t1, t2, nq, nd);
  const int nq3 = nq * nq * nq;
  for (int q = 0; q < nq3; ++q)
    for (int l = 0; l < lanes; ++l) {
      const double* d = qd6 + ((size_t)q * 6) * lanes;
      double gx = Gx[(size_t)q * lanes + l];
      double gy = Gy[(size_t)q * lanes + l];
      double gz = Gz[(size_t)q * lanes + l];
      double rx = d[0 * lanes + l] * gx + d[1 * lanes + l] * gy + d[2 * lanes + l] * gz;
      double ry = d[1 * lanes + l] * gx + d[3 * lanes + l] * gy + d[4 * lanes + l] * gz;
      double rz = d[2 * lanes + l] * gx + d[4 * lanes + l] * gy + d[5 * lanes + l] * gz;
      Gx[(size_t)q * lanes + l] = rx;
      Gy[(size_t)q * lanes + l] = ry;
      Gz[(size_t)q * lanes + l] = rz;
    }
  grad3d_T(Kv, Kd, Bt, dBt, Gx, Gy, Gz, ve, acc, t1, t2, nq, nd);
}

// ------------------------------------------------------- shared FLOP reference
// ONE standard sum-factorization FLOP count, applied identically to every
// implementation (naive / avx2 / blocked / even-odd) so that an algorithmic
// FLOP reduction (even-odd) shows up purely as a wall-clock GFLOP/s gain rather
// than as a different denominator. This mirrors mf-kernels' existing convention
// and is the convention reported to all four implementations in the comparison.
inline double bp1_flops_per_cell(int nq, int nd) {
  double interp = 2.0 * ((double)nq * nd * nd * nd + (double)nq * nq * nd * nd +
                         (double)nq * nq * nq * nd);
  double D      = (double)nq * nq * nq;                 // one multiply per quad pt
  double integ  = 2.0 * ((double)nd * nq * nq * nq + (double)nd * nd * nq * nq +
                         (double)nd * nd * nd * nq);
  return interp + D + integ;
}
inline double bp3_flops_per_cell(int nq, int nd) {
  // three gradient interpolations + symmetric 3x3 apply + three integrations
  double grad   = 3.0 * 2.0 * ((double)nq * nd * nd * nd + (double)nq * nq * nd * nd +
                               (double)nq * nq * nq * nd);
  double D      = 15.0 * (double)nq * nq * nq;          // sym 3x3 mat-vec: 9 mul + 6 add
  double integ  = 3.0 * 2.0 * ((double)nd * nq * nq * nq + (double)nd * nd * nq * nq +
                               (double)nd * nd * nd * nq);
  return grad + D + integ;
}

} // namespace mfk
