// numerics.hpp - GLL nodes, GL quadrature, Lagrange basis at quad points.
// Self-contained, MFEM-independent. Used to build B (n_q x n_d) and verify
// the sum-factorized BP operator against a dense reference.
#pragma once
#include <vector>
#include <cmath>
#include <functional>

namespace num {

// Legendre P_n and derivative via recurrence.
inline void legendre(int n, double x, double& P, double& dP) {
  double p0 = 1.0, p1 = x;
  if (n == 0) { P = 1.0; dP = 0.0; return; }
  if (n == 1) { P = x;   dP = 1.0; return; }
  for (int k = 2; k <= n; ++k) {
    double p2 = ((2*k-1)*x*p1 - (k-1)*p0)/k;
    p0 = p1; p1 = p2;
  }
  P = p1;
  dP = n*(x*p1 - p0)/(x*x - 1.0);
}

// Gauss-Legendre nodes/weights on [-1,1], n points. Newton on P_n.
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.resize(n); w.resize(n);
  for (int i = 0; i < n; ++i) {
    double xi = std::cos(M_PI*(i + 0.75)/(n + 0.5)); // initial guess
    for (int it = 0; it < 100; ++it) {
      double P, dP; legendre(n, xi, P, dP);
      double dx = -P/dP; xi += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    double P, dP; legendre(n, xi, P, dP);
    x[i] = xi;
    w[i] = 2.0/((1.0 - xi*xi)*dP*dP);
  }
  // sort ascending
  for (int i = 0; i < n; ++i)
    for (int j = i+1; j < n; ++j)
      if (x[j] < x[i]) { std::swap(x[i],x[j]); std::swap(w[i],w[j]); }
}

// Gauss-Lobatto-Legendre nodes on [-1,1], n points (endpoints included).
// Interior nodes are roots of P'_{n-1}. Newton iteration.
inline void gauss_lobatto(int n, std::vector<double>& x) {
  x.resize(n);
  x[0] = -1.0; x[n-1] = 1.0;
  if (n == 2) return;
  // interior roots of P'_{n-1}: use Chebyshev-Gauss-Lobatto initial guess
  for (int i = 1; i < n-1; ++i) {
    double xi = -std::cos(M_PI*i/(n-1));
    for (int it = 0; it < 100; ++it) {
      // f = P'_{n-1}(xi). Use that (1-x^2)P'_{n-1} = ... ; easier: Newton on
      // derivative of Legendre via finite-recurrence for P'_{m} and P''_{m}.
      int m = n-1;
      // compute P_m, P'_m, P''_m
      double Pm, dPm; legendre(m, xi, Pm, dPm);
      // P''_m from ODE: (1-x^2)P'' - 2x P' + m(m+1)P = 0
      double ddPm = (2*xi*dPm - m*(m+1)*Pm)/(1.0 - xi*xi);
      double dx = -dPm/ddPm; xi += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    x[i] = xi;
  }
  for (int i = 0; i < n; ++i)
    for (int j = i+1; j < n; ++j)
      if (x[j] < x[i]) std::swap(x[i],x[j]);
}

// Lagrange basis values and derivatives at point xq for nodes {x_i}.
inline void lagrange_at(const std::vector<double>& nodes, double xq,
                        std::vector<double>& val, std::vector<double>& der) {
  int nd = nodes.size();
  val.assign(nd, 0.0); der.assign(nd, 0.0);
  for (int i = 0; i < nd; ++i) {
    double Li = 1.0;
    for (int j = 0; j < nd; ++j) if (j!=i) Li *= (xq - nodes[j])/(nodes[i]-nodes[j]);
    val[i] = Li;
    // derivative
    double d = 0.0;
    for (int k = 0; k < nd; ++k) if (k!=i) {
      double term = 1.0/(nodes[i]-nodes[k]);
      for (int j = 0; j < nd; ++j) if (j!=i && j!=k) term *= (xq - nodes[j])/(nodes[i]-nodes[j]);
      d += term;
    }
    der[i] = d;
  }
}

// Build B (n_q x n_d) and dB (n_q x n_d): basis values/derivs of GLL nodal
// basis (n_d=p+1 nodes) evaluated at GL quad points (n_q=p+2). Row-major [q*n_d+i].
inline void build_B(int p, std::vector<double>& B, std::vector<double>& dB,
                    std::vector<double>& qx, std::vector<double>& qw,
                    std::vector<double>& nodes) {
  int nd = p+1, nq = p+2;
  gauss_lobatto(nd, nodes);
  gauss_legendre(nq, qx, qw);
  B.assign(nq*nd, 0.0); dB.assign(nq*nd, 0.0);
  std::vector<double> val, der;
  for (int q = 0; q < nq; ++q) {
    lagrange_at(nodes, qx[q], val, der);
    for (int i = 0; i < nd; ++i) { B[q*nd+i]=val[i]; dB[q*nd+i]=der[i]; }
  }
}

} // namespace num
