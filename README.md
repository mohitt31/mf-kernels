# mf-kernels: High-Performance Matrix-Free Tensor-Product FEM Kernels

[![Build Status](https://github.com/mohitt31/mf-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/mohitt31/mf-kernels/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Five C++17 shape-matrix application kernels and a 3D sum-factorization driver that drastically reduce the $\mathcal{O}(n_q^3 n_d^3)$ cost of high-order finite element evaluation by leveraging sum-factorization down to $\mathcal{O}(n_d^{d+1})$.

High-order finite element methods evaluate field values at quadrature points by applying a tensor-product operator $V = (S \otimes S \otimes S) U$. Sum factorization rearranges this into three 1D contractions. This repository explores optimizing the extremely hot inner loop of this contraction to reach peak hardware utilization.

## Kernel Variants

The inner loop applies a 1D shape matrix across a spectator dimension. This repository implements it in five ways:

- **`naive`**: Direct triple loop. Relies on the compiler's auto-vectorizer and serves as the honest baseline.
- **`pitfall`**: "Helps" the compiler by using an explicit per-lane accumulator array, which severely backfires on x86 due to `vpermpd`-based broadcasts.
- **`avx2`**: Explicit AVX2 SIMD with `_mm256_broadcast_sd` and `_mm256_fmadd_pd`, using a single accumulator per quadrature point.
- **`avx2_blocked`**: Same as `avx2`, but uses 2-way register blocking across quadrature points to amortize the shape-matrix load and expose more independent FMA chains.
- **`evenodd_avx2`**: Algorithmic optimization exploiting the symmetry of the shape matrix to halve the total FMA count.

## Build Instructions

Requirements: A C++17 compiler (GCC 11+ or Clang 14+) and an x86-64 CPU with AVX2 and FMA (Haswell or Zen onwards).

```bash
git clone https://github.com/mohitt31/mf-kernels.git
cd mf-kernels
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

By default, the build uses `-O3 -march=native`. Ensure your compiler targets an architecture with AVX2/FMA for the optimized kernels.

## Benchmark Results

**Configuration:** Intel Xeon (x86-64, AVX2 — no AVX-512), gcc 13.3, `-O3 -march=native`.
Measured on a GitHub Codespace. Throughput is reported in GFLOP/s using the *standard* algorithm's FLOP count, so the even-odd algorithmic variant's speedup is visible as a pure wall-clock gain. Median of 5 runs.

| p | naive | pitfall | avx2 | blocked | evenodd | best vs naive |
|---|------:|--------:|-----:|--------:|--------:|--------------:|
| 5 | 11.4  |   3.5   |  9.7 |  17.0   |  20.8   |  **1.82×**    |
| 7 | 14.1  |   3.8   |  9.5 |  16.9   |  24.7   |  **1.75×**    |
| 9 | 15.5  |   3.7   | 10.4 |  20.9   |  25.7   |  **1.65×**    |

> **Note**: For polynomials $p \ge 5$ (the regime where matrix-free FEM is strictly preferred), the algorithmic exploitation of symmetry (`evenodd_avx2`) massively outperforms the compiler's auto-vectorization (`naive`). Straight `avx2` can sometimes underperform the `naive` baseline because single-accumulator chains suffer from a 4-5 cycle FMA latency; the `blocked` variant restores parity by holding two explicit accumulators to hide latency.

## Reproducibility

You can build the repository and reproduce the exact benchmark table above with a single script:

```bash
./bench.sh
```

## Portability: ARM / Apple Silicon

On `aarch64` / Apple M-series, the AVX2 paths are `#if`-guarded out, leaving only the `naive` and `pitfall` variants. Interestingly, the `pitfall` variant—which causes a 3-4× slowdown on x86 gcc due to poor scalar broadcasting instructions—becomes 3-4× *faster* under Apple Clang's NEON auto-vectorizer, as its explicit per-lane accumulator maps cleanly onto NEON. 

This illustrates a broader lesson: there is no architecture-independent "SIMD-friendly" C++. Code that tricks the auto-vectorizer perfectly on one ISA may be a trap on another, requiring explicit per-ISA vector abstractions (like `deal.II`'s `VectorizedArray`).

## References

1. M. Kronbichler & K. Kormann. *A generic interface for parallel cell-based finite element operator application.* Computers & Fluids, 63:135–147 (2012). [doi:10.1016/j.compfluid.2012.04.012](https://doi.org/10.1016/j.compfluid.2012.04.012)
