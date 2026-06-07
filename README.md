# mf-kernels: High-Performance Matrix-Free Tensor-Product FEM Kernels

[![Build Status](https://github.com/mohitt31/mf-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/mohitt31/mf-kernels/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Five C++17 shape-matrix application kernels and a 3D sum-factorization driver that drastically reduce the $\mathcal{O}(p^{2d})$ cost of high-order finite element evaluation by leveraging sum-factorization down to $\mathcal{O}(d \cdot p^{d+1})$.

High-order finite element methods evaluate field values at quadrature points by applying a tensor-product operator $V = (S \otimes S \otimes S) U$. Sum factorization rearranges this into three 1D contractions. This repository explores optimizing the extremely hot inner loop of this contraction to reach peak hardware utilization.

## Kernel Variants

The inner loop applies a 1D shape matrix across a spectator dimension. This repository implements it in five ways:

- **`naive`**: Direct triple loop. Relies on the compiler's auto-vectorizer and serves as the honest baseline.
- **`pitfall`**: Attempts to "help" the compiler by using an explicit per-lane accumulator array. While it slightly outperforms the scalar `naive` code, it severely underperforms explicit AVX2 by ~4×. This happens because `gcc 13.3` lowers the broadcasts via 8 occurrences of `vpermpd` instead of `vbroadcastsd` (verified in `asm/pitfall_vs_naive.s`), capping its throughput.
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

**Configuration:** AMD EPYC (GitHub Codespace, x86-64, AVX2 — no AVX-512), gcc 13.3.0, `-O3 -march=native`, `--target-seconds 2.0`, median of 5 runs, with a `DoNotOptimize` barrier so the naive baseline is not dead-code-eliminated.
Throughput is reported in GFLOP/s using the *standard* algorithm's FLOP count, so the even-odd algorithmic variant's speedup is visible as a pure wall-clock gain.

| p | naive | pitfall | avx2 | blocked | evenodd | best vs naive |
|---|------:|--------:|-----:|--------:|--------:|--------------:|
| 5 |  2.53 |  3.76   | 15.28|  20.82  |  21.73  |  **8.6×**     |
| 7 |  2.56 |  3.91   | 15.13|  21.58  |  25.92  | **10.1×**     |
| 9 |  2.84 |  3.86   | 15.41|  21.09  |  28.36  | **10.0×**     |

> **Note**: An earlier un-guarded measurement made the `naive` baseline look artificially fast (~15 GFLOP/s) because the compiler eliminated the unused output loop. Adding an escape barrier corrected this, revealing the honest scalar throughput is ~2.5 GFLOP/s. This transparency is a strength: against this true scalar baseline, explicit `avx2` yields a ~6× speedup; 2-way register blocking (`blocked`) pushes that to ~8×; and the algorithmic exploitation of symmetry (`evenodd_avx2`) reaches an impressive ~8.6–10×.
> 
> The `pitfall` variant, while slightly faster than the scalar baseline, plateaus around ~3.8 GFLOP/s and underperforms explicit AVX2 by ~4×. The generated assembly explains why it cannot catch explicit AVX2: its "helpful" hints cause the compiler to emit `vpermpd` instead of `vbroadcastsd`.

## Reproducibility

You can build the repository and reproduce the exact benchmark table above with a single script:

```bash
./bench.sh
```

## Portability: ARM / Apple Silicon

On `aarch64` / Apple M-series, the AVX2 paths are `#if`-guarded out, leaving only the `naive` and `pitfall` variants. Interestingly, the `pitfall` variant—which underperforms explicit AVX2 by ~4× on x86 gcc due to poor scalar broadcasting instructions—becomes highly efficient under Apple Clang's NEON auto-vectorizer, as its explicit per-lane accumulator maps cleanly onto NEON. 

This illustrates a broader lesson: there is no architecture-independent "SIMD-friendly" C++. Code that tricks the auto-vectorizer perfectly on one ISA may be a trap on another, requiring explicit per-ISA vector abstractions (like `deal.II`'s `VectorizedArray`).

## References

1. M. Kronbichler & K. Kormann. *A generic interface for parallel cell-based finite element operator application.* Computers & Fluids, 63:135–147 (2012). [doi:10.1016/j.compfluid.2012.04.012](https://doi.org/10.1016/j.compfluid.2012.04.012)
