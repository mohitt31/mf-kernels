# mf-kernels: High-Performance Matrix-Free Tensor-Product FEM Kernels

[![Build Status](https://github.com/mohitt31/mf-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/mohitt31/mf-kernels/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Five C++17 shape-matrix application kernels and a 3D sum-factorization driver that reduce the $\mathcal{O}(p^{2d})$ cost of high-order finite element evaluation to $\mathcal{O}(d \cdot p^{d+1})$ via sum-factorization (Kronbichler & Kormann, 2012).

High-order finite element methods evaluate field values at quadrature points by applying a tensor-product operator $V = (S \otimes S \otimes S) U$. Sum factorization rearranges this into three 1D contractions. This repository explores optimizing the hot inner loop of that contraction, and separately, extends it to a full operator and benchmarks it against MFEM and libCEED.

## Kernel Variants

The inner loop applies a 1D shape matrix across a spectator dimension, implemented five ways:

- **`naive`**: Direct triple loop. Relies on the compiler's auto-vectorizer; the honest baseline.
- **`pitfall`**: Attempts to "help" the compiler with an explicit per-lane accumulator array. Underperforms explicit AVX2 by ~4× on x86 because gcc 13.3 lowers the broadcasts to `vpermpd` instead of `vbroadcastsd` (verified in generated assembly, see `asm/README.md`).
- **`avx2`**: Explicit AVX2 SIMD, `_mm256_broadcast_sd` + `_mm256_fmadd_pd`, one accumulator per quadrature point.
- **`avx2_blocked`**: Same as `avx2`, 2-way register blocking across quadrature points.
- **`evenodd_avx2`**: Exploits shape-matrix symmetry to halve the FMA count.

## Build

Requirements: C++17 compiler (GCC 11+ or Clang 14+), x86-64 with AVX2/FMA (Haswell/Zen onward).

```bash
git clone https://github.com/mohitt31/mf-kernels.git
cd mf-kernels
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Default build uses `-O3 -march=native`.

## Bare-kernel benchmark

**Setup:** AMD EPYC (GitHub Codespace, x86-64, AVX2 — no AVX-512), gcc 13.3.0, `-O3 -march=native`, median of 5 runs, `DoNotOptimize` barrier so the naive baseline can't be dead-code-eliminated. GFLOP/s uses the standard algorithm's FLOP count, so even-odd's algorithmic saving shows as a wall-clock gain.

| p | naive | pitfall | avx2 | blocked | evenodd | best vs naive |
|---|------:|--------:|-----:|--------:|--------:|--------------:|
| 5 |  2.53 |  3.76   | 15.28|  20.82  |  21.73  |  **8.6×**     |
| 7 |  2.56 |  3.91   | 15.13|  21.58  |  25.92  | **10.1×**     |
| 9 |  2.84 |  3.86   | 15.41|  21.09  |  28.36  | **10.0×**     |

Reproduce with `./bench.sh`.

> An earlier, un-guarded measurement made `naive` look artificially fast (~15 GFLOP/s) because the compiler eliminated the unused output loop; the escape barrier corrected it to the honest ~2.5 GFLOP/s. `pitfall` plateaus at ~3.8 GFLOP/s and underperforms explicit AVX2 by ~4× — traced in the generated assembly to `vpermpd` vs `vbroadcastsd`.

## Full-operator benchmark: mf-kernels vs MFEM partial assembly vs libCEED

The kernel above is a bare 1D contraction. To find out whether it's actually competitive as a finite-element operator — not just a microbenchmark — I extended it to the full CEED BP1 (mass) and BP3 (Poisson) bake-off operators and benchmarked it against MFEM's native partial assembly and libCEED's CPU backends, on the same machine, same build, same DOF/FLOP accounting. Full writeup, methodology caveats, correctness proof, and reproduction scripts: **[`benchmarks/mfem-comparison`](benchmarks/mfem-comparison)**.

**Setup:** AMD EPYC 7763 (Zen 3), single core, `taskset`-pinned, gcc 13.3.0, `-O3 -march=native`. GLL nodal basis, Gauss-Legendre quadrature at the CEED `q = p+2` rule, order sweep `p = 1..8`. Correctness validated two ways: against a dense O(p⁶) reference (worst diff 8.9e-15) and against MFEM's own output on identical inputs (worst diff 3.8e-15) — see [`WRITEUP.md`](benchmarks/mfem-comparison/WRITEUP.md) §3.

**Results (unique DOFs/sec, millions), BP1 mass operator:**

| p | mf-k blocked | mf-k even-odd | MFEM PA | libCEED avx | libCEED xsmm |
|---|----:|----:|----:|----:|----:|
| 1 | 14.0 | 11.9 |  8.2 |  5.3 |  4.5 |
| 4 | 79.4 | 73.8 | 44.7 | 48.4 | 77.3 |
| 8 | 96.2 | 95.8 | 27.6\* | 72.7 | 124.9 |

Full BP1 and BP3 tables (p = 1..8, five implementations each): [`WRITEUP.md` §4](benchmarks/mfem-comparison/WRITEUP.md).

**Honest interpretation:**
- mf-kernels' blocked variant beats MFEM native partial assembly at every order on both operators, often by close to 2×.
- It also beats libCEED's own AVX blocked backend across the whole order range — the fairest single comparison, since both batch elements into SIMD lanes.
- The only implementation that beats mf-kernels is libCEED with LIBXSMM, and only at higher order (mf-kernels trails by ~15-25% at p=8); mf-kernels leads at low order.
- No claim is made that mf-kernels beats MFEM or libCEED as a system — this measures one operator apply on a structured, constant-geometry mesh. See [`methodology_flags.md`](benchmarks/mfem-comparison/methodology_flags.md) for the full list of caveats (quadrature convention, single-core scope, cloud-VM timing jitter, and more), listed weaknesses-first on purpose.

Reproduce:
```bash
cd benchmarks/mfem-comparison
bash scripts/build.sh   # builds LIBXSMM + libCEED + MFEM + binaries
bash scripts/run.sh     # env capture, both sides, 1e-13 diff check, plots
```

## Portability: ARM / Apple Silicon

On `aarch64`/Apple M-series, AVX2 paths are `#if`-guarded out, leaving `naive` and `pitfall`. Interestingly, `pitfall` — which underperforms AVX2 by ~4× on x86 gcc — becomes efficient under Apple Clang's NEON auto-vectorizer, since its explicit per-lane accumulator maps cleanly onto NEON. There is no architecture-independent "SIMD-friendly" C++: code that helps the auto-vectorizer on one ISA can be a trap on another, which is why deal.II's `VectorizedArray` abstraction exists.

All benchmark numbers above are from a remote x86-64 EPYC box (I develop on Apple Silicon, which has no AVX2/AVX-512); the code builds and runs on both.

## References

1. M. Kronbichler & K. Kormann. *A generic interface for parallel cell-based finite element operator application.* Computers & Fluids, 63:135–147 (2012). [doi:10.1016/j.compfluid.2012.04.012](https://doi.org/10.1016/j.compfluid.2012.04.012)
