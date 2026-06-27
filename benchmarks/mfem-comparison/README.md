# mf-kernels vs MFEM partial assembly + libCEED

A like-for-like benchmark of the `mf-kernels` matrix-free operator against
MFEM's native partial assembly (PA) and libCEED's CPU backends, on the two
standard CEED bake-off problems:

- **BP1** mass operator, `v = G^T B^T (w detJ) B G u`
- **BP3** Poisson operator, `v = G^T B^T D B G u` with `D` the symmetric metric

3D hexahedral elements, order-`p` Gauss-Lobatto nodal basis (`n_d = p+1` nodes
per dimension), Gauss-Legendre quadrature with `n_q = p+2` points per dimension
(the CEED convention, integration rule order `2p+3`). Order sweep `p = 1..8`.

Four implementations are compared on the same machine, same build, same mesh,
same DOF/FLOP accounting:

1. `mf-kernels` apply (variants: naive, avx2, avx2_blocked, even-odd)
2. MFEM native PA (`AssemblyLevel::PARTIAL`)
3. libCEED `/cpu/self/avx/blocked`
4. libCEED `/cpu/self/xsmm/blocked` (LIBXSMM)

## Layout

```
include/
  bp_operator.hpp   full BP1/BP3 element operator on mf-kernels' 1D kernels
  evenodd_rect.hpp  rectangular even-odd kernels (values + derivatives)
  numerics.hpp      self-contained GLL/GL basis (standalone path only)
src/
  validate_bp.cpp   MFEM-independent correctness vs a dense reference
  bench_bp.cpp      mf-kernels full-operator timing (all variants), CSV out
  mfem_bp.cpp       MFEM PA + libCEED timing; exports data for the diff check
  diff_bp.cpp       mf-kernels vs MFEM on identical data, 1e-13 check
scripts/
  build.sh          build LIBXSMM + libCEED + MFEM + the binaries
  env_capture.sh    record CPU/cache/topology + toolchain
  run.sh            pinned single-core run of the whole comparison
  plots.py          throughput + GFLOP/s figures
results/            CSVs, env.txt, figures land here
Makefile
```

## Quick start

mf-kernels side only (no external deps), to sanity-check on any AVX2 box:

```bash
make mfk
taskset -c 0 ./results/validate_bp        # correctness vs dense reference
taskset -c 0 ./results/bench_bp --op both # timing dry-run, all variants
```

Full comparison on the benchmark node (AMD EPYC):

```bash
export ROOT=$HOME/bp-bench
bash scripts/build.sh                     # builds deps + all binaries
bash scripts/run.sh                       # env capture, both sides, diff, plots
```

## What makes it a fair comparison

- **Same operator.** `mf-kernels` is extended here from the bare 1D contraction
  to the full `G^T B^T D B G` operator, so it ingests and produces the same
  global vectors as MFEM. `bench_bp` times gather + element apply + scatter, the
  same span MFEM's `Mult` covers.
- **Same numerics.** On the benchmark machine the basis `B`, derivative `dB`,
  per-quad data `D`, and the element restriction map are exported from MFEM and
  fed to `mf-kernels`, so the only difference is the arithmetic, not the inputs.
  `diff_bp` confirms the two agree to `<= 1e-13` on the global vector per `p`.
- **Same accounting.** Throughput is unique global DOFs/sec (CEED convention).
  GFLOP/s uses one shared standard sum-factorization FLOP count applied to all
  four implementations, so the even-odd variant's algorithmic FLOP reduction
  appears as a wall-clock gain rather than a different denominator.

`mf-kernels`' cross-element SIMD batching (4 cells per AVX2 register) is most
directly comparable to libCEED's blocked CPU backends, which also batch elements
into vector lanes. MFEM native PA is the general, less hand-tuned reference.

## Honest scope

`mf-kernels` is a contraction-focused kernel set, not a finite element library.
The comparison measures one operator apply on a structured mesh with constant
geometry; it does not exercise unstructured meshes, mixed orders, assembly, or
the solver stack that MFEM/libCEED provide. Where `mf-kernels` is competitive it
is competitive on this specific hot path, which is the honest claim. See
`methodology_flags.md` for the full list of caveats and open questions.
