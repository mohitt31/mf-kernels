# mf-kernels vs MFEM partial assembly and libCEED: a like-for-like benchmark

**Author:** Mohit Prajapati · **Status:** draft, numbers pending the AMD EPYC run
**Scope:** CEED BP1 (mass) and BP3 (Poisson), 3D hex, order `p = 1..8`, single
core then single socket.

> Placeholders are written as `[[…]]`. Every `[[…]]` is filled from
> `results/env.txt`, `results/mfk.csv`, `results/mfem.csv`, and the `diff_*`
> logs produced by `scripts/run.sh`. Do not hand-edit numbers; re-run and paste.

## 1. Purpose

`mf-kernels` is a small set of hand-tuned matrix-free tensor-product contraction
kernels. The question this benchmark answers is narrow and concrete: on the two
standard CEED bake-off operators, how does the `mf-kernels` apply compare, on the
same machine and with identical accounting, to MFEM's native partial assembly and
to libCEED's CPU backends? The goal is an honest like-for-like measurement, not a
headline.

## 2. Setup

**Hardware / toolchain (from `results/env.txt`).**
CPU `[[model]]`, `[[sockets]]` socket(s), `[[cores]]` cores, `[[L1/L2/L3]]` cache,
ISA `[[avx2 | avx2+avx512]]`. Compiler `[[gcc 13.3.x]]`, flags `-O3 -march=native`.
Frequency governor `[[performance]]`. Single-core runs pinned with
`taskset -c 0`, `OMP_NUM_THREADS=1`, `OMP_PLACES=cores`.

**Discretization.** Order-`p` Gauss-Lobatto nodal H1 basis (`n_d = p+1` nodes/dim).
Gauss-Legendre quadrature, `n_q = p+2` points/dim (rule order `2p+3`), the CEED BP
convention. The 1D operator is therefore rectangular `(p+2) x (p+1)`. Structured
`n x n x n` hex mesh on the unit cube, `n` chosen so the unique-DOF count is
`~[[target]]` per `p`.

**Operators.**
- BP1 (mass): `v = G^T B^T (w·detJ) B G u`.
- BP3 (Poisson): `v = G^T B^T D B G u`, `D` the symmetric `3x3` metric
  `w·detJ·J^{-1}J^{-T}` per quadrature point.

`G` is the element gather/scatter. On the benchmark machine `B`, `dB`, `D`, and
`G` are exported from MFEM and reused by `mf-kernels`, so the two run identical
numerics and differ only in arithmetic.

**Implementations.** (1) `mf-kernels` apply, variants naive / avx2 / avx2_blocked
/ even-odd; (2) MFEM native PA; (3) libCEED `/cpu/self/avx/blocked`; (4) libCEED
`/cpu/self/xsmm/blocked`. MFEM built with libCEED + LIBXSMM, same compiler/flags.

**Accounting.** Throughput in unique global DOFs/sec. GFLOP/s (mf-kernels axis)
uses one shared standard sum-factorization FLOP count across all variants, so the
even-odd algorithmic saving shows as a wall-clock gain. Timing is warmup + `[[reps]]`
repeats, median with inter-quartile spread.

## 3. Correctness

Two independent checks.

- **MFEM-independent** (`validate_bp`): the sum-factorized BP1/BP3 operator vs a
  dense `O(p^6)` reference built from a self-contained GLL/GL basis, all four
  variants, `p = 1..8`. Worst max-abs difference `8.9e-15` (BP3, `p = 8`); BP1
  at `~5e-17`; naive vs avx2 bitwise identical.
- **Against MFEM** (`diff_bp`): `mf-kernels` applied to MFEM's exported basis,
  quad data, and restriction, on MFEM's own input vector, vs MFEM's output.

| p | BP1 max-abs | BP3 max-abs |
|---|-------------|-------------|
| 1 | `[[…]]` | `[[…]]` |
| … | `[[…]]` | `[[…]]` |
| 8 | `[[…]]` | `[[…]]` |

All `<= 1e-13` (threshold met / not met: `[[…]]`).

## 4. Results

Numbers pasted from `results/mfk.csv` and `results/mfem.csv`; figures in
`results/`.

**BP1, single core, unique DOFs/sec (G).**

| p | mf-k avx2 | mf-k blocked | mf-k even-odd | MFEM PA | libCEED avx | libCEED xsmm |
|---|-----------|--------------|---------------|---------|-------------|--------------|
| 1 | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` |
| … | | | | | | |
| 8 | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` |

**BP3, single core, unique DOFs/sec (G).**

| p | mf-k avx2 | mf-k blocked | mf-k even-odd | MFEM PA | libCEED avx | libCEED xsmm |
|---|-----------|--------------|---------------|---------|-------------|--------------|
| 1 | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` |
| … | | | | | | |
| 8 | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` | `[[…]]` |

Single-socket (all cores) summary: `[[…]]` (mf-kernels single-socket only if the
colored scatter is in; otherwise report MFEM/libCEED socket scaling and note
mf-kernels socket result is deferred).

![BP1 throughput](results/dofs_per_sec_bp1.png)
![BP3 throughput](results/dofs_per_sec_bp3.png)

## 5. Interpretation (honest)

- Where `mf-kernels` lands relative to **libCEED blocked** is the fair headline,
  since both batch elements into SIMD lanes. Expected pattern from the dry runs:
  `mf-kernels` is competitive on the contraction, with the gap narrowing once
  gather/scatter is included (the full operator is more memory-bound than the
  bare kernel). Fill the actual standing here: `[[mf-kernels is within X% of /
  ahead of / behind libCEED-blocked at p = …]]`.
- The even-odd variant's advantage over plain avx2 **shrinks at the full-operator
  level** compared to the bare contraction, because gather/scatter and the
  pointwise `D` apply dilute the contraction's share of the runtime. State the
  crossover `p` observed: `[[…]]`.
- MFEM native PA is the general reference; any `mf-kernels` lead over it reflects
  hand-tuning on a narrow path, not a better finite element method.
- No claim is made that `mf-kernels` beats MFEM or libCEED as a system. The
  measurement is one operator apply on a structured, constant-geometry mesh.

## 6. Threats to validity

See `methodology_flags.md`. Most load-bearing: the `q = p+2` convention, the
three different SIMD strategies, unique-vs-E-vector DOF counting, the shared
GFLOP/s convention, and (if Zen4) AVX2-only `mf-kernels` vs AVX-512-capable
references.

## 7. Reproduce

```bash
bash scripts/build.sh      # LIBXSMM + libCEED + MFEM + binaries
bash scripts/run.sh        # env capture, both sides, 1e-13 check, plots
```
