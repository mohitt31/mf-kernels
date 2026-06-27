# Methodology flags and open questions

Things a careful reviewer (and Dr. Kolev) should know before reading the numbers.
Listed weaknesses-first on purpose.

## Comparison fairness

1. **Three different parallelization strategies, not one.** `mf-kernels` batches
   4 distinct *cells* into each AVX2 register (SoA-of-4, deal.II style). libCEED's
   `/blocked` backends also batch elements into lanes, so those two are the most
   apples-to-apples pair. MFEM native PA vectorizes differently and is the more
   general reference. Reading the gap between mf-kernels and libCEED-blocked is
   more meaningful than the gap to MFEM native.

2. **mf-kernels is a kernel set, not a library.** The benchmark wraps it in the
   full operator on a *structured, constant-geometry* mesh. Real workloads have
   curved/unstructured geometry (per-quad metric varies, more data movement),
   mixed orders, and the assembly/solver stack. None of that is exercised here.
   A win on this hot path does not imply a win at application level.

3. **Quadrature convention.** We use the CEED BP rule `n_q = p+2` (rule order
   `2p+3`), which makes the 1D operator rectangular `(p+2) x (p+1)`. If a
   collocated `n_q = n_d = p+1` rule were used instead, the contraction counts,
   the even-odd structure, and the relative standings would all shift. The choice
   here matches libCEED's BP examples so the comparison is against the convention
   MFEM/libCEED actually benchmark.

## DOF and FLOP accounting

4. **Throughput is unique (true) DOFs/sec.** On the structured grid that is
   `(n p + 1)^3`. The element-vector (E-vector) DOF count is larger because of
   shared interface dofs; we report unique dofs to match CEED. Stating which one
   is used matters because the ratio grows with `p`.

5. **GFLOP/s uses a single shared "standard sum-fact" FLOP count** for every
   implementation, including even-odd (which does fewer real FLOPs). This is
   deliberate and matches mf-kernels' existing convention: it turns an
   algorithmic FLOP reduction into a visible wall-clock gain rather than hiding
   it in a smaller denominator. It also means the GFLOP/s number is a throughput
   proxy, not a hardware-efficiency measurement; do not compare it to peak FMA
   without that caveat. MFEM/libCEED GFLOP/s are not emitted to avoid mixing two
   different internal FLOP conventions on one axis; their throughput is compared
   on DOFs/sec.

## Correctness

6. **The 1e-13 check is a same-inputs check.** `diff_bp` feeds mf-kernels the
   basis, quad data, and restriction map exported from MFEM, so agreement to
   round-off validates that the two compute the same operator, not that
   mf-kernels independently reproduces MFEM's discretization. The independent
   check is `validate_bp` (vs a dense reference built from a self-contained
   GLL/GL basis), which passes to ~9e-15.

7. **Restriction replication.** The element gather/scatter uses MFEM's
   `ElementRestriction` with `LEXICOGRAPHIC` ordering, assumed conforming with no
   essential BCs (true for the BP setup). On the first real run, confirm a single
   element matches before trusting the sweep; this is the most likely spot for a
   layout mismatch.

## Timing

8. **Single core first, then single socket.** Headline numbers are pinned to one
   core (`taskset` + `OMP_PLACES=cores`). Single-socket scaling for mf-kernels
   would need a thread-safe scatter (coloring or atomics); MFEM/libCEED already
   have this. Multi-thread mf-kernels is out of scope for this round and is
   flagged as future work, not silently compared.

9. **Memory-bandwidth regime.** At high `p` the full operator may be
   bandwidth-bound rather than FLOP-bound; the even-odd FLOP saving then matters
   less than at the bare-contraction level. The dry runs already show even-odd's
   advantage shrinking once gather/scatter is included. A STREAM roofline on the
   benchmark node would make this quantitative and is worth adding.

10. **AVX-512.** If the EPYC node is Zen4 (AVX-512 capable), `mf-kernels`'
    AVX2-intrinsic kernels leave the wider unit unused while libCEED/MFEM may use
    it. Report the ISA actually compiled (`-march=native`) and note this; the
    earlier Codespace run was AVX2-only.

## Open questions for Dr. Kolev

- Is the `q = p+2` rule the convention you want compared against, or would you
  also like a collocated `q = p+1` variant?
- For the throughput axis, do you prefer unique-DOF/sec only, or unique plus
  E-vector for context?
- Is single-core the comparison you care about first, or should single-socket
  (with a colored mf-kernels scatter) be in scope for the first pass?
