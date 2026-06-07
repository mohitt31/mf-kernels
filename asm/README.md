# Assembly Verifications

This directory contains the assembly output demonstrating the compiler behaviors described in the main `README.md`.

## `pitfall_vs_naive.s`

Generated with:
```bash
g++ -S -O3 -march=native -I include -x c++ ...
```

**Compiler:** gcc 13.3.0
**Architecture:** x86-64, AVX2 (AMD EPYC)

This file confirms that the "helpful" `pitfall` variant causes `gcc 13.3.0` to lower the broadcasts using `vpermpd` (8 occurrences in the loop), whereas the `naive` variant compiles to a clean `vbroadcastsd` (1 occurrence). This difference is the source of the 3–4× performance degradation.
