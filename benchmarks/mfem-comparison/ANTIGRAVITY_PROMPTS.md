# Antigravity prompts — machine-side steps

Hand these to Antigravity one block at a time, on the AMD EPYC box (or the EPYC
GitHub Codespace), arm64 Mac is NOT the target for the final numbers. Each block
is self-contained. Stop and check the stated success condition before the next.

---

## 0. Sanity: confirm the box and toolchain

```
Run: lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)'; gcc --version | head -1;
     grep -m1 -o -E 'avx2|avx512f' /proc/cpuinfo | sort -u
Tell me the CPU model, socket/core count, whether avx512f is present, and the gcc version.
```
Success: gcc is 13.3.x and the CPU is the EPYC we expect. If avx512f shows up,
flag it (mf-kernels is AVX2-only; note it for the write-up).

---

## 1. Pull the benchmark folder (already in the repo)

```
cd into the mf-kernels checkout. Confirm benchmarks/mfem-comparison/ exists with
src/, include/, scripts/, Makefile. If the repo is not cloned yet:
git clone https://github.com/mohitt31/mf-kernels.git && cd mf-kernels
```
Success: `ls benchmarks/mfem-comparison/src` shows the four .cpp files.

---

## 2. Build the mf-kernels side first (no external deps) and validate

```
cd benchmarks/mfem-comparison
make mfk
taskset -c 0 ./results/validate_bp
```
Success: last line is `PASS: sum-fact operator == dense reference to ~1e-13`.
If it fails, stop and send me the full output — do not proceed.

---

## 3. Build the dependency stack (LIBXSMM + libCEED + MFEM)

```
cd benchmarks/mfem-comparison
export ROOT=$HOME/bp-bench
bash scripts/build.sh
```
This clones and builds LIBXSMM, libCEED (with XSMM), MFEM (with libCEED + OpenMP),
then mfem_bp. It is incremental. If MFEM's config step rejects a flag, paste the
exact error to me before changing anything.
Success: `ls results/mfem_bp` exists and `./results/mfem_bp --device cpu --op bp1 --target 50000`
prints per-p lines without crashing.

---

## 4. Verify the libCEED backends actually exist in this build

```
cd benchmarks/mfem-comparison
./results/mfem_bp --device "ceed-cpu:/cpu/self/avx/blocked" --op bp1 --target 50000 | head
./results/mfem_bp --device "ceed-cpu:/cpu/self/xsmm/blocked" --op bp1 --target 50000 | head
```
Success: both run and `Device::Print()` near the top shows the ceed backend was
selected (not silently fell back to cpu). If a backend string is rejected, send me
the Device::Print() output so I can correct the resource path.

---

## 5. Correctness against MFEM (the 1e-13 check)

```
cd benchmarks/mfem-comparison
./results/mfem_bp --op bp1 --device cpu --export results/export --n 3
./results/mfem_bp --op bp3 --device cpu --export results/export --n 3
for V in naive avx2 blocked evenodd; do ./results/diff_bp results/export bp1 $V; done
for V in naive avx2 blocked evenodd; do ./results/diff_bp results/export bp3 $V; done
```
Success: every line ends `PASS (threshold 1e-13)`. If any FAIL, send me the full
diff_bp output for that case — this is almost certainly the restriction-map layout
(flag #7 in methodology_flags.md) and I will patch mfem_bp's export.

---

## 6. Full pinned run + plots

```
cd benchmarks/mfem-comparison
CORE=0 TARGET=2000000 REPS=21 bash scripts/run.sh
```
Success: results/ now has env.txt, mfk.csv, mfem.csv, diff_*.txt, and the .png
plots. Send me all of results/*.csv and results/env.txt and the diff_*.txt logs.
I will fill the write-up and the Kolev email from them.

---

## 7. Commit

```
cd <repo root>
git add benchmarks/mfem-comparison
git commit -m "Add MFEM PA + libCEED vs mf-kernels BP1/BP3 benchmark (harness, correctness, scripts)"
git push
```
Do NOT commit results/ binaries or results/export (they are gitignored). The CSVs,
env.txt, and plots may be committed if we want them in the repo; ask me first.

---

### If anything stalls
- MFEM config failing on libCEED: send me the `make config` tail.
- A libCEED backend missing: send `Device::Print()` output.
- diff_bp FAIL: send the per-p maxabs/maxrel lines.
Each of these I can fix from the message without you debugging it.
