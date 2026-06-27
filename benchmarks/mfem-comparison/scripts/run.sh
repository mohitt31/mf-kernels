#!/usr/bin/env bash
# run.sh - run the full comparison, pinned to a single core, single socket.
#
# Produces:
#   results/env.txt              machine + toolchain
#   results/mfk.csv              mf-kernels (naive/avx2/blocked/evenodd), BP1+BP3
#   results/mfem.csv             MFEM native PA + libCEED avx + libCEED xsmm, BP1+BP3
#   results/export/              small-mesh dump for the correctness check
#   results/diff_*.txt           1e-13 correctness logs
#
# Pinning: taskset -c $CORE plus OMP_NUM_THREADS=1 and OMP_PLACES=cores so the
# single-core numbers are clean. Edit CORE / TARGET as needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$HERE/results"; mkdir -p "$RES" "$RES/export"
CORE="${CORE:-0}"
TARGET="${TARGET:-2000000}"
REPS="${REPS:-21}"
PIN="taskset -c $CORE"
export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close

echo "==> environment"
bash "$HERE/scripts/env_capture.sh" >/dev/null || true

echo "==> mf-kernels"
: > "$RES/mfk.csv"
$PIN "$RES/bench_bp" --op both --target "$TARGET" --reps "$REPS" --csv "$RES/mfk.csv"

echo "==> MFEM native PA + libCEED backends"
: > "$RES/mfem.csv"
echo "op,p,device,NE,uDOF,median_s,iqr_s,dofs_per_s" > "$RES/mfem.csv"
for OP in bp1 bp3; do
  for DEV in "cpu" "ceed-cpu:/cpu/self/avx/blocked" "ceed-cpu:/cpu/self/xsmm/blocked"; do
    echo "    [$OP] $DEV"
    $PIN "$RES/mfem_bp" --op "$OP" --device "$DEV" --target "$TARGET" --reps "$REPS" --csv "$RES/mfem.csv"
  done
done

echo "==> correctness export (small mesh) + diff"
for OP in bp1 bp3; do
  $PIN "$RES/mfem_bp" --op "$OP" --device cpu --export "$RES/export" --n 3
  for V in naive avx2 blocked evenodd; do
    $PIN "$RES/diff_bp" "$RES/export" "$OP" "$V" | tee "$RES/diff_${OP}_${V}.txt"
  done
done

echo "==> plots"
python3 "$HERE/scripts/plots.py" "$RES" || echo "plots skipped (matplotlib?)"
echo "==> done. see $RES/"
