#!/usr/bin/env bash
# env_capture.sh - record the exact machine + toolchain for the write-up.
# Writes results/env.txt. Run on the benchmark node before the benchmarks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/results/env.txt"
mkdir -p "$HERE/results"
{
  echo "=== date ==="; date -u
  echo; echo "=== uname ==="; uname -a
  echo; echo "=== compiler ==="; ${CXX:-g++} --version | head -2
  echo; echo "=== CPU (lscpu) ==="; lscpu 2>/dev/null
  echo; echo "=== flags (avx2/avx512/fma) ==="
  grep -m1 -o -E "avx2|avx512f|avx512dq|avx512vl|fma" /proc/cpuinfo | sort -u | tr '\n' ' '; echo
  echo; echo "=== cache ==="; lscpu -C 2>/dev/null || cat /sys/devices/system/cpu/cpu0/cache/index*/size 2>/dev/null
  echo; echo "=== NUMA topology ==="; numactl --hardware 2>/dev/null || echo "numactl not available"
  echo; echo "=== governor ==="; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
} | tee "$OUT"
echo "wrote $OUT"
