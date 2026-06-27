#!/usr/bin/env bash
# build.sh - build everything needed for the comparison on the benchmark machine.
#
# Builds, in order: LIBXSMM, libCEED (with LIBXSMM), MFEM (serial, with libCEED),
# then the benchmark binaries. Tuned for a single-socket AMD EPYC node with
# gcc 13.3, -O3 -march=native. Edit JOBS / paths as needed.
#
# Re-run is incremental: it skips a dependency if its install marker exists.
set -euo pipefail

ROOT="${ROOT:-$HOME/bp-bench}"
JOBS="${JOBS:-$(nproc)}"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
OPTFLAGS="${OPTFLAGS:--O3 -march=native}"
mkdir -p "$ROOT"
cd "$ROOT"

echo "==> toolchain"; $CXX --version | head -1

# ---- LIBXSMM ----
if [ ! -f "$ROOT/libxsmm/lib/libxsmm.a" ]; then
  echo "==> LIBXSMM"
  [ -d libxsmm ] || git clone --depth 1 https://github.com/libxsmm/libxsmm.git
  make -C libxsmm -j"$JOBS" STATIC=1 CC="$CC" CXX="$CXX" FC= \
    BLAS=0 INTRINSICS=2
fi
export LIBXSMM_DIR="$ROOT/libxsmm"

# ---- libCEED (with LIBXSMM CPU backends) ----
if [ ! -f "$ROOT/libCEED/lib/libceed.so" ] && [ ! -f "$ROOT/libCEED/lib/libceed.a" ]; then
  echo "==> libCEED"
  [ -d libCEED ] || git clone --depth 1 https://github.com/CEED/libCEED.git
  make -C libCEED -j"$JOBS" \
    CC="$CC" CXX="$CXX" OPT="$OPTFLAGS" \
    XSMM=1 XSMM_DIR="$LIBXSMM_DIR"
fi
export CEED_DIR="$ROOT/libCEED"

# ---- MFEM (serial, partial assembly, libCEED enabled) ----
if [ ! -f "$ROOT/mfem/libmfem.a" ]; then
  echo "==> MFEM"
  [ -d mfem ] || git clone --depth 1 https://github.com/mfem/mfem.git
  make -C mfem config \
    MFEM_USE_CEED=YES CEED_DIR="$CEED_DIR" \
    MFEM_USE_OPENMP=YES \
    CXXFLAGS="$OPTFLAGS -std=c++17" CXX="$CXX"
  make -C mfem -j"$JOBS"
fi
export MFEM_DIR="$ROOT/mfem"

# ---- benchmark binaries ----
echo "==> benchmark binaries"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
make -C "$HERE" mfk
make -C "$HERE" mfem MFEM_DIR="$MFEM_DIR"

echo "==> done. binaries in $HERE/results/"
echo "    MFEM_DIR=$MFEM_DIR"
