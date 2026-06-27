# Makefile - mf-kernels vs MFEM benchmark
#
# mf-kernels-only targets (validate_bp, bench_bp, diff_bp) need no external deps.
# The mfem_bp target needs a real MFEM build configured with libCEED + LIBXSMM;
# point MFEM_DIR at it (an install prefix with include/ and lib/, or a build dir
# with a config.mk). See scripts/build.sh for the full recipe.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -funroll-loops -Wall
INCLUDE  := -Iinclude -I../../include/mfk
OUT      := results

MFEM_DIR ?= $(HOME)/mfem
MFEM_INC := -I$(MFEM_DIR)/include
MFEM_LIB := -L$(MFEM_DIR)/lib -lmfem

.PHONY: all mfk mfem clean
all: mfk

# --- mf-kernels side (self-contained) ---
mfk: $(OUT)/validate_bp $(OUT)/bench_bp $(OUT)/diff_bp

$(OUT)/validate_bp: src/validate_bp.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@
$(OUT)/bench_bp: src/bench_bp.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@
$(OUT)/diff_bp: src/diff_bp.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@

# --- MFEM side (needs MFEM_DIR) ---
mfem: $(OUT)/mfem_bp
$(OUT)/mfem_bp: src/mfem_bp.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) $(MFEM_INC) $(INCLUDE) $< -o $@ $(MFEM_LIB)

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -f $(OUT)/validate_bp $(OUT)/bench_bp $(OUT)/diff_bp $(OUT)/mfem_bp
