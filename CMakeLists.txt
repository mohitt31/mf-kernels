cmake_minimum_required(VERSION 3.16)
project(mf_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Header-only library target.
add_library(mfk INTERFACE)
target_include_directories(mfk INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# -march=native turns on AVX2/FMA on any Haswell-or-later x86 and lets the
# compiler pick the best matching tune flags. On non-x86 hosts (Apple silicon,
# ARM Linux) the SIMD code paths in mfk are #if-guarded and silently skipped.
if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "-O")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" MFK_HAS_MARCH_NATIVE)
if(MFK_HAS_MARCH_NATIVE)
  target_compile_options(mfk INTERFACE -march=native)
endif()
check_cxx_compiler_flag("-mavx2" MFK_HAS_AVX2)
check_cxx_compiler_flag("-mfma"  MFK_HAS_FMA)
if(MFK_HAS_AVX2)
  target_compile_options(mfk INTERFACE -mavx2)
endif()
if(MFK_HAS_FMA)
  target_compile_options(mfk INTERFACE -mfma)
endif()

# Warnings for our own code only — not propagated via the INTERFACE target so
# downstream users aren't pelted with our preferences.
add_library(mfk_warnings INTERFACE)
target_compile_options(mfk_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wcast-align)

# --- tests ---
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE mfk mfk_warnings)
add_test(NAME correctness COMMAND test_correctness)

# --- bench ---
add_executable(bench_sumfact bench/bench_sumfact.cpp)
target_link_libraries(bench_sumfact PRIVATE mfk mfk_warnings)
