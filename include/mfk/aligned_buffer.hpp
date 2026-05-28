// SPDX-License-Identifier: MIT
// mfk/aligned_buffer.hpp — small RAII wrapper around 32-byte aligned storage.
//
// AVX2 loads/stores want 32-byte alignment. std::vector does not guarantee this.
// posix_memalign / std::aligned_alloc both work; we wrap one in a unique_ptr
// deleter so it composes cleanly with std::unique_ptr.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace mfk {

inline constexpr std::size_t cacheline_bytes = 64;
inline constexpr std::size_t avx2_align      = 32;

// std::aligned_alloc requires size to be a multiple of alignment on some libcs.
// Round up to be safe.
inline std::size_t round_up(std::size_t n, std::size_t a) noexcept {
  return (n + a - 1) / a * a;
}

struct AlignedDeleter {
  void operator()(void* p) const noexcept { std::free(p); }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter>;

template <typename T>
inline AlignedPtr<T> make_aligned(std::size_t n_elems, std::size_t align = avx2_align) {
  static_assert(std::is_trivial_v<T>, "make_aligned is intended for POD types");
  const std::size_t bytes = round_up(n_elems * sizeof(T), align);
  void* raw = std::aligned_alloc(align, bytes);
  if (!raw) throw std::bad_alloc{};
  std::memset(raw, 0, bytes);
  return AlignedPtr<T>(static_cast<T*>(raw));
}

} // namespace mfk
