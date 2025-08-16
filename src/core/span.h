// Copyright (c) 2025 yanghuafang
// SPDX-License-Identifier: MIT

#ifndef ACCEL_CORE_SPAN_H_
#define ACCEL_CORE_SPAN_H_

#include <cstddef>
#include <type_traits>
#include <vector>

namespace accel {

// Non-owning, bounds-carrying view over a contiguous range of `T`.
//
// A deliberately minimal stand-in for C++20's `std::span`. The project pins a
// single language level (C++17) for both host translation units and NVCC
// device compilation, so the standard type is not available; rolling a 60-line
// view is cheaper than fragmenting the codebase across two language levels.
//
// Ownership: none. The referenced storage must outlive every Span that refers
// to it — passing a Span constructed from a temporary `std::vector` into a
// function that stores it is a dangling-reference bug the type cannot detect.
//
// Thread-safety: as thread-safe as the underlying storage. Concurrent reads
// through distinct `Span<const T>` instances are safe.
//
//   T  Element type. Use `Span<const T>` for read-only views; the
//      conversion from `Span<T>` to `Span<const T>` is implicit.
template <typename T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using pointer = T*;
  using reference = T&;
  using iterator = T*;

  constexpr Span() noexcept = default;

  // Precondition: `data` is non-null whenever `size` is non-Zero.
  constexpr Span(pointer data, size_type size) noexcept
      : data_(data), size_(size) {}

  // Binds a mutable vector. SFINAE keeps `Span<float>` from binding a
  // `std::vector<double>` or a const vector.
  template <typename U, typename = std::enable_if_t<
                            std::is_convertible<U (*)[], T (*)[]>::value>>
  Span(std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

  // Binds a const vector; only viable when `T` is itself const-qualified.
  template <typename U, typename = std::enable_if_t<
                            std::is_convertible<const U (*)[], T (*)[]>::value>>
  Span(const std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

  // Qualification conversion `Span<U>` -> `Span<const U>`.
  template <typename U, typename = std::enable_if_t<
                            std::is_convertible<U (*)[], T (*)[]>::value>>
  constexpr Span(const Span<U>& other) noexcept
      : data_(other.data()), size_(other.size()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type SizeBytes() const noexcept { return size_ * sizeof(T); }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }

  // Unchecked. Callers validate extents once, up front, rather than
  //          paying a branch per element in the GEMM inner loops.
  constexpr reference operator[](size_type index) const noexcept {
    return data_[index];
  }

 private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

}  // namespace accel

#endif  // ACCEL_CORE_SPAN_H_
