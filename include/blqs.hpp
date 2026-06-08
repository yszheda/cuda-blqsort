#pragma once
#include "blqs_types.hpp"
#include "blqs_errors.hpp"
#include "blqs_impl.hpp"
#include "blqs_keyvalue_impl.hpp"
#include <type_traits>
#include <stdexcept>

namespace blqs {

// ── Struct-based sort (unstable) ──────────────────────────────────────────

template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator cmp = Comparator()) {
    if (n <= 0) return;
    if (d_data == nullptr) {
        throw std::invalid_argument("sort: d_data is nullptr");
    }
    detail::sort_driver(d_data, n, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Struct-based sort (stable) ────────────────────────────────────────────

template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator cmp = Comparator()) {
    if (n <= 0) return;
    if (d_data == nullptr) {
        throw std::invalid_argument("sort_stable: d_data is nullptr");
    }
    // Use radix sort for numeric types (stable by nature)
    if constexpr (std::is_arithmetic<T>::value) {
        detail::radix_sort_driver<T, void>(d_data, nullptr, n);
    } else {
        // For custom types, use stable merge sort
        detail::stable_merge_sort(d_data, n, cmp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Key-value sort (unstable) ─────────────────────────────────────────────

template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr) {
        throw std::invalid_argument("sort_by_key: keys or values is nullptr");
    }
    auto cmp = [] __device__ (const K& a, const K& b) { return a < b; };
    detail::sort_by_key_driver(d_keys, d_values, n, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Key-value sort (stable) ───────────────────────────────────────────────

template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr) {
        throw std::invalid_argument("sort_by_key_stable: keys or values is nullptr");
    }
    // Stable sort uses radix sort for numeric keys
    if constexpr (std::is_arithmetic<K>::value) {
        detail::radix_sort_driver<K, V>(d_keys, d_values, n);
    } else {
        // For custom key types, use stable merge sort
        detail::stable_sort_by_key_driver(d_keys, d_values, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace blqs
