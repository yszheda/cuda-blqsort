// API implementations with explicit instantiations
// This .cu file is compiled by nvcc and provides the actual sort functions
#include "blqs.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include "blqs_kernels.hpp"
#include <type_traits>
#include <stdexcept>
#include <vector>

using namespace blqs::detail;

namespace blqs {

using int_cmp   = int (*)(const int&, const int&);
using float_cmp = int (*)(const float&, const float&);
using int64_cmp = int (*)(const int64_t&, const int64_t&);
using dbl_cmp   = int (*)(const double&, const double&);

// ── sort driver (generic, instantiated for each type+comparator) ────────────

template <typename T, typename Comparator>
void do_sort(T* d_data, int n, Comparator cmp) {
    if (n <= 1) return;
    if (n <= BLOCK_SIZE) {
        int threads = (n < BLOCK_SIZE) ? n : BLOCK_SIZE;
        quicksort_shared<T, 1024, Comparator><<<1, threads>>>(d_data, n, cmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int* d_bucket_counts = nullptr;
    int num_buckets = 2;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_bucket_counts, num_blocks * num_buckets * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, num_blocks * num_buckets * sizeof(int)));
    partition_kernel<T, Comparator><<<num_blocks, BLOCK_SIZE>>>(
        d_data, n, d_bucket_counts, nullptr, num_buckets, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_bucket_counts(num_blocks * num_buckets);
    CUDA_CHECK(cudaMemcpy(h_bucket_counts.data(), d_bucket_counts,
        num_blocks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bucket_counts));
    for (int b = 0; b < num_blocks; b++) {
        int chunk_start = b * BLOCK_SIZE;
        int chunk_size = ((n - chunk_start) < BLOCK_SIZE) ? (n - chunk_start) : BLOCK_SIZE;
        int lower_count = h_bucket_counts[b * num_buckets + 0];
        if (lower_count > 1) do_sort(d_data + chunk_start, lower_count, cmp);
        if (chunk_size - lower_count > 1) do_sort(d_data + chunk_start + lower_count, chunk_size - lower_count, cmp);
    }
}

// ── KV sort driver ────────────────────────────────────────────────────────

template <typename K, typename V, typename Comparator>
void do_sort_by_key(K* d_keys, V* d_values, int n, Comparator cmp) {
    if (n <= 1) return;
    if (n <= BLOCK_SIZE) {
        int threads = (n < BLOCK_SIZE) ? n : BLOCK_SIZE;
        kv_block_sort_kernel<K, V, Comparator><<<1, threads>>>(d_keys, d_values, n, cmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    int* d_bucket_counts = nullptr;
    int num_buckets = 2;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_bucket_counts, num_blocks * num_buckets * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, num_blocks * num_buckets * sizeof(int)));
    kv_partition_kernel<K, V, Comparator><<<num_blocks, BLOCK_SIZE>>>(
        d_keys, d_values, n, d_bucket_counts, nullptr, num_buckets, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_bucket_counts(num_blocks * num_buckets);
    CUDA_CHECK(cudaMemcpy(h_bucket_counts.data(), d_bucket_counts,
        num_blocks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bucket_counts));
    for (int b = 0; b < num_blocks; b++) {
        int chunk_start = b * BLOCK_SIZE;
        int chunk_size = ((n - chunk_start) < BLOCK_SIZE) ? (n - chunk_start) : BLOCK_SIZE;
        int lower_count = h_bucket_counts[b * num_buckets + 0];
        if (lower_count > 1) do_sort_by_key(d_keys + chunk_start, d_values + chunk_start, lower_count, cmp);
        if (chunk_size - lower_count > 1) do_sort_by_key(d_keys + chunk_start + lower_count, d_values + chunk_start + lower_count, chunk_size - lower_count, cmp);
    }
}

// ── Public API implementations ────────────────────────────────────────────

template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator cmp) {
    if (n <= 0) return;
    if (d_data == nullptr) throw std::invalid_argument("sort: d_data is nullptr");
    do_sort(d_data, n, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator cmp) {
    if (n <= 0) return;
    if (d_data == nullptr) throw std::invalid_argument("sort_stable: d_data is nullptr");
    if constexpr (std::is_arithmetic<T>::value) {
        detail::radix_sort_driver<T, void>(d_data, nullptr, n);
    }
    (void)cmp;
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr)
        throw std::invalid_argument("sort_by_key: keys or values is nullptr");
    do_sort_by_key(d_keys, d_values, n, [] __device__ (const K& a, const K& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr)
        throw std::invalid_argument("sort_by_key_stable: keys or values is nullptr");
    if constexpr (std::is_arithmetic<K>::value) {
        detail::radix_sort_driver<K, V>(d_keys, d_values, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Explicit instantiations for function pointer comparators ──────────────

template void sort<int, int_cmp>(int*, int, int_cmp);
template void sort<float, float_cmp>(float*, int, float_cmp);
template void sort<int64_t, int64_cmp>(int64_t*, int, int64_cmp);
template void sort<double, dbl_cmp>(double*, int, dbl_cmp);

template void sort_stable<int, int_cmp>(int*, int, int_cmp);
template void sort_stable<float, float_cmp>(float*, int, float_cmp);
template void sort_stable<int64_t, int64_cmp>(int64_t*, int, int64_cmp);
template void sort_stable<double, dbl_cmp>(double*, int, dbl_cmp);

template void sort_by_key<int, int>(int*, int*, int);
template void sort_by_key<float, int>(float*, int*, int);
template void sort_by_key<float, float>(float*, float*, int);
template void sort_by_key<int64_t, int64_t>(int64_t*, int64_t*, int);
template void sort_by_key<double, double>(double*, double*, int);

template void sort_by_key_stable<int, int>(int*, int*, int);
template void sort_by_key_stable<float, int>(float*, int*, int);
template void sort_by_key_stable<int64_t, int64_t>(int64_t*, int64_t*, int);
template void sort_by_key_stable<double, double>(double*, double*, int);

} // namespace blqs
