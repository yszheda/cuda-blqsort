// Template declarations for API functions (struct-based sort)
// All kernel definitions are in blqs_kernels.hpp (header-only)
#pragma once
#include <cuda_runtime.h>
#include "blqs_config.hpp"
#include "blqs_errors.hpp"
#include "blqs_kernels.hpp"

namespace blqs {
namespace detail {

// ── Sort Driver (implementation in header for on-demand template instantiation) ──

template <typename T, typename Comparator>
void sort_driver(T* d_data, int n, Comparator cmp) {
    if (n <= 1) return;

    if (n <= BLOCK_SIZE) {
        int threads = min(n, BLOCK_SIZE);
        int blocks = 1;
        quicksort_shared<T, 1024, Comparator><<<blocks, threads>>>(d_data, n, cmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int* d_bucket_counts = nullptr;
    int* h_bucket_counts = nullptr;
    int num_buckets = 2;
    int num_blocks = get_num_blocks(n);

    CUDA_CHECK(cudaMalloc(&d_bucket_counts, num_blocks * num_buckets * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, num_blocks * num_buckets * sizeof(int)));
    h_bucket_counts = new int[num_blocks * num_buckets];

    partition_kernel<T, Comparator><<<num_blocks, BLOCK_SIZE>>>(
        d_data, n, d_bucket_counts, nullptr, num_buckets, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_bucket_counts, d_bucket_counts,
        num_blocks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bucket_counts));

    for (int b = 0; b < num_blocks; b++) {
        int chunk_start = b * BLOCK_SIZE;
        int chunk_size = min(BLOCK_SIZE, n - chunk_start);
        int lower_count = h_bucket_counts[b * num_buckets + 0];

        if (lower_count > 1) sort_driver(d_data + chunk_start, lower_count, cmp);
        if (chunk_size - lower_count > 1) sort_driver(d_data + chunk_start + lower_count, chunk_size - lower_count, cmp);
    }

    delete[] h_bucket_counts;
}

template <typename T, typename Comparator>
void stable_merge_sort(T* d_data, int n, Comparator cmp) {
    radix_sort_driver<T, void>(d_data, nullptr, n);
    (void)cmp;
}

// ── KV Sort Driver ────────────────────────────────────────────────────────

template <typename K, typename V, typename Comparator>
void sort_by_key_driver(K* d_keys, V* d_values, int n, Comparator cmp) {
    if (n <= 1) return;

    if (n <= BLOCK_SIZE) {
        int threads = min(n, BLOCK_SIZE);
        kv_block_sort_kernel<K, V, Comparator><<<1, threads>>>(d_keys, d_values, n, cmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    int* d_bucket_counts = nullptr;
    int* h_bucket_counts = nullptr;
    int num_buckets = 2;
    int num_blocks = get_num_blocks(n);

    CUDA_CHECK(cudaMalloc(&d_bucket_counts, num_blocks * num_buckets * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, num_blocks * num_buckets * sizeof(int)));
    h_bucket_counts = new int[num_blocks * num_buckets];

    kv_partition_kernel<K, V, Comparator><<<num_blocks, BLOCK_SIZE>>>(
        d_keys, d_values, n, d_bucket_counts, nullptr, num_buckets, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_bucket_counts, d_bucket_counts,
        num_blocks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bucket_counts));

    for (int b = 0; b < num_blocks; b++) {
        int chunk_start = b * BLOCK_SIZE;
        int chunk_size = min(BLOCK_SIZE, n - chunk_start);
        int lower_count = h_bucket_counts[b * num_buckets + 0];

        if (lower_count > 1) sort_by_key_driver(d_keys + chunk_start, d_values + chunk_start, lower_count, cmp);
        if (chunk_size - lower_count > 1) sort_by_key_driver(d_keys + chunk_start + lower_count, d_values + chunk_start + lower_count, chunk_size - lower_count, cmp);
    }

    delete[] h_bucket_counts;
}

template <typename K, typename V>
void stable_sort_by_key_driver(K* d_keys, V* d_values, int n) {
    radix_sort_driver<K, V>(d_keys, d_values, n);
}

} // namespace detail
} // namespace blqs
