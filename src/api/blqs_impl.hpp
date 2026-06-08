// Template declarations + implementations for API functions
// Moved implementations to header for on-demand template instantiation
#pragma once
#include <cuda_runtime.h>
#include "blqs_config.hpp"
#include "blqs_errors.hpp"

namespace blqs {
namespace detail {

// ── Kernel declarations (defined in partition.cu) ──────────────────────────

template <typename T, typename Comparator>
__global__ void partition_kernel(
    T* data, int n, int* bucket_counts, int* bucket_offsets,
    int num_buckets, Comparator cmp);

// ── Kernel declarations (defined in block_sort.cu) ─────────────────────────

template <typename T, int MaxSize, typename Comparator>
__global__ void quicksort_shared(T* data, int n, Comparator cmp);

// ── Forward declarations (defined in radix_sort.cu / blqs_keyvalue_impl.cu) ─

template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n);

template <typename K, typename V, typename Comparator>
void sort_by_key_driver(K* d_keys, V* d_values, int n, Comparator cmp);

template <typename K, typename V>
void stable_sort_by_key_driver(K* d_keys, V* d_values, int n);

// ── Sort Driver (implementation in header for template instantiation) ──────

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

        if (lower_count > 1) {
            sort_driver(d_data + chunk_start, lower_count, cmp);
        }
        if (chunk_size - lower_count > 1) {
            sort_driver(d_data + chunk_start + lower_count, chunk_size - lower_count, cmp);
        }
    }

    delete[] h_bucket_counts;
}

template <typename T, typename Comparator>
void stable_merge_sort(T* d_data, int n, Comparator cmp) {
    radix_sort_driver<T, void>(d_data, nullptr, n);
    (void)cmp;
}

} // namespace detail
} // namespace blqs
