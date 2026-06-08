// API implementations for cuda-blqsort (key-value sort)
// Task 3: Key-value sort driver
#include "blqs_keyvalue_impl.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include <cuda_runtime.h>

namespace blqs {
namespace detail {

// Forward kernel declarations (defined in partition.cu)
template <typename K, typename V, typename Comparator>
__global__ void kv_partition_kernel(
    K* keys, V* values, int n, int* bucket_counts, int* bucket_offsets,
    int num_buckets, Comparator cmp);

// Forward kernel declarations (defined in block_sort.cu)
template <typename K, typename V, typename Comparator>
__global__ void kv_block_sort_kernel(
    K* keys, V* values, int n, Comparator cmp);

// Forward radix sort driver (defined in radix_sort.cu)
template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n);

// ── Task 3: Host-Side Recursive KV Sort Driver ─────────────────────────────

template <typename K, typename V, typename Comparator>
void sort_by_key_driver(K* d_keys, V* d_values, int n, Comparator cmp) {
    if (n <= 1) return;

    if (n <= BLOCK_SIZE) {
        int threads = min(n, BLOCK_SIZE);
        kv_block_sort_kernel<K, V, Comparator><<<1, threads>>>(
            d_keys, d_values, n, cmp);
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

        if (lower_count > 1) {
            sort_by_key_driver(d_keys + chunk_start, d_values + chunk_start, lower_count, cmp);
        }
        if (chunk_size - lower_count > 1) {
            sort_by_key_driver(d_keys + chunk_start + lower_count, d_values + chunk_start + lower_count,
                chunk_size - lower_count, cmp);
        }
    }

    delete[] h_bucket_counts;
}

// ── Task 4: Stable KV Sort (always uses radix for numeric keys) ────────────

template <typename K, typename V>
void stable_sort_by_key_driver(K* d_keys, V* d_values, int n) {
    radix_sort_driver<K, V>(d_keys, d_values, n);
}

// ── Explicit Instantiations ────────────────────────────────────────────────

template void sort_by_key_driver<int, int, int (*)(const int&, const int&)>(
    int*, int*, int, int (*)(const int&, const int&));
template void sort_by_key_driver<float, int, int (*)(const float&, const float&)>(
    float*, int*, int, int (*)(const float&, const float&));
template void sort_by_key_driver<int64_t, int64_t, int (*)(const int64_t&, const int64_t&)>(
    int64_t*, int64_t*, int, int (*)(const int64_t&, const int64_t&));
template void sort_by_key_driver<double, double, int (*)(const double&, const double&)>(
    double*, double*, int, int (*)(const double&, const double&));

template void stable_sort_by_key_driver<int, int>(int*, int*, int);
template void stable_sort_by_key_driver<float, int>(float*, int*, int);
template void stable_sort_by_key_driver<int64_t, int64_t>(int64_t*, int64_t*, int);
template void stable_sort_by_key_driver<double, double>(double*, double*, int);

} // namespace detail
} // namespace blqs
