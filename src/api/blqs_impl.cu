// API implementations for cuda-blqsort (struct-based sort)
// Task 2: Baseline sort_driver, Task 4: Stable sort APIs
#include "blqs_impl.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>

namespace blqs {
namespace detail {

// Forward kernel declarations (defined in partition.cu)
template <typename T, typename Comparator>
__global__ void partition_kernel(
    T* data, int n, int* bucket_counts, int* bucket_offsets, int num_buckets, Comparator cmp);

// Forward kernel declarations (defined in block_sort.cu)
template <typename T, int MaxSize, typename Comparator>
__global__ void quicksort_shared(T* data, int n, Comparator cmp);

// Forward radix sort driver (defined in radix_sort.cu)
template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n);

// ── Task 2: Host-Side Recursive Sort Driver ────────────────────────────────

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

// ── Task 4: Stable Merge Sort Driver ──────────────────────────────────────

template <typename T, typename Comparator>
void stable_merge_sort(T* d_data, int n, Comparator cmp) {
    radix_sort_driver<T, void>(d_data, nullptr, n);
    (void)cmp;
}

// ── Explicit Instantiations ────────────────────────────────────────────────

template void sort_driver<int, int (*)(const int&, const int&)>(int*, int, int (*)(const int&, const int&));
template void sort_driver<float, int (*)(const float&, const float&)>(float*, int, int (*)(const float&, const float&));
template void sort_driver<int64_t, int (*)(const int64_t&, const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
template void sort_driver<double, int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));

template void stable_merge_sort<int, int (*)(const int&, const int&)>(int*, int, int (*)(const int&, const int&));
template void stable_merge_sort<float, int (*)(const float&, const float&)>(float*, int, int (*)(const float&, const float&));
template void stable_merge_sort<int64_t, int (*)(const int64_t&, const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
template void stable_merge_sort<double, int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));

} // namespace detail
} // namespace blqs
