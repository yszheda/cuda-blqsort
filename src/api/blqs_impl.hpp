#pragma once
#include <cuda_runtime.h>

namespace blqs {
namespace detail {

// Kernel declarations (defined in partition.cu)
template <typename T, typename Comparator>
__global__ void partition_kernel(
    T* data, int n, int* bucket_counts, int* bucket_offsets,
    int num_buckets, Comparator cmp);

// Kernel declarations (defined in block_sort.cu)
template <typename T, int MaxSize, typename Comparator>
__global__ void quicksort_shared(T* data, int n, Comparator cmp);

// Host-side recursive sort driver (defined in blqs_impl.cu)
template <typename T, typename Comparator>
void sort_driver(T* d_data, int n, Comparator cmp);

// Stable merge sort driver (defined in blqs_impl.cu)
template <typename T, typename Comparator>
void stable_merge_sort(T* d_data, int n, Comparator cmp);

} // namespace detail
} // namespace blqs
