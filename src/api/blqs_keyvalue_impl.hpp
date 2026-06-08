// Template declarations for key-value sort API functions
#pragma once
#include <cuda_runtime.h>

namespace blqs {
namespace detail {

// Kernel declarations (defined in partition.cu)
template <typename K, typename V, typename Comparator>
__global__ void kv_partition_kernel(
    K* keys, V* values, int n, int* bucket_counts, int* bucket_offsets,
    int num_buckets, Comparator cmp);

// Kernel declarations (defined in block_sort.cu)
template <typename K, typename V, typename Comparator>
__global__ void kv_block_sort_kernel(
    K* keys, V* values, int n, Comparator cmp);

// Host-side recursive KV sort driver (defined in blqs_keyvalue_impl.cu)
template <typename K, typename V, typename Comparator>
void sort_by_key_driver(K* d_keys, V* d_values, int n, Comparator cmp);

// Stable KV sort driver (defined in blqs_keyvalue_impl.cu)
template <typename K, typename V>
void stable_sort_by_key_driver(K* d_keys, V* d_values, int n);

} // namespace detail
} // namespace blqs
