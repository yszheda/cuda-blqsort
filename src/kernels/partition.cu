// Partition kernels for cuda-blqsort
// Task 2: Baseline partition, Task 3: KV partition, Task 5: Warp-level optimizations
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "blqs_config.hpp"

namespace blqs {
namespace detail {

// ── Pivot Selection (Median-of-Three) ──────────────────────────────────────

template <typename T, typename Comparator>
__device__ int select_pivot(const T* data, int n, Comparator cmp) {
    int mid = n / 2;
    int last = n - 1;

    if (cmp(data[last], data[0])) {
        if (cmp(data[0], data[mid])) return 0;
        if (cmp(data[last], data[mid])) return last;
        return mid;
    } else {
        if (cmp(data[last], data[mid])) return last;
        if (cmp(data[0], data[mid])) return mid;
        return 0;
    }
}

// ── Task 2: Baseline Partition Kernel ──────────────────────────────────────

template <typename T, typename Comparator>
__global__ void partition_kernel(
    T* data,
    int n,
    int* bucket_counts,
    int* bucket_offsets,
    int num_buckets,
    Comparator cmp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    __shared__ T pivot;
    __shared__ int chunk_start;
    __shared__ int chunk_size;

    if (threadIdx.x == 0) {
        chunk_start = blockIdx.x * blockDim.x;
        chunk_size = min(blockDim.x, n - chunk_start);
        int pivot_idx = chunk_start + select_pivot(data + chunk_start, chunk_size, cmp);
        pivot = data[pivot_idx];
    }
    __syncthreads();

    T val = data[tid];
    int bucket = cmp(val, pivot) ? 0 : 1;

    int pos = atomicAdd(&bucket_counts[blockIdx.x * num_buckets + bucket], 1);

    int offset = (bucket == 0) ? chunk_start
                               : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];
    data[offset + pos] = val;
}

// ── Task 3: Key-Value Partition Kernel ─────────────────────────────────────

template <typename K, typename V, typename Comparator>
__global__ void kv_partition_kernel(
    K* keys,
    V* values,
    int n,
    int* bucket_counts,
    int* bucket_offsets,
    int num_buckets,
    Comparator cmp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    __shared__ K pivot;
    __shared__ int chunk_start;
    __shared__ int chunk_size;

    if (threadIdx.x == 0) {
        chunk_start = blockIdx.x * blockDim.x;
        chunk_size = min(blockDim.x, n - chunk_start);
        int pivot_idx = chunk_start + select_pivot(keys + chunk_start, chunk_size, cmp);
        pivot = keys[pivot_idx];
    }
    __syncthreads();

    K key = keys[tid];
    V val = values[tid];
    int bucket = cmp(key, pivot) ? 0 : 1;

    int pos = atomicAdd(&bucket_counts[blockIdx.x * num_buckets + bucket], 1);
    int offset = (bucket == 0) ? chunk_start
                               : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];

    keys[offset + pos] = key;
    values[offset + pos] = val;
}

// ── Task 5: Warp-Level Partition Helper ────────────────────────────────────

template <typename T, typename Comparator>
__device__ int warp_partition(T* shared_data, int low, int high, Comparator cmp) {
    int tid = threadIdx.x;
    T pivot = shared_data[high];

    unsigned int mask = __ballot_sync(0xFFFFFFFF, tid < high && !cmp(pivot, shared_data[tid]));
    int rank = __popc(mask & ((1u << tid) - 1));

    return rank;
}

// ── Explicit Instantiations ────────────────────────────────────────────────

template __global__ void partition_kernel<int, int (*)(const int&, const int&)>(
    int*, int, int*, int*, int, int (*)(const int&, const int&));
template __global__ void partition_kernel<float, int (*)(const float&, const float&)>(
    float*, int, int*, int*, int, int (*)(const float&, const float&));
template __global__ void partition_kernel<int64_t, int (*)(const int64_t&, const int64_t&)>(
    int64_t*, int, int*, int*, int, int (*)(const int64_t&, const int64_t&));
template __global__ void partition_kernel<double, int (*)(const double&, const double&)>(
    double*, int, int*, int*, int, int (*)(const double&, const double&));

template __global__ void kv_partition_kernel<int, int, int (*)(const int&, const int&)>(
    int*, int*, int, int*, int*, int, int (*)(const int&, const int&));
template __global__ void kv_partition_kernel<float, int, int (*)(const float&, const float&)>(
    float*, int*, int, int*, int*, int, int (*)(const float&, const float&));
template __global__ void kv_partition_kernel<int64_t, int64_t, int (*)(const int64_t&, const int64_t&)>(
    int64_t*, int64_t*, int, int*, int*, int, int (*)(const int64_t&, const int64_t&));
template __global__ void kv_partition_kernel<double, double, int (*)(const double&, const double&)>(
    double*, double*, int, int*, int*, int, int (*)(const double&, const double&));

} // namespace detail
} // namespace blqs
