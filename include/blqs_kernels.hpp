// All CUDA kernel template definitions for cuda-blqsort
// Kernels use operator< directly (no custom comparator parameter)
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <type_traits>
#include "blqs_config.hpp"
#include "blqs_errors.hpp"

// All kernel definitions are gated behind __CUDACC__ so .cpp files can
// include this header without nvcc-specific syntax errors.
#ifdef __CUDACC__

namespace blqs {
namespace detail {

// ── Partition Kernel (uses operator<) ──────────────────────────────────────

template <typename T>
__global__ void partition_kernel(
    T* data, int n, int* bucket_counts, int* bucket_offsets, int num_buckets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    __shared__ T pivot;
    __shared__ int chunk_start;
    __shared__ int chunk_size;

    if (threadIdx.x == 0) {
        chunk_start = blockIdx.x * blockDim.x;
        int cs = (n - chunk_start < BLOCK_SIZE) ? (n - chunk_start) : BLOCK_SIZE;
        chunk_size = cs;
        // Median-of-three pivot
        int a = chunk_start, b = chunk_start + cs / 2, c = chunk_start + cs - 1;
        T va = data[a], vb = data[b], vc = data[c];
        if (vc < va) { if (va < vb) pivot = va; else if (vc < vb) pivot = vc; else pivot = vb; }
        else { if (vc < vb) pivot = vc; else if (va < vb) pivot = vb; else pivot = va; }
    }
    __syncthreads();

    T val = data[tid];
    int bucket = (val < pivot) ? 0 : 1;
    int pos = atomicAdd(&bucket_counts[blockIdx.x * num_buckets + bucket], 1);
    int offset = (bucket == 0) ? chunk_start
                               : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];
    data[offset + pos] = val;
}

// ── KV Partition Kernel (uses operator< on keys) ───────────────────────────

template <typename K, typename V>
__global__ void kv_partition_kernel(
    K* keys, V* values, int n, int* bucket_counts, int* bucket_offsets, int num_buckets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    __shared__ K pivot;
    __shared__ int chunk_start;
    __shared__ int chunk_size;

    if (threadIdx.x == 0) {
        chunk_start = blockIdx.x * blockDim.x;
        int cs = (n - chunk_start < BLOCK_SIZE) ? (n - chunk_start) : BLOCK_SIZE;
        chunk_size = cs;
        int a = chunk_start, b = chunk_start + cs / 2, c = chunk_start + cs - 1;
        K va = keys[a], vb = keys[b], vc = keys[c];
        if (vc < va) { if (va < vb) pivot = va; else if (vc < vb) pivot = vc; else pivot = vb; }
        else { if (vc < vb) pivot = vc; else if (va < vb) pivot = vb; else pivot = va; }
    }
    __syncthreads();

    K key = keys[tid];
    V val = values[tid];
    int bucket = (key < pivot) ? 0 : 1;
    int pos = atomicAdd(&bucket_counts[blockIdx.x * num_buckets + bucket], 1);
    int offset = (bucket == 0) ? chunk_start
                               : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];
    keys[offset + pos] = key;
    values[offset + pos] = val;
}

// ── Insertion Sort (Base Case) ─────────────────────────────────────────────

template <typename T>
__device__ void insertion_sort(T* data, int n) {
    for (int i = 1; i < n; i++) {
        T key = data[i];
        int j = i - 1;
        while (j >= 0 && key < data[j]) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

// ── Partition Block (Lomuto) ───────────────────────────────────────────────

template <typename T>
__device__ T partition_block(T* data, int low, int high) {
    T pivot = data[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (!(pivot < data[j])) {
            i++;
            T tmp = data[i]; data[i] = data[j]; data[j] = tmp;
        }
    }
    T tmp = data[i + 1]; data[i + 1] = data[high]; data[high] = tmp;
    return i + 1;
}

// ── Shared-Memory Quicksort Kernel ─────────────────────────────────────────

template <typename T, int MaxSize = 1024>
__global__ void quicksort_shared(T* data, int n) {
    __shared__ T shared_mem[MaxSize];
    int tid = threadIdx.x;
    if (tid < n) shared_mem[tid] = data[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    int stack_low[MaxSize], stack_high[MaxSize], top = -1;
    stack_low[++top] = 0;
    stack_high[top] = n - 1;

    while (top >= 0) {
        int low = stack_low[top];
        int high = stack_high[top--];
        if (low >= high) continue;
        if (high - low + 1 <= BASE_CASE_THRESHOLD) {
            insertion_sort(shared_mem + low, high - low + 1);
            continue;
        }
        int pi = partition_block(shared_mem, low, high);
        stack_low[++top] = low; stack_high[top] = pi - 1;
        stack_low[++top] = pi + 1; stack_high[top] = high;
    }
    __syncthreads();

    if (tid < n) data[blockIdx.x * blockDim.x + tid] = shared_mem[tid];
}

// ── KV Block Sort Kernel ───────────────────────────────────────────────────

template <typename K, typename V>
__global__ void kv_block_sort_kernel(K* keys, V* values, int n) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int chunk_size = (n - block_offset < BLOCK_SIZE) ? (n - block_offset) : BLOCK_SIZE;

    __shared__ K shared_keys[1024];
    __shared__ V shared_values[1024];
    if (tid < chunk_size) {
        shared_keys[tid] = keys[block_offset + tid];
        shared_values[tid] = values[block_offset + tid];
    }
    __syncthreads();

    int stack_low[1024], stack_high[1024], top = -1;
    stack_low[++top] = 0; stack_high[top] = chunk_size - 1;

    while (top >= 0) {
        int low = stack_low[top], high = stack_high[top--];
        if (low >= high) continue;
        K pivot = shared_keys[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (!(pivot < shared_keys[j])) {
                i++;
                K tk = shared_keys[i]; shared_keys[i] = shared_keys[j]; shared_keys[j] = tk;
                V tv = shared_values[i]; shared_values[i] = shared_values[j]; shared_values[j] = tv;
            }
        }
        i++;
        K tk = shared_keys[i]; shared_keys[i] = shared_keys[high]; shared_keys[high] = tk;
        V tv = shared_values[i]; shared_values[i] = shared_values[high]; shared_values[high] = tv;
        stack_low[++top] = low; stack_high[top] = i - 1;
        stack_low[++top] = i + 1; stack_high[top] = high;
    }
    __syncthreads();

    if (tid < chunk_size) {
        keys[block_offset + tid] = shared_keys[tid];
        values[block_offset + tid] = shared_values[tid];
    }
}

// ── Warp-Level Partition Helper ────────────────────────────────────────────

template <typename T>
__device__ int warp_partition(T* shared_data, int low, int high) {
    int tid = threadIdx.x;
    T pivot = shared_data[high];
    unsigned int mask = __ballot_sync(0xFFFFFFFF, tid < high && !(pivot < shared_data[tid]));
    return __popc(mask & ((1u << tid) - 1));
}

// ── Stable Merge Kernel ────────────────────────────────────────────────────

template <typename T>
__global__ void stable_merge_kernel(
    const T* input, T* output, const int* indices, int* output_indices,
    int n, int left_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int target = tid, l = 0, r = left_size;
    for (int i = 0; i <= target; i++) {
        if (l >= left_size) {
            if (i == target) { output[i] = input[left_size + r]; output_indices[i] = indices[left_size + r]; }
            r++;
        } else if (r >= n) {
            if (i == target) { output[i] = input[l]; output_indices[i] = indices[l]; }
            l++;
        } else if (!(input[left_size + r] < input[l])) {
            if (i == target) { output[i] = input[l]; output_indices[i] = indices[l]; }
            l++;
        } else {
            if (i == target) { output[i] = input[left_size + r]; output_indices[i] = indices[left_size + r]; }
            r++;
        }
    }
}

// ── Radix Sort Utilities ───────────────────────────────────────────────────

constexpr int RADIX_BITS = 8;
constexpr int RADIX_BINS = 1 << RADIX_BITS;
constexpr int RADIX_PASSES_32 = 4;
constexpr int RADIX_PASSES_64 = 8;

template <typename T>
struct RadixTraits {
    using UnsignedT = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
    static constexpr int passes = sizeof(T) == 4 ? RADIX_PASSES_32 : RADIX_PASSES_64;
};

template <typename T>
__device__ __forceinline__ unsigned int get_radix_digit(T val, int pass) {
    typename RadixTraits<T>::UnsignedT uval = reinterpret_cast<typename RadixTraits<T>::UnsignedT&>(val);
    if constexpr (std::is_signed<T>::value) {
        if constexpr (sizeof(T) == 4) uval ^= 0x80000000U;
        else uval ^= 0x8000000000000000ULL;
    }
    return (uval >> (pass * RADIX_BITS)) & (RADIX_BINS - 1);
}

template <typename K, typename V>
__global__ void radix_sort_pass_kernel(
    const K* input_keys, const V* input_values,
    K* output_keys, V* output_values, int n, int pass) {
    extern __shared__ int s_mem[];
    int* s_counters = s_mem;
    int* s_offsets = s_mem + RADIX_BINS;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < RADIX_BINS) s_counters[threadIdx.x] = 0;
    __syncthreads();

    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        atomicAdd(&s_counters[digit], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_BINS) {
        int sum = 0;
        for (int i = 0; i < threadIdx.x; i++) sum += s_counters[i];
        s_offsets[threadIdx.x] = sum;
    }
    __syncthreads();

    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        int pos = atomicAdd(&s_offsets[digit], 1);
        output_keys[pos] = input_keys[tid];
        if constexpr (!std::is_same<V, void>::value) output_values[pos] = input_values[tid];
    }
}

template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n) {
    if (n <= 1) return;
    constexpr int passes = RadixTraits<K>::passes;

    K *d_buf_keys[2];
    V *d_buf_values[2];

    CUDA_CHECK(cudaMalloc(&d_buf_keys[0], n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_buf_keys[1], n * sizeof(K)));
    CUDA_CHECK(cudaMemcpy(d_buf_keys[0], d_keys, n * sizeof(K), cudaMemcpyDeviceToDevice));

    if constexpr (!std::is_same<V, void>::value) {
        CUDA_CHECK(cudaMalloc(&d_buf_values[0], n * sizeof(V)));
        CUDA_CHECK(cudaMalloc(&d_buf_values[1], n * sizeof(V)));
        CUDA_CHECK(cudaMemcpy(d_buf_values[0], d_values, n * sizeof(V), cudaMemcpyDeviceToDevice));
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = RADIX_BINS * 2 * sizeof(int);

    for (int pass = 0; pass < passes; pass++) {
        int src = pass % 2, dst = 1 - src;
        radix_sort_pass_kernel<K, V><<<blocks, threads, shared_mem_size>>>(
            d_buf_keys[src], d_buf_values[src], d_buf_keys[dst], d_buf_values[dst], n, pass);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int last = passes % 2;
    CUDA_CHECK(cudaMemcpy(d_keys, d_buf_keys[last], n * sizeof(K), cudaMemcpyDeviceToDevice));
    if constexpr (!std::is_same<V, void>::value) {
        CUDA_CHECK(cudaMemcpy(d_values, d_buf_values[last], n * sizeof(V), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_buf_values[0]));
        CUDA_CHECK(cudaFree(d_buf_values[1]));
    }
    CUDA_CHECK(cudaFree(d_buf_keys[0]));
    CUDA_CHECK(cudaFree(d_buf_keys[1]));
}

} // namespace detail
} // namespace blqs

#endif // __CUDACC__
