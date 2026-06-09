// All CUDA kernel template definitions for cuda-blqsort
// Kernels use operator< directly, single global pivot strategy
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <type_traits>
#include "blqs_config.hpp"
#include "blqs_errors.hpp"

#ifdef __CUDACC__

namespace blqs {
namespace detail {

// ── Partition Kernel: single global pivot, two-pass (count + scatter) ──────

// Pass 1: Count elements less than pivot (result in d_less_count)
template <typename T>
__global__ void count_less_kernel(const T* data, int n, T pivot, int* d_less_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (data[tid] < pivot) atomicAdd(d_less_count, 1);
}

// Pass 2: Scatter elements using separate output buffer to avoid read/write race
template <typename T>
__global__ void partition_scatter_kernel(
    const T* input, T* output, int n, T pivot, int less_count,
    int* d_less_written, int* d_ge_written) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    T val = input[tid];
    if (val < pivot) {
        int pos = atomicAdd(d_less_written, 1);
        output[pos] = val;
    } else {
        int pos = atomicAdd(d_ge_written, 1);
        output[less_count + pos] = val;
    }
}

// ── KV Partition: single global pivot ──────────────────────────────────────

template <typename K, typename V>
__global__ void kv_count_less_kernel(const K* keys, int n, K pivot, int* d_less_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (keys[tid] < pivot) atomicAdd(d_less_count, 1);
}

template <typename K, typename V>
__global__ void kv_partition_scatter_kernel(
    const K* in_keys, const V* in_values,
    K* out_keys, V* out_values, int n, K pivot, int less_count,
    int* d_less_written, int* d_ge_written) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    K key = in_keys[tid];
    V val = in_values[tid];
    if (key < pivot) {
        int pos = atomicAdd(d_less_written, 1);
        out_keys[pos] = key;
        out_values[pos] = val;
    } else {
        int pos = atomicAdd(d_ge_written, 1);
        out_keys[less_count + pos] = key;
        out_values[less_count + pos] = val;
    }
}

// ── Small-array Quicksort (single block, shared memory) ────────────────────

template <typename T>
__device__ void insertion_sort_dev(T* data, int n) {
    for (int i = 1; i < n; i++) {
        T key = data[i];
        int j = i - 1;
        while (j >= 0 && key < data[j]) { data[j + 1] = data[j]; j--; }
        data[j + 1] = key;
    }
}

template <typename T>
__device__ T partition_block_dev(T* data, int low, int high) {
    T pivot = data[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (!(pivot < data[j])) { i++; T t = data[i]; data[i] = data[j]; data[j] = t; }
    }
    T t = data[i + 1]; data[i + 1] = data[high]; data[high] = t;
    return i + 1;
}

template <typename T, int MaxSize = 1024>
__global__ void quicksort_shared_kernel(T* data, int n) {
    __shared__ T sdata[MaxSize];
    int tid = threadIdx.x;
    if (tid < n) sdata[tid] = data[tid];
    __syncthreads();

    // Only thread 0 performs the sort in shared memory
    if (tid == 0) {
        constexpr int STACK_DEPTH = 20;
        int s_low[STACK_DEPTH], s_high[STACK_DEPTH], top = -1;
        s_low[++top] = 0; s_high[top] = n - 1;

        while (top >= 0) {
            int lo = s_low[top], hi = s_high[top--];
            if (lo >= hi) continue;
            if (hi - lo + 1 <= BASE_CASE_THRESHOLD) {
                insertion_sort_dev(sdata + lo, hi - lo + 1);
                continue;
            }
            int pi = partition_block_dev(sdata, lo, hi);
            s_low[++top] = lo; s_high[top] = pi - 1;
            s_low[++top] = pi + 1; s_high[top] = hi;
        }
    }
    __syncthreads();
    if (tid < n) data[tid] = sdata[tid];
}

// ── KV Block Sort ──────────────────────────────────────────────────────────

template <typename K, typename V>
__global__ void kv_block_sort_kernel(K* keys, V* values, int n) {
    int tid = threadIdx.x;
    __shared__ K sk[1024];
    __shared__ V sv[1024];
    if (tid < n) { sk[tid] = keys[tid]; sv[tid] = values[tid]; }
    __syncthreads();

    // Only thread 0 sorts
    if (tid == 0) {
        constexpr int STACK_DEPTH = 20;
        int sl[STACK_DEPTH], sh[STACK_DEPTH], top = -1;
        sl[++top] = 0; sh[top] = n - 1;

        while (top >= 0) {
            int lo = sl[top], hi = sh[top--];
            if (lo >= hi) continue;
            K pivot = sk[hi];
            int i = lo - 1;
            for (int j = lo; j < hi; j++) {
                if (!(pivot < sk[j])) {
                    i++;
                    K tk = sk[i]; sk[i] = sk[j]; sk[j] = tk;
                    V tv = sv[i]; sv[i] = sv[j]; sv[j] = tv;
                }
            }
            i++;
            K tk = sk[i]; sk[i] = sk[hi]; sk[hi] = tk;
            V tv = sv[i]; sv[i] = sv[hi]; sv[hi] = tv;
            sl[++top] = lo; sh[top] = i - 1;
            sl[++top] = i + 1; sh[top] = hi;
        }
    }
    __syncthreads();
    if (tid < n) { keys[tid] = sk[tid]; values[tid] = sv[tid]; }
}

// ── Radix Sort ─────────────────────────────────────────────────────────────

constexpr int RADIX_BITS = 8;
constexpr int RADIX_BINS = 1 << RADIX_BITS;

template <typename T>
struct RadixTraits {
    using UnsignedT = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
    static constexpr int passes = sizeof(T) == 4 ? 4 : 8;
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
    const K* in_k, const V* in_v, K* out_k, V* out_v, int n, int pass) {
    extern __shared__ int s_mem[];
    int* s_cnt = s_mem;
    int* s_off = s_mem + RADIX_BINS;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < RADIX_BINS) s_cnt[threadIdx.x] = 0;
    __syncthreads();
    if (tid < n) atomicAdd(&s_cnt[get_radix_digit(in_k[tid], pass)], 1);
    __syncthreads();

    if (threadIdx.x < RADIX_BINS) {
        int s = 0;
        for (int i = 0; i < threadIdx.x; i++) s += s_cnt[i];
        s_off[threadIdx.x] = s;
    }
    __syncthreads();

    if (tid < n) {
        int d = get_radix_digit(in_k[tid], pass);
        int p = atomicAdd(&s_off[d], 1);
        out_k[p] = in_k[tid];
        if constexpr (!std::is_same<V, void>::value) out_v[p] = in_v[tid];
    }
}

template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n) {
    if (n <= 1) return;
    constexpr int passes = RadixTraits<K>::passes;

    K *dk[2]; V *dv[2];
    CUDA_CHECK(cudaMalloc(&dk[0], n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&dk[1], n * sizeof(K)));
    CUDA_CHECK(cudaMemcpy(dk[0], d_keys, n * sizeof(K), cudaMemcpyDeviceToDevice));
    if constexpr (!std::is_same<V, void>::value) {
        CUDA_CHECK(cudaMalloc(&dv[0], n * sizeof(V)));
        CUDA_CHECK(cudaMalloc(&dv[1], n * sizeof(V)));
        CUDA_CHECK(cudaMemcpy(dv[0], d_values, n * sizeof(V), cudaMemcpyDeviceToDevice));
    }

    // Use single block for correctness (avoids multi-block prefix sum issues)
    int threads = (n < 1024) ? n : 1024;
    int blocks = 1;
    size_t smem = RADIX_BINS * 2 * sizeof(int);
    for (int p = 0; p < passes; p++) {
        int s = p % 2, d = 1 - s;
        radix_sort_pass_kernel<K, V><<<blocks, threads, smem>>>(
            dk[s], dv[s], dk[d], dv[d], n, p);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int last = passes % 2;
    CUDA_CHECK(cudaMemcpy(d_keys, dk[last], n * sizeof(K), cudaMemcpyDeviceToDevice));
    if constexpr (!std::is_same<V, void>::value) {
        CUDA_CHECK(cudaMemcpy(d_values, dv[last], n * sizeof(V), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(dv[0])); CUDA_CHECK(cudaFree(dv[1]));
    }
    CUDA_CHECK(cudaFree(dk[0])); CUDA_CHECK(cudaFree(dk[1]));
}

} // namespace detail
} // namespace blqs

#endif // __CUDACC__
