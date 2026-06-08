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

// ── Small-array Quicksort (multi-thread cooperative, shared memory) ────────

template <typename T>
__device__ void insertion_sort_dev(T* data, int n) {
    for (int i = 1; i < n; i++) {
        T key = data[i];
        int j = i - 1;
        while (j >= 0 && key < data[j]) { data[j + 1] = data[j]; j--; }
        data[j + 1] = key;
    }
}

// Multi-thread cooperative partition: all threads participate
// Returns the pivot index
template <typename T>
__device__ int partition_block_mt(T* data, int lo, int hi, int block_threads) {
    T pivot = data[hi];
    int count = hi - lo + 1; // elements to partition

    // Each thread counts how many of its elements go left of pivot
    int local_less = 0, local_ge = 0;
    for (int i = threadIdx.x; i < count; i += block_threads) {
        if (data[lo + i] < pivot) local_less++;
        else local_ge++;
    }

    // Block-wide prefix sum to get global offsets
    __shared__ int s_less_total[1];
    __shared__ int s_ge_total[1];

    // Use warp-level ballot for fast sum (works for any block size)
    __shared__ int s_less[1024]; // per-thread partial
    __shared__ int s_ge[1024];
    s_less[threadIdx.x] = local_less;
    s_ge[threadIdx.x] = local_ge;
    __syncthreads();

    if (threadIdx.x == 0) {
        int lt = 0, gt = 0;
        for (int i = 0; i < block_threads && i < count; i++) {
            lt += s_less[i];
            gt += s_ge[i];
        }
        s_less_total[0] = lt;
        s_ge_total[0] = gt;
    }
    __syncthreads();

    // Compute per-thread starting offset (prefix sum of partials)
    __shared__ int s_less_offset[1024];
    __shared__ int s_ge_offset[1024];
    if (threadIdx.x == 0) {
        int lt = 0, gt = 0;
        for (int i = 0; i < block_threads && i < count; i++) {
            s_less_offset[i] = lt;
            s_ge_offset[i] = gt;
            lt += s_less[i];
            gt += s_ge[i];
        }
    }
    __syncthreads();

    // Scatter to temp buffer
    __shared__ T s_tmp[1024];
    int less_off = s_less_offset[threadIdx.x];
    int ge_off = s_ge_offset[threadIdx.x];
    for (int i = threadIdx.x; i < count; i += block_threads) {
        if (data[lo + i] < pivot) {
            s_tmp[less_off] = data[lo + i];
            less_off += block_threads;
        } else {
            s_tmp[s_less_total[0] + ge_off] = data[lo + i];
            ge_off += block_threads;
        }
    }
    __syncthreads();

    // Copy back
    for (int i = threadIdx.x; i < count; i += block_threads) {
        data[lo + i] = s_tmp[i];
    }
    __syncthreads();

    return lo + s_less_total[0]; // pivot index (first >= pivot)
}

template <typename T, int MaxSize = 1024>
__global__ void quicksort_shared_kernel(T* data, int n) {
    __shared__ T sdata[MaxSize];
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    if (tid < n) sdata[tid] = data[tid];
    __syncthreads();

    // Cooperative iterative quicksort
    constexpr int STACK_DEPTH = 20;
    __shared__ int s_low[STACK_DEPTH], s_high[STACK_DEPTH];
    __shared__ int s_top;

    if (tid == 0) { s_low[0] = 0; s_high[0] = n - 1; s_top = 0; }
    __syncthreads();

    while (true) {
        int lo, hi;
        if (tid == 0) {
            if (s_top < 0) { lo = -1; hi = -1; }
            else { lo = s_low[s_top]; hi = s_high[s_top]; s_top--; }
        }
        // Broadcast lo, hi to all threads
        lo = __shfl_sync(0xFFFFFFFF, lo, 0);
        hi = __shfl_sync(0xFFFFFFFF, hi, 0);
        if (lo < 0 || lo >= hi) break;

        if (hi - lo + 1 <= BASE_CASE_THRESHOLD) {
            // Small: thread 0 does insertion sort
            if (tid == 0) insertion_sort_dev(sdata + lo, hi - lo + 1);
            __syncthreads();
            continue;
        }

        // Cooperative partition
        int pi = partition_block_mt(sdata, lo, hi, block_threads);
        // pi is the first element >= pivot; pivot itself is at hi
        // Need to place pivot at pi
        T pivot_val = sdata[hi];
        if (pi < hi) {
            // Swap sdata[pi] and sdata[hi] only if needed
            // Actually the partition already put >= elements starting at pi
            // The pivot value may have moved; find it and put it at pi
            for (int i = pi; i <= hi; i++) {
                if (sdata[i] == pivot_val) {
                    T tmp = sdata[i]; sdata[i] = sdata[pi]; sdata[pi] = tmp;
                    break;
                }
            }
        }

        // Push sub-ranges to stack (larger first for better cache behavior)
        int left_size = pi - lo;
        int right_size = hi - pi;
        if (tid == 0) {
            if (left_size > 1 && right_size > 1) {
                if (left_size > right_size) {
                    s_top++; s_low[s_top] = lo; s_high[s_top] = pi - 1;
                    s_top++; s_low[s_top] = pi + 1; s_high[s_top] = hi;
                } else {
                    s_top++; s_low[s_top] = pi + 1; s_high[s_top] = hi;
                    s_top++; s_low[s_top] = lo; s_high[s_top] = pi - 1;
                }
            } else if (left_size > 1) {
                s_top++; s_low[s_top] = lo; s_high[s_top] = pi - 1;
            } else if (right_size > 1) {
                s_top++; s_low[s_top] = pi + 1; s_high[s_top] = hi;
            }
        }
        __syncthreads();
    }

    if (tid < n) data[tid] = sdata[tid];
}

// ── KV Block Sort (multi-thread cooperative) ───────────────────────────────

template <typename K, typename V>
__device__ int kv_partition_block_mt(K* keys, V* values, int lo, int hi, int block_threads) {
    K pivot = keys[hi];
    int count = hi - lo + 1;

    int local_less = 0, local_ge = 0;
    for (int i = threadIdx.x; i < count; i += block_threads) {
        if (keys[lo + i] < pivot) local_less++;
        else local_ge++;
    }

    __shared__ int s_less[1024];
    __shared__ int s_ge[1024];
    __shared__ int s_less_total[1];
    __shared__ int s_less_offset[1024];
    __shared__ int s_ge_offset[1024];
    s_less[threadIdx.x] = local_less;
    s_ge[threadIdx.x] = local_ge;
    __syncthreads();

    if (threadIdx.x == 0) {
        int lt = 0, gt = 0;
        for (int i = 0; i < block_threads && i < count; i++) {
            s_less_offset[i] = lt;
            s_ge_offset[i] = gt;
            lt += s_less[i];
            gt += s_ge[i];
        }
        s_less_total[0] = lt;
    }
    __syncthreads();

    __shared__ K sk_tmp[1024];
    __shared__ V sv_tmp[1024];
    int less_off = s_less_offset[threadIdx.x];
    int ge_off = s_ge_offset[threadIdx.x];
    for (int i = threadIdx.x; i < count; i += block_threads) {
        if (keys[lo + i] < pivot) {
            sk_tmp[less_off] = keys[lo + i];
            sv_tmp[less_off] = values[lo + i];
            less_off += block_threads;
        } else {
            sk_tmp[s_less_total[0] + ge_off] = keys[lo + i];
            sv_tmp[s_less_total[0] + ge_off] = values[lo + i];
            ge_off += block_threads;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += block_threads) {
        keys[lo + i] = sk_tmp[i];
        values[lo + i] = sv_tmp[i];
    }
    __syncthreads();

    return lo + s_less_total[0];
}

template <typename K, typename V>
__global__ void kv_block_sort_kernel(K* keys, V* values, int n) {
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    __shared__ K sk[1024];
    __shared__ V sv[1024];
    if (tid < n) { sk[tid] = keys[tid]; sv[tid] = values[tid]; }
    __syncthreads();

    constexpr int STACK_DEPTH = 20;
    __shared__ int sl[STACK_DEPTH], sh[STACK_DEPTH], s_top;

    if (tid == 0) { sl[0] = 0; sh[0] = n - 1; s_top = 0; }
    __syncthreads();

    while (true) {
        int lo, hi;
        if (tid == 0) {
            if (s_top < 0) { lo = -1; hi = -1; }
            else { lo = sl[s_top]; hi = sh[s_top]; s_top--; }
        }
        lo = __shfl_sync(0xFFFFFFFF, lo, 0);
        hi = __shfl_sync(0xFFFFFFFF, hi, 0);
        if (lo < 0 || lo >= hi) break;

        if (hi - lo + 1 <= BASE_CASE_THRESHOLD) {
            if (tid == 0) {
                for (int i = 1; i <= hi - lo; i++) {
                    K k = sk[lo + i];
                    V v = sv[lo + i];
                    int j = i - 1;
                    while (j >= 0 && k < sk[lo + j]) {
                        sk[lo + j + 1] = sk[lo + j];
                        sv[lo + j + 1] = sv[lo + j];
                        j--;
                    }
                    sk[lo + j + 1] = k;
                    sv[lo + j + 1] = v;
                }
            }
            __syncthreads();
            continue;
        }

        int pi = kv_partition_block_mt(sk, sv, lo, hi, block_threads);
        K pivot_val = sk[hi];
        if (pi < hi) {
            for (int i = pi; i <= hi; i++) {
                if (sk[i] == pivot_val) {
                    K tk = sk[i]; sk[i] = sk[pi]; sk[pi] = tk;
                    V tv = sv[i]; sv[i] = sv[pi]; sv[pi] = tv;
                    break;
                }
            }
        }

        int left_size = pi - lo;
        int right_size = hi - pi;
        if (tid == 0) {
            if (left_size > 1 && right_size > 1) {
                if (left_size > right_size) {
                    s_top++; sl[s_top] = lo; sh[s_top] = pi - 1;
                    s_top++; sl[s_top] = pi + 1; sh[s_top] = hi;
                } else {
                    s_top++; sl[s_top] = pi + 1; sh[s_top] = hi;
                    s_top++; sl[s_top] = lo; sh[s_top] = pi - 1;
                }
            } else if (left_size > 1) {
                s_top++; sl[s_top] = lo; sh[s_top] = pi - 1;
            } else if (right_size > 1) {
                s_top++; sl[s_top] = pi + 1; sh[s_top] = hi;
            }
        }
        __syncthreads();
    }

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
