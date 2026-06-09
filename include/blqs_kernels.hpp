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

// Cooperative multi-thread partition kernel
// All 256 threads participate: count -> prefix sum -> scatter -> copy back
template <typename T, int MaxSize = 256>
__device__ void cooperative_partition(T* sdata, int low, int high) {
    extern __shared__ char smem_raw[];
    T* temp = sdata + MaxSize;  // temp buffer after sdata
    int block_threads = blockDim.x;
    int n = high - low + 1;
    T pivot = sdata[high];
    int tid = threadIdx.x;

    // Phase 1: Each thread counts elements < pivot in its stride
    __shared__ int less_cnt[256];
    __shared__ int ge_cnt[256];
    int lc = 0, gc = 0;
    for (int i = tid; i < n; i += block_threads) {
        if (sdata[low + i] < pivot) lc++;
        else gc++;
    }
    less_cnt[tid] = lc;
    ge_cnt[tid] = gc;
    __syncthreads();

    // Phase 2: Thread 0 computes prefix sums for per-thread offsets
    __shared__ int less_off[256];   // starting offset in "less" region for each thread
    __shared__ int ge_off[256];     // starting offset in "ge" region for each thread
    __shared__ int total_less_val;
    if (tid == 0) {
        less_off[0] = 0;
        ge_off[0] = 0;
        for (int t = 1; t < block_threads; t++) {
            less_off[t] = less_off[t - 1] + less_cnt[t - 1];
            ge_off[t] = ge_off[t - 1] + ge_cnt[t - 1];
        }
        total_less_val = less_off[block_threads - 1] + less_cnt[block_threads - 1];
    }
    __syncthreads();

    // Phase 3: All threads scatter to temp buffer cooperatively
    // Each thread uses a LOCAL counter initialized from prefix sum offset
    // Key fix: increment pos by 1 for each element (not block_threads)
    int less_pos = less_off[tid];  // local counter, starts at precomputed offset
    int ge_pos = ge_off[tid];      // local counter for ge region
    for (int i = tid; i < n; i += block_threads) {
        T val = sdata[low + i];
        if (val < pivot) {
            temp[less_pos] = val;  // place at current position
            less_pos++;             // increment by 1 (NOT block_threads!)
        } else {
            temp[total_less_val + ge_pos] = val;
            ge_pos++;              // increment by 1
        }
    }
    __syncthreads();

    // Phase 4: All threads copy back cooperatively
    for (int i = tid; i < n; i += block_threads) {
        sdata[low + i] = temp[i];
    }
    __syncthreads();
}

template <typename T, int MaxSize = 256>
__global__ void quicksort_shared_kernel(T* data, int n) {
    // Shared memory layout: sdata[MaxSize] + temp[MaxSize]
    extern __shared__ char smem_raw[];
    T* sdata = reinterpret_cast<T*>(smem_raw);
    int tid = threadIdx.x;
    if (tid < n) sdata[tid] = data[tid];
    __syncthreads();

    // Thread 0 manages the quicksort stack
    // Partition step uses all 256 threads cooperatively
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
            // Cooperative partition: all threads participate
            cooperative_partition<T, MaxSize>(sdata, lo, hi);
            // Find pivot position after partition
            T pivot = sdata[hi];
            int pi = lo;
            for (int j = lo; j < hi; j++) {
                if (sdata[j] < pivot) pi++;
            }
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

    // Thread 0 sorts
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

// Multi-block radix sort: histogram phase (computes per-block digit counts)
template <typename K, typename V>
__global__ void radix_sort_histogram_kernel(
    const K* in_k, int* block_hist, int n, int pass) {
    __shared__ int s_cnt[RADIX_BINS];
    if (threadIdx.x < RADIX_BINS) s_cnt[threadIdx.x] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int d = get_radix_digit(in_k[tid], pass);
        atomicAdd(&s_cnt[d], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_BINS) {
        block_hist[blockIdx.x * RADIX_BINS + threadIdx.x] = s_cnt[threadIdx.x];
    }
}

// Multi-block radix sort: scatter phase using precomputed global offsets
template <typename K, typename V>
__global__ void radix_sort_scatter_kernel(
    const K* in_k, const V* in_v, K* out_k, V* out_v,
    const int* global_offsets, int n, int pass, int num_blocks_launch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int d = get_radix_digit(in_k[tid], pass);
    // Compute position: global offset for digit + per-block offset + rank within block
    int block_offset = global_offsets[blockIdx.x * RADIX_BINS + d];
    int rank = 0;
    // Count elements in this block with same digit before this thread
    int block_start = blockIdx.x * blockDim.x;
    int block_n = min((int)blockDim.x, n - block_start);
    for (int i = 0; i < threadIdx.x && (block_start + i) < n; i++) {
        if (get_radix_digit(in_k[block_start + i], pass) == d) rank++;
    }
    out_k[block_offset + rank] = in_k[tid];
    if constexpr (!std::is_same<V, void>::value) out_v[block_offset + rank] = in_v[tid];
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

    // Use multi-block for large arrays, single-block for small
    int num_blocks = (n + 255) / 256;
    if (num_blocks > 1) {
        // Multi-block radix sort
        std::vector<int> block_hist(num_blocks * RADIX_BINS);
        std::vector<int> global_offsets(num_blocks * RADIX_BINS);
        int* d_block_hist = nullptr;
        int* d_global_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_hist, num_blocks * RADIX_BINS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_global_offsets, num_blocks * RADIX_BINS * sizeof(int)));

        size_t smem = RADIX_BINS * 2 * sizeof(int);
        for (int p = 0; p < passes; p++) {
            int s = p % 2, d = 1 - s;

            // Phase 1: Compute block histograms
            radix_sort_histogram_kernel<K, V><<<num_blocks, 256>>>(
                dk[s], d_block_hist, n, p);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy histograms to host
            CUDA_CHECK(cudaMemcpy(block_hist.data(), d_block_hist,
                num_blocks * RADIX_BINS * sizeof(int), cudaMemcpyDeviceToHost));

            // Phase 2: Compute global offsets (prefix sum across blocks)
            for (int digit = 0; digit < RADIX_BINS; digit++) {
                int offset = 0;
                for (int b = 0; b < num_blocks; b++) {
                    global_offsets[b * RADIX_BINS + digit] = offset;
                    offset += block_hist[b * RADIX_BINS + digit];
                }
            }

            // Copy global offsets to device
            CUDA_CHECK(cudaMemcpy(d_global_offsets, global_offsets.data(),
                num_blocks * RADIX_BINS * sizeof(int), cudaMemcpyHostToDevice));

            // Phase 3: Scatter
            radix_sort_scatter_kernel<K, V><<<num_blocks, 256>>>(
                dk[s], dv[s], dk[d], dv[d], d_global_offsets, n, p, num_blocks);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaFree(d_block_hist));
        CUDA_CHECK(cudaFree(d_global_offsets));
    } else {
        // Single-block for small arrays
        int threads = (n < 1024) ? n : 1024;
        int blocks = 1;
        size_t smem = RADIX_BINS * 2 * sizeof(int);
        for (int p = 0; p < passes; p++) {
            int s = p % 2, d = 1 - s;
            radix_sort_pass_kernel<K, V><<<blocks, threads, smem>>>(
                dk[s], dv[s], dk[d], dv[d], n, p);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
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
