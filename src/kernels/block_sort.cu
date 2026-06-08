// Block sort kernels for cuda-blqsort
// Task 2: Baseline block quicksort, Task 5: Sorting network optimization
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "blqs_config.hpp"

namespace blqs {
namespace detail {

// ── Insertion Sort (Base Case) ─────────────────────────────────────────────

template <typename T, typename Comparator>
__device__ void insertion_sort(T* data, int n, Comparator cmp) {
    for (int i = 1; i < n; i++) {
        T key = data[i];
        int j = i - 1;
        while (j >= 0 && cmp(key, data[j])) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

// ── Task 5: Bitonic Sorting Network (64 elements) ──────────────────────────

template <typename T, typename Comparator>
__device__ void sorting_network_64(T* data, Comparator cmp) {
    for (int k = 2; k <= 64; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < 64; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool swap = ((i & k) == 0) ? cmp(data[ixj], data[i])
                                               : cmp(data[i], data[ixj]);
                    if (swap) {
                        T tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            }
        }
    }
}

// ── Partition Block (Lomuto Scheme) ────────────────────────────────────────

template <typename T, typename Comparator>
__device__ T partition_block(T* data, int low, int high, Comparator cmp) {
    T pivot = data[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (!cmp(pivot, data[j])) {
            i++;
            T tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
    T tmp = data[i + 1];
    data[i + 1] = data[high];
    data[high] = tmp;
    return i + 1;
}

// ── Shared-Memory Quicksort (Task 2) ───────────────────────────────────────

template <typename T, int MaxSize = 1024, typename Comparator>
__global__ void quicksort_shared(T* data, int n, Comparator cmp) {
    __shared__ T shared_mem[MaxSize];

    int tid = threadIdx.x;
    if (tid < n) {
        shared_mem[tid] = data[blockIdx.x * blockDim.x + tid];
    }
    __syncthreads();

    int stack_low[MaxSize];
    int stack_high[MaxSize];
    int top = -1;

    stack_low[++top] = 0;
    stack_high[top] = n - 1;

    while (top >= 0) {
        int low = stack_low[top];
        int high = stack_high[top--];

        if (low >= high) continue;
        if (high - low + 1 <= BASE_CASE_THRESHOLD) {
            insertion_sort(shared_mem + low, high - low + 1, cmp);
            continue;
        }

        int pi = partition_block(shared_mem, low, high, cmp);
        stack_low[++top] = low;
        stack_high[top] = pi - 1;
        stack_low[++top] = pi + 1;
        stack_high[top] = high;
    }

    __syncthreads();

    if (tid < n) {
        data[blockIdx.x * blockDim.x + tid] = shared_mem[tid];
    }
}

// ── Key-Value Block Sort Kernel (Task 3) ───────────────────────────────────

template <typename K, typename V, typename Comparator>
__global__ void kv_block_sort_kernel(
    K* keys,
    V* values,
    int n,
    Comparator cmp
) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int chunk_size = min(blockDim.x, n - block_offset);

    __shared__ K shared_keys[1024];
    __shared__ V shared_values[1024];

    if (tid < chunk_size) {
        shared_keys[tid] = keys[block_offset + tid];
        shared_values[tid] = values[block_offset + tid];
    }
    __syncthreads();

    int stack_low[1024];
    int stack_high[1024];
    int top = -1;
    stack_low[++top] = 0;
    stack_high[top] = chunk_size - 1;

    while (top >= 0) {
        int low = stack_low[top];
        int high = stack_high[top--];
        if (low >= high) continue;

        K pivot = shared_keys[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (!cmp(pivot, shared_keys[j])) {
                i++;
                K tmp_k = shared_keys[i]; shared_keys[i] = shared_keys[j]; shared_keys[j] = tmp_k;
                V tmp_v = shared_values[i]; shared_values[i] = shared_values[j]; shared_values[j] = tmp_v;
            }
        }
        i++;
        K tmp_k = shared_keys[i]; shared_keys[i] = shared_keys[high]; shared_keys[high] = tmp_k;
        V tmp_v = shared_values[i]; shared_values[i] = shared_values[high]; shared_values[high] = tmp_v;

        stack_low[++top] = low;
        stack_high[top] = i - 1;
        stack_low[++top] = i + 1;
        stack_high[top] = high;
    }

    __syncthreads();

    if (tid < chunk_size) {
        keys[block_offset + tid] = shared_keys[tid];
        values[block_offset + tid] = shared_values[tid];
    }
}

// ── Explicit Instantiations ────────────────────────────────────────────────

template __global__ void quicksort_shared<int, 1024, int (*)(const int&, const int&)>(int*, int, int (*)(const int&, const int&));
template __global__ void quicksort_shared<float, 1024, int (*)(const float&, const float&)>(float*, int, int (*)(const float&, const float&));
template __global__ void quicksort_shared<int64_t, 1024, int (*)(const int64_t&, const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
template __global__ void quicksort_shared<double, 1024, int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));

template __global__ void kv_block_sort_kernel<int, int, int (*)(const int&, const int&)>(int*, int*, int, int (*)(const int&, const int&));
template __global__ void kv_block_sort_kernel<float, int, int (*)(const float&, const float&)>(float*, int*, int, int (*)(const float&, const float&));
template __global__ void kv_block_sort_kernel<int64_t, int64_t, int (*)(const int64_t&, const int64_t&)>(int64_t*, int64_t*, int, int (*)(const int64_t&, const int64_t&));
template __global__ void kv_block_sort_kernel<double, double, int (*)(const double&, const double&)>(double*, double*, int, int (*)(const double&, const double&));

} // namespace detail
} // namespace blqs
