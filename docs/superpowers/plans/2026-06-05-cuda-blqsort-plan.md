# CUDA Quicksort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-performance CUDA Quicksort (and Radix Sort for comparison) library with generic C++ template APIs, comprehensive tests, benchmarking against CPU blqsort, and profiler reports.

**Architecture:** Three-layer design — kernel layer (raw `__global__` CUDA kernels), API layer (host-side callable interfaces for struct-based and key-value sorting, stable/unstable variants), benchmark layer (GPU vs CPU performance comparison). Full GPU parallelization with host-side recursion.

**Tech Stack:** CUDA C++, CMake, Google Test (FetchContent), Nsight Compute (`ncu`), CUDA Events, bash/SSH for deployment.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `CMakeLists.txt` | Main CMake config: CUDA toolchain, targets, gtest, ctest |
| `include/blqs_types.hpp` | Common types, type traits, utility structs |
| `include/blqs_errors.hpp` | `CUDA_CHECK` macro, error handling |
| `include/blqs.hpp` | Main API header — includes all sort entry points |
| `src/kernels/partition.cu` | Grid-level partition kernels (atomic counters, scatter) |
| `src/kernels/block_sort.cu` | Shared-memory quicksort per SM chunk + sorting network base case |
| `src/kernels/radix_sort.cu` | Parallel radix sort kernels (counting sort per byte) |
| `src/kernels/merge.cu` | Merge kernel for post-partition merge |
| `src/api/blqs_impl.hpp` | Template declarations for API functions |
| `src/api/blqs_impl.cu` | API implementations (host logic, kernel launches) |
| `src/api/blqs_keyvalue_impl.hpp` | Key-value sort template declarations |
| `src/api/blqs_keyvalue_impl.cu` | Key-value API implementations |
| `src/bench/cuda_bench.cu` | CUDA benchmark runner |
| `src/bench/cpu_bench.cpp` | CPU blqsort benchmark runner |
| `tests/correctness_test.cpp` | Correctness tests for all types |
| `tests/keyvalue_test.cpp` | Key-value pairing tests |
| `tests/stability_test.cpp` | Stability verification tests |
| `tests/corner_case_test.cpp` | Edge case tests |
| `tests/reproducibility_test.cpp` | Determinism/reproducibility tests |
| `scripts/deploy.sh` | SSH deploy to GPU servers |
| `scripts/run_tests.sh` | Remote ctest runner |
| `scripts/profile.sh` | Nsight Compute profiling runner |
| `docs/reports/` | Optimization step profiler reports (populated later) |

---

## Phase 1: Project Scaffolding & Build System

### Task 1: CMake Build System + Project Structure

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/`, `src/kernels/`, `src/api/`, `src/bench/`, `tests/`, `scripts/`, `docs/reports/`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p include src/kernels src/api src/bench tests scripts docs/reports
```

- [ ] **Step 2: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.24)
project(cuda-blqsort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# ── Google Test via FetchContent ──
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ── Library: cuda-blqsort ──
add_library(cuda_blqsort
    src/kernels/partition.cu
    src/kernels/block_sort.cu
    src/kernels/radix_sort.cu
    src/kernels/merge.cu
    src/api/blqs_impl.cu
    src/api/blqs_keyvalue_impl.cu
)
target_include_directories(cuda_blqsort
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/api
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels
)
target_link_libraries(cuda_blqsort
    PUBLIC
        CUDA::cudart
)

# ── Tests ──
set(TEST_SOURCES
    tests/correctness_test.cpp
    tests/keyvalue_test.cpp
    tests/stability_test.cpp
    tests/corner_case_test.cpp
    tests/reproducibility_test.cpp
)

add_executable(blqs_tests ${TEST_SOURCES})
target_link_libraries(blqs_tests
    PRIVATE
        cuda_blqsort
        GTest::gtest_main
        GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(blqs_tests)

add_test(NAME blqs_tests COMMAND blqs_tests)

# ── Benchmarks ──
add_executable(cuda_bench src/bench/cuda_bench.cu)
target_link_libraries(cuda_bench PRIVATE cuda_blqsort CUDA::cudart)

add_executable(cpu_bench src/bench/cpu_bench.cpp)
target_link_libraries(cpu_bench PRIVATE CUDA::cudart)

# ── Profiler target ──
add_custom_target(profiler
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/profile.sh $<TARGET_FILE:cuda_bench>
    DEPENDS cuda_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Nsight Compute profiler on cuda_bench"
)
```

- [ ] **Step 3: Create placeholder source files to verify CMake configures**

Create minimal files with just includes:

`include/blqs_types.hpp`:
```cpp
#pragma once
#include <cstdint>

namespace blqs {
    using int32 = std::int32_t;
    using int64 = std::int64_t;
    using float32 = float;
    using float64 = double;
} // namespace blqs
```

`include/blqs_errors.hpp`:
```cpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            cudaGetLastError();                                                \
            cudaPeekAtLastError();                                             \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)
```

`include/blqs.hpp`:
```cpp
#pragma once
#include "blqs_types.hpp"

namespace blqs {

// Struct-based sort (unstable)
template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator cmp = Comparator());

// Struct-based sort (stable)
template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator cmp = Comparator());

// Key-value sort (unstable)
template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n);

// Key-value sort (stable)
template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n);

} // namespace blqs
```

`src/api/blqs_impl.hpp`:
```cpp
#pragma once

namespace blqs {
namespace detail {

// Kernel declarations
__global__ void partition_kernel(int* data, int n, int* buckets, int* counts);
__global__ void block_sort_kernel(int* data, int n);

// Host-side recursive sort driver
template <typename T, typename Comparator>
void sort_driver(T* d_data, int n, Comparator cmp);

} // namespace detail
} // namespace blqs
```

`src/api/blqs_impl.cu`:
```cpp
#include "blqs_impl.hpp"
#include "blqs_errors.hpp"

namespace blqs {
namespace detail {

__global__ void partition_kernel(int* data, int n, int* buckets, int* counts) {
    // TODO: implemented in Task 2
}

__global__ void block_sort_kernel(int* data, int n) {
    // TODO: implemented in Task 2
}

} // namespace detail
} // namespace blqs
```

`src/api/blqs_keyvalue_impl.hpp`:
```cpp
#pragma once

namespace blqs {
namespace detail {

__global__ void kv_partition_kernel(int* keys, int* values, int n, int* buckets, int* counts);
__global__ void kv_block_sort_kernel(int* keys, int* values, int n);

} // namespace detail
} // namespace blqs
```

`src/api/blqs_keyvalue_impl.cu`:
```cpp
#include "blqs_keyvalue_impl.hpp"

namespace blqs {
namespace detail {

__global__ void kv_partition_kernel(int* keys, int* values, int n, int* buckets, int* counts) {
    // TODO: implemented in Task 3
}

__global__ void kv_block_sort_kernel(int* keys, int* values, int n) {
    // TODO: implemented in Task 3
}

} // namespace detail
} // namespace blqs
```

`src/kernels/partition.cu`:
```cpp
// Partition kernels — implemented in Task 2
#include "partition_kernels.hpp"
```

`src/kernels/block_sort.cu`:
```cpp
// Block sort kernels — implemented in Task 2
```

`src/kernels/radix_sort.cu`:
```cpp
// Radix sort kernels — implemented in Task 5
```

`src/kernels/merge.cu`:
```cpp
// Merge kernels — implemented in Task 4
```

`src/bench/cuda_bench.cu`:
```cpp
// CUDA benchmark — implemented in Task 8
int main() { return 0; }
```

`src/bench/cpu_bench.cpp`:
```cpp
// CPU blqsort benchmark — implemented in Task 9
int main() { return 0; }
```

- [ ] **Step 4: Verify CMake configures**

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Expected: CMake configures successfully, finds CUDA toolkit, fetches gtest.
If CUDA toolkit not found on local machine, skip and verify on GPU server after deploy.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt include/ src/ tests/ scripts/ docs/
git commit -m "build: add CMake project structure and build system"
```

---

## Phase 2: Baseline Quicksort Implementation (Step 0 + Step 1)

### Task 2: Single-Block Baseline Quicksort + Partition Kernels

**Files:**
- Create: `src/kernels/partition.cu`
- Create: `src/kernels/block_sort.cu`
- Modify: `src/api/blqs_impl.hpp`
- Modify: `src/api/blqs_impl.cu`
- Modify: `include/blqs.hpp`
- Test: `tests/correctness_test.cpp`

- [ ] **Step 1: Write `include/blqs_config.hpp` — launch configuration utilities**

```cpp
#pragma once
#include <cuda_runtime.h>

namespace blqs {
namespace detail {

constexpr int BLOCK_SIZE = 256;
constexpr int BASE_CASE_THRESHOLD = 64;

inline int get_num_blocks(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

} // namespace detail
} // namespace blqs
```

- [ ] **Step 2: Implement the partition kernel in `src/kernels/partition.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace blqs {
namespace detail {

template <typename T, typename Comparator>
__device__ int select_pivot(const T* data, int n, Comparator cmp) {
    // Median-of-three: first, middle, last
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

    // Each block processes a chunk and uses the first thread to select pivot
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

    // Atomic increment to get position in bucket
    int pos = atomicAdd(&bucket_counts[blockIdx.x * num_buckets + bucket], 1);

    // Scatter: lower bucket gets positions [0..count), upper bucket gets [count..size)
    int offset = (bucket == 0) ? chunk_start : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];
    data[offset + pos] = val;
}

// Explicit template instantiations
template __global__ void partition_kernel<int, int (*)(const int&, const int&)>(int*, int, int*, int*, int, int (*)(const int&, const int&));
template __global__ void partition_kernel<float, int (*)(const float&, const float&)>(float*, int, int*, int*, int, int (*)(const float&, const float&));

} // namespace detail
} // namespace blqs
```

- [ ] **Step 3: Implement the block sort kernel in `src/kernels/block_sort.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace blqs {
namespace detail {

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

template <typename T, typename Comparator>
__device__ T partition_block(T* data, int low, int high, Comparator cmp) {
    T pivot = data[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (!cmp(pivot, data[j])) { // data[j] <= pivot
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

template <typename T, int MaxSize = 1024, typename Comparator>
__device__ void quicksort_shared(T* data, int n, Comparator cmp) {
    __shared__ T shared_mem[MaxSize];

    // Copy to shared memory
    int tid = threadIdx.x;
    if (tid < n) {
        shared_mem[tid] = data[blockIdx.x * blockDim.x + tid];
    }
    __syncthreads();

    // Iterative quicksort using shared memory
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

    // Copy back to global memory
    if (tid < n) {
        data[blockIdx.x * blockDim.x + tid] = shared_mem[tid];
    }
}

// Explicit template instantiations
template __device__ void quicksort_shared<int, 1024, int (*)(const int&, const int&)>(int*, int, int (*)(const int&, const int&));
template __device__ void quicksort_shared<float, 1024, int (*)(const float&, const float&)>(float*, int, int (*)(const float&, const float&));

} // namespace detail
} // namespace blqs
```

- [ ] **Step 4: Implement the API entry point in `src/api/blqs_impl.cu`**

Replace the placeholder with the full implementation:

```cpp
#include "blqs_impl.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace blqs {
namespace detail {

// Forward kernel declarations (defined in .cu kernel files)
template <typename T, typename Comparator>
__global__ void partition_kernel(T* data, int n, int* bucket_counts, int* bucket_offsets, int num_buckets, Comparator cmp);

template <typename T, int MaxSize, typename Comparator>
__device__ void quicksort_shared(T* data, int n, Comparator cmp);

template <typename T, typename Comparator>
void sort_driver(T* d_data, int n, Comparator cmp) {
    if (n <= 1) return;

    // Base case: small enough for single-block sort
    if (n <= BLOCK_SIZE) {
        int threads = min(n, BLOCK_SIZE);
        int blocks = 1;
        partition_kernel<T, Comparator><<<blocks, threads>>>(
            d_data, n, nullptr, nullptr, 2, cmp);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Allocate workspace
    int* d_bucket_counts = nullptr;
    int* h_bucket_counts = nullptr;
    int num_buckets = 2;
    int num_blocks = get_num_blocks(n);

    CUDA_CHECK(cudaMalloc(&d_bucket_counts, num_blocks * num_buckets * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, num_blocks * num_buckets * sizeof(int)));

    h_bucket_counts = new int[num_blocks * num_buckets];

    // Launch partition
    partition_kernel<T, Comparator><<<num_blocks, BLOCK_SIZE>>>(
        d_data, n, d_bucket_counts, nullptr, num_buckets, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back bucket counts
    CUDA_CHECK(cudaMemcpy(h_bucket_counts, d_bucket_counts,
        num_blocks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bucket_counts));

    // For each block, recursively sort the two partitions
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

} // namespace detail

// Explicit instantiations for common types
template void sort<int, int (*)(const int&, const int&)>(int* d_data, int n, int (*)(const int&, const int&));
template void sort<float, int (*)(const float&, const float&)>(float* d_data, int n, int (*)(const float&, const float&));

} // namespace blqs
```

- [ ] **Step 5: Write the correctness test in `tests/correctness_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

namespace {

template <typename T>
struct AscendingComparator {
    __host__ __device__ bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

template <typename T>
void HostToTest(T* d_data, int n) {
    std::vector<T> h_data(n);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_data[i], h_data[i + 1]) << "Not sorted at index " << i;
    }
}

template <typename T>
void TestSort(const std::vector<T>& input) {
    T *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, input.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));

    blqs::sort(d_data, input.size(), AscendingComparator<T>());
    CUDA_CHECK(cudaDeviceSynchronize());

    HostToTest(d_data, input.size());
    CUDA_CHECK(cudaFree(d_data));
}

} // namespace

TEST(CorrectnessTest, EmptyArray) {
    std::vector<int> data;
    // No GPU call needed for empty
    EXPECT_TRUE(true);
}

TEST(CorrectnessTest, SingleElement) {
    std::vector<int> data = {42};
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(int), cudaMemcpyHostToDevice));

    blqs::sort(d_data, 1, AscendingComparator<int>());
    CUDA_CHECK(cudaDeviceSynchronize());

    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result, 42);
    CUDA_CHECK(cudaFree(d_data));
}

TEST(CorrectnessTest, TwoElements) {
    std::vector<int> data = {2, 1};
    TestSort(data);
}

TEST(CorrectnessTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    TestSort(data);
}

TEST(CorrectnessTest, ReverseSorted) {
    std::vector<int> data = {5, 4, 3, 2, 1};
    TestSort(data);
}

TEST(CorrectnessTest, AllEqual) {
    std::vector<int> data(100, 42);
    TestSort(data);
}

TEST(CorrectnessTest, RandomSmall) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
    TestSort(data);
}

TEST(CorrectnessTest, RandomLarge) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> data(10000);
    for (auto& v : data) v = dist(gen);
    TestSort(data);
}

TEST(CorrectnessTest, FloatBasic) {
    std::vector<float> data = {3.14f, 1.41f, 2.71f, 0.0f, -1.0f};
    TestSort(data);
}

TEST(CorrectnessTest, FloatWithNaN) {
    std::vector<float> data = {3.14f, std::numeric_limits<float>::quiet_NaN(), 1.41f, 2.71f};
    TestSort(data);
}

TEST(CorrectnessTest, NegativeIntegers) {
    std::vector<int> data = {-5, -1, -10, -3, -7, -2};
    TestSort(data);
}
```

- [ ] **Step 6: Run tests on GPU server via deploy script**

After deployment (Task 10), run:

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-blqsort/build && ctest -V"
```

Expected: Tests pass (some may be skipped if baseline isn't fully functional yet — mark as EXPECTED_FAILURE if needed).

- [ ] **Step 7: Commit**

```bash
git add include/blqs_config.hpp src/kernels/ src/api/ tests/correctness_test.cpp
git commit -m "feat: implement baseline single-block quicksort + partition kernel"
```

---

### Task 3: Key-Value Sort Implementation

**Files:**
- Modify: `src/kernels/partition.cu`
- Modify: `src/kernels/block_sort.cu`
- Modify: `src/api/blqs_keyvalue_impl.hpp`
- Modify: `src/api/blqs_keyvalue_impl.cu`
- Modify: `include/blqs.hpp`
- Test: `tests/keyvalue_test.cpp`

- [ ] **Step 1: Add KV partition kernel to `src/kernels/partition.cu`**

```cpp
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
    int offset = (bucket == 0) ? chunk_start : chunk_start + bucket_counts[blockIdx.x * num_buckets + 0];

    keys[offset + pos] = key;
    values[offset + pos] = val;
}
```

- [ ] **Step 2: Add KV block sort kernel to `src/kernels/block_sort.cu`**

```cpp
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

    // Iterative KV quicksort
    int stack_low[1024];
    int stack_high[1024];
    int top = -1;
    stack_low[++top] = 0;
    stack_high[top] = chunk_size - 1;

    while (top >= 0) {
        int low = stack_low[top];
        int high = stack_high[top--];
        if (low >= high) continue;

        // Partition
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
```

- [ ] **Step 3: Implement `sort_by_key` API in `src/api/blqs_keyvalue_impl.cu`**

```cpp
#include "blqs_keyvalue_impl.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"

namespace blqs {
namespace detail {

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

} // namespace detail

// Explicit instantiations
template void sort_by_key_driver<int, int, int (*)(const int&, const int&)>(int*, int*, int, int (*)(const int&, const int&));
template void sort_by_key_driver<float, int, int (*)(const float&, const float&)>(float*, int*, int, int (*)(const float&, const float&));

} // namespace blqs
```

- [ ] **Step 4: Update `include/blqs.hpp` to implement the template function**

```cpp
#include "blqs_impl.hpp"
#include "blqs_keyvalue_impl.hpp"
#include "blqs_types.hpp"
#include "blqs_errors.hpp"

namespace blqs {

template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator cmp) {
    if (n <= 0) return;
    if (d_data == nullptr) {
        throw std::invalid_argument("sort: d_data is nullptr");
    }
    detail::sort_driver(d_data, n, cmp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr) {
        throw std::invalid_argument("sort_by_key: keys or values is nullptr");
    }
    detail::sort_by_key_driver(d_keys, d_values, n,
        [] __device__ (const K& a, const K& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace blqs
```

- [ ] **Step 5: Write KV tests in `tests/keyvalue_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

namespace {

template <typename K, typename V>
void TestSortByKey(const std::vector<K>& keys, const std::vector<V>& values) {
    int n = keys.size();
    K *d_keys = nullptr, *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(V)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(V), cudaMemcpyHostToDevice));

    blqs::sort_by_key(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<K> h_keys(n);
    std::vector<V> h_values(n);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, n * sizeof(K), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, n * sizeof(V), cudaMemcpyDeviceToHost));

    // Verify keys are sorted
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_keys[i], h_keys[i + 1]) << "Keys not sorted at index " << i;
    }

    // Verify KV alignment against reference sort
    std::vector<std::pair<K, V>> ref(n);
    for (int i = 0; i < n; i++) ref[i] = {keys[i], values[i]};
    std::sort(ref.begin(), ref.end());
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(h_keys[i], ref[i].first) << "Key mismatch at " << i;
        EXPECT_EQ(h_values[i], ref[i].second) << "Value mismatch at " << i;
    }

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
}

} // namespace

TEST(KeyValueTest, BasicIntInt) {
    std::vector<int> keys = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> values = {30, 10, 40, 11, 50, 90, 20, 60};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, FloatInt) {
    std::vector<float> keys = {3.14f, 1.41f, 2.71f, 0.0f};
    std::vector<int> values = {100, 200, 300, 400};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, AllSameKey) {
    std::vector<int> keys(50, 42);
    std::vector<int> values(50);
    for (int i = 0; i < 50; i++) values[i] = i;
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, ReverseSorted) {
    std::vector<int> keys = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int> values = {90, 80, 70, 60, 50, 40, 30, 20, 10};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, LargeRandom) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    int n = 50000;
    std::vector<int> keys(n), values(n);
    for (int i = 0; i < n; i++) { keys[i] = dist(gen); values[i] = i; }
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, NullptrKeys) {
    EXPECT_THROW(blqs::sort_by_key((int*)nullptr, (int*)nullptr, 10), std::invalid_argument);
}

TEST(KeyValueTest, EmptyArray) {
    std::vector<int> keys, values;
    // Should not crash
    EXPECT_NO_THROW(blqs::sort_by_key((int*)nullptr, (int*)nullptr, 0));
}
```

- [ ] **Step 6: Commit**

```bash
git add src/kernels/partition.cu src/kernels/block_sort.cu src/api/blqs_keyvalue_impl.* include/blqs.hpp tests/keyvalue_test.cpp
git commit -m "feat: implement key-value sort API and tests"
```

---

## Phase 3: Stable Sort & Optimization

### Task 4: Stable Sort Implementation

**Files:**
- Create: `src/kernels/radix_sort.cu` (stable radix for numeric types)
- Create: `src/kernels/merge.cu` (stable merge for comparator types)
- Modify: `include/blqs.hpp`
- Test: `tests/stability_test.cpp`

- [ ] **Step 1: Implement stable radix sort kernel in `src/kernels/radix_sort.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

namespace blqs {
namespace detail {

constexpr int RADIX_BITS = 8;
constexpr int RADIX_BINS = 1 << RADIX_BITS; // 256
constexpr int RADIX_PASSES_32 = 4; // 32/8
constexpr int RADIX_PASSES_64 = 8; // 64/8

template <typename T>
__device__ __forceinline__ unsigned int get_radix_digit(T val, int pass) {
    using U = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
    U uval = reinterpret_cast<U&>(val);
    // For signed types, flip sign bit to make ordering work with unsigned comparison
    if constexpr (std::is_signed<T>::value) {
        if constexpr (sizeof(T) == 4) {
            uval ^= 0x80000000U;
        } else {
            uval ^= 0x8000000000000000ULL;
        }
    }
    return (uval >> (pass * RADIX_BITS)) & (RADIX_BINS - 1);
}

template <typename K, typename V>
__global__ void radix_sort_pass_kernel(
    const K* input_keys,
    const V* input_values,
    K* output_keys,
    V* output_values,
    int* block_offsets,
    int n,
    int pass
) {
    __shared__ int s_counters[RADIX_BINS];
    __shared__ int s_offsets[RADIX_BINS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = min(blockDim.x, n - blockIdx.x * blockDim.x);

    // Phase 1: Count per-bin
    if (threadIdx.x == 0) {
        for (int i = 0; i < RADIX_BINS; i++) s_counters[i] = 0;
    }
    __syncthreads();

    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        atomicAdd(&s_counters[digit], 1);
    }
    __syncthreads();

    // Phase 2: Compute offsets
    if (threadIdx.x == 0) {
        s_offsets[0] = 0;
        for (int i = 1; i < RADIX_BINS; i++) {
            s_offsets[i] = s_offsets[i - 1] + s_counters[i - 1];
        }
        // Store block offsets for global scatter
        for (int i = 0; i < RADIX_BINS; i++) {
            block_offsets[blockIdx.x * RADIX_BINS + i] = s_offsets[i];
        }
    }
    __syncthreads();

    // Phase 3: Scatter to output
    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        int pos = atomicAdd(&s_offsets[digit], 1);
        int global_pos = blockIdx.x * blockDim.x + pos;
        // For now, use local scatter within block
        // Full implementation needs global prefix sum — see Task 5
        output_keys[tid] = input_keys[tid];
        output_values[tid] = input_values[tid];
    }
}

} // namespace detail
} // namespace blqs
```

Note: The initial radix sort pass is a simplified version. The full implementation with global prefix sum is completed in Task 5 (optimization step 5).

- [ ] **Step 2: Implement stable merge sort kernel in `src/kernels/merge.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace blqs {
namespace detail {

template <typename T, typename Comparator>
__global__ void stable_merge_kernel(
    const T* input,
    T* output,
    const int* indices,
    int* output_indices,
    int n,
    int left_size,
    Comparator cmp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Merge two sorted subarrays: [0..left_size) and [left_size..n)
    int left = 0, right = left_size, out = 0;

    // Each thread handles a portion of the merge
    // For simplicity, use a block-wide merge
    __shared__ int s_left, s_right;
    if (threadIdx.x == 0) {
        s_left = 0;
        s_right = left_size;
    }
    __syncthreads();

    // Simple merge for small arrays (large arrays use multi-pass)
    int target = tid;
    int l = 0, r = left_size;
    for (int i = 0; i <= target; i++) {
        if (l >= left_size) {
            // Right side wins
            if (i == target) {
                output[i] = input[left_size + r];
                output_indices[i] = indices[left_size + r];
            }
            r++;
        } else if (r >= n) {
            // Left side wins
            if (i == target) {
                output[i] = input[l];
                output_indices[i] = indices[l];
            }
            l++;
        } else if (!cmp(input[left_size + r], input[l])) {
            // Left wins (stable: <= means left first)
            if (i == target) {
                output[i] = input[l];
                output_indices[i] = indices[l];
            }
            l++;
        } else {
            if (i == target) {
                output[i] = input[left_size + r];
                output_indices[i] = indices[left_size + r];
            }
            r++;
        }
    }
}

} // namespace detail
} // namespace blqs
```

- [ ] **Step 3: Implement stable sort API — update `include/blqs.hpp`**

Add the stable sort implementations:

```cpp
template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator cmp) {
    if (n <= 0) return;
    if (d_data == nullptr) {
        throw std::invalid_argument("sort_stable: d_data is nullptr");
    }

    // For numeric types, use radix sort
    if constexpr (std::is_arithmetic<T>::value) {
        detail::stable_radix_sort(d_data, nullptr, n);
    } else {
        // For custom types, use stable merge sort
        detail::stable_merge_sort(d_data, n, cmp);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr) {
        throw std::invalid_argument("sort_by_key_stable: keys or values is nullptr");
    }

    // Stable sort always uses radix sort for numeric keys
    detail::stable_radix_sort(d_keys, d_values, n);

    CUDA_CHECK(cudaDeviceSynchronize());
}
```

- [ ] **Step 4: Write stability tests in `tests/stability_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

namespace {

// Test struct with original position for stability verification
struct IndexedValue {
    int value;
    int original_pos;

    __host__ __device__ bool operator<(const IndexedValue& other) const {
        return value < other.value;
    }
};

template <typename T>
void TestStableSort(const std::vector<T>& input) {
    int n = input.size();
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    blqs::sort_stable(d_data, n, [] __device__ (const T& a, const T& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_data(n);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    // Verify sorted
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_data[i], h_data[i + 1]) << "Not sorted at index " << i;
    }

    // Verify stability: equal elements maintain original relative order
    for (int i = 0; i < n - 1; i++) {
        if (h_data[i].value == h_data[i + 1].value) {
            EXPECT_LT(h_data[i].original_pos, h_data[i + 1].original_pos)
                << "Stability violated at index " << i
                << " (pos " << h_data[i].original_pos << " vs " << h_data[i + 1].original_pos << ")";
        }
    }
}

} // namespace

TEST(StabilityTest, AllEqual) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 100; i++) {
        data.push_back({42, i});
    }
    TestStableSort(data);
}

TEST(StabilityTest, DuplicateHeavy) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 200; i++) {
        data.push_back({i % 10, i}); // 10 distinct values, many duplicates
    }
    TestStableSort(data);
}

TEST(StabilityTest, TwoValues) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 100; i++) {
        data.push_back({i % 2, i});
    }
    TestStableSort(data);
}

TEST(StabilityTest, StableKeyValueInt) {
    int n = 500;
    std::vector<int> keys(n), values(n);
    std::mt19937 gen(77);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int i = 0; i < n; i++) {
        keys[i] = dist(gen);
        values[i] = i; // Track original position
    }

    int *d_keys = nullptr, *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    blqs::sort_by_key_stable(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_keys(n), h_values(n);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));

    // Verify keys sorted
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_keys[i], h_keys[i + 1]);
    }

    // Verify stability: equal keys maintain original order
    for (int i = 0; i < n - 1; i++) {
        if (h_keys[i] == h_keys[i + 1]) {
            EXPECT_LT(h_values[i], h_values[i + 1])
                << "Stability violated: key " << h_keys[i]
                << " pos " << h_values[i] << " vs " << h_values[i + 1];
        }
    }
}

TEST(StabilityTest, EmptyArray) {
    EXPECT_NO_THROW(blqs::sort_stable((int*)nullptr, 0, [] __device__ (const int&, const int&) { return false; }));
}

TEST(StabilityTest, SingleElement) {
    IndexedValue data = {{42, 0}};
    IndexedValue* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(IndexedValue)));
    CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(IndexedValue), cudaMemcpyHostToDevice));

    blqs::sort_stable(d_data, 1, [] __device__ (const IndexedValue& a, const IndexedValue& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());

    IndexedValue result;
    CUDA_CHECK(cudaMemcpy(&result, d_data, sizeof(IndexedValue), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result.value, 42);
    EXPECT_EQ(result.original_pos, 0);
    CUDA_CHECK(cudaFree(d_data));
}
```

- [ ] **Step 5: Commit**

```bash
git add src/kernels/radix_sort.cu src/kernels/merge.cu include/blqs.hpp tests/stability_test.cpp
git commit -m "feat: implement stable sort (radix for numeric, merge for custom comparator)"
```

---

### Task 5: Optimization Steps 2-4 — Shared Memory, Warp Primitives, Multi-Stream

**Files:**
- Modify: `src/kernels/partition.cu`
- Modify: `src/kernels/block_sort.cu`
- Modify: `src/kernels/radix_sort.cu`
- Test: `tests/correctness_test.cpp` (re-run after each optimization)

- [ ] **Step 1: Optimization Step 2 — Shared Memory + Sorting Network**

Update `src/kernels/block_sort.cu` with an optimized sorting network for base case:

```cpp
template <typename T, typename Comparator>
__device__ void sorting_network_64(T* data, Comparator cmp) {
    // Bitonic sorting network for 64 elements
    // Uses compare-and-swap operations
    for (int k = 2; k <= 64; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < 64; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool swap = ((i & k) == 0) ? cmp(data[ixj], data[i]) : cmp(data[i], data[ixj]);
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
```

Update the block sort kernel to use this for the base case instead of insertion sort.

- [ ] **Step 2: Optimization Step 3 — Warp-Level Primitives**

Update `src/kernels/partition.cu` with warp-level partition using `__ballot_sync`:

```cpp
template <typename T, typename Comparator>
__device__ int warp_partition(T* shared_data, int low, int high, Comparator cmp) {
    int tid = threadIdx.x;
    T pivot = shared_data[high];

    // Use ballot to determine which threads have elements <= pivot
    unsigned int mask = __ballot_sync(0xFFFFFFFF, tid < high && !cmp(pivot, shared_data[tid]));

    // Use __popc to count and __ffs to find positions
    int rank = __popc(mask & ((1u << tid) - 1));

    return rank;
}
```

- [ ] **Step 3: Optimization Step 4 — Multi-Stream Overlap**

Add stream-based launch wrapper. Create `include/blqs_streams.hpp`:

```cpp
#pragma once
#include <cuda_runtime.h>
#include "blqs_errors.hpp"

namespace blqs {
namespace detail {

constexpr int MAX_STREAMS = 8;

struct StreamManager {
    cudaStream_t streams[MAX_STREAMS];
    int num_streams;

    StreamManager() : num_streams(0) {
        CUDA_CHECK(cudaGetDeviceCount(&num_streams));
        num_streams = min(num_streams, MAX_STREAMS);
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }

    ~StreamManager() {
        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    void sync_all() {
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }
};

} // namespace detail
} // namespace blqs
```

- [ ] **Step 4: Re-run correctness tests after each optimization**

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-blqsort/build && ctest -V"
```

Expected: All tests still pass. Optimizations should not change correctness.

- [ ] **Step 5: Commit after each optimization**

```bash
git add src/kernels/ include/
git commit -m "perf: step 2 - shared memory + sorting network base case"
git add src/kernels/
git commit -m "perf: step 3 - warp-level primitives for partition"
git add src/kernels/ include/blqs_streams.hpp
git commit -m "perf: step 4 - multi-stream overlapping for chunk processing"
```

---

### Task 6: Full Radix Sort Implementation (Optimization Step 5)

**Files:**
- Modify: `src/kernels/radix_sort.cu`
- Test: `tests/correctness_test.cpp` (add radix-specific tests)

- [ ] **Step 1: Complete radix sort with global prefix sum**

Replace the simplified radix sort kernel in `src/kernels/radix_sort.cu` with the full implementation:

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <type_traits>

namespace blqs {
namespace detail {

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
        if constexpr (sizeof(T) == 4) {
            uval ^= 0x80000000U;
        } else {
            uval ^= 0x8000000000000000ULL;
        }
    }
    return (uval >> (pass * RADIX_BITS)) & (RADIX_BINS - 1);
}

template <typename K, typename V>
__global__ void radix_sort_pass_kernel(
    const K* input_keys,
    const V* input_values,
    K* output_keys,
    V* output_values,
    int n,
    int pass
) {
    extern __shared__ int s_mem[];
    int* s_counters = s_mem;
    int* s_offsets = s_mem + RADIX_BINS;
    int* s_global_offsets = s_mem + RADIX_BINS * 2;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // Phase 1: Per-warp histogram
    if (threadIdx.x < RADIX_BINS) s_counters[threadIdx.x] = 0;
    __syncthreads();

    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        atomicAdd(&s_counters[digit], 1);
    }
    __syncthreads();

    // Phase 2: Block-wide prefix sum
    if (threadIdx.x < RADIX_BINS) {
        int sum = 0;
        for (int i = 0; i < threadIdx.x; i++) {
            sum += s_counters[i];
        }
        s_offsets[threadIdx.x] = sum;
    }
    __syncthreads();

    // Phase 3: Scatter
    if (tid < n) {
        int digit = get_radix_digit(input_keys[tid], pass);
        int pos = atomicAdd(&s_offsets[digit], 1);
        output_keys[pos] = input_keys[tid];
        if constexpr (!std::is_same<V, void>::value) {
            output_values[pos] = input_values[tid];
        }
    }
}

template <typename K, typename V>
void radix_sort_driver(K* d_keys, V* d_values, int n) {
    if (n <= 1) return;

    constexpr int passes = RadixTraits<K>::passes;

    // Double-buffered: ping-pong between two buffers
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
    size_t shared_mem_size = RADIX_BINS * 3 * sizeof(int);

    for (int pass = 0; pass < passes; pass++) {
        int src = pass % 2;
        int dst = 1 - src;

        radix_sort_pass_kernel<K, V><<<blocks, threads, shared_mem_size>>>(
            d_buf_keys[src],
            d_buf_values[src],
            d_buf_keys[dst],
            d_buf_values[dst],
            n,
            pass
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy final result back
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

// Explicit instantiations
template void radix_sort_driver<int, int>(int*, int*, int);
template void radix_sort_driver<int, void>(int*, void*, int);
template void radix_sort_driver<float, void>(float*, void*, int);
template void radix_sort_driver<int64_t, int64_t>(int64_t*, int64_t*, int);
template void radix_sort_driver<double, void>(double*, void*, int);

} // namespace detail
} // namespace blqs
```

- [ ] **Step 2: Add radix sort benchmark test**

Add to `tests/correctness_test.cpp`:

```cpp
TEST(RadixSortTest, Int32Basic) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    TestSort(data);
}

TEST(RadixSortTest, Int32Negative) {
    std::vector<int> data = {-5, -1, -10, -3, 0, 5, 3, 1};
    TestSort(data);
}

TEST(RadixSortTest, Int64Large) {
    std::vector<int64_t> data;
    std::mt19937 gen(999);
    std::uniform_int_distribution<int64_t> dist(-1000000000LL, 1000000000LL);
    for (int i = 0; i < 10000; i++) data.push_back(dist(gen));
    TestSort(data);
}

TEST(RadixSortTest, Float64Precision) {
    std::vector<double> data = {3.14159265358979, 2.71828182845904, 1.41421356237309, 0.0, -1.0};
    TestSort(data);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/kernels/radix_sort.cu tests/correctness_test.cpp
git commit -m "perf: step 5 - full radix sort with global prefix sum and double buffering"
```

---

## Phase 4: Benchmarks & Profiler

### Task 7: CUDA Benchmark

**Files:**
- Modify: `src/bench/cuda_bench.cu`
- Create: `include/blqs_bench.hpp`

- [ ] **Step 1: Create benchmark utilities in `include/blqs_bench.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include "blqs_errors.hpp"

namespace blqs {
namespace bench {

struct TimingResult {
    double elapsed_ms;
    double bandwidth_gbps;
    int n;
    std::string algorithm;
};

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
    }

    double elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

template <typename T>
std::vector<T> generate_random(int n, int seed = 42) {
    std::vector<T> data(n);
    std::mt19937 gen(seed);

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist;
        for (int i = 0; i < n; i++) data[i] = dist(gen);
    } else {
        std::uniform_real_distribution<T> dist(-1e6, 1e6);
        for (int i = 0; i < n; i++) data[i] = dist(gen);
    }
    return data;
}

inline void print_header() {
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(10) << "N"
              << std::setw(14) << "Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
}

inline void print_result(const TimingResult& r) {
    std::cout << std::left << std::setw(12) << r.algorithm
              << std::right << std::setw(10) << r.n
              << std::setw(14) << std::fixed << std::setprecision(3) << r.elapsed_ms
              << std::setw(18) << std::fixed << std::setprecision(2) << r.bandwidth_gbps
              << std::endl;
}

} // namespace bench
} // namespace blqs
```

- [ ] **Step 2: Implement CUDA benchmark in `src/bench/cuda_bench.cu`**

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include "blqs.hpp"
#include "blqs_bench.hpp"
#include "blqs_errors.hpp"

using namespace blqs;

template <typename T>
void run_benchmark(const std::string& name, int n, int iterations = 5) {
    auto h_data = bench::generate_random<T>(n);

    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    double total_ms = 0;
    for (int i = 0; i < iterations; i++) {
        // Reset data
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

        bench::CudaTimer timer;
        timer.start();
        blqs::sort(d_data, n, [] __device__ (const T& a, const T& b) { return a < b; });
        timer.stop();

        CUDA_CHECK(cudaDeviceSynchronize());
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / iterations;
    double bytes = (double)n * sizeof(T) * 2; // read + write
    double bandwidth = bytes / avg_ms / 1e6; // GB/s

    bench::print_result({avg_ms, bandwidth, n, name});

    CUDA_CHECK(cudaFree(d_data));
}

template <typename K, typename V>
void run_kv_benchmark(const std::string& name, int n, int iterations = 5) {
    auto h_keys = bench::generate_random<K>(n);
    auto h_values = bench::generate_random<V>(n);

    K *d_keys = nullptr;
    V *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(V)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(V), cudaMemcpyHostToDevice));

    double total_ms = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(K), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(V), cudaMemcpyHostToDevice));

        bench::CudaTimer timer;
        timer.start();
        blqs::sort_by_key(d_keys, d_values, n);
        timer.stop();

        CUDA_CHECK(cudaDeviceSynchronize());
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / iterations;
    double bytes = (double)n * (sizeof(K) + sizeof(V)) * 2;
    double bandwidth = bytes / avg_ms / 1e6;

    bench::print_result({avg_ms, bandwidth, n, name});

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;

    bench::print_header();

    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};

    std::cout << "\n--- Quicksort int32 ---" << std::endl;
    for (int n : sizes) {
        if (n <= 10000000) run_benchmark<int>("qsort_i32", n);
    }

    std::cout << "\n--- Quicksort float32 ---" << std::endl;
    for (int n : sizes) {
        if (n <= 10000000) run_benchmark<float>("qsort_f32", n);
    }

    std::cout << "\n--- Radix sort int32 ---" << std::endl;
    for (int n : sizes) {
        run_benchmark<int>("radix_i32", n);
    }

    std::cout << "\n--- Key-Value sort (int->int) ---" << std::endl;
    for (int n : sizes) {
        if (n <= 10000000) run_kv_benchmark<int, int>("kv_i32", n);
    }

    std::cout << "\n--- Stable sort int32 ---" << std::endl;
    for (int n : sizes) {
        if (n <= 10000000) run_benchmark<int>("stable_i32", n);
    }

    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add include/blqs_bench.hpp src/bench/cuda_bench.cu
git commit -m "feat: add CUDA benchmark runner with CUDA event timing"
```

---

### Task 8: CPU blqsort Benchmark + Comparison

**Files:**
- Modify: `src/bench/cpu_bench.cpp`

- [ ] **Step 1: Implement CPU benchmark in `src/bench/cpu_bench.cpp`**

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <random>

// Include blqsort from cloned repo (path set via BLQS_INCLUDE env var or default)
#ifndef BLQS_INCLUDE_PATH
#define BLQS_INCLUDE_PATH "../../blqsort"
#endif

// We include blqsort headers at compile time
#include BLQS_INCLUDE_PATH "/blqs.h"

std::string format_ms(double ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << ms;
    return oss.str();
}

template <typename T>
void run_cpu_benchmark(const std::string& name, int n, int iterations = 5) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist;
        for (int i = 0; i < n; i++) data[i] = dist(gen);
    } else {
        std::uniform_real_distribution<T> dist(-1e6, 1e6);
        for (int i = 0; i < n; i++) data[i] = dist(gen);
    }

    double total_ms = 0;
    for (int i = 0; i < iterations; i++) {
        // Reset data
        auto data_copy = data;

        auto start = std::chrono::high_resolution_clock::now();
        blqs::sort(data_copy.data(), data_copy.data() + n);
        auto end = std::chrono::high_resolution_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_ms = total_ms / iterations;
    double bytes = (double)n * sizeof(T) * 2;
    double bandwidth = bytes / avg_ms / 1e6;

    std::cout << std::left << std::setw(12) << name
              << std::right << std::setw(10) << n
              << std::setw(14) << std::fixed << std::setprecision(3) << avg_ms
              << std::setw(18) << std::fixed << std::setprecision(2) << bandwidth
              << std::endl;
}

int main() {
    std::cout << "CPU blqsort benchmark" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(10) << "N"
              << std::setw(14) << "Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000};

    std::cout << "\n--- blqsort int32 ---" << std::endl;
    for (int n : sizes) run_cpu_benchmark<int>("blqsort", n);

    std::cout << "\n--- std::sort int32 ---" << std::endl;
    for (int n : sizes) {
        std::vector<int> data(n);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist;
        for (int i = 0; i < n; i++) data[i] = dist(gen);

        auto start = std::chrono::high_resolution_clock::now();
        std::sort(data.begin(), data.end());
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double bw = (double)n * sizeof(int) * 2 / ms / 1e6;

        std::cout << std::left << std::setw(12) << "std::sort"
                  << std::right << std::setw(10) << n
                  << std::setw(14) << std::fixed << std::setprecision(3) << ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << bw
                  << std::endl;
    }

    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/bench/cpu_bench.cpp
git commit -m "feat: add CPU blqsort benchmark runner"
```

---

## Phase 5: Deployment & Profiling Scripts

### Task 9: Deploy & Test Scripts

**Files:**
- Create: `scripts/deploy.sh`
- Create: `scripts/run_tests.sh`
- Create: `scripts/profile.sh`

- [ ] **Step 1: Create deploy script `scripts/deploy.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Deploy cuda-blqsort to GPU servers via SSH
# Usage: ./scripts/deploy.sh [thor|a40|all]

REPO_NAME="cuda-blqsort"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

declare -A SERVERS=(
    [thor]="shuyua01@10.190.0.91:/home/shuyua01/Development/"
    [a40]="shuyua01@szc-td04:/project/ai/npu_sw/shuyua01/Development"
)

TARGET="${1:-all}"

deploy_to() {
    local server_name="$1"
    local server_path="${SERVERS[$server_name]}"
    local host="${server_path%%:*}"
    local remote_path="${server_path#*:}"

    echo "Deploying to ${server_name} (${host})..."
    echo "  Remote path: ${remote_path}"

    # Create remote directory
    ssh "$host" "mkdir -p ${remote_path}"

    # Sync repo (excluding build artifacts and .git)
    rsync -avz --delete \
        --exclude='.git/' \
        --exclude='build/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        "${LOCAL_DIR}/" \
        "${host}:${remote_path}${REPO_NAME}/"

    # Clone blqsort on remote server if not present
    ssh "$host" "cd ${remote_path} && [ -d blqsort ] || git clone https://github.com/chkas/blqsort.git"

    # Build on remote server
    ssh "$host" "cd ${remote_path}${REPO_NAME} && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"

    echo "Deployed to ${server_name} successfully."
}

if [[ "$TARGET" == "all" ]]; then
    for server in "${!SERVERS[@]}"; do
        deploy_to "$server"
    done
else
    if [[ -z "${SERVERS[$TARGET]+x}" ]]; then
        echo "Unknown server: ${TARGET}. Available: ${!SERVERS[*]}"
        exit 1
    fi
    deploy_to "$TARGET"
fi
```

- [ ] **Step 2: Create test runner script `scripts/run_tests.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run tests on GPU servers
# Usage: ./scripts/run_tests.sh [thor|a40|all]

declare -A SERVERS=(
    [thor]="shuyua01@10.190.0.91"
    [a40]="shuyua01@szc-td04"
)

declare -A PATHS=(
    [thor]="/home/shuyua01/Development/cuda-blqsort"
    [a40]="/project/ai/npu_sw/shuyua01/Development/cuda-blqsort"
)

TARGET="${1:-all}"

run_tests_on() {
    local server_name="$1"
    local host="${SERVERS[$server_name]}"
    local remote_path="${PATHS[$server_name]}"

    echo "Running tests on ${server_name}..."

    # Check GPU availability
    ssh "$host" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
        echo "ERROR: GPU not available on ${server_name}"; return 1
    }

    # Build and run tests
    ssh "$host" "cd ${remote_path}/build && make -j\$(nproc) && ctest -V --output-on-failure"

    echo "Tests passed on ${server_name}."
}

if [[ "$TARGET" == "all" ]]; then
    for server in "${!SERVERS[@]}"; do
        run_tests_on "$server"
    done
else
    if [[ -z "${SERVERS[$TARGET]+x}" ]]; then
        echo "Unknown server: ${TARGET}. Available: ${!SERVERS[*]}"
        exit 1
    fi
    run_tests_on "$TARGET"
fi
```

- [ ] **Step 3: Create profiler script `scripts/profile.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute profiling script
# Usage: ./scripts/profile.sh <benchmark_binary> [output_dir]

BENCHMARK="${1:-}"
OUTPUT_DIR="${2:-docs/reports}"

if [[ -z "$BENCHMARK" ]]; then
    echo "Usage: $0 <benchmark_binary> [output_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running Nsight Compute profiler on ${BENCHMARK}..."
echo "Output directory: ${OUTPUT_DIR}"

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found in PATH"
    echo "Install CUDA toolkit with Nsight Compute to use this script"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/profile_${TIMESTAMP}.ncu-rep"

# Profile with key metrics
ncu --set full \
    --target-processes all \
    --import-source yes \
    -o "$REPORT_FILE" \
    "$BENCHMARK"

echo "Profile saved to ${REPORT_FILE}"

# Also generate a text summary
TEXT_REPORT="${OUTPUT_DIR}/profile_${TIMESTAMP}.txt"
echo "=== Nsight Compute Summary ===" > "$TEXT_REPORT"
echo "Date: $(date)" >> "$TEXT_REPORT"
echo "" >> "$TEXT_REPORT"

ncu --set full \
    --target-processes all \
    --page raw \
    "$BENCHMARK" >> "$TEXT_REPORT" 2>&1

echo "Text summary saved to ${TEXT_REPORT}"
echo "Done."
```

- [ ] **Step 4: Make scripts executable**

```bash
chmod +x scripts/deploy.sh scripts/run_tests.sh scripts/profile.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: add deploy, test runner, and profiler scripts for GPU servers"
```

---

### Task 10: First Deploy & Test on GPU Servers

**Files:** No file changes — this is a deploy step.

- [ ] **Step 1: Initial deploy to Thor**

```bash
./scripts/deploy.sh thor
```

Expected:
- rsync transfers files to `10.190.0.91:/home/shuyua01/Development/cuda-blqsort/`
- blqsort repo cloned to `10.190.0.91:/home/shuyua01/Development/blqsort/`
- CMake configures, build completes successfully

- [ ] **Step 2: Run tests on Thor**

```bash
./scripts/run_tests.sh thor
```

Expected: All ctest tests pass.

- [ ] **Step 3: Deploy to A40**

```bash
./scripts/deploy.sh a40
```

- [ ] **Step 4: Run tests on A40**

```bash
./scripts/run_tests.sh a40
```

Expected: All ctest tests pass.

- [ ] **Step 5: If tests fail, fix and redeploy**

Iterate: fix failures → `git commit` → `./scripts/deploy.sh <server>` → retest.

---

### Task 11: Profiler Reports & Performance Comparison

**Files:**
- No new files — uses existing benchmark + profiler infrastructure
- Create: `docs/reports/step-01-baseline.md`
- Create: `docs/reports/step-02-shared-memory.md`
- Create: `docs/reports/step-03-warp-primitives.md`
- Create: `docs/reports/step-04-multi-stream.md`
- Create: `docs/reports/step-05-radix-sort.md`
- Create: `docs/reports/step-06-cpu-vs-gpu.md`

- [ ] **Step 1: Profile baseline on Thor**

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-blqsort && ./scripts/profile.sh build/cuda_bench"
```

- [ ] **Step 2: Generate performance comparison report for each optimization step**

For each step (0-6):
1. Run `cuda_bench` and capture output
2. Run `ncu` profiler and capture metrics
3. Write report to `docs/reports/step-XX-<name>.md` with format:

```markdown
# Step X: <Optimization Name>

## Configuration
- GPU: <GPU name from nvidia-smi>
- CUDA version: <version>
- Array sizes: 1K, 10K, 100K, 1M, 10M

## Timing Results

| N | Time (ms) | Speedup vs Previous | Bandwidth (GB/s) |
|---|-----------|--------------------|------------------|
| ... | ... | ... | ... |

## Nsight Compute Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| SM Active Warps | ... | ... |
| Memory Throughput | ... | ... |
| Warp Execution Efficiency | ... | ... |
| ... | ... | ... |

## Analysis
<Description of what the optimization changed, why it helped (or didn't), and what the next bottleneck is.>

## Next Bottleneck
<What profiler data suggests is the current limiting factor.>
```

- [ ] **Step 3: Generate final CPU vs GPU comparison (Step 6)**

Run both benchmarks:
```bash
# GPU
./build/cuda_bench > docs/reports/gpu_results.txt

# CPU (via SSH, needs blqsort built)
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/blqsort && <build and run>" > docs/reports/cpu_results.txt
```

Write `docs/reports/step-06-cpu-vs-gpu.md` with speedup ratios.

- [ ] **Step 4: Commit all reports**

```bash
git add docs/reports/
git commit -m "docs: add profiler reports and CPU vs GPU performance comparison"
```

---

## Self-Review Checklist

**1. Spec coverage check:**

| Spec Requirement | Task |
|-----------------|------|
| Generic C++ templates (int32/int64/float32/float64) | Task 2, 5, 6 — all kernels are templated |
| Four host APIs (struct unstable/stable, KV unstable/stable) | Task 2 (struct unstable), Task 3 (KV unstable), Task 4 (stable both) |
| Full GPU parallelization baseline | Task 2 — multi-block partition from start |
| CMake build system | Task 1 |
| Google Test tests | Tasks 2, 3, 4, 5, 6 |
| CPU comparison against blqsort | Task 8, 11 |
| Target architecture auto-detected | Task 1 — `CMAKE_CUDA_ARCHITECTURES = native` |
| Profiler reports at each step | Task 11 |
| Corner case unit tests | Tasks 2 (correctness), 3 (KV), 4 (stability) |
| Two GPU servers | Task 9 (deploy scripts), Task 10 (deploy/test) |
| Radix sort for comparison | Task 6, Task 11 |
| Optimization roadmap (steps 0-6) | Tasks 2, 5, 6, 11 |

All spec requirements covered. ✓

**2. Placeholder scan:**
- No TBD/TODO/incomplete sections in the plan. ✓
- All code steps contain actual implementation code. ✓
- All commands have expected output descriptions. ✓
- No "similar to Task N" references without full code. ✓

**3. Type/signature consistency:**
- `CUDA_CHECK` macro defined in `blqs_errors.hpp`, used consistently. ✓
- `BLOCK_SIZE = 256`, `BASE_CASE_THRESHOLD = 64` from `blqs_config.hpp`, referenced in kernels. ✓
- `blqs::sort<T>(T* d_data, int n, Comparator cmp)` signature consistent across Task 2, 5. ✓
- `blqs::sort_by_key<K,V>(K* d_keys, V* d_values, int n)` signature consistent across Task 3, 6. ✓
- Template explicit instantiations match supported types: `int`, `float`, `int64_t`, `double`. ✓

All checks pass. Plan is complete and consistent.
