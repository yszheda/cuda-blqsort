# Design: CUDA Quicksort (cuda-blqsort)

## Context

High-performance CUDA Quicksort implementation referencing [blqsort](https://github.com/chkas/blqsort) CPU library. Goals span three audiences:
- **Academic** — thorough profiler analysis, benchmarking, technical reports
- **Production** — practical drop-in CUDA sort with good API, well-tested
- **Learning** — clear step-by-step optimization progression with documented speedups

Both Quicksort and Radix Sort are implemented for performance comparison.

## Requirements

- **Generic API** via C++ templates, minimum support for `int32`, `int64`, `float32`, `float64`
- **Four host APIs**: struct-based (custom comparator, unstable), struct-based stable, key-value separate-array (unstable), key-value stable
- **Full GPU parallelization** as baseline, iterative step-by-step optimization
- **CMake** build system, Google Test for tests
- **CPU comparison** against blqsort (cloned separately on GPU servers at runtime)
- **Target architecture** auto-detected from server CUDA toolkit version
- **Profiler reports** at each optimization step: Nsight Compute (`ncu`) deep analysis + CUDA event timings
- **Corner case unit tests** validated on two GPU servers
- **Two GPU servers**: Thor (`ssh shuyua01@10.190.0.91`, `PROJECT_ROOT=/home/shuyua01/Development/`) and A40 (`ssh shuyua01@szc-td04`, `PROJECT_ROOT=/project/ai/npu_sw/shuyua01/Development`)

## Architecture

### Three-layer design

1. **Kernel layer** (`src/kernels/`) — raw `__global__` CUDA kernels
   - `partition_kernel` — grid-level element redistribution via atomic counters
   - `block_sort_kernel` — shared-memory quicksort per SM chunk
   - `radix_sort_kernel` — parallel radix sort (comparison baseline)
   - `merge_kernel` — post-partition merge step

2. **API layer** (`src/api/`) — host-side callable interfaces
   - `blqs::sort<T>(T* d_data, int n)` — struct-based, custom comparator (unstable)
   - `blqs::sort_stable<T>(T* d_data, int n)` — stable struct-based sort
   - `blqs::sort_by_key<K,V>(K* keys, V* values, int n)` — separate arrays (unstable)
   - `blqs::sort_by_key_stable<K,V>(K* keys, V* values, int n)` — stable separate-array sort

3. **Benchmark layer** (`src/bench/`) — performance comparison
   - `cuda_bench` — runs cuda-blqsort on GPU
   - `cpu_bench` — runs blqsort on CPU (cloned from GitHub at runtime)
   - Generates timing tables, speedup ratios, and profiler reports

## Kernel Internals

### Partition Strategy
- Grid-wide bucket-based partition
- First pass: count elements per pivot bucket using atomic counters
- Second pass: scatter to destination
- Pivot selection: median-of-three sampled across grid

### Recursion Model
- Handled on the host
- After each kernel launch, host reads back bucket boundaries and launches sub-kernels for non-base-case chunks
- Compatible with CUDA streams for overlapping execution

### Base Case
- ≤64 elements → sorting network in registers
- Shared-memory execution, no recursion overhead

### Stable Sort
- Numeric types: stable radix sort internally (counting sort per digit)
- Non-numeric types (custom comparator): stable merge sort variant — stable partition via index tracking, then recursive merge. This avoids the O(n) extra space issue by using device workspace allocated per call.

### Radix Sort
- 32-bit keys: 256-bin counting sort per byte (4 passes for `uint32`, 8 for `uint64`)
- Key-value pairs sorted simultaneously

### Memory Model
- Device allocation per kernel call — no persistent buffers across calls
- Shared memory: per-block working set, dynamically sized based on `n / num_blocks`
- Host pinned memory for boundary arrays and small result copies

## Data Flow

```
Host API call (e.g. blqs::sort_by_key)
  → Validate input (nullptr check, size check)
  → Compute grid/block dimensions based on N
  → Allocate device workspace (bucket counters, boundaries)
  → [CUDA events start]
  → Kernel launch: partition or radix sort
  → Copy back bucket boundaries (cudaMemcpyDeviceToHost)
  → If chunks need recursion → relaunch per-chunk kernels
  → [CUDA events end, sync]
  → Free device workspace
  → Return timing stats
```

## Error Handling

- **CUDA errors**: `CUDA_CHECK` macro — calls `cudaGetLastError()` + `cudaPeekAtLastError()`, throws `std::runtime_error` with file:line and error string
- **Invalid input**: host-side validation before any device call, throws `std::invalid_argument`
- **OOM**: check `cudaMalloc` return code, throw descriptive `std::runtime_error`
- **Server connectivity**: SSH test scripts check `nvidia-smi` before running, fail fast

## Optimization Roadmap

| Step | Description | Expected vs Previous |
|------|-------------|---------------------|
| 0    | Baseline: naive single-block quicksort | Reference point |
| 1    | Multi-block partition with atomic counters | 2-5x step 0 |
| 2    | Shared-memory optimization + sorting network base case | 1.5-2x step 1 |
| 3    | Warp-level primitives for partition | 1.3-1.5x step 2 |
| 4    | Multi-stream overlapping for multiple chunks | 1.2-1.5x step 3 |
| 5    | Radix sort comparison baseline | Dominates on numeric types |
| 6    | Final CPU vs GPU comparison (blqsort vs cuda-blqsort) | Report speedup ratio |

Each step has its own profiler report with Nsight Compute metrics (SM occupancy, memory bandwidth, warp efficiency) + CUDA event timings.

## Testing Strategy

### Unit Tests (Google Test)

- **Correctness**: Sort arrays of each supported type and verify sorted output
- **Key-Value pairing**: Verify key-value pairs stay aligned after sort
- **Corner cases**:
  - Empty array (n=0), single element (n=1)
  - Already sorted / reverse sorted
  - All elements equal
  - Two distinct values
  - Large array (10M+ elements)
  - Duplicate-heavy arrays
  - NaN/Inf for floating point
  - Negative numbers for signed types
- **Stability**: For stable variants, verify equal elements maintain original relative order
- **Struct-based**: Define a test struct with custom `operator<`, verify correctness
- **CPU comparison**: Run same inputs through both CPU blqsort and GPU, verify identical output
- **Reproducibility**: Same input → same output across multiple runs

### Benchmark Tests

- Array sizes: 1K, 10K, 100K, 1M, 10M, 100M elements
- Both GPU servers (Thor + A40)
- Generate CSV + markdown performance tables

## CMake Configuration

- `CMAKE_CUDA_ARCHITECTURES = native` for auto-detection
- Tests linked against gtest (fetched via `FetchContent`)
- `CTest` integration for `ctest` runner
- `profiler` target generates Nsight Compute reports + CUDA event timings

## Repository Structure

```
cuda-blqsort/
├── CMakeLists.txt
├── src/
│   ├── kernels/          # CUDA kernels (.cu files)
│   │   ├── partition.cu
│   │   ├── block_sort.cu
│   │   ├── radix_sort.cu
│   │   └── merge.cu
│   ├── api/              # Host API (.hpp + .cu)
│   │   ├── blqs.hpp
│   │   ├── blqs_keyvalue.hpp
│   │   └── blqs_stable.hpp
│   └── bench/            # Benchmark tools
│       ├── cuda_bench.cpp
│       └── cpu_bench.cpp
├── tests/
│   ├── correctness_test.cpp
│   ├── keyvalue_test.cpp
│   ├── stability_test.cpp
│   ├── corner_case_test.cpp
│   └── reproducibility_test.cpp
├── scripts/
│   ├── deploy.sh         # SSH deploy to GPU servers
│   ├── run_tests.sh      # Run ctest on remote servers
│   └── profile.sh        # Nsight Compute profiling
└── docs/
    ├── reports/          # Optimization step profiler reports
    └── superpowers/specs/
```
