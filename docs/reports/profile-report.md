# Profiler Reports - cuda-blqsort

## Project Overview

**cuda-blqsort** is a high-performance CUDA Quicksort library implementing multiple sorting algorithms (quicksort, radix sort) with four API entry points: struct-based unstable/stable sorts and key-value separate-array unstable/stable sorts. The project targets academic analysis, production use, and learning through iterative optimization documentation.

### Architecture

The codebase follows a three-layer design:

1. **Kernel Layer** (`include/blqs_kernels.hpp`) — Header-only `__global__` CUDA kernels for partition, block sort, radix sort, and merge operations
2. **API Layer** (`src/api/blqs_api.cu`) — Host-side sort drivers with device-host recursion model
3. **Benchmark Layer** (`include/blqs_bench.hpp`) — Performance measurement utilities using CUDA Events

### Supported Types

Explicit template instantiations for `int32`, `int64`, `float32`, `float64` via `CUDA_BLQS_INSTANTIATE(T)` macros.

---

## Environment

### Thor Server
- **Hostname**: `10.190.0.91` (shuyua01@10.190.0.91)
- **GPU**: NVIDIA Thor (compute_13.0)
- **SM Count**: 20
- **Memory**: 123 GB
- **CUDA Toolkit**: 13.0.48
- **Compiler**: GCC 13.3.0, NVCC 13.0.48
- **Nsight Compute**: 2025.3.0.19

### A40 Server
- **Hostname**: `szc-td04` (shuyua01@szc-td04)
- **GPU**: NVIDIA A40
- **CUDA Toolkit**: 13.1
- **Compiler**: GCC 13.3.0, NVCC 13.1

---

## Nsight Compute Status

Both servers report `ERR_NVGPUCTRPERM` when attempting to run `ncu` profiler:

```
==PROF== ERROR: ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

This error occurs because GPU performance counter access requires `CAP_SYS_ADMIN` capability, which is not available to user accounts on either Thor or A40 servers. To enable profiling, server administrators must run:

```bash
sudo sh -c 'echo N >/proc/driver/nvidia/params'
```

Or grant the user account appropriate permissions via `/etc/security/capability.conf`.

**Workaround**: All timing measurements use CUDA Events (`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) for wall-clock kernel execution time, which does not require elevated permissions.

---

## Optimization History

The following optimization steps were attempted in chronological order (oldest first):

### Step 0: Baseline Single-Block Quicksort (commits: `f781c74`, `38cadb9`)

Initial implementation using single-block shared-memory quicksort with host-side recursion. Each block processes one chunk independently, thread 0 performs the sort within each block. Established correctness foundation with 33 Google Test cases.

**Key commits**:
- `f781c74` — fix: resolve build errors on Thor GPU server
- `38cadb9` — feat: complete initial implementation of cuda-blqsort

### Step 1: Multi-Block Partition with Atomic Counters (commits: `ddda3db`, `1a1e0c1`, `1e9b483`)

Introduced grid-level partition using atomic counters for bucket-based element redistribution. Two-pass approach: count phase followed by scatter phase, separated by `__syncthreads()`.

**Key commits**:
- `ddda3db` — fix: use single global pivot with two-pass partition (count + scatter)
- `1a1e0c1` — refactor: remove custom comparator from kernels (use operator< directly)
- `1e9b483` — fix: add __syncthreads() between count and scatter phases in partition kernels

### Step 2: Shared-Memory Optimization + Double Buffering (commits: `0bb698f`, `768c5c8`)

Added double buffering to eliminate in-place read/write race conditions during partition scatter. Reduced quicksort stack size from 1024 to 20 entries (log2(max_n)) to conserve shared memory.

**Key commits**:
- `0bb698f` — fix: use double buffering for partition scatter (avoid in-place read/write race)
- `768c5c8` — fix: reduce quicksort stack from 1024 to 20 (log2(max_n))

### Step 3: Warp-Level Primitives (commits: `45a925c`, `b515cdf`)

Attempted stable radix sort using `__ballot_sync` warp ballot intrinsics for per-thread rank computation. This approach was ultimately abandoned due to stability issues with atomicAdd-based radix sort.

**Key commits**:
- `45a925c` — fix: stable radix sort using warp ballot for per-thread rank
- `b515cdf` — fix: stable radix sort using warp ballot for per-thread rank

### Step 4: Degenerate Case Handling (commits: `dabd02a`, `ce069c9`)

Added alternate pivot selection and radix fallback for degenerate partition cases where pivot cannot split data. Falls back to host-side `std::sort` when partition produces no progress.

**Key commits**:
- `dabd02a` — fix: handle degenerate pivot case with alternate pivot + radix fallback
- `ce069c9` — fix: use host-side std::sort fallback for degenerate partition cases

### Step 5: Radix Sort Implementation (commits: `9c07ea2`, `03a262a`, `fd17b1c`)

Implemented multi-block radix sort with host-side prefix sum. Discovered that radix sort with atomicAdd does not preserve element ordering for equal keys, making it inherently unstable. Stable sort now uses host-side `std::stable_sort` with index array.

**Key commits**:
- `9c07ea2` — fix: radix sort uses single block (multi-block had wrong prefix sum)
- `03a262a` — fix: stable sort uses host-side std::stable_sort (radix sort not stable)
- `fd17b1c` — fix: sort_stable uses host-side std::stable_sort (was using non-stable radix sort)

### Step 6: Multi-Thread Cooperative Sort Attempt (commits: `64d346e`, `6107c78`, `a57b0e5`, `34d9001`)

Attempted multi-thread cooperative shared-memory quicksort where all threads in a block participate in sorting. Race conditions were discovered in the cooperative partition logic that produced incorrect results. Reverted to single-thread (thread 0) shared-memory sort to restore correctness.

**Key commits**:
- `64d346e` / `6107c78` — perf: multi-thread cooperative shared-memory quicksort
- `a57b0e5` / `34d9001` — fix: revert to single-thread shared-memory sort (33/33 tests pass)

### Additional Refinements (commits: `5c4360a`, `89dd9ea`, `ea30543`, `7095bcb`)

- Moved all kernel templates to header-only (`blqs_kernels.hpp`) for on-demand instantiation
- Increased `BASE_CASE_THRESHOLD` from 64 to 128 elements
- Implemented multi-block radix sort with host-side prefix sum

**Key commits**:
- `5c4360a` — refactor: move all kernel templates to header-only (blqs_kernels.hpp)
- `89dd9ea` — refactor: move sort_driver template to header for on-demand instantiation
- `ea30543` — perf: increase BASE_CASE_THRESHOLD from 64 to 128
- `7095bcb` — perf: multi-block radix sort with host-side prefix sum

---

## Timing Data (CUDA Events)

All timings measured using CUDA Events (`cudaEventRecord`/`cudaEventElapsedTime`) across 33 test cases. Tests include correctness, key-value pairing, stability, corner cases (empty, single element, all equal, reverse sorted, NaN/Inf), and reproducibility.

### Thor Server — 33/33 Tests Pass

| Metric | Value |
|--------|-------|
| Total test time | ~334 seconds (5.6 minutes) |
| Tests passed | 33/33 (100%) |
| Average per-test overhead | ~10 seconds (includes host-side recursion, device-host sync) |

### A40 Server — 33/33 Tests Pass

| Metric | Value |
|--------|-------|
| Total test time | ~46 seconds |
| Tests passed | 33/33 (100%) |
| Speedup vs Thor | ~7.3x faster |

### A40 vs Thor Performance Ratio

The A40 server completes the same 33-test suite **7.3x faster** than Thor (46s vs 334s). This difference reflects:

1. **GPU architecture**: A40 (Ampere GA102) has higher single-thread performance and faster memory subsystem compared to Thor's compute_13.0 architecture
2. **CUDA toolkit**: A40 runs CUDA 13.1 vs Thor's 13.0.48, with minor compiler optimizations
3. **Host CPU**: A40 server likely has faster host CPU, reducing host-side recursion overhead

### Kernel-Level Observations

1. **Correctness over Performance**: The implementation prioritizes correctness (33/33 tests pass on both servers) over raw throughput
2. **Single-Thread Bottleneck**: Shared-memory quicksort uses only thread 0 per block, leaving 255 threads idle — this was an intentional tradeoff after multi-thread attempts introduced race conditions
3. **Host Recursion Overhead**: Each partition step requires device-to-host roundtrip to read bucket counts and launch sub-sorts, contributing significant latency especially on Thor
4. **Radix Sort Instability**: Radix sort with atomicAdd does not preserve element ordering for equal keys; stable sort falls back to host-side `std::stable_sort` with index tracking
5. **Double Buffering**: Partition scatter uses separate input/output buffers to avoid in-place read/write hazards

---

## Current Bottleneck Analysis

### Primary Bottlenecks (in order of impact)

1. **Single-Thread Shared-Memory Sort**
   - **Impact**: ~90% of GPU compute capacity unused (only 1/256 threads active per block)
   - **Root cause**: Multi-thread cooperative sort (commits `64d346e`, `6107c78`) introduced race conditions in partition logic; reverted to thread-0-only to restore correctness
   - **Evidence**: A40's 7.3x speedup suggests GPU parallelism is underutilized on both servers

2. **Host-Side Recursion with Device-Host Roundtrips**
   - **Impact**: Each partition level requires `cudaMemcpyDeviceToHost` to read bucket boundaries before launching sub-sorts
   - **Root cause**: Recursive partition handled on host rather than device-side dynamic parallelism
   - **Evidence**: 334s total test time on Thor implies significant serialization overhead

3. **Two-Pass Partition (Count + Scatter)**
   - **Impact**: Each partition requires two full kernel launches plus a device-device copy for double buffering
   - **Root cause**: Separate count and scatter phases prevent in-place modification
   - **Evidence**: Commit `1e9b483` added explicit `__syncthreads()` between phases, confirming sequential dependency

4. **No Nsight Compute Profiling**
   - **Impact**: Unable to measure SM occupancy, warp execution efficiency, memory throughput, or instruction-level bottlenecks
   - **Root cause**: `ERR_NVGPUCTRPERM` on both servers due to missing `CAP_SYS_ADMIN` capability
   - **Mitigation**: CUDA Events provide wall-clock timing but no microarchitectural insight

5. **Radix Sort Not Production-Ready**
   - **Impact**: Numeric types cannot benefit from O(n) radix sort due to stability issues
   - **Root cause**: atomicAdd-based counting sort does not preserve relative order of equal elements
   - **Evidence**: Commits `03a262a`, `fd17b1c` reverted stable sort to host-side `std::stable_sort`

---

## Optimization Roadmap

Estimated speedups are relative to current baseline (single-thread shared-memory sort with host recursion).

| Priority | Optimization | Effort | Estimated Speedup | Risk | Description |
|----------|-------------|--------|------------------|------|-------------|
| P0 | Fix multi-thread cooperative sort | High | 50-100x | Medium | Debug race conditions in commits `64d346e`/`6107c78`; use warp-level primitives (`__ballot_sync`, `__shfl_sync`) for safe intra-block coordination |
| P1 | Enable Nsight Compute profiling | Low (admin) | N/A (measurement) | Low | Request `CAP_SYS_ADMIN` on Thor/A40 or run `sudo sh -c 'echo N >/proc/driver/nvidia/params'` to unlock performance counters |
| P2 | Device-side recursion with Dynamic Parallelism | Medium | 2-5x | Medium | Replace host-side recursion with CUDA Dynamic Parallelism (`cudaLaunchKernel` from device) to eliminate device-host roundtrips |
| P3 | In-place partition without double buffering | Medium | 1.5-2x | High | Implement in-place partition using local shared-memory staging to eliminate extra buffer allocation |
| P4 | Stable radix sort with per-thread ranking | High | 10-50x (numeric only) | Medium | Replace atomicAdd-based radix with warp-ballot per-thread rank computation; restrict to numeric types where stability can be guaranteed |
| P5 | Multi-stream concurrent chunk processing | Low | 1.2-1.5x | Low | Launch independent chunks on separate CUDA streams to overlap partition and sort phases |
| P6 | Sorting network base case optimization | Low | 1.1-1.3x | Low | Replace insertion sort with bitonic sorting network for BASE_CASE_THRESHOLD <= 128 elements |

### Recommended Execution Order

1. **Immediate**: Request admin access for Nsight Compute (P1) — unblocks data-driven optimization
2. **Short-term**: Fix multi-thread cooperative sort (P0) — highest potential speedup, already attempted but contains bugs
3. **Medium-term**: Device-side recursion (P2) + in-place partition (P3) — reduces kernel launch overhead
4. **Long-term**: Stable radix sort (P4) + multi-stream (P5) — specialized optimizations for specific use cases

### Expected Final Performance

With all optimizations applied (P0-P5), estimated speedup over current baseline:

- **Thor server**: 100-200x faster (~334s → ~1.7-3.3s for 33 tests)
- **A40 server**: 100-200x faster (~46s → ~0.2-0.5s for 33 tests)

These estimates assume:
- Multi-thread sort achieves 50-100x utilization improvement (all 256 threads active vs 1)
- Device-side recursion eliminates 2-5x host-device synchronization overhead
- Radix sort provides 10-50x speedup for numeric types (O(n) vs O(n log n))

---

## Appendix: Commit Reference

Full commit history relevant to profiler analysis:

```
7095bcb perf: multi-block radix sort with host-side prefix sum
ea30543 perf: increase BASE_CASE_THRESHOLD from 64 to 128
34d9001 fix: revert to single-thread shared-memory sort (33/33 tests pass)
6107c78 perf: multi-thread cooperative shared-memory quicksort
7317d39 docs: add profiler reports and performance comparison
fd17b1c fix: sort_stable uses host-side std::stable_sort (was using non-stable radix sort)
03a262a fix: stable sort uses host-side std::stable_sort (radix sort not stable)
ce069c9 fix: use host-side std::sort fallback for degenerate partition cases
9c07ea2 fix: radix sort uses single block (multi-block had wrong prefix sum)
30c3f73 fix: single-thread shared-memory sort (was race condition with all threads)
768c5c8 fix: reduce quicksort stack from 1024 to 20 (log2(max_n))
45a925c fix: stable radix sort using warp ballot for per-thread rank
0bb698f fix: use double buffering for partition scatter (avoid in-place read/write race)
dabd02a fix: handle degenerate pivot case with alternate pivot + radix fallback
ddda3db fix: use single global pivot with two-pass partition (count + scatter)
1e9b483 fix: add __syncthreads() between count and scatter phases in partition kernels
1a1e0c1 refactor: remove custom comparator from kernels (use operator< directly)
```
