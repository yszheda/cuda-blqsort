# Profiler Reports - cuda-blqsort

## Environment
- **GPU Server**: Thor (`shuyua01@10.190.0.91`)
- **GPU**: NVIDIA Thor (compute_13.0, SM count: 20, Memory: 123 GB)
- **CUDA**: 13.0.48
- **Nsight Compute**: 2025.3.0.19

## Nsight Compute Status

The `ncu` profiler requires elevated permissions (`CAP_SYS_ADMIN`) to access GPU performance counters.
On Thor server, this permission is not available in the user account:
```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

To enable profiling, run: `sudo sh -c 'echo N >/proc/driver/nvidia/params'` or contact the server admin.

## Timing Data (CUDA Events)

### Quicksort int32

| N | Time (ms) | Notes |
|---|-----------|-------|
| 1,000 | ~387 | Kernel launch overhead dominates |
| 10,000 | ~3,700 | Host-side recursion overhead |
| 100,000 | ~4,250 | Single-thread sort bottleneck |
| 1,000,000 | ~54,111 | O(n log n) but per-thread |

### Key Observations

1. **Correctness over Performance**: The implementation prioritizes correctness (33/33 tests pass) over raw performance
2. **Single-Thread Bottleneck**: Shared-memory quicksort uses only thread 0 per block
3. **Host Recursion Overhead**: Each partition requires device-host roundtrip
4. **Radix Sort Not Stable**: Radix sort with atomicAdd doesn't preserve order; stable sort falls back to host-side std::stable_sort

## A40 Server

- **A40**: 33/33 tests PASS in 11.38s (vs Thor's 47.77s)
- **CUDA**: 13.1 on A40 (vs 13.0 on Thor)
- **Compiler**: GCC 13.3.0, NVCC 13.1

## Optimization Roadmap

The codebase is structured for iterative optimization with clear extension points:
- `include/blqs_kernels.hpp` — kernel templates (easy to swap implementations)
- `src/api/blqs_api.cu` — host-side sort drivers
- `include/blqs_bench.hpp` — benchmark utilities

Future optimization passes should focus on:
1. Multi-thread shared-memory sort (warp-level cooperation)
2. Multi-block partition with global prefix sum
3. Radix sort with stable per-thread ordering
4. Stream-based concurrent chunk processing
