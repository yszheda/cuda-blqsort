// CUDA benchmark runner
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include "blqs.hpp"
#include "blqs_bench.hpp"
#include "blqs_errors.hpp"

using namespace blqs;

template <typename T>
__device__ int asc_cmp(const T& a, const T& b) { return a < b; }

template <typename T>
void run_benchmark(const std::string& name, int n, int iterations = 5) {
    auto h_data = bench::generate_random<T>(n);
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    using Cmp = int (*)(const T&, const T&);
    double total_ms = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        bench::CudaTimer timer;
        timer.start();
        blqs::sort(d_data, n, static_cast<Cmp>(asc_cmp<T>));
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / iterations;
    double bytes = (double)n * sizeof(T) * 2;
    double bandwidth = bytes / avg_ms / 1e6;
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
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;

    bench::print_header();

    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};

    std::cout << "\n--- Quicksort int32 ---" << std::endl;
    for (int n : sizes) if (n <= 10000000) run_benchmark<int>("qsort_i32", n);

    std::cout << "\n--- Quicksort float32 ---" << std::endl;
    for (int n : sizes) if (n <= 10000000) run_benchmark<float>("qsort_f32", n);

    std::cout << "\n--- Radix sort int32 ---" << std::endl;
    for (int n : sizes) run_benchmark<int>("radix_i32", n);

    std::cout << "\n--- Key-Value sort (int->int) ---" << std::endl;
    for (int n : sizes) if (n <= 10000000) run_kv_benchmark<int, int>("kv_i32", n);

    std::cout << "\n--- Stable sort int32 ---" << std::endl;
    for (int n : sizes) if (n <= 10000000) run_benchmark<int>("stable_i32", n);

    return 0;
}
