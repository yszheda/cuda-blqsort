// Benchmark utility header
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
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
