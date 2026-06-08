// CPU blqsort benchmark runner
// Compares CPU blqsort and std::sort performance
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <random>

// Include blqsort from cloned repo (path set via BLQS_INCLUDE_PATH)
#ifndef BLQS_INCLUDE_PATH
#define BLQS_INCLUDE_PATH "../../blqsort"
#endif

#include BLQS_INCLUDE_PATH "/blqs.h"

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
