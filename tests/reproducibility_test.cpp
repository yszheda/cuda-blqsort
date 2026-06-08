// Reproducibility/determinism tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <random>

namespace {

template <typename T>
struct AscendingComparator {
    __host__ __device__ bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

// Run sort multiple times and verify identical output
template <typename T>
void TestReproducibility(const std::vector<T>& input, int runs = 3) {
    int n = static_cast<int>(input.size());
    std::vector<T> first_run(n);

    for (int r = 0; r < runs; r++) {
        T* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_data, input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

        blqs::sort(d_data, n, AscendingComparator<T>());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> h_data(n);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data));

        if (r == 0) {
            first_run = h_data;
        } else {
            for (int i = 0; i < n; i++) {
                EXPECT_EQ(first_run[i], h_data[i])
                    << "Non-deterministic result at index " << i
                    << " (run 0 vs run " << r << ")";
            }
        }
    }
}

} // namespace

TEST(ReproducibilityTest, DeterministicInt32) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<int> data(1000);
    for (auto& v : data) v = dist(gen);
    TestReproducibility(data);
}

TEST(ReproducibilityTest, DeterministicFloat) {
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> data(500);
    for (auto& v : data) v = dist(gen);
    TestReproducibility(data);
}

TEST(ReproducibilityTest, DeterministicLarge) {
    std::mt19937 gen(777);
    std::uniform_int_distribution<int> dist;
    std::vector<int> data(100000);
    for (auto& v : data) v = dist(gen);
    TestReproducibility(data, 2);
}
