// Reproducibility/determinism tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <random>

int cmp_int(const int& a, const int& b) { return a < b; }

int cmp_float(const float& a, const float& b) { return a < b; }

namespace {

void TestReproducibility(const std::vector<int>& input, int runs = 3) {
    int n = static_cast<int>(input.size());
    if (n == 0) return;
    std::vector<int> first_run;

    for (int r = 0; r < runs; r++) {
        int* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        blqs::sort(d, n, cmp_int);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        if (r == 0) {
            first_run = h;
        } else {
            for (int i = 0; i < n; i++) {
                EXPECT_EQ(first_run[i], h[i])
                    << "Non-deterministic at index " << i << " (run 0 vs run " << r << ")";
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

    int n = 500;
    std::vector<float> first_run;
    for (int r = 0; r < 3; r++) {
        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        blqs::sort(d, n, cmp_float);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        if (r == 0) first_run = h;
        else {
            for (int i = 0; i < n; i++) EXPECT_EQ(first_run[i], h[i]);
        }
    }
}

TEST(ReproducibilityTest, DeterministicLarge) {
    std::mt19937 gen(777);
    std::uniform_int_distribution<int> dist;
    std::vector<int> data(100000);
    for (auto& v : data) v = dist(gen);
    TestReproducibility(data, 2);
}
