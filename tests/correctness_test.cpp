// Correctness tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

// Plain function pointer comparators (matching explicit instantiations)
int cmp_int(const int& a, const int& b) { return a < b; }
int cmp_float(const float& a, const float& b) { return a < b; }
int cmp_int64(const int64_t& a, const int64_t& b) { return a < b; }
int cmp_double(const double& a, const double& b) { return a < b; }

namespace {

template <typename T>
void HostVerifySorted(T* d_data, int n) {
    std::vector<T> h_data(n);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_data[i], h_data[i + 1]) << "Not sorted at index " << i;
    }
}

void TestSortInt(const std::vector<int>& input) {
    if (input.empty()) return;
    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, input.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));
    blqs::sort(d, static_cast<int>(input.size()), cmp_int);
    HostVerifySorted(d, static_cast<int>(input.size()));
    CUDA_CHECK(cudaFree(d));
}

void TestSortFloat(const std::vector<float>& input) {
    if (input.empty()) return;
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    blqs::sort(d, static_cast<int>(input.size()), cmp_float);
    HostVerifySorted(d, static_cast<int>(input.size()));
    CUDA_CHECK(cudaFree(d));
}

void TestSortInt64(const std::vector<int64_t>& input) {
    if (input.empty()) return;
    int64_t* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, input.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d, input.data(), input.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    blqs::sort(d, static_cast<int>(input.size()), cmp_int64);
    HostVerifySorted(d, static_cast<int>(input.size()));
    CUDA_CHECK(cudaFree(d));
}

void TestSortDouble(const std::vector<double>& input) {
    if (input.empty()) return;
    double* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, input.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice));
    blqs::sort(d, static_cast<int>(input.size()), cmp_double);
    HostVerifySorted(d, static_cast<int>(input.size()));
    CUDA_CHECK(cudaFree(d));
}

} // namespace

TEST(CorrectnessTest, EmptyArray) {
    std::vector<int> data;
    TestSortInt(data);
}

TEST(CorrectnessTest, SingleElement) {
    std::vector<int> data = {42};
    TestSortInt(data);
}

TEST(CorrectnessTest, TwoElements) {
    std::vector<int> data = {2, 1};
    TestSortInt(data);
}

TEST(CorrectnessTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    TestSortInt(data);
}

TEST(CorrectnessTest, ReverseSorted) {
    std::vector<int> data = {5, 4, 3, 2, 1};
    TestSortInt(data);
}

TEST(CorrectnessTest, AllEqual) {
    std::vector<int> data(100, 42);
    TestSortInt(data);
}

TEST(CorrectnessTest, RandomSmall) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
    TestSortInt(data);
}

TEST(CorrectnessTest, RandomLarge) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> data(10000);
    for (auto& v : data) v = dist(gen);
    TestSortInt(data);
}

TEST(CorrectnessTest, FloatBasic) {
    std::vector<float> data = {3.14f, 1.41f, 2.71f, 0.0f, -1.0f};
    TestSortFloat(data);
}

TEST(CorrectnessTest, NegativeIntegers) {
    std::vector<int> data = {-5, -1, -10, -3, -7, -2};
    TestSortInt(data);
}

TEST(CorrectnessTest, Int64Basic) {
    std::vector<int64_t> data = {3, 1, 4, 1, 5, 9, 2, 6};
    TestSortInt64(data);
}

TEST(CorrectnessTest, DoubleBasic) {
    std::vector<double> data = {3.14159, 1.41421, 2.71828, 0.0, -1.0};
    TestSortDouble(data);
}
