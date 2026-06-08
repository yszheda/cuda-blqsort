// Correctness tests for cuda-blqsort
// Tests all supported types and corner cases for blqs::sort()
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
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

    blqs::sort(d_data, static_cast<int>(input.size()), AscendingComparator<T>());
    CUDA_CHECK(cudaDeviceSynchronize());

    HostToTest(d_data, static_cast<int>(input.size()));
    CUDA_CHECK(cudaFree(d_data));
}

} // namespace

TEST(CorrectnessTest, EmptyArray) {
    std::vector<int> data;
    EXPECT_NO_THROW(TestSort(data));
}

TEST(CorrectnessTest, SingleElement) {
    std::vector<int> data = {42};
    TestSort(data);
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

TEST(CorrectnessTest, Int64Basic) {
    std::vector<int64_t> data = {3, 1, 4, 1, 5, 9, 2, 6};
    TestSort(data);
}

TEST(CorrectnessTest, DoubleBasic) {
    std::vector<double> data = {3.14159, 1.41421, 2.71828, 0.0, -1.0};
    TestSort(data);
}
