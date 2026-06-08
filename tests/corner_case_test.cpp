// Corner case tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <random>
#include <numeric>

namespace {

template <typename T>
struct AscendingComparator {
    __host__ __device__ bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

template <typename T>
void TestSort(const std::vector<T>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return;
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    blqs::sort(d_data, n, AscendingComparator<T>());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_data(n);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_data[i], h_data[i + 1]) << "Not sorted at index " << i;
    }
}

} // namespace

TEST(CornerCaseTest, TwoDistinctValues) {
    std::vector<int> data = {1, 0, 1, 0, 1, 0, 1, 0};
    TestSort(data);
}

TEST(CornerCaseTest, DuplicateHeavy) {
    std::vector<int> data;
    for (int i = 0; i < 1000; i++) {
        data.push_back(i % 5); // only 5 distinct values
    }
    TestSort(data);
}

TEST(CornerCaseTest, BlockSizeBoundary) {
    // Test around BLOCK_SIZE=256 boundary
    for (int n : {255, 256, 257, 511, 512, 513}) {
        std::vector<int> data(n);
        std::mt19937 gen(n);
        std::uniform_int_distribution<int> dist;
        for (int i = 0; i < n; i++) data[i] = dist(gen);
        TestSort(data);
    }
}

TEST(CornerCaseTest, NegativeNumbers) {
    std::vector<int> data;
    for (int i = -1000; i <= 1000; i++) data.push_back(i);
    std::shuffle(data.begin(), data.end(), std::mt19937(42));
    TestSort(data);
}

TEST(CornerCaseTest, FloatInfAndNaN) {
    std::vector<float> data = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        0.0f, 1.0f, -1.0f
    };
    // Should not crash — behavior with NaN is implementation-defined
    TestSort(data);
}

TEST(CornerCaseTest, NearlySorted) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    // Swap a few adjacent pairs
    for (int i = 0; i < 100; i += 2) {
        std::swap(data[i], data[i + 1]);
    }
    TestSort(data);
}
