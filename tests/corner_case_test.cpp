// Corner case tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <random>
#include <numeric>

int cmp_int(const int& a, const int& b) { return a < b; }

namespace {

void TestSortInt(const std::vector<int>& input) {
    if (input.empty()) return;
    int n = static_cast<int>(input.size());
    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    blqs::sort(d, n, cmp_int);
    std::vector<int> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h[i], h[i + 1]) << "Not sorted at index " << i;
    }
}

} // namespace

TEST(CornerCaseTest, TwoDistinctValues) {
    std::vector<int> data = {1, 0, 1, 0, 1, 0, 1, 0};
    TestSortInt(data);
}

TEST(CornerCaseTest, DuplicateHeavy) {
    std::vector<int> data;
    for (int i = 0; i < 1000; i++) data.push_back(i % 5);
    TestSortInt(data);
}

TEST(CornerCaseTest, BlockSizeBoundary) {
    for (int n : {255, 256, 257, 511, 512, 513}) {
        std::vector<int> data(n);
        std::mt19937 gen(n);
        std::uniform_int_distribution<int> dist;
        for (int i = 0; i < n; i++) data[i] = dist(gen);
        TestSortInt(data);
    }
}

TEST(CornerCaseTest, NegativeNumbers) {
    std::vector<int> data;
    for (int i = -1000; i <= 1000; i++) data.push_back(i);
    std::shuffle(data.begin(), data.end(), std::mt19937(42));
    TestSortInt(data);
}

TEST(CornerCaseTest, NearlySorted) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    for (int i = 0; i < 100; i += 2) std::swap(data[i], data[i + 1]);
    TestSortInt(data);
}
