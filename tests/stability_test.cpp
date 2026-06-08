// Stability verification tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

namespace {

struct IndexedValue {
    int value;
    int original_pos;

    __host__ __device__ bool operator<(const IndexedValue& other) const {
        return value < other.value;
    }
};

template <typename T>
void TestStableSort(const std::vector<T>& input) {
    int n = static_cast<int>(input.size());
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    blqs::sort_stable(d_data, n, [] __device__ (const T& a, const T& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_data(n);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_data[i].value, h_data[i + 1].value) << "Not sorted at index " << i;
    }

    for (int i = 0; i < n - 1; i++) {
        if (h_data[i].value == h_data[i + 1].value) {
            EXPECT_LT(h_data[i].original_pos, h_data[i + 1].original_pos)
                << "Stability violated at index " << i
                << " (pos " << h_data[i].original_pos << " vs " << h_data[i + 1].original_pos << ")";
        }
    }
}

} // namespace

TEST(StabilityTest, AllEqual) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 100; i++) {
        data.push_back({42, i});
    }
    TestStableSort(data);
}

TEST(StabilityTest, DuplicateHeavy) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 200; i++) {
        data.push_back({i % 10, i});
    }
    TestStableSort(data);
}

TEST(StabilityTest, TwoValues) {
    std::vector<IndexedValue> data;
    for (int i = 0; i < 100; i++) {
        data.push_back({i % 2, i});
    }
    TestStableSort(data);
}

TEST(StabilityTest, StableKeyValueInt) {
    int n = 500;
    std::vector<int> keys(n), values(n);
    std::mt19937 gen(77);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int i = 0; i < n; i++) {
        keys[i] = dist(gen);
        values[i] = i;
    }

    int *d_keys = nullptr, *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    blqs::sort_by_key_stable(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_keys(n), h_values(n);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));

    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_keys[i], h_keys[i + 1]);
    }

    for (int i = 0; i < n - 1; i++) {
        if (h_keys[i] == h_keys[i + 1]) {
            EXPECT_LT(h_values[i], h_values[i + 1])
                << "Stability violated: key " << h_keys[i]
                << " pos " << h_values[i] << " vs " << h_values[i + 1];
        }
    }
}

TEST(StabilityTest, EmptyArray) {
    EXPECT_NO_THROW(blqs::sort_stable((int*)nullptr, 0, [] __device__ (const int&, const int&) { return false; }));
}

TEST(StabilityTest, SingleElement) {
    IndexedValue data = {{42, 0}};
    IndexedValue* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(IndexedValue)));
    CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(IndexedValue), cudaMemcpyHostToDevice));

    blqs::sort_stable(d_data, 1, [] __device__ (const IndexedValue& a, const IndexedValue& b) { return a < b; });
    CUDA_CHECK(cudaDeviceSynchronize());

    IndexedValue result;
    CUDA_CHECK(cudaMemcpy(&result, d_data, sizeof(IndexedValue), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result.value, 42);
    EXPECT_EQ(result.original_pos, 0);
    CUDA_CHECK(cudaFree(d_data));
}
