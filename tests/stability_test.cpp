// Stability verification tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

// Test stability by using KV sort: values track original position
// After stable sort, equal keys should have values in ascending order

TEST(StabilityTest, StableKVIntAllEqual) {
    int n = 100;
    std::vector<int> keys(n, 42);
    std::vector<int> values(n);
    for (int i = 0; i < n; i++) values[i] = i;

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
        EXPECT_LT(h_values[i], h_values[i + 1])
            << "Stability violated at index " << i;
    }
}

TEST(StabilityTest, StableKVDuplicateHeavy) {
    int n = 200;
    std::vector<int> keys(n);
    std::vector<int> values(n);
    for (int i = 0; i < n; i++) { keys[i] = i % 10; values[i] = i; }

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
        if (h_keys[i] == h_keys[i + 1]) {
            EXPECT_LT(h_values[i], h_values[i + 1])
                << "Stability violated: key " << h_keys[i]
                << " pos " << h_values[i] << " vs " << h_values[i + 1];
        }
    }
}

TEST(StabilityTest, StableKVFloat) {
    int n = 100;
    std::vector<float> keys(n);
    std::vector<int> values(n);
    std::mt19937 gen(77);
    std::uniform_int_distribution<int> dist(0, 5);
    for (int i = 0; i < n; i++) { keys[i] = static_cast<float>(dist(gen)); values[i] = i; }

    float *d_keys = nullptr;
    int *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    blqs::sort_by_key_stable(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_keys(n);
    std::vector<int> h_values(n);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));

    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_keys[i], h_keys[i + 1]);
        if (h_keys[i] == h_keys[i + 1]) {
            EXPECT_LT(h_values[i], h_values[i + 1]);
        }
    }
}

TEST(StabilityTest, EmptyArray) {
    EXPECT_NO_THROW(blqs::sort_by_key_stable((int*)nullptr, (int*)nullptr, 0));
}

TEST(StabilityTest, SingleElement) {
    int key = 42, value = 0;
    int *d_key = nullptr, *d_value = nullptr;
    CUDA_CHECK(cudaMalloc(&d_key, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_value, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_key, &key, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, &value, sizeof(int), cudaMemcpyHostToDevice));

    blqs::sort_by_key_stable(d_key, d_value, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    int r_key, r_value;
    CUDA_CHECK(cudaMemcpy(&r_key, d_key, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&r_value, d_value, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(r_key, 42);
    EXPECT_EQ(r_value, 0);
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
}
