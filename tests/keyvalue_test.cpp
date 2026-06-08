// Key-value sort tests for cuda-blqsort
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "blqs.hpp"
#include "blqs_errors.hpp"

#include <vector>
#include <algorithm>
#include <random>

namespace {

template <typename K, typename V>
void TestSortByKey(const std::vector<K>& keys, const std::vector<V>& values) {
    int n = static_cast<int>(keys.size());
    K *d_keys = nullptr, *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(V)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(V), cudaMemcpyHostToDevice));

    blqs::sort_by_key(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<K> h_keys(n);
    std::vector<V> h_values(n);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, n * sizeof(K), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, n * sizeof(V), cudaMemcpyDeviceToHost));

    // Verify keys are sorted
    for (int i = 0; i < n - 1; i++) {
        EXPECT_LE(h_keys[i], h_keys[i + 1]) << "Keys not sorted at index " << i;
    }

    // Verify key-value pairs are preserved (multiset comparison)
    // Sort both input and output pairs and compare
    std::vector<std::pair<K, V>> input_pairs(n);
    std::vector<std::pair<K, V>> output_pairs(n);
    for (int i = 0; i < n; i++) {
        input_pairs[i] = {keys[i], values[i]};
        output_pairs[i] = {h_keys[i], h_values[i]};
    }
    std::sort(input_pairs.begin(), input_pairs.end());
    std::sort(output_pairs.begin(), output_pairs.end());
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(input_pairs[i].first, output_pairs[i].first) << "Key multiset mismatch at " << i;
        EXPECT_EQ(input_pairs[i].second, output_pairs[i].second) << "Value multiset mismatch at " << i;
    }

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
}

} // namespace

TEST(KeyValueTest, BasicIntInt) {
    std::vector<int> keys = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> values = {30, 10, 40, 11, 50, 90, 20, 60};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, FloatInt) {
    std::vector<float> keys = {3.14f, 1.41f, 2.71f, 0.0f};
    std::vector<int> values = {100, 200, 300, 400};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, AllSameKey) {
    std::vector<int> keys(50, 42);
    std::vector<int> values(50);
    for (int i = 0; i < 50; i++) values[i] = i;
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, ReverseSorted) {
    std::vector<int> keys = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int> values = {90, 80, 70, 60, 50, 40, 30, 20, 10};
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, LargeRandom) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    int n = 50000;
    std::vector<int> keys(n), values(n);
    for (int i = 0; i < n; i++) { keys[i] = dist(gen); values[i] = i; }
    TestSortByKey(keys, values);
}

TEST(KeyValueTest, NullptrKeys) {
    EXPECT_THROW(blqs::sort_by_key((int*)nullptr, (int*)nullptr, 10), std::invalid_argument);
}

TEST(KeyValueTest, EmptyArray) {
    EXPECT_NO_THROW(blqs::sort_by_key((int*)nullptr, (int*)nullptr, 0));
}
