// API implementations — two-pass partition with single global pivot
#include "blqs.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include "blqs_kernels.hpp"
#include <type_traits>
#include <stdexcept>
#include <vector>

using namespace blqs::detail;

namespace blqs {

// ── sort driver: two-pass global partition ────────────────────────────────

template <typename T>
void do_sort(T* d_data, int n) {
    if (n <= 1) return;

    // Small arrays: single-block shared-memory quicksort
    if (n <= BLOCK_SIZE) {
        int threads = (n < BLOCK_SIZE) ? n : BLOCK_SIZE;
        quicksort_shared_kernel<T, 1024><<<1, threads>>>(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Read pivot candidates from device
    std::vector<T> h_samples(3);
    CUDA_CHECK(cudaMemcpy(&h_samples[0], d_data, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[1], d_data + n / 2, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[2], d_data + n - 1, sizeof(T), cudaMemcpyDeviceToHost));

    // Median-of-three pivot
    T pivot;
    if (h_samples[2] < h_samples[0]) {
        if (h_samples[0] < h_samples[1]) pivot = h_samples[0];
        else if (h_samples[2] < h_samples[1]) pivot = h_samples[2];
        else pivot = h_samples[1];
    } else {
        if (h_samples[2] < h_samples[1]) pivot = h_samples[2];
        else if (h_samples[0] < h_samples[1]) pivot = h_samples[1];
        else pivot = h_samples[0];
    }

    // Pass 1: count elements less than pivot
    int h_less_count = 0;
    int* d_less_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_less_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_less_count, 0, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_less_kernel<T><<<blocks, threads>>>(d_data, n, pivot, d_less_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_less_count, d_less_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_less_count));

    // Handle degenerate cases (all elements on one side of pivot)
    if (h_less_count == 0 || h_less_count == n) {
        // Try again with a different pivot: use the element at index n/4
        std::vector<T> h_alt(1);
        CUDA_CHECK(cudaMemcpy(&h_alt[0], d_data + n / 4, sizeof(T), cudaMemcpyDeviceToHost));
        T pivot2 = h_alt[0];

        int* d_lc2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lc2, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_lc2, 0, sizeof(int)));
        count_less_kernel<T><<<blocks, threads>>>(d_data, n, pivot2, d_lc2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_less_count, d_lc2, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_lc2));

        if (h_less_count > 0 && h_less_count < n) {
            // Second pivot worked, proceed with scatter
            int* d_lw = nullptr, *d_gw = nullptr;
            CUDA_CHECK(cudaMalloc(&d_lw, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_gw, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_lw, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_gw, 0, sizeof(int)));
            partition_scatter_kernel<T><<<blocks, threads>>>(
                d_data, n, pivot2, h_less_count, d_lw, d_gw);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_lw));
            CUDA_CHECK(cudaFree(d_gw));
            do_sort(d_data, h_less_count);
            do_sort(d_data + h_less_count, n - h_less_count);
            return;
        }

        // Still degenerate: use radix sort as fallback (stable, always works)
        radix_sort_driver<T, void>(d_data, nullptr, n);
        return;
    }

    // Pass 2: scatter elements
    int* d_less_written = nullptr;
    int* d_ge_written = nullptr;
    CUDA_CHECK(cudaMalloc(&d_less_written, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ge_written, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_less_written, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ge_written, 0, sizeof(int)));

    partition_scatter_kernel<T><<<blocks, threads>>>(
        d_data, n, pivot, h_less_count, d_less_written, d_ge_written);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_less_written));
    CUDA_CHECK(cudaFree(d_ge_written));

    // Recursively sort both halves
    do_sort(d_data, h_less_count);
    do_sort(d_data + h_less_count, n - h_less_count);
}

// ── KV sort driver ────────────────────────────────────────────────────────

template <typename K, typename V>
void do_sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 1) return;

    if (n <= BLOCK_SIZE) {
        int threads = (n < BLOCK_SIZE) ? n : BLOCK_SIZE;
        kv_block_sort_kernel<K, V><<<1, threads>>>(d_keys, d_values, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    std::vector<K> h_samples(3);
    CUDA_CHECK(cudaMemcpy(&h_samples[0], d_keys, sizeof(K), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[1], d_keys + n / 2, sizeof(K), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[2], d_keys + n - 1, sizeof(K), cudaMemcpyDeviceToHost));

    K pivot;
    if (h_samples[2] < h_samples[0]) {
        if (h_samples[0] < h_samples[1]) pivot = h_samples[0];
        else if (h_samples[2] < h_samples[1]) pivot = h_samples[2];
        else pivot = h_samples[1];
    } else {
        if (h_samples[2] < h_samples[1]) pivot = h_samples[2];
        else if (h_samples[0] < h_samples[1]) pivot = h_samples[1];
        else pivot = h_samples[0];
    }

    int h_less_count = 0;
    int* d_less_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_less_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_less_count, 0, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kv_count_less_kernel<K, V><<<blocks, threads>>>(d_keys, n, pivot, d_less_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_less_count, d_less_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_less_count));

    if (h_less_count == 0 || h_less_count == n) {
        // Try alternate pivot
        std::vector<K> h_alt(1);
        CUDA_CHECK(cudaMemcpy(&h_alt[0], d_keys + n / 4, sizeof(K), cudaMemcpyDeviceToHost));
        K pivot2 = h_alt[0];

        int* d_lc2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lc2, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_lc2, 0, sizeof(int)));
        kv_count_less_kernel<K, V><<<blocks, threads>>>(d_keys, n, pivot2, d_lc2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_less_count, d_lc2, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_lc2));

        if (h_less_count > 0 && h_less_count < n) {
            int* d_lw = nullptr, *d_gw = nullptr;
            CUDA_CHECK(cudaMalloc(&d_lw, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_gw, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_lw, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_gw, 0, sizeof(int)));
            kv_partition_scatter_kernel<K, V><<<blocks, threads>>>(
                d_keys, d_values, n, pivot2, h_less_count, d_lw, d_gw);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_lw));
            CUDA_CHECK(cudaFree(d_gw));
            do_sort_by_key(d_keys, d_values, h_less_count);
            do_sort_by_key(d_keys + h_less_count, d_values + h_less_count, n - h_less_count);
            return;
        }

        radix_sort_driver<K, V>(d_keys, d_values, n);
        return;
    }

    int* d_less_written = nullptr;
    int* d_ge_written = nullptr;
    CUDA_CHECK(cudaMalloc(&d_less_written, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ge_written, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_less_written, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ge_written, 0, sizeof(int)));

    kv_partition_scatter_kernel<K, V><<<blocks, threads>>>(
        d_keys, d_values, n, pivot, h_less_count, d_less_written, d_ge_written);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_less_written));
    CUDA_CHECK(cudaFree(d_ge_written));

    do_sort_by_key(d_keys, d_values, h_less_count);
    do_sort_by_key(d_keys + h_less_count, d_values + h_less_count, n - h_less_count);
}

// ── Public API ──────────────────────────────────────────────────────────────

template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator) {
    if (n <= 0) return;
    if (d_data == nullptr) throw std::invalid_argument("sort: d_data is nullptr");
    do_sort(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator) {
    if (n <= 0) return;
    if (d_data == nullptr) throw std::invalid_argument("sort_stable: d_data is nullptr");
    if constexpr (std::is_arithmetic<T>::value) {
        radix_sort_driver<T, void>(d_data, nullptr, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr)
        throw std::invalid_argument("sort_by_key: keys or values is nullptr");
    do_sort_by_key(d_keys, d_values, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n) {
    if (n <= 0) return;
    if (d_keys == nullptr || d_values == nullptr)
        throw std::invalid_argument("sort_by_key_stable: keys or values is nullptr");
    if constexpr (std::is_arithmetic<K>::value) {
        radix_sort_driver<K, V>(d_keys, d_values, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Explicit instantiations ───────────────────────────────────────────────

template void sort<int,       int (*)(const int&,    const int&)>(int*, int, int (*)(const int&, const int&));
template void sort<float,     int (*)(const float&,  const float&)>(float*, int, int (*)(const float&, const float&));
template void sort<int64_t,   int (*)(const int64_t&,const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
template void sort<double,    int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));

template void sort_stable<int,       int (*)(const int&,    const int&)>(int*, int, int (*)(const int&, const int&));
template void sort_stable<float,     int (*)(const float&,  const float&)>(float*, int, int (*)(const float&, const float&));
template void sort_stable<int64_t,   int (*)(const int64_t&,const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
template void sort_stable<double,    int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));

template void sort_by_key<int, int>(int*, int*, int);
template void sort_by_key<float, int>(float*, int*, int);
template void sort_by_key<float, float>(float*, float*, int);
template void sort_by_key<int64_t, int64_t>(int64_t*, int64_t*, int);
template void sort_by_key<double, double>(double*, double*, int);

template void sort_by_key_stable<int, int>(int*, int*, int);
template void sort_by_key_stable<float, int>(float*, int*, int);
template void sort_by_key_stable<int64_t, int64_t>(int64_t*, int64_t*, int);
template void sort_by_key_stable<double, double>(double*, double*, int);

} // namespace blqs
