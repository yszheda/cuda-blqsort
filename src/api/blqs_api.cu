// API implementations — two-pass partition with single global pivot, double buffering
#include "blqs.hpp"
#include "blqs_errors.hpp"
#include "blqs_config.hpp"
#include "blqs_kernels.hpp"
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <vector>

using namespace blqs::detail;

namespace blqs {

// ── sort driver: two-pass global partition with temp buffer ───────────────

template <typename T>
void do_sort_impl(T* d_data, T* d_buf, int n) {
    if (n <= 1) return;

    // Small arrays: single-block shared-memory quicksort
    if (n <= BLOCK_SIZE) {
        int threads = (n < BLOCK_SIZE) ? n : BLOCK_SIZE;
        size_t smem = 2 * 256 * sizeof(T);  // sdata[256] + temp[256]
        quicksort_shared_kernel<T, 256><<<1, threads, smem>>>(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Read pivot candidates
    std::vector<T> h_samples(3);
    CUDA_CHECK(cudaMemcpy(&h_samples[0], d_data, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[1], d_data + n / 2, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_samples[2], d_data + n - 1, sizeof(T), cudaMemcpyDeviceToHost));

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

    // Handle degenerate: try alternate pivot, fallback to radix
    if (h_less_count == 0 || h_less_count == n) {
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
            int* d_lw = nullptr, *d_gw = nullptr;
            CUDA_CHECK(cudaMalloc(&d_lw, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_gw, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_lw, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_gw, 0, sizeof(int)));
            partition_scatter_kernel<T><<<blocks, threads>>>(
                d_data, d_buf, n, pivot2, h_less_count, d_lw, d_gw);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Copy result back to d_data
            CUDA_CHECK(cudaMemcpy(d_data, d_buf, n * sizeof(T), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_gw));
            do_sort_impl(d_data, d_buf, h_less_count);
            do_sort_impl(d_data + h_less_count, d_buf + h_less_count, n - h_less_count);
            return;
        }
        // Still degenerate: copy to host, sort, copy back
        std::vector<T> h_data(n);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
        std::sort(h_data.begin(), h_data.end());
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        return;
    }

    // Pass 2: scatter to temp buffer (avoid in-place read/write race)
    int* d_lw = nullptr, *d_gw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lw, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gw, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lw, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_gw, 0, sizeof(int)));

    partition_scatter_kernel<T><<<blocks, threads>>>(
        d_data, d_buf, n, pivot, h_less_count, d_lw, d_gw);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy partitioned result back
    CUDA_CHECK(cudaMemcpy(d_data, d_buf, n * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_gw));

    // Recursively sort both halves
    do_sort_impl(d_data, d_buf, h_less_count);
    do_sort_impl(d_data + h_less_count, d_buf + h_less_count, n - h_less_count);
}

template <typename T>
void do_sort(T* d_data, int n) {
    if (n <= 1) return;
    T* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(T)));
    do_sort_impl(d_data, d_buf, n);
    CUDA_CHECK(cudaFree(d_buf));
}

// ── KV sort driver ────────────────────────────────────────────────────────

template <typename K, typename V>
void do_sort_by_key_impl(K* d_keys, V* d_values, K* d_kbuf, V* d_vbuf, int n) {
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
                d_keys, d_values, d_kbuf, d_vbuf, n, pivot2, h_less_count, d_lw, d_gw);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(d_keys, d_kbuf, n * sizeof(K), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_values, d_vbuf, n * sizeof(V), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_gw));
            do_sort_by_key_impl(d_keys, d_values, d_kbuf, d_vbuf, h_less_count);
            do_sort_by_key_impl(d_keys + h_less_count, d_values + h_less_count, d_kbuf + h_less_count, d_vbuf + h_less_count, n - h_less_count);
            return;
        }
        // Still degenerate: copy to host, sort, copy back
        std::vector<K> h_k(n);
        std::vector<V> h_v(n);
        CUDA_CHECK(cudaMemcpy(h_k.data(), d_keys, n * sizeof(K), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_values, n * sizeof(V), cudaMemcpyDeviceToHost));
        std::vector<std::pair<K, V>> pairs(n);
        for (int i = 0; i < n; i++) pairs[i] = {h_k[i], h_v[i]};
        std::sort(pairs.begin(), pairs.end());
        for (int i = 0; i < n; i++) { h_k[i] = pairs[i].first; h_v[i] = pairs[i].second; }
        CUDA_CHECK(cudaMemcpy(d_keys, h_k.data(), n * sizeof(K), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, h_v.data(), n * sizeof(V), cudaMemcpyHostToDevice));
        return;
    }

    int* d_lw = nullptr, *d_gw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lw, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gw, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lw, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_gw, 0, sizeof(int)));

    kv_partition_scatter_kernel<K, V><<<blocks, threads>>>(
        d_keys, d_values, d_kbuf, d_vbuf, n, pivot, h_less_count, d_lw, d_gw);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_keys, d_kbuf, n * sizeof(K), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, d_vbuf, n * sizeof(V), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_gw));

    do_sort_by_key_impl(d_keys, d_values, d_kbuf, d_vbuf, h_less_count);
    do_sort_by_key_impl(d_keys + h_less_count, d_values + h_less_count, d_kbuf + h_less_count, d_vbuf + h_less_count, n - h_less_count);
}

template <typename K, typename V>
void do_sort_by_key(K* d_keys, V* d_values, int n) {
    if (n <= 1) return;
    K* d_kbuf = nullptr;
    V* d_vbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kbuf, n * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vbuf, n * sizeof(V)));
    do_sort_by_key_impl(d_keys, d_values, d_kbuf, d_vbuf, n);
    CUDA_CHECK(cudaFree(d_kbuf));
    CUDA_CHECK(cudaFree(d_vbuf));
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
        // Use host-side std::stable_sort (radix sort with atomicAdd is not stable)
        std::vector<T> h_data(n);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
        std::stable_sort(h_data.begin(), h_data.end());
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
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
        // Use host-side std::sort for stability (radix sort isn't stable with atomicAdd)
        std::vector<K> h_k(n);
        std::vector<V> h_v(n);
        CUDA_CHECK(cudaMemcpy(h_k.data(), d_keys, n * sizeof(K), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_values, n * sizeof(V), cudaMemcpyDeviceToHost));
        std::vector<size_t> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&h_k](size_t a, size_t b) { return h_k[a] < h_k[b]; });
        std::vector<K> sk(n);
        std::vector<V> sv(n);
        for (int i = 0; i < n; i++) { sk[i] = h_k[idx[i]]; sv[i] = h_v[idx[i]]; }
        CUDA_CHECK(cudaMemcpy(d_keys, sk.data(), n * sizeof(K), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, sv.data(), n * sizeof(V), cudaMemcpyHostToDevice));
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
