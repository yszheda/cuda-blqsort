// Public API — header-only when compiled by nvcc, extern declarations for .cpp
#pragma once
#include "blqs_types.hpp"
#include <stdexcept>
#include <type_traits>

namespace blqs {

template <typename T, typename Comparator>
void sort(T* d_data, int n, Comparator cmp = Comparator());

template <typename T, typename Comparator>
void sort_stable(T* d_data, int n, Comparator cmp = Comparator());

template <typename K, typename V>
void sort_by_key(K* d_keys, V* d_values, int n);

template <typename K, typename V>
void sort_by_key_stable(K* d_keys, V* d_values, int n);

} // namespace blqs

// For .cpp files: extern template declarations so the compiler knows these exist.
// The actual definitions come from blqs_api.cu (compiled by nvcc).
// These only apply when NOT using nvcc.
#ifndef __CUDACC__
extern template void blqs::sort<int,       int (*)(const int&,    const int&)>(int*, int, int (*)(const int&, const int&));
extern template void blqs::sort<float,     int (*)(const float&,  const float&)>(float*, int, int (*)(const float&, const float&));
extern template void blqs::sort<int64_t,   int (*)(const int64_t&,const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
extern template void blqs::sort<double,    int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));
extern template void blqs::sort_stable<int,       int (*)(const int&,    const int&)>(int*, int, int (*)(const int&, const int&));
extern template void blqs::sort_stable<float,     int (*)(const float&,  const float&)>(float*, int, int (*)(const float&, const float&));
extern template void blqs::sort_stable<int64_t,   int (*)(const int64_t&,const int64_t&)>(int64_t*, int, int (*)(const int64_t&, const int64_t&));
extern template void blqs::sort_stable<double,    int (*)(const double&, const double&)>(double*, int, int (*)(const double&, const double&));
extern template void blqs::sort_by_key<int, int>(int*, int*, int);
extern template void blqs::sort_by_key<float, int>(float*, int*, int);
extern template void blqs::sort_by_key<int64_t, int64_t>(int64_t*, int64_t*, int);
extern template void blqs::sort_by_key<double, double>(double*, double*, int);
extern template void blqs::sort_by_key_stable<int, int>(int*, int*, int);
extern template void blqs::sort_by_key_stable<float, int>(float*, int*, int);
extern template void blqs::sort_by_key_stable<int64_t, int64_t>(int64_t*, int64_t*, int);
extern template void blqs::sort_by_key_stable<double, double>(double*, double*, int);
#endif
