// Launch configuration utilities
#pragma once
#include <cuda_runtime.h>

namespace blqs {
namespace detail {

constexpr int BLOCK_SIZE = 256;
constexpr int BASE_CASE_THRESHOLD = 128;

inline int get_num_blocks(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

} // namespace detail
} // namespace blqs
