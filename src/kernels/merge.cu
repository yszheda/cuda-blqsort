// Merge kernels for cuda-blqsort
// Task 4: Stable merge sort kernel for non-numeric types
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace blqs {
namespace detail {

template <typename T, typename Comparator>
__global__ void stable_merge_kernel(
    const T* input,
    T* output,
    const int* indices,
    int* output_indices,
    int n,
    int left_size,
    Comparator cmp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int target = tid;
    int l = 0, r = left_size;
    for (int i = 0; i <= target; i++) {
        if (l >= left_size) {
            if (i == target) {
                output[i] = input[left_size + r];
                output_indices[i] = indices[left_size + r];
            }
            r++;
        } else if (r >= n) {
            if (i == target) {
                output[i] = input[l];
                output_indices[i] = indices[l];
            }
            l++;
        } else if (!cmp(input[left_size + r], input[l])) {
            if (i == target) {
                output[i] = input[l];
                output_indices[i] = indices[l];
            }
            l++;
        } else {
            if (i == target) {
                output[i] = input[left_size + r];
                output_indices[i] = indices[left_size + r];
            }
            r++;
        }
    }
}

} // namespace detail
} // namespace blqs
