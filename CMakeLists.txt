cmake_minimum_required(VERSION 3.24)
project(cuda-blqsort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture
set(CMAKE_CUDA_ARCHITECTURES "native")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Find CUDA toolkit (support non-standard paths)
find_package(CUDAToolkit REQUIRED)

# Set explicit CUDA compiler if auto-detection fails
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Support CUDA 13.0 which may need newer cmake policies
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# ── Google Test via FetchContent ──
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ── Library: cuda-blqsort ──
add_library(cuda_blqsort
    src/kernels/partition.cu
    src/kernels/block_sort.cu
    src/kernels/radix_sort.cu
    src/kernels/merge.cu
    src/api/blqs_api.cu
)
# Enable separate compilation for relocatable device code (kernel template linking)
set_target_properties(cuda_blqsort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_blqsort
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/api
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels
)
target_link_libraries(cuda_blqsort
    PUBLIC
        CUDA::cudart
)

# ── Tests ──
set(TEST_SOURCES
    tests/correctness_test.cpp
    tests/keyvalue_test.cpp
    tests/stability_test.cpp
    tests/corner_case_test.cpp
    tests/reproducibility_test.cpp
)

add_executable(blqs_tests ${TEST_SOURCES})
target_link_libraries(blqs_tests
    PRIVATE
        cuda_blqsort
        GTest::gtest_main
        GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(blqs_tests)

add_test(NAME blqs_tests COMMAND blqs_tests)

# ── Benchmarks ──
add_executable(cuda_bench src/bench/cuda_bench.cu)
target_link_libraries(cuda_bench PRIVATE cuda_blqsort CUDA::cudart)

add_executable(cpu_bench src/bench/cpu_bench.cpp)
target_compile_definitions(cpu_bench PRIVATE BLQS_INCLUDE_PATH="${PROJECT_SOURCE_DIR}/../blqsort/cpp")
target_include_directories(cpu_bench PRIVATE "${PROJECT_SOURCE_DIR}/../blqsort/cpp")
target_link_libraries(cpu_bench PRIVATE CUDA::cudart)

# ── Profiler target ──
add_custom_target(profiler
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/profile.sh $<TARGET_FILE:cuda_bench>
    DEPENDS cuda_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Nsight Compute profiler on cuda_bench"
)
