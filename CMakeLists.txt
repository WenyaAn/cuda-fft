cmake_minimum_required(VERSION 3.22)
project(cuda_fft_256 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(fft256_benchmark
    benchmark.cu
    fft256.cu
)
target_compile_features(fft256_benchmark PRIVATE cxx_std_17)
set_target_properties(fft256_benchmark PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(fft256_benchmark PRIVATE CUDA::cudart CUDA::cufft)

enable_testing()
add_test(NAME fft256_correctness_and_benchmark COMMAND fft256_benchmark)
