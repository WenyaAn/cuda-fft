cmake_minimum_required(VERSION 3.22)
project(cuda_fft LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(fft_benchmark
    benchmark.cu
    fft.cu
)
target_compile_features(fft_benchmark PRIVATE cxx_std_17)
set_target_properties(fft_benchmark PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(fft_benchmark PRIVATE CUDA::cudart CUDA::cufft)

enable_testing()
add_test(NAME fft_correctness_and_benchmark COMMAND fft_benchmark)
