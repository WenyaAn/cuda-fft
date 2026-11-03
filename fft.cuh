#pragma once

#include <cuda_runtime.h>

constexpr int kFftSize256 = 256;
constexpr int kFftThreads256 = kFftSize256;

__global__ void fft256_decimation_in_frequency(const float2* input,
                                              float2* output);

cudaError_t launch_fft(const float2* input, float2* output, int fft_size);
cudaError_t launch_fft_config(const float2* input, float2* output, int fft_size,
                              int thread_count);
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                    int fft_size, int iterations,
                                    cudaEvent_t start, cudaEvent_t stop,
                                    float* average_us);
cudaError_t benchmark_fft_config(const float2* input, float2* output,
                                 int fft_size, int thread_count, int iterations,
                                 cudaEvent_t start, cudaEvent_t stop,
                                 float* average_us);
