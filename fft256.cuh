#pragma once

#include <cuda_runtime.h>

constexpr int kFftSize = 256;
constexpr int kFftThreads = 32;

__global__ void fft256_cooley_tukey(const float2* input, float2* output);
