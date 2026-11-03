#include "fft256.cuh"

#include <cmath>

namespace {

__device__ __forceinline__ unsigned int reverse_bits_8(unsigned int value) {
    value = ((value & 0x55u) << 1) | ((value >> 1) & 0x55u);
    value = ((value & 0x33u) << 2) | ((value >> 2) & 0x33u);
    value = ((value & 0x0fu) << 4) | ((value >> 4) & 0x0fu);
    return value;
}

__device__ __forceinline__ float2 complex_multiply(float2 lhs, float2 rhs) {
    return make_float2(lhs.x * rhs.x - lhs.y * rhs.y,
                       lhs.x * rhs.y + lhs.y * rhs.x);
}

}  // namespace

__global__ void fft256_cooley_tukey(const float2* input, float2* output) {
    __shared__ float2 values[kFftSize];

    const int thread = threadIdx.x;
    for (int index = thread; index < kFftSize; index += kFftThreads) {
        values[index] = input[reverse_bits_8(static_cast<unsigned int>(index))];
    }
    __syncthreads();

    constexpr int kButterfliesPerThread = (kFftSize / 2) / kFftThreads;
    for (int stage_size = 2; stage_size <= kFftSize; stage_size <<= 1) {
        const int half_size = stage_size / 2;
        for (int item = 0; item < kButterfliesPerThread; ++item) {
            const int butterfly = thread * kButterfliesPerThread + item;
            const int group = butterfly / half_size;
            const int offset = butterfly % half_size;
            const int even_index = group * stage_size + offset;
            const int odd_index = even_index + half_size;

            const float angle = -6.2831853071795864769f *
                                static_cast<float>(offset) /
                                static_cast<float>(stage_size);
            float sine;
            float cosine;
            sincosf(angle, &sine, &cosine);
            const float2 even = values[even_index];
            const float2 odd = complex_multiply(
                values[odd_index], make_float2(cosine, sine));

            values[even_index] = make_float2(even.x + odd.x, even.y + odd.y);
            values[odd_index] = make_float2(even.x - odd.x, even.y - odd.y);
        }
        __syncthreads();
    }

    for (int index = thread; index < kFftSize; index += kFftThreads) {
        output[index] = values[index];
    }
}
