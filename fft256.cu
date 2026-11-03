#include "fft256.cuh"

namespace {

__device__ __forceinline__ unsigned int reverse_bits_8(unsigned int value) {
    value = ((value & 0x55u) << 1) | ((value >> 1) & 0x55u);
    value = ((value & 0x33u) << 2) | ((value >> 2) & 0x33u);
    value = ((value & 0x0fu) << 4) | ((value >> 4) & 0x0fu);
    return value;
}

__device__ __forceinline__ float2 twiddle_factor(int index,
                                                int fft_size = 256) {
    constexpr float kTwoPi = 6.2831853071795864769f;
    float sine;
    float cosine;
    __sincosf(-kTwoPi * static_cast<float>(index) / fft_size, &sine,
              &cosine);
    return make_float2(cosine, sine);
}

__host__ __device__ constexpr int fft_log2(int fft_size) {
    int bits = 0;
    while (fft_size > 1) {
        fft_size >>= 1;
        ++bits;
    }
    return bits;
}

__device__ __forceinline__ float2 complex_multiply(float2 lhs, float2 rhs) {
    return make_float2(__fmaf_rn(-lhs.y, rhs.y, lhs.x * rhs.x),
                       __fmaf_rn(lhs.x, rhs.y, lhs.y * rhs.x));
}

template <int FftSize, int ThreadCount>
__global__ void fft_radix2_decimation_in_frequency(const float2* input,
                                                   float2* output) {
    __shared__ float2 values[FftSize];

    const int thread = threadIdx.x;
    for (int index = thread; index < FftSize; index += blockDim.x) {
        values[index] = input[index];
    }
    __syncthreads();

    for (int stage_size = FftSize; stage_size >= 64; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        const int twiddle_step = FftSize / stage_size;
        for (int butterfly = thread; butterfly < FftSize / 2;
             butterfly += blockDim.x) {
            const int group = butterfly / half_size;
            const int offset = butterfly % half_size;
            const int even_index = group * stage_size + offset;
            const int odd_index = even_index + half_size;
            const float2 even = values[even_index];
            const float2 odd = values[odd_index];
            values[even_index] =
                make_float2(even.x + odd.x, even.y + odd.y);

            const float2 difference =
                make_float2(even.x - odd.x, even.y - odd.y);
            values[odd_index] = complex_multiply(
                difference,
                twiddle_factor(offset * twiddle_step, FftSize));
        }
        __syncthreads();
    }

    float2 local_values[FftSize / ThreadCount];
    #pragma unroll
    for (int item = 0; item < FftSize / ThreadCount; ++item) {
        local_values[item] = values[thread + item * ThreadCount];
    }

    #pragma unroll
    for (int stage_size = 32; stage_size >= 2; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        const int twiddle_index =
            (thread & (half_size - 1)) * (FftSize / stage_size);
        #pragma unroll
        for (int item = 0; item < FftSize / ThreadCount; ++item) {
            const float2 value = local_values[item];
            const float2 partner = make_float2(
                __shfl_xor_sync(0xffffffffu, value.x, half_size),
                __shfl_xor_sync(0xffffffffu, value.y, half_size));
            if ((thread & half_size) == 0) {
                local_values[item] =
                    make_float2(value.x + partner.x, value.y + partner.y);
            } else {
                const float2 difference =
                    make_float2(partner.x - value.x, partner.y - value.y);
                local_values[item] = complex_multiply(
                    difference, twiddle_factor(twiddle_index, FftSize));
            }
        }
    }

    #pragma unroll
    for (int item = 0; item < FftSize / ThreadCount; ++item) {
        const int index = thread + item * ThreadCount;
        output[__brev(static_cast<unsigned int>(index)) >>
               (32 - fft_log2(FftSize))] = local_values[item];
    }
}

template <int FftSize, int ThreadCount>
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     int iterations, float* average_us) {
    cudaError_t status = cudaEventRecord(start);
    if (status != cudaSuccess) {
        return status;
    }
    for (int i = 0; i < iterations; ++i) {
        if constexpr (FftSize == kFftSize256) {
            fft256_decimation_in_frequency<<<1, kFftThreads256>>>(input,
                                                                 output);
        } else {
            fft_radix2_decimation_in_frequency<FftSize, ThreadCount>
                <<<1, ThreadCount>>>(input, output);
        }
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaEventRecord(stop);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaEventSynchronize(stop);
    if (status != cudaSuccess) {
        return status;
    }
    float elapsed_ms = 0.0f;
    status = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (status == cudaSuccess) {
        *average_us = elapsed_ms * 1000.0f / iterations;
    }
    return status;
}

}  // namespace

__global__ void fft256_decimation_in_frequency(const float2* input,
                                              float2* output) {
    __shared__ float2 shared_values[2][kFftSize256];

    const int index = threadIdx.x;
    if (index < kFftSize256 / 2) {
        const float2 even = input[index];
        const float2 odd = input[index + kFftSize256 / 2];
        const float2 difference =
            make_float2(even.x - odd.x, even.y - odd.y);
        shared_values[0][index] =
            make_float2(even.x + odd.x, even.y + odd.y);
        shared_values[0][index + kFftSize256 / 2] =
            complex_multiply(difference, twiddle_factor(index));
    }
    __syncthreads();

    if ((index & 64) == 0) {
        const float2 even = shared_values[0][index];
        const float2 odd = shared_values[0][index | 64];
        shared_values[1][index] =
            make_float2(even.x + odd.x, even.y + odd.y);
    } else {
        const float2 even = shared_values[0][index & ~64];
        const float2 odd = shared_values[0][index];
        const float2 difference =
            make_float2(even.x - odd.x, even.y - odd.y);
        shared_values[1][index] = complex_multiply(
            difference, twiddle_factor((index & 63) * 2));
    }
    __syncthreads();

    if ((index & 32) == 0) {
        const float2 even = shared_values[1][index];
        const float2 odd = shared_values[1][index | 32];
        shared_values[0][index] =
            make_float2(even.x + odd.x, even.y + odd.y);
    } else {
        const float2 even = shared_values[1][index & ~32];
        const float2 odd = shared_values[1][index];
        const float2 difference =
            make_float2(even.x - odd.x, even.y - odd.y);
        shared_values[0][index] = complex_multiply(
            difference, twiddle_factor((index & 31) * 4));
    }
    __syncthreads();

    float2 value = shared_values[0][index];
    #pragma unroll
    for (int stage_size = 32; stage_size >= 2; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        const float2 partner = make_float2(
            __shfl_xor_sync(0xffffffffu, value.x, half_size),
            __shfl_xor_sync(0xffffffffu, value.y, half_size));
        if ((index & half_size) == 0) {
            value = make_float2(value.x + partner.x, value.y + partner.y);
        } else {
            const float2 difference =
                make_float2(partner.x - value.x, partner.y - value.y);
            const int twiddle_index =
                (index & (half_size - 1)) * (kFftSize256 / stage_size);
            value = complex_multiply(difference,
                                     twiddle_factor(twiddle_index));
        }
    }

    output[reverse_bits_8(static_cast<unsigned int>(index))] = value;
}

cudaError_t launch_fft(const float2* input, float2* output, int fft_size) {
    const int thread_count =
        fft_size >= 1024 ? 1024 : (fft_size == 512 ? 512 : kFftThreads256);
    return launch_fft_config(input, output, fft_size, thread_count);
}

cudaError_t launch_fft_config(const float2* input, float2* output, int fft_size,
                              int thread_count) {
    switch (fft_size) {
        case 256:
            if (thread_count != kFftThreads256) {
                return cudaErrorInvalidConfiguration;
            }
            fft256_decimation_in_frequency<<<1, kFftThreads256>>>(input,
                                                                 output);
            break;
        case 512:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<512, 256>
                    <<<1, 256>>>(input, output);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<512, 512>
                    <<<1, 512>>>(input, output);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 1024:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<1024, 256>
                    <<<1, 256>>>(input, output);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<1024, 512>
                    <<<1, 512>>>(input, output);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<1024, 1024>
                    <<<1, 1024>>>(input, output);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 2048:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<2048, 256>
                    <<<1, 256>>>(input, output);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<2048, 512>
                    <<<1, 512>>>(input, output);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<2048, 1024>
                    <<<1, 1024>>>(input, output);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 4096:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<4096, 256>
                    <<<1, 256>>>(input, output);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<4096, 512>
                    <<<1, 512>>>(input, output);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<4096, 1024>
                    <<<1, 1024>>>(input, output);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                     int fft_size, int iterations,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     float* average_us) {
    switch (fft_size) {
        case 256:
            return benchmark_fft_config(input, output, fft_size, 256,
                                        iterations, start, stop, average_us);
        case 512:
            return benchmark_fft_config(input, output, fft_size, 512,
                                        iterations, start, stop, average_us);
        case 1024:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us);
        case 2048:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us);
        case 4096:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t benchmark_fft_config(const float2* input, float2* output,
                                 int fft_size, int thread_count, int iterations,
                                 cudaEvent_t start, cudaEvent_t stop,
                                 float* average_us) {
    switch (fft_size) {
        case 256:
            if (thread_count == 256) {
                return benchmark_fft_iterations<256, 256>(
                    input, output, start, stop, iterations, average_us);
            }
            break;
        case 512:
            if (thread_count == 256) {
                return benchmark_fft_iterations<512, 256>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<512, 512>(
                    input, output, start, stop, iterations, average_us);
            }
            break;
        case 1024:
            if (thread_count == 256) {
                return benchmark_fft_iterations<1024, 256>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<1024, 512>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<1024, 1024>(
                    input, output, start, stop, iterations, average_us);
            }
            break;
        case 2048:
            if (thread_count == 256) {
                return benchmark_fft_iterations<2048, 256>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<2048, 512>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<2048, 1024>(
                    input, output, start, stop, iterations, average_us);
            }
            break;
        case 4096:
            if (thread_count == 256) {
                return benchmark_fft_iterations<4096, 256>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<4096, 512>(
                    input, output, start, stop, iterations, average_us);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<4096, 1024>(
                    input, output, start, stop, iterations, average_us);
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaErrorInvalidConfiguration;
}
