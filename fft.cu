#include "fft.cuh"

namespace {

/**
 * @brief 将 8 位无符号整数的位顺序反转。
 * @param value 待反转的整数，低 8 位参与计算。
 * @return 低 8 位反转后的整数。
 */
__device__ __forceinline__ unsigned int reverse_bits_8(unsigned int value) {
    value = ((value & 0x55u) << 1) | ((value >> 1) & 0x55u);
    value = ((value & 0x33u) << 2) | ((value >> 2) & 0x33u);
    value = ((value & 0x0fu) << 4) | ((value >> 4) & 0x0fu);
    return value;
}

/**
 * @brief 根据变换方向生成一个复数旋转因子。
 * @param index 旋转因子的频率索引。
 * @param fft_size 当前 FFT 点数。
 * @param inverse 是否生成反向变换使用的共轭方向旋转因子。
 * @return 复数旋转因子 exp(±2πi·index/fft_size)。
 */
__device__ __forceinline__ float2 twiddle_factor(int index,
                                                int fft_size = 256,
                                                bool inverse = false) {
    constexpr float kTwoPi = 6.2831853071795864769f;
    float sine;
    float cosine;

    const float direction = inverse ? 1.0f : -1.0f;
    __sincosf(direction * kTwoPi * static_cast<float>(index) / fft_size, &sine,
              &cosine);
    return make_float2(cosine, sine);
}

/**
 * @brief 计算二的幂点数对应的以 2 为底的对数。
 * @param fft_size FFT 点数。
 * @return fft_size 的 log2 整数值。
 */
__host__ __device__ constexpr int fft_log2(int fft_size) {
    int bits = 0;
    while (fft_size > 1) {
        fft_size >>= 1;
        ++bits;
    }
    return bits;
}

/**
 * @brief 执行两个复数的乘法，并使用融合乘加计算分量。
 * @param lhs 左操作数。
 * @param rhs 右操作数。
 * @return lhs 与 rhs 的复数乘积。
 */
__device__ __forceinline__ float2 complex_multiply(float2 lhs, float2 rhs) {
    return make_float2(__fmaf_rn(-lhs.y, rhs.y, lhs.x * rhs.x),
                       __fmaf_rn(lhs.x, rhs.y, lhs.y * rhs.x));
}

/**
 * @brief 以频率抽取算法执行指定长度的单精度复数 FFT。
 * @tparam FftSize FFT 点数，必须是支持的二次幂长度。
 * @tparam ThreadCount 当前线程块的线程数，必须整除 FftSize。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param inverse 是否执行反向变换；反向变换不做 1/N 归一化。
 * @return 无；输出以自然频率顺序写入 output。
 */
template <int FftSize, int ThreadCount>
__global__ void fft_radix2_decimation_in_frequency(const float2* input,
                                                   float2* output,
                                                   bool inverse) {
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
                twiddle_factor(offset * twiddle_step, FftSize, inverse));
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
                    difference,
                    twiddle_factor(twiddle_index, FftSize, inverse));
            }
        }
    }

    #pragma unroll
    for (int item = 0; item < FftSize / ThreadCount; ++item) {
        const int index = thread + item * ThreadCount;
        output[__brev(static_cast<unsigned int>(index)) >>
               (32 - fft_log2(FftSize))] =
            local_values[item];
    }
}

/**
 * @brief 重复启动指定模板配置的 FFT，并用 CUDA events 计算平均耗时。
 * @tparam FftSize FFT 点数。
 * @tparam ThreadCount 当前线程块的线程数。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param start 计时开始 event。
 * @param stop 计时结束 event。
 * @param iterations FFT kernel 的启动次数。
 * @param average_us 用于接收单次平均执行时间，单位为微秒。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码。
 */
template <int FftSize, int ThreadCount>
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     int iterations, float* average_us,
                                     bool inverse) {
    cudaError_t status = cudaEventRecord(start);
    if (status != cudaSuccess) {
        return status;
    }
    for (int i = 0; i < iterations; ++i) {
        if constexpr (FftSize == kFftSize256) {
            fft256_decimation_in_frequency<<<1, kFftThreads256>>>(input,
                                                                 output,
                                                                 inverse);
        } else {
            fft_radix2_decimation_in_frequency<FftSize, ThreadCount>
                <<<1, ThreadCount>>>(input, output, inverse);
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

}

/**
 * @brief 执行专门优化的 256 点单精度复数 FFT。
 * @param input 指向 256 个设备端复数输入元素。
 * @param output 用于保存 256 个设备端复数输出元素。
 * @param inverse 是否执行反向变换；输出不进行 1/N 归一化。
 * @return 无；输出按自然频率顺序写入。
 */
__global__ void fft256_decimation_in_frequency(const float2* input,
                                              float2* output, bool inverse) {
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
            complex_multiply(difference, twiddle_factor(index, kFftSize256,
                                                        inverse));
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
            difference, twiddle_factor((index & 63) * 2, kFftSize256,
                                       inverse));
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
            difference, twiddle_factor((index & 31) * 4, kFftSize256,
                                       inverse));
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
                                     twiddle_factor(twiddle_index,
                                                    kFftSize256, inverse));
        }
    }

    output[reverse_bits_8(static_cast<unsigned int>(index))] =
        value;
}

/**
 * @brief 选择默认线程配置并启动对应 FFT kernel。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param fft_size FFT 点数，支持 256、512、1024、2048 和 4096。
 * @param inverse 是否执行反向变换；反向变换不做归一化。
 * @return CUDA 状态码；成功返回 cudaSuccess。
 */
cudaError_t launch_fft(const float2* input, float2* output, int fft_size,
                       bool inverse) {
    const int thread_count =
        fft_size >= 1024 ? 1024 : (fft_size == 512 ? 512 : kFftThreads256);
    return launch_fft_config(input, output, fft_size, thread_count, inverse);
}

/**
 * @brief 按 FFT 点数和线程数选择 kernel 并提交设备执行。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param fft_size FFT 点数。
 * @param thread_count 单个线程块的线程数。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码；不支持的点数或线程配置返回相应错误。
 */
cudaError_t launch_fft_config(const float2* input, float2* output, int fft_size,
                              int thread_count, bool inverse) {
    switch (fft_size) {
        case 256:
            if (thread_count != kFftThreads256) {
                return cudaErrorInvalidConfiguration;
            }
            fft256_decimation_in_frequency<<<1, kFftThreads256>>>(input,
                                                                 output,
                                                                 inverse);
            break;
        case 512:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<512, 256>
                    <<<1, 256>>>(input, output, inverse);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<512, 512>
                    <<<1, 512>>>(input, output, inverse);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 1024:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<1024, 256>
                    <<<1, 256>>>(input, output, inverse);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<1024, 512>
                    <<<1, 512>>>(input, output, inverse);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<1024, 1024>
                    <<<1, 1024>>>(input, output, inverse);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 2048:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<2048, 256>
                    <<<1, 256>>>(input, output, inverse);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<2048, 512>
                    <<<1, 512>>>(input, output, inverse);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<2048, 1024>
                    <<<1, 1024>>>(input, output, inverse);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 4096:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<4096, 256>
                    <<<1, 256>>>(input, output, inverse);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<4096, 512>
                    <<<1, 512>>>(input, output, inverse);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<4096, 1024>
                    <<<1, 1024>>>(input, output, inverse);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

/**
 * @brief 按默认线程配置重复运行 FFT，并计算单次平均耗时。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param fft_size FFT 点数。
 * @param iterations kernel 启动次数。
 * @param start 计时开始 event。
 * @param stop 计时结束 event。
 * @param average_us 用于接收平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码；不支持的点数返回 cudaErrorInvalidValue。
 */
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                     int fft_size, int iterations,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     float* average_us, bool inverse) {
    switch (fft_size) {
        case 256:
            return benchmark_fft_config(input, output, fft_size, 256,
                                        iterations, start, stop, average_us,
                                        inverse);
        case 512:
            return benchmark_fft_config(input, output, fft_size, 512,
                                        iterations, start, stop, average_us,
                                        inverse);
        case 1024:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us,
                                        inverse);
        case 2048:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us,
                                        inverse);
        case 4096:
            return benchmark_fft_config(input, output, fft_size, 1024,
                                        iterations, start, stop, average_us,
                                        inverse);
        default:
            return cudaErrorInvalidValue;
    }
}

/**
 * @brief 按指定线程配置重复运行 FFT，并计算单次平均耗时。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param fft_size FFT 点数。
 * @param thread_count 单个线程块的线程数。
 * @param iterations kernel 启动次数。
 * @param start 计时开始 event。
 * @param stop 计时结束 event。
 * @param average_us 用于接收平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码；不支持的点数或线程配置返回错误。
 */
cudaError_t benchmark_fft_config(const float2* input, float2* output,
                                 int fft_size, int thread_count, int iterations,
                                 cudaEvent_t start, cudaEvent_t stop,
                                 float* average_us, bool inverse) {
    switch (fft_size) {
        case 256:
            if (thread_count == 256) {
                return benchmark_fft_iterations<256, 256>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            break;
        case 512:
            if (thread_count == 256) {
                return benchmark_fft_iterations<512, 256>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<512, 512>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            break;
        case 1024:
            if (thread_count == 256) {
                return benchmark_fft_iterations<1024, 256>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<1024, 512>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<1024, 1024>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            break;
        case 2048:
            if (thread_count == 256) {
                return benchmark_fft_iterations<2048, 256>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<2048, 512>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<2048, 1024>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            break;
        case 4096:
            if (thread_count == 256) {
                return benchmark_fft_iterations<4096, 256>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<4096, 512>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<4096, 1024>(
                    input, output, start, stop, iterations, average_us,
                    inverse);
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaErrorInvalidConfiguration;
}
