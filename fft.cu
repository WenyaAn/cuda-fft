#include "fft.cuh"

#include <cmath>
#include <memory>
#include <new>
#include <vector>

struct FFTPlan {
    int fft_size;
    int thread_count;
    int device;
    float2* twiddle_factors;
};

__global__ void fft256_decimation_in_frequency(
    const float2* input, float2* output, bool inverse,
    const float2* twiddle_factors);

namespace {

bool is_supported_fft_size(int fft_size) {
    return fft_size == 256 || fft_size == 512 || fft_size == 1024 ||
           fft_size == 2048 || fft_size == 4096;
}

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
__device__ __forceinline__ float2 twiddle_factor(
    int index, bool inverse, const float2* twiddle_factors) {
    const float2 factor = twiddle_factors[index];
    return inverse ? make_float2(factor.x, -factor.y) : factor;
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
 * @tparam FFTSize FFT 点数，必须是支持的二次幂长度。
 * @tparam ThreadCount 当前线程块的线程数，必须整除 FFTSize。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param inverse 是否执行反向变换；反向变换不做 1/N 归一化。
 * @return 无；输出以自然频率顺序写入 output。
 */
template <int FFTSize, int ThreadCount>
__global__ void fft_radix2_decimation_in_frequency(const float2* input,
                                                   float2* output,
                                                   bool inverse,
                                                   const float2* twiddle_factors) {
    __shared__ float2 values[FFTSize];
    __shared__ float2 shared_twiddle_factors[FFTSize / 2];

    const int thread = threadIdx.x;
    const size_t base_offset = static_cast<size_t>(blockIdx.x) * FFTSize;
    for (int index = thread; index < FFTSize; index += blockDim.x) {
        values[index] = input[base_offset + index];
    }
    for (int index = thread; index < FFTSize / 2; index += blockDim.x) {
        shared_twiddle_factors[index] = twiddle_factors[index];
    }
    __syncthreads();

    for (int stage_size = FFTSize; stage_size >= 64; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        const int twiddle_step = FFTSize / stage_size;
        for (int butterfly = thread; butterfly < FFTSize / 2;
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
            values[odd_index] =
                complex_multiply(difference,
                                 twiddle_factor(offset * twiddle_step, inverse,
                                                shared_twiddle_factors));
        }
        __syncthreads();
    }

    float2 local_values[FFTSize / ThreadCount];
    #pragma unroll
    for (int item = 0; item < FFTSize / ThreadCount; ++item) {
        local_values[item] = values[thread + item * ThreadCount];
    }

    #pragma unroll
    for (int stage_size = 32; stage_size >= 2; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        const int twiddle_index =
            (thread & (half_size - 1)) * (FFTSize / stage_size);
        #pragma unroll
        for (int item = 0; item < FFTSize / ThreadCount; ++item) {
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
                    twiddle_factor(twiddle_index, inverse,
                                   shared_twiddle_factors));
            }
        }
    }

    #pragma unroll
    for (int item = 0; item < FFTSize / ThreadCount; ++item) {
        const int index = thread + item * ThreadCount;
        output[base_offset +
               (__brev(static_cast<unsigned int>(index)) >>
                (32 - fft_log2(FFTSize)))] =
            local_values[item];
    }
}

/**
 * @brief 重复启动指定模板配置的 FFT，并用 CUDA events 计算平均耗时。
 * @tparam FFTSize FFT 点数。
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
template <int FFTSize, int ThreadCount>
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                     cudaEvent_t start, cudaEvent_t stop,
                                     int iterations, float* average_us,
                                     bool inverse,
                                     const float2* twiddle_factors) {
    cudaError_t status = cudaEventRecord(start);
    if (status != cudaSuccess) {
        return status;
    }
    for (int i = 0; i < iterations; ++i) {
        if constexpr (FFTSize == kFFTSize256) {
            fft256_decimation_in_frequency
                <<<1, kFFTThreads256>>>(input, output, inverse,
                                        twiddle_factors);
        } else {
            fft_radix2_decimation_in_frequency<FFTSize, ThreadCount>
                <<<1, ThreadCount>>>(input, output, inverse, twiddle_factors);
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
__device__ void fft256_decimation_in_frequency_body(
    const float2* input, float2* output, bool inverse,
    const float2* twiddle_factors) {
    __shared__ float2 shared_values[2][kFFTSize256];
    __shared__ float2 shared_twiddle_factors[kFFTSize256 / 2];

    const int index = threadIdx.x;
    const size_t base_offset =
        static_cast<size_t>(blockIdx.x) * kFFTSize256;
    for (int twiddle_index = index; twiddle_index < kFFTSize256 / 2;
         twiddle_index += blockDim.x) {
        shared_twiddle_factors[twiddle_index] =
            twiddle_factors[twiddle_index];
    }
    __syncthreads();

    if (index < kFFTSize256 / 2) {
        const float2 even = input[base_offset + index];
        const float2 odd = input[base_offset + index + kFFTSize256 / 2];
        const float2 difference =
            make_float2(even.x - odd.x, even.y - odd.y);
        shared_values[0][index] =
            make_float2(even.x + odd.x, even.y + odd.y);
        shared_values[0][index + kFFTSize256 / 2] =
            complex_multiply(difference,
                             twiddle_factor(index, inverse,
                                            shared_twiddle_factors));
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
            difference,
            twiddle_factor((index & 63) * 2, inverse,
                           shared_twiddle_factors));
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
            difference,
            twiddle_factor((index & 31) * 4, inverse,
                           shared_twiddle_factors));
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
                (index & (half_size - 1)) * (kFFTSize256 / stage_size);
            value = complex_multiply(difference,
                                     twiddle_factor(twiddle_index, inverse,
                                                    shared_twiddle_factors));
        }
    }

    output[base_offset + reverse_bits_8(static_cast<unsigned int>(index))] =
        value;
}

__global__ void fft256_decimation_in_frequency(
    const float2* input, float2* output, bool inverse,
    const float2* twiddle_factors) {
    fft256_decimation_in_frequency_body(input, output, inverse,
                                        twiddle_factors);
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
namespace {

cudaError_t launch_fft_plan_config_impl(const float2* input, float2* output,
                                        int fft_size, int thread_count,
                                        int grid_size, bool inverse,
                                        const float2* twiddle_factors) {
    switch (fft_size) {
        case 256:
            if (thread_count != kFFTThreads256) {
                return cudaErrorInvalidConfiguration;
            }
            fft256_decimation_in_frequency<<<grid_size, kFFTThreads256>>>(
                input, output, inverse, twiddle_factors);
            break;
        case 512:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<512, 256>
                    <<<grid_size, 256>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<512, 512>
                    <<<grid_size, 512>>>(input, output, inverse, twiddle_factors);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 1024:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<1024, 256>
                    <<<grid_size, 256>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<1024, 512>
                    <<<grid_size, 512>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<1024, 1024>
                    <<<grid_size, 1024>>>(input, output, inverse, twiddle_factors);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 2048:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<2048, 256>
                    <<<grid_size, 256>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<2048, 512>
                    <<<grid_size, 512>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<2048, 1024>
                    <<<grid_size, 1024>>>(input, output, inverse, twiddle_factors);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 4096:
            if (thread_count == 256) {
                fft_radix2_decimation_in_frequency<4096, 256>
                    <<<grid_size, 256>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 512) {
                fft_radix2_decimation_in_frequency<4096, 512>
                    <<<grid_size, 512>>>(input, output, inverse, twiddle_factors);
            } else if (thread_count == 1024) {
                fft_radix2_decimation_in_frequency<4096, 1024>
                    <<<grid_size, 1024>>>(input, output, inverse, twiddle_factors);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

}

cudaError_t create_fft_plan(FFTPlan** plan, int fft_size) {
    if (plan == nullptr || *plan != nullptr ||
        !is_supported_fft_size(fft_size)) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    std::unique_ptr<FFTPlan> new_plan(new (std::nothrow) FFTPlan{});
    if (!new_plan) {
        return cudaErrorMemoryAllocation;
    }
    std::vector<float2> host_twiddle_factors;
    try {
        host_twiddle_factors.resize(static_cast<size_t>(fft_size / 2));
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    }
    constexpr double kTwoPi = 6.2831853071795864769;
    for (int index = 0; index < fft_size / 2; ++index) {
        const double angle =
            -kTwoPi * static_cast<double>(index) / fft_size;
        host_twiddle_factors[index] =
            make_float2(static_cast<float>(std::cos(angle)),
                        static_cast<float>(std::sin(angle)));
    }
    status = cudaMalloc(&new_plan->twiddle_factors,
                        host_twiddle_factors.size() * sizeof(float2));
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemcpy(new_plan->twiddle_factors,
                        host_twiddle_factors.data(),
                        host_twiddle_factors.size() * sizeof(float2),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(new_plan->twiddle_factors);
        return status;
    }
    new_plan->fft_size = fft_size;
    new_plan->thread_count =
        fft_size >= 1024 ? 1024 : (fft_size == 512 ? 512 : kFFTThreads256);
    new_plan->device = device;
    *plan = new_plan.release();
    return cudaSuccess;
}

cudaError_t destroy_fft_plan(FFTPlan** plan) {
    if (plan == nullptr || *plan == nullptr) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (device != (*plan)->device) {
        return cudaErrorInvalidDevice;
    }
    status = cudaFree((*plan)->twiddle_factors);
    if (status != cudaSuccess) {
        return status;
    }
    delete *plan;
    *plan = nullptr;
    return cudaSuccess;
}

cudaError_t launch_fft_plan(const FFTPlan* plan, const float2* input,
                            float2* output, bool inverse) {
    if (plan == nullptr) {
        return cudaErrorInvalidValue;
    }
    return launch_fft_plan_config(plan, input, output, plan->thread_count,
                                  inverse);
}

cudaError_t launch_fft_plan_config(const FFTPlan* plan, const float2* input,
                                   float2* output, int thread_count,
                                   bool inverse) {
    if (plan == nullptr || input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (device != plan->device) {
        return cudaErrorInvalidDevice;
    }
    return launch_fft_plan_config_impl(input, output, plan->fft_size,
                                       thread_count, 1, inverse,
                                       plan->twiddle_factors);
}

cudaError_t launch_fft_plan_batch(const FFTPlan* plan, const float2* input,
                                  float2* output, int batch_count,
                                  bool inverse) {
    if (plan == nullptr || input == nullptr || output == nullptr ||
        batch_count <= 0) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (device != plan->device) {
        return cudaErrorInvalidDevice;
    }
    return launch_fft_plan_config_impl(input, output, plan->fft_size,
                                       plan->thread_count, batch_count, inverse,
                                       plan->twiddle_factors);
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
namespace {

cudaError_t benchmark_fft_plan_config_impl(
    const float2* input, float2* output, int fft_size, int thread_count,
    int iterations, cudaEvent_t start, cudaEvent_t stop, float* average_us,
    bool inverse, const float2* twiddle_factors) {
    switch (fft_size) {
        case 256:
            if (thread_count == 256) {
                return benchmark_fft_iterations<256, 256>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            break;
        case 512:
            if (thread_count == 256) {
                return benchmark_fft_iterations<512, 256>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<512, 512>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            break;
        case 1024:
            if (thread_count == 256) {
                return benchmark_fft_iterations<1024, 256>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<1024, 512>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<1024, 1024>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            break;
        case 2048:
            if (thread_count == 256) {
                return benchmark_fft_iterations<2048, 256>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<2048, 512>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<2048, 1024>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            break;
        case 4096:
            if (thread_count == 256) {
                return benchmark_fft_iterations<4096, 256>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 512) {
                return benchmark_fft_iterations<4096, 512>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            if (thread_count == 1024) {
                return benchmark_fft_iterations<4096, 1024>(
                    input, output, start, stop, iterations, average_us, inverse,
                    twiddle_factors);
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaErrorInvalidConfiguration;
}

}

cudaError_t benchmark_fft_plan_config(const FFTPlan* plan,
                                      const float2* input, float2* output,
                                      int thread_count, int iterations,
                                      cudaEvent_t start, cudaEvent_t stop,
                                      float* average_us, bool inverse) {
    if (plan == nullptr || input == nullptr || output == nullptr ||
        iterations <= 0 || average_us == nullptr) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (device != plan->device) {
        return cudaErrorInvalidDevice;
    }
    return benchmark_fft_plan_config_impl(
        input, output, plan->fft_size, thread_count, iterations, start, stop,
        average_us, inverse, plan->twiddle_factors);
}

cudaError_t benchmark_fft_plan_iterations(const FFTPlan* plan,
                                          const float2* input, float2* output,
                                          int iterations, cudaEvent_t start,
                                          cudaEvent_t stop, float* average_us,
                                          bool inverse) {
    if (plan == nullptr) {
        return cudaErrorInvalidValue;
    }
    return benchmark_fft_plan_config(plan, input, output, plan->thread_count,
                                     iterations, start, stop, average_us,
                                     inverse);
}
