#include "fft.cuh"

#include <cmath>
#include <memory>
#include <new>
#include <vector>

struct FFTPlan {
    int fft_size;             // 每次变换的复数元素数量。
    int thread_count;         // 默认每个变换线程块的线程数。
    int device;               // 创建此 plan 时所使用的 CUDA 设备。
    float2* twiddle_factors;  // 设备端正向变换旋转因子表。
};

struct FFTPlanFp64 {
    int fft_size;
    int thread_count;
    int device;
    double2* twiddle_factors;
};

__global__ void fft256_decimation_in_frequency(
    const float2* input, float2* output, bool inverse,
    const float2* twiddle_factors);

namespace {

/**
 * @brief 检查 FFT 点数是否为当前实现支持的长度。
 * @param fft_size 待检查的 FFT 点数。
 * @return 支持 256、512、1024、2048 或 4096 点时返回 true，否则返回 false。
 */
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

__device__ __forceinline__ double2 twiddle_factor_fp64(
    int index, bool inverse, const double2* twiddle_factors) {
    const double2 factor = twiddle_factors[index];
    return inverse ? make_double2(factor.x, -factor.y) : factor;
}

__device__ __forceinline__ double2 complex_multiply_fp64(double2 lhs,
                                                          double2 rhs) {
    return make_double2(__fma_rn(-lhs.y, rhs.y, lhs.x * rhs.x),
                        __fma_rn(lhs.x, rhs.y, lhs.y * rhs.x));
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
    // 大尺寸蝶形阶段在共享内存中交换数据；旋转因子表也暂存于共享内存。
    __shared__ float2 values[FFTSize];
    __shared__ float2 shared_twiddle_factors[FFTSize / 2];

    const int thread = threadIdx.x;
    const size_t base_offset = static_cast<size_t>(blockIdx.x) * FFTSize;
    // 当前 block 负责一个 FFT，线程协作加载输入和旋转因子。
    for (int index = thread; index < FFTSize; index += blockDim.x) {
        values[index] = input[base_offset + index];
    }
    for (int index = thread; index < FFTSize / 2; index += blockDim.x) {
        shared_twiddle_factors[index] = twiddle_factors[index];
    }
    __syncthreads();

    // DIF 蝶形按段长从 FFTSize 逐级减半；跨 warp 数据通过共享内存交换，
    // 每轮结束后同步，确保下一阶段读取到完整结果。
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

    // 将剩余数据从共享内存读入每个线程的寄存器，供 warp 内阶段使用。
    float2 local_values[FFTSize / ThreadCount];
    #pragma unroll
    for (int item = 0; item < FFTSize / ThreadCount; ++item) {
        local_values[item] = values[thread + item * ThreadCount];
    }

    // 段长不超过一个 warp 后，通过 shuffle 在 lane 间交换配对值，
    // 避免这些蝶形阶段继续访问共享内存和进行 block 级同步。
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
            // 每对蝶形中，一侧计算和，另一侧计算差并乘旋转因子。
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

    // DIF 的末级结果为位反转顺序，写出时重排为自然频率顺序。
    #pragma unroll
    for (int item = 0; item < FFTSize / ThreadCount; ++item) {
        const int index = thread + item * ThreadCount;
        output[base_offset +
               (__brev(static_cast<unsigned int>(index)) >>
                (32 - fft_log2(FFTSize)))] =
            local_values[item];
    }
}

template <int FFTSize, int ThreadCount>
__global__ void fft_radix2_decimation_in_frequency_fp64(
    const double2* input, double2* output, bool inverse,
    const double2* twiddle_factors) {
    extern __shared__ double2 values[];

    const int thread = threadIdx.x;
    const size_t base_offset = static_cast<size_t>(blockIdx.x) * FFTSize;
    for (int index = thread; index < FFTSize; index += blockDim.x) {
        values[index] = input[base_offset + index];
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
            const double2 even = values[even_index];
            const double2 odd = values[odd_index];
            values[even_index] =
                make_double2(even.x + odd.x, even.y + odd.y);

            const double2 difference =
                make_double2(even.x - odd.x, even.y - odd.y);
            values[odd_index] = complex_multiply_fp64(
                difference,
                twiddle_factor_fp64(offset * twiddle_step, inverse,
                                    twiddle_factors));
        }
        __syncthreads();
    }

    double2 local_values[FFTSize / ThreadCount];
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
            const double2 value = local_values[item];
            const double2 partner = make_double2(
                __longlong_as_double(__shfl_xor_sync(
                    0xffffffffu, __double_as_longlong(value.x), half_size)),
                __longlong_as_double(__shfl_xor_sync(
                    0xffffffffu, __double_as_longlong(value.y), half_size)));
            if ((thread & half_size) == 0) {
                local_values[item] =
                    make_double2(value.x + partner.x, value.y + partner.y);
            } else {
                const double2 difference =
                    make_double2(partner.x - value.x, partner.y - value.y);
                local_values[item] = complex_multiply_fp64(
                    difference,
                    twiddle_factor_fp64(twiddle_index, inverse,
                                        twiddle_factors));
            }
        }
    }

    #pragma unroll
    for (int item = 0; item < FFTSize / ThreadCount; ++item) {
        const int index = thread + item * ThreadCount;
        output[base_offset +
               (__brev(static_cast<unsigned int>(index)) >>
                (32 - fft_log2(FFTSize)))] = local_values[item];
    }
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
    // 每个 block 处理一个 256 点 FFT；先将旋转因子表协作加载到共享内存。
    for (int twiddle_index = index; twiddle_index < kFFTSize256 / 2;
         twiddle_index += blockDim.x) {
        shared_twiddle_factors[twiddle_index] =
            twiddle_factors[twiddle_index];
    }
    __syncthreads();

    // 第一轮 DIF 蝶形配对距离为 128：每个线程计算一组和与差，
    // 差值乘对应旋转因子后写入共享内存。
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

    // 继续完成段长 128 的蝶形阶段，配对距离为 64。
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

    // 完成段长 64 的蝶形阶段，配对距离为 32；这些跨 warp 数据仍通过共享内存交换。
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

    // 将每个线程的当前值保存在寄存器中，后续小阶段完全在 warp 内完成。
    float2 value = shared_values[0][index];
    #pragma unroll
    for (int stage_size = 32; stage_size >= 2; stage_size >>= 1) {
        const int half_size = stage_size / 2;
        // 通过 lane 索引异或 half_size 找到本轮蝶形配对线程。
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

    // 将 DIF 产生的位反转排列写出为自然频率顺序。
    output[base_offset + reverse_bits_8(static_cast<unsigned int>(index))] =
        value;
}

/**
 * @brief 256 点 FFT kernel 的全局入口，转发到专用设备端实现。
 * @param input 指向设备端复数输入数组。
 * @param output 用于保存自然频率顺序结果的设备端数组。
 * @param inverse 是否执行不进行 1/N 归一化的逆向变换。
 * @param twiddle_factors 指向设备端预计算的旋转因子表。
 * @return 无；kernel 执行结果写入 output。
 */
__global__ void fft256_decimation_in_frequency(
    const float2* input, float2* output, bool inverse,
    const float2* twiddle_factors) {
    fft256_decimation_in_frequency_body(input, output, inverse,
                                        twiddle_factors);
}

namespace {

/**
 * @brief 根据 FFT 点数和线程配置选择 kernel 并提交到 CUDA grid 执行。
 * @param input 指向设备端复数输入数组。
 * @param output 指向设备端复数输出数组。
 * @param fft_size FFT 点数。
 * @param thread_count 每个 block 的线程数。
 * @param grid_size CUDA grid 中的 block 数；批量执行时通常等于 FFT 数量。
 * @param inverse 是否执行不进行 1/N 归一化的逆向变换。
 * @param twiddle_factors 指向设备端预计算的旋转因子表。
 * @return CUDA 状态码；点数或线程配置不受支持时返回错误。
 */
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

cudaError_t launch_fft_plan_config_fp64_impl(
    const double2* input, double2* output, int fft_size, int thread_count,
    int grid_size, bool inverse, const double2* twiddle_factors) {
#define LAUNCH_FP64(FFT_SIZE, THREAD_COUNT)                                  \
    fft_radix2_decimation_in_frequency_fp64<FFT_SIZE, THREAD_COUNT>         \
        <<<grid_size, THREAD_COUNT, FFT_SIZE * sizeof(double2)>>>(           \
            input, output, inverse, twiddle_factors)
    switch (fft_size) {
        case 256:
            if (thread_count != 256) {
                return cudaErrorInvalidConfiguration;
            }
            LAUNCH_FP64(256, 256);
            break;
        case 512:
            if (thread_count == 256) {
                LAUNCH_FP64(512, 256);
            } else if (thread_count == 512) {
                LAUNCH_FP64(512, 512);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 1024:
            if (thread_count == 256) {
                LAUNCH_FP64(1024, 256);
            } else if (thread_count == 512) {
                LAUNCH_FP64(1024, 512);
            } else if (thread_count == 1024) {
                LAUNCH_FP64(1024, 1024);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 2048:
            if (thread_count == 256) {
                LAUNCH_FP64(2048, 256);
            } else if (thread_count == 512) {
                LAUNCH_FP64(2048, 512);
            } else if (thread_count == 1024) {
                LAUNCH_FP64(2048, 1024);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        case 4096:
            if (thread_count == 256) {
                LAUNCH_FP64(4096, 256);
            } else if (thread_count == 512) {
                LAUNCH_FP64(4096, 512);
            } else if (thread_count == 1024) {
                LAUNCH_FP64(4096, 1024);
            } else {
                return cudaErrorInvalidConfiguration;
            }
            break;
        default:
            return cudaErrorInvalidValue;
    }
#undef LAUNCH_FP64
    return cudaSuccess;
}

}

/**
 * @brief 创建 FFT plan，预计算旋转因子并上传到当前 CUDA 设备。
 * @param plan 接收新 plan 的指针；必须非空且初始值为 nullptr。
 * @param fft_size FFT 点数，支持 256、512、1024、2048 和 4096。
 * @return CUDA 状态码；参数、分配或 CUDA 操作失败时返回相应错误。
 */
cudaError_t create_fft_plan(FFTPlan** plan, int fft_size) {
    // 避免覆盖已有 plan，并拒绝当前 kernel 不支持的变换点数。
    if (plan == nullptr || *plan != nullptr ||
        !is_supported_fft_size(fft_size)) {
        return cudaErrorInvalidValue;
    }

    // 记录创建时的设备，后续执行和销毁都要求切换回该设备。
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    std::unique_ptr<FFTPlan> new_plan(new (std::nothrow) FFTPlan{});
    if (!new_plan) {
        return cudaErrorMemoryAllocation;
    }

    // 在主机上预计算正向旋转因子，逆向变换时通过取共轭复用该表。
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

    // 将旋转因子表上传到当前设备；上传失败时释放已分配的设备内存。
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

    // 按变换规模设置默认 block 大小，并在初始化完成后才交付 plan 句柄。
    new_plan->fft_size = fft_size;
    new_plan->thread_count =
        fft_size >= 1024 ? 1024 : (fft_size == 512 ? 512 : kFFTThreads256);
    new_plan->device = device;
    *plan = new_plan.release();
    return cudaSuccess;
}

/**
 * @brief 释放 plan 持有的设备端旋转因子表，并销毁 plan。
 * @param plan 指向有效 plan 句柄的指针；成功后句柄置为 nullptr。
 * @return CUDA 状态码；当前设备与 plan 创建设备不同时返回错误。
 */
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

/**
 * @brief 使用 plan 中记录的默认线程配置启动一次 FFT。
 * @param plan 已创建且当前设备匹配的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 用于保存设备端复数输出的缓冲区。
 * @param inverse 是否执行不进行 1/N 归一化的逆向变换。
 * @return CUDA 状态码；参数无效或设备不匹配时返回相应错误。
 */
cudaError_t launch_fft_plan(const FFTPlan* plan, const float2* input,
                            float2* output, bool inverse) {
    if (plan == nullptr) {
        return cudaErrorInvalidValue;
    }
    return launch_fft_plan_config(plan, input, output, plan->thread_count,
                                  inverse);
}

/**
 * @brief 使用指定线程数启动一次 FFT。
 * @param plan 已创建且当前设备匹配的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 用于保存设备端复数输出的缓冲区。
 * @param thread_count 每个 CUDA block 的线程数。
 * @param inverse 是否执行不进行 1/N 归一化的逆向变换。
 * @return CUDA 状态码；参数、设备或线程配置无效时返回相应错误。
 */
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

/**
 * @brief 并行启动一批独立 FFT，每个 block 处理一个变换。
 * @param plan 已创建且当前设备匹配的 FFT plan。
 * @param input 连续存放 batch_count 个输入变换的设备缓冲区。
 * @param output 用于保存连续批量结果的设备缓冲区。
 * @param batch_count 批次中的变换数量，必须大于零。
 * @param inverse 是否执行不进行 1/N 归一化的逆向变换。
 * @return CUDA 状态码；参数无效或设备不匹配时返回相应错误。
 */
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

cudaError_t create_fft_plan_fp64(FFTPlanFp64** plan, int fft_size) {
    if (plan == nullptr || *plan != nullptr ||
        !is_supported_fft_size(fft_size)) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (fft_size == 4096) {
        constexpr int kSharedBytes = 4096 * sizeof(double2);
        status = cudaFuncSetAttribute(
            fft_radix2_decimation_in_frequency_fp64<4096, 256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        if (status != cudaSuccess) {
            return status;
        }
        status = cudaFuncSetAttribute(
            fft_radix2_decimation_in_frequency_fp64<4096, 512>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        if (status != cudaSuccess) {
            return status;
        }
        status = cudaFuncSetAttribute(
            fft_radix2_decimation_in_frequency_fp64<4096, 1024>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        if (status != cudaSuccess) {
            return status;
        }
    }

    std::unique_ptr<FFTPlanFp64> new_plan(new (std::nothrow) FFTPlanFp64{});
    if (!new_plan) {
        return cudaErrorMemoryAllocation;
    }

    std::vector<double2> host_twiddle_factors;
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
            make_double2(std::cos(angle), std::sin(angle));
    }

    status = cudaMalloc(&new_plan->twiddle_factors,
                        host_twiddle_factors.size() * sizeof(double2));
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemcpy(new_plan->twiddle_factors,
                        host_twiddle_factors.data(),
                        host_twiddle_factors.size() * sizeof(double2),
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

cudaError_t destroy_fft_plan_fp64(FFTPlanFp64** plan) {
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

cudaError_t launch_fft_plan_fp64(const FFTPlanFp64* plan,
                                 const double2* input, double2* output,
                                 bool inverse) {
    if (plan == nullptr) {
        return cudaErrorInvalidValue;
    }
    return launch_fft_plan_config_fp64(plan, input, output,
                                       plan->thread_count, inverse);
}

cudaError_t launch_fft_plan_config_fp64(const FFTPlanFp64* plan,
                                        const double2* input,
                                        double2* output, int thread_count,
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
    return launch_fft_plan_config_fp64_impl(
        input, output, plan->fft_size, thread_count, 1, inverse,
        plan->twiddle_factors);
}

cudaError_t launch_fft_plan_batch_fp64(const FFTPlanFp64* plan,
                                       const double2* input, double2* output,
                                       int batch_count, bool inverse) {
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
    return launch_fft_plan_config_fp64_impl(
        input, output, plan->fft_size, plan->thread_count, batch_count,
        inverse, plan->twiddle_factors);
}
