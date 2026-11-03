#pragma once

#include <cuda_runtime.h>

constexpr int kFftSize256 = 256;
constexpr int kFftThreads256 = kFftSize256;

/**
 * @brief 执行 256 点基 2 频率抽取复数 FFT。
 * @param input 指向 256 个复数输入元素的设备内存。
 * @param output 用于保存 256 个复数输出元素的设备内存。
 * @param inverse 为 true 时执行反向变换；反变换不进行 1/N 归一化。
 * @return 无；CUDA 启动错误通过 CUDA Runtime 查询。
 */
__global__ void fft256_decimation_in_frequency(const float2* input,
                                              float2* output, bool inverse);

/**
 * @brief 按点数选择默认线程配置并启动 FFT kernel。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param fft_size FFT 点数，支持 256、512、1024、2048 和 4096。
 * @param inverse 是否执行反向变换；反向变换不做归一化。
 * @return CUDA 状态码；成功返回 cudaSuccess，不支持的点数返回错误。
 */
cudaError_t launch_fft(const float2* input, float2* output, int fft_size,
                       bool inverse = false);

/**
 * @brief 使用指定线程数启动对应点数的 FFT kernel。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param fft_size FFT 点数。
 * @param thread_count 单个线程块的线程数。
 * @param inverse 是否执行反向变换；反向变换不做归一化。
 * @return CUDA 状态码；点数或线程配置不受支持时返回错误。
 */
cudaError_t launch_fft_config(const float2* input, float2* output, int fft_size,
                              int thread_count, bool inverse = false);

/**
 * @brief 按点数默认配置重复启动 FFT 并测量平均执行时间。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param fft_size FFT 点数。
 * @param iterations kernel 启动次数，必须大于零。
 * @param start 用于计时的起始 CUDA event。
 * @param stop 用于计时的结束 CUDA event。
 * @param average_us 用于接收单次平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换；反向变换不做归一化。
 * @return CUDA 状态码。
 */
cudaError_t benchmark_fft_iterations(const float2* input, float2* output,
                                    int fft_size, int iterations,
                                    cudaEvent_t start, cudaEvent_t stop,
                                    float* average_us, bool inverse = false);

/**
 * @brief 使用指定线程配置重复启动 FFT 并测量平均执行时间。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param fft_size FFT 点数。
 * @param thread_count 单个线程块的线程数。
 * @param iterations kernel 启动次数，必须大于零。
 * @param start 用于计时的起始 CUDA event。
 * @param stop 用于计时的结束 CUDA event。
 * @param average_us 用于接收单次平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换；反向变换不做归一化。
 * @return CUDA 状态码。
 */
cudaError_t benchmark_fft_config(const float2* input, float2* output,
                                 int fft_size, int thread_count, int iterations,
                                 cudaEvent_t start, cudaEvent_t stop,
                                 float* average_us, bool inverse = false);
