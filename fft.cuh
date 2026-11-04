#pragma once

#include <cuda_runtime.h>

constexpr int kFFTSize256 = 256;
constexpr int kFFTThreads256 = kFFTSize256;

struct FFTPlan;

/**
 * @brief 创建可复用的 FFT plan，并在设备端分配和初始化旋转因子表。
 * @param plan 接收 plan 的指针；必须指向 nullptr。
 * @param fft_size FFT 点数，支持 256、512、1024、2048 和 4096。
 * @return CUDA 状态码。
 */
cudaError_t create_fft_plan(FFTPlan** plan, int fft_size);

/**
 * @brief 释放 FFT plan 持有的设备端旋转因子表并将句柄置空。
 * @param plan 指向有效 FFT plan 句柄的指针。
 * @return CUDA 状态码。
 */
cudaError_t destroy_fft_plan(FFTPlan** plan);

/**
 * @brief 使用 plan 的默认线程配置执行 FFT。
 * @param plan 已创建的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param inverse 是否执行反向变换；反向变换不做 1/N 归一化。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan(const FFTPlan* plan, const float2* input,
                            float2* output, bool inverse = false);

/**
 * @brief 使用 plan 和指定线程数执行 FFT。
 * @param plan 已创建的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param thread_count 单个线程块的线程数。
 * @param inverse 是否执行反向变换；反向变换不做 1/N 归一化。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan_config(const FFTPlan* plan, const float2* input,
                                   float2* output, int thread_count,
                                   bool inverse = false);

/**
 * @brief 使用一个 CUDA block 并行执行批量独立 FFT。
 * @param plan 已创建的 FFT plan。
 * @param input 指向连续存放的 batch_count 个输入变换。
 * @param output 用于保存连续存放的 batch_count 个输出变换。
 * @param batch_count 批次中的变换数量，必须大于零。
 * @param inverse 是否执行反向变换；反向变换不做 1/N 归一化。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan_batch(const FFTPlan* plan, const float2* input,
                                  float2* output, int batch_count,
                                  bool inverse = false);

/**
 * @brief 使用 plan 的默认线程配置重复执行 FFT 并测量平均耗时。
 * @param plan 已创建的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param iterations kernel 启动次数，必须大于零。
 * @param start 用于计时的起始 CUDA event。
 * @param stop 用于计时的结束 CUDA event。
 * @param average_us 用于接收单次平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码。
 */
cudaError_t benchmark_fft_plan_iterations(const FFTPlan* plan,
                                          const float2* input, float2* output,
                                          int iterations, cudaEvent_t start,
                                          cudaEvent_t stop, float* average_us,
                                          bool inverse = false);

/**
 * @brief 使用 plan 和指定线程配置重复执行 FFT 并测量平均耗时。
 * @param plan 已创建的 FFT plan。
 * @param input 指向设备端复数输入数据。
 * @param output 指向设备端复数输出缓冲区。
 * @param thread_count 单个线程块的线程数。
 * @param iterations kernel 启动次数，必须大于零。
 * @param start 用于计时的起始 CUDA event。
 * @param stop 用于计时的结束 CUDA event。
 * @param average_us 用于接收单次平均耗时，单位为微秒。
 * @param inverse 是否执行反向变换。
 * @return CUDA 状态码。
 */
cudaError_t benchmark_fft_plan_config(const FFTPlan* plan,
                                      const float2* input, float2* output,
                                      int thread_count, int iterations,
                                      cudaEvent_t start, cudaEvent_t stop,
                                      float* average_us, bool inverse = false);
