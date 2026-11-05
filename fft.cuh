#pragma once

#include <cuda_runtime.h>

constexpr int kFFTSize256 = 256;
constexpr int kFFTThreads256 = kFFTSize256;

struct FFTPlan;
struct FFTPlanFp64;

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
 * @brief 为批次中的每个独立 FFT 启动一个 CUDA block 并行执行。
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
 * @brief 创建可复用的双精度复数 FFT plan。
 * @param plan 接收 plan 的指针；必须指向 nullptr。
 * @param fft_size FFT 点数，支持 256、512、1024、2048 和 4096。
 * @return CUDA 状态码。
 */
cudaError_t create_fft_plan_fp64(FFTPlanFp64** plan, int fft_size);

/**
 * @brief 释放双精度 FFT plan 并将句柄置空。
 * @param plan 指向有效 FFT plan 句柄的指针。
 * @return CUDA 状态码。
 */
cudaError_t destroy_fft_plan_fp64(FFTPlanFp64** plan);

/**
 * @brief 使用 plan 的默认线程配置执行双精度 FFT。
 * @param plan 已创建的双精度 FFT plan。
 * @param input 指向设备端 double2 复数输入数据。
 * @param output 指向设备端 double2 复数输出缓冲区。
 * @param inverse 是否执行未归一化的反向变换。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan_fp64(const FFTPlanFp64* plan,
                                 const double2* input, double2* output,
                                 bool inverse = false);

/**
 * @brief 使用指定线程数执行双精度 FFT。
 * @param plan 已创建的双精度 FFT plan。
 * @param input 指向设备端 double2 复数输入数据。
 * @param output 指向设备端 double2 复数输出缓冲区。
 * @param thread_count 单个线程块的线程数。
 * @param inverse 是否执行未归一化的反向变换。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan_config_fp64(const FFTPlanFp64* plan,
                                        const double2* input,
                                        double2* output, int thread_count,
                                        bool inverse = false);

/**
 * @brief 并行执行一批双精度 FFT，每个 block 处理一个变换。
 * @param plan 已创建的双精度 FFT plan。
 * @param input 连续存放 batch_count 个输入变换。
 * @param output 用于保存连续批量结果的缓冲区。
 * @param batch_count 批次中的变换数量，必须大于零。
 * @param inverse 是否执行未归一化的反向变换。
 * @return CUDA 状态码。
 */
cudaError_t launch_fft_plan_batch_fp64(const FFTPlanFp64* plan,
                                       const double2* input, double2* output,
                                       int batch_count, bool inverse = false);
