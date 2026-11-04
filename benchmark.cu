#include "fft.cuh"

#include <cufft.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1.0e-3f;
constexpr float kRelativeTolerance = 2.0e-4f;
constexpr int kWarmupIterations = 100;
constexpr int kBenchmarkIterations = 5000;
constexpr int kBenchmarkSamples = 100;
constexpr int kConfigBenchmarkIterations = 1000;
constexpr int kConfigBenchmarkSamples = 30;

/**
 * @brief 检查 CUDA Runtime 调用结果，失败时输出错误并终止程序。
 * @param status CUDA Runtime 返回的状态码。
 * @param operation 出错的操作名称。
 * @return 无；状态非成功时以失败状态退出进程。
 */
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief 检查 cuFFT 调用结果，失败时输出状态并终止程序。
 * @param status cuFFT 返回的状态码。
 * @param operation 出错的操作名称。
 * @return 无；状态非成功时以失败状态退出进程。
 */
void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "%s failed with cuFFT status %d\n", operation,
                     static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief 构造指定模式的单精度复数测试输入。
 * @param test_name 输入模式名称，如 impulse、constant、single tone 或 random。
 * @param fft_size 输入数组长度。
 * @return 含有 fft_size 个复数值的主机端输入数组。
 */
std::vector<float2> make_input(const std::string& test_name, int fft_size) {
    std::vector<float2> input(fft_size, make_float2(0.0f, 0.0f));
    if (test_name == "impulse") {
        input[0] = make_float2(1.0f, 0.0f);
    } else if (test_name == "shifted impulse") {
        input[37] = make_float2(1.0f, -0.5f);
    } else if (test_name == "constant") {
        std::fill(input.begin(), input.end(), make_float2(0.25f, -0.75f));
    } else if (test_name == "single tone") {
        for (int i = 0; i < fft_size; ++i) {
            const float angle = 6.2831853071795864769f * 13.0f *
                                static_cast<float>(i) / fft_size;
            input[i] = make_float2(std::cos(angle), std::sin(angle));
        }
    } else {
        std::mt19937 generator(123456u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float2& value : input) {
            value = make_float2(distribution(generator), distribution(generator));
        }
    }
    return input;
}

/**
 * @brief 比较两组复数结果，并检查逐元素绝对与相对误差容限。
 * @param actual 待检查的输出结果。
 * @param expected 参考结果。
 * @param test_name 测试名称，用于错误报告。
 * @return 所有元素中的最大复数绝对误差；超出容限时终止程序。
 */
float compare_outputs(const std::vector<float2>& actual,
                      const std::vector<float2>& expected,
                      const std::string& test_name) {
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float dx = actual[i].x - expected[i].x;
        const float dy = actual[i].y - expected[i].y;
        const float error = std::hypot(dx, dy);
        const float reference_magnitude =
            std::hypot(expected[i].x, expected[i].y);
        const float tolerance =
            kAbsoluteTolerance + kRelativeTolerance * reference_magnitude;
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::fprintf(stderr,
                         "%s mismatch at bin %zu: error %.7g exceeds %.7g\n",
                         test_name.c_str(), i, error, tolerance);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("PASS %-16s max error %.7g\n", test_name.c_str(), max_error);
    return max_error;
}

/**
 * @brief 重复执行 cuFFT C2C 变换并测量单次平均耗时。
 * @param plan 已创建的 cuFFT 一维变换计划。
 * @param device_input cuFFT 使用的设备端输入缓冲区。
 * @param device_output cuFFT 写入的设备端输出缓冲区。
 * @param start 计时开始 event。
 * @param stop 计时结束 event。
 * @param direction CUFFT_FORWARD 或 CUFFT_INVERSE。
 * @return 单次变换的平均耗时，单位为微秒；CUDA/cuFFT 失败时终止程序。
 */
float benchmark_cufft(cufftHandle plan,
                      cufftComplex* device_input,
                      cufftComplex* device_output,
                      cudaEvent_t start,
                      cudaEvent_t stop,
                      int direction) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        check_cufft(cufftExecC2C(plan, device_input, device_output, direction),
                    "cuFFT benchmark");
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime");
    return elapsed_ms * 1000.0f / kBenchmarkIterations;
}

struct TimingStats {
    double mean;
    double median;
    double standard_deviation;
    float minimum;
    float maximum;
};

/**
 * @brief 汇总计时样本的均值、中位数、标准差和极值。
 * @param samples 每个元素代表一次批量计时得到的单次平均耗时。
 * @return 包含统计值的 TimingStats；samples 不应为空。
 */
TimingStats summarize_timings(const std::vector<float>& samples) {
    std::vector<float> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    double sum = 0.0;
    for (float sample : samples) {
        sum += sample;
    }
    const double mean = sum / samples.size();

    double squared_difference_sum = 0.0;
    for (float sample : samples) {
        const double difference = sample - mean;
        squared_difference_sum += difference * difference;
    }

    const size_t middle = sorted_samples.size() / 2;
    const double median =
        sorted_samples.size() % 2 == 0
            ? (sorted_samples[middle - 1] + sorted_samples[middle]) / 2.0
            : sorted_samples[middle];
    return {mean,
            median,
            std::sqrt(squared_difference_sum / samples.size()),
            sorted_samples.front(),
            sorted_samples.back()};
}

/**
 * @brief 将一组计时统计结果格式化输出到标准输出。
 * @param label 计时对象名称。
 * @param stats 计时统计结果。
 * @return 无。
 */
void print_timing_stats(const char* label, const TimingStats& stats) {
    std::printf("  %-12s mean %7.3f us; median %7.3f us; stddev %6.3f us; "
                "min/max %.3f/%.3f us\n",
                label, stats.mean, stats.median, stats.standard_deviation,
                stats.minimum, stats.maximum);
}

}

/**
 * @brief 对支持的 FFT 点数执行正确性校验、配置调优和性能比较。
 * @return 所有测试成功时返回 EXIT_SUCCESS；CUDA/cuFFT 调用失败时退出程序。
 */
int main() {
    int device = 0;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    std::printf("GPU: %s; custom FFT launch: 1 block x %d threads\n",
                properties.name, kFFTThreads256);
    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const std::vector<int> fft_sizes = {256, 512, 1024, 2048, 4096};
    const std::vector<std::string> test_names = {
        "impulse", "shifted impulse", "constant", "single tone", "random"};
    for (int fft_size : fft_sizes) {
        std::printf("\n=== %d-point C2C FFT ===\n", fft_size);
        FFTPlan* custom_plan = nullptr;
        check_cuda(create_fft_plan(&custom_plan, fft_size), "create_fft_plan");
        float2* device_input = nullptr;
        float2* custom_output = nullptr;
        cufftComplex* cufft_output = nullptr;
        const size_t bytes = fft_size * sizeof(float2);
        check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
        check_cuda(cudaMalloc(&custom_output, bytes),
                   "cudaMalloc(custom output)");
        check_cuda(cudaMalloc(&cufft_output, bytes),
                   "cudaMalloc(cuFFT output)");

        cufftHandle plan;
        check_cufft(cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1),
                    "cufftPlan1d");
        std::vector<float2> host_custom(fft_size);
        std::vector<float2> host_cufft(fft_size);
        for (const std::string& test_name : test_names) {
            const std::vector<float2> host_input =
                make_input(test_name, fft_size);
            check_cuda(cudaMemcpy(device_input, host_input.data(), bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(input)");
            for (bool inverse : {false, true}) {
                check_cuda(launch_fft_plan(custom_plan, device_input,
                                           custom_output, inverse),
                           "custom FFT launch");
                check_cuda(cudaGetLastError(), "custom FFT launch");
                check_cufft(cufftExecC2C(
                                plan,
                                reinterpret_cast<cufftComplex*>(device_input),
                                cufft_output,
                                inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                            "cufftExecC2C");
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                check_cuda(cudaMemcpy(host_custom.data(), custom_output, bytes,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy(custom output)");
                check_cuda(cudaMemcpy(host_cufft.data(), cufft_output, bytes,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy(cuFFT output)");
                compare_outputs(
                    host_custom, host_cufft,
                    std::to_string(fft_size) + " " +
                        (inverse ? "inverse " : "forward ") + test_name);
            }
        }

        const std::vector<float2> round_trip_input =
            make_input("random", fft_size);
        check_cuda(cudaMemcpy(device_input, round_trip_input.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(round-trip input)");
        check_cuda(launch_fft_plan(custom_plan, device_input, custom_output),
                   "custom forward round-trip launch");
        check_cuda(cudaGetLastError(), "custom forward round-trip launch");
        check_cuda(launch_fft_plan(custom_plan, custom_output, device_input,
                                   true),
                   "custom inverse round-trip launch");
        check_cuda(cudaGetLastError(), "custom inverse round-trip launch");
        check_cuda(cudaDeviceSynchronize(), "round-trip synchronization");
        check_cuda(cudaMemcpy(host_custom.data(), device_input, bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(round-trip output)");
        std::vector<float2> expected_round_trip(fft_size);
        for (int i = 0; i < fft_size; ++i) {
            expected_round_trip[i] =
                make_float2(round_trip_input[i].x * fft_size,
                            round_trip_input[i].y * fft_size);
        }
        compare_outputs(host_custom, expected_round_trip,
                        std::to_string(fft_size) + " round trip (N*x)");

        std::vector<int> thread_counts = {256};
        if (fft_size >= 512) {
            thread_counts.push_back(512);
        }
        if (fft_size >= 1024) {
            thread_counts.push_back(1024);
        }

        std::vector<std::vector<float>> config_samples(thread_counts.size());
        for (size_t config = 0; config < thread_counts.size(); ++config) {
            const int thread_count = thread_counts[config];
            config_samples[config].reserve(kConfigBenchmarkSamples);
            for (bool inverse : {false, true}) {
                check_cufft(cufftExecC2C(
                                plan,
                                reinterpret_cast<cufftComplex*>(device_input),
                                cufft_output,
                                inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                            "cufftExecC2C(config correctness)");
                check_cuda(cudaDeviceSynchronize(),
                           "FFT config reference synchronization");
                check_cuda(cudaMemcpy(host_cufft.data(), cufft_output, bytes,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy(config reference)");
                check_cuda(launch_fft_plan_config(
                               custom_plan, device_input, custom_output,
                               thread_count, inverse),
                           "FFT config correctness launch");
                check_cuda(cudaGetLastError(), "FFT config correctness launch");
                check_cuda(cudaDeviceSynchronize(),
                           "FFT config correctness synchronization");
                check_cuda(cudaMemcpy(host_custom.data(), custom_output, bytes,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy(config output)");
                compare_outputs(
                    host_custom, host_cufft,
                    std::to_string(fft_size) + " " +
                        (inverse ? "inverse " : "forward ") +
                        std::to_string(thread_count) + " threads");
            }
            for (int i = 0; i < kWarmupIterations; ++i) {
                check_cuda(launch_fft_plan_config(
                               custom_plan, device_input, custom_output,
                               thread_count),
                           "FFT config warmup launch");
            }
            check_cuda(cudaGetLastError(), "FFT config warmup launch");
        }
        check_cuda(cudaDeviceSynchronize(), "FFT config warmup synchronization");

        for (int sample = 0; sample < kConfigBenchmarkSamples; ++sample) {
            for (size_t config = 0; config < thread_counts.size(); ++config) {
                float custom_us = 0.0f;
                check_cuda(benchmark_fft_plan_config(
                               custom_plan, device_input, custom_output,
                               thread_counts[config],
                               kConfigBenchmarkIterations, start, stop,
                               &custom_us),
                           "FFT config benchmark");
                config_samples[config].push_back(custom_us);
            }
        }

        size_t fastest_config = 0;
        TimingStats fastest_stats = summarize_timings(config_samples[0]);
        std::printf("Thread configuration tuning (%d samples x %d FFTs):\n",
                    kConfigBenchmarkSamples, kConfigBenchmarkIterations);
        for (size_t config = 0; config < thread_counts.size(); ++config) {
            const TimingStats stats = summarize_timings(config_samples[config]);
            char label[32];
            std::snprintf(label, sizeof(label), "%d threads",
                          thread_counts[config]);
            print_timing_stats(label, stats);
            if (stats.median < fastest_stats.median) {
                fastest_config = config;
                fastest_stats = stats;
            }
        }
        std::printf("  fastest measured: %d threads (median %.3f us)\n",
                    thread_counts[fastest_config], fastest_stats.median);

        for (int i = 0; i < kWarmupIterations; ++i) {
            check_cuda(launch_fft_plan(custom_plan, device_input,
                                       custom_output),
                       "custom FFT warmup launch");
            check_cuda(cudaGetLastError(), "custom FFT warmup launch");
            check_cuda(launch_fft_plan(custom_plan, device_input, custom_output,
                                       true),
                       "custom inverse FFT warmup launch");
            check_cuda(cudaGetLastError(),
                       "custom inverse FFT warmup launch");
            check_cufft(cufftExecC2C(
                            plan, reinterpret_cast<cufftComplex*>(device_input),
                            cufft_output, CUFFT_FORWARD),
                        "cuFFT warmup");
            check_cufft(cufftExecC2C(
                            plan, reinterpret_cast<cufftComplex*>(device_input),
                            cufft_output, CUFFT_INVERSE),
                        "cuFFT inverse warmup");
        }
        check_cuda(cudaDeviceSynchronize(), "benchmark warmup synchronization");

        for (bool inverse : {false, true}) {
            std::vector<float> custom_samples;
            std::vector<float> on_the_fly_samples;
            std::vector<float> cufft_samples;
            custom_samples.reserve(kBenchmarkSamples);
            on_the_fly_samples.reserve(kBenchmarkSamples);
            cufft_samples.reserve(kBenchmarkSamples);
            const int direction =
                inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
            for (int sample = 0; sample < kBenchmarkSamples; ++sample) {
                if (sample % 2 == 0) {
                    float custom_us = 0.0f;
                    float on_the_fly_us = 0.0f;
                    check_cuda(benchmark_fft_plan_iterations(
                                   custom_plan, device_input, custom_output,
                                   kBenchmarkIterations, start, stop,
                                   &custom_us, inverse),
                               "planned FFT benchmark");
                    check_cuda(benchmark_fft_iterations(
                                   device_input, custom_output, fft_size,
                                   kBenchmarkIterations, start, stop,
                                   &on_the_fly_us, inverse),
                               "on-the-fly FFT benchmark");
                    custom_samples.push_back(custom_us);
                    on_the_fly_samples.push_back(on_the_fly_us);
                    cufft_samples.push_back(benchmark_cufft(
                        plan, reinterpret_cast<cufftComplex*>(device_input),
                        cufft_output, start, stop, direction));
                } else {
                    cufft_samples.push_back(benchmark_cufft(
                        plan, reinterpret_cast<cufftComplex*>(device_input),
                        cufft_output, start, stop, direction));
                    float custom_us = 0.0f;
                    float on_the_fly_us = 0.0f;
                    check_cuda(benchmark_fft_iterations(
                                   device_input, custom_output, fft_size,
                                   kBenchmarkIterations, start, stop,
                                   &on_the_fly_us, inverse),
                               "on-the-fly FFT benchmark");
                    check_cuda(benchmark_fft_plan_iterations(
                                   custom_plan, device_input, custom_output,
                                   kBenchmarkIterations, start, stop,
                                   &custom_us, inverse),
                               "planned FFT benchmark");
                    custom_samples.push_back(custom_us);
                    on_the_fly_samples.push_back(on_the_fly_us);
                }
            }

            const TimingStats custom_stats = summarize_timings(custom_samples);
            const TimingStats on_the_fly_stats =
                summarize_timings(on_the_fly_samples);
            const TimingStats cufft_stats = summarize_timings(cufft_samples);
            std::printf(
                "%s performance (%d samples x %d FFTs/sample, transfers "
                "excluded):\n",
                inverse ? "Inverse" : "Forward", kBenchmarkSamples,
                kBenchmarkIterations);
            print_timing_stats("Planned FFT", custom_stats);
            print_timing_stats("On-the-fly", on_the_fly_stats);
            print_timing_stats("cuFFT C2C", cufft_stats);
            std::printf("  cuFFT/custom mean-time ratio: %.2fx\n",
                        cufft_stats.mean / custom_stats.mean);
        }

        check_cufft(cufftDestroy(plan), "cufftDestroy");
        check_cuda(destroy_fft_plan(&custom_plan), "destroy_fft_plan");
        check_cuda(cudaFree(device_input), "cudaFree(input)");
        check_cuda(cudaFree(custom_output), "cudaFree(custom output)");
        check_cuda(cudaFree(cufft_output), "cudaFree(cuFFT output)");
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return EXIT_SUCCESS;
}
