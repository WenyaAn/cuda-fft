#include "fft256.cuh"

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

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "%s failed with cuFFT status %d\n", operation,
                     static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

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

float benchmark_cufft(cufftHandle plan,
                      cufftComplex* device_input,
                      cufftComplex* device_output,
                      cudaEvent_t start,
                      cudaEvent_t stop) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        check_cufft(cufftExecC2C(plan, device_input, device_output, CUFFT_FORWARD),
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

void print_timing_stats(const char* label, const TimingStats& stats) {
    std::printf("  %-12s mean %7.3f us; median %7.3f us; stddev %6.3f us; "
                "min/max %.3f/%.3f us\n",
                label, stats.mean, stats.median, stats.standard_deviation,
                stats.minimum, stats.maximum);
}

}  // namespace

int main() {
    int device = 0;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    std::printf("GPU: %s; custom FFT launch: 1 block x %d threads\n",
                properties.name, kFftThreads256);
    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const std::vector<int> fft_sizes = {256, 512, 1024, 2048, 4096};
    const std::vector<std::string> test_names = {
        "impulse", "shifted impulse", "constant", "single tone", "random"};
    for (int fft_size : fft_sizes) {
        std::printf("\n=== %d-point C2C FFT ===\n", fft_size);
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
            check_cuda(launch_fft(device_input, custom_output, fft_size),
                       "custom FFT launch");
            check_cuda(cudaGetLastError(), "custom FFT launch");
            check_cufft(cufftExecC2C(
                            plan, reinterpret_cast<cufftComplex*>(device_input),
                            cufft_output, CUFFT_FORWARD),
                        "cufftExecC2C");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            check_cuda(cudaMemcpy(host_custom.data(), custom_output, bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(custom output)");
            check_cuda(cudaMemcpy(host_cufft.data(), cufft_output, bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(cuFFT output)");
            compare_outputs(host_custom, host_cufft,
                            std::to_string(fft_size) + " " + test_name);
        }

        for (int i = 0; i < kWarmupIterations; ++i) {
            check_cuda(launch_fft(device_input, custom_output, fft_size),
                       "custom FFT warmup launch");
            check_cuda(cudaGetLastError(), "custom FFT warmup launch");
            check_cufft(cufftExecC2C(
                            plan, reinterpret_cast<cufftComplex*>(device_input),
                            cufft_output, CUFFT_FORWARD),
                        "cuFFT warmup");
        }
        check_cuda(cudaDeviceSynchronize(), "benchmark warmup synchronization");

        std::vector<float> custom_samples;
        std::vector<float> cufft_samples;
        custom_samples.reserve(kBenchmarkSamples);
        cufft_samples.reserve(kBenchmarkSamples);
        for (int sample = 0; sample < kBenchmarkSamples; ++sample) {
            if (sample % 2 == 0) {
                float custom_us = 0.0f;
                check_cuda(benchmark_fft_iterations(
                               device_input, custom_output, fft_size,
                               kBenchmarkIterations, start, stop, &custom_us),
                           "custom FFT benchmark");
                custom_samples.push_back(custom_us);
                cufft_samples.push_back(benchmark_cufft(
                    plan, reinterpret_cast<cufftComplex*>(device_input),
                    cufft_output, start, stop));
            } else {
                cufft_samples.push_back(benchmark_cufft(
                    plan, reinterpret_cast<cufftComplex*>(device_input),
                    cufft_output, start, stop));
                float custom_us = 0.0f;
                check_cuda(benchmark_fft_iterations(
                               device_input, custom_output, fft_size,
                               kBenchmarkIterations, start, stop, &custom_us),
                           "custom FFT benchmark");
                custom_samples.push_back(custom_us);
            }
        }

        const TimingStats custom_stats = summarize_timings(custom_samples);
        const TimingStats cufft_stats = summarize_timings(cufft_samples);
        std::printf(
            "Performance (%d samples x %d FFTs/sample, transfers excluded):\n",
            kBenchmarkSamples, kBenchmarkIterations);
        print_timing_stats("Custom FFT", custom_stats);
        print_timing_stats("cuFFT C2C", cufft_stats);
        std::printf("  cuFFT/custom mean-time ratio: %.2fx\n",
                    cufft_stats.mean / custom_stats.mean);

        check_cufft(cufftDestroy(plan), "cufftDestroy");
        check_cuda(cudaFree(device_input), "cudaFree(input)");
        check_cuda(cudaFree(custom_output), "cudaFree(custom output)");
        check_cuda(cudaFree(cufft_output), "cudaFree(cuFFT output)");
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return EXIT_SUCCESS;
}
