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

std::vector<float2> make_input(const std::string& test_name) {
    std::vector<float2> input(kFftSize, make_float2(0.0f, 0.0f));
    if (test_name == "impulse") {
        input[0] = make_float2(1.0f, 0.0f);
    } else if (test_name == "shifted impulse") {
        input[37] = make_float2(1.0f, -0.5f);
    } else if (test_name == "constant") {
        std::fill(input.begin(), input.end(), make_float2(0.25f, -0.75f));
    } else if (test_name == "single tone") {
        for (int i = 0; i < kFftSize; ++i) {
            const float angle = 6.2831853071795864769f * 13.0f *
                                static_cast<float>(i) / kFftSize;
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
    for (int i = 0; i < kFftSize; ++i) {
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
                         "%s mismatch at bin %d: error %.7g exceeds %.7g\n",
                         test_name.c_str(), i, error, tolerance);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("PASS %-16s max error %.7g\n", test_name.c_str(), max_error);
    return max_error;
}

float benchmark_custom_kernel(const float2* device_input,
                              float2* device_output,
                              cudaEvent_t start,
                              cudaEvent_t stop) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        fft256_cooley_tukey<<<1, kFftThreads>>>(device_input, device_output);
    }
    check_cuda(cudaGetLastError(), "custom FFT warmup launch");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        fft256_cooley_tukey<<<1, kFftThreads>>>(device_input, device_output);
    }
    check_cuda(cudaGetLastError(), "custom FFT benchmark launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime");
    return elapsed_ms * 1000.0f / kBenchmarkIterations;
}

float benchmark_cufft(cufftHandle plan,
                      cufftComplex* device_input,
                      cufftComplex* device_output,
                      cudaEvent_t start,
                      cudaEvent_t stop) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        check_cufft(cufftExecC2C(plan, device_input, device_output, CUFFT_FORWARD),
                    "cuFFT warmup");
    }
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

}  // namespace

int main() {
    int device = 0;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    std::printf("GPU: %s; FFT: %d points; custom launch: 1 block x %d threads\n",
                properties.name, kFftSize, kFftThreads);

    float2* device_input = nullptr;
    float2* custom_output = nullptr;
    cufftComplex* cufft_output = nullptr;
    const size_t bytes = kFftSize * sizeof(float2);
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
    check_cuda(cudaMalloc(&custom_output, bytes), "cudaMalloc(custom output)");
    check_cuda(cudaMalloc(&cufft_output, bytes), "cudaMalloc(cuFFT output)");

    cufftHandle plan;
    check_cufft(cufftPlan1d(&plan, kFftSize, CUFFT_C2C, 1), "cufftPlan1d");
    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const std::vector<std::string> test_names = {
        "impulse", "shifted impulse", "constant", "single tone", "random"};
    std::vector<float2> host_custom(kFftSize);
    std::vector<float2> host_cufft(kFftSize);
    for (const std::string& test_name : test_names) {
        const std::vector<float2> host_input = make_input(test_name);
        check_cuda(cudaMemcpy(device_input, host_input.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(input)");
        fft256_cooley_tukey<<<1, kFftThreads>>>(device_input, custom_output);
        check_cuda(cudaGetLastError(), "custom FFT launch");
        check_cufft(cufftExecC2C(plan,
                                 reinterpret_cast<cufftComplex*>(device_input),
                                 cufft_output, CUFFT_FORWARD),
                    "cufftExecC2C");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        check_cuda(cudaMemcpy(host_custom.data(), custom_output, bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(custom output)");
        check_cuda(cudaMemcpy(host_cufft.data(), cufft_output, bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(cuFFT output)");
        compare_outputs(host_custom, host_cufft, test_name);
    }

    const float custom_us =
        benchmark_custom_kernel(device_input, custom_output, start, stop);
    const float cufft_us =
        benchmark_cufft(plan, reinterpret_cast<cufftComplex*>(device_input),
                        cufft_output, start, stop);
    std::printf("Average execution time (%d iterations, transfers excluded):\n",
                kBenchmarkIterations);
    std::printf("  Custom Cooley-Tukey: %.3f us\n", custom_us);
    std::printf("  cuFFT C2C:           %.3f us\n", cufft_us);
    std::printf("  cuFFT/custom ratio:  %.2fx\n", cufft_us / custom_us);

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cufft(cufftDestroy(plan), "cufftDestroy");
    check_cuda(cudaFree(device_input), "cudaFree(input)");
    check_cuda(cudaFree(custom_output), "cudaFree(custom output)");
    check_cuda(cudaFree(cufft_output), "cudaFree(cuFFT output)");
    return EXIT_SUCCESS;
}
