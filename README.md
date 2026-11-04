# CUDA radix-2 FFT

This project implements forward and inverse, single-precision, complex-to-
complex radix-2 FFTs for 256, 512, 1024, 2048, and 4096 points. The 256-point
kernel uses 256 threads, the 512-point kernel uses 512 threads, and larger
sizes use 1024 threads. Shared memory handles cross-warp butterfly stages,
while warp shuffles and registers handle the final stages. A reusable
`FFTPlan` caches the transform size, default thread configuration, and CUDA
device. It uses the same on-the-fly twiddle computation as the legacy
`launch_fft` APIs; precomputed twiddle lookup was benchmarked and was slower
for the tested kernels. The custom implementation does not use cuFFT or
cuBLAS.

The inverse follows cuFFT's unnormalized convention: applying an inverse FFT
after a forward FFT returns the input multiplied by the FFT length. The test
executable compares both directions against cuFFT C2C and checks this
round-trip result. It reports custom-kernel and cuFFT timings after warmup;
timing excludes host-device transfers and cuFFT plan creation.
The custom FFT plan is also created before timing begins, so its setup cost is
excluded.

## Reusable FFT plan

Create one plan per transform size and reuse it for forward and inverse
transforms. Plans are associated with the CUDA device current at creation time;
launch and destroy them with that device selected.

```cpp
FFTPlan* plan = nullptr;
cudaError_t status = create_fft_plan(&plan, 2048);
if (status != cudaSuccess) {
    // Handle the CUDA error.
}

status = launch_fft_plan(plan, device_input, device_output);
if (status != cudaSuccess) {
    // Handle the CUDA error.
}

status = destroy_fft_plan(&plan);
if (status != cudaSuccess) {
    // Handle the CUDA error.
}
```

## Build and run

```sh
cmake -S . -B build
cmake --build build -j
./build/fft_benchmark
```

Run the same correctness checks and benchmark through CTest:

```sh
ctest --test-dir build --output-on-failure
```

The default CUDA target is `sm_89` (GeForce RTX 4090). Override it with
`-DCMAKE_CUDA_ARCHITECTURES=<architecture>` when building for another GPU.
