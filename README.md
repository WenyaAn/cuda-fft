# CUDA radix-2 FFT

This project implements forward and inverse, single-precision, complex-to-
complex radix-2 FFTs for 256, 512, 1024, 2048, and 4096 points. The 256-point
kernel uses 256 threads, the 512-point kernel uses 512 threads, and larger
sizes use 1024 threads. Shared memory handles cross-warp butterfly stages,
while warp shuffles and registers handle the final stages. A reusable
`FFTPlan` stores the transform size, default thread configuration, and CUDA
device. Plan creation allocates device memory for and uploads a precomputed
twiddle table. Each FFT block stages that table into shared memory once and
reuses it throughout the butterfly stages. Destroy each plan on its associated
device to release the table. FFT execution uses the plan-based API. The custom
implementation does not use cuFFT or cuBLAS.

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
launch and destroy them with that device selected. Plan creation allocates
`fft_size / 2` complex twiddle factors on the GPU and precomputes the forward
factors on the host. Inverse launches use their complex conjugates. Each block
copies the table into shared memory before computing; the kernel needs no
additional device workspace. The former planless FFT launch and benchmark
functions have been removed; create a plan and use `launch_fft_plan` or
`launch_fft_plan_config`.

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

For many independent, contiguous transforms, use the batch API. It launches
one block per transform; the batch contains `batch_count` adjacent arrays of
`fft_size` complex values:

```cpp
status = launch_fft_plan_batch(plan, batch_input, batch_output, batch_count);
if (status != cudaSuccess) {
    // Handle the CUDA error.
}
```

The benchmark checks batched forward and inverse results against both
`cufftPlanMany` and repeated executions of a single-transform cuFFT plan, and
reports throughput for a 1024-transform batch.

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
或
```sh
ctest --test-dir build --output-on-failure -V
```

The default CUDA target is `sm_89` (GeForce RTX 4090). Override it with
`-DCMAKE_CUDA_ARCHITECTURES=<architecture>` when building for another GPU.
