# CUDA radix-2 FFT

This project implements forward, single-precision, complex-to-complex radix-2
FFTs for 256, 512, 1024, 2048, and 4096 points. The kernels use a 256-thread
block, shared memory for cross-warp butterfly stages, and warp shuffles plus
registers for the final stages. Twiddle factors are computed on the fly. The
custom implementation does not use cuFFT or cuBLAS.

The test executable compares each custom FFT against cuFFT C2C for impulse,
shifted impulse, constant, single-tone, and deterministic random inputs. It
then reports custom-kernel and cuFFT execution times after warmup. Timing
excludes host-device transfers and cuFFT plan creation.

## Build and run

```sh
cmake -S . -B build
cmake --build build -j
./build/fft256_benchmark
```

Run the same correctness checks and benchmark through CTest:

```sh
ctest --test-dir build --output-on-failure
```

The default CUDA target is `sm_89` (GeForce RTX 4090). Override it with
`-DCMAKE_CUDA_ARCHITECTURES=<architecture>` when building for another GPU.
