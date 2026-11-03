# cuda-fft: 256-point CUDA FFT

This project implements a forward, single-precision, 256-point complex FFT
using an iterative radix-2 Cooley-Tukey decimation-in-time algorithm. The custom
implementation uses CUDA runtime primitives only: one block of exactly 32
threads loads bit-reversed input into shared memory, performs the eight FFT
stages, and writes the result in natural order. It does not use cuFFT, cuBLAS,
or other CUDA libraries.

The test executable compares the custom output with cuFFT C2C for impulse,
shifted impulse, constant, single-tone, and deterministic random inputs. It
then reports average custom-kernel and cuFFT execution times after warmup.
Timing includes kernel/library execution but excludes host-device transfers and
cuFFT plan creation.

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
