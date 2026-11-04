# CUDA FFT Performance

## Test environment

- Date: 2026-11-04
- GPU: NVIDIA GeForce RTX 4090 (benchmark ran on CUDA device 0)
- Driver: 590.44.01
- CUDA runtime reported by `nvidia-smi`: 13.1
- Precision and transform: single-precision complex-to-complex (C2C)
- Sizes: 256, 512, 1024, 2048, and 4096 points

## Method

Results below were collected by building and running `fft_benchmark` in this
repository on 2026-11-04. The program performs 100 warmup launches, then
collects 100 timing samples of 5,000 transforms each. CUDA events measure
device execution; host-device transfers, custom FFT plan creation, and cuFFT
plan creation are excluded. The program alternates which implementation runs
first between samples. Times in the table are the benchmark's median
per-transform times, to limit the impact of occasional GPU scheduling outliers.

The local FFT uses a reusable `FFTPlan` and one thread block. The plan stores
the transform size, default thread count, CUDA device, and a precomputed
device-side twiddle table. Each block copies the table into shared memory once
before executing the FFT. The default thread count is 256 for 256 points, 512
for 512 points, and 1024 for 1024 points and above. cuFFT uses a single
`CUFFT_C2C` plan. Forward and inverse results are reported separately; inverse
transforms are unnormalized.

## Results

All times are in microseconds per transform. The last column is
`cuFFT median / planned FFT median`; values below 1 mean cuFFT was faster.

| Points | Direction | Planned FFT | cuFFT | cuFFT / plan |
|---:|---|---:|---:|---:|
| 256 | Forward | 3.074 | 3.064 | 1.00x |
| 256 | Inverse | 3.061 | 3.063 | 1.00x |
| 512 | Forward | 3.388 | 2.278 | 0.67x |
| 512 | Inverse | 3.344 | 2.278 | 0.68x |
| 1024 | Forward | 5.945 | 2.702 | 0.45x |
| 1024 | Inverse | 5.946 | 2.703 | 0.45x |
| 2048 | Forward | 9.785 | 3.555 | 0.36x |
| 2048 | Inverse | 9.820 | 3.555 | 0.36x |
| 4096 | Forward | 17.299 | 4.194 | 0.24x |
| 4096 | Inverse | 17.353 | 4.194 | 0.24x |

The benchmark's correctness checks passed for all tested sizes and both
directions, including impulse, shifted impulse, constant, single-tone, random,
and forward/inverse round-trip cases.

## Batched throughput

The batch API launches one block per independent transform. The benchmark
compares a contiguous batch of 1024 transforms against 1024 sequential
`cufftExecC2C` calls using a single-transform cuFFT 1D plan; it does not use
`cufftPlanMany`. Plan creation and host-device transfers are excluded.
Correctness is checked for every element in the batch in both directions. The
table shows forward medians from 30 samples. Each custom sample averages 100
batch launches; each cuFFT sample executes the single-transform plan once for
each of the 1024 inputs. Per-transform time is batch time divided by 1024.

| Points | Custom batch (us/batch) | Custom batch (us/FFT) | Sequential cuFFT plan (us/batch) | Sequential cuFFT plan (us/FFT) | Custom batch speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 6.267 | 0.0061 | 2319.360 | 2.2650 | 370.10x |
| 512 | 20.992 | 0.0205 | 2355.200 | 2.3000 | 112.20x |
| 1024 | 52.500 | 0.0513 | 2753.552 | 2.6890 | 52.45x |
| 2048 | 103.096 | 0.1007 | 3647.488 | 3.5620 | 35.38x |
| 4096 | 204.052 | 0.1993 | 4322.816 | 4.2215 | 21.18x |

This is a comparison of execution strategies for a batch, not just FFT kernel
arithmetic: the custom implementation launches one grid for the full batch,
whereas the cuFFT plan is called once per transform. CUDA-event timings include
the GPU timeline gaps caused by sequential submissions, so the custom speedup
primarily reflects avoiding thousands of individual launches. It must not be
interpreted as the custom FFT kernel being hundreds of times faster than cuFFT.
For a comparison of batched kernel throughput with launch overhead amortized on
both sides, use a cuFFT batched plan. Inverse measurements were within 1% of
the forward timings.

## Notes

- At 256 points, the local kernel and cuFFT have similar measured latency.
- For the larger tested sizes, cuFFT is faster in this run; the gap grows with
  transform size.
- Plan setup is excluded. The plan path stages its precomputed table from global
  memory to shared memory once per FFT block, then uses shared-memory lookups
  during butterfly stages. This version removes runtime trigonometric
  computation, but the table and FFT data compete for shared-memory capacity;
  cuFFT remains substantially faster for larger sizes. Against the previous
  global-table lookup version, shared staging improved the measured 4096-point
  median by roughly 3%; it did not close the gap to cuFFT.
- These are measurements of repeated, single-transform kernel execution on one
  GPU, not end-to-end application throughput. Results can vary with GPU clocks,
  system load, CUDA/cuFFT versions, and workload shape. Re-run the benchmark on
  the target system before using these figures for performance decisions.

To reproduce:

```sh
cmake -S . -B build
cmake --build build -j
./build/fft_benchmark
```
