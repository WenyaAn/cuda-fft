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
first between samples. Times in the table are the benchmark's mean
per-transform times.

The local FFT uses a reusable `FFTPlan` and one thread block. The plan caches
the transform size and default thread count. Its kernel computes twiddle
factors on the fly. The default thread count is 256 for 256 points, 512 for 512
points, and 1024 for 1024 points and above. cuFFT uses a single `CUFFT_C2C`
plan. Forward and inverse results are reported separately; inverse transforms
are unnormalized.

## Results

All times are in microseconds per transform. The benchmark compares the
plan-based API and the legacy on-the-fly API in the same run. The last column
is `cuFFT mean / planned FFT mean`; values below 1 mean cuFFT was faster.

| Points | Direction | Planned FFT | On-the-fly FFT | cuFFT | cuFFT / plan |
|---:|---|---:|---:|---:|---:|
| 256 | Forward | 3.143 | 3.158 | 3.048 | 0.97x |
| 256 | Inverse | 2.423 | 2.426 | 2.411 | 1.00x |
| 512 | Forward | 2.899 | 2.899 | 2.279 | 0.79x |
| 512 | Inverse | 2.899 | 2.899 | 2.279 | 0.79x |
| 1024 | Forward | 3.724 | 3.724 | 2.743 | 0.74x |
| 1024 | Inverse | 3.724 | 3.724 | 2.743 | 0.74x |
| 2048 | Forward | 5.864 | 5.864 | 3.555 | 0.61x |
| 2048 | Inverse | 5.862 | 5.862 | 3.555 | 0.61x |
| 4096 | Forward | 9.937 | 9.937 | 4.248 | 0.43x |
| 4096 | Inverse | 9.937 | 9.938 | 4.250 | 0.43x |

The benchmark's correctness checks passed for all tested sizes and both
directions, including impulse, shifted impulse, constant, single-tone, random,
and forward/inverse round-trip cases.

## Notes

- At 256 points, the local kernel and cuFFT have similar measured latency.
- For the larger tested sizes, cuFFT is faster in this run; the gap grows with
  transform size.
- The plan and on-the-fly results are measured side-by-side. Plan setup is
  excluded; the plan path uses the same kernel as the legacy on-the-fly API.
  Storing a full precomputed twiddle table was slower in testing, so the plan
  intentionally does not retain or upload that table.
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
