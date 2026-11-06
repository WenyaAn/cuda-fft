# CUDA Radix-2 FFT

本项目实现了单精度（FP32）和双精度（FP64）复数到复数（C2C）的 Radix-2
正向和逆向 FFT，支持 256、512、1024、2048 和 4096 点。默认配置下，256 点
使用 256 个线程，512 点使用 512 个线程，更大尺寸使用 1024 个线程。跨 warp
的蝶形阶段使用共享内存，最后几个阶段通过 warp shuffle 和寄存器完成。

可复用的 `FFTPlanFp32` 保存 FFT 点数、默认线程配置和 CUDA 设备信息。创建 plan
时会在 GPU 上分配并上传预计算的旋转因子表。每个 FFT block 会先将旋转因子表
载入共享内存，并在各蝶形阶段中重复使用。请在创建 plan 时对应的 CUDA 设备
上执行和销毁该 plan。FFT 执行使用基于 plan 的 API。

FFT kernel 直接使用 CUDA Runtime API 进行设备管理、内存操作和 kernel 启动；没有
调用 cuFFT、cuBLAS、cuRAND 等 CUDA 数学或计算库，也没有直接调用 CUDA Driver
API。只有测试和基准程序使用 cuFFT，作为正确性参考和性能对比；cuBLAS、
cuRAND 等库不参与本项目的构建和运行。

逆向变换遵循 cuFFT 的非归一化约定：对输入先执行正向 FFT，再执行逆向 FFT，
结果为原始输入乘以 FFT 点数。测试程序会将 FP32 和 FP64 的正向、逆向及批量
结果分别与 cuFFT C2C 对比，并检查正逆变换的往返结果。性能基准覆盖 FP32 和
FP64 的单次及批量变换，并与 cuFFT 对比；计时不包括数据传输和 plan 创建；
具体测量配置和结果见性能报告。

当前测试覆盖的所有点数和两个变换方向均通过正确性检查。在 RTX 4090 上，FP32
单次变换中 256 点自定义 FFT 与 cuFFT 延迟接近，512 点及以上 cuFFT 更快；
FP64 单次变换中 256 点两者接近，512 点及以上 cuFFT 约快 1.6–1.7 倍。对 FP32
和 FP64，cuFFT PlanMany 批量变换均快于自定义批量实现。以上性能结论仅适用于
报告中的测试环境和方法；详细数据见 [PERFORMANCE.md](./PERFORMANCE.md)。

## 精度误差

下表记录 `fft_benchmark` 一次运行中 FP32 自定义 FFT 与 cuFFT 的最大复数绝对误差；
每个正向/逆向单元格中，数值按“正向 / 逆向”排列。单次测试覆盖冲激、移位冲激、
常数、单频正弦和随机输入；线程配置列汇总该尺寸所有受支持配置的误差。批量误差
比较自定义批量 FFT 与 cuFFT PlanMany。往返误差比较自定义正向再逆向的结果与
`N·x`（逆向变换不归一化）。

| 点数 | 单次 FFT / cuFFT（正向 / 逆向） | 线程配置 FFT / cuFFT（正向 / 逆向） | 批量 FFT / PlanMany（正向 / 逆向） | 往返 FFT / `N·x` |
|---:|---:|---:|---:|---:|
| 256 | 1.72e-5 / 1.54e-5 | 1.38e-3 / 1.12e-3 | 9.55e-6 / 9.00e-6 | 6.87e-5 |
| 512 | 1.83e-5 / 5.64e-5 | 4.56e-3 / 4.63e-3 | 1.64e-5 / 1.45e-5 | 1.64e-4 |
| 1024 | 2.76e-5 / 1.55e-4 | 1.57e-2 / 1.61e-2 | 2.44e-5 / 2.67e-5 | 4.49e-4 |
| 2048 | 6.95e-5 / 2.06e-4 | 6.03e-2 / 4.47e-2 | 3.96e-5 / 3.48e-5 | 8.19e-4 |
| 4096 | 8.79e-5 / 4.26e-4 | 1.40e-1 / 1.33e-1 | 5.29e-5 / 5.42e-5 | 1.97e-3 |

误差定义为复数差的模；逐元素检查条件为 `error ≤ 1e-3 + 2e-4 × |reference|`。
以上比较以及所有正向、逆向和往返检查均通过。cuFFT PlanMany 与逐个调用单变换
cuFFT plan 的比较在本次运行中两个方向、所有尺寸的最大误差均为 `0`。实际数值
可能随 GPU、CUDA/cuFFT 版本和测试环境变化。

FP64 使用固定种子的随机复数输入与 cuFFT `CUFFT_Z2Z` 比较；线程配置列汇总
该尺寸所有受支持配置中的最大值。批量误差比较自定义批量 FFT 与 cuFFT PlanMany。
往返误差比较自定义正向再逆向结果与 `N·x`。

| 点数 | 单次 FFT / cuFFT（正向 / 逆向） | 线程配置 FFT / cuFFT（正向 / 逆向） | 批量 FFT / PlanMany（正向 / 逆向） | 往返 FFT / `N·x` |
|---:|---:|---:|---:|---:|
| 256 | 1.32e-14 / 1.24e-14 | 1.32e-14 / 1.24e-14 | 2.56e-14 / 2.42e-14 | 1.91e-13 |
| 512 | 2.86e-14 / 2.25e-14 | 2.86e-14 / 2.25e-14 | 3.84e-14 / 4.16e-14 | 4.86e-13 |
| 1024 | 4.55e-14 / 5.00e-14 | 4.55e-14 / 5.00e-14 | 7.11e-14 / 6.55e-14 | 7.28e-13 |
| 2048 | 6.55e-14 / 6.04e-14 | 6.55e-14 / 6.04e-14 | 1.03e-13 / 1.02e-13 | 1.84e-12 |
| 4096 | 1.11e-13 / 1.29e-13 | 1.11e-13 / 1.29e-13 | 1.88e-13 / 1.71e-13 | 4.31e-12 |

FP64 的误差为复数差的模，逐元素检查条件为 `error ≤ 1e-10 + 1e-12 × |reference|`。
所有 FP64 正向、逆向、线程配置、批量和往返检查均通过。这里的误差数值来自一次
RTX 4090 测试运行，可能随 GPU 和 CUDA/cuFFT 版本变化。

## 代码结构

- `CMakeLists.txt`：配置 CUDA/C++ 构建目标、链接 CUDA Runtime 和 cuFFT，并注册 CTest。
- `fft.cuh`：声明 `FFTPlanFp32` 的创建、销毁、单次执行和批量执行接口。
- `fft.cu`：实现 FFT plan 管理、旋转因子准备，以及不同尺寸的 CUDA FFT kernel。
- `benchmark.cu`：程序入口；运行正确性检查、线程配置测试和性能基准，并使用 cuFFT
  作为参考实现。
- `PERFORMANCE.md`：记录基准测试环境、方法和测量结果。
- `OPTIMIZATION.md`：记录当前实现特点及后续优化方向。

整体调用流程是：`benchmark.cu` 创建并复用 `FFTPlanFp32`，通过 `fft.cuh` 中的 API
调用 `fft.cu` 的 kernel；正确性检查和性能对比由基准程序负责。

## 可复用 FFT plan

每种变换尺寸创建一个 plan，并在正向和逆向变换中复用。Plan 与创建时当前的
CUDA 设备关联；调用和销毁 plan 时应选择同一设备。创建 plan 时会在 GPU 上
分配 `fft_size / 2` 个复数旋转因子，并在主机上预计算正向变换使用的因子。
逆向变换通过取复共轭获得对应的旋转因子。计算前，每个 block 都会将旋转因子
表复制到共享内存；kernel 不需要额外的设备工作区。

原先不使用 plan 的 FFT 启动和基准函数已移除。单精度接口使用 `FFTPlanFp32` 和
`float2` 数据，函数名统一带有 `_fp32` 后缀；双精度接口使用 `FFTPlanFp64`、
`double2` 和 `_fp64` 后缀。创建单精度 plan 后，可调用
`launch_fft_plan_fp32` 或 `launch_fft_plan_config_fp32`：

```cpp
FFTPlanFp32* plan = nullptr;
cudaError_t status = create_fft_plan_fp32(&plan, 2048);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}

status = launch_fft_plan_fp32(plan, device_input, device_output);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}

status = destroy_fft_plan_fp32(&plan);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}
```

双精度接口使用独立的 `FFTPlanFp64` 和 `double2` 缓冲区；销毁、单次执行、
线程配置和批量接口分别为 `destroy_fft_plan_fp64`、`launch_fft_plan_fp64`、
`launch_fft_plan_config_fp64` 和 `launch_fft_plan_batch_fp64`。例如：

```cpp
FFTPlanFp64* fp64_plan = nullptr;
cudaError_t status = create_fft_plan_fp64(&fp64_plan, 2048);
if (status == cudaSuccess) {
    status = launch_fft_plan_fp64(fp64_plan, fp64_input, fp64_output);
}
if (status == cudaSuccess) {
    status = destroy_fft_plan_fp64(&fp64_plan);
}
```

## 批量 FFT

对于多个相互独立且连续存储的变换，可使用批量 API。每个变换占用连续的
`fft_size` 个复数值，kernel 为每个变换启动一个 block：

```cpp
status = launch_fft_plan_batch_fp32(plan, batch_input, batch_output,
                                    batch_count);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}
```

基准程序会检查批量正向和逆向变换，并与 `cufftPlanMany` 及重复调用单变换
cuFFT plan 的结果对比；同时报告 1024 个变换组成的批次的吞吐量。
FP64 单次与批量基准同样对比 cuFFT `CUFFT_Z2Z` / `cufftPlanMany`，结果和
测量配置见 [PERFORMANCE.md](./PERFORMANCE.md)。

## 构建和运行

项目要求 **C++17**。CMake 会为 C++ 编译目标启用 C++17；CUDA `.cu` 源文件也
使用了 C++17 特性（例如 `if constexpr`），因此需要支持 C++17 的 CUDA 编译器。

```sh
cmake -S . -B build
cmake --build build -j
./build/fft_benchmark
```

也可以通过 CTest 运行相同的正确性检查和基准：

```sh
ctest --test-dir build --output-on-failure
```

使用 `-V` 可以查看测试程序的详细输出，包括正确性检查和性能数据：

```sh
ctest --test-dir build --output-on-failure -V
```

默认 CUDA 编译目标为 `sm_89`（GeForce RTX 4090）。如需为其他 GPU 构建，
可通过 `-DCMAKE_CUDA_ARCHITECTURES=<架构编号>` 指定目标架构。
