# CUDA Radix-2 FFT

本项目实现了单精度复数到复数（C2C）的 Radix-2 正向和逆向 FFT，支持
256、512、1024、2048 和 4096 点。256 点 kernel 使用 256 个线程，512 点
kernel 使用 512 个线程，更大尺寸使用 1024 个线程。跨 warp 的蝶形阶段使用
共享内存，最后几个阶段通过 warp shuffle 和寄存器完成。

可复用的 `FFTPlan` 保存 FFT 点数、默认线程配置和 CUDA 设备信息。创建 plan
时会在 GPU 上分配并上传预计算的旋转因子表。每个 FFT block 会先将旋转因子表
载入共享内存，并在各蝶形阶段中重复使用。请在创建 plan 时对应的 CUDA 设备
上执行和销毁该 plan。FFT 执行使用基于 plan 的 API。自定义 FFT 实现不依赖
cuFFT 或 cuBLAS；测试和基准程序会使用 cuFFT 作为正确性参考和性能对比。

逆向变换遵循 cuFFT 的非归一化约定：对输入先执行正向 FFT，再执行逆向 FFT，
结果为原始输入乘以 FFT 点数。测试程序会将两个方向的结果与 cuFFT C2C 对比，
并检查正逆变换的往返结果。程序预热后会报告自定义 kernel 和 cuFFT 的耗时；
计时不包括主机与 GPU 之间的数据传输、cuFFT plan 创建以及自定义 FFT plan
创建开销。

## 可复用 FFT plan

每种变换尺寸创建一个 plan，并在正向和逆向变换中复用。Plan 与创建时当前的
CUDA 设备关联；调用和销毁 plan 时应选择同一设备。创建 plan 时会在 GPU 上
分配 `fft_size / 2` 个复数旋转因子，并在主机上预计算正向变换使用的因子。
逆向变换通过取复共轭获得对应的旋转因子。计算前，每个 block 都会将旋转因子
表复制到共享内存；kernel 不需要额外的设备工作区。

原先不使用 plan 的 FFT 启动和基准函数已移除。创建 plan 后，可调用
`launch_fft_plan` 或 `launch_fft_plan_config`：

```cpp
FFTPlan* plan = nullptr;
cudaError_t status = create_fft_plan(&plan, 2048);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}

status = launch_fft_plan(plan, device_input, device_output);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}

status = destroy_fft_plan(&plan);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}
```

## 批量 FFT

对于多个相互独立且连续存储的变换，可使用批量 API。每个变换占用连续的
`fft_size` 个复数值，kernel 为每个变换启动一个 block：

```cpp
status = launch_fft_plan_batch(plan, batch_input, batch_output, batch_count);
if (status != cudaSuccess) {
    // 处理 CUDA 错误。
}
```

基准程序会检查批量正向和逆向变换，并与 `cufftPlanMany` 及重复调用单变换
cuFFT plan 的结果对比；同时报告 1024 个变换组成的批次的吞吐量。

## 构建和运行

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
