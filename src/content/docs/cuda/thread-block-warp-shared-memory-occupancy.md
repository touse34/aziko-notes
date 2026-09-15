---
title: Thread、Block、Warp 与 Shared Memory 如何影响 CUDA 理论占用率？
description: 用统一公式和对比算例解释线程数、线程块、warp 与 shared memory 如何共同限制每个 SM 的驻留资源，以及为什么更高 occupancy 不一定更快。
---

结论：**CUDA 理论占用率由每个 SM 上能够同时驻留的 warp 数决定，而 thread、block、shared memory 和 register 都是在限制“最多能驻留多少个 block”。最终结果由最先耗尽的那项资源决定。**

所以不能孤立地说：

- “block 越大，occupancy 越高”；
- “threads per block 是 32 的倍数，就一定是 100%”；
- “shared memory 只影响访存，不影响调度”；
- “理论 occupancy 达到 100%，kernel 就一定最快”。

## 什么是理论占用率

CUDA 中的 occupancy 定义为：

```text
Theoretical Occupancy
    = Active Warps per SM / Maximum Warps per SM
```

这里的 active 指**可以同时驻留在 SM 上、具备执行资格的 warp**，不代表这些 warp 每个周期都在执行，也不代表 warp 内 32 个 lane 都在做有效工作。

理论占用率来自 launch 配置和静态资源限制；Nsight Compute 中的 Achieved Occupancy 则是运行时采样得到的实际活跃 warp 比例。两者必须分开。

## 四个概念各自影响什么

| 概念 | 资源粒度 | 对理论 occupancy 的主要影响 | 常见误区 |
| --- | --- | --- | --- |
| Thread | 单个 CUDA 线程 | 决定每个 block 的线程总数，并消耗每线程 register | 线程越多不等于 occupancy 越高 |
| Warp | 固定为 32 个线程 | 是 occupancy 的直接计数单位 | active warp 不等于 32 个 lane 都有效 |
| Thread block | 不可拆分的驻留单位 | SM 只能接收整数个完整 block，还受 blocks/SM 上限约束 | 大 block 不一定比小 block更好 |
| Shared memory | 按 block 分配 | 每驻留一个 block，就要在 SM 上复制一份该 block 的 shared memory | 只看每 SM 总容量，忽略向下取整和分配粒度 |

register 也不能从计算中省略：它通常按线程使用、按 warp/block 的硬件粒度分配，同样会限制驻留 block 数。关于类型与 register 的关系，可参阅上一篇笔记：[CUDA 中 int 换成 bool 会减少寄存器吗？](/cuda/int-vs-bool-registers/)

## 统一计算框架：先算 blocks/SM，再取最小值

设：

- `T_block`：threads per block；
- `W = 32`：warp size；
- `W_block = ceil(T_block / W)`：warps per block；
- `T_SM`：每个 SM 最多驻留的线程数；
- `W_SM`：每个 SM 最多驻留的 warp 数；
- `B_SM`：每个 SM 最多驻留的 block 数；
- `S_SM`：每个 SM 可用于 block 的 shared memory；
- `S_block`：每个 block 的 static shared memory 与 dynamic shared memory 之和；
- `R_SM`：每个 SM 的 register 数；
- `R_block`：每个 block 实际分配的 register 数。

先分别计算各资源允许的驻留 block 数：

```text
B_thread   = floor(T_SM / T_block)
B_warp     = floor(W_SM / W_block)
B_shared   = floor(S_SM / S_block)
B_register = floor(R_SM / R_block)
```

最终能够驻留的 block 数是：

```text
B_active = min(B_SM, B_thread, B_warp, B_shared, B_register)
```

于是：

```text
W_active  = B_active × W_block
Occupancy = W_active / W_SM
```

如果 kernel 不使用 shared memory，可以把 `B_shared` 看成不构成限制；其他资源同理。

:::caution[手算只能用于理解]
真实 GPU 会按架构规定的粒度对 warp、register 和 shared memory 分配量进行向上取整，还可能受到 shared-memory carveout、每 block 资源上限和 cluster 配置影响。因此手算适合找 limiter，最终结果应以 CUDA Occupancy API 或 Nsight Compute Occupancy Calculator 为准。
:::

## Thread：数量最终会被换算成 warp

GPU 并不是逐个调度线程，而是把一个 block 中的线程按连续 thread ID 划分为 warp：

```text
W_block = ceil(T_block / 32)
```

例如：

| Threads/block | 分配的 warps/block | 最后一个 warp 的有效线程数 |
| ---: | ---: | ---: |
| 32 | 1 | 32 |
| 33 | 2 | 1 |
| 64 | 2 | 32 |
| 128 | 4 | 32 |
| 160 | 5 | 32 |
| 256 | 8 | 32 |

`33 threads/block` 会分配两个 warp。第二个 warp 虽然只有一个有效线程，但在 occupancy 计算中仍然占用一个完整 warp 的驻留槽位。

这说明两个重要区别：

- **theoretical occupancy** 统计驻留 warp；
- **warp execution efficiency / active threads per warp** 关心每个 warp 有多少 lane 真正在工作。

因此，一个 kernel 可以同时拥有很高的 occupancy 和很差的 lane 利用率。

### 极端例子：100% occupancy，但近一半 lane 空闲

假设某个抽象 SM 最多驻留：

- 64 warps；
- 32 blocks；
- 2048 threads。

忽略 register 和 shared memory 限制，使用 `33 threads/block`：

- 每个 block 分配 2 个 warp；
- warp 限制允许 `64 / 2 = 32` 个 block；
- block 上限也是 32；
- 共驻留 `32 × 2 = 64` 个 warp；
- 理论 occupancy 为 100%。

但是每个 block 实际只有 33 个线程，却占用了 64 个 lane 对应的 warp 槽位，单看线程填充率只有：

```text
33 / 64 ≈ 51.6%
```

所以“100% occupancy”绝不等于“100% SIMD lane 利用率”。

## Warp：理论占用率的直接分子

warp 是 SM 保存执行上下文和调度指令的基本单位。一个 warp 因内存依赖、数据依赖或同步而暂时不能发射指令时，warp scheduler 可以选择另一个 ready warp，以此隐藏延迟。

occupancy 的意义就在这里：驻留 warp 太少时，可供 scheduler 切换的候选通常不足；但驻留 warp 足够多之后，继续提高 occupancy 不一定增加 eligible warp，也不一定提升吞吐。

还要区分这些指标：

| 指标 | 回答的问题 |
| --- | --- |
| Theoretical Occupancy | 资源上限允许驻留多少 warp？ |
| Achieved Occupancy | 运行过程中实际观察到多少活跃 warp？ |
| Eligible Warps per Scheduler | 每个周期真正有多少 warp 已准备好发射？ |
| Active Threads per Warp | 一个 warp 中平均多少 lane 在参与执行？ |
| SM Throughput | SM 的执行管线实际有多忙？ |

高 theoretical occupancy 只回答第一行，不能替代后面几项。

## Block：不可拆分带来的阶梯效应

SM 以完整 block 为单位接收工作。即使剩余资源足够容纳若干 warp，只要放不下一个完整 block，这部分资源就无法被该 kernel 使用。

继续使用“64 warps、2048 threads、32 blocks per SM”的抽象设备，并忽略其他资源：

| Threads/block | Warps/block | Active blocks/SM | Active warps/SM | 理论 occupancy |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 32（受 block 上限限制） | 32 | 50% |
| 64 | 2 | 32 | 64 | 100% |
| 128 | 4 | 16 | 64 | 100% |
| 160 | 5 | 12 | 60 | 93.75% |
| 192 | 6 | 10 | 60 | 93.75% |
| 256 | 8 | 8 | 64 | 100% |
| 1024 | 32 | 2 | 64 | 100% |

这个表揭示了三个现象：

1. **block 太小可能先撞到 blocks/SM 上限。** 32-thread block 即使资源很轻，也可能无法填满所有 warp 槽位。
2. **某些 block size 会留下无法拼成完整 block 的资源碎片。** 160 和 192 都只能驻留 60 个 warp。
3. **更大的 block 不代表更高 occupancy。** 64、128、256 和 1024 在这个理想算例中都是 100%。

实际调优通常可以从 128、256 threads/block 开始比较，但这只是搜索起点，不是固定答案。频繁使用 `__syncthreads()` 的 kernel 还要考虑：若每个 SM 只能驻留一个大 block，这个 block 等待同步时，SM 上可能没有另一个 block 可以接替执行。

## Shared memory：每个 block 都要复制一份

一个 block 使用的 shared memory 为：

```text
S_block = S_static + S_dynamic
```

- static shared memory 来自 `__shared__` 静态声明；
- dynamic shared memory 来自 `extern __shared__`，大小由 kernel launch 的第三个参数指定。

当一个 SM 驻留多个 block 时，每个 block 都要拥有独立的 shared-memory 区域。因此 shared memory 对 occupancy 的影响是明显的阶梯函数。

### 对比算例

假设抽象 SM 有 64 KiB shared memory，kernel 使用 256 threads/block，即 8 warps/block，并暂时忽略 register 限制：

| Shared memory/block | Shared memory 允许的 blocks/SM | Active warps/SM | 理论 occupancy |
| ---: | ---: | ---: | ---: |
| 0 KiB | 不构成限制 | 64 | 100% |
| 8 KiB | 8 | 64 | 100% |
| 16 KiB | 4 | 32 | 50% |
| 20 KiB | 3 | 24 | 37.5% |
| 32 KiB | 2 | 16 | 25% |
| 48 KiB | 1 | 8 | 12.5% |

这里假设 thread/warp/block 上限允许最多 8 个 256-thread block，并忽略真实硬件的分配粒度。可以看到，shared memory 从 8 KiB 增加到 16 KiB 时，理论 occupancy 不是平滑下降一点，而是从 100% 跳到 50%。

反过来也一样：将每 block shared memory 从 20 KiB 降到 17 KiB，可能仍只能驻留 3 个 block，性能不会因为“省了 3 KiB”自动改善。只有跨过允许第 4 个 block 驻留的阈值，理论 occupancy 才会跳变。

### Shared memory 少了也不保证更快

shared memory 常被用来：

- 合并或重排 global-memory 访问；
- 在线程之间复用数据；
- 减少重复加载；
- 构建 block 内协作算法。

为了提高 occupancy 把数据从 shared memory 移回 global memory，可能增加长延迟访存；减少 tile 大小也可能降低数据复用。正确问题不是“shared memory 能不能再少一点”，而是：

> 减少 shared memory 后是否跨过驻留阈值，并且新增加的访存或计算代价是否小于更多并发 warp 带来的收益？

## 组合对比：真正的 limiter 是最小值

假设某个 kernel 根据不同资源分别可以驻留：

| 限制项 | 最多允许的 blocks/SM |
| --- | ---: |
| 硬件 blocks/SM 上限 | 24 |
| threads 上限 | 12 |
| warps 上限 | 12 |
| registers | 8 |
| shared memory | 5 |

那么最终只能驻留：

```text
B_active = min(24, 12, 12, 8, 5) = 5
```

此时 register 从“允许 8 blocks”优化到“允许 10 blocks”不会改变 occupancy，因为 shared memory 仍然把结果卡在 5 blocks。必须先确认 limiter，再优化对应资源。

同样，如果 shared memory 从“允许 5 blocks”优化到“允许 9 blocks”，limiter 会切换成 register，最终只能驻留 8 blocks，而不是 9。

## 一个真实反例：达到 100% occupancy 后反而更慢

在一次 RTX 4060 Laptop（SM 8.9）的 Fast32 kernel 隔离实验中，生产基线约为：

- 64 threads/block；
- 48 registers/thread；
- 4000 B shared memory/block；
- 20 active blocks/SM；
- 83.33% theoretical occupancy。

仅把 register cap 从 48 压到 40 后，shared memory 仍然只允许 20 blocks/SM，因此理论 occupancy 仍是 83.33%。这正是“优化非 limiter 不改变最终结果”的例子。

随后另一个实验版本同时改变存储布局，将 shared memory 降到约 2976 B/block，并使用 40 registers/thread，驱动 Occupancy API 验证达到 24 blocks/SM、100% theoretical occupancy；Nsight Compute 测得约 92.1% achieved occupancy。

但它并没有更快：硬寄存器约束引发更多编译器生成的 local-memory 流量，eligible warps per scheduler 反而从约 1.49 降到 1.41。固定工作量和完整训练均变慢，完整训练回退约 5%～8%。最终结论是**正确但更慢，不部署**。

这个反例说明：

1. 降低 register 只有在 register 是当前 limiter 时才可能提高 occupancy；
2. 为跨过 shared-memory 阈值而移动数据，会带来新的访存成本；
3. theoretical occupancy、achieved occupancy 和 eligible warps 是不同指标；
4. occupancy 是诊断线索，不是最终优化目标。

## 用 CUDA Occupancy API 得到可信结果

不要把手算结果直接当成硬件事实。可以让 CUDA Runtime 根据目标 GPU、最终编译出的 kernel、block size 和 dynamic shared memory 计算 active blocks/SM：

```cpp
int block_size = 256;
std::size_t dynamic_smem = 16 * 1024;
int active_blocks = 0;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    my_kernel,
    block_size,
    dynamic_smem
);

cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);

int warps_per_block =
    (block_size + prop.warpSize - 1) / prop.warpSize;
int max_warps_per_sm =
    prop.maxThreadsPerMultiProcessor / prop.warpSize;

double theoretical_occupancy =
    static_cast<double>(active_blocks * warps_per_block) /
    max_warps_per_sm;
```

这里传入的必须是**实际 launch 使用的 block size 和 dynamic shared-memory 字节数**。还应检查每个 CUDA API 的返回值；示例省略错误处理只是为了突出计算过程。

如果 dynamic shared memory 随 block size 改变，可以使用 `cudaOccupancyMaxPotentialBlockSizeVariableSMem`；如果已确定希望每个 SM 驻留的 block 数，可用 `cudaOccupancyAvailableDynamicSMemPerBlock` 反推每 block 可用的 dynamic shared memory。

## 用编译器和 Nsight Compute 验证

### 1. 记录最终静态资源

```powershell
nvcc -O3 -arch=sm_89 kernel.cu --resource-usage -o kernel.exe
```

或者使用：

```powershell
nvcc -O3 -arch=sm_89 kernel.cu -Xptxas=-v -Xptxas=-warn-spills -o kernel.exe
```

记录每个 kernel 的 registers/thread、static shared memory、stack frame、spill loads 和 spill stores。不要只记录源码中的 `__shared__` 数组大小，也不要假设 `--maxrregcount` 一定等于最终资源表现。

### 2. 采集 launch 与 occupancy 数据

```powershell
ncu --section LaunchStats --section Occupancy .\kernel.exe
```

至少比较：

- block size、grid size；
- registers/thread；
- static/dynamic shared memory per block；
- active blocks/warps per SM；
- theoretical occupancy；
- achieved occupancy；
- occupancy limiter。

然后结合 Scheduler Statistics 查看 eligible warps，并结合 Memory Workload Analysis 检查为了省 shared memory 或 register 是否增加了 local/global-memory 流量。

### 3. 做 occupancy 敏感性实验

NVIDIA 建议的一种方法是：在不改变 kernel 有效计算的情况下，增加 launch 的 dynamic shared-memory 数量，人为降低可驻留 block 数，然后观察运行时间如何变化。

如果 occupancy 从 100% 降到 75% 或 50%，kernel 时间几乎不变，说明原 kernel 可能已经拥有足够的延迟隐藏能力，继续追求 100% 的价值有限。反之，如果时间显著恶化，occupancy 更可能是当前性能敏感因素。

## 理论 occupancy 与实际性能为什么不等价

高 occupancy 主要提供更多 warp 用于隐藏延迟，但 kernel 性能还取决于：

- warp 中有效 lane 的比例与分支分歧；
- eligible warps 是否足够；
- 指令级并行度（ILP）；
- global/local memory 延迟与带宽；
- shared-memory bank conflict；
- register spill；
- 算术管线、Tensor Core 或特殊函数单元吞吐；
- grid 是否足够大，能否覆盖所有 SM；
- block 尾部效应与负载不均衡。

尤其要注意：理论 occupancy 假设每个 SM 有足够多的 block 可以驻留。如果整个 grid 的 block 数太少，某些 SM 根本分不到足够工作，即使 Occupancy Calculator 给出 100%，运行时也达不到那个状态。

## 调优时的推荐顺序

1. **确认目标 GPU。** 查询 compute capability、warp size、threads/warps/blocks per SM、shared memory 和 register 容量。
2. **记录真实 launch。** 包括 grid、block、static/dynamic shared memory。
3. **读取最终 kernel 资源。** 使用 `ptxas -v`、`--resource-usage` 或 `cudaFuncGetAttributes`。
4. **用 Occupancy API/Calculator 找 limiter。** 不要只做粗略手算。
5. **只优化当前 limiter。** 如果 shared memory 卡住 blocks/SM，单独压 register 通常无效。
6. **检查是否跨过离散阈值。** 资源减少但 active blocks/SM 不变时，theoretical occupancy 不会变。
7. **profile 动态行为。** 同时看 achieved occupancy、eligible warps、active lanes、stall 和内存流量。
8. **做正确性与交错计时。** 用固定工作量、CUDA event 和 A/B/ABBA 顺序比较，最后再看端到端时间。

## 一张表记住核心关系

| 你修改了什么 | 可能改善什么 | 什么时候不会提高理论 occupancy | 可能付出的代价 |
| --- | --- | --- | --- |
| 增加 threads/block | 每 block warp 数，减少 blocks/SM 上限浪费 | 被 shared memory/register 限制时 | 并行粒度过粗、同步等待、尾部效应 |
| 减少 threads/block | 允许更多小 block 驻留 | 先撞到 blocks/SM 上限时 | warp 数不足、调度开销和碎片 |
| 调成 32 的倍数 | 避免最后一个不完整 warp | occupancy 本来受其他资源限制时 | 通常代价较小，但算法映射可能变差 |
| 减少 shared memory/block | 可能允许更多 block 驻留 | 没跨过下一个 block 阈值，或 limiter 是 register 时 | 更多 global-memory 访问、复用下降 |
| 减少 registers/thread | 可能允许更多 block/warp 驻留 | limiter 是 shared memory、block 或 thread 时 | spill、额外指令、ILP 下降 |
| 增加 grid blocks | 改善全 GPU 实际覆盖 | 单 SM 理论资源配置不变 | 更多总工作或调度开销 |

## 总结

理解 occupancy 最可靠的方式，是始终沿着这条链条推导：

> **threads/block → warps/block → 每项资源允许的 blocks/SM → 取最小值 → active warps/SM → theoretical occupancy**

thread 决定 block 中有多少 warp；warp 是 occupancy 的计数单位；block 是不可拆分的驻留单位；shared memory 和 register 决定一个 SM 能复制多少份 block 的执行上下文。任何一项都可能成为 limiter，而且 limiter 会随着代码和 launch 配置变化。

最后要记住：**理论占用率是资源容量模型，不是性能分数。** 优化是否成立，必须同时看正确性、实际 occupancy、eligible warps、lane 利用率、spill/内存流量、kernel 时间和端到端时间。

## 参考资料

- [CUDA C++ Best Practices Guide：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy)
- [CUDA C++ Best Practices Guide：Thread and Block Heuristics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#thread-and-block-heuristics)
- [CUDA C++ Best Practices Guide：Effects of Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effects-of-shared-memory)
- [CUDA Programming Guide：Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)
- [CUDA Runtime API：Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)
- [Nsight Compute：Occupancy Calculator](https://docs.nvidia.com/nsight-compute/NsightCompute/#occupancy-calculator)
