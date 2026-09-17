# H02 OPENBLAS_NUM_THREADS：OpenBLAS 线程数

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5637-5723 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(H02, 第 400-406 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 82 行, 状态=已用)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`H02 OPENBLAS_NUM_THREADS` 定义 OpenBLAS（开源 BLAS/LAPACK 库）使用的线程数，固定为 **2**。OpenBLAS 是 numpy/PyTorch 底层负责矩阵乘法、SVD、特征分解的加速库；默认会使用全部 CPU 核心（如 24 核），导致内存消耗成倍增长。设置方式（**必须在 import numpy/torch 之前**）：

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
```

## ② 公式

线程数 → 内存峰值的机制：对 $H\in\mathbb{R}^{B\times d}$ 执行 `svdvals(H)` 时，OpenBLAS/LAPACK 创建 $n_{\text{threads}}$ 个工作线程，每个线程分配临时工作区。总内存近似：

$$\text{Mem}_{\text{SVD}}(n) \approx \text{base} + n\cdot \text{scratch}$$

其中 `base` 为主矩阵 $O(d^2)$，`scratch` 为每线程临时缓冲。OOM 条件：

$$\text{Mem}_{\text{SVD}}(n) > \text{RSS 上限}\;(3.16\ \text{GB}) \iff \text{OOM}$$

限制 $n=2$ 使 `Mem(2)` 落在限制内，而默认 `Mem(24)` 越限崩溃。

## ③ 物理/直觉与用途

**物理直觉**：并行线程数是一把双刃剑——更多线程意味着更快与更多内存。在超算/受限环境（3.16GB RSS）下，SVD 的每线程工作区会把内存峰值推过红线。`OPENBLAS_NUM_THREADS=2` 是在"内存安全"与"计算速度"之间取平衡：2 线程既保留足够并行度，又把每线程内存开销压到 2 份。

**用途**（防止 SVD 计算时的堆内存溢出）：

| 线程数 | 内存峰值 | SVD 速度 | 风险 |
|---|---|---|---|
| 默认（24 核） | 极高 | 快 | ❌ OOM |
| 8 | 高 | 中 | ⚠️ 可能 OOM |
| **2** | **低** | **可接受** | **✅ 安全** |
| 1 | 最低 | 慢 | ✅ 安全 |

实测：在 3.16GB RSS 限制下，`OPENBLAS_NUM_THREADS=2` 使 SVD 平稳运行，默认 24 线程导致内存溢出崩溃。

## ④ 推导过程

**推导 1：SVD 内存爆炸机制**。`torch.linalg.svdvals(H)`（LAPACK `?gesdd` 分治 SVD）为加速而并行化，每个线程持有工作区副本。设工作区大小 $W(d)$，则：

$$\text{Mem}(n) = \underbrace{O(d^2)}_{\text{主矩阵}} + \underbrace{n\cdot W(d)}_{\text{线程工作区}}$$

当 $H$ 的 $d$ 较大时，$n\cdot W(d)$ 项随 $n$ 线性增长。24 线程比 2 线程多出 22 份工作区，恰是压垮 3.16GB 上限的最后一根稻草。

**推导 2：为何选 2 而非 1**。1 线程最安全但 SVD 显著变慢（串行）；2 线程在典型双通道 CPU 上几乎跑满单 socket 吞吐，内存开销仅 2 份工作区。故 2 是"安全且不慢"的驻点。

**推导 3：为何必须 import 前设置**。OpenBLAS 在首次初始化时读取环境变量并锁定线程池大小；import 之后设置对已初始化的运行时无效。因此：

```python
os.environ["OPENBLAS_NUM_THREADS"] = "2"   # 必须先于 import numpy/torch
import numpy as np
import torch
```

## ⑤ 数值验证

**实测状态**：H02 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 82 行：3.16GB RSS 限制下实测平稳运行）。

**实测值（主文档第 5705-5719 行）**：
- 设置 OPENBLAS_NUM_THREADS=2，可用核心 24
- SVD 运行状态：平稳完成 ✅，内存峰值 3.16GB RSS（无 OOM）
- 对比（默认 24 线程）：SVD OOM 崩溃 ❌

**合成数值验证（[verify.py](verify.py)）**：
- 在 import numpy 前设置 `OPENBLAS_NUM_THREADS=2`，断言环境变量生效 ✅
- 尝试读取实际 OpenBLAS 线程数（openblas_get_num_threads / threadpoolctl，如可用）✅
- 执行中等规模 SVD（如 4096×4096 float32 ≈ 64MB），验证分解正确（重建误差 < 1e-4）且平稳完成 ✅
- 记录峰值内存（tracemalloc/psutil，如可用），确认远低于 3.16GB 上限 ✅
- 说明：实际线程数受构建后端影响，核心验证点是"环境变量正确设置 + SVD 平稳完成"

## ⑥ 与关联参数的关系

| 参数 | 作用 | 关系 |
|---|---|---|
| H03 OMP_NUM_THREADS | OMP 线程数 | 两者同时设置为 2，共同控制并行度 |
| G03 max_dim | 跳过超大层 | 共同防 OOM：G03 跳过 lm_head（维度层面），H02 控制 SVD 内存（线程层面） |
| H05 dtype | 模型精度 | float32 下协方差 4 字节/元素；dtype 与线程数共同决定内存预算 |
| H01 seed | 随机种子 | 线程数不影响数值结果（确定性 LAPACK），只影响内存/速度 |
