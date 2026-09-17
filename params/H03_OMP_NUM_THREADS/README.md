# H03 OMP_NUM_THREADS：OMP 线程数

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5724-5821 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(H03, 第 408-414 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 83 行, 状态=已用)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`H03 OMP_NUM_THREADS` 定义 OpenMP（Open Multi-Processing）并行框架使用的线程数，固定为 **2**。OpenMP 是 C/C++/Fortran 并行编程的行业标准接口，被 PyTorch、NumPy、SciPy 底层（Intel MKL、LLVM libomp）用于并行化循环与矩阵运算。设置方式（**必须在 import torch 之前**）：

```python
import os
os.environ["OMP_NUM_THREADS"] = "2"
```

## ② 公式

OpenMP 并行化把循环/张量运算分给 $n$ 个线程，每个线程持有部分工作区。内存峰值近似：

$$\text{Mem}_{\text{parallel}}(n) \approx \text{base} + n\cdot \text{scratch}$$

OOM 条件与 H02 相同：$\text{Mem}(n) > 3.16\ \text{GB} \Rightarrow \text{OOM}$。PyTorch 在 import 时读取 OMP_NUM_THREADS 并设置默认并行线程数：

$$n_{\text{torch}} = \min(\text{OMP\_NUM\_THREADS},\; \text{可用核心})$$

## ③ 物理/直觉与用途

**物理直觉**：与 H02 同源——并行线程越多，内存峰值越高。现代 CPU 多为 24 核，OpenMP 默认启用全部核心；在 3.16GB RSS 的受限环境（超算平台）中，多线程同时做矩阵运算使内存峰值大幅上升。限制到 2 线程可有效控制内存压力、防止 OOM。

**用途**（与 H02 共同构成并行度控制）：

| 参数 | 控制对象 | 影响操作 |
|---|---|---|
| H02 | OpenBLAS | SVD、矩阵乘法、特征分解 |
| H03 | OpenMP | 循环并行、张量操作、MKL 运算 |

两者通常设置为相同值（都是 2），确保所有并行库使用一致的线程数。

## ④ 推导过程

**推导 1：PyTorch 线程继承链**。`torch` 底层算子（`ATen`）通过 OpenMP 并行化，其线程数在首次初始化时由 `OMP_NUM_THREADS` 决定。因此必须 import 前设置；import 后可读回验证：

```python
import torch
torch.get_num_threads()  # 期望 2（OMP_NUM_THREADS=2）
```

**推导 2：为何与 H02 相同值**。一个进程内 OpenBLAS 与 OpenMP 线程池同时存在。若 H02=8、H03=8，则可能同时有 16 个线程抢占内存与 CPU，峰值叠加；统一设为 2 使总并行度受控、可预期。

**推导 3：内存-速度权衡**。线程数从 24 降到 2，SVD 变慢（实测 2-3 秒量级仍可接受），但内存峰值从"必然 OOM"降到"3.16GB 内安全"。在受限环境中，安全优先于速度。

## ⑤ 数值验证

**实测状态**：H03 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 83 行：3.16GB RSS 限制下实测平稳）。

**实测值（主文档第 5799-5817 行）**：
- 设置 OPENBLAS_NUM_THREADS=2、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2
- 可用核心 24，实际并行线程 2
- SVD 平稳完成 ✅，内存峰值 3.16GB RSS（无 OOM）
- 对比（默认 24 线程）：SVD 触发 OOM 崩溃 ❌

**合成数值验证（[verify.py](verify.py)）**：
- import 前设置 OMP_NUM_THREADS=2，断言环境变量生效 ✅
- `torch.get_num_threads() == 2`（若 torch 可用）✅
- threadpoolctl 探测 OMP/OpenBLAS 线程数 ≤ 2（若可用）✅
- 用 torch 执行 SVD/矩阵乘法验证正确性（重建误差 < 1e-4）与内存受控 ✅
- 对照说明：本机实测 `torch.get_num_threads()` 默认 8，设置后收敛到 2

## ⑥ 与关联参数的关系

| 参数 | 控制范围 | 关系 |
|---|---|---|
| H02 OPENBLAS_NUM_THREADS | OpenBLAS 后端 | 两者同时设置为 2，保持并行度一致 |
| H05 dtype | 模型精度 | float32 比 float16 占用更多内存，线程数限制更为关键 |
| G03 max_dim | 跳过超大层 | 共同防 OOM：G03 控制协方差矩阵规模，H03 控制并行内存 |
| H01 seed | 随机种子 | 线程数不影响数值结果，与 H01 正交 |
