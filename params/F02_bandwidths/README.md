# F02 bandwidths：带宽预测档

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 4251-4360 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(F02, 第 310-316 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 71 行, 状态=理论)
> 源码：[._local_whitebox_detect.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_local_whitebox_detect.py)（第 151-154 行：有效带宽估计）
> 实测日志：[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)（tok/s=8.39，有效带宽 16.6GB/s）
> 验证脚本：[verify.py](verify.py)

## ① 定义

`F02 bandwidths` 定义三档硬件内存带宽参考值，用于预测不同硬件上的 decode 推理速度上限：

```python
bandwidths = [30e9, 2e12, 3.35e12]   # 单位: bytes/s
```

| 档位 | 带宽值 | 代表硬件 | 说明 |
|---|---|---|---|
| 档 1 | 30 GB/s | CPU DDR4-3200 | 单通道理论峰值，实际有效约 20-25GB/s（本机实测 16.6GB/s） |
| 档 2 | 2 TB/s | NVIDIA A100 (80GB) | HBM2e，NVIDIA 官方规格 2.0TB/s |
| 档 3 | 3.35 TB/s | NVIDIA H100 (80GB) | HBM3，NVIDIA 官方规格 3.35TB/s |

这三个值是**理论上限**而非实测值：30e9 对应 DDR4-3200 的典型带宽（保守估计，高于本机实测 16.6GB/s）；2e12 与 3.35e12 是 NVIDIA 官方规格。

## ② 公式

解码阶段为 memory-bound（每生成一个 token 需读取一遍全部权重），故：

$$tok/s \approx \frac{B_{mem}}{W_{bytes}},\qquad W_{bytes} = N_{params}\times\text{dtype\_bytes}$$

其中 $B_{mem}$ 为带宽（bytes/s），$W_{bytes}$ 为权重字节数。对 Qwen2.5-0.5B（fp32，W≈1.98GB）：

$$tok/s_{DDR4} = \frac{30\text{e}9}{1.98\text{e}9} \approx 15.2,\qquad
tok/s_{A100} = \frac{2\text{e}12}{1.98\text{e}9} \approx 1010,\qquad
tok/s_{H100} = \frac{3.35\text{e}12}{1.98\text{e}9} \approx 1692$$

有效带宽（实测反推）与效率：

$$BW_{eff} = W_{bytes}\times tok/s_{实测},\qquad \eta = \frac{tok/s_{实测}}{B_{mem}/W_{bytes}}$$

## ③ 物理/几何直觉与用途

**物理直觉**：自回归生成是"权重流"——每个新 token 都要把整套权重从内存搬运到计算单元。DDR4 每秒只能搬 30GB，而权重就有 1.98GB，于是 CPU 上限约 15 token/s；HBM3 每秒搬 3.35TB，把上限抬到 1690 token/s。**带宽是解码吞吐的硬天花板**，与算力（FLOPs）无关，这是 LLM 推理"内存受限"的本质。

几何视角：指纹测量需要"看"模型内部激活（k_proj 逐层 spl_gamma、曲率采样）。CPU 上每 token 要花 ~0.12 秒，48 步就要 ~5.7 秒——带宽决定了一次指纹采集的时间成本。三档带宽实际上界定了同一份指纹数据在三类硬件上的采集耗时跨度（约 100 倍）。

**用途**：① 预测不同硬件的 tok/s 上限，为指纹采集做时间预算；② 与实测 tok/s 对比得到效率系数 $\eta$（本机 CPU η≈55%），判断推理是否处于权重读取受限区；③ 判断 KV 压缩的收益边界（KV/W 占比大时压缩才显著）。

## ④ 推导过程

**推导 1：权重大小**。Qwen2.5-0.5B 参数量约 0.5B，fp32 下 $W\approx0.5\times10^9\times4=2\text{GB}$；实测日志记为 1.98GB（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log) 第 14 行，对应 494M 参数）。

**推导 2：memory-bound 判据**。decode 阶段单 token 计算量 ∝ 序列长度（attention）与权重（线性层），而权重读取量固定为 $W_{bytes}$。当 $W_{bytes}/B_{mem} \gg$ 单 token 计算时间时，带宽成为瓶颈（对 0.5B 模型在 CPU 上严格成立）。故 $tok/s=B_{mem}/W_{bytes}$ 是可靠的上限估计。

**推导 3：效率系数的意义**。实测 CPU tok/s=8.39，理论 15.2，$\eta=8.39/15.15\approx55.4\%$。差值来源于：内存控制器效率（DDR4 实际有效带宽约为理论峰值 55-70%）、kernel launch 开销、预填充与采样 overhead。用同一 η 可把 A100/H100 的理论值修正为更现实的"预期实测"区间。

## ⑤ 数值验证

**实测状态**：F02 = 理论（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 71 行：三档带宽为规格值，仅 CPU 档有实测对照）。

**实测值（Qwen2.5-0.5B-Instruct，CPU，[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)）**：

| 硬件 | 带宽 | 理论 tok/s 上限 | 实测 tok/s | 效率 |
|---|---|---|---|---|
| CPU DDR4 | 30 GB/s | 30/1.98 ≈ 15.2 | 8.39 | 55.4% |
| A100 | 2 TB/s | 2000/1.98 ≈ 1010.1 | — | — |
| H100 | 3.35 TB/s | 3350/1.98 ≈ 1691.9 | — | — |

有效带宽反推：$BW_{eff}=1.98\text{GB}\times8.39=16.6\text{GB/s}$（实测日志第 31 行）✅

**合成数值验证（[verify.py](verify.py)）**：
- 三档带宽值精确匹配 [30e9, 2e12, 3.35e12] ✅
- tok/s=带宽/权重字节：15.2 / 1010.1 / 1691.9，与主文档输出一致（偏差 <0.5%）✅
- 效率 η=8.39/15.15≈55.4%（主文档 ~55%）✅
- 有效带宽 1.98e9×8.39≈16.6GB/s ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| F05 gen_len_eng | 48 步生成测得 tok/s=8.39，正是 F02 的 CPU 档实测输入；F02 反过来用三档带宽预测 tok/s 上限 |
| A01 model | 权重字节数由模型参数量×dtype 决定，tok/s 预测依赖该值 |
| H05 dtype | 公式中 dtype_bytes：fp16 权重减半 → 同带宽下 tok/s 翻倍 |
| 501 tok/s | 输出指标：实测 8.39（Qwen0.5B CPU），与 F02 CPU 档预测对比 |
| 505 内存带宽 | 输出指标：实测 16.6GB/s = W_bytes×tok/s，即 F02 有效带宽公式的直接产物 |
| 502 KV_bytes / 503 KV/W | KV/W 越大，decode 越受 KV 读取影响，bandwidth 模型的"纯权重读取"假设越需要修正 |
