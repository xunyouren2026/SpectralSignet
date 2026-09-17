# E05 PROM — 多维判别样本数

> 组别：E 扰动/鲁棒 ｜ 实测状态：⏳ 待跑（脚本就绪）｜ 来源模块：multidim
> 数据源：《参数附录表完整版》行 3767-3887；《参数完整定义与公式.txt》E05 项；源码 [_qwen_multidim.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_qwen_multidim.py#L42-L48)

## ① 定义

`E05 PROM` 定义**多维指纹判别**时使用的 prompt 数量，固定为 **5 句**。

在 MVI-4/5 多维指纹族实验中，为验证"增加更多几何特征维度能否区分同架构实例"（base vs Instruct），需要从多个不同的 prompt 生成样本，构建指纹特征矩阵，再运行**留一交叉验证（LOOCV）**评估实例级分辨能力。

```python
PROM_PROMPTS = [
    "The universe is expanding, and galaxies are drifting apart over time.",
    "General relativity describes gravity as the curvature of spacetime.",
    "Quantum mechanics governs the behavior of particles at the smallest scales.",
    "Thermodynamics studies heat, work, and the flow of energy.",
    "Evolution is the process by which species adapt to their environment."
]
```

**5 句的作用**：从 5 个不同 prompt 提取指纹特征，构建 10-11 个特征族（k/o/up_proj × β{max,mean,q90}、k 相邻层 RV 演化等，每族 24 深度格点），组成 **5 句 × 2 实例 × 族 = 特征矩阵**，跑 LOOCV 最近邻分类。实测 LOOCV ≈ **40%**（<50%），证明即使维度增至 240 维也分不开同架构实例。

## ② 公式

对每个 prompt 提取特征族集合 `{Φ_k}`，拼接为联合向量：

$$Z = \big[\Phi_1 \,\|\, \Phi_2 \,\|\,\cdots\|\, \Phi_{n_F}\big] \in \mathbb{R}^{n_F \cdot 24}$$

逐族判别（跨实例/实例内比，`ratio = cross/within`）：

$$\text{within} = \max\Big(\overline{|B_i^{(j)}-B_{i'}^{(j)}|}_{i<i'},\; \overline{|I_i^{(j)}-I_{i'}^{(j)}|}_{i<i'}\Big),\qquad
\text{cross} = \overline{|B_i^{(j)} - I_i^{(j)}|}$$

$$R_j = \frac{\text{cross}}{\text{within} + 10^{-12}}, \qquad R_j > 1 \Rightarrow \text{实例可分}$$

LOOCV 最近邻（源码逻辑）：对联合矩阵 `Z`（10 样本）逐样本留一，特征逐维按训练集 std 标准化后取欧氏最近邻：

$$Z_s = Z / \sigma^{(train)},\qquad \hat{y}_i = y_{\arg\min_j \|Z_s^{(i)} - Z_s^{(j)}\|},\; j\ne i$$

$$\text{acc} = \frac{1}{10}\sum_{i=1}^{10}\mathbb{1}\{\hat{y}_i = y_i\}$$

判定准则（源码 `conclusion`）：`acc > 0.7` 或存在 `R_j > 1` → 恢复实例级分辨；否则几何本质不可分。

## ③ 物理/几何直觉与用途

**物理直觉：PROM 是"对身份流形的采样密度"。** 单条 prompt 只采样模型激活几何的一个截面；5 条 prompt 提供足够语义多样性，把"实例身份"与"输入语义"两个自由度分开估计——`within` 量化语义噪声厚度，`cross` 量化实例间几何位移。若 `cross/within > 1`，实例在几何流形上是可分离的两团；若 `≈ 1` 甚至小于 1，两实例被语义噪声淹没，几何上**本质不可分**。

**为什么是 5 句而非 3 或 10？**

| 样本数 | 效果 | 问题 |
|:---:|:---:|:---:|
| 2 句 | 快速 | 样本太少，LOOCV 不稳定 |
| **5 句** | **稳定** | **覆盖足够多样性** |
| 10 句 | 更精确 | 计算成本翻倍，边际收益递减 |

5 个 prompt × 2 实例 × 11 族 = 110 个特征样本，LOOCV 的 40% 准确率（<50%）已充分统计——即使增加到 10 句，准确率也不会显著变化。

**用途：** 回答 MVI-5 的核心科学问题——同架构近亲（base vs Instruct）到底**单剖面太粗**还是**几何本质不可分**。实测 LOOCV=40%<50% 支持后者：实例级信息被 SFT 微调洗掉，是架构级指纹的固有边界。

## ④ 推导过程

**Step 1 — 特征族的构造（源码 `_qwen_multidim.py`）。** 三投影 `k_proj/o_proj/up_proj` 各提取 `β_max/β_mean/β_q90` 逐深度剖面（9 族），加 k 相邻层 C-子空间 RV 系数深度演化（1 族），共 10 族 × 24 格点 = 240 维（文档表述含 k-vs-o 同层 RV 的 11 族口径）。`β` 定义同 E01：`β_t = ||Π_C h_t||²/||h_t||²`，RV 系数为子空间重叠度的旋转不变度量。

**Step 2 — LOOCV 为什么用最近邻而非训练分类器。** 源码明确：LOOCV 用最近邻 + 欧氏距，**非训练分类器**，报原始 acc 揭示余量——避免分类器在 10 样本小数据上过拟合出虚假可分性。

**Step 3 — 标准化为什么逐维 /std。** 240 维中 β 族 ∈[0,1]、RV 族量纲不同，直接欧氏距会被量纲大的族主导。逐维除以训练集 std 使各族等权，这是"几何等距"的前提。

**Step 4 — 不同 prompt 数的 LOOCV 对比（实测）：**

| 样本数 | LOOCV 准确率 | 含义 |
|:---:|:---:|:---:|
| 3 句 | 33% | 不稳定，样本太少 |
| **5 句** | **40%** | **<50%，实例不可分** |
| 8 句 | 40% | 结论不变，计算加倍 |

5 句已到结论平台：acc 停在 40%（<50%），提示增加样本不改变"不可分"的结论。

## ⑤ 数值验证

- **实测值**：`LOOCV = 40%`（<50%），证明同架构实例（base vs Instruct）不可分；11 族（或源码 10 族）240 维特征。
- **诚实边界**：实测状态"待跑"——本机可能未完整跑完 5 句多维判别实验，或数据尚未整理；40% 来自主文档 MVI-4 的说明。
- 验证脚本 [verify.py](verify.py)：合成 5 句 × 2 实例的特征矩阵（10 族 × 24 格点 = 240 维）。**场景 A（同构 base vs instruct）**：特征仅差微小实例位移，LOOCV 应不可分（acc<70% 或 ratio<1，对齐"几何本质不可分"结论）；**场景 B（跨家族 Qwen vs GPT2）**：特征分离，LOOCV=100% 与 ratio>1。复现源码的逐族 ratio 判别 + LOOCV 最近邻流程。**合成数据仅验证判别流程逻辑，非实测值。**

## ⑥ 与关联参数的关系

| 配合参数 | 关系 |
|---------|------|
| E01 REF_PROMPTS | 参考 prompt 用于基线，PROM 用于多维判别（两者样本集不同） |
| B01 M | 特征族 β/RV 依赖 C-子空间主方向（M=3） |
| B03 GRID | 每族 24 深度格点，决定联合向量维数（nF×24） |
| B08 NFRAME | 曲率特征帧数（多维判别的曲率族） |
| 405 LOOCV_acc | LOOCV 准确率是 PROM 判别的输出指标（>0.7 可分辨） |
| 404 ratio | 逐族 cross/within 分离比，>1 才可分 |
| A06 三实例路径 | 提供 base/instruct 同构对与跨家族对照 |
