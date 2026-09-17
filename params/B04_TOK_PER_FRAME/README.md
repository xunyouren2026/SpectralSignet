# B04 TOK_PER_FRAME — 每帧 token 数

> 组别：B 几何计算 ｜ 实测状态：✅ 已用（TOK=8）｜ 来源模块：whitebox / ai_versionA
> 数据源：《参数附录表完整版》行 995-1116；《参数完整定义与公式.txt》B04 项

## ① 定义

`B04 TOK_PER_FRAME` 定义每帧（frame）包含的 token 数量，固定为 **8**。在曲率分析中，生成过程被划分为若干帧，每帧聚合 8 个 token 的激活进行统计，而不是逐 token 计算曲率。

**为什么要分帧？** 曲率是流形的几何性质，需要局部邻域内的多个点才能计算——单个 token 激活 `h_t∈R^d` 只是一个点，无法定义曲率。每帧 8 个 token 提供了采样池（配合 `B05=18` 有放回采样），足以拟合 2D 主截面并估算曲率。

## ② 公式

总帧数：

$$N_{frame} = \frac{\text{gen\_len}}{\text{TOK\_PER\_FRAME}}$$

帧聚合（每帧的曲率指标在帧内 token 上统计）：

$$\text{DEFF}_f = \text{mean}_{t\in frame_f}\, \text{DEFF}(\kappa_1^{(t)},\kappa_2^{(t)})$$

帧间波动（非稳态告警指标 T02）：

$$\text{CV} = \frac{\text{std}(\text{DEFF}_f)}{\text{mean}(\text{DEFF}_f)}$$

## ③ 物理/几何直觉与用途

**物理直觉：TOK_PER_FRAME 是"时间分辨率旋钮"。** 生成过程中流形几何随 token 演化，分帧把连续生成流离散成帧序列：

- 帧数太少 → 无法观察生成过程中的快速变化（时间分辨率低）；
- 帧内 token 太少 → 采样点不足，曲率估计方差大（统计稳定性低）；
- `TOK=8` 在二者间平衡：64 token=8 帧、96 token=12 帧、256 token=32 帧。

**关键洞察**：最终曲率指纹是"帧级聚合值"而非逐 token 原始数据，`TOK_PER_FRAME` 直接控制聚合粒度与曲率曲线的分辨率。

**用途**：
1. 决定帧数 `gen_len / TOK_PER_FRAME`；
2. 提供每帧采样池（8 token）供 `B05` 做 18 次有放回采样；
3. 影响曲率统计收敛（帧间 CV）：TOK=4→CV 3.2%、TOK=8→CV 1.5%、TOK=16→CV 1.1%。

## ④ 推导过程

**Step 1 — 帧内平均的统计效应。** 设每 token 的瞬时曲率量 `q_t = μ + ε_t`，`ε_t~N(0,σ²)` 独立同分布。帧均值 `q̄_f = (1/TOK)Σ_{t∈f}q_t` 满足：

$$\text{Var}(\bar{q}_f) = \frac{\sigma^2}{\text{TOK}},\qquad \text{CV} \propto \frac{1}{\sqrt{\text{TOK}}}$$

故 TOK 越大，帧序列越平滑、帧间 CV 越小（文档实测 3.2% → 1.5% → 1.1% 单调递减）。

**Step 2 — DEFF 均值对 TOK 的不敏感性。** DEFF 是主曲率角的函数（`DEFF = 1 + sin 2φ`），其期望是流形几何的固有量，与分帧粒度无关——故三种 TOK 下 DEFF 均值都 ≈1.58。TOK 只影响估计方差，不影响期望。

**Step 3 — TOK 取值的权衡表。**

| TOK | 帧数(gen_len=96) | 时间分辨率 | 统计稳定性 | CV |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 24 | 高 | 低 | 3.2% |
| 8 | 12 | 中 | 高 | 1.5% |
| 16 | 6 | 低 | 高 | 1.1% |

## ⑤ 数值验证

- **实测值**：TOK=8（已用）；64 token=8 帧、96 token=12 帧、256 token=32 帧；DEFF 均值≈1.58（与 TOK 无关），末帧 K% 45%；生成速度快的模型可支持更大 TOK。
- 验证脚本 [verify.py](verify.py)：验证 `gen_len/TOK` 整除关系与帧数，合成每 token 曲率噪声演示"帧均值方差 ∝ 1/√TOK、CV 随 TOK 递减、DEFF 均值不变"。

## ⑥ 与关联参数的关系

| 配合参数 | 关系 |
|---------|------|
| A03 gen_len | 总帧数 = gen_len / TOK_PER_FRAME，共同决定曲率统计的样本量 |
| B05 SAMPLES_PER_FRAME | 每帧采样点数量，与 TOK 共同决定统计稳定性 |
| B08 NFRAME | 用于曲率特征的帧数，TOK 决定每帧粒度；NFRAME 取前几帧 |
| 202 DEFF_A | DEFF 在每帧上计算，帧数越多 DEFF 曲线越精细 |
