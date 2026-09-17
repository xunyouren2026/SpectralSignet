# E02 REWRITES — 改写扰动数

> 组别：E 扰动/鲁棒 ｜ 实测状态：✅ 已用（rewrite n=12）｜ 来源模块：robustness
> 数据源：《参数附录表完整版》行 3443-3551；《参数完整定义与公式.txt》E02 项；源码 [_qwen_robustness.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_qwen_robustness.py#L45-L50)

## ① 定义

`E02 REWRITES` 定义用于测试**改写扰动（rewrite perturbation）**的样本数量，固定为 **4 句**。

改写是文本扰动家族的第一类：**同一语义、不同措辞**。系统对每个参考 prompt 生成若干语义相同但句法/措辞不同的版本，计算改写后文本的指纹与基线指纹的距离（MAD），验证指纹是否在改写扰动下保持稳定。由于有 3 个溯源实例（Qwen base / Qwen Instruct / GPT-2），**4 句改写 × 3 实例 = 12 个测试样本（n=12）**。

```python
REWRITES = [
    "The universe is expanding, galaxies drifting apart over time.",
    "Galaxies are moving away from each other as the universe expands.",
    "Over time, the universe has been expanding, causing galaxies to drift apart.",
    "The expansion of the universe causes galaxies to separate from each other."
]
```

（源码中为 4 条与参考句完全不同的独立改写句。）

## ② 公式

设基线指纹 `baseline = mean(REF_PROMPTS profiles)`（E01），改写样本指纹为 `gⱼ`。改写扰动距离用 **MAD（Mean Absolute Deviation，平均绝对偏差）**：

$$\text{MAD} \;=\; \frac{1}{J}\sum_{j=1}^{J}\Bigg[\frac{1}{d}\sum_{k=1}^{d}\big|g_{jk} - \text{baseline}_k\big|\Bigg]$$

家族判定（48 维特征的 L1 最近邻，源码 `family_dist`）：

$$d_Q = \min\Big(\overline{|g - \bar{f}_{\text{qwen\_b}}|},\; \overline{|g - \bar{f}_{\text{qwen\_i}}|}\Big),\qquad
d_G = \overline{|g - \bar{f}_{\text{gpt2}}|}$$

$$\text{pred} = \begin{cases} \text{Qwen0.5B} & d_Q \le d_G \\ \text{GPT2} & d_Q > d_G \end{cases}$$

其中 `\bar{f}` 为各实例参考簇基线（E01 流程对 REF_PROMPTS 取均值），`\overline{|·|}` 表示逐维绝对差取平均（即 L1/MAD 距离）。

## ③ 物理/几何直觉与用途

**物理直觉：改写扰动是"身份流形上的小扰动"。** 指纹刻画的是模型的**结构身份**（β 谱能量集中 + 曲率特征），而非文本的内容。改写保留语义、只改变措辞，理想情况下指纹应几乎不动——就像同一物体的不同照片，形状不变。若指纹对措辞敏感，说明它混入了"文本内容噪声"，身份信号不纯。

改写扰动覆盖三种措辞变换：
- **同义替换**：改变措辞但保留语义 → 指纹变化应最小；
- **句式变换**：主动/被动语态互换 → 指纹变化应最小；
- **措辞调整**：同义词替换 → 指纹变化应最小。

**用途：** 验证指纹对**语言表面形式**的鲁棒性。改写是溯源鲁棒性的第一道门槛——若连措辞变化都能改变家族归属，指纹不可信。

## ④ 推导过程

**Step 1 — 为什么用 MAD（L1）而非欧氏距离。** 48 维特征中 β 谱（∈[0,1]）与曲率特征（归一化后 ∈[0,1]）量纲一致但分布不同。L1 距离对逐维离群点更鲁棒（不像 L2 会被大偏差平方放大），且天然可解释为"每维平均偏差"。实测改写样本 `MAD(Qwen)=0.040`，即每个特征维度平均偏离基线 `4%`。

**Step 2 — 为什么取 min（到 qwen_b/qwen_i 的较小者）。** Qwen base 与 Qwen Instruct 同属 Qwen0.5B 家族，家族判定应把两者视为**同一参考簇**。取 `min` 等价于"最近邻家族成员"，允许样本漂向家族内任一成员；`min` 还使改写在 base/Instruct 间的微小差异不误伤家族判定。

**Step 3 — 改写样本数的鲁棒性对比（实测）：**

| 改写样本数 | MAD | 溯源准确率 | 含义 |
|:---:|:---:|:---:|:---:|
| 2 句 | 0.035 | 100% | 样本偏少 |
| **4 句** | **0.040** | **100%** | **标准配置** |
| 8 句 | 0.042 | 100% | 更全面但耗时加倍 |

4 句已是标准配置：MAD 稳定在 0.04 附近，溯源 100%，更多样本边际收益递减。

## ⑤ 数值验证

- **实测值**：`MAD(Qwen)=0.040`，改写扰动下溯源准确率 `100%`，指纹对语义改写鲁棒。
- 实测环境：MVI-4 鲁棒性实验，48 维特征（β24 + 版本 A 曲率 24），L1 最近邻家族判定。
- 验证脚本 [verify.py](verify.py)：合成 48 维特征（β24 + 曲率 24），构造 Qwen 双参考簇与 GPT-2 参考簇；4 句改写样本在 Qwen 基线附近加小扰动（MAD≈0.040）；用 L1 最近邻复现 4 句→n=12 的家族判定，验证 Qwen vs GPT2 距离判别 100% 正确。**合成数据仅验证算法逻辑，数值对齐实测量级。**

## ⑥ 与关联参数的关系

| 配合参数 | 关系 |
|---------|------|
| E01 REF_PROMPTS | 提供基线指纹（改写样本 MAD 的比对参照） |
| E03 trunc | 改写（保留语义）与截断（保留部分原文）同属文本扰动测试集 |
| E04 GEN_PROMPT | 改写与重生成（全新内容）共同构成"内容保留→内容重造"的扰动谱 |
| B08 NFRAME | 叠加曲率特征帧数（8 帧×3 = 24 维），与 β24 拼成 48 维溯源特征 |
| B09 n_sample | 曲率每帧采样数，影响曲率特征稳定性 |
| 401 溯源正确率 | 改写样本的家族判定正确率（100%） |
