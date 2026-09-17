# G02 keep_ratio：token 保留比例

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 4957-5188 行（存在两处 G02 章节：4957 与 5073，内容一致）
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(G02, 第 358-365 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 77 行, 状态=理论)
> 源码：[spectral_pruning.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_pruning.py)、[spectral_analysis.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_analysis.py)（Gamma_i 定义）
> 验证脚本：[verify.py](verify.py)

## ① 定义

`G02 keep_ratio` 定义谱记忆（`SpectralMemory`）与谱剪枝（`SpectralTokenPruner`）模块中 **token 的保留比例**，固定为 **0.5（50%）**。系统根据每个 token 在 C-子空间的能量占比 `Gamma_i` 降序排序，保留前 `keep_ratio` 比例的 token、丢弃其余。

## ② 公式

每个 token 的重要性 = 其在 C-子空间的能量占比（108 Gamma_i，与 [spectral_analysis.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_analysis.py) 的逐 token 投影一致）：

$$\Gamma_i = \frac{\|\Pi_C h_t\|^2}{\|h_t\|^2},\qquad t=1,\dots,T$$

保留策略：

```python
keep = int(seq_len * keep_ratio)          # 0.5 → 保留 50%
sorted_indices = np.argsort(gamma_i)[::-1]  # 按 Gamma_i 降序
kept_indices   = sorted_indices[:keep]      # 前 keep 个保留
dropped_indices = sorted_indices[keep:]     # 其余丢弃
```

## ③ 物理/几何直觉与用途

**物理直觉**：C-子空间承载因果决策信息。`Gamma_i` 衡量第 $t$ 个 token 的激活有多少能量落在 C-子空间内——`Gamma_i` 高的 token 是"决策核心 token"，低的则是冗余/填充 token。按 `Gamma_i` 排序保留，等价于**在 token 维度上做谱能量驱动的注意力剪枝**：丢弃的信息是最不"聚焦"的。

**用途**（压缩强度控制旋钮）：

| 保留比例 | 压缩强度 | 精度风险 | 适用场景 |
|---|---|---|---|
| 0.7 | 弱 | 几乎无 | 保守压缩 |
| 0.5 | 中等 | 低 | 标准配置 |
| 0.3 | 强 | 中 | 激进压缩 |
| 0.1 | 极强 | 高 | 极限测试 |

核心权衡：压缩越多省 token 越多，但信息损失风险越大。`keep_ratio=0.5` 是"省 token"与"保精度"之间的平衡点。

## ④ 推导过程

**推导 1：Gamma_i 的几何含义**。设 $\Pi_C$ 为到 C-子空间（$\dim C = 3$）的正交投影，则每个 token 的激活可分解为 $h_t = \Pi_C h_t + r_t$，其中 $r_t\perp C$。由正交性：

$$\|h_t\|^2 = \|\Pi_C h_t\|^2 + \|r_t\|^2 \;\Rightarrow\; \Gamma_i = 1 - \frac{\|r_t\|^2}{\|h_t\|^2}$$

即 $\Gamma_i$ 直接度量 token 激活"落在 C-子空间外的比例"。$\Gamma_i\to 1$ 表示激活几乎全在 C-子空间内（高度聚焦）。

**推导 2：谱剪枝 vs 随机剪枝的精度差异来源**。设标签 $y$ 与 C-子空间方向强相关。随机剪枝等概率丢弃 token，等于以 $1-\rho$（$\rho$=保留率）的概率丢失信息，期望信息损失与 $\rho$ 线性相关；谱剪枝丢弃的是 $\Gamma_i$ 最小的 token（C-能量占比最低），保留的信息量近似为：

$$\text{Retained} = \sum_{t\in\text{kept}}\Gamma_i \;\approx\; \rho \cdot T \cdot \mathbb{E}[\Gamma_i \mid \Gamma_i\ge q_\rho] \;\geq\; \rho \cdot T \cdot \bar{\Gamma}$$

因此谱剪枝在相同保留率下保留更多 C-子空间能量，故精度更高。实测：keep_ratio=0.3 时谱剪枝 100→30 token 准确率仍 1.0000，随机剪枝跌到 0.9220（损失 7.8%）。

**推导 3：为何 0.5 是安全值**。若 token 级 $Gamma_i$ 分布的 C-能量高度集中于高 $\Gamma_i$ 的 top 子集，则保留 top 50% 即可覆盖绝大多数 C-子空间能量：

$$\frac{\sum_{t\in\text{top }50\%}\Gamma_i}{\sum_t \Gamma_i} \approx 0.9\sim1.0$$

实测 50% 保留下准确率仍 1.000（0% 相对损失）。

## ⑤ 数值验证

**实测状态**：G02 = 理论（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 77 行：定义与公式完整，未在本次会话实测）。

**实测值（主文档第 5025-5033、5063-5068 行）**：

| keep_ratio | 保留 token | 准确率 | 相对损失 |
|---|---|---|---|
| 1.0 | 100 | 1.000 | 0% |
| 0.5 | 50 | 1.000 | 0% |
| 0.3 | 30 | 1.000 | 0% |
| 0.2 | 20 | 0.980 | 2% |
| 0.1 | 10 | 0.950 | 5% |

随机剪枝对比：keep_ratio=0.3 时准确率跌至 0.9220（-7.8%）。

**合成数值验证（[verify.py](verify.py)）**：
- 合成 200 token、含低秩结构的激活，计算逐 token Gamma_i ✅
- keep=int(200×0.5)=100，保留集合恰为 Gamma_i 最高的 100 个 ✅
- 保留集合平均 Gamma_i 显著高于丢弃集合 ✅
- 谱剪枝保留的 C-子空间能量占比 ≥ 随机剪枝（多随机种子平均）✅
- 参数敏感性：keep_ratio=0.3/0.5/0.7 时保留数 = int(T×ratio) ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| 108 Gamma_i | token 重要性是 keep_ratio 排序的直接依据 |
| C01 energy_thr | 两者共同描述谱能量的分布特性（能量阈值 vs 保留比例） |
| G01 gate_strength | 门控提升 C-子空间集中度后，Gamma_i 排序更可信，剪枝更精准 |
| G03 max_dim | 剪枝作用于 token 维度（seq_len），与层维度（d）无关，二者正交 |
| B01 M | C-子空间维数 M=3 决定 Gamma_i 计算的投影算子 |
| 106 ratio | 谱压缩比（head_dim/k90）与 keep_ratio 同属"压缩强度"坐标系 |
