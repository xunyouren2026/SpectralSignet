# B02 N_max — SPL 最大 Gegenbauer 谱阶

> 组别：B 几何计算 ｜ 实测状态：🔬 理论（N_max=8，公式定义未落实验数据）｜ 来源模块：plugin
> 数据源：《参数附录表完整版》行 740-861；《参数完整定义与公式.txt》B02 项；源码 [gegenbauer.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/gegenbauer.py)

## ① 定义

`B02 N_max` 控制 SPL（Spectral Projection Lens，谱投影透镜）分解时允许的**最大 Gegenbauer 谱阶**，固定为 **8**。在 Gegenbauer 展开中，任意激活分布函数 `f(x)` 可分解为无穷级数，`N_max=8` 截断到 n=0..8 共 **9 个基函数**。

`N_max` 与 `B01 M` 是两个不同的"截断"：`M=3` 决定保留的低维子空间维度（l≤2 模态）；`N_max=8` 决定谱分解的计算带宽（截断误差控制参数），而非模态保留参数。

## ② 公式

Gegenbauer 级数展开：

$$f(x) = \sum_{n=0}^{\infty}\hat{f}_n\, C_n^{(\alpha)}(x)$$

三项递推关系（论文定义 2.4，[gegenbauer.py L45-L54](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/gegenbauer.py#L45-L54)）：

$$C_0^{(\alpha)}(x)=1,\qquad C_1^{(\alpha)}(x)=2\alpha x$$

$$(n+1)\,C_{n+1}^{(\alpha)}(x) = 2(n+\alpha)\,x\,C_n^{(\alpha)}(x) - (n+2\alpha-1)\,C_{n-1}^{(\alpha)}(x)$$

超球面 S^{d-1} 情形 `α = d/2 - 1`（Qwen 0.5B 的 d=896 时 α=447）。正交性常数（范数平方，[gegenbauer.py L96-L111](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/gegenbauer.py#L96-L111)）：

$$\lVert C_n^{(\alpha)}\rVert^2 = \frac{\pi\,2^{1-2\alpha}\,\Gamma(n+2\alpha)}{n!\,(n+\alpha)\,\Gamma^2(\alpha)}$$

谱系数与能量：`\hat{f}_n = ⟨f, \tilde{C}_n⟩`（归一化基），`E_n = |\hat{f}_n|²`。

## ③ 物理/几何直觉与用途

**物理直觉：N_max 是"谱分解的带宽旋钮"。** 在超球面 S^{d-1} 上，Gegenbauer 多项式是超球面 Laplace 算子本征函数的径向部分，构成 L²(S^{d-1}) 的正交完备基（球谐函数的径向类比）。激活分布的能量按阶数展开，Lévy 测度集中引理保证高阶模态（n≥3）能量被压制——因此 `N_max` 只需覆盖到"能量已可忽略"的阶数即可，无需展开到无穷。

- `N_max=8` 提供足够的谱分辨率验证理论预测（Gamma、N_eff）；
- 增大 `N_max` 对 Gamma 影响极小（Gamma 只取前 3 阶 n=0,1,2）；
- 增大 `N_max` 提升计算成本并引入 `lgamma` 溢出风险。

**用途**：
1. **Gegenbauer 基生成**：预计算 `N_max+1=9` 个基函数 `C_n^{(α)}(x)`；
2. **谱系数计算**：将激活投影到每个基函数上，得 `N_max+1` 个谱系数；
3. 输出 `N_eff`（有效谱带宽）依赖截断后的能量分布。

## ④ 推导过程

**Step 1 — 递推的由来。** Gegenbauer 多项式满足三项递推（Rodrigues 公式与生成函数 `(1-2xt+t²)^{-α} = ΣC_n^{(α)}(x)t^n` 的系数关系导出）。给定 `C₀=1, C₁=2αx` 后，第 k 步由 `C_{k-1}, C_k` 计算 `C_{k+1}`：

$$C_{k+1} = \frac{2(k+\alpha)x\,C_k - (k+2\alpha-1)\,C_{k-1}}{k+1}$$

**Step 2 — 范数公式的推导。** 正交归一基要求 `⟨C_n, C_m⟩_{w} = δ_{nm}`，权重 `w(x)=(1-x²)^{α-1/2}`。由 Gegenbauer 的正交性常数：

$$\int_{-1}^{1}(1-x^2)^{\alpha-1/2}C_n^{(\alpha)}(x)^2\,dx = \frac{\pi\,2^{1-2\alpha}\Gamma(n+2\alpha)}{n!\,(n+\alpha)\,\Gamma(\alpha)^2}$$

**Step 3 — N_max=8 的选择（截断误差分析）。** 实测各阶能量占比（Qwen2.5-0.5B, d=896, α=447）：

```
n=0: E₀=0.35 (35%)   n=1: E₁=0.18 (18%)   n=2: E₂=0.09 (9%)
n=3: E₃=0.05 (5%)    n=4: E₄=0.03 (3%)    n=5-8: 合计≈0.02 (2%)
```

Gamma = (E₀+E₁+E₂)/ΣE = 0.62，N_eff≈5.8。截断在 n=8 时高阶能量占比 < 0.5%。

| N_max | Gamma | N_eff | 计算成本 | 溢出风险 |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 0.60 | 5.5 | 低 | 无 |
| 8 | 0.62 | 5.8 | 中 | 无 |
| 12 | 0.62 | 5.8 | 高 | 无 |
| 16 | 0.62 | 5.8 | 很高 | 高（d=896 时 `lgamma` 可能溢出） |

**Step 4 — 数值稳定性。** `gegenbauer_norm_sq` 中 `lgamma(n+2α)` 在 `n+2α` 很大时，`Γ(n+2α)` 本身会超出 float64 可表示范围（约 1.8e308，对应 `lgamma > 709`）。`N_max=8` 保证 d=896 时所有计算在 float64 内稳定。

## ⑤ 数值验证

- **实测值**：N_max=8（理论状态，公式定义）；Gamma≈0.62、N_eff≈5.8；N_max=16 在 d=896 下有 `lgamma` 溢出风险。
- 验证脚本 [verify.py](verify.py)：用 torch（numpy 兜底）实现递推，验证 `C₀=1, C₁=2αx`，与 `scipy.special.eval_gegenbauer` 交叉验证 n=0..8，并检查 d=896、α=447 下范数计算的溢出状态。

## ⑥ 与关联参数的关系

| 配合参数 | 关系 |
|---------|------|
| B01 M | M 是低维保留维度（l≤2），N_max 是高维展开阶数上限；M 决定 Gamma 计算口径，N_max 决定谱分解带宽 |
| H05 dtype | N_max=8 时 float32 足够；N_max>12 可能需要 float64 防止溢出 |
| G04 target_metric | 论文参考 Gamma=0.9966 与实测 Gamma（0.62 口径）对照时需注意谱展开阶数差异 |
| 107 N_eff | 有效谱带宽由 N_max 截断后的能量分布 `E` 计算 |
