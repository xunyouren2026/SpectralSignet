# G01 gate_strength：谱门控强度

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 4851-4956 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(G01, 第 351-357 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 76 行, 状态=理论)
> 源码：[spectral_gating.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_gating.py)、[plugin.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/plugin.py)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`G01 gate_strength`（记作 $s$）定义推理时 C-子空间（l≤2 低阶 Gegenbauer 模态）增强的强度，即**谱门控强度**，取值 $s\in[0,1]$。它控制激活向量 $H$ 被"拉向"C-子空间投影 $\Pi_C H$ 的比例，是 `SpectralCognitionPlugin` 中谱门控功能的核心旋钮。

典型取值（插件默认 `gate_strength=0.3`）：
- $s=0.0$：**纯检测模式**——只测量 Gamma，不修改激活（指纹提取/诊断）
- $s=0.3$：**标准增强模式**——激活向 C-子空间方向拉 30%，信息更集中

## ② 公式

核心公式（论文 8.3.2 谱门控，源码 [spectral_gating.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_gating.py) 第 81-86 行）：

$$H_{\text{gated}} = H + s \cdot (\Pi_C H - H)$$

其中：
- $H\in\mathbb{R}^{B\times d}$：原始激活张量（$B$=token 数, $d$=隐藏维）
- $\Pi_C H$：$H$ 在 C-子空间上的正交投影
- $s$：门控强度（gate_strength）

C-子空间的工程近似（源码第 44-79 行 `project_to_c_subspace`）：
- $n=0$ 常数模态 → 激活均值方向 $\mu = \text{mean}(H)$
- $n=1$ 偶极模态 → 第一主成分方向 $v_1$（SVD 右奇异向量）
- $n=2$ 四极模态 → 第二主成分方向 $v_2$

$$\Pi_C H = \mu + (H-\mu) V V^\top,\qquad V=[v_1, v_2]\in\mathbb{R}^{d\times 2}$$

门控的两种退化情形：
- $s=0\Rightarrow H_{\text{gated}}=H$（恒等映射，无操作）
- $s=1\Rightarrow H_{\text{gated}}=\Pi_C H$（完全投影到 C-子空间）

## ③ 物理/几何直觉与用途

**物理直觉**：高维残差系统的信息能量会坍缩到低阶模态（谱能量集中定理）。C-子空间承载"因果决策方向"，高阶模态是冗余噪声。`gate_strength` 相当于在"保持原激活"与"完全聚焦到决策方向"之间做线性插值——一个连续化的**能量集中旋钮**。

几何上，$H_{\text{gated}} = H + s(\Pi_C H - H)$ 是把每个 token 的激活沿"指向 C-子空间的连线"移动 $s$ 的比例：
- $s$ 越大，激活越贴向低维子空间 → Gamma（前 3 主成分能量占比）越高
- 但 $s$ 过大（如 1.0）会把激活完全压到 2 维平面内，破坏生成多样性，输出可能跳变（实测出现绕口令现象）

**用途**：
| 强度 | 模式 | 效果 | 适用场景 |
|---|---|---|---|
| 0.0 | 纯检测 | 只测量，不修改 | 指纹提取/诊断 |
| 0.1-0.2 | 轻度增强 | Gamma 微升，输出稳定 | 质量敏感场景 |
| 0.3 | 标准增强 | Gamma 明显提升，输出可接受 | 通用增强 |
| 0.5+ | 激进增强 | Gamma 大幅提升，可能改变输出 | 实验探索 |

## ④ 推导过程

**推导 1：门控公式的等价形式**。设残差 $R = H - \Pi_C H$（$R\perp$ C-子空间，即与 $V$ 正交）。则：

$$H_{\text{gated}} = H + s(\Pi_C H - H) = (1-s)H + s\,\Pi_C H = \Pi_C H + (1-s)R$$

**推导 2：与 C-子空间的距离收缩**。$\|\cdot\|_F$ 为 Frobenius 范数：

$$\|H_{\text{gated}} - \Pi_C H\|_F = (1-s)\|H - \Pi_C H\|_F$$

即门控把激活到 C-子空间的"垂直距离"压缩为原来的 $1-s$。$s=0.3$ 时垂直分量保留 70%，$s=1.0$ 时归零。

**推导 3：Gamma 变化方向**。设 C-子空间承载方差占比 $\Gamma = \sum_{i\le 3}s_i^2/\sum_i s_i^2$（奇异值平方口径）。投影压缩垂直残差后，谱能量向低阶模态集中，故门控后 Gamma 单调不减：

$$\Gamma(H_{\text{gated}}) \geq \Gamma(H),\qquad \forall s\in[0,1]$$

$s=0$ 时取等号。这一性质正是"门控提升 Gamma"的机理，也是 verify.py 的验证目标之一。

**推导 4：为何选 0.3**。设增强后 Gamma 增量近似线性于 $s$（实测 Qwen 0.5B）：$s=0.1\to+0.01$, $s=0.3\to+0.03$, $s=0.5\to+0.06$, $s=1.0\to+0.15$。0.3 落在"可观测提升 $\approx +0.02\sim0.05$"且"输出分布不变"的交集内。

## ⑤ 数值验证

**实测状态**：G01 = 理论（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 76 行：公式完整，设计实现于源码；T07 谱门控增强为设计公式）。

**实测值（主文档第 4946-4949 行，Qwen2.5-0.5B-Instruct，锚定层 layer12.k_proj）**：

| gate_strength | Gamma | 增量 |
|---|---|---|
| 0.0（检测） | 0.1694 | — |
| 0.3（增强） | 0.1901 | +0.0207 |
| 0.5 | 0.2031 | +0.0337 |
| 1.0 | 0.3241 | +0.1547（输出跳变） |

结论：0.3 是最佳平衡点——提升 Gamma 且不破坏输出分布。

**合成数值验证（[verify.py](verify.py)）**：
- $s=0$：`H_gated == H`（max|Δ|=0，精确无操作）✅
- $s=0.3$：验证插值性质 $\|H_{\text{gated}}-\Pi_C H\|_F = 0.7\|H-\Pi_C H\|_F$，Gamma 提升 ✅
- $s=1$：`H_gated == Π_C H`（完全投影），残差 $H-\Pi_C H$ 与主方向 $v_1,v_2$ 正交 ✅
- Gamma 随 $s$ 单调不减 ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| B01 M | C-子空间投影依赖主方向数 M=3（均值+2 主成分），门控即投影叠加 |
| C01 energy_thr | 门控本质是强制能量向低秩方向集中，与 energy_thr 协同描述谱能量分布 |
| 101 spl_gamma | 门控的直接观测对象：Gamma 增量随 $s$ 增大 |
| T07 谱门控增强 | 同公式的阈值面：$H_{\text{gated}}=H+s(\Pi_C H-H)$，$s=0.3$ |
| G02 keep_ratio | 门控提升 C-子空间集中度后，token 级 Gamma_i 排序更可信，影响剪枝效果（未验证 K% 变化） |
| H05 dtype | 门控内部用 float32 计算后转回原 dtype（源码第 119-121 行），避免破坏残差连接与数值稳定性 |
