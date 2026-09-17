# G04 target_metric：参考 Gamma（论文值）

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5311-5420 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(G04, 第 375-381 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 79 行, 状态=对照)
> 源码：[spectral_analysis.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_analysis.py)（SPL Gamma 实现）
> 验证脚本：[verify.py](verify.py)

## ① 定义

`G04 target_metric` 定义论文中报告的参考 Gamma 值，固定为 **0.9966**。该值来自论文核心发现——**Post-Norm 残差网络充分训练后，其激活在 C-子空间的能量占比可高达 0.99 以上**。它作为"参照点"，用于对比当前实测模型的 Gamma，判断其几何行为是接近 Post-Norm（凝聚型）还是 Pre-Norm（弥散型）。

$$\Gamma_{\text{target}} = 0.9966$$

## ② 公式

Gamma（C-子空间能量占比，前 $M=3$ 主成分）：

$$\Gamma = \frac{\sum_{i\le 3} s_i^2}{\sum_i s_i^2}\quad(\text{能量口径})\qquad \text{或}\qquad \Gamma = \frac{\sum_{i\le 3} s_i}{\sum_i s_i}\quad(\text{SPL 奇异值口径, 见源码})$$

其中 $s_i$ 为中心化激活矩阵 $H_c = H-\bar{H}$ 的奇异值（SVD）。target_metric 即论文口径下的参考值 $0.9966$。

对比判定（主文档第 5373-5383 行）：

$$\text{arch} = \begin{cases} \text{Post-Norm 凝聚型}, & \Gamma > 0.90 \\ \text{混合/早期架构}, & 0.50 < \Gamma \le 0.90 \\ \text{Pre-Norm 弥散型}, & 0.20 < \Gamma \le 0.50 \\ \text{高度弥散型}, & \Gamma \le 0.20 \end{cases}$$

## ③ 物理/几何直觉与用途

**物理直觉**：谱能量集中定理指出，高维残差系统的信息能量会坍缩到 $l\le2$ 的低阶 C-子空间。Post-Norm 架构（残差在归一化之后加回）经过充分训练后，激活几乎全部落在低维子空间内，Gamma 趋近 1（论文值 0.9966）；而 Pre-Norm 架构（归一化在残差之前，现代 LLM 标配）激活高度弥散，Gamma 显著低（0.12-0.46）。因此 **Gamma 是架构几何的"分水岭"**，target_metric 是这个分水岭的 Post-Norm 端参照。

**用途**：作为**参照基准**用于对比验证：

| 实测 Gamma | 与 target 对比 | 判定 |
|---|---|---|
| ≈0.99 | 接近 | Post-Norm 型（凝聚） |
| ≈0.16-0.46 | 显著更低 | Pre-Norm 型（弥散） |
| 约 0.3-0.5 | 中等 | 混合型 |

## ④ 推导过程

**推导 1：Gamma 的谱几何来源**。激活矩阵 $H\in\mathbb{R}^{B\times d}$ 中心化后做 SVD $H_c = U\Sigma V^\top$。奇异值平方 $s_i^2$ 即各主方向承载的方差。C-子空间取前 3 个主方向，故 Gamma = 前 3 个主方向的方差占比。若数据近似落在一个 3 维子空间内，则 $s_4,\dots\to 0$，Gamma $\to 1$。

**推导 2：Post-Norm 为何 Gamma→1（论文口径 0.9966）**。Post-Norm 残差块 $x_{l+1} = \text{LN}(x_l + f_l(x_l))$，残差分支 $f_l$ 的低秩结构使能量沿少数方向累积；充分训练后激活流形被"压扁"到低维 C-子空间附近，垂直弥散分量趋于零：

$$\Gamma = 1 - \frac{\sum_{i>3}s_i^2}{\sum_i s_i^2} \approx 1 - \varepsilon,\qquad \varepsilon \ll 1$$

论文报告充分训练后 $\Gamma \approx 0.9966$，即 $\varepsilon \approx 0.0034$。

**推导 3：Pre-Norm 为何 Gamma 低**。Pre-Norm $x_{l+1} = x_l + f_l(\text{LN}(x_l))$ 中归一化不断"重缩放"激活，抑制能量坍缩，激活在 $d$ 维空间中近似各向同性弥散。若奇异值近似均匀 $s_i \approx \sigma$，则（能量口径）：

$$\Gamma \approx \frac{3\sigma^2}{d\sigma^2} = \frac{3}{d}$$

对 $d=896$（Qwen 0.5B）理论值 $\approx 0.0033$，但实测 0.16 说明仍有部分结构——量级上远低于 0.9966，符合"弥散型"判定。

## ⑤ 数值验证

**实测状态**：G04 = 对照（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 79 行：论文参考值，作为对照基准而非可调参数）。

**实测对照表（主文档第 5385-5394 行）**：

| 模型 | 实测 Gamma | 相对 target | 判定 |
|---|---|---|---|
| ResNet | 0.9992 | +0.26% | 达到/超过论文水平 |
| target_metric | 0.9966 | 0% | 论文参考 |
| GPT-2 | 0.457 | -54% | Pre-Norm 早期架构 |
| Qwen 0.5B | 0.1625 | -84% | Pre-Norm 现代架构 |
| RWKV-5 | 0.1237 | -88% | 纯 RNN 弥散型 |

**合成数值验证（[verify.py](verify.py)）**：
- 构造"Post-Norm 型"近 3 维数据（低秩信号 + 极弱噪声）→ Gamma ≈ 0.99，与 0.9966 相对差距 < 5% ✅
- 构造"Pre-Norm 型"弥散数据（各向同性）→ Gamma ≈ 3/d，远低于 target ✅
- 架构类型判定器（>0.90 / >0.50 / >0.20 阈值）对论文实测值逐一判定正确 ✅
- 对合成 Post-Norm 数据与 target 0.9966 的对比：差距标注 ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| 101 spl_gamma | 实测 Gamma 值，与 target_metric 对比（注意口径：审计实测 0.625 为特定高集中度投影口径，全模型平均 0.16-0.31） |
| B01 M | Gamma 取前 M=3 主成分，M 决定 Gamma 数值 |
| T01 DEFF 平台 | DEFF 平台告警与 target_metric 同属"参照系"（几何参照 vs 谱参照） |
| G01 gate_strength | 门控提升实测 Gamma（Qwen 0.17→0.19-0.32），向 target 方向逼近 |
| G05 max_ctx / H 组 | 与参照基准无关的工程参数，不影响 target 值本身 |
