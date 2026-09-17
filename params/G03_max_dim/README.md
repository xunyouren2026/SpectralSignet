# G03 max_dim：跳过超大输出层阈值

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5188-5308 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(G03, 第 367-374 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 78 行, 状态=理论)
> 源码：[plugin.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/plugin.py)（第 48-56、115-117 行 max_dim 检查）
> 验证脚本：[verify.py](verify.py)

## ① 定义

`G03 max_dim` 定义自动层检测中**跳过超大输出层的维度阈值**，固定为 **8192**。在自动层检测时，系统遍历模型的所有线性/卷积子模块注册前向钩子；若某层输出维度 $d > \text{max\_dim}$，则跳过该层、不注册钩子，防止对超大层（如 `lm_head` 输出层，词表维度可达数万）做谱分析导致内存溢出（OOM）。

```text
检查：该层的输出维度 d = 152064（lm_head）
判断：d > max_dim → 跳过
结果：不捕获该层，内存安全
```

## ② 公式

$$\text{capture}(d) = \begin{cases} \text{True}, & d \le \text{max\_dim} \\ \text{False}, & d > \text{max\_dim} \end{cases}$$

内存风险源头——协方差矩阵：

$$C = \frac{H^\top H}{B-1} \in \mathbb{R}^{d\times d},\qquad \text{Memory}(C) = d^2 \times 4\ \text{bytes (float32)}$$

对 `lm_head`（$d=152064$）：$152064^2 \times 4 \approx 92.5$ GB——单层协方差矩阵即超过任何单卡显存，必然 OOM。（主文档第 5229 行估算为 86.5GB，系采用不同词表尺寸口径；本处按文档给出的 152064 计算。）

## ③ 物理/几何直觉与用途

**物理直觉**：谱分析（Gamma、SVD）的复杂度随维度 $d$ 的**平方**（协方差）至**立方**（特征分解）增长。`lm_head`/嵌入层本质是"词表 → 特征"的巨型投影，其维度由词表大小决定（数万），远超模型隐藏维（数百到数千）。这些层的激活位于 $\mathbb{R}^{152064}$ 这样的超高维空间，对它们做 SVD 既无几何意义（谱集中度被维数稀释）又必然内存爆炸。`max_dim` 是**以维度为判据的防御性截断**。

**用途**（内存安全保护机制）：

| 层类型 | 典型维度 | 是否跳过 | 原因 |
|---|---|---|---|
| k_proj / up_proj / o_proj / gate_proj / down_proj | 896 | ✅ 捕获 | 正常层 |
| LLaMA 隐藏层 | 4096 | ✅ 捕获 | 正常层 |
| DeepSeek-V3 隐藏层 | 7168 | ✅ 捕获 | 正常层（保守 8192 仍覆盖） |
| lm_head | 152064 | ❌ 跳过 | 防 OOM |
| 嵌入层 | 152064 | ❌ 跳过 | 防 OOM |

边界行为：$d = 8192$ 时 `8192 > 8192` 为 False → **捕获**；$d = 8193$ → **跳过**。

## ④ 推导过程

**推导 1：为何 8192 恰好分隔两类层**。所有主流模型隐藏维都远小于 8192（Qwen 0.5B=896、GPT-2=768、LLaMA=4096、DeepSeek-V3=7168），而所有词表投影层都远大于 8192（152064、50257、32000、129280）。8192 落在两者之间的空白区间：

$$\max(\text{hidden dim}) = 7168 < 8192 < 152064 = \min(\text{lm\_head dim})$$

因此 8192 是"排除 lm_head 而非排除正常层"的保守上界。

**推导 2：OOM 的维度来源**。谱分析对激活 $H\in\mathbb{R}^{B\times d}$（$B\approx256$ 样本）构造协方差 $C=H^\top H/(B-1)\in\mathbb{R}^{d\times d}$。内存随 $d$ 二次增长：

$$\frac{\text{Memory}(d=152064)}{\text{Memory}(d=896)} = \left(\frac{152064}{896}\right)^2 \approx 2.88\times 10^4$$

即 lm_head 的协方差内存是正常层的约 2.9 万倍——这就是必须用 `max_dim` 硬截断而非"优化计算"的原因。

**推导 3：与 SVD 策略的互补**。即使采用"高维低样本直接对 $H$ 做 SVD"（[spectral_analysis.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_analysis.py) 第 96-102 行，避免构造 $d\times d$），$B\times d$ 的 SVD 本身在 $d=152064$ 时开销依然过高且 Gamma 无鉴别力，`max_dim` 从源头跳过。

## ⑤ 数值验证

**实测状态**：G03 = 理论（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 78 行：设计逻辑明确，未专门验证 8192 普适性）。源码 [plugin.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/plugin.py) 第 115-117 行已实现。

**实测值（主文档第 5296-5306 行，Qwen2.5-0.5B-Instruct）**：
- 捕获 48 层（k/up/o/gate/down，d=896）
- 跳过 lm_head（d=152064）
- 内存占用：无 OOM，正常运行

**合成数值验证（[verify.py](verify.py)）**：
- 构造典型层维度表（Qwen/GPT-2/LLaMA/DeepSeek 隐藏层 + lm_head/嵌入层），逐一断言捕获/跳过 ✅
- 边界检查：d=8192 捕获、d=8193 跳过 ✅
- 协方差内存计算：d=896 ≈ 3.2 MB（安全），d=152064 ≈ 92.5 GB（OOM）✅
- 模拟 `_auto_layer_names` 逻辑返回正确的目标层列表 ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| A04 layer_names | 若手动指定层名，max_dim 不生效；若 auto 自动检测则生效 |
| F01 ctx | 两者共同防 OOM：ctx 控制 KV 缓存内存，max_dim 控制协方差矩阵内存 |
| H02/H03 | 线程数限制也用于防 OOM；三者共同构成内存安全体系 |
| G02 keep_ratio | G02 作用于 token 维度（seq_len），G03 作用于层维度（d），正交 |
| H05 dtype | float32 下协方差矩阵 4 字节/元素；若用 float16 内存减半，但谱分析精度受威胁 |
