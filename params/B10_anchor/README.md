# B10 anchor — 锚定层

> 组别：B 几何计算 ｜ 实测状态：🎨 设计（layer12.k_proj，设计值未落实验）｜ 来源模块：realtime
> 数据源：《参数附录表完整版》行 1618-1739；《参数完整定义与公式.txt》B10 项

## ① 定义

`B10 anchor` 定义用于 **β_t 实时追踪** 的锚定层，固定为 `layer{N//2}.k_proj`，即模型中间层的 k_proj 投影。对 Qwen 0.5B（24 层），`N//2 = 12`，锚定层为 `model.layers.12.self_attn.k_proj`。锚定层是生成过程中实时 β_t 测量的钩子目标：在一次短前向中完成校准建立 C-子空间基 U，随后在生成中持续捕获该层激活，实时计算 β_t。

## ② 公式

锚定层索引与名称：

$$\text{anchor\_layer} = \left\lfloor \frac{N}{2} \right\rfloor,\qquad \text{anchor\_name} = \texttt{"model.layers.\{N//2\}.self\_attn.k\_proj"}$$

校准（一次短前向 → C-子空间基）：

$$U = \text{PCA}(H_{cal}, M=3),\qquad H_c = H_{cal} - \overline{H}_{cal}$$

逐 token 实时集中度（β_t）：

$$\beta_t = \frac{\lVert \Pi_C h_t\rVert^2}{\lVert h_t\rVert^2} = \frac{(U^{\top}h_t)^{\top}(U^{\top}h_t)}{h_t^{\top}h_t}, \qquad \Pi_C = UU^{\top}$$

## ③ 物理/几何直觉与用途

**物理直觉：中间层是"高保真传输区"。** 三层几何结构分析表明激活流形沿深度分为三个区域：

- **浅层（0-6）＝输入编码重塑区**：β 值低（0.2-0.3），对具体 token 敏感，不适合稳定监控；
- **中层（10-14）＝高保真传输区**：β 值高（0.5-0.8），跨输入稳定，几何集中度最高——最适合做 β_t 实时追踪；
- **深层（18-23）＝输出前重组区**：β 值波动（0.3-0.6），反映输出准备而非推理状态。

选择 k_proj（而非 q/v/o）是因为 key 投影承载 token 间的结构性关联，其激活在 C-子空间（l≤2 模态）上的集中度跨输入最稳定。

**用途**：
1. 生成过程中实时测量 β_t（O(d) 开销），监控推理状态（聚焦 vs 混沌）；
2. 校准阶段用短 prompt（几个 token）建立 C-子空间基 U；
3. 支撑谱门控（G01）等实时干预的反馈信号。

## ④ 推导过程

**Step 1 — N//2 的普适性。** 对任意 Transformer 层数 N，`N//2` 定位几何结构的中部。适配表：

| 模型 | 层数 | 中间层 | anchor |
|------|:---:|:---:|------|
| Qwen 0.5B | 24 | 12 | layer12.k_proj |
| Qwen 7B | 32 | 16 | layer16.k_proj |
| GPT-2 124M | 12 | 6 | layer6.k_proj |
| Llama 7B | 32 | 16 | layer16.k_proj |
| DeepSeek-V3 | 61 | 30 | layer30.k_proj |
| GPT-3 175B | 96 | 48 | layer48.k_proj |

**Step 2 — β_t 的几何含义。** `Π_C` 把 h_t 投影到 C-子空间（前 3 主方向），`β_t` 度量"该 token 激活能量中落在低维谱结构内的比例"。β_t→1 表示激活高度集中在 C-子空间（聚焦状态）；β_t→0 表示激活散布在高维（混沌/探索状态）。β 谱指纹（102 β_max）即各层 max_t β_t。

**Step 3 — 校准-追踪两阶段。** 校准：短前向捕获锚定层激活 H_cal ∈ R^{T×d}，中心化后 PCA 取前 3 主方向得 U ∈ R^{d×3}。追踪：每生成一步捕获 h_t（钩子），计算 β_t = ||U^T h_t||²/||h_t||²。

## ⑤ 数值验证

- **实测值**：anchor=layer12.k_proj（设计状态，未落实验）；β 值实测（参考 γ 排序）：o=0.78 > k=0.72 > q=0.68；k_proj 的 γ=0.7227（白盒实测）。
- 验证脚本 [verify.py](verify.py)：验证 `N//2` 中间层索引对多模型成立，验证锚定名生成，验证校准 PCA(M=3) → U，验证 β_t = ||Π_C h_t||²/||h_t||² 数值正确性（∈[0,1] 且与解析投影一致）。

## ⑥ 与关联参数的关系

| 配合参数 | 关系 |
|---------|------|
| A01 model | 层数 N 决定中间层 N//2 |
| A04 layer_names | anchor 是层列表中的特定位点（中间层） |
| A05 projs | anchor 固定用 k_proj 投影；不同投影 β 值不同（o>k>q） |
| B01 M | 校准建立 U 用 M=3 维 C-子空间（M 是 U 的列数） |
| 101 spl_gamma | β_t 逐 token 版本；spl_gamma 是跨 token 聚合版本 |
| G01 gate_strength | 谱门控依赖实时 β_t 作为反馈信号 |
