# F01 ctx：KV 测试上下文长度

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 4128-4250 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(F01, 第 302-308 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 70 行, 状态=已用)
> 源码：[._local_whitebox_detect.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_local_whitebox_detect.py)（`--ctx` 参数、KV 降档循环）
> 实测日志：[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`F01 ctx` 定义 KV 缓存（key-value cache）测试的目标上下文长度，默认 **4096**，内存不足时自动降档到 **1024**。它控制 `_local_whitebox_detect.py` 第 ② 步"KV cache 字节实测"的输入规模：构造指定 ctx 长度的输入做一次前向传播，测量实际 KV 缓存字节数与 KV/权重比。

```python
ap.add_argument("--ctx", type=int, default=4096)   # 源码第 55 行
```

实际运行中使用 `--ctx 1024`（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log) 第 4 行），实测输出 `seq=1014 (ctx实际=1024) → KV=24.9MB`。

## ② 公式

KV 缓存总字节数（每层 K 与 V 各一份，每 token 需要 n_kv 个头 × head_dim 维 × 精度字节数）：

$$KV_{bytes} = 2 \times L \times n_{kv} \times h_d \times ctx \times 4$$

其中：L=层数（Qwen2.5-0.5B 为 24），n_kv=KV 头数，h_d=每头维度，4=float32 字节数（float16 则为 2）。

每 token 的 KV 成本（架构常数）：

$$kv_{per\_tok} = 2 \times L \times n_{kv} \times h_d \times \text{dtype\_bytes}$$

**重要口径标注**：主文档第 4140-4143 行示例取 n_kv=8、h_d=112，得到 $kv_{per\_tok}=2\times24\times8\times112\times4=172{,}032\text{ B}\approx168\text{KB/tok}$；但实测日志显示 Qwen2.5-0.5B-Instruct 真实配置为 **24L / 2KV / 64hd**（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log) 第 14 行 `arch: 24L 2KV 64hd`），故：

$$kv_{per\_tok}^{实际} = 2\times24\times2\times64\times4 = 24{,}576\text{ B} \approx 24.6\text{KB/tok}$$

主文档 168KB/tok 对应的是 n_kv=8/h_d=112 的示例配置（疑为笔误或他型号口径），与实测 24.6KB/tok 相差约 7 倍。本 README 与 verify.py 以实测口径为主、主文档口径为辅，并如实标注差异。

自动降档规则（源码第 96-111 行：OOM 减半重试）：

$$ctx_{actual} = \max\{ctx \in \{4096, 2048, 1024, \dots, 64\} : \text{前向不触发 OOM}\}$$

## ③ 物理/几何直觉与用途

**物理直觉**：自回归解码时，每个已生成 token 的 key/value 向量必须常驻内存，供后续每一步 attention 查询。KV 缓存是"注意力的凝固历史"——上下文越长，保留的历史点越多，内存线性增长。KV 是长上下文推理的"显存地板"：模型权重一次加载即可，KV 却随对话长度持续累积。

几何视角：KV 缓存正是谱指纹（Γ、C-子空间）采样的对象空间。`ctx` 决定流形（激活轨迹）上被"物化"的历史点数——点数越多，曲率采样与谱集中度统计越充分，指纹越稳；但点数越多，物理内存压力越大。`F01 ctx` 就是这两个目标的折中旋钮。

| ctx 值 | KV（168KB/tok 口径） | KV（24.6KB/tok 口径） | 适用场景 |
|---|---|---|---|
| 1024 | 172MB | 25.2MB | CPU 安全运行（实测 24.9MB） |
| 2048 | 344MB | 50.3MB | CPU 可运行 |
| 4096 | 688MB | 100.7MB | 需降档或 GPU |
| 8192 | 1.38GB | 201.3MB | 需 GPU |

## ④ 推导过程

**推导 1：kv_per_tok 的由来**。每层注意力保留两组缓存：K 缓存与 V 缓存，故因子 2。每个 token 在每层产生 n_kv 个 key（维度 h_d）与 n_kv 个 value（维度 h_d），故每层每 token 占用 $2\times n_{kv}\times h_d$ 个元素；乘以层数 L 与 dtype 字节数即上式。

**推导 2：降档判据**。以权重 W≈1.98GB（fp32）、内存上限 3.16GB RSS 为例。源码注释给出关键额外项：4096 ctx 的 lm_head 中间激活约 2.3GB（[._local_whitebox_detect.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_local_whitebox_detect.py) 第 96 行）。峰值估算：

$$peak(4096) \approx W + 2\cdot KV(4096) + act(4096) \approx 1.98\text{GB} + 2\times100.7\text{MB} + 2.3\text{GB} \approx 4.48\text{GB} > 3.16\text{GB}$$

$$peak(1024) \approx 1.98\text{GB} + 2\times25.2\text{MB} + 0.575\text{GB} \approx 2.61\text{GB} < 3.16\text{GB}$$

故请求 4096 时触发降档至 1024（KV×2 为临时张量余量）。这正是实测日志中 `--ctx 1024`、`RSS=3.16GB` 的机理。

**推导 3：理论 vs 实测 KV 的差异**。理论公式在 ctx=1024 时给出 $KV_{th}=24{,}576\times1024=25{,}165{,}824\text{ B}=25.165824\text{MB}$；实测 24.9MB 的差异来源是**实际前向序列长度 seq=1014**（重复填充后不足 1024 token 的输入按 base prompt 长度截断），$24{,}576\times1014=24{,}920{,}064\text{ B}\approx24.9\text{MB}$。相对偏差 ≈0.98%，属序列取整效应而非公式误差。

## ⑤ 数值验证

**实测状态**：F01 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 70 行）。

**实测值（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)，Qwen2.5-0.5B-Instruct，CPU）**：

| 量 | 值 | 来源 |
|---|---|---|
| 架构 | 24L 2KV 64hd，W=1.98GB (fp32) | wb_full.log:14 |
| kv_per_tok | 24.6KB（公式 24,576B） | wb_full.log:14 |
| 运行 ctx | 1024（--ctx 1024） | wb_full.log:4 |
| 实际 seq | 1014 | wb_full.log:20 |
| 实测 KV | 24.9MB | wb_full.log:20 |
| KV/W | 1.3% | wb_full.log:20 |
| RSS | 3.16GB | wb_full.log:23 |

**合成数值验证（[verify.py](verify.py)）**：
- 公式精确复算：2×24×2×64×1024×4 = 25,165,824 B = 25.165824MB ✅
- 与实测 24.9MB 比对：相对偏差 1.06%；用实测 seq=1014 复算 24.920064MB ≈ 24.9MB，偏差 0.98%，差异来源为序列取整 ✅
- 降档逻辑：请求 4096 → 峰值估算超 3.16GB → 返回 1024；请求 1024 → 保留 ✅
- 主文档 168KB/tok 口径（n_kv=8/h_d=112）：kv_per_tok=172,032B，ctx=4096 理论 704.6MB（十进制）✅（标注与实测口径差异）

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| F03 kv_ctx | 同一物理量的两个口径：F01 定义**测多少**（ctx），F03 定义**怎么算**（kv_per_tok×ctx）。实测 24.9MB@1024 即 F01 的 ctx 与 F03 的 kv_per_tok 共同作用 |
| F04 ctx_scan | F04 是显存公式档位扫描 [1024..65536]，F01 是实际前向测试的单一 ctx（默认 4096 降档 1024） |
| G05 max_ctx | 黑盒 API 计时上限 4096；F01 是白盒 KV 前向测试，约束来源不同（内存 vs API 成本） |
| 502 KV_bytes | 输出指标：实测 24.9MB@ctx1024，公式即 F01 的 KV 公式 |
| 503 KV/W | 输出指标：实测 1.3%（24.9MB/1.98GB）；该值越小 KV 越可压缩 |
| 504 RSS/VRAM | 降档的触发对象：3.16GB RSS 限制下 4096 峰值超限 → 自动降档到 1024 |
| H05 dtype | 公式中字节因子：float32=4、float16=2；fp16 可将 KV 减半 |
