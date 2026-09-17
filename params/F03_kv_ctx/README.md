# F03 kv_ctx：KV 预测上下文字节

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 4361-4481 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(F03, 第 318-324 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 72 行, 状态=已用 24.6KB/tok)
> 源码：[._local_whitebox_detect.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/_local_whitebox_detect.py)（第 73 行：kv_per_tok = 2·n_layers·n_kv·hd·4）
> 实测日志：[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)（KV/每token=24.6KB；KV=24.9MB@1024；KV/W=1.3%）
> 验证脚本：[verify.py](verify.py)

## ① 定义

`F03 kv_ctx` 定义 KV 缓存总字节数的预测公式：用架构常数 `kv_per_tok`（每个 token 消耗的 KV 字节）乘以任意目标上下文长度 `ctx`，得到该上下文下的 KV 缓存大小。它是显存规划的核心工具——长上下文部署前必须先回答"KV 会吃掉多少显存"。

$$KV_{bytes} = kv_{per\_tok} \times ctx$$

审计口径（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 72 行）：kv_per_tok = **24.6KB/tok**（已用，实测）。

## ② 公式

kv_per_tok 由架构决定（同 F01）：

$$kv_{per\_tok} = 2 \times L \times n_{kv} \times h_d \times \text{dtype\_bytes}$$

Qwen2.5-0.5B 实测配置（L=24, n_kv=2, h_d=64, float32）：

$$kv_{per\_tok} = 2\times24\times2\times64\times4 = 24{,}576\text{ B} \approx 24.6\text{KB/tok}$$

```python
def predict_kv(ctx, kv_per_tok):
    return kv_per_tok * ctx          # KV_bytes = kv_per_tok × ctx
```

**口径标注**：主文档第 4374-4383 行以 n_kv=8/h_d=112 推得 172,032B（≈168KB/tok），并在第 4385 行注明"24.6KB/tok 对应的是某种轻量配置（可能是 float16+较小头维），而 168KB/tok 是 Qwen 0.5B float32 的实际值"。但实测日志（_local_whitebox_detect.py 直接读 config）确认真实配置为 2KV/64hd → 24.6KB/tok。主文档的 168KB/tok 应为口径误差；F03 的已用实测值以 24.6KB/tok 为准。

## ③ 物理/几何直觉与用途

**物理直觉**：KV 缓存是解码时"必须随身携带的全部历史"——每来一个新 token，都要在**所有层**向**所有历史 token** 的 key 上做 attention，因此历史不能丢。它是唯一随对话线性增长的内存项：权重只付一次费，KV 按 token 持续付费。`kv_per_tok` 就是"每 token 的存储租金"。

几何视角：KV 中存的就是激活向量经线性投影后的 key/value 流形坐标。kv_per_tok 越大，意味着每采样一个流形点要付出的存储越多——这正是 F04 计算"KV 何时反超权重"、以及 106 压缩比（head_dim/k90）判断"KV 流形可压缩性"的物理前提。

**用途**：① 推理部署前的显存预算；② 判断长上下文可行性（如 128K 上下文需要多少 KV）；③ 对比 KV 压缩收益。

## ④ 推导过程

**推导 1：预测公式的线性性**。设 kv_per_tok 为常数 $c$，则 $KV(ctx)=c\cdot ctx$，直接给出线性标度律：$KV(2\,ctx)=2\,KV(ctx)$。ctx 从 1K 到 128K，KV 线性增长 128 倍。

**推导 2：预测与实测的差异**。预测：$24{,}576\times1024=25{,}165{,}824\text{ B}=25.17\text{MB}$；实测 24.9MB。差异来自实测前向序列 seq=1014（<1024，输入重复填充截断）：$24{,}576\times1014=24{,}920{,}064\text{ B}\approx24.9\text{MB}$，相对偏差 0.98%。预测公式本身零误差，偏差完全来自序列取整。

**推导 3：KV 相对权重占比**。$KV/W = (c\cdot ctx)/W_{bytes}$。Qwen0.5B 实测（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)）：$24.9\text{MB}/1.98\text{GB}=1.3\%$。主文档用 168KB/tok 口径在 4K 得 688MB/1.98GB≈35%——占比随 kv_per_tok 与 ctx 两者线性放大，不同口径差异巨大，说明"KV/W"必须绑定具体模型配置与上下文才有意义。

## ⑤ 数值验证

**实测状态**：F03 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 72 行：kv_per_tok=24.6KB/tok，实测 24.9MB@1024）。

**实测值（[wb_full.log](file:///c:/Users/小冰/Desktop/宇宙/AIQ/wb_full.log)，Qwen2.5-0.5B-Instruct）**：

| 量 | 值 |
|---|---|
| kv_per_tok（公式） | 24,576 B = 24.6KB/tok（日志） |
| 预测 KV(1024) | 25.17MB |
| 实测 KV(seq=1014) | 24.9MB（偏差 0.98%） |
| KV/W | 1.3% |

主文档 F03 示例表（168KB/tok 口径，供对照）：

| 上下文长度 | KV（168KB/tok） | 适用场景 |
|---|---|---|
| 1K | 168MB | 短对话 |
| 4K | 688MB | 标准 RAG |
| 16K | 2.75GB | 长文档 |
| 32K | 5.5GB | 长视频/书籍 |
| 128K | 22GB | 超长上下文 |

**合成数值验证（[verify.py](verify.py)）**：
- kv_per_tok = 2×24×2×64×4 = 24,576 B = 24.6KB/tok（与日志一致）✅
- kv_ctx = kv_per_tok × ctx：ctx=1024 → 25.165824MB；实测 24.9MB（seq=1014 复算 24.920064MB，偏差 0.98%）✅
- 线性性：KV(2×ctx) = 2×KV(ctx) ✅
- 168KB/tok 口径对照表（1K/4K/16K/32K/128K），标注十进制/二进制单位差异 ✅
- KV/W：实测口径 1.3%（vs 主文档 35% 为 168KB/tok@4K 口径）✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| F01 ctx | 同一公式的两面：F03 用 kv_per_tok×ctx 做**预测**，F01 用该公式决定**测多大的 ctx**（默认 4096 降档 1024）；实测 KV=24.9MB@1024 是两者共同产物 |
| F04 ctx_scan | F03 的预测公式是 F04 扫描四档 [1024,4096,16384,65536] 的实现内核 |
| 502 KV_bytes | 输出指标：实测 24.9MB@1024，公式即 F03 的 kv_per_tok×ctx |
| 503 KV/W | 输出指标：实测 1.3% = KV/权重，由 F03 的 KV 预测除以权重得到 |
| 106 ratio | KV 压缩比 head_dim/k90：kv_per_tok 越小（压缩空间越大），长上下文越经济 |
| H05 dtype | dtype_bytes 因子：fp16 使 kv_per_tok 减半（84KB/tok 口径），KV 预测随之减半 |
