# H05 dtype：模型精度

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5934-6037 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(H05, 第 424-430 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 85 行, 状态=已用)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`H05 dtype` 定义模型加载与推理时使用的数值精度，固定为 **`torch.float32`**。精度决定内存占用、计算速度与数值稳定性：

```python
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
```

**为什么是 float32 而非 float16/int8**：
- float16 更省内存（减半），但 CPU 上部分算子不兼容、国产 DCU 上 SVD 可能不稳定
- float32 数值稳定性更好，避免梯度/激活溢出
- Qwen 0.5B 在 float32 下权重约 2.0GB，仍可接受
- 谱分析（SVD、协方差）在 float32 下数值稳健

## ② 公式

内存占用 = 参数量 × 每参数字节数：

$$\text{Mem} = N_{\text{params}} \times \text{bytes}/\text{param}$$

| dtype | 每参数字节数 | Qwen 0.5B（约 0.494B 参数） |
|---|---|---|
| float32 | 4 | ≈ 2.0 GB |
| float16 | 2 | ≈ 1.0 GB |
| int8 | 1 | ≈ 0.5 GB |

float32 的机器精度（单位舍入）：

$$\epsilon_{32} = 2^{-24} \approx 5.96\times 10^{-8}$$

float16 上限 $65504$，超出即溢出为 inf；float32 上限约 $3.4\times 10^{38}$，正常激活（量级 $O(1\sim10^3)$）远不会溢出。

## ③ 物理/直觉与用途

**物理直觉**：数值精度是"内存-稳定性"天平上的滑块。float32 有 23 位尾数（约 7 位十进制有效数字），对协方差计算、SVD 分解这类**累加放大误差**的谱分析而言是安全底线：float16 只有 10 位尾数，长序列求和时舍入误差累积可能导致 Gamma 计算漂移甚至 DCU 上 SVD 不收敛。float32 用翻倍的内存换取"谱分析结果可信"。

**用途**（控制模型的内存占用和计算精度）：

| dtype | 权重内存 | 推理时 RSS | 建议场景 |
|---|---|---|---|
| float32 | ~2.0GB | ~3.16GB | CPU 安全，DCU 稳健（H05 固定值） |
| float16 | ~1.0GB | ~2.0GB | GPU 部署，NVIDIA 卡 |
| int8 | ~0.5GB | ~1.5GB | 边缘设备部署 |

## ④ 推导过程

**推导 1：float32 相对误差界**。对元素量级 $O(1)$ 的矩阵 $H$，协方差 $C=H^\top H/(B-1)$ 每步乘加引入 $\epsilon_{32}\approx 6\times10^{-8}$ 相对误差；$B=256$ 行的求和经约 $\log_2 256=8$ 次合并，累计相对误差约 $O(\epsilon_{32}\cdot \sqrt{B})$，即 $10^{-6}$ 量级。而 float16 的 $\epsilon_{16}=2^{-11}\approx 4.9\times10^{-4}$，累计误差到 $10^{-2}\sim10^{-3}$，足以让 Gamma 的第三位小数失真。

**推导 2：float16 溢出窗口**。激活在层归一化前可到 $O(10^2\sim10^3)$，平方后 $O(10^4\sim10^6)$；float16 上限 $65504$ 使得协方差元素（平方求和）极易上溢为 inf → SVD 输入含 inf → 结果 NaN。float32 上限 $3.4\times10^{38}$ 提供约 10 个数量级的安全余量。

**推导 3：DCU 兼容性**。国产 DCU（海光）的 `torch.linalg.svdvals` 在 float16 下偶发 cusolver 收敛失败（源码 [spectral_analysis.py](file:///c:/Users/小冰/Desktop/宇宙/AIQ/spectral-cognition/modules/spectral_analysis.py) 第 100-102 行注释），float32 + CPU LAPACK 是最稳健路径。

## ⑤ 数值验证

**实测状态**：H05 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 85 行：`_local_whitebox_detect.py` 固定 float32）。

**实测值（主文档第 6020-6037 行，Qwen2.5-0.5B-Instruct）**：
- 权重 ~2.0GB，RSS 峰值 3.16GB（在限制内可用 ✅）
- SVD 稳定 ✅，协方差计算稳定 ✅
- 结论：float32 是 CPU/DCU 环境下最稳健的选择

**合成数值验证（[verify.py](verify.py)）**：
- 同一矩阵在 float32 与 float64 下计算 SVD/Gamma/均值，相对误差 ≈ 1e-6（ε₃₂ 量级）✅
- float16 溢出演示：`1e5²` 类协方差元素在 float16 上溢为 inf，float32 保持有限 ✅
- 内存计算：0.494B 参数 × 4B ≈ 1.98GB（与文档 ~2.0GB 一致）；float16 减半 ✅
- float32 下 Gamma 计算稳定（有限、合理范围）✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| H02/H03 | 线程数限制用于防 OOM，与 dtype 共同控制内存占用 |
| F01 ctx / F03 kv_ctx | 上下文与 KV 缓存内存随 dtype 缩放（float32×4, float16×2） |
| B01/B02 | 谱分析（SVD/Gegenbauer）的数值稳定性与 dtype 直接相关 |
| H01 seed | 固定 dtype + 固定 seed 共同保证数值级可复现 |
| G04 target_metric | 实测 Gamma 与论文值 0.9966 的可比性依赖稳定 dtype |
