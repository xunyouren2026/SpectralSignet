# H01 seed：随机种子

> 数据源：《几何指纹：AI模型的家族级溯源与抗微调鲁棒性参数附录表完整版.md》第 5427-5519 行
> 配套：《AI几何指纹插件_参数完整定义与公式.txt》(H01, 第 392-398 行)、《AI几何指纹插件_参数审计与实验报告.txt》(第 81 行, 状态=已用)
> 验证脚本：[verify.py](verify.py)

## ① 定义

`H01 seed` 定义全局随机种子，固定为 **0**。随机数生成器（PRNG）以种子为初值产生确定性的随机序列；固定种子保证每次运行时随机序列完全一致，从而确保实验结果可复现。

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

0 本身无特殊含义，只是一个惯例默认值——只要固定，任何数字都等价。

## ② 公式

PRNG 序列生成（以 PCG64/MT19937 为例）：

$$x_{n+1} = f(x_n),\qquad x_0 = \text{seed}(0)$$

固定种子即固定 $x_0$，从而固定整条随机序列 $\{x_n\}$。指纹计算 $F$（含随机采样、蒙特卡洛、dropout 等）成为种子的函数：

$$F(\text{seed}=0) \equiv \text{const},\qquad \forall \text{ run}$$

## ③ 物理/直觉与用途

**物理直觉**：实验可复现性是科学方法论的基石。指纹提取涉及多处随机性（token 采样、MC 理论分布、dropout、数据切分），不固定种子时每次运行得到不同指纹，无法区分"模型差异"与"随机波动"。`seed=0` 把随机性"冻结"，使指纹差异只可能来自模型本身。

**用途**（控制所有随机过程的可复现性）：

| 随机过程 | 影响 | 后果（若不固定） |
|---|---|---|
| 模型权重初始化 | 每次训练起点不同 | 训练结果不可复现 |
| Dropout | 每次丢弃不同神经元 | 推理结果波动 |
| 数据采样 | 每次采样不同 token/样本 | 指纹每次不同 |
| 蒙特卡洛采样（D08 mc_n） | 每次生成不同理论分布 | χ²/p 值每次不同 |

`H01 seed=0` 保证"同一份代码，在不同时间、不同机器上，跑出完全一样的结果"。

## ④ 推导过程

**推导 1：种子确定性的来源**。PRNG 是确定性状态机 $x_{n+1}=f(x_n)$。若两次运行都以 $x_0=s$ 出发，则序列逐项相等（对纯确定性算法）。numpy/torch 分别维护独立 PRNG 状态，因此需要 `np.random.seed(0)` 与 `torch.manual_seed(0)` 分别固定（`random.seed(0)` 固定 Python 内置随机）。

**推导 2：并行/GPU 下的复现限制**。GPU 上某些算子（如 cudnn benchmark、原子加法的约简顺序）非确定性，因此还需要：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

对于 CPU 上的谱分析（SVD、协方差），LAPACK 是确定性的，`seed=0` 足以完全复现。

**推导 3：指纹函数视角**。指纹 $G(\theta)$ 可写为：

$$G(\theta) = \mathcal{F}(\text{model}, \text{prompt}, \underbrace{\text{seed}}_{\text{H01}}, \text{dtype}, \text{threads}, \dots)$$

固定 H01 使随机维度退化，指纹成为模型结构的稳定函数——这正是"家族级溯源"所需的可重复性前提。

## ⑤ 数值验证

**实测状态**：H01 = 已用（[审计报告](file:///c:/Users/小冰/Desktop/宇宙/AIQ/AI几何指纹插件_参数审计与实验报告.txt)第 81 行：所有脚本均固定 seed=0）。

**实测值（主文档第 5507-5516 行）**：
- 两次独立运行（重新加载模型）指纹差异 = 0.000（完全一致）
- 判定：可复现 ✅

**合成数值验证（[verify.py](verify.py)）**：
- `set_seed(0)` 后两次生成随机矩阵/SVD 指纹 → 逐元素全等（max|Δ|=0）✅
- 不同种子（0 vs 1）→ 结果不同（证明种子确实控制随机性）✅
- 模拟"指纹管道"（随机采样→SVD→Gamma→MC 直方图）→ 两次运行完全一致 ✅
- torch 路径（如可用）：`torch.manual_seed(0)` 后生成张量全等 ✅

## ⑥ 与关联参数的关系

| 参数 | 关系 |
|---|---|
| D08 mc_n | 蒙特卡洛采样随机性由 H01 控制，保证 χ²/p 值可复现 |
| A02 prompt / A03 gen_len | 生成采样随机性受 H01 影响（temperature 采样） |
| H05 dtype | 固定 dtype 与固定 seed 共同保证数值级可复现（float32 下 SVD 确定性） |
| H02/H03 | 线程数不影响数值结果（确定性 LAPACK），只影响内存/速度；与 H01 正交 |
| G04 target_metric | 复现性使实测 Gamma 可与论文值 0.9966 可靠对比 |
