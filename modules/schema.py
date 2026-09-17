"""
aiq-geometric-forensics.modules.schema — 单一事实源（阈值 / 权重 / 证据级 / 展示口径）
=====================================================================
把散落在 health.py / forensics.py / cli.py 的判据阈值、权重、证据分级
集中到本模块，消除"同一数值散落多处、可能漂移"的生产风险（大厂读码要求）。

每个条目标注参数来源级别（Evidence level）：
  measured   由真实实测标定 / 可由实测直接得出（实证）
  theory     数学常数或理论锚点（如 π/2）
  heuristic  经验切点，尚无实验标定（推测）—— 严禁当实测引用
"""
from __future__ import annotations

# ---------------------------------------------------------------- 数学/理论锚点
PI_HALF: float = 1.5707963267948966            # 理论 · DEFF 平台锚点 π/2

# ---------------------------------------------------------------- AIQ 权重（实测·设计权重）
# 权重视为已标定设计参数（和=1），判定默认诠释的因子权重。
WEIGHTS = (0.20, 0.25, 0.20, 0.20, 0.15)

# ---------------------------------------------------------------- 健康判据（heuristic 为主）
K_SLOT = (20.0, 57.0)                          # heuristic · K<0% 健康槽（鞍形主导区间）
DEFF_TOL = 0.15                                # heuristic · 平台锁定容差（abs 偏差）
H_CONV_THR = 0.05                              # heuristic · 「H 收敛」判定阈值（f3 下界）

# ---------------------------------------------------------------- 可压缩性分级（heuristic）
COMPRESS_HIGH = 0.45                           # γ≥0.45 高可压（attention 侧 o/k）
COMPRESS_MID = 0.35                            # γ∈[0.35,0.45) 中可压
COMPRESS_LOW = 0.35                            # γ<0.35 难压缩（前馈扩展 up/v）

# ---------------------------------------------------------------- 压缩建议档位（heuristic 配套）
# 与 COMPRESS_* 阈值配套的剪枝建议（keep_ratio 保留比例区间），集中管理避免散落。
COMPRESS_HIGH_RATIO = (0.25, 0.4)              # 高可压 → 激进剪枝 keep_ratio 0.25-0.4
COMPRESS_MID_RATIO = (0.5, 0.6)                # 中可压 → 温和压缩 keep_ratio 0.5-0.6
COMPRESS_LOW_RATIO = (1.0, 1.0)                # 难压缩 → 保持全精度 keep_ratio 1.0

# ---------------------------------------------------------------- 家族溯源阈值（measured 标定）
FTC_EPS = 0.01                                 # 指纹均值差 < 0.01 → 同族（SFT 不改身份）
MARGIN_EPS = 0.29                              # 明显异族门控（远亲低可靠度）

# ---------------------------------------------------------------- 指纹剖面（工程定）
PROFILE_K = 64                                 # 层谱对相对深度重采样的恒定维数（跨层数可比）

# ---------------------------------------------------------------- 溯源鲁棒性（stability，heuristic）
# 存活判据 = 扰动后归属认回自身（实测：scale 50%/trunc 90%/sft σ=0.1 仍 100% 认回，
# 证明剖面形状不变性；位移仅监控）。档位覆盖"由轻到毒"，用于呈现强度-归属灰度。
STAB_MAD_THR = 0.020                           # 位移监控带（仅展示扰动强度，不否决存活）
PERT_SFT_EPS = (0.001, 0.005, 0.010, 0.050, 0.100)   # 微调微扰幅度档（含强微调）
PERT_TRUNC_K = (0.10, 0.25, 0.50, 0.80, 0.90)        # 层谱截断丢弃比例档（含近乎全抹）
PERT_SCALE_DELTA = (0.05, 0.10, 0.30, 0.50)          # 幅度改写比例档（含强缩放）
PERT_N_TRIALS = 30                             # 随机扰动试验次数（种子可复现）

# ---------------------------------------------------------------- 家族族谱（family，heuristic）
# 族谱距离口径 = 指纹**剖面段 MAD**（64 维重采样 Gamma 剖面）。
# 标量段（K%≈45 量级）若用欧氏会主导并淹没剖面差异（实测 5 档全部 2.0~4.9 全"远亲"），
# 故族谱只取剖面段。阈值按实测剖面 MAD 分层标定：
#   Qwen base↔Instruct=0.0031（SFT 级） < 0.5B↔1.5B=0.087 / ↔TinyLlama=0.073（近亲）
#   ↔BLOOM=0.143 / TinyLlama↔BLOOM=0.128（远亲）
FAM_ATTR_DIST = 0.010                          # 剖面 MAD < 0.01 → 同族（SFT 级亲缘，实测 0.0031）
FAM_AFFINITY_DIST = 0.100                      # 剖面 MAD < 0.10 → 近亲（同家族扩参/邻近架构）

# ---------------------------------------------------------------- 展示口径（纯展示非科学量）
AIQ_YMAX = 60.0                                # 综合报告 AIQ 条形图显示上限（仅图表纵轴）

# ---------------------------------------------------------------- 证据分级标注
# 与 report.py 的徽标体系对应；heuristic 项必须在报告/文档显式标注"推测"。
EVIDENCE = {
    "PI_HALF": "theory",
    "DEFF_TOL": "heuristic",
    "K_SLOT": "heuristic",
    "H_CONV_THR": "heuristic",
    "COMPRESS_HIGH": "heuristic",
    "COMPRESS_MID": "heuristic",
    "COMPRESS_LOW": "heuristic",
    "COMPRESS_HIGH_RATIO": "heuristic",
    "COMPRESS_MID_RATIO": "heuristic",
    "COMPRESS_LOW_RATIO": "heuristic",
    "FTC_EPS": "measured",
    "MARGIN_EPS": "heuristic",
    "STAB_MAD_THR": "heuristic",
    "PERT_SFT_EPS": "heuristic",
    "PERT_TRUNC_K": "heuristic",
    "PERT_SCALE_DELTA": "heuristic",
    "FAM_ATTR_DIST": "heuristic",
    "FAM_AFFINITY_DIST": "heuristic",
    "WEIGHTS": "design",
}
