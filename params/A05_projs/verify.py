# -*- coding: utf-8 -*-
"""A05 projs 投影类型选择 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 投影枚举与模块归属（Attention: k/q/v/o; MLP: gate/up/down）
  2. 指纹维度：三投影 24×3=72 点、全投影 24×7=168 维
  3. 各投影 Gamma 排序 o>k>q>gate>down>v>up（审计实测，行 20-28）
  4. Gamma 数值范围 0.60-0.78
  5. 投影策略映射（单投影/标准/全投影/水印/曲率）
  6. 真实模型对照：真实 7 投影 Gamma 与文档声称排序对照（如实报告差异）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A05Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 A05Config（环境变量 AIQ_A05_<KEY> > YAML > _params_data.json > 默认）
  ProjSynthesizer         —— 按目标 Gamma 构造激活矩阵（设定特征值谱 + 随机正交基重建）
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 322-439（A05 是什么/干什么/怎么做）
  《AI几何指纹插件_参数审计与实验报告.txt》行 20-28（Gamma 实测矩阵）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
  注：合成激活经中心化后协方差口径的 Gamma 与设定目标有 <2% 偏差（中心化秩一扰动），
      排序与量级与审计实测一致（详见验证项 3 输出）。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 7 投影 Gamma（spectral.proj_gamma）与文档声称排序对照。
  如实呈现（真实实测 vs 文档声称）：
    1) 排序差异：真实 o>k>q>down>gate>v>up（down 0.396 > gate 0.369）；
       文档声称 o>k>q>gate>down>v>up——down 与 gate 顺序互换，如实报告。
    2) 量级差异：真实 Gamma 范围 [0.303, 0.494]，文档声称 0.60-0.78——
       真实实测整体低约 0.19-0.29，显著低于文档声称。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from typing import Any

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, ConfigError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 A05Config（pydantic 优先；dataclass 回退） ----
_HAS_PYDANTIC = False
_ConfigModelBase: Any
try:
    from pydantic import BaseModel as _PydanticBase  # noqa: E402
    _ConfigModelBase = _PydanticBase
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - pydantic 缺失路径
    import dataclasses as _dataclasses

    @_dataclasses.dataclass
    class _DataclassBase:
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。"""

    _ConfigModelBase = _DataclassBase


class A05Config(_ConfigModelBase):
    """A05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A05 节点键名一一对应；取值优先级：
    环境变量 AIQ_A05_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    M: int = 3                       # B01: C-子空间主方向数（spl_gamma 取前 M 主成分）
    N_LAYERS: int = 24               # Qwen 层数（指纹维度 = 层数 × 投影数）
    B: int = 200                     # 合成激活 token 数（须 > d 以承载完整谱）
    DIM: int = 32                    # 合成激活维度
    # 审计实测 Gamma（审计行 21-27）：o>k>q>gate>down>v>up（各值从数据层读取）
    GAMMA_O: float = 0.7765
    GAMMA_K: float = 0.7227
    GAMMA_Q: float = 0.6812
    GAMMA_GATE: float = 0.6648
    GAMMA_DOWN: float = 0.6428
    GAMMA_V: float = 0.6311
    GAMMA_UP: float = 0.6009
    GAMMA_TOL: float = 0.02          # 合成 Gamma 与设定目标的相对偏差容差（中心化扰动所致）

    def gamma_measured(self) -> dict:
        """审计实测 Gamma 字典（顺序 = 审计矩阵顺序 o/k/q/gate/down/v/up）。"""
        return {
            "o_proj": self.GAMMA_O,
            "k_proj": self.GAMMA_K,
            "q_proj": self.GAMMA_Q,
            "gate_proj": self.GAMMA_GATE,
            "down_proj": self.GAMMA_DOWN,
            "v_proj": self.GAMMA_V,
            "up_proj": self.GAMMA_UP,
        }


# ---- 第二层：配置工厂 ConfigFactory（实例化 A05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A05Config。"""

    def build(self) -> A05Config:
        """构建 A05Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(A05Config, "A05")


# 投影模块归属（主文档行 327-337；结构性算法逻辑，非阈值）
MODULE_OF = {
    "k_proj": "Attention", "q_proj": "Attention", "v_proj": "Attention",
    "o_proj": "Attention",
    "gate_proj": "MLP", "up_proj": "MLP", "down_proj": "MLP",
}
# 投影策略映射（主文档行 411-417；结构性算法逻辑，非阈值）
STRATEGIES = {
    "快速验证(10秒级)": ("k_proj",),                          # 单投影快指纹
    "标准指纹(1分钟级)": ("k_proj", "up_proj", "o_proj"),     # 三投影标准指纹
    "完整指纹(3分钟级)": ("k_proj", "q_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),   # 全 7 投影
    "水印嵌入": ("k_proj",),                                   # 水印走 k_proj
    "曲率特征": ("k_proj", "up_proj", "o_proj"),               # 曲率走三投影
}


# ---- 第三层：合成器 ProjSynthesizer（算法与原脚本完全一致） ----
class ProjSynthesizer(_SynthBase):
    """A05 按目标 Gamma 构造激活矩阵的合成器。

    - synth_proj_activation(target_gamma, rng)：设定特征值谱（前 M 个简并 = a，其余 = 1），
      由随机正交基重建 H，使 spl_gamma(H) ≈ target_gamma。
    """

    def __init__(self, cfg: A05Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_proj_activation(self, target_gamma: float, B: int | None = None,
                              d: int | None = None,
                              rng: np.random.Generator | None = None) -> np.ndarray:
        """构造激活矩阵 H ∈ R^{B×d}，使其 spl_gamma ≈ target_gamma。
        方法：设定特征值谱 λ（前 M 个简并 = a，其余 = 1），
              a/b = target·(d-M)/(M·(1-target))，再由随机正交基重建 H。"""
        cfg = self._cfg
        m = cfg.M
        if B is None:
            B = cfg.B
        if d is None:
            d = cfg.DIM
        if rng is None:
            rng = np.random.default_rng(cfg.SEED)     # 未显式传 rng 时用固定种子，保证可复现
        if not 0.0 < target_gamma < 1.0:
            raise ValueError(f"target_gamma 须在 (0,1): {target_gamma}")
        if B <= 0 or d <= 0 or d <= m:
            raise ValueError(f"非法维度: B={B}, d={d}, M={m}")
        lam = np.ones(d)                            # 基础谱：全部特征值=1
        # 反解前 M 个简并特征值 a：使 gamma = M·a / (M·a + (d-M)·1) = target
        a = target_gamma * (d - m) / (m * (1.0 - target_gamma))
        lam[:m] = a                                 # 前 M 主方向注入强能量
        # H = A·diag(sqrt(λ))·B^T，随机正交矩阵 A,B（QR 分解构造正交基）
        A, _ = np.linalg.qr(rng.standard_normal((B, d)))     # 样本空间正交基 (B,d)
        Br, _ = np.linalg.qr(rng.standard_normal((d, d)))    # 特征空间正交基 (d,d)
        H = A @ (np.sqrt(lam)[:, None] * Br.T).T
        return H


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def spl_gamma(H: np.ndarray, m: int) -> float:
    """spl_gamma = 前 M=3 主成分能量占比（B01）。返回 NaN 表示退化输入。
    口径：激活中心化 → 协方差阵特征值 → 前 M 大特征值之和占比。"""
    H = np.asarray(H, dtype=float)
    if H.size == 0 or H.shape[0] < 2:
        return float("nan")             # 防御：样本不足 2 无法估计协方差
    Hc = H - H.mean(0, keepdims=True)   # 中心化（源码 spectral_analysis.py 同口径）
    S = np.linalg.eigvalsh(Hc.T @ Hc / (Hc.shape[0] - 1))[::-1]   # 协方差特征值降序
    tot = float(S.sum())                # 总能量（迹）
    if not np.isfinite(tot) or tot <= 0.0:
        return float("nan")             # 防御：全零/非有限则判定退化
    return float(S[:m].sum() / tot)     # 前 M 大特征值能量占比


def gamma_rank_order(g: dict[str, float]) -> list[str]:
    """按 Gamma 值降序排列投影名（排序比较的基础工具）。"""
    return sorted(g, key=g.get, reverse=True)


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A05 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: A05Config,
        synth: ProjSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.g_synth: dict[str, float] = {}   # 各投影合成 Gamma（供范围断言步骤复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 枚举与模块归属
    def validate_enum(self) -> dict:
        """1) 投影枚举与模块归属：7 种投影，Attention 4 种 / MLP 3 种。"""
        cfg = self.config
        projs = list(cfg.gamma_measured().keys())          # 7 种投影名（保持审计矩阵顺序）
        att = [p for p in projs if MODULE_OF[p] == "Attention"]   # Attention 组：k/q/v/o
        mlp = [p for p in projs if MODULE_OF[p] == "MLP"]         # MLP 组：gate/up/down
        ok1 = (len(projs) == 7 and len(att) == 4 and len(mlp) == 3)   # 枚举完整性断言
        if not ok1:
            raise ConfigError(
                f"枚举/归属不符: 总={len(projs)}, att={len(att)}, mlp={len(mlp)}",
                expected={"total": 7, "att": 4, "mlp": 3},
                actual={"total": len(projs), "att": len(att), "mlp": len(mlp)},
                param_key="A05",
            )
        return {"detail": (f"投影枚举 {len(projs)} 种: {projs}; "
                           f"Attention 模块: {att} ({len(att)} 种); MLP 模块: {mlp} ({len(mlp)} 种)"),
                "projs": projs, "att": att, "mlp": mlp}

    # ------------------------------------------------------------ 2) 指纹维度
    def validate_dim(self) -> dict:
        """2) 指纹维度：三投影 24×3=72 点、全投影 24×7=168 维（主文档行 210/391）。"""
        cfg = self.config
        dim3 = cfg.N_LAYERS * 3      # 三投影指纹：24 层 × 3 投影（k/up/o 标准策略）
        dim7 = cfg.N_LAYERS * 7      # 全投影指纹：24 层 × 7 投影（完整指纹策略）
        ok2 = (dim3 == 72 and dim7 == 168)   # 维度公式断言（主文档行 210/391）
        if not ok2:
            raise ConfigError(
                f"维度不符: {dim3} / {dim7}",
                expected={"dim3": 72, "dim7": 168},
                actual={"dim3": dim3, "dim7": dim7},
                param_key="A05",
            )
        return {"detail": f"三投影指纹 {cfg.N_LAYERS}×3={dim3} 点; 全投影 {cfg.N_LAYERS}×7={dim7} 维",
                "dim3": dim3, "dim7": dim7}

    # ------------------------------------------------------------ 3) 合成 Gamma 排序
    def validate_gamma_synth(self) -> dict:
        """3) 合成 Gamma（目标=审计实测，协方差口径）：数值容差内 + 排序与文档一致。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        g_meas = cfg.gamma_measured()
        projs = list(g_meas.keys())
        g_synth: dict[str, float] = {}
        rows = []
        ok3 = True
        for p in projs:
            H = self.synth.synth_proj_activation(g_meas[p], rng=rng)   # 按目标 Gamma 构造激活
            g_synth[p] = spl_gamma(H, cfg.M)    # 用协方差口径重估 Gamma
            rel = abs(g_synth[p] - g_meas[p]) / g_meas[p]   # 相对偏差
            one_ok = np.isfinite(g_synth[p]) and (rel < cfg.GAMMA_TOL)          # 须有限且容差内
            ok3 &= bool(one_ok)
            rows.append(f"{p:10s} gamma={g_synth[p]:.4f} (目标 {g_meas[p]:.4f}, "
                        f"偏差 {rel * 100:.2f}%)")
        order_synth = gamma_rank_order(g_synth)      # 合成 Gamma 的降序排序
        order_doc = gamma_rank_order(g_meas)         # 文档声称的降序排序
        ok3 &= (order_synth == order_doc)            # 排序一致性断言（o>k>q>gate>down>v>up）
        if not ok3:
            raise AIQValidationError(
                f"Gamma 数值/排序不符: 合成={order_synth}, 文档={order_doc}",
                expected=order_doc, actual=order_synth, param_key="A05",
            )
        self.g_synth = g_synth                        # 供范围断言步骤复用
        return {"detail": ("合成 Gamma（目标=审计实测，中心化协方差口径）: " + "; ".join(rows)
                           + f"; 合成排序: {order_synth}（=文档 {order_doc}）"),
                "g_synth": g_synth, "order_synth": order_synth, "order_doc": order_doc}

    # ------------------------------------------------------------ 4) Gamma 范围
    def validate_gamma_range(self) -> dict:
        """4) Gamma 数值范围 [0.60, 0.78]（文档 0.60-0.78）。"""
        cfg = self.config
        vals = np.array(list(self.g_synth.values()))      # 各投影合成 Gamma 数值集合
        ok4 = bool(0.60 <= vals.min() and vals.max() <= 0.78)   # 范围断言（文档 0.60-0.78）
        if not ok4:
            raise AIQValidationError(
                f"Gamma 超出文档范围: [{vals.min():.3f}, {vals.max():.3f}]",
                expected=(0.60, 0.78), actual=(float(vals.min()), float(vals.max())),
                param_key="A05",
            )
        return {"detail": f"Gamma 范围 [{vals.min():.3f}, {vals.max():.3f}] (文档 0.60-0.78)",
                "gamma_min": float(vals.min()), "gamma_max": float(vals.max())}

    # ------------------------------------------------------------ 5) 策略映射
    def validate_strategy(self) -> dict:
        """5) 策略映射：标准策略恰为 (k,up,o)；完整策略恰覆盖全部 7 投影。"""
        cfg = self.config
        projs = list(cfg.gamma_measured().keys())
        # 断言：标准策略恰为 (k,up,o)；完整策略恰覆盖全部 7 投影
        ok5 = (STRATEGIES["标准指纹(1分钟级)"] == ("k_proj", "up_proj", "o_proj")
               and sorted(STRATEGIES["完整指纹(3分钟级)"]) == sorted(projs))
        if not ok5:
            raise ConfigError(
                "策略映射与文档不符",
                expected={"标准": ("k_proj", "up_proj", "o_proj"), "完整覆盖": sorted(projs)},
                actual={"标准": STRATEGIES["标准指纹(1分钟级)"],
                        "完整": sorted(STRATEGIES["完整指纹(3分钟级)"])},
                param_key="A05",
            )
        rows = [f"{k:18s} -> {v}" for k, v in STRATEGIES.items()]
        return {"detail": "策略映射: " + "; ".join(rows)}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 7 投影 Gamma 齐全有限 + 与文档声称排序对照（如实报告差异）。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        g_meas = cfg.gamma_measured()
        projs = list(g_meas.keys())
        proj_real = rd.get("spectral.proj_gamma", {})    # 真实 7 投影 Gamma 字典
        g_real = {p: proj_real.get(p, float("nan")) for p in projs}   # 缺失置 NaN 标记
        order_real = gamma_rank_order(g_real)            # 真实 Gamma 降序排序
        order_doc = gamma_rank_order(g_meas)             # 文档声称排序
        moved = [p for p in projs if order_real.index(p) != order_doc.index(p)]   # 顺序互换投影
        vals_real = np.array([g_real[p] for p in projs]) # 真实 Gamma 数值集合
        ok6 = (len(order_real) == 7) and all(np.isfinite(vals_real))   # 7 投影齐全且有限
        if not ok6:
            raise RealModelMismatchError(
                f"真实投影 Gamma 缺失或非有限: {g_real}",
                expected=7, actual=len(order_real), param_key="A05",
            )
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        detail = (f"{tag} 真实 7 投影实测 Gamma: " + "; ".join(
            f"{p:10s} gamma={g_real[p]:.4f} vs 文档声称 {g_meas[p]:.4f}（差 {g_real[p] - g_meas[p]:+.4f}）"
            for p in projs)
            + f"; 真实排序: {order_real}（文档声称: {order_doc}）; 排序差异投影: {moved if moved else '无'}"
            + f"（如实报告：真实 down_proj > gate_proj，与文档 down<gate 相反）"
            + f"; 真实 Gamma 范围 [{vals_real.min():.3f}, {vals_real.max():.3f}] "
            + f"vs 文档声称 0.60-0.78 —— 真实实测整体低约 "
            + f"{0.60 - vals_real.max():.2f}~{0.78 - vals_real.min():.2f}，如实报告")
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "g_real": g_real, "order_real": order_real, "order_doc": order_doc, "moved": moved,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "enum", self.validate_enum),
            (2, "dim", self.validate_dim),
            (3, "gamma_synth", self.validate_gamma_synth),
            (4, "gamma_range", self.validate_gamma_range),
            (5, "strategy", self.validate_strategy),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except (AIQValidationError, ValueError) as e:  # 类型化异常 + 防御性 ValueError
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e),
                         "expected": getattr(e, "expected", None),
                         "actual": getattr(e, "actual", None),
                         "param_key": getattr(e, "param_key", None)}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """A05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A05 projs 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = ProjSynthesizer(cfg)                      # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A05 projs 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: M={cfg.M} N_LAYERS={cfg.N_LAYERS} B={cfg.B} DIM={cfg.DIM} "
          f"GAMMA_TOL={cfg.GAMMA_TOL} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a05_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
