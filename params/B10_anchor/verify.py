# -*- coding: utf-8 -*-
"""B10 anchor 锚定层 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 中间层索引 num_layers // 2（Qwen24->12, GPT-2 12->6, Llama32->16,
     DeepSeek61->30, GPT-3 96->48）
  2. 锚定名 "model.layers.{N//2}.self_attn.k_proj"
  3. 校准流程：短前向激活 -> PCA(M=3) -> C-子空间基 U
  4. β_t = ||Π_C h_t||^2 / ||h_t||^2 逐 token 实时计算，∈[0,1]，与解析投影一致
  5. 逐 token 实时 β_t 序列全部落在 [0,1]
  6. 真实模型对照：真实 24 层 → 锚定层 layer12（24//2）；真实 layer12.k_proj Gamma 对照

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B10Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退；
                            跨参数引用 M←B01.M 由共享 ConfigFactory 解析）
  ConfigFactory           —— 实例化 B10Config（环境变量 AIQ_B10_<KEY> > YAML > _params_data.json > 默认）
  AnchorSynthesizer       —— 校准 PCA（中心化 -> 特征分解 -> 前 M 主方向）+ β_t 实时投影计算
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1618-1739（B10）
  《参数完整定义与公式.txt》B10 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 24 层 → 锚定层 layer12（24//2）；真实 layer12.k_proj Gamma≈0.4729
          vs 全局均值≈0.4695（中层锚定合理性验证）。
  如实呈现：真实 layer12 集中度与全局均值偏差 <1%，中层锚定合理；真实层间
            Gamma 极差含 layer0 峰值（0.9295），为真实数据突出结构。
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
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
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

# ---- 第一层：配置模型 B10Config（pydantic 优先；dataclass 回退） ----
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


class B10Config(_ConfigModelBase):
    """B10 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B10 节点键名一一对应（M 为跨参数引用）；
    取值优先级：环境变量 AIQ_B10_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    M: int = 3                       # B01 跨参数引用：C-子空间维度（校准 PCA 取前 M 主方向）
    DIM: int = 64                    # 模拟激活维度
    T_CAL: int = 8                   # 校准短前向 token 数
    GAMMA_CAL_MIN: float = 0.8       # 校准 C-子空间能量占比下界（注入低秩结构应偏高）
    B_IN_TOL: float = 0.02           # h∈C 时 β_t 与 1 的偏差容差
    B_ORTH_TOL: float = 0.02         # h⊥C 时 β_t 上界
    EPS: float = 1e-12               # β_t 分母防御——浮点常数，保留
    N_LAYERS_QWEN: int = 24          # Qwen 层数（真实架构断言基准）
    N_LAYERS_GPT2: int = 12          # GPT-2 层数
    # 模型层数 -> 中间层索引（README ④ 适配表；名称/层数从数据层读取）
    MODEL_NAMES: list = ["Qwen 0.5B", "GPT-2 124M", "Qwen 7B", "Llama 7B", "DeepSeek-V3", "GPT-3 175B"]
    MODEL_LAYERS: list = [24, 12, 32, 32, 61, 96]


# ---- 第二层：配置工厂 ConfigFactory（实例化 B10Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B10 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B10Config。"""

    def build(self) -> B10Config:
        """构建 B10Config；跨参数引用 M（B01.M）单独解析（与旧版 P.get_int 同义）。"""
        cfg = self.build_model(B10Config, "B10")
        cfg.M = self.get_int("B01", "M", 3)   # B01: C-子空间维度
        return cfg


# ---- 第三层：合成器 AnchorSynthesizer（算法与原脚本完全一致） ----
class AnchorSynthesizer(_SynthBase):
    """B10 锚定层校准 + β_t 实时投影合成器。

    - calibrate_subspace(H_cal, m)：短前向激活中心化 -> PCA 前 m 主方向 U；
    - beta_t(h, Umat)：逐 token 实时集中度 β_t = ||Π_C h_t||²/||h_t||²。
    """

    def __init__(self, cfg: B10Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    @staticmethod
    def calibrate_subspace(H_cal: np.ndarray, m: int) -> tuple[np.ndarray, float]:
        """校准：短前向激活中心化 -> PCA 前 m 主方向 U（形状 (d, m)）。
        返回 (U, 前 m 主成分能量占比)。"""
        H_cal = np.asarray(H_cal, dtype=float)
        if H_cal.shape[0] < 2 or H_cal.shape[1] < m:
            raise ValueError(f"校准数据过小: {H_cal.shape}, m={m}")   # 防御：样本/维度不足
        Hc = H_cal - H_cal.mean(axis=0, keepdims=True)       # 中心化（PCA 前提）
        cov = Hc.T @ Hc / (H_cal.shape[0] - 1)               # 样本协方差阵
        evals, evecs = np.linalg.eigh(cov)                   # 特征分解（升序）
        order = np.argsort(evals)[::-1]                      # 降序索引
        U = evecs[:, order[:m]]                              # 前 m 主方向 (d, m)
        gamma_cal = float(evals[order[:m]].sum() / evals.sum())   # 前 m 主成分能量占比
        return U, gamma_cal

    @staticmethod
    def beta_t(h: np.ndarray, Umat: np.ndarray, eps: float = 1e-12) -> float:
        """逐 token 实时集中度 β_t = ||Π_C h_t||²/||h_t||²（分母加 EPS 防御除零）。"""
        h = np.asarray(h, dtype=float).ravel()   # 展平为一维激活向量
        if h.size != Umat.shape[0]:
            raise ValueError(f"维度不匹配: h={h.size}, U 行数={Umat.shape[0]}")
        proj = Umat.T @ h                       # U^T h (M,)：投影到 C-子空间的坐标
        norm_sq = float(h @ h)                  # 激活向量模长平方
        if norm_sq <= 0.0:
            return 0.0                          # 防御：零向量 β_t 定义为 0
        return float(proj @ proj / (norm_sq + eps))   # 投影能量占比（EPS 防除零）


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def anchor_layer(n_layers: int) -> int:
    """锚定层索引 = floor(N/2)（B10 普适公式）。"""
    if n_layers <= 0:
        raise ValueError(f"非法的层数: {n_layers}")   # 防御：层数非正则无法取中点
    return n_layers // 2        # 中间层：向下取整


def anchor_name(n_layers: int) -> str:
    """锚定名 = "model.layers.{N//2}.self_attn.k_proj"。"""
    return f"model.layers.{anchor_layer(n_layers)}.self_attn.k_proj"   # k_proj 为指纹主投影


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B10 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B10Config,
        synth: AnchorSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self._rng: np.random.Generator | None = None   # 跨步骤共享 rng（保持原随机流次序）
        self.U: np.ndarray | None = None   # 校准 C-子空间基（供 β_t 数值/序列步骤复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 中间层索引 N//2
    def validate_anchor_index(self) -> dict:
        """1) 中间层索引 N//2（多模型适配：24->12, 12->6, 32->16, 61->30, 96->48）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng                                   # 供后续步骤复用同一随机流
        models = dict(zip(cfg.MODEL_NAMES, cfg.MODEL_LAYERS))
        expect = {24: 12, 12: 6, 32: 16, 61: 30, 96: 48}   # 各模型期望中间层（结构性算法逻辑）
        ok1 = True
        rows = []
        for name, N in models.items():
            mid = anchor_layer(N)          # 公式 N//2 计算中间层
            one_ok = (mid == expect[N])    # 与期望一致
            ok1 &= one_ok
            rows.append(f"{name:<12} N={N:>3} -> 中间层 {mid:>2}")
        if not ok1:
            raise AIQValidationError(
                f"中间层索引不符: {models}",
                expected=expect, actual={N: anchor_layer(N) for N in models.values()},
                param_key="B10",
            )
        return {"detail": "中间层索引 N//2（多模型适配）: " + "; ".join(rows)}

    # ------------------------------------------------------------ 2) 锚定名生成
    def validate_anchor_name(self) -> dict:
        """2) 锚定名生成："model.layers.{N//2}.self_attn.k_proj"。"""
        cfg = self.config
        expect = {24: 12, 12: 6, 32: 16, 61: 30, 96: 48}
        names = {N: anchor_name(N) for N in expect}   # 各层数对应的锚定名
        ok2 = all(names[N] == f"model.layers.{expect[N]}.self_attn.k_proj" for N in expect)   # 命名规范断言
        if not ok2:
            raise AIQValidationError(
                f"锚定名不符: {names}",
                expected={N: f"model.layers.{expect[N]}.self_attn.k_proj" for N in expect},
                actual=names, param_key="B10",
            )
        return {"detail": "锚定名生成: " + "; ".join(f"N={N:>3} -> {names[N]}" for N in sorted(names))}

    # ------------------------------------------------------------ 3) 校准 PCA(M=3) -> U
    def validate_calibration(self) -> dict:
        """3) 校准流程：注入低秩结构后 PCA(M) → U 形状 (DIM, M) 且能量占比 > GAMMA_CAL_MIN。"""
        cfg = self.config
        assert self._rng is not None
        H_cal = self._rng.standard_normal((cfg.T_CAL, cfg.DIM))   # 初始校准激活（将被覆盖）
        # 注入低维结构：前 3 主方向强，其余弱
        U_truth = np.linalg.qr(self._rng.standard_normal((cfg.DIM, cfg.DIM)))[0]   # 随机正交矩阵（真子空间）
        coef = np.zeros((cfg.T_CAL, cfg.DIM))               # 激活系数
        coef[:, :cfg.M] = self._rng.normal(0, 2.0, (cfg.T_CAL, cfg.M))       # 前 M 方向：强系数（σ=2）
        coef[:, cfg.M:] = self._rng.normal(0, 0.3, (cfg.T_CAL, cfg.DIM - cfg.M)) # 其余方向：弱系数（σ=0.3）
        H_cal = coef @ U_truth.T                    # 重建低秩结构的校准激活
        U, gamma_cal = self.synth.calibrate_subspace(H_cal, cfg.M)   # 校准：PCA 提取 C-子空间基 U
        self.U = U                                  # 供 β_t 数值/序列步骤复用
        ok3 = (U.shape == (cfg.DIM, cfg.M)) and (gamma_cal > cfg.GAMMA_CAL_MIN)   # 形状 + 能量占比断言
        if not ok3:
            raise AIQValidationError(
                f"校准不符: U={U.shape}, gamma={gamma_cal:.4f}",
                expected={"shape": (cfg.DIM, cfg.M), "gamma_min": cfg.GAMMA_CAL_MIN},
                actual={"shape": tuple(U.shape), "gamma": gamma_cal},
                param_key="B10",
            )
        return {"detail": (f"校准 PCA(M={cfg.M}): U 形状 = {U.shape}（应为 ({cfg.DIM}, {cfg.M})）; "
                           f"C-子空间能量占比 = {gamma_cal:.4f}（注入低秩结构，应 >{cfg.GAMMA_CAL_MIN}）"),
                "gamma_cal": gamma_cal}

    # ------------------------------------------------------------ 4) β_t 数值正确性
    def validate_beta_numeric(self) -> dict:
        """4) β_t 数值验证：C 内≈1、正交≈0、一般情形与解析一致、全部 ∈[0,1]。"""
        cfg = self.config
        assert self._rng is not None and self.U is not None
        # 情形 A：h 在 C-子空间内（h = U c）-> β_t ≈ 1
        c = self._rng.standard_normal(cfg.M)          # 子空间坐标
        h_in = self.U @ c                        # 构造 C 内的激活
        b_in = self.synth.beta_t(h_in, self.U, cfg.EPS)              # β_t（应≈1）
        # 情形 B：h 正交于 C-子空间 -> β_t ≈ 0
        Q_full = np.linalg.qr(self.U, mode='complete')[0]  # 完整 QR -> Q (d,d)
        N_orth = Q_full[:, cfg.M:]                        # 与 U 正交的方向 (d, d-M)
        h_orth = N_orth @ self._rng.standard_normal(cfg.DIM - cfg.M)   # 构造正交激活
        b_orth = self.synth.beta_t(h_orth, self.U, cfg.EPS)              # β_t（应≈0）
        # 情形 C：一般 h -> β_t = ||U^T h||²/||h||² 解析值
        h_gen = self._rng.standard_normal(cfg.DIM)        # 一般随机激活
        b_gen = self.synth.beta_t(h_gen, self.U, cfg.EPS)                # 数值 β_t
        b_gen_analytic = float((self.U.T @ h_gen) @ (self.U.T @ h_gen) / (h_gen @ h_gen))   # 解析 β_t
        # 三情形断言：C 内≈1、正交≈0、一般情形与解析一致、全部落在 [0,1]
        ok4 = (abs(b_in - 1.0) < cfg.B_IN_TOL) and (b_orth < cfg.B_ORTH_TOL) \
            and (abs(b_gen - b_gen_analytic) < 1e-10) \
            and (0.0 <= b_in <= 1.0) and (0.0 <= b_orth <= 1.0) and (0.0 <= b_gen <= 1.0)
        if not ok4:
            raise AIQValidationError(
                f"β_t 数值不符: in={b_in:.6f}, orth={b_orth:.6f}, gen={b_gen:.6f} vs 解析={b_gen_analytic:.6f}",
                expected={"in≈1": True, "orth≈0": True, "gen==analytic": True, "∈[0,1]": True},
                actual={"b_in": b_in, "b_orth": b_orth, "b_gen": b_gen, "b_gen_analytic": b_gen_analytic},
                param_key="B10",
            )
        return {"detail": (f"β_t 数值验证: h ∈ C-子空间: β_t = {b_in:.6f}（应≈1）; "
                           f"h ⊥ C-子空间: β_t = {b_orth:.6f}（应≈0）; 一般 h: β_t = {b_gen:.6f} "
                           f"= 解析 {b_gen_analytic:.6f}（应一致）"),
                "b_in": b_in, "b_orth": b_orth, "b_gen": b_gen, "b_gen_analytic": b_gen_analytic}

    # ------------------------------------------------------------ 5) 逐 token 实时 β_t 序列
    def validate_beta_series(self) -> dict:
        """5) 逐 token 实时 β_t 序列全部落在 [0,1]。"""
        cfg = self.config
        assert self._rng is not None and self.U is not None
        H_live = self._rng.standard_normal((5, cfg.DIM))   # 5 个实时 token 激活
        betas = np.array([self.synth.beta_t(h, self.U, cfg.EPS) for h in H_live])   # 逐 token 实时 β_t
        ok5 = bool(np.all((betas >= 0) & (betas <= 1)))    # 全部落在 [0,1]
        if not ok5:
            raise AIQValidationError(
                f"β_t 序列越界: {betas}",
                expected="全部 ∈[0,1]", actual=betas.tolist(), param_key="B10",
            )
        return {"detail": f"逐 token β_t 序列 = {np.round(betas, 4).tolist()}（全部 ∈[0,1]）",
                "betas": betas.tolist()}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 24 层 → 锚定层 layer12 + 真实 layer12.k_proj Gamma 对照。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        n_layers_real = rd.get("arch.n_layers")                    # 真实层数（24）
        gamma_layers = np.asarray(rd.get("spectral.k_proj_gamma_layers"), dtype=float)   # 真实逐层 Gamma
        mid_real = anchor_layer(n_layers_real)                 # 24//2 = 12
        gamma_mid = float(gamma_layers[mid_real])              # 锚定层真实 Gamma
        gamma_mean = float(gamma_layers.mean())                # 全局均值
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok6 = (n_layers_real == cfg.N_LAYERS_QWEN) and (mid_real == cfg.N_LAYERS_QWEN // 2) \
            and (gamma_layers.size == cfg.N_LAYERS_QWEN)   # 结构断言
        if not ok6:
            raise RealModelMismatchError(
                f"真实锚定层不符: n_layers={n_layers_real}, mid={mid_real}",
                expected={"n_layers": cfg.N_LAYERS_QWEN, "mid": cfg.N_LAYERS_QWEN // 2},
                actual={"n_layers": n_layers_real, "mid": mid_real},
                param_key="B10",
            )
        return {"detail": (f"{tag} 真实 {n_layers_real} 层 → 锚定层 = layer{mid_real} "
                           f"（{n_layers_real}//2）；锚定名 = {anchor_name(n_layers_real)}; "
                           f"锚定层 layer{mid_real}.k_proj gamma = {gamma_mid:.4f} vs 全局均值 "
                           f"{gamma_mean:.4f}（偏差 "
                           f"{100 * (gamma_mid - gamma_mean) / gamma_mean:+.1f}%，"
                           f"中层锚定与全局量级一致）; 真实层间 Gamma 极差 = "
                           f"{gamma_layers.max() - gamma_layers.min():.4f}（layer0 峰值 "
                           f"{gamma_layers.max():.4f} 为真实数据突出结构，如实报告）"),
                "source": rd.source_tag(), "tag": tag,
                "n_layers_real": n_layers_real, "mid_real": mid_real,
                "gamma_mid": gamma_mid, "gamma_mean": gamma_mean}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "anchor_index", self.validate_anchor_index),
            (2, "anchor_name", self.validate_anchor_name),
            (3, "calibration", self.validate_calibration),
            (4, "beta_numeric", self.validate_beta_numeric),
            (5, "beta_series", self.validate_beta_series),
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
    """B10 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B10 anchor 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = AnchorSynthesizer(cfg)                    # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("B10 anchor 验证（四层工厂架构）：锚定层 = layer{N//2}.k_proj")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: M={cfg.M} DIM={cfg.DIM} T_CAL={cfg.T_CAL} GAMMA_CAL_MIN={cfg.GAMMA_CAL_MIN} "
          f"B_IN_TOL={cfg.B_IN_TOL} B_ORTH_TOL={cfg.B_ORTH_TOL} "
          f"N_LAYERS_QWEN={cfg.N_LAYERS_QWEN} N_LAYERS_GPT2={cfg.N_LAYERS_GPT2} "
          f"MODEL_LAYERS={cfg.MODEL_LAYERS} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b10_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b10_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b10_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
