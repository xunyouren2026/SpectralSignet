# -*- coding: utf-8 -*-
"""T07 谱门控增强 激活沿 C-子空间门控 — 操作公式验证（SpectralCognitionPlugin）
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 公式 H_gated = H + s*(Π_C H - H)，等价凸组合 (1-s)H + s·Π_C H
  2. s=0 -> H_gated == H（纯检测模式）；s=1 -> H_gated == Π_C H（完全投影）
  3. 正交性：⟨Π_C H, H - Π_C H⟩ ≈ 0（投影算子性质，README ④Step3）
  4. Gamma（前 M=3 主成分能量占比）随 s 单调不减，且 Γ(1) = 1（谱集中增强）
  5. s=0.3 提升定性对照（Qwen0.5B: Γ 0.17 -> 0.19-0.23，README ③/④Step4）
  6. 投影算子幂等性：Π_C(Π_C H) == Π_C H
  7. 工程化防御：空输入/NaN/Inf/n_comp 越界/全零方差（除零）显式处理
  8. 真实模型实测对照：真实 k_proj Gamma 基准 + 逐层 + 与审计对照
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T07Config / ConfigFactory / T07Synthesizer / T07Validator /
  ReportGenerator / main —— 同 A01（env AIQ_T07_<KEY> 覆盖由共享工厂处理）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T07 节（行 10091-10153）
  源码 spectral-cognition/modules/spectral_gating.py（SpectralGate 类，行 7-14）
  《AI几何指纹插件_参数审计与实验报告.txt》T07（G01 gate_strength 推荐 s=0.3）
真实模型对照（关键发现）：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 k_proj 逐层 Gamma 均值
  =0.4695（_real_metrics.json spectral.k_proj_gamma_mean，24 层），作为 s=0.3
  门控增强的真实基准（Γ0）。真实 0.4695 显著低于文档审计 0.625 —— 这正是
  T04 中 AIQ 由 58.20 降至 49.94 的根因（f1/f4=Gamma）。如实呈现，真实实测为准。
  输出标注：[真实实测]（数据齐全）/ [审计回退]（缺失）。
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
from _errors import AIQValidationError, RealModelMismatchError, SynthesisError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 T07Config（pydantic 优先；dataclass 回退） ----
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


class T07Config(_ConfigModelBase):
    """T07 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T07 节点键名一一对应；取值优先级：
    环境变量 AIQ_T07_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 42               # 合成激活矩阵固定种子（保证可复现，README ⑤）
    B: int = 32                  # 合成激活形状：B=32 批次/帧
    D: int = 16                  # 合成激活形状：D=16 特征维（测试构造）
    N_COMPONENTS: int = 3        # C-子空间维数：n=0 均值+n=1 偶极+n=2 四极（B01 M，README ②）
    M_GAMMA: int = 3             # Gamma 谱集中度主成分数 M=3（README ③）
    GATE_STRENGTH: float = 0.3   # 推荐门控强度（G01 gate_strength，README ①）
    GAMMA_AUDIT: float = 0.625   # 文档审计 k_proj Gamma（AIQ f1/f4 用值）


# ---- 第二层：配置工厂 ConfigFactory（实例化 T07Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T07 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T07Config。"""

    def build(self) -> T07Config:
        """构建 T07Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T07Config, "T07")


# ---- 第三层：合成器 T07Synthesizer（激活合成 + C-子空间投影/门控/谱集中度） ----
class T07Synthesizer(_SynthBase):
    """T07 合成器：合成激活矩阵 + C-子空间投影 + 谱门控 + Gamma 谱集中度（与原脚本一致）。"""

    def __init__(self, cfg: T07Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def make_data(self) -> np.ndarray:
        """合成激活矩阵：随机背景 + 低秩注入 + 去均值（可复现）。

        注入秩 3 结构使谱集中度可测；去均值保证 PCA 数值稳定。
        """
        cfg = self._cfg
        if cfg.B <= 0 or cfg.D <= 0:
            raise SynthesisError(f"make_data: 形状必须为正，收到 {(cfg.B, cfg.D)}",
                                 actual=(cfg.B, cfg.D))
        rng = np.random.default_rng(self._seed)   # 固定种子 => 可复现的合成激活
        H = rng.normal(size=(cfg.B, cfg.D))       # 随机背景（满秩噪声）
        # 注入秩 3 低秩结构：增强前 3 个主方向能量，使 Γ 可测且有梯度
        H = H @ rng.normal(size=(cfg.D, cfg.D)) + 2.0 * rng.normal(size=(cfg.B, 3)) @ rng.normal(size=(3, cfg.D))
        H = H - H.mean(0, keepdims=True)          # 去均值：消除 DC 分量，PCA 数值稳定
        if not np.isfinite(H).all():
            raise SynthesisError("make_data: 生成矩阵含 NaN/Inf", actual=H.shape)
        return H

    def project_c(self, H, n_comp: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """C-子空间投影：均值 + 前 n_comp-1 个主方向（对应 spectral_gating.py）。

        返回 (Π_C H, 均值 μ, 主方向矩阵 V)。空输入、NaN/Inf、n_comp 越界显式报错。
        """
        cfg = self._cfg
        if n_comp is None:
            n_comp = cfg.N_COMPONENTS
        H = np.asarray(H, dtype=float)
        assert H.ndim == 2 and H.shape[0] > 0 and H.shape[1] > 0, \
            f"project_c: 需非空 (B,D) 矩阵，收到 {H.shape}"
        assert np.isfinite(H).all(), f"project_c: 输入含 NaN/Inf: {H}"
        assert n_comp >= 2, f"project_c: n_comp 至少为 2（均值+1主方向），收到 {n_comp}"
        mu = H.mean(0, keepdims=True)             # 常数模态：行均值（n=0 模态）
        Hc = H - mu                               # 去均值激活（PCA 输入）
        cov = Hc.T @ Hc / max(H.shape[0] - 1, 1)  # 样本协方差（贝塞尔校正，防除零）
        evals, evecs = np.linalg.eigh(cov)        # 对称矩阵特征分解（升序）
        idx = np.argsort(evals)[::-1][: n_comp - 1]  # 取前 n_comp-1 个主方向（降序）
        V = evecs[:, idx]
        return mu + (Hc @ V) @ V.T, mu, V         # Π_C H = μ + 去均值投影 + 回加均值

    def gate(self, H, s: float, Pi=None) -> np.ndarray:
        """谱门控：H_gated = H + s*(Π_C H - H) = (1-s)H + s·Π_C H（README ②）。

        s=0 恒等（检测模式），s=1 完全投影到 C-子空间。
        """
        assert np.isfinite(s), f"gate: 门控强度 s 非有限: {s}"
        if Pi is None:
            Pi, _, _ = self.project_c(H)          # 未预投影时现场计算
        return H + s * (Pi - H)                   # 沿 C-子空间方向的线性插值

    def gamma_of(self, X, M: int | None = None) -> float:
        """前 M 个主成分能量占比（谱集中度 Γ，README ③）。

        全零方差（tot=0）返回 nan（防除零，不崩溃）；空/NaN 输入显式报错。
        """
        cfg = self._cfg
        if M is None:
            M = cfg.M_GAMMA
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[0] > 0, f"gamma_of: 需非空 (B,D) 矩阵，收到 {X.shape}"
        assert np.isfinite(X).all(), f"gamma_of: 输入含 NaN/Inf: {X}"
        assert M >= 1, f"gamma_of: M 必须 >= 1，收到 {M}"
        Xc = X - X.mean(0, keepdims=True)         # 去均值（消除 DC 对谱集中度的稀释）
        S = np.linalg.eigvalsh(Xc.T @ Xc)[::-1]   # 协方差特征值（升序->降序）
        tot = S.sum()                             # 总能量 = 特征值之和
        return float(S[:M].sum() / tot) if tot > 0 else float("nan")  # 全零方差防除零->nan


# ---- 第四层：验证引擎 T07Validator（9 项验证 + 结构化日志 + 类型化异常） ----
class T07ValidationError(AIQValidationError):
    """T07 谱门控增强验证失败。"""


class T07Validator(_EngineBase):
    """T07 验证引擎：顺序执行 9 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 合成激活/投影在步骤 1 生成并缓存（self.H/self.Pi_C），后续步骤复用。
    """

    def __init__(
        self,
        config: T07Config,
        synth: T07Synthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.H: np.ndarray | None = None        # 合成激活（步骤 1 生成）
        self.Pi_C: np.ndarray | None = None     # C-子空间投影（步骤 1 生成）
        self.g0: float | None = None            # 门控前基准谱集中度（步骤 5 计算）
        self.gs: list | None = None             # 各 s 的谱集中度（步骤 5 计算）
        self.lift: float | None = None          # s=0.3 相对提升倍数（步骤 6 计算）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) s=0 恒等
    def validate_s0_identity(self) -> dict:
        """1) s=0 -> H_gated == H（纯检测模式）。"""
        cfg = self.config
        assert self.synth is not None
        H = self.synth.make_data()               # 数据准备（固定 SEED，可复现）
        Pi_C, _, _ = self.synth.project_c(H)
        self.H, self.Pi_C = H, Pi_C
        Hg0 = self.synth.gate(H, 0.0, Pi_C)
        ok = bool(np.allclose(Hg0, H, atol=1e-12))
        if not ok:
            raise T07ValidationError(
                f"s=0 非恒等: max|H_gated-H|={np.max(np.abs(Hg0 - H)):.3e}",
                expected=0.0, actual=float(np.max(np.abs(Hg0 - H))), param_key="T07",
            )
        return {"detail": f"s=0 -> H_gated == H（纯检测模式）", "ok": ok}

    # ------------------------------------------------------------ 2) s=1 完全投影
    def validate_s1_project(self) -> dict:
        """2) s=1 -> H_gated == Π_C H（完全投影）。"""
        assert self.synth is not None and self.H is not None and self.Pi_C is not None
        Hg1 = self.synth.gate(self.H, 1.0, self.Pi_C)
        ok = bool(np.allclose(Hg1, self.Pi_C, atol=1e-9))
        if not ok:
            raise T07ValidationError(
                f"s=1 未达完全投影: max|H_gated-Π_CH|={np.max(np.abs(Hg1 - self.Pi_C)):.3e}",
                expected=0.0, actual=float(np.max(np.abs(Hg1 - self.Pi_C))), param_key="T07",
            )
        return {"detail": "s=1 -> H_gated == Pi_C H（完全投影）", "ok": ok}

    # ------------------------------------------------------------ 3) 正交性
    def validate_orthogonality(self) -> dict:
        """3) 正交性：Π_C H ⊥ (H - Π_C H)（投影定理，归一化内积 ≈ 0）。"""
        assert self.H is not None and self.Pi_C is not None
        resid = self.H - self.Pi_C                 # 补空间残差
        orth = abs(float(np.sum(self.Pi_C * resid))) / (
            np.linalg.norm(self.Pi_C) * np.linalg.norm(resid) + 1e-12)  # 归一化内积
        ok = bool(orth < 1e-10)
        if not ok:
            raise T07ValidationError(
                f"正交性不成立: 归一化内积 {orth:.2e} 应 < 1e-10",
                expected=1e-10, actual=orth, param_key="T07",
            )
        return {"detail": (f"正交性: <Pi_C H, H-Pi_C H>/‖·‖ = {orth:.2e} ≈ 0"),
                "orth": orth}

    # ------------------------------------------------------------ 4) 凸组合
    def validate_convex(self) -> dict:
        """4) 凸组合/线性性：H_gated(0.3) = 0.7H + 0.3·Π_C H。"""
        cfg = self.config
        assert self.synth is not None and self.H is not None and self.Pi_C is not None
        s_ref = cfg.GATE_STRENGTH
        Hg3 = self.synth.gate(self.H, s_ref, self.Pi_C)
        expect3 = (1 - s_ref) * self.H + s_ref * self.Pi_C
        ok = bool(np.allclose(Hg3, expect3, atol=1e-9))
        if not ok:
            raise T07ValidationError(
                f"凸组合不成立: max|H_gated-expect|={np.max(np.abs(Hg3 - expect3)):.3e}",
                expected=0.0, actual=float(np.max(np.abs(Hg3 - expect3))), param_key="T07",
            )
        return {"detail": f"凸组合: H_gated({s_ref}) = {1 - s_ref}H + {s_ref}·Pi_C H", "ok": ok}

    # ------------------------------------------------------------ 5) Gamma 单调性
    def validate_gamma_mono(self) -> dict:
        """5) Gamma 随 s 单调不减（谱集中增强），且 Γ(1) = 1。"""
        assert self.synth is not None and self.H is not None and self.Pi_C is not None
        g0 = self.synth.gamma_of(self.H)          # 门控前基准谱集中度
        gs = [self.synth.gamma_of(self.synth.gate(self.H, s, self.Pi_C))
              for s in [0.0, 0.15, 0.3, 0.5, 0.8, 1.0]]
        self.g0, self.gs = g0, gs
        mono = bool(all(gs[i] <= gs[i + 1] + 1e-12 for i in range(len(gs) - 1)))
        ok = bool(mono and np.isclose(gs[-1], 1.0, atol=1e-9, rtol=0.0))
        if not ok:
            raise T07ValidationError(
                f"Gamma 单调性或 Γ(1)=1 不成立: gs={gs}",
                expected={"mono": True, "gamma1": 1.0},
                actual={"mono": mono, "gamma1": gs[-1]}, param_key="T07",
            )
        return {"detail": (f"Gamma 随 s 单调不减: {[f'{v:.4f}' for v in gs]} "
                           f"(s=0..1), Γ(1)=1.0"),
                "g0": g0, "gs": gs}

    # ------------------------------------------------------------ 6) s=0.3 提升
    def validate_lift(self) -> dict:
        """6) s=0.3 提升定性对照（推荐强度应带来谱集中提升）。"""
        cfg = self.config
        assert self.gs is not None and self.g0 is not None
        # gs 索引约定：s 序列 [0.0, 0.15, 0.3, 0.5, 0.8, 1.0] -> gs[2] 即 s=0.3
        lift = self.gs[2] / max(self.g0, 1e-12)   # 门控后相对提升倍数（ε 防除零）
        self.lift = lift
        ok = bool(lift > 1.0)
        if not ok:
            raise T07ValidationError(
                f"s={cfg.GATE_STRENGTH} 未提升: Γ {self.g0:.4f} -> {self.gs[2]:.4f}",
                expected="> 1.0x", actual=lift, param_key="T07",
            )
        return {"detail": (f"s={cfg.GATE_STRENGTH} 提升对照: Γ {self.g0:.4f} -> "
                           f"{self.gs[2]:.4f} (提升 {lift:.2f}x; 文档 Qwen0.5B "
                           f"0.17->0.19-0.23 定性一致)"),
                "g0": self.g0, "g03": self.gs[2], "lift": lift}

    # ------------------------------------------------------------ 7) 幂等性
    def validate_idempotent(self) -> dict:
        """7) 投影算子幂等性：Π_C(Π_C H) == Π_C H。"""
        assert self.synth is not None and self.Pi_C is not None
        Pi2, _, _ = self.synth.project_c(self.Pi_C)
        ok = bool(np.allclose(Pi2, self.Pi_C, atol=1e-9))
        if not ok:
            raise T07ValidationError(
                f"幂等性不成立: max|Π_C(Π_CH)-Π_CH|={np.max(np.abs(Pi2 - self.Pi_C)):.3e}",
                expected=0.0, actual=float(np.max(np.abs(Pi2 - self.Pi_C))), param_key="T07",
            )
        return {"detail": "幂等性: Π_C(Π_C H) == Π_C H（投影算子）", "ok": ok}

    # ------------------------------------------------------------ 8) 防御
    def validate_guards(self) -> dict:
        """8) 工程化防御：空输入 / NaN / n_comp 越界 / s 非有限 显式报错。"""
        cfg = self.config
        assert self.synth is not None and self.H is not None and self.Pi_C is not None
        guards = [
            ("project_c 空输入", lambda: self.synth.project_c(np.empty((0, cfg.D)))),
            ("project_c n_comp=1", lambda: self.synth.project_c(self.H, n_comp=1)),
            ("gamma_of 空输入", lambda: self.synth.gamma_of(np.empty((0, cfg.D)))),
            ("gamma_of NaN", lambda: self.synth.gamma_of(self.H * float("nan"))),
            ("gate s=NaN", lambda: self.synth.gate(self.H, float("nan"), self.Pi_C)),
        ]
        guard_oks = []
        for _gname, fn in guards:
            try:
                fn()                             # 未抛异常 => 防御缺失
                guard_oks.append(False)
            except AssertionError:
                guard_oks.append(True)           # 显式报错 => 防御通过
        ok_guard = all(guard_oks)
        if not ok_guard:
            raise T07ValidationError(
                f"边界防御失败: {[g[0] for g, o in zip(guards, guard_oks) if not o]}",
                expected="all raise", actual=guard_oks, param_key="T07",
            )
        return {"detail": f"防御: 空/NaN/n_comp越界/s非有限 均显式处理 {guard_oks}",
                "guard_oks": guard_oks}

    # ------------------------------------------------------------ 9) 全零方差
    def validate_zerovar(self) -> dict:
        """9) 防御：全零方差 Gamma -> nan（防除零）。"""
        cfg = self.config
        assert self.synth is not None
        g_zero = self.synth.gamma_of(np.zeros((cfg.B, cfg.D)))
        ok = bool(np.isnan(g_zero))
        if not ok:
            raise T07ValidationError(
                f"全零方差应返回 nan，实得 {g_zero}",
                expected="nan", actual=g_zero, param_key="T07",
            )
        return {"detail": f"防御: 全零方差 Gamma -> nan（防除零） {g_zero}",
                "g_zero": g_zero}

    # ------------------------------------------------------------ 10) 真实模型对照
    def validate_real(self) -> dict:
        """10) 真实模型实测对照：真实 k_proj Gamma 基准（24 层均值）域校验 + 逐层 + vs 审计。"""
        cfg = self.config
        rd = self._get_real_data()
        src = rd.source_tag()
        gamma_real = rd.get("spectral.k_proj_gamma_mean", None)   # 24 层均值
        gamma_layers_real = rd.get("spectral.k_proj_gamma_layers", None)  # 逐层 24 维
        if gamma_real is None:
            return {"detail": (f"[审计回退] 真实 Gamma 缺失，维持文档审计基准 "
                               f"{cfg.GAMMA_AUDIT}"), "fallback": True}
        g_baseline_real = float(gamma_real)
        ok_base = bool(0.0 <= g_baseline_real <= 1.0)   # Gamma 值域 [0,1] 校验
        if not ok_base:
            raise RealModelMismatchError(
                f"真实 Gamma 越出 [0,1]: {g_baseline_real}",
                expected="[0,1]", actual=g_baseline_real, param_key="T07",
            )
        details = [(f"真实 k_proj Gamma 基准(24层均值) = {g_baseline_real:.4f} "
                    f"(域 [0,1] 校验)")]
        extra: dict[str, Any] = {"source": src, "gamma_real": g_baseline_real}
        if gamma_layers_real is not None:
            gl = np.asarray(gamma_layers_real, dtype=float)
            ok_layer = bool(gl.size == 24 and np.isfinite(gl).all())  # 逐层数据健康
            if not ok_layer:
                raise RealModelMismatchError(
                    f"真实逐层 Gamma 异常: n={gl.size}, finite={np.isfinite(gl).all()}",
                    expected={"n": 24, "finite": True},
                    actual={"n": int(gl.size), "finite": bool(np.isfinite(gl).all())},
                    param_key="T07",
                )
            details.append(f"真实逐层 24 维: min={gl.min():.3f} max={gl.max():.3f} n={gl.size}")
            extra["gamma_layers"] = [float(x) for x in gl]
        # 关键发现：真实 Gamma vs 文档审计（AIQ f1/f4 用值，真实实测为准）
        g_audit = rd.audit("k_proj_gamma_mean", cfg.GAMMA_AUDIT)
        diff_g = g_baseline_real - g_audit
        ok_gdiff = bool(diff_g < 0)               # 真实 Gamma 应低于审计值
        if not ok_gdiff:
            raise RealModelMismatchError(
                f"真实 Gamma {g_baseline_real} 应低于文档审计 {g_audit}",
                expected=f"< {g_audit}", actual=g_baseline_real, param_key="T07",
            )
        details.append(f"真实 vs 审计: k_proj Gamma 真实={g_baseline_real:.4f} "
                       f"审计={g_audit:.3f} (Δ={diff_g:+.4f}); 影响: 文档 AIQ 以 0.625 作 "
                       f"f1/f4；真实更低 -> AIQ 58.20 -> 49.94（见 T04 真实对照）")
        # s=0.3 增强基准锚定：真实 Γ0 为门控起点（公式级增强已在合成层验证）
        ok_anchor = bool(g_baseline_real > 0.0)
        if not ok_anchor:
            raise RealModelMismatchError(
                f"真实基准 Γ0={g_baseline_real} 非正，锚定失效",
                expected="> 0", actual=g_baseline_real, param_key="T07",
            )
        details.append(f"s=0.3 增强基准: 真实 Γ0={g_baseline_real:.4f} 为门控起点 "
                       f"(合成层已验证 Γ(s=0.3)={self.gs[2]:.4f} > Γ(0)={self.g0:.4f}, "
                       f"提升 {self.lift:.2f}x; 真实激活的门控需另存激活重测); "
                       f"对比: 合成基准 Γ0={self.g0:.4f} vs 真实 Γ0={g_baseline_real:.4f}")
        extra["diff_vs_audit"] = diff_g
        extra["g_audit"] = g_audit
        return {"detail": f"[{src}] " + "; ".join(details), **extra}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 10 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "s0_identity", self.validate_s0_identity),
            (2, "s1_project", self.validate_s1_project),
            (3, "orthogonality", self.validate_orthogonality),
            (4, "convex", self.validate_convex),
            (5, "gamma_mono", self.validate_gamma_mono),
            (6, "lift", self.validate_lift),
            (7, "idempotent", self.validate_idempotent),
            (8, "guards", self.validate_guards),
            (9, "zerovar", self.validate_zerovar),
            (10, "real", self.validate_real),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:  # 类型化异常：携带 expected/actual
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e), "expected": e.expected,
                         "actual": e.actual, "param_key": e.param_key}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """T07 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T07 谱门控增强 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = T07Synthesizer(cfg)                       # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T07Validator(cfg, synth, report, real_data=RD)  # ③ 验证层

    print("=" * 74)
    print("T07 谱门控增强  H_gated = H + s*(Pi_C H - H)")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: GATE_STRENGTH={cfg.GATE_STRENGTH} N_COMPONENTS={cfg.N_COMPONENTS} "
          f"M_GAMMA={cfg.M_GAMMA} B={cfg.B} D={cfg.D}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t07_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t07_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t07_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
