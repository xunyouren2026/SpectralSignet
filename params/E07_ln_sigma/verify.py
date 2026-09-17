# -*- coding: utf-8 -*-
"""E07 ln_sigma lognormal 扰动强度 — 尺度不变性与比值-尺度解耦验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. lognormal 乘性扰动 H_pert = H * exp(eps), eps~N(0, ln_sigma^2), ln_sigma=0.8
  2. 比值型 DEFF 平台漂移 ≈ 0（尺度不变量，解析恒等 ~1e-16）
  3. 尺度型 energy_mean 漂移大（相对漂移 ≈ E[s]-1 = e^{sigma^2/2}-1 ≈ 37.7%）
  4. 比值-尺度解耦：corr(logR, phi) ≈ 0（尺度自由度与形状自由度独立）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E07Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 E07Config（env AIQ_E07_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3996-4127（E07 ln_sigma）
  关联指标 307 energy_mean、301 DEFF平台（对照逻辑）
  《参数审计与实验报告.txt》（状态=已用 lnσ=0.8, 平台漂移≈0）
说明：纯数值合成数据（逐点曲率对模拟），不加载任何大模型。

真实模型对照：
  经 _real_data 惰性接入真实曲率对（RD.phi_pairs()，经 _cfg 相对定位），在
  真实曲率对（6303 点）上施加 lognormal 公共尺度扰动，验证 DEFF 平台漂移≈0、
  能量漂移≈37.7%、φ 不变，并如实展示真实比值-尺度解耦 corr(logR,φ)
  （真实 0.1206 vs 审计理想 ≈0）。数据来源标注：[真实实测] 或 [审计回退]。
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
from _errors import (  # noqa: E402
    AIQValidationError,
    ConfigError,
    RealModelMismatchError,
    ReproducibilityError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 E07Config（pydantic 优先；dataclass 回退） ----
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


class E07Config(_ConfigModelBase):
    """E07 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E07/E06 节点键名对应（LN_SIGMA，N_FRAMES/LAM
    联动 E06）；取值优先级：env AIQ_E07_<KEY> > YAML > _params_data.json >
    本模型默认值。
    """

    SEED: int = 0                   # 固定随机种子：保证可复现（算法逻辑常量）
    LN_SIGMA: float = 0.8           # E07: lognormal 扰动强度（README ①）
    N_FRAMES: int = 96              # E06: 联动帧数（README ⑥）
    N_POINT: int = 500              # 每帧逐点曲率对数（原始构造）
    LAMBDA: float = 0.3196          # 曲率比 |k2|/|k1|（DEFF≈1.58 平台，README ④Step 1）
    K1_BASE: float = 1.0            # |k1| 基准（原始构造）
    K1_NOISE: float = 0.10          # |k1| 结构噪声幅度（原始构造）
    K2_NOISE: float = 0.03          # |k2| 附加噪声幅度（原始构造）
    DRIFT_EPS: float = 1e-9         # DEFF 漂移 / E_pert=s*E_orig 解析恒等的判定阈值
    ENERGY_MIN_REL: float = 0.15    # 尺度型能量相对漂移下限（理论 37.7%）
    CORR_MAX: float = 0.10          # 比值-尺度解耦：|corr(logR, phi)| 上限
    PHI_TOL: float = 1e-9           # φ 在乘性扰动下不变性的角度容差（弧度）
    DEFF_EPS: float = 1e-12         # deff_of/phi_of 分母下限
    LOG_EPS: float = 1e-12          # log 输入下限
    REAL_SEED: int = 77             # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E07Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E07 配置工厂：按优先级实例化 E07Config。"""

    def build(self) -> E07Config:
        """构建 E07Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E07Config, "E07")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def deff_of(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """逐点 DEFF = (|a|+|b|)^2 / (a^2 + b^2)（比值型，尺度不变量）。"""
    return (np.abs(a) + np.abs(b)) ** 2 / (a ** 2 + b ** 2 + eps)


def phi_of(a: np.ndarray, b: np.ndarray, eps: float, n_frames: int) -> np.ndarray:
    """主曲率角 phi = atan(|b|/|a|)（比值型，尺度不变量），逐帧取中位。

    返回 (N_FRAMES,) 弧度数组。
    """
    ang = np.arctan(np.abs(b) / (np.abs(a) + eps))  # 比值取反正切（防御除零）
    return np.median(ang, axis=1)                    # 每帧 500 点取中位


def synthesize_curvature(rng: np.random.Generator, cfg: E07Config) -> tuple:
    """合成逐点曲率对（N_FRAMES x N_POINT）。

    模拟鞍面主导结构：|k1|~1, |k2|~lambda, 符号交替；加独立结构噪声使
    phi 有帧间波动。返回 (k1, k2)。
    """
    k1 = cfg.K1_BASE + cfg.K1_NOISE * rng.standard_normal((cfg.N_FRAMES, cfg.N_POINT))
    # k2 = λ·k1 + 附加噪声，并随机取正负号（鞍面主导：符号交替）
    k2 = (cfg.LAMBDA * k1 + cfg.K2_NOISE * rng.standard_normal((cfg.N_FRAMES, cfg.N_POINT))) \
         * np.where(rng.standard_normal((cfg.N_FRAMES, cfg.N_POINT)) > 0, 1.0, -1.0)
    return k1, k2


def _real_phi_deg(k1: np.ndarray, k2: np.ndarray, eps: float) -> np.ndarray:
    """真实口径 φ = arctan2(min, max) 度数（与 harness 一致）。"""
    a1, a2 = np.abs(k1), np.abs(k2)
    small = np.minimum(a1, a2)      # 主曲率较小者（min）
    large = np.maximum(a1, a2)      # 主曲率较大者（max）
    return np.degrees(np.arctan2(small, large + eps))


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E07 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享合成状态在 _synthesize() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: E07Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 单 rng 流一致）----
        self.deff_orig: np.ndarray | None = None
        self.E_orig: np.ndarray | None = None
        self.phi_orig: np.ndarray | None = None
        self.deff_pert: np.ndarray | None = None
        self.E_pert: np.ndarray | None = None
        self.phi_pert: np.ndarray | None = None
        self.drift_deff = 0.0
        self.mean_E_orig = 0.0
        self.rel_energy = 0.0
        self.rel_err_energy = 0.0
        self.phi_mean_orig = 0.0
        self.phi_mean_pert = 0.0
        self.corr_lr_phi = 0.0
        self.E_S = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性合成共享数据（与原脚本 main 的单一 rng 流一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        k1, k2 = synthesize_curvature(rng, cfg)  # 合成逐点曲率对（96x500）
        # ---- 原始（无扰动）----
        self.deff_orig = np.mean(deff_of(k1, k2, cfg.DEFF_EPS), axis=1)   # (96,) 逐帧 DEFF
        self.E_orig = np.mean(np.abs(k1) + np.abs(k2), axis=1)            # (96,) 逐帧能量
        self.phi_orig = phi_of(k1, k2, cfg.DEFF_EPS, cfg.N_FRAMES)        # (96,) 主曲率角
        # ---- lognormal 乘性扰动 ----
        s = np.exp(rng.normal(0, cfg.LN_SIGMA, cfg.N_FRAMES))             # 0.45 ~ 2.23（单步）
        k1p, k2p = k1 * s[:, None], k2 * s[:, None]                       # 公共尺度扰动
        self.deff_pert = np.mean(deff_of(k1p, k2p, cfg.DEFF_EPS), axis=1)
        self.E_pert = np.mean(np.abs(k1p) + np.abs(k2p), axis=1)          # = s * E_orig
        self.phi_pert = phi_of(k1p, k2p, cfg.DEFF_EPS, cfg.N_FRAMES)
        # ---- 指标 ----
        self.drift_deff = float(np.mean(np.abs(self.deff_pert - self.deff_orig)))
        self.mean_E_orig = float(np.mean(self.E_orig))
        self.rel_energy = (float(abs(np.mean(self.E_pert) - self.mean_E_orig)) / self.mean_E_orig
                           if self.mean_E_orig != 0.0 else float("inf"))
        # 解析恒等：E_pert = s * E_orig（逐帧严格成立）
        self.rel_err_energy = float(np.max(
            np.abs(self.E_pert - s * self.E_orig) / (s * self.E_orig + cfg.DEFF_EPS)))
        self.phi_mean_orig = float(np.mean(self.phi_orig)) * 180 / np.pi
        self.phi_mean_pert = float(np.mean(self.phi_pert)) * 180 / np.pi
        # 比值-尺度解耦：corr(logR, phi)，R=逐帧能量（尺度量），phi=主曲率角（比质量）
        logR = np.log(self.E_pert + cfg.LOG_EPS)
        self.corr_lr_phi = float(np.corrcoef(logR, self.phi_pert)[0, 1])
        # lognormal 一阶矩 E[s]=e^{σ²/2}
        self.E_S = float(np.exp(cfg.LN_SIGMA ** 2 / 2))
        self._synced = True

    # ------------------------------------------------------------ 1) DEFF 平台漂移
    def validate_deff_drift(self) -> dict:
        """1) DEFF 漂移≈0（尺度不变量，浮点恒等）。"""
        cfg = self.config
        self._synthesize()
        ok = self.drift_deff < cfg.DRIFT_EPS
        if not ok:
            raise ReproducibilityError(
                f"DEFF 漂移应≈0（浮点恒等），实际 {self.drift_deff:.2e}",
                expected=cfg.DRIFT_EPS, actual=self.drift_deff, param_key="E07",
            )
        return {"detail": f"DEFF 平台漂移: {self.drift_deff:.1e} < 1e-9（尺度不变量）",
                "drift": self.drift_deff}

    # ------------------------------------------------------------ 2) E_pert = s*E_orig 恒等
    def validate_energy_identity(self) -> dict:
        """2) E_pert = s*E_orig 解析恒等（max 相对误差≈0）。"""
        cfg = self.config
        self._synthesize()
        ok = self.rel_err_energy < cfg.DRIFT_EPS
        if not ok:
            raise ReproducibilityError(
                f"E_pert = s*E_orig 应解析恒等，max 相对误差 {self.rel_err_energy:.1e}",
                expected=cfg.DRIFT_EPS, actual=self.rel_err_energy, param_key="E07",
            )
        return {"detail": f"E_pert=s*E_orig 解析恒等: max 相对误差 {self.rel_err_energy:.1e}",
                "rel_err": self.rel_err_energy}

    # ------------------------------------------------------------ 3) 尺度型能量漂移
    def validate_energy_drift(self) -> dict:
        """3) 尺度型能量相对漂移显著（>15%，理论 E[s]-1≈37.7%）。"""
        cfg = self.config
        self._synthesize()
        ok = self.rel_energy > cfg.ENERGY_MIN_REL
        if not ok:
            raise ConfigError(
                f"尺度型能量相对漂移应显著 >15%，实际 {self.rel_energy*100:.1f}%",
                expected=cfg.ENERGY_MIN_REL, actual=self.rel_energy, param_key="E07",
            )
        return {
            "detail": (f"energy_mean 相对漂移: {self.rel_energy*100:.1f}% > 15% "
                       f"(理论 E[s]-1={self.E_S-1:.1%})"),
            "rel_energy": self.rel_energy, "e_s": self.E_S,
        }

    # ------------------------------------------------------------ 4) 比值-尺度解耦
    def validate_decoupling(self) -> dict:
        """4) 比值-尺度解耦 corr(logR, phi) ≈ 0（|corr| < CORR_MAX）。"""
        cfg = self.config
        self._synthesize()
        ok = abs(self.corr_lr_phi) < cfg.CORR_MAX
        if not ok:
            raise ConfigError(
                f"比值-尺度应解耦 corr≈0，实际 {self.corr_lr_phi:+.4f}",
                expected=cfg.CORR_MAX, actual=self.corr_lr_phi, param_key="E07",
            )
        return {"detail": f"比值-尺度解耦: corr(logR, φ)={self.corr_lr_phi:+.4f} ≈ 0",
                "corr": self.corr_lr_phi}

    # ------------------------------------------------------------ 5) φ 扰动不变性
    def validate_phi_invariance(self) -> dict:
        """5) φ 严格不受乘性尺度扰动影响（差值 < PHI_TOL 弧度）。"""
        cfg = self.config
        self._synthesize()
        ok = abs(self.phi_mean_pert - self.phi_mean_orig) < cfg.PHI_TOL * 180 / np.pi
        if not ok:
            raise ConfigError(
                f"φ 应严格不受乘性尺度扰动影响: {self.phi_mean_orig:.6f}° vs {self.phi_mean_pert:.6f}°",
                expected=cfg.PHI_TOL * 180 / np.pi,
                actual=abs(self.phi_mean_pert - self.phi_mean_orig), param_key="E07",
            )
        return {
            "detail": (f"φ 扰动不变性: {self.phi_mean_orig:.2f}° -> {self.phi_mean_pert:.2f}° "
                       f"(差值 {abs(self.phi_mean_pert-self.phi_mean_orig):.2e}°)"),
            "phi_orig_deg": self.phi_mean_orig, "phi_pert_deg": self.phi_mean_pert,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实曲率对上施加 lognormal 公共尺度扰动，验证 DEFF 平台漂移。

        对照实测存档：DEFF=1.5920 / K<0%=45.50% / Hmed=1.3152 / φmean=20.26°
        / λ=0.8516；真实解耦 corr(logR,φ)=0.1206 如实展示（真实为准）。
        数据缺失回退且不判失败。
        """
        cfg = self.config
        rd = self._get_real_data()
        deff_plat_meas = rd.get("curvature.DEFF_plat")
        K_neg = rd.get("curvature.K_neg_pct")
        H_med = rd.get("curvature.H_median")
        phi_mean_meas = rd.get("curvature.phi_mean_deg")
        lam_meas = rd.get("curvature.lambda_ratio")
        corr_real = rd.get("curvature.corr_logR_phi")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        pairs = rd.phi_pairs()   # 真实曲率对（经 _cfg.phi_pairs_path() 相对定位）
        if pairs is None:        # 数据文件缺失：回退审计值不判失败
            return {
                "detail": f"{tag} phi_pairs_all.npy 缺失 -> 对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        k1, k2 = pairs[:, 0], pairs[:, 1]
        deff_plat = float(np.mean(deff_of(k1, k2, cfg.DEFF_EPS)))   # 真实 DEFF 平台均值
        E0 = np.abs(k1) + np.abs(k2)                                # 真实能量（尺度型）
        phi0 = _real_phi_deg(k1, k2, cfg.DEFF_EPS)                  # 真实主曲率角（度数）
        frames = cfg.N_FRAMES
        per = len(k1) // frames                       # 每帧真实曲率点数
        k1f = k1[:per * frames].reshape(frames, per)  # 按帧切分
        k2f = k2[:per * frames].reshape(frames, per)
        rng_r = np.random.default_rng(cfg.REAL_SEED)  # 固定种子：扰动可复现
        s = np.exp(rng_r.normal(0, cfg.LN_SIGMA, frames))  # lognormal 公共尺度因子
        deff_pert = np.mean(deff_of(k1f * s[:, None], k2f * s[:, None], cfg.DEFF_EPS), axis=1)
        deff_orig = np.mean(deff_of(k1f, k2f, cfg.DEFF_EPS), axis=1)
        drift = float(np.mean(np.abs(deff_pert - deff_orig)))  # DEFF 尺度漂移
        E_orig = np.mean(E0[:per * frames].reshape(frames, per), axis=1)
        E_pert = np.mean(np.abs(k1f * s[:, None]) + np.abs(k2f * s[:, None]), axis=1)
        mE0 = float(np.mean(E_orig))
        rel_energy = float(abs(np.mean(E_pert) - mE0) / mE0) if mE0 != 0.0 else float("inf")
        phi_pert = np.mean(_real_phi_deg(k1f * s[:, None], k2f * s[:, None], cfg.DEFF_EPS), axis=1)
        phi_orig = np.mean(_real_phi_deg(k1f, k2f, cfg.DEFF_EPS), axis=1)
        dphi = float(abs(phi_pert.mean() - phi_orig.mean()))  # φ 扰动前后差
        ok1 = abs(deff_plat - deff_plat_meas) < 1e-3   # 平台对齐实测存档
        ok2 = drift < cfg.DRIFT_EPS                    # DEFF 尺度不变量
        ok3 = rel_energy > cfg.ENERGY_MIN_REL          # 能量显著漂移
        ok4 = dphi < cfg.PHI_TOL * 180 / np.pi         # φ 严格不变
        if not (ok1 and ok2 and ok3 and ok4):
            raise RealModelMismatchError(
                f"真实曲率对扰动验证不符: DEFF平台 {deff_plat:.4f}(需≈{deff_plat_meas:.4f}), "
                f"漂移 {drift:.1e}, 能量漂移 {rel_energy*100:.1f}%, dφ={dphi:.2e}°",
                expected={"plat": deff_plat_meas, "drift<": cfg.DRIFT_EPS,
                          "rel_energy>": cfg.ENERGY_MIN_REL, "dphi<": cfg.PHI_TOL * 180 / np.pi},
                actual={"plat": deff_plat, "drift": drift,
                        "rel_energy": rel_energy, "dphi": dphi}, param_key="E07",
            )
        return {
            "detail": (f"{tag} DEFF 平台 {deff_plat:.4f} vs 实测存档 {deff_plat_meas:.4f}; "
                       f"曲率身份: K<0%={K_neg:.2f}% | Hmed={H_med:.4f} | "
                       f"φmean={phi_mean_meas:.2f}° | λ={lam_meas:.4f}; "
                       f"{frames} 帧公共尺度扰动 lnσ={cfg.LN_SIGMA}: DEFF 漂移 {drift:.1e}≈0; "
                       f"能量相对漂移 {rel_energy*100:.1f}%（理论 E[s]-1≈37.7%）; "
                       f"φ 扰动不变性: 差值 {dphi:.2e}°; "
                       f"比值-尺度解耦 corr(logR,φ) = {corr_real:+.4f}（真实实测为准）"),
            "source": rd.source_tag(), "tag": tag,
            "deff_plat": deff_plat, "deff_plat_meas": deff_plat_meas,
            "drift": drift, "rel_energy": rel_energy, "dphi": dphi, "corr_real": corr_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "deff_drift", self.validate_deff_drift),
            (2, "energy_identity", self.validate_energy_identity),
            (3, "energy_drift", self.validate_energy_drift),
            (4, "decoupling", self.validate_decoupling),
            (5, "phi_invariance", self.validate_phi_invariance),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e), "expected": e.expected,
                         "actual": e.actual, "param_key": e.param_key}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """E07 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E07 ln_sigma 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = ValidatorEngine(cfg, report, real_data=RD)

    print("=" * 76)
    print("E07 ln_sigma 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: LN_SIGMA={cfg.LN_SIGMA} N_FRAMES={cfg.N_FRAMES} "
          f"N_POINT={cfg.N_POINT} LAMBDA={cfg.LAMBDA} SEED={cfg.SEED}")
    print("=" * 76)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e07_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e07_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e07_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
