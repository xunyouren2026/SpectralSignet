# -*- coding: utf-8 -*-
"""H04 data_file 逐点 (κ1,κ2) 数据集 — 数据驱动指标复算与审计对照验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 加载真实数据集 phi_pairs_all.npy，断言形状 (6303, 2) 与 dtype float64
  2. 复算全部核心指标并与审计报告实测值对比（容差内一致）
     报告值（AI几何指纹插件_参数审计与实验报告.txt 第一部分）：
       DEFF平台=1.5920 | λ=0.8516 | K<0%=45.50% | 同号率=0.5450
       corr(κ1,κ2)=-0.9153 | corr(|κ1|,|κ2|)=0.9339
       φ均值=20.26°  φ中位=19.75° | Hmed=1.3152 | 能量均值=6.1855
  3. DEFF 平台 ≈ π/2（1.5708），K<0% 落在槽 [20,57]
  4. 比值-尺度解耦 corr(logR, φ) ≈ 0（审计报告附加项）
  5. φ 直方图分布摘要（附注，不参与判定）
  6. 真实模型对照：与 _real_metrics.json 真实测量曲率指标逐项对照
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  H04Config                —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory            —— 实例化 H04Config（env AIQ_H04_<KEY> 覆盖）
  H04DataSynthesizer       —— 合成回退数据生成（真实文件缺失时结构验证）
  H04Validator(ValidatorEngine) —— 6 项验证 + 结构化 JSON 日志（_logging）
                            + 类型化异常（_errors，携带 expected/actual）
  ReportGenerator / main   —— 报告 + 退出码 0/1（复用 _factory/_common 基类）
数据源：
  主文档行 5821-5930（H04 章节，README ⑤ 对照表）
  数据文件 phi_pairs_all.npy（经 _cfg.phi_pairs_path() 相对定位，零硬编码）
  《参数审计与实验报告.txt》行 84（状态=已用）
真实模型对照：
  本参数本就是真实数据：phi_pairs_all.npy（6303×2 float64）；本地复算值
  与 _real_metrics.json 曲率实测（DEFF=1.5920、λ=0.8516、K<0%=45.50% 等）
  逐项对照（同源数据应一致）。真实数据缺失时退化为合成数据做结构验证
  （不加载任何大模型）。
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
    SynthesisError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（CFG 供数据文件相对定位）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 H04Config（pydantic 优先；dataclass 回退） ----
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


class H04Config(_ConfigModelBase):
    """H04 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 H04 节点键名一一对应；取值优先级：
    环境变量 AIQ_H04_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # 合成回退数据固定种子（H01 语义）
    N_SAMPLES: int = 6303      # 数据集样本数（README ①）
    DEFF_REF: float = 1.5920   # DEFF 平台审计参考（202）
    LAMBDA_REF: float = 0.8516  # λ = σ2/σ1 审计参考（306）
    K_NEG_REF: float = 45.50   # K<0% 审计参考（201）
    SAME_SIGN_REF: float = 0.5450   # 同号率审计参考
    CORR_RAW_REF: float = -0.9153   # corr(κ1,κ2) 审计参考
    CORR_ABS_REF: float = 0.9339    # corr(|κ1|,|κ2|) 审计参考
    PHI_MEAN_REF: float = 20.26     # φ 均值（度）审计参考
    PHI_MED_REF: float = 19.75      # φ 中位（度）审计参考
    HMED_REF: float = 1.3152        # |H| 中位数审计参考
    ENERGY_REF: float = 6.1855      # 能量均值 = mean(|κ1|+|κ2|) 审计参考
    K_NEG_SLOT: list = [20.0, 57.0]  # K<0% 告警槽 [20,57]（T03）
    CORR_DECOUPLE_TOL: float = 0.05  # corr(logR,φ) 解耦阈值（README ⑤）


# ---- 第二层：配置工厂 ConfigFactory（实例化 H04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """H04 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 H04Config。"""

    def build(self) -> H04Config:
        """构建 H04Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(H04Config, "H04")


# ---------------- 算法逻辑常量（数学公式/结构常量，非参数阈值） ----------------
DEFF_TARGET = np.pi / 2      # DEFF 平台理论值 = π/2（README ④ 推导 3，数学公式）
EPS = 1e-12                  # 除零保护小量
# 审计对照容差（验证方法论常量，与原脚本一致：相对/绝对分项容差）
METRICS_TOL = [  # (指标名, 容差模式, 容差)；顺序与 EXPECT 对照表一致
    ("rel", 1e-3), ("rel", 1e-3), ("rel", 1e-2), ("rel", 1e-2),
    ("abs", 1e-3), ("abs", 1e-3), ("abs", 0.05), ("abs", 0.05),
    ("rel", 1e-3), ("rel", 1e-3),
]


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def compute_metrics(k1: np.ndarray, k2: np.ndarray) -> dict:
    """由 (κ1,κ2) 复算全部核心指标（README ② 公式，与原脚本一致）。"""
    a1, a2 = np.abs(k1), np.abs(k2)                      # 主曲率绝对值
    big = np.maximum(a1, a2)                             # 较大 |κ|
    small = np.minimum(a1, a2)                           # 较小 |κ|
    phi = np.degrees(np.arctan2(small, big))             # φ = arctan(|κ2|/|κ1|) 按 |κ1|>=|κ2| 约定
    deff = (a1 + a2) ** 2 / (k1 ** 2 + k2 ** 2 + EPS)    # DEFF 平台公式（EPS 防零）
    lam = float(k2.std() / (k1.std() + EPS))             # λ = σ2/σ1
    k_neg = 100.0 * np.mean(k1 * k2 < 0)                 # K<0% 百分比
    same_sign = float(np.mean(k1 * k2 > 0))              # 同号率
    corr_raw = float(np.corrcoef(k1, k2)[0, 1])          # corr(κ1,κ2)
    corr_abs = float(np.corrcoef(a1, a2)[0, 1])          # corr(|κ1|,|κ2|)
    R = np.sqrt(k1 ** 2 + k2 ** 2)                       # 幅度 R
    logR = np.log(R + EPS)                               # 对数幅度（EPS 防 log0）
    return {
        "deff": float(np.mean(deff)),                    # DEFF 平台均值
        "lam": lam,                                      # 主轴比
        "k_neg": float(k_neg),                           # K<0%
        "same_sign": same_sign,                          # 同号率
        "corr_raw": corr_raw,                            # 原始相关系数
        "corr_abs": corr_abs,                            # 绝对值相关系数
        "phi_mean": float(phi.mean()),                   # φ 均值
        "phi_median": float(np.median(phi)),             # φ 中位数
        "hmed": float(np.median(np.abs((k1 + k2) / 2))),  # |H| 中位数（平均曲率）
        "energy": float(np.mean(a1 + a2)),               # 能量均值
        "corr_logR_phi": float(np.corrcoef(logR, phi)[0, 1]),  # 比值-尺度解耦度
    }


# ---- 第三层：合成器 H04DataSynthesizer（真实文件缺失时的回退数据） ----
class H04DataSynthesizer(_SynthBase):
    """H04 合成回退数据生成器：结构近似真实集（负相关 κ1,κ2 + 独立幅度）。"""

    def __init__(self, cfg: H04Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_fallback(self, n_samples: int | None = None) -> np.ndarray:
        """合成回退数据：结构近似真实集（负相关 κ1,κ2 + 独立幅度），仅供结构验证。"""
        cfg = self._cfg
        if n_samples is None:
            n_samples = cfg.N_SAMPLES
        if n_samples <= 0:
            raise SynthesisError(f"非法的样本数: {n_samples}", actual=n_samples)
        rng = np.random.default_rng(self._seed)          # 固定种子随机源
        R = rng.lognormal(mean=1.2, sigma=0.6, size=n_samples)   # 对数正态幅度
        phi = rng.uniform(0.0, 45.0, size=n_samples)     # 各向同性 φ 均匀分布
        phi_r = np.radians(phi)                          # 转弧度用于三角函数
        k1 = R * np.cos(phi_r)                           # κ1 = R·cos φ
        k2 = R * np.sin(phi_r)                           # κ2 = R·sin φ
        sign = rng.choice([-1.0, 1.0], size=n_samples, p=[0.55, 0.45])  # 55% 负号：K<0 结构
        k2 = k2 * sign                                   # 施加符号（κ2 可为负）
        out = np.column_stack([k1, k2]).astype(np.float64)  # 组装 (N,2) float64 数组
        if not np.isfinite(out).all():
            raise SynthesisError("合成回退数据含 NaN/Inf", actual=out.shape)
        return out


# ---- 第四层：验证引擎 H04Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class H04ValidationError(AIQValidationError):
    """H04 数据驱动指标验证失败（复算指标与审计/理论不符）。"""


class H04Validator(_EngineBase):
    """H04 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 真实数据经 _cfg.phi_pairs_path() 相对定位（零硬编码），缺失时合成回退。
    """

    def __init__(
        self,
        config: H04Config,
        synth: H04DataSynthesizer | None,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data          # 惰性注入（None 时 validate_real_model 方法内 import）
        self.data: np.ndarray | None = None  # 当前生效数据（真实或合成）
        self.m: dict | None = None           # 复算核心指标
        self.synthetic_mode: bool = False    # 是否合成回退模式

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 加载与形状
    def validate_shape_dtype(self) -> dict:
        """1) 加载数据（真实 phi_pairs_all.npy 或合成回退），断言形状与 dtype。"""
        cfg = self.config
        rd = self._get_real_data()
        data = rd.phi_pairs()                # 经 _cfg 相对定位加载真实数据（缺失返回 None）
        if data is None:                     # 真实数据缺失 -> 合成回退（仅结构验证）
            self.synthetic_mode = True
            assert self.synth is not None, "合成器缺失但真实数据不可用"
            data = self.synth.synth_fallback()
            src_desc = "合成回退数据"
        else:
            self.synthetic_mode = False
            src_desc = CFG.phi_pairs_path() if hasattr(CFG, "phi_pairs_path") else "phi_pairs_all.npy"
        self.data = np.asarray(data, dtype=float)
        ok = bool(self.data.shape == (cfg.N_SAMPLES, 2) and self.data.dtype == np.float64)
        if not ok:
            raise H04ValidationError(
                f"[1] 形状/dtype 不符：{self.data.shape}, {self.data.dtype}",
                expected=(cfg.N_SAMPLES, 2, "float64"),
                actual=(self.data.shape, str(self.data.dtype)),
                param_key="H04",
            )
        # 复算核心指标（真实/合成一致执行，供 [2]-[4] 复用）
        self.m = compute_metrics(self.data[:, 0], self.data[:, 1])
        return {
            "detail": (f"[1] 数据来源: {src_desc}; 形状 {self.data.shape} + "
                       f"{self.data.dtype}: PASS" + ("（合成回退模式）" if self.synthetic_mode else "")),
            "shape": list(self.data.shape), "dtype": str(self.data.dtype),
            "synthetic_mode": self.synthetic_mode,
        }

    # ------------------------------------------------------------ 2) 指标复算 vs 审计
    def validate_metrics_audit(self) -> dict:
        """2) 核心指标复算 vs 审计报告实测值（仅真实数据模式做对照）。"""
        cfg = self.config
        if self.synthetic_mode:              # 合成模式不做报告值对照（原脚本语义）
            return {"detail": "[2] 合成模式：不做审计报告值对照（结构验证见 [3][4]）",
                    "synthetic_mode": True, "checked": 0}
        assert self.m is not None
        refs = [  # 与 _params_data.json H04 节点键名一一对应
            ("DEFF 平台", self.m["deff"], cfg.DEFF_REF),
            ("λ = σ2/σ1", self.m["lam"], cfg.LAMBDA_REF),
            ("K<0% (%)", self.m["k_neg"], cfg.K_NEG_REF),
            ("同号率", self.m["same_sign"], cfg.SAME_SIGN_REF),
            ("corr(κ1,κ2)", self.m["corr_raw"], cfg.CORR_RAW_REF),
            ("corr(|κ1|,|κ2|)", self.m["corr_abs"], cfg.CORR_ABS_REF),
            ("φ均值 (°)", self.m["phi_mean"], cfg.PHI_MEAN_REF),
            ("φ中位 (°)", self.m["phi_median"], cfg.PHI_MED_REF),
            ("Hmed |H|", self.m["hmed"], cfg.HMED_REF),
            ("能量均值", self.m["energy"], cfg.ENERGY_REF),
        ]
        details, failed = [], []
        for (name, val, exp), (mode, tol) in zip(refs, METRICS_TOL):
            if mode == "rel":                # 相对容差模式
                ok = bool(np.isclose(val, exp, rtol=tol, atol=1e-9))
            else:                            # 绝对容差模式
                ok = bool(np.isclose(val, exp, rtol=0.0, atol=tol))
            details.append(f"{name}={val:.6f}(报告 {exp:.6f})" + ("PASS" if ok else "FAIL"))
            if not ok:
                failed.append((name, val, exp, tol))
        if failed:
            raise H04ValidationError(
                f"[2] 指标复算与审计报告超容差: {failed[0][0]} 实测={failed[0][1]:.6f} "
                f"报告={failed[0][2]:.6f}",
                expected={n: e for n, _, e, _ in failed},
                actual={n: v for n, v, _, _ in failed},
                param_key="H04",
            )
        return {"detail": "[2] " + "; ".join(details), "n_checked": len(refs)}

    # ------------------------------------------------------------ 3) 平台与槽
    def validate_platform_slot(self) -> dict:
        """3) DEFF 平台 ≈ π/2（5% 内）且 K<0% 落入告警槽 [20,57]。"""
        cfg = self.config
        assert self.m is not None
        dev_deff = abs(self.m["deff"] - DEFF_TARGET) / DEFF_TARGET * 100.0  # 相对 π/2 偏差
        ok3a = bool(np.isclose(self.m["deff"], DEFF_TARGET, rtol=0.05))     # 平台须在 5% 内
        lo, hi = float(cfg.K_NEG_SLOT[0]), float(cfg.K_NEG_SLOT[1])
        ok3b = bool(lo <= self.m["k_neg"] <= hi)                            # K<0% 须落入告警槽
        if not (ok3a and ok3b):
            raise H04ValidationError(
                f"[3] 平台/槽不符: DEFF={self.m['deff']:.4f}(偏差 {dev_deff:.2f}%), "
                f"K<0%={self.m['k_neg']:.2f}%",
                expected={"deff_tol_5pct": True, "k_slot": [lo, hi]},
                actual={"deff": self.m["deff"], "k_neg": self.m["k_neg"]},
                param_key="H04",
            )
        return {
            "detail": (f"[3] DEFF 平台 vs π/2={DEFF_TARGET:.4f}: 偏差 {dev_deff:.2f}% PASS; "
                       f"K<0%={self.m['k_neg']:.2f}% ∈ 槽[{lo},{hi}] PASS"),
            "deff_dev_pct": dev_deff, "k_neg": self.m["k_neg"], "k_slot": [lo, hi],
        }

    # ------------------------------------------------------------ 4) 比值-尺度解耦
    def validate_decouple(self) -> dict:
        """4) 比值-尺度解耦：corr(logR, φ) ≈ 0（幅度与形状近似独立）。"""
        cfg = self.config
        assert self.m is not None
        corr_lr = self.m["corr_logR_phi"]
        ok = abs(corr_lr) < cfg.CORR_DECOUPLE_TOL
        if not ok:
            raise H04ValidationError(
                f"[4] corr(logR, φ)={corr_lr:+.4f} 未解耦（|·|≥{cfg.CORR_DECOUPLE_TOL}）",
                expected=cfg.CORR_DECOUPLE_TOL, actual=abs(corr_lr), param_key="H04",
            )
        return {"detail": (f"[4] 比值-尺度解耦 corr(logR, φ) = {corr_lr:+.4f} "
                           f"(审计报告 +0.0056, |·|<{cfg.CORR_DECOUPLE_TOL})"),
                "corr_logR_phi": corr_lr, "tol": cfg.CORR_DECOUPLE_TOL}

    # ------------------------------------------------------------ 5) φ 直方图（附注）
    def validate_phi_histogram(self) -> dict:
        """5) φ 分布摘要 + 直方图（附注，不参与判定；原脚本 [5] 语义）。"""
        assert self.data is not None
        k1, k2 = self.data[:, 0], self.data[:, 1]
        phi_full = np.degrees(np.arctan2(     # 全体点 φ（按 |κ1|>=|κ2| 约定）
            np.minimum(np.abs(k1), np.abs(k2)), np.maximum(np.abs(k1), np.abs(k2))))
        mu = float(phi_full.mean())
        sigma = float(phi_full.std())
        hist, _ = np.histogram(phi_full, bins=np.arange(0, 46.5, 0.5))  # 0~46° 每 0.5° 直方图
        return {"detail": (f"[5] φ 分布摘要: mean={mu:.2f}°, std={sigma:.2f}° "
                           f"(各向同性→22.5°); 直方图 46 格, 峰值格频数={int(hist.max())}"),
                "phi_mean": mu, "phi_std": sigma, "hist_peak": int(hist.max())}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：本地复算与 _real_metrics.json 曲率实测逐项对照。"""
        rd = self._get_real_data()
        tag = rd.source_tag()
        if self.synthetic_mode or rd.get("curvature.DEFF_plat", None) is None:
            return {"detail": (f"[6] [{tag}] 真实曲率测量不可用或当前为合成回退模式，"
                               f"跳过对照"), "skipped": True}
        assert self.m is not None
        pairs = [  # (指标名, 本地复算值, 真实 JSON 字段, 相对容差)
            ("DEFF 平台", self.m["deff"], rd.get("curvature.DEFF_plat"), 1e-3),
            ("λ = σ2/σ1", self.m["lam"], rd.get("curvature.lambda_ratio"), 1e-3),
            ("K<0% (%)", self.m["k_neg"], rd.get("curvature.K_neg_pct"), 1e-2),
            ("同号率", self.m["same_sign"], rd.get("curvature.same_sign_ratio"), 1e-2),
            ("corr(κ1,κ2)", self.m["corr_raw"], rd.get("curvature.corr_k1k2"), 1e-3),
            ("corr(|κ1|,|κ2|)", self.m["corr_abs"], rd.get("curvature.corr_abs"), 1e-3),
            ("φ均值 (°)", self.m["phi_mean"], rd.get("curvature.phi_mean_deg"), 1e-3),
            ("φ中位 (°)", self.m["phi_median"], rd.get("curvature.phi_med_deg"), 1e-3),
            ("Hmed |H|", self.m["hmed"], rd.get("curvature.H_median"), 1e-3),
            ("能量均值", self.m["energy"], rd.get("curvature.energy_mean"), 1e-3),
        ]
        details, failed = [], []
        for name, val, real, tol in pairs:
            if real is None:                 # 真实 JSON 缺字段：仅展示不判失败
                details.append(f"{name}={val:.6f} (真实 JSON 缺该字段)")
                continue
            ok = bool(np.isclose(val, real, rtol=tol, atol=1e-6))  # 同源数据应逐项一致
            details.append(f"{name}={val:.6f} vs 真实={real:.6f}" + ("PASS" if ok else "FAIL"))
            if not ok:
                failed.append((name, val, real))
        # 真实解耦度（口径不同，仅展示不判 PASS）
        corr_lr_real = rd.get("curvature.corr_logR_phi")
        note = ""
        if corr_lr_real is not None:
            note = (f"; corr(logR,φ) 口径差异: 本地(log R)={self.m['corr_logR_phi']:+.4f} "
                    f"vs 真实(log|κ1|+|κ2|)={corr_lr_real:+.4f}（仅展示）")
        if failed:
            raise RealModelMismatchError(
                f"[6] 本地复算与真实测量不一致: {failed[0][0]} "
                f"本地={failed[0][1]:.6f} vs 真实={failed[0][2]:.6f}",
                expected={n: r for n, _, r in failed},
                actual={n: v for n, v, _ in failed},
                param_key="H04",
            )
        return {"detail": f"[6] [{tag}] " + "; ".join(details) + note,
                "source": tag, "n_checked": len(pairs)}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "shape_dtype", self.validate_shape_dtype),
            (2, "metrics_audit", self.validate_metrics_audit),
            (3, "platform_slot", self.validate_platform_slot),
            (4, "decouple", self.validate_decouple),
            (5, "phi_histogram", self.validate_phi_histogram),
            (6, "real_model", self.validate_real_model),
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
    """H04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="H04 data_file 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = H04DataSynthesizer(cfg)                   # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = H04Validator(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("H04 data_file 逐点(κ1,κ2)数据集验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_SAMPLES={cfg.N_SAMPLES} K_NEG_SLOT={cfg.K_NEG_SLOT} "
          f"CORR_DECOUPLE_TOL={cfg.CORR_DECOUPLE_TOL}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "h04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "h04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "h04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
