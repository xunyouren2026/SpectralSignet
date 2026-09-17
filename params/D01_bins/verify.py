# -*- coding: utf-8 -*-
"""D01 bins — φ 分布直方图分格：定义域与分格逻辑验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. φ = arctan(|κ₂|/|κ₁|) 定义域 ∈ [0°,45°]，|κ₂|≤|κ₁| 排序正确
  2. 46 格直方图 linspace(0,45,47)：格数=46，格宽一致
  3. 0.5° 格直方图 arange(0,45.5,0.5)：格数=90（0-45°@0.5°严格为 90 格）
     + 文档示例 arange(0,46.5,0.5) 实际为 92 格（审计：文档-源码口径不一致）
  4. 合成截断高斯 N(14°,25°)|[0,45] 直方图峰值格 ≈14°（实测 μ=14.1°）
  5. 合成 φ 均值 ≈ 实测 φmean=20.26°，直方图形状单峰中段
  6. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 (κ1,κ2) 数据 6303 点；
     corr(logR,φ)=0.1206 vs 审计报告 ≈0 的差异如实标注）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D01Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D01Config（环境变量 AIQ_D01_<KEY>
                            > YAML > _params_data.json > 模型默认值）
  PhiSynthesizer         —— φ 分布合成：synth_phi_trunc_gauss（拒绝采样）/
                            synth_kappa_pairs（各向异性高斯对）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 1935-2060（D01 六步流程）
  源码 _phi_fit_chi2.py 行 82-94（bins=np.linspace(0,45,46) 实际口径）
  《参数审计与实验报告.txt》行 11、17、52（状态=已用）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from math import erf, sqrt
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

# ---- 第一层：配置模型 D01Config（pydantic 优先；dataclass 回退） ----
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
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。

        子类 D01Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D01Config(_ConfigModelBase):
    """D01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 D01 节点键名一一对应；取值优先级：
    环境变量 AIQ_D01_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    BINS: int = 46                          # 46 格直方图格数（README ①）
    PHI_MAX_DEG: float = 45.0               # φ 定义域上界（README ①）
    MU_MEAS: float = 14.0                   # 实测 μ（M3 截断高斯，审计报告行 17）
    SIG_MEAS: float = 25.0                  # 实测 σ（审计报告行 17）
    PHI_MEAN_MEAS: float = 20.26            # 实测 φ 均值（审计报告行 11）
    N: int = 200000                         # 合成样本数（拒绝采样目标）
    LAM_CURV: float = 0.85                  # 曲率幅度尺度比 λ=σ₂/σ₁（≈实测 0.8516）
    PEAK_TOL: float = 1.5                   # 合成峰值格容差（对照实测 μ=14.1°）
    MEAN_TOL: float = 1.0                   # 合成 φ 均值容差（对照实测 20.26°）
    SHAPE_CENTER: float = 20.0              # 直方图形状参考中心（中段）
    SHAPE_TOL: float = 15.0                 # 直方图形状中段容差
    PEAK_MU_TOL: float = 6.0                # 真实直方图峰值 vs 文档 μ=14° 容差
    CORR_TOL: float = 0.02                  # corr(logR,φ) 一致性容差
    CORR_MIN_ABS: float = 0.05              # corr 非零下界（弱相关存在性）
    FALLBACK_MEAN_TOL: float = 2.0          # 审计回退 φ 均值容差
    SEED: int = 0                           # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 D01Config。"""

    def build(self) -> D01Config:
        """构建 D01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(D01Config, "D01")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def phi_from_pairs(k1: np.ndarray, k2: np.ndarray) -> tuple:
    """D01 定义：|κ₁|≥|κ₂| 排序 -> φ=arctan(min/max)∈[0,45°]（README ①）。

    返回 (phi_deg, n_valid)；边界防御：无有效样本时报错。
    """
    a1, a2 = np.abs(k1), np.abs(k2)
    big = np.maximum(a1, a2)
    small = np.minimum(a1, a2)
    valid = big > 1e-12
    n_valid = int(valid.sum())
    assert n_valid > 0, f"phi_from_pairs: 无有效样本（|κ|≤1e-12 占满），n_valid={n_valid}"
    phi = np.degrees(np.arctan(small[valid] / big[valid]))
    return phi, n_valid


def load_phi_real(rd: Any) -> np.ndarray | None:
    """真实 (κ1,κ2) -> φ(度) 数组；真实数据缺失返回 None。

    参数：
      rd: 共享真实数据模块（_real_data，经 setup_env 注入或惰性导入）
    """
    pairs = rd.phi_pairs()
    if pairs is None:
        return None
    k1, k2 = pairs[:, 0], pairs[:, 1]
    a1, a2 = np.abs(k1), np.abs(k2)
    big = np.maximum(a1, a2)
    small = np.minimum(a1, a2)
    valid = big > 1e-12
    if not valid.any():
        return None
    return np.degrees(np.arctan(small[valid] / big[valid]))


# ---- 第三层：φ 合成器 PhiSynthesizer（算法与原脚本完全一致） ----
class PhiSynthesizer(_SynthBase):
    """D01 φ 分布合成器：截断高斯拒绝采样 + 各向异性高斯 (κ1,κ2) 对。

    - synth_phi_trunc_gauss：N(μ,σ)|[lo,hi] 拒绝采样（与源码 _phi_fit_chi2 一致）；
    - synth_kappa_pairs：κ₁~N(0,1)，κ₂~N(0,λ²)，λ 为曲率幅度尺度比。
    """

    def __init__(self, cfg: D01Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_phi_trunc_gauss(
        self, mu: float, sigma: float, lo: float, hi: float,
        n: int, seed: int | None = None,
    ) -> np.ndarray:
        """拒绝采样：截断高斯 N(μ,σ)|[lo,hi]（与源码 _phi_fit_chi2 一致）。

        边界防御：n>0、拒绝采样卡死（迭代上限）保护。
        """
        assert n > 0, f"synth_phi_trunc_gauss: 样本数 n={n} 必须>0"
        rng = np.random.default_rng(self._seed if seed is None else seed)
        out = []
        total = 0
        guard = 0
        while total < n:
            guard += 1
            assert guard < 1000, "synth_phi_trunc_gauss: 拒绝采样未收敛（接受率过低）"
            x = rng.normal(mu, sigma, n - total)
            x = x[(x >= lo) & (x <= hi)]
            out.append(x)
            total += len(x)
        return np.concatenate(out)[:n]

    def synth_kappa_pairs(self, lam: float, n: int, seed: int = 1) -> tuple:
        """合成 (κ₁,κ₂)：κ₁~N(0,1)，κ₂~N(0,λ²)。λ=σ₂/σ₁ 为曲率幅度尺度比。"""
        assert n > 0, f"synth_kappa_pairs: 样本数 n={n} 必须>0"
        rng = np.random.default_rng(seed)
        k1 = rng.normal(0.0, 1.0, n)
        k2 = rng.normal(0.0, lam, n)
        return k1, k2


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D01 验证引擎：顺序执行 6 项验证（5 合成 + 1 真实模型对照）。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（方法内 import 或经 _common.setup_env 注入）。
    """

    def __init__(
        self,
        config: D01Config,
        synth: PhiSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 定义域
    def validate_domain(self) -> dict:
        """1) φ=arctan(|κ₂|/|κ₁|) 定义域 [0°,45°]（README ④第 1 步）。"""
        cfg = self.config
        k1, k2 = self.synth.synth_kappa_pairs(lam=cfg.LAM_CURV, n=cfg.N)
        phi_pairs, n_valid = phi_from_pairs(k1, k2)
        ok = bool(float(phi_pairs.min() >= 0.0) and float(phi_pairs.max() <= 45.0))
        if not ok:
            raise AIQValidationError(
                f"φ 定义域越界: [{phi_pairs.min():.4f}, {phi_pairs.max():.4f}]",
                expected=[0.0, 45.0],
                actual=[float(phi_pairs.min()), float(phi_pairs.max())],
                param_key="D01",
            )
        return {
            "detail": f"n={n_valid}, φ_min={phi_pairs.min():.4f}°, "
                      f"φ_max={phi_pairs.max():.4f}°",
            "n_valid": n_valid,
        }

    # ------------------------------------------------------------ 2) 46 格
    def validate_bins46(self) -> dict:
        """2) 46 格直方图 linspace(0,45,47)（README ②，主文档行 1992 口径）。"""
        cfg = self.config
        k1, k2 = self.synth.synth_kappa_pairs(lam=cfg.LAM_CURV, n=cfg.N)
        phi_pairs, _ = phi_from_pairs(k1, k2)
        bins46 = np.linspace(0, cfg.PHI_MAX_DEG, cfg.BINS + 1)
        c46, e46 = np.histogram(phi_pairs, bins=bins46)
        width = np.diff(e46)
        ok = (len(c46) == cfg.BINS) and bool(np.allclose(width, width[0]))
        if not ok:
            raise AIQValidationError(
                f"46 格直方图分格不符: 格数={len(c46)}",
                expected=cfg.BINS, actual=len(c46), param_key="D01",
            )
        return {
            "detail": f"格数={len(c46)}, 格宽一致, 格宽={width[0]:.4f}°",
            "n_bins": len(c46), "width": float(width[0]),
        }

    # ------------------------------------------------------------ 3) 0.5° 格审计
    def validate_half_deg_audit(self) -> dict:
        """3) 0.5° 格审计（README ⑤审计发现：文档-源码口径不一致）。"""
        cfg = self.config
        k1, k2 = self.synth.synth_kappa_pairs(lam=cfg.LAM_CURV, n=cfg.N)
        phi_pairs, _ = phi_from_pairs(k1, k2)
        bins_half = np.arange(0, 45.5, 0.5)     # 严格 0-45°@0.5°
        c_half, _ = np.histogram(phi_pairs, bins=bins_half)
        bins_doc = np.arange(0, 46.5, 0.5)      # 主文档示例代码
        c_doc, _ = np.histogram(phi_pairs, bins=bins_doc)
        ok = (len(c_half) == 90) and (len(c_doc) == 92)
        if not ok:
            raise AIQValidationError(
                f"0.5° 格审计不符: arange(0,45.5)={len(c_half)}格, "
                f"arange(0,46.5)={len(c_doc)}格",
                expected={"half": 90, "doc": 92},
                actual={"half": len(c_half), "doc": len(c_doc)},
                param_key="D01",
            )
        return {
            "detail": f"arange(0,45.5,0.5)={len(c_half)}格, "
                      f"arange(0,46.5,0.5)={len(c_doc)}格",
            "half_bins": len(c_half), "doc_bins": len(c_doc),
        }

    # ------------------------------------------------------------ 4) 峰值格
    def validate_trunc_gauss_peak(self) -> dict:
        """4) 合成截断高斯 -> 峰值格（实测 μ=14.1°，README ⑤表）。"""
        cfg = self.config
        phi_s = self.synth.synth_phi_trunc_gauss(
            cfg.MU_MEAS, cfg.SIG_MEAS, 0.0, 45.0, cfg.N)
        bins46 = np.linspace(0, cfg.PHI_MAX_DEG, cfg.BINS + 1)
        c_s, e_s = np.histogram(phi_s, bins=bins46)
        centers = e_s[:-1] + 0.5 * np.diff(e_s)
        peak_bin = int(np.argmax(c_s))
        peak_deg = centers[peak_bin]
        mean_phi = float(phi_s.mean())
        ok = (abs(peak_deg - cfg.MU_MEAS) <= cfg.PEAK_TOL
              and abs(mean_phi - cfg.PHI_MEAN_MEAS) <= cfg.MEAN_TOL)
        if not ok:
            raise AIQValidationError(
                f"合成峰值格/均值偏离: 峰值格={peak_deg:.2f}°, 均值={mean_phi:.2f}°",
                expected={"peak tol": cfg.PEAK_TOL, "mean tol": cfg.MEAN_TOL},
                actual={"peak_deg": float(peak_deg), "mean_phi": mean_phi},
                param_key="D01",
            )
        return {
            "detail": f"峰值格={peak_bin}(格心{peak_deg:.2f}°), "
                      f"φ均值={mean_phi:.2f}° (实测 μ=14.1°, φmean=20.26°)",
            "peak_bin": peak_bin, "peak_deg": float(peak_deg), "mean_phi": mean_phi,
        }

    # ------------------------------------------------------------ 5) 形状
    def validate_shape(self) -> dict:
        """5) 直方图形状：单峰、峰值在中段（README ③几何直觉）。"""
        cfg = self.config
        phi_s = self.synth.synth_phi_trunc_gauss(
            cfg.MU_MEAS, cfg.SIG_MEAS, 0.0, 45.0, cfg.N)
        bins46 = np.linspace(0, cfg.PHI_MAX_DEG, cfg.BINS + 1)
        c_s, e_s = np.histogram(phi_s, bins=bins46)
        centers = e_s[:-1] + 0.5 * np.diff(e_s)
        argmax = int(np.argmax(c_s))
        mid_bin = cfg.BINS // 2
        ok = abs(centers[argmax] - cfg.SHAPE_CENTER) < cfg.SHAPE_TOL
        if not ok:
            raise AIQValidationError(
                f"直方图形状非单峰中段: 峰值在 {centers[argmax]:.1f}°",
                expected=cfg.SHAPE_CENTER, actual=float(centers[argmax]),
                param_key="D01",
            )
        return {
            "detail": f"峰值在 {centers[argmax]:.1f}° (中段 {centers[mid_bin]:.1f}° 附近), "
                      f"总计数={int(c_s.sum())}",
            "peak_deg": float(centers[argmax]), "total": int(c_s.sum()),
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real(self) -> dict:
        """6) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 直方图）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins46 = np.linspace(0, cfg.PHI_MAX_DEG, cfg.BINS + 1)
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成高斯对
            rng = np.random.default_rng(cfg.SEED)
            k1f = rng.normal(0.0, 1.0, cfg.N)
            k2f = rng.normal(0.0, cfg.LAM_CURV, cfg.N)
            phif, _ = phi_from_pairs(k1f, k2f)
            ok = float(phif.mean() - cfg.PHI_MEAN_MEAS) < cfg.FALLBACK_MEAN_TOL
            if not ok:
                raise AIQValidationError(
                    f"回退合成 φ 均值偏离过大: {phif.mean():.2f}°",
                    expected=cfg.PHI_MEAN_MEAS, actual=float(phif.mean()),
                    param_key="D01",
                )
            return {"detail": f"φ均值={phif.mean():.2f}°（真实数据未就绪）",
                    "tag": "[审计回退]", "mean_phi": float(phif.mean()),
                    "fallback": True}
        tag = "[真实实测]"
        cr, er = np.histogram(phi_real, bins=bins46)
        ctr = er[:-1] + 0.5 * np.diff(er)
        mean_r = float(phi_real.mean())
        peak_idx_r = int(np.argmax(cr))
        peak_deg_r = ctr[peak_idx_r]
        phi_mean_meas = rd.get("curvature.phi_mean_deg", cfg.PHI_MEAN_MEAS)
        # 均值一致性容差 0.05 为同一数据源的浮点恒等校验（EPS 量级）
        okr1 = (len(cr) == cfg.BINS) and abs(mean_r - phi_mean_meas) < 0.05
        if not okr1:
            raise RealModelMismatchError(
                f"真实 φ 直方图分格/均值不符: 格数={len(cr)}, 均值={mean_r:.2f}°",
                expected={"bins": cfg.BINS, "mean": phi_mean_meas},
                actual={"bins": len(cr), "mean": mean_r}, param_key="D01",
            )
        okr1b = abs(peak_deg_r - cfg.MU_MEAS) <= cfg.PEAK_MU_TOL
        if not okr1b:
            raise RealModelMismatchError(
                f"真实直方图峰值偏离文档 μ=14° 过大: 峰值={peak_deg_r:.1f}°",
                expected=cfg.MU_MEAS, actual=float(peak_deg_r), param_key="D01",
            )
        # corr(logR, φ)：真实 0.1206 vs 审计报告 ≈0（如实标注差异）
        pairs_r = rd.phi_pairs()
        logr_r = np.log(np.abs(pairs_r[:, 0]) + np.abs(pairs_r[:, 1]) + 1e-16)
        corr_r = float(np.corrcoef(logr_r, phi_real)[0, 1])
        stored_corr = rd.get("curvature.corr_logR_phi", corr_r)
        okr2 = (abs(corr_r - stored_corr) < cfg.CORR_TOL) and (abs(corr_r) > cfg.CORR_MIN_ABS)
        if not okr2:
            raise RealModelMismatchError(
                f"corr(logR,φ) 校验失败: corr={corr_r:.4f}",
                expected={"tol": cfg.CORR_TOL, "min_abs": cfg.CORR_MIN_ABS},
                actual=corr_r, param_key="D01",
            )
        return {
            "detail": (f"{tag} 真实 φ 直方图（{len(phi_real)} 点，{cfg.BINS} 格）: "
                       f"峰值格={peak_idx_r}({peak_deg_r:.1f}°), 均值={mean_r:.2f}° = 实测 "
                       f"{phi_mean_meas:.2f}°; corr(logR,φ)={corr_r:.4f} vs 审计报告≈0"
                       f"（真实数据上比值-尺度弱相关，非完全解耦）"),
            "source": rd.source_tag(), "tag": tag,
            "n": int(len(phi_real)), "mean_deg": mean_r,
            "peak_deg": float(peak_deg_r), "corr_logR_phi": corr_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "domain", self.validate_domain),
            (2, "bins46", self.validate_bins46),
            (3, "half_deg_audit", self.validate_half_deg_audit),
            (4, "trunc_gauss_peak", self.validate_trunc_gauss_peak),
            (5, "shape", self.validate_shape),
            (6, "real_model", self.validate_real),
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
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """D01 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D01 bins 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = PhiSynthesizer(cfg)                # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D01 bins 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: BINS={cfg.BINS} PHI_MAX_DEG={cfg.PHI_MAX_DEG} MU_MEAS={cfg.MU_MEAS} "
          f"SIG_MEAS={cfg.SIG_MEAS} PHI_MEAN_MEAS={cfg.PHI_MEAN_MEAS} N={cfg.N} "
          f"LAM_CURV={cfg.LAM_CURV} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
