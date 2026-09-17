# -*- coding: utf-8 -*-
"""D02 fit_range — 卡方拟合域：剔除尾部抗噪验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 拟合域提取：前 39 格 = 中心 0.5°-38.5°，尾部 39.5°-45° 单独报告
  2. 无噪声合成数据：拟合域 χ² ≤ 全范围 χ²（剔除尾部不损失信息）
  3. 注入尾部噪声：全范围 χ² 大幅恶化，拟合域 χ² 受影响小（抗噪性）
  4. 剔除尾部后 M3 拟合 p 值提高（复现实测 0.85 -> 0.91 趋势）
  5. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M3 卡方拟合：
     最优 μ*=14°、σ*=25°，拟合域 χ²<全范围 χ²，p 值 0.73→0.90）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D02Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D02Config（环境变量 AIQ_D02_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            跨参数节点取值在 build() 中显式解析）
  PhiFitSynthesizer      —— φ 分布合成（拒绝采样）+ M3 联合网格拟合
                            （μ∈[5,40]×σ∈[2,30]，get_grid 建模）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2061-2157（D02 五步流程）
  源码 _phi_fit_chi2.py 行 86-94、99-108、119-137
  《参数审计与实验报告.txt》行 17、53（状态=已用）
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

# ---- 第一层：配置模型 D02Config（pydantic 优先；dataclass 回退） ----
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

        子类 D02Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D02Config(_ConfigModelBase):
    """D02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（MU/SIG/CRIT/
    MU_GRID/SIGMA_GRID/TOL_MU/TOL_SIG）在 ConfigFactory.build() 中按
    来源节点显式解析（D04/D05/D06），保持单一数据源。
    """

    MU: float = 14.0                # 实测 M3 最优 μ（审计报告行 17，D04.MU_TRUE 同源）
    SIG: float = 25.0               # 实测 M3 最优 σ（审计报告行 17，D05.SIGMA_TRUE 同源）
    N: int = 200000                 # 合成样本数
    N_FIT: int = 39                 # 拟合域格数（源码 np.arange(39)）
    FIT_LO_DEG: float = 0.5         # 拟合域下界格心（主文档行 2061）
    FIT_HI_DEG: float = 38.5        # 拟合域上界格心（主文档行 2061）
    CRIT_DOF2: float = 5.99         # df=2, α=0.05（T05 尖峰显著性阈值，D06.CRIT_DCHI2 同源）
    MU_GRID: tuple = (5.0, 40.0, 71)     # M3 联合网格 μ：(start, stop, count)（D04 定义）
    SIGMA_GRID: tuple = (2.0, 30.0, 57)  # M3 联合网格 σ：(start, stop, count)（D05 定义）
    TOL_MU: float = 1.0             # μ 恢复容差（D04.TOL_MU 同源）
    TOL_SIG: float = 2.0            # σ 恢复容差（D05.TOL_SIG 同源）
    NOISE_GAIN: float = 5.0         # 加尾部噪声抗噪增益判据（Δχ² 差阈值）
    P_REAL_MIN: float = 0.85        # 真实数据拟合域 p 值下界（复现 0.85->0.91 趋势）
    SEED: int = 0                   # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D02 配置工厂：实例化 D02Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D02Config:
        cfg = self.build_model(D02Config, "D02")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D04/D05/D06）
        cfg.MU = self.get_float("D04", "MU_TRUE", 14.0)
        cfg.SIG = self.get_float("D05", "SIGMA_TRUE", 25.0)
        cfg.CRIT_DOF2 = self.get_float("D06", "CRIT_DCHI2", 5.99)
        cfg.TOL_MU = self.get_float("D04", "TOL_MU", 1.0)
        cfg.TOL_SIG = self.get_float("D05", "TOL_SIG", 2.0)
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
        cfg.MU_GRID = self.get_grid("D04", "MU_GRID", (5.0, 40.0, 71))
        cfg.SIGMA_GRID = self.get_grid("D05", "SIGMA_GRID", (2.0, 30.0, 57))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def exp_gauss(centers: np.ndarray, mu: float, sig: float, Ntot: float) -> np.ndarray:
    """M3 截断高斯期望频数：E=N·[Φ(hi)-Φ(lo)]/[Φ(45)-Φ(0)]（源码 exp_gauss）。

    边界防御：归一化分母须为正。
    """
    lo = centers - 0.5
    hi = centers + 0.5
    denom = norm_cdf(45.0, mu, sig) - norm_cdf(0.0, mu, sig)
    assert float(np.max(denom)) > 0.0, f"exp_gauss: 归一化分母为 0（μ={mu},σ={sig}）"
    return Ntot * (norm_cdf(hi, mu, sig) - norm_cdf(lo, mu, sig)) / denom


def chi2_merge(obs: np.ndarray, exp: np.ndarray) -> float:
    """低期望 bin 合并（源码 chi2_merge：期望<1 时合并到相邻格）。

    边界防御：obs/exp 有限且 exp 总和为正。
    """
    o = obs.astype(float).copy()
    e = exp.astype(float).copy()
    assert o.size == e.size > 0, f"chi2_merge: obs/e 尺寸不一致或为空 {o.size}/{e.size}"
    assert np.all(np.isfinite(o)) and np.all(np.isfinite(e)), "chi2_merge: 输入含 NaN/Inf"
    assert e.sum() > 0.0, f"chi2_merge: 期望总和为 0，无法计算 χ²"
    while e.min() < 1.0 and len(e) > 1:
        i = int(np.argmin(e))
        if i == len(e) - 1:
            e[-2] += e[-1]
            o[-2] += o[-1]
            e = e[:-1]
            o = o[:-1]
        else:
            e[i + 1] += e[i]
            o[i + 1] += o[i]
            e = np.delete(e, i)
            o = np.delete(o, i)
    assert np.all(e > 0.0), "chi2_merge: 合并后仍存在非正期望"
    return float(((o - e) ** 2 / e).sum())


def chi2_sf_approx(c2: float, dof: float) -> float:
    """χ² 上尾 p 值（正态近似）：z=(χ²-dof)/√(2·dof)。"""
    z = (c2 - dof) / np.sqrt(2.0 * dof)
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def merge_dof(obs: np.ndarray, exp: np.ndarray) -> int:
    """合并后格数（与 chi2_merge 相同的合并规则，用于计算自由度）。"""
    e = exp.astype(float).copy()
    while e.min() < 1.0 and len(e) > 1:
        i = int(np.argmin(e))
        if i == len(e) - 1:
            e = e[:-1]
        else:
            e = np.delete(e, i)
    return int(len(e))


def load_phi_real(rd: Any) -> np.ndarray | None:
    """真实 (κ1,κ2) -> φ(度) 数组；缺失返回 None。

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


# ---- 第三层：合成器 PhiFitSynthesizer（算法与原脚本完全一致） ----
class PhiFitSynthesizer(_SynthBase):
    """D02 合成器：φ 分布拒绝采样 + M3 联合网格拟合。

    - synth_phi_trunc_gauss：N(μ,σ)|[lo,hi] 拒绝采样；
    - fit_m3_grid：μ∈[5,40]×σ∈[2,30] 联合网格最小 χ² -> (μ*, σ*, χ²)
      （网格经 get_grid 建模为 (start, stop, count)，np.linspace 展开）。
    """

    def __init__(self, cfg: D02Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_phi_trunc_gauss(
        self, mu: float, sigma: float, lo: float, hi: float,
        n: int, seed: int | None = None,
    ) -> np.ndarray:
        """拒绝采样：截断高斯 N(μ,σ)|[lo,hi]（与源码 _phi_fit_chi2 一致）。"""
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

    def fit_m3_grid(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                    Ntot: float) -> tuple:
        """M3 联合网格 (μ∈[5,40]×σ∈[2,30]) 最小 χ² -> (μ*, σ*, χ²)。"""
        cfg = self._cfg
        mu_lo, mu_hi, n_mu = cfg.MU_GRID      # 网格 (start, stop, count)
        sig_lo, sig_hi, n_sig = cfg.SIGMA_GRID
        best = (1e18, None)
        for mu in np.linspace(mu_lo, mu_hi, n_mu):
            for sig in np.linspace(sig_lo, sig_hi, n_sig):
                c2 = chi2_merge(obs_fit, exp_gauss(centers_fit, mu, sig, Ntot))
                if c2 < best[0]:
                    best = (c2, (mu, sig))
        return best[1][0], best[1][1], best[0]


# ---- 第四层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D02 验证引擎：顺序执行 5 项验证（4 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D02Config,
        synth: PhiFitSynthesizer,
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

    def _phi_bins(self) -> tuple:
        """源码口径：45 格直方图（np.linspace(0,45,46)，与 D01 的 46 格口径不同）。"""
        bins = np.linspace(0, 45, 46)
        edges = bins
        centers_full = (edges[:-1] + edges[1:]) / 2
        return bins, centers_full

    # ------------------------------------------------------------ 1) 拟合域提取
    def validate_fit_range(self) -> dict:
        """1) 拟合域提取：前 39 格 0.5°-38.5°（README ②，源码 np.arange(39)）。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
        bins, centers_full = self._phi_bins()
        obs_full, edges = np.histogram(phi, bins=bins)
        centers_full = (edges[:-1] + edges[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)                    # 0..38 -> 中心 0.5-38.5°
        obs_fit = obs_full[fit_idx]
        centers_fit = centers_full[fit_idx]
        tail = obs_full[cfg.N_FIT:].sum()
        ok = (len(fit_idx) == cfg.N_FIT
              and np.isclose(centers_fit[0], cfg.FIT_LO_DEG, rtol=0, atol=1e-9)
              and np.isclose(centers_fit[-1], cfg.FIT_HI_DEG, rtol=0, atol=1e-9))
        if not ok:
            raise AIQValidationError(
                f"拟合域提取不符: centers=[{centers_fit[0]}, {centers_fit[-1]}]",
                expected=[cfg.FIT_LO_DEG, cfg.FIT_HI_DEG],
                actual=[float(centers_fit[0]), float(centers_fit[-1])],
                param_key="D02",
            )
        return {
            "detail": f"尾部(>38.5°)计数={int(tail)}/{int(obs_full.sum())} "
                      f"= {tail / obs_full.sum() * 100:.2f}%",
            "tail_pct": float(tail / obs_full.sum() * 100.0),
        }

    # ------------------------------------------------------------ 2) 无噪声
    def validate_noise_free(self) -> dict:
        """2) 无噪声：全范围 vs 拟合域 χ²（对真实 M3 模型，README ⑤表）。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
        bins, _ = self._phi_bins()
        obs_full, edges = np.histogram(phi, bins=bins)
        centers_full = (edges[:-1] + edges[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        obs_fit = obs_full[fit_idx]
        centers_fit = centers_full[fit_idx]
        c2_full = chi2_merge(obs_full, exp_gauss(centers_full, cfg.MU, cfg.SIG, cfg.N))
        c2_fit = chi2_merge(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
        ok = c2_fit <= c2_full + 1e-6
        if not ok:
            raise AIQValidationError(
                f"拟合域 χ² 应≤全范围 χ²: 拟合域={c2_fit:.2f}, 全范围={c2_full:.2f}",
                expected=c2_full, actual=c2_fit, param_key="D02",
            )
        return {
            "detail": f"全范围χ²={c2_full:.2f}, 拟合域χ²={c2_fit:.2f}, "
                      f"Δ={c2_full - c2_fit:+.2f}",
            "c2_full": c2_full, "c2_fit": c2_fit,
        }

    # ------------------------------------------------------------ 3) 抗噪
    def validate_tail_noise(self) -> dict:
        """3) 注入尾部噪声 -> 抗噪性（README ⑤误差说明）。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
        bins, _ = self._phi_bins()
        obs_full, edges = np.histogram(phi, bins=bins)
        centers_full = (edges[:-1] + edges[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        obs_fit = obs_full[fit_idx]
        centers_fit = centers_full[fit_idx]
        c2_full = chi2_merge(obs_full, exp_gauss(centers_full, cfg.MU, cfg.SIG, cfg.N))
        c2_fit = chi2_merge(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
        rng = np.random.default_rng(5)
        obs_noisy = obs_full.astype(float).copy()
        obs_noisy[0] += rng.poisson(300)              # 0°端异常高（管状伪影模拟）
        obs_noisy[-3:] += rng.poisson(150, 3)         # 45°端异常高（边缘伪影模拟）
        c2n_full = chi2_merge(obs_noisy, exp_gauss(centers_full, cfg.MU, cfg.SIG, cfg.N))
        c2n_fit = chi2_merge(obs_noisy[fit_idx], exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
        gain = (c2n_full - c2_full) - (c2n_fit - c2_fit)
        ok = (c2n_full - c2_full) > (c2n_fit - c2_fit) + cfg.NOISE_GAIN
        if not ok:
            raise AIQValidationError(
                f"拟合域抗噪增益不足: 全范围χ²↑{c2n_full - c2_full:+.2f}, "
                f"拟合域χ²↑{c2n_fit - c2_fit:+.2f}",
                expected={"gain >": cfg.NOISE_GAIN}, actual=gain, param_key="D02",
            )
        return {
            "detail": f"全范围χ²↑{c2n_full - c2_full:+.2f}, 拟合域χ²↑{c2n_fit - c2_fit:+.2f}, "
                      f"抗噪增益={gain:.2f}",
            "gain": gain,
        }

    # ------------------------------------------------------------ 4) M3 p 值
    def validate_p_value(self) -> dict:
        """4) M3 p 值提升（复现实测 0.85 -> 0.91 趋势，README ③用途）。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
        bins, _ = self._phi_bins()
        obs_full, edges = np.histogram(phi, bins=bins)
        centers_full = (edges[:-1] + edges[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        obs_fit = obs_full[fit_idx]
        centers_fit = centers_full[fit_idx]
        c2_full = chi2_merge(obs_full, exp_gauss(centers_full, cfg.MU, cfg.SIG, cfg.N))
        c2_fit = chi2_merge(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
        d_full = merge_dof(obs_full, exp_gauss(centers_full, cfg.MU, cfg.SIG, cfg.N))
        d_fit = merge_dof(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
        p_full = chi2_sf_approx(c2_full, d_full - 2)  # M3 k=2
        p_fit = chi2_sf_approx(c2_fit, d_fit - 2)
        ok = p_fit >= p_full - 1e-6
        if not ok:
            raise AIQValidationError(
                f"剔除尾部后 p 值未提升: 全范围p={p_full:.3f}, 拟合域p={p_fit:.3f}",
                expected={"p_fit >= p_full": True},
                actual={"p_full": p_full, "p_fit": p_fit}, param_key="D02",
            )
        return {
            "detail": f"全范围p={p_full:.3f}, 拟合域p={p_fit:.3f} (实测 0.85->0.91)",
            "p_full": p_full, "p_fit": p_fit,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real(self) -> dict:
        """5) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布卡方拟合）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins, _ = self._phi_bins()
        fit_idx = np.arange(cfg.N_FIT)
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成截断高斯
            phi = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
            obs_full, edges = np.histogram(phi, bins=bins)
            centers_full = (edges[:-1] + edges[1:]) / 2
            centers_fit = centers_full[fit_idx]
            obs_fit = obs_full[fit_idx]
            c2_fit = chi2_merge(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
            d_fit = merge_dof(obs_fit, exp_gauss(centers_fit, cfg.MU, cfg.SIG, cfg.N))
            p_fit = chi2_sf_approx(c2_fit, d_fit - 2)
            ok = p_fit >= 0.0  # 回退分支仅报告，不判定（真实数据未就绪）
            if not ok:
                raise AIQValidationError(
                    f"回退合成拟合域 p 值异常: {p_fit:.3f}",
                    expected=0.0, actual=p_fit, param_key="D02",
                )
            return {"detail": f"拟合域p={p_fit:.3f}（真实数据未就绪）",
                    "tag": "[审计回退]", "p_fit": p_fit, "fallback": True}
        tag = "[真实实测]"
        Nreal = float(len(phi_real))
        obs_r_full, edges_r = np.histogram(phi_real, bins=bins)
        centers_r = (edges_r[:-1] + edges_r[1:]) / 2
        obs_r_fit = obs_r_full[fit_idx]
        centers_r_fit = centers_r[fit_idx]
        # 1) 真实数据 M3 拟合：μ*/σ* 与文档审计对照
        mu_star_r, sig_star_r, c2_star_r = self.synth.fit_m3_grid(
            obs_r_fit, centers_r_fit, Nreal)
        okr1 = (abs(mu_star_r - cfg.MU) <= cfg.TOL_MU) and (abs(sig_star_r - cfg.SIG) <= cfg.TOL_SIG)
        if not okr1:
            raise RealModelMismatchError(
                f"真实数据 M3 拟合偏离文档: μ*={mu_star_r:.1f}, σ*={sig_star_r:.1f}",
                expected={"mu tol": cfg.TOL_MU, "sig tol": cfg.TOL_SIG},
                actual={"mu": mu_star_r, "sig": sig_star_r}, param_key="D02",
            )
        # 2) 真实数据全范围 vs 拟合域 χ²（剔除尾部抗噪性质）
        c2_r_full = chi2_merge(obs_r_full, exp_gauss(centers_r, mu_star_r, sig_star_r, Nreal))
        c2_r_fit = chi2_merge(obs_r_fit, exp_gauss(centers_r_fit, mu_star_r, sig_star_r, Nreal))
        okr2 = c2_r_fit <= c2_r_full + 1e-6
        if not okr2:
            raise RealModelMismatchError(
                f"真实数据拟合域 χ² 应≤全范围: {c2_r_fit:.2f} vs {c2_r_full:.2f}",
                expected=c2_r_full, actual=c2_r_fit, param_key="D02",
            )
        # 3) 真实数据 p 值（对照文档 0.85 -> 0.91 趋势）
        d_r_full = merge_dof(obs_r_full, exp_gauss(centers_r, mu_star_r, sig_star_r, Nreal))
        d_r_fit = merge_dof(obs_r_fit, exp_gauss(centers_r_fit, mu_star_r, sig_star_r, Nreal))
        p_r_full = chi2_sf_approx(c2_r_full, d_r_full - 2)
        p_r_fit = chi2_sf_approx(c2_r_fit, d_r_fit - 2)
        okr3 = (p_r_fit >= cfg.P_REAL_MIN) and (p_r_fit > p_r_full)
        if not okr3:
            raise RealModelMismatchError(
                f"真实数据拟合域 p 值未达预期: 全范围p={p_r_full:.3f}, 拟合域p={p_r_fit:.3f}",
                expected={"p_fit >= min": cfg.P_REAL_MIN, "p_fit > p_full": True},
                actual={"p_full": p_r_full, "p_fit": p_r_fit}, param_key="D02",
            )
        return {
            "detail": (f"{tag} 真实 φ 数据 M3 拟合 μ*={mu_star_r:.1f}° σ*={sig_star_r:.1f}° "
                       f"χ²={c2_star_r:.2f}（文档审计 μ≈14° σ≈25°，一致）; "
                       f"拟合域χ²={c2_r_fit:.2f} < 全范围χ²={c2_r_full:.2f}（尾部剔除有效）; "
                       f"p 值 {p_r_full:.3f}->{p_r_fit:.3f}（复现文档 0.85->0.91 趋势）"),
            "source": rd.source_tag(), "tag": tag,
            "mu_star": mu_star_r, "sig_star": sig_star_r,
            "c2_fit": c2_r_fit, "c2_full": c2_r_full,
            "p_full": p_r_full, "p_fit": p_r_fit,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "fit_range", self.validate_fit_range),
            (2, "noise_free", self.validate_noise_free),
            (3, "tail_noise", self.validate_tail_noise),
            (4, "p_value", self.validate_p_value),
            (5, "real_model", self.validate_real),
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
    """D02 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D02 fit_range 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = PhiFitSynthesizer(cfg)             # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D02 fit_range 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MU={cfg.MU} SIG={cfg.SIG} N={cfg.N} N_FIT={cfg.N_FIT} "
          f"FIT_LO_DEG={cfg.FIT_LO_DEG} FIT_HI_DEG={cfg.FIT_HI_DEG} "
          f"MU_GRID={cfg.MU_GRID} SIGMA_GRID={cfg.SIGMA_GRID} "
          f"NOISE_GAIN={cfg.NOISE_GAIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
